"""Reconstructed-controller benchmark with classical planners and MAPPO.

The original simulator/checkpoints were not supplied.  This script builds a
fully specified normalized 3-D environment, trains a parameter-sharing MAPPO
baseline and the dual-stream MAPPO actor, and evaluates them together with APF,
A*, and RRT* on 30 fixed test seeds.  Results are explicitly a reconstructed
reproducibility benchmark, not a replacement for the restricted field data.
"""

from __future__ import annotations

import heapq
import json
import math
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn
from torch.distributions import Normal


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controller"
META = ROOT / "data" / "metadata"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_AGENT, N_OBS = 5, 6
TRAIN_SEEDS = [71, 72, 73, 74, 75]
TEST_SEEDS = list(range(9101, 9131))


def make_world_np(seed: int):
    rng = np.random.default_rng(seed)
    starts = np.c_[np.full(N_AGENT, 0.06), np.linspace(0.12, 0.88, N_AGENT), np.linspace(0.22, 0.38, N_AGENT)]
    perm = rng.permutation(N_AGENT)
    goals = np.c_[np.full(N_AGENT, 0.94), np.linspace(0.12, 0.88, N_AGENT)[perm], np.linspace(0.62, 0.82, N_AGENT)]
    centers = np.c_[rng.uniform(0.28, 0.76, N_OBS), rng.uniform(0.12, 0.88, N_OBS), rng.uniform(0.20, 0.78, N_OBS)]
    radii = rng.uniform(0.065, 0.095, N_OBS)
    # Reject obstacles that overlap starts/goals.
    for j in range(N_OBS):
        for _ in range(30):
            if min(np.linalg.norm(starts - centers[j], axis=1).min(),
                   np.linalg.norm(goals - centers[j], axis=1).min()) > radii[j] + 0.10:
                break
            centers[j] = [rng.uniform(0.28, 0.76), rng.uniform(0.12, 0.88), rng.uniform(0.20, 0.78)]
    return starts.astype(np.float32), goals.astype(np.float32), centers.astype(np.float32), radii.astype(np.float32)


class VecEnv:
    def __init__(self, batch: int, seed: int):
        worlds = [make_world_np(seed * 1000 + i) for i in range(batch)]
        self.pos = torch.tensor(np.stack([w[0] for w in worlds]), device=DEVICE)
        self.goal = torch.tensor(np.stack([w[1] for w in worlds]), device=DEVICE)
        self.centers = torch.tensor(np.stack([w[2] for w in worlds]), device=DEVICE)
        self.radii = torch.tensor(np.stack([w[3] for w in worlds]), device=DEVICE)
        self.vel = torch.zeros_like(self.pos)
        self.prev_dist = torch.linalg.norm(self.goal - self.pos, dim=-1)
        self.collision_count = torch.zeros(batch, device=DEVICE)

    def obs(self):
        b = self.pos.shape[0]
        goal = self.goal - self.pos
        od = self.centers[:, None, :, :] - self.pos[:, :, None, :]
        oc = torch.linalg.norm(od, dim=-1) - self.radii[:, None, :]
        oi = oc.argmin(dim=-1)
        nearest_o = torch.gather(od, 2, oi[..., None, None].expand(b, N_AGENT, 1, 3)).squeeze(2)
        clearance = torch.gather(oc, 2, oi[..., None]).squeeze(-1)[..., None]
        ad = self.pos[:, None, :, :] - self.pos[:, :, None, :]
        dist = torch.linalg.norm(ad, dim=-1) + torch.eye(N_AGENT, device=DEVICE)[None] * 99
        ai = dist.argmin(dim=-1)
        nearest_a = torch.gather(ad, 2, ai[..., None, None].expand(b, N_AGENT, 1, 3)).squeeze(2)
        agent_dist = torch.gather(dist, 2, ai[..., None]).squeeze(-1)[..., None]
        centroid = self.pos.mean(dim=1, keepdim=True) - self.pos
        degree = (dist < 0.70).float().sum(dim=-1, keepdim=True) / (N_AGENT - 1)
        return torch.cat([goal, self.vel, nearest_o, clearance, nearest_a, agent_dist, centroid, degree], dim=-1)

    def safety_filter(self, action):
        obs = self.obs()
        ovec, clear = obs[..., 6:9], obs[..., 9:10]
        avec, adist = obs[..., 10:13], obs[..., 13:14]
        rep_o = -ovec / (torch.linalg.norm(ovec, dim=-1, keepdim=True) + 1e-6) * torch.relu(0.14 - clear) * 9.0
        rep_a = -avec / (torch.linalg.norm(avec, dim=-1, keepdim=True) + 1e-6) * torch.relu(0.10 - adist) * 10.0
        return torch.clamp(action + rep_o + rep_a, -1, 1)

    def step(self, action):
        action = torch.clamp(action, -1, 1)
        self.vel = self.vel * 0.82 + action * 0.018
        speed = torch.linalg.norm(self.vel, dim=-1, keepdim=True)
        self.vel = self.vel * torch.clamp(0.055 / (speed + 1e-8), max=1.0)
        self.pos = torch.clamp(self.pos + self.vel, 0.02, 0.98)
        d = torch.linalg.norm(self.goal - self.pos, dim=-1)
        progress = self.prev_dist - d
        self.prev_dist = d
        od = torch.linalg.norm(self.centers[:, None] - self.pos[:, :, None], dim=-1) - self.radii[:, None]
        obs_hit = od.min(dim=-1).values < 0.015
        ad = torch.linalg.norm(self.pos[:, None] - self.pos[:, :, None], dim=-1)
        pair_mask = ~torch.eye(N_AGENT, dtype=torch.bool, device=DEVICE)[None]
        near = (ad < 0.055) & pair_mask
        agent_hit = near.any(dim=-1)
        centroid_d = torch.linalg.norm(self.pos.mean(dim=1, keepdim=True) - self.pos, dim=-1)
        reward = 8.0 * progress - 3.5 * obs_hit.float() - 3.0 * agent_hit.float()
        reward -= 0.25 * (centroid_d > 0.72).float() + 0.006 * (action ** 2).sum(-1)
        reward += 3.0 * (d < 0.055).float()
        self.collision_count += (obs_hit | agent_hit).float().mean(-1)
        return reward, d < 0.055


class Actor(nn.Module):
    def __init__(self, dual: bool):
        super().__init__()
        self.dual = dual
        if dual:
            self.collision = nn.Sequential(nn.Linear(18, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))
            self.communication = nn.Sequential(nn.Linear(18, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 3))
            self.mix_logit = nn.Parameter(torch.tensor(1.1))
        else:
            self.net = nn.Sequential(nn.Linear(18, 128), nn.Tanh(), nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 3))
        self.log_std = nn.Parameter(torch.full((3,), -0.65))

    def forward(self, x):
        if self.dual:
            lam = torch.sigmoid(self.mix_logit)
            mean = lam * self.collision(x) + (1 - lam) * self.communication(x)
        else:
            mean = self.net(x)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(18 * N_AGENT, 192), nn.Tanh(), nn.Linear(192, 96), nn.Tanh(), nn.Linear(96, 1))

    def forward(self, obs):
        return self.net(obs.flatten(1)).squeeze(-1)


def policy_mean(actor, obs):
    """Shared goal-direction prior plus a learned MAPPO residual.

    The prior is identical for both neural baselines and represents the nominal
    waypoint tracker; MAPPO learns coordination/avoidance residuals around it.
    """
    goal = obs[..., :3]
    prior = goal / (torch.linalg.norm(goal, dim=-1, keepdim=True) + 1e-6)
    return torch.clamp(0.90 * prior + 0.55 * actor(obs), -1, 1)


def train_mappo(seed: int, dual: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    actor, critic = Actor(dual).to(DEVICE), Critic().to(DEVICE)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    gamma, lam_gae, clip = 0.98, 0.95, 0.20
    updates, batch, horizon = 32, 18, 64
    for upd in range(updates):
        env = VecEnv(batch, seed * 100 + upd)
        obs_buf, act_buf, log_buf, rew_buf, val_buf = [], [], [], [], []
        for _ in range(horizon):
            obs = env.obs()
            with torch.no_grad():
                mean = policy_mean(actor, obs)
                dist = Normal(mean, actor.log_std.exp())
                action = dist.sample()
                logp = dist.log_prob(action).sum(-1)
                value = critic(obs)
            exec_action = env.safety_filter(action) if dual else action
            reward, _ = env.step(exec_action)
            obs_buf.append(obs); act_buf.append(action); log_buf.append(logp)
            rew_buf.append(reward.mean(-1)); val_buf.append(value)
        obs_t = torch.stack(obs_buf)        # T,B,A,O
        act_t = torch.stack(act_buf)
        oldlog_t = torch.stack(log_buf)
        rew_t = torch.stack(rew_buf)
        val_t = torch.stack(val_buf)
        adv = torch.zeros_like(rew_t)
        gae = torch.zeros(batch, device=DEVICE)
        next_v = torch.zeros(batch, device=DEVICE)
        for t in reversed(range(horizon)):
            delta = rew_t[t] + gamma * next_v - val_t[t]
            gae = delta + gamma * lam_gae * gae
            adv[t] = gae
            next_v = val_t[t]
        ret = adv + val_t
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(3):
            mean = policy_mean(actor, obs_t)
            dist = Normal(mean, actor.log_std.exp())
            logp = dist.log_prob(act_t).sum(-1)
            ratio = torch.exp(logp - oldlog_t)
            adv_a = adv[..., None].expand_as(ratio)
            policy_loss = -torch.minimum(ratio * adv_a, torch.clamp(ratio, 1 - clip, 1 + clip) * adv_a).mean()
            value = critic(obs_t.flatten(0, 1)).reshape(horizon, batch)
            value_loss = 0.5 * (value - ret).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()
            loss = policy_loss + 0.5 * value_loss - 0.004 * entropy
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.7); opt.step()
    return actor.eval()


def simulate_actor(actor: Actor, seed: int, dual: bool):
    env = VecEnv(1, seed)
    path = [env.pos[0].detach().cpu().numpy().copy()]
    t0 = time.perf_counter()
    reached = torch.zeros(N_AGENT, dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        for _ in range(180):
            obs = env.obs(); action = policy_mean(actor, obs)
            if dual: action = env.safety_filter(action)
            _, done = env.step(action)
            reached |= done[0]
            path.append(env.pos[0].cpu().numpy().copy())
    runtime = time.perf_counter() - t0
    return path_metrics(np.asarray(path), reached.cpu().numpy(), runtime,
                        env.centers[0].cpu().numpy(), env.radii[0].cpu().numpy())


def apf(seed: int):
    starts, goals, centers, radii = make_world_np(seed)
    pos, vel = starts.copy(), np.zeros_like(starts)
    path = [pos.copy()]; reached = np.zeros(N_AGENT, bool)
    t0 = time.perf_counter()
    for _ in range(180):
        action = (goals - pos) * 3.0
        for i in range(N_AGENT):
            od = centers - pos[i]
            clearance = np.linalg.norm(od, axis=1) - radii
            j = np.argmin(clearance)
            if clearance[j] < 0.18:
                action[i] -= od[j] / (np.linalg.norm(od[j]) + 1e-6) * (0.18 - clearance[j]) * 18
            ad = pos - pos[i]; dd = np.linalg.norm(ad, axis=1); dd[i] = 99
            j = np.argmin(dd)
            if dd[j] < 0.12:
                action[i] -= ad[j] / (dd[j] + 1e-6) * (0.12 - dd[j]) * 18
        action = np.clip(action, -1, 1)
        vel = 0.82 * vel + 0.018 * action
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        vel *= np.minimum(1, 0.055 / (speed + 1e-8))
        pos = np.clip(pos + vel, 0.02, 0.98)
        reached |= np.linalg.norm(goals - pos, axis=1) < 0.055
        path.append(pos.copy())
    return path_metrics(np.asarray(path), reached, time.perf_counter() - t0, centers, radii)


def collision_free(a, b, centers, radii, margin=0.025):
    n = max(2, int(np.linalg.norm(b - a) / 0.02) + 1)
    pts = np.linspace(a, b, n)
    return np.all(np.linalg.norm(pts[:, None] - centers[None], axis=2) > radii[None] + margin)


def rrt_star_path(start, goal, centers, radii, rng, samples=520):
    nodes = [start.copy()]; parent = [-1]; cost = [0.0]
    for it in range(samples):
        target = goal if rng.random() < 0.12 else rng.uniform([0.03, 0.03, 0.08], [0.97, 0.97, 0.94])
        arr = np.asarray(nodes); ni = int(np.argmin(np.linalg.norm(arr - target, axis=1)))
        d = target - arr[ni]; new = arr[ni] + d / (np.linalg.norm(d) + 1e-9) * min(0.085, np.linalg.norm(d))
        if not collision_free(arr[ni], new, centers, radii): continue
        near = np.flatnonzero(np.linalg.norm(arr - new, axis=1) < 0.14)
        bestp, bestc = ni, cost[ni] + np.linalg.norm(new - arr[ni])
        for j in near:
            cc = cost[j] + np.linalg.norm(new - arr[j])
            if cc < bestc and collision_free(arr[j], new, centers, radii): bestp, bestc = int(j), float(cc)
        nodes.append(new); parent.append(bestp); cost.append(bestc)
        if np.linalg.norm(new - goal) < 0.10 and collision_free(new, goal, centers, radii):
            nodes.append(goal.copy()); parent.append(len(nodes) - 2); cost.append(bestc + np.linalg.norm(new - goal)); break
    if np.linalg.norm(nodes[-1] - goal) > 1e-6: return None
    out, k = [], len(nodes) - 1
    while k >= 0: out.append(nodes[k]); k = parent[k]
    return np.asarray(out[::-1])


def astar_path(start, goal, centers, radii):
    shape = np.array([21, 21, 11]); step = 1 / (shape - 1)
    si = tuple(np.clip(np.rint(start / step), 0, shape - 1).astype(int))
    gi = tuple(np.clip(np.rint(goal / step), 0, shape - 1).astype(int))
    moves = [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1) if (i, j, k) != (0, 0, 0)]
    q = [(0.0, si)]; g = {si: 0.0}; prev = {}
    while q:
        _, cur = heapq.heappop(q)
        if cur == gi: break
        for mv in moves:
            nb = tuple(np.asarray(cur) + mv)
            if any(nb[d] < 0 or nb[d] >= shape[d] for d in range(3)): continue
            p = np.asarray(nb) * step
            if np.any(np.linalg.norm(centers - p, axis=1) <= radii + 0.035): continue
            ng = g[cur] + np.linalg.norm(np.asarray(mv) * step)
            if ng < g.get(nb, 1e9):
                g[nb] = ng; prev[nb] = cur
                heapq.heappush(q, (ng + np.linalg.norm((np.asarray(nb) - gi) * step), nb))
    if gi not in g: return None
    out, cur = [np.asarray(gi) * step], gi
    while cur != si: cur = prev[cur]; out.append(np.asarray(cur) * step)
    return np.asarray(out[::-1])


def planner(seed: int, kind: str):
    starts, goals, centers, radii = make_world_np(seed); rng = np.random.default_rng(seed + 55)
    t0 = time.perf_counter(); paths = []
    for i in range(N_AGENT):
        p = astar_path(starts[i], goals[i], centers, radii) if kind == "A*" else rrt_star_path(starts[i], goals[i], centers, radii, rng)
        paths.append(p)
    runtime = time.perf_counter() - t0
    reached = np.array([p is not None for p in paths])
    maxn = max([len(p) if p is not None else 2 for p in paths]) + 2 * (N_AGENT - 1)
    traj = np.zeros((maxn, N_AGENT, 3))
    for i, p in enumerate(paths):
        if p is None: p = np.repeat(starts[i][None], 2, axis=0)
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
        q = np.linspace(0, dist[-1] if dist[-1] else 1, maxn - 2 * i)
        interp = np.c_[[np.interp(q, dist if dist[-1] else [0, 1], p[:, d] if dist[-1] else [p[0, d], p[0, d]]) for d in range(3)]]
        interp = interp.reshape(3, -1).T
        traj[:2 * i, i] = p[0]; traj[2 * i:, i] = interp
    return path_metrics(traj, reached, runtime, centers, radii)


def path_metrics(path, reached, runtime, centers, radii):
    length = np.linalg.norm(np.diff(path, axis=0), axis=2).sum(axis=0).mean()
    min_sep = 99.0
    collision = False
    for t in range(path.shape[0]):
        d = np.linalg.norm(path[t, :, None] - path[t, None, :], axis=2) + np.eye(N_AGENT) * 99
        min_sep = min(min_sep, float(d.min()))
        collision |= d.min() < 0.045
    clear = np.linalg.norm(path[:, :, None] - centers[None, None], axis=3) - radii[None, None]
    collision |= clear.min() < 0.0
    success = reached.astype(float)
    if collision: success *= 0.8
    return {"success_pct": success.mean() * 100, "minimum_separation": min_sep,
            "mean_path_length": float(length), "runtime_s": runtime, "collision": int(collision)}


def run():
    print(f"Device: {DEVICE}", flush=True)
    models = {"MAPPO": [], "Dual-stream MAPPO": []}
    for dual, name in [(False, "MAPPO"), (True, "Dual-stream MAPPO")]:
        for seed in TRAIN_SEEDS:
            print(f"Training {name}, seed {seed}", flush=True)
            models[name].append(train_mappo(seed, dual))
    rows = []
    for j, seed in enumerate(TEST_SEEDS):
        print(f"Evaluating test seed {seed}", flush=True)
        for name, fn in [("APF", apf), ("A*", lambda z: planner(z, "A*")), ("RRT*", lambda z: planner(z, "RRT*"))]:
            rows.append({"seed": seed, "method": name, **fn(seed)})
        model_i = j % len(TRAIN_SEEDS)
        rows.append({"seed": seed, "method": "MAPPO", **simulate_actor(models["MAPPO"][model_i], seed, False)})
        rows.append({"seed": seed, "method": "Dual-stream MAPPO", **simulate_actor(models["Dual-stream MAPPO"][model_i], seed, True)})
    raw = pd.DataFrame(rows); raw.to_csv(OUT / "controller_raw_30_test_seeds.csv", index=False)
    summary = raw.groupby("method", as_index=False).agg(
        success_mean=("success_pct", "mean"), success_sd=("success_pct", "std"),
        min_sep_mean=("minimum_separation", "mean"), min_sep_sd=("minimum_separation", "std"),
        path_length_mean=("mean_path_length", "mean"), path_length_sd=("mean_path_length", "std"),
        runtime_mean_s=("runtime_s", "mean"), collision_rate=("collision", "mean"))
    summary.to_csv(OUT / "controller_summary.csv", index=False)
    pivot = raw.pivot(index="seed", columns="method", values="success_pct")
    tests = []
    for b in ["APF", "A*", "RRT*", "MAPPO"]:
        try: stat, p = wilcoxon(pivot["Dual-stream MAPPO"], pivot[b], alternative="greater")
        except ValueError: stat, p = 0.0, 1.0
        tests.append({"baseline": b, "W": stat, "p_raw": p})
    pd.DataFrame(tests).to_csv(OUT / "controller_wilcoxon.csv", index=False)
    (META / "controller_benchmark_metadata.json").write_text(json.dumps({
        "training_seeds": TRAIN_SEEDS, "test_seeds": TEST_SEEDS, "device": str(DEVICE),
        "workspace": "normalized 3-D [0,1]^3 (reported as 1000 x 1000 x 200 m after axis scaling)",
        "uavs": N_AGENT, "obstacles": N_OBS, "training_updates": 32, "rollout_horizon": 64,
        "note": "Reconstructed benchmark because original simulator and checkpoints were not supplied."
    }, indent=2), encoding="utf-8")
    figure(raw, summary)
    print(summary.to_string(index=False))


def figure(raw, summary):
    order = ["APF", "A*", "RRT*", "MAPPO", "Dual-stream MAPPO"]
    colors = ["#8b8b8b", "#377eb8", "#6a5acd", "#2ca25f", "#e83e4d"]
    s = summary.set_index("method").loc[order]
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.6), constrained_layout=True)
    metrics = [("success_mean", "success_sd", "Task success (%)"),
               ("min_sep_mean", "min_sep_sd", "Minimum separation (normalized)"),
               ("path_length_mean", "path_length_sd", "Mean path length (normalized)"),
               ("runtime_mean_s", None, "Runtime (s)")]
    for i, (ax, (m, sd, yl)) in enumerate(zip(axes.flat, metrics)):
        err = s[sd] if sd else None
        ax.bar(np.arange(5), s[m], yerr=err, color=colors, capsize=3, edgecolor="black", linewidth=0.6)
        ax.set_xticks(np.arange(5)); ax.set_xticklabels(order, rotation=18, ha="right")
        ax.set_ylabel(yl); ax.grid(axis="y", ls=":", alpha=0.6)
        ax.set_title(f"({chr(97+i)}) {yl}", loc="left", fontweight="bold")
    fig.suptitle("Reconstructed 3-D controller benchmark (30 fixed test seeds)", fontsize=12, fontweight="bold")
    fig.savefig(FIG / "Fig8_controller_baselines.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Fig8_controller_baselines.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
