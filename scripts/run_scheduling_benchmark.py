"""Reproducible multi-UAV scheduling benchmark for the revision.

The institutional task coordinates are unavailable in the supplied files.  This
script therefore creates a public, anonymized synthetic benchmark with the same
19-instance/574-task size profile reported in the manuscript.  It compares all
algorithms under an identical objective, evaluation budget, and 30 fixed random
seeds; writes raw CSV data and statistical tests; and generates a 600-dpi PNG
and vector PDF figure.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scheduling"
DATA = ROOT / "data" / "synthetic"
META = ROOT / "data" / "metadata"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)
META.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

TASK_COUNTS = [13, 15, 16, 19, 22, 25, 26, 28, 29, 31, 32, 34, 35, 37, 39, 41, 42, 44, 46]
SEEDS = list(range(202601, 202631))
N_UAV = 5
POP = 16
ITER = 25
EVAL_BUDGET = POP * (ITER + 1)


@dataclass(frozen=True)
class Scenario:
    scenario: int
    xyz: np.ndarray
    priority: np.ndarray
    service: np.ndarray
    task_type: np.ndarray
    earliest: np.ndarray
    latest: np.ndarray
    bases: np.ndarray


def make_scenario(index: int, n: int) -> Scenario:
    rng = np.random.default_rng(41000 + index)
    corridor = rng.integers(0, 3, n)
    x = np.linspace(90, 930, n) + rng.normal(0, 28, n)
    rng.shuffle(x)
    centerlines = np.array([235.0, 500.0, 765.0])
    y = centerlines[corridor] + 42 * np.sin(x / 145 + corridor) + rng.normal(0, 22, n)
    z = 55 + 18 * corridor + rng.normal(0, 7, n)
    xyz = np.c_[np.clip(x, 40, 960), np.clip(y, 40, 960), np.clip(z, 25, 155)]
    task_type = rng.integers(0, 6, n)
    priority = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.08, 0.18, 0.34, 0.26, 0.14])
    base_service = np.array([45, 60, 75, 40, 55, 70], float)
    service = base_service[task_type] + rng.normal(0, 4, n)
    earliest = rng.uniform(0, 180, n)
    latest = earliest + rng.uniform(420, 900, n) - 30 * (priority - 1)
    bases = np.c_[np.full(N_UAV, 20.0), np.linspace(160, 840, N_UAV), np.full(N_UAV, 60.0)]
    return Scenario(index, xyz, priority.astype(float), service, task_type, earliest, latest, bases)


SCENARIOS = [make_scenario(i + 1, n) for i, n in enumerate(TASK_COUNTS)]


def decode(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, N_UAV - 1).astype(np.int16)


def schedule_cost(assign_or_x: np.ndarray, s: Scenario) -> float:
    assign = decode(assign_or_x)
    completion = np.zeros(len(assign))
    total_distance = 0.0
    end_times = np.zeros(N_UAV)
    route_loads = np.zeros(N_UAV)

    for u in range(N_UAV):
        idx = np.flatnonzero(assign == u)
        t = 0.0
        # Fixed priority/deadline order makes the common cost function fast and
        # ensures that every algorithm is evaluated with exactly the same route decoder.
        idx = idx[np.lexsort((s.xyz[idx, 0], s.latest[idx], -s.priority[idx]))]
        pos = s.bases[u].copy()
        for k in idx:
            leg = float(np.linalg.norm(s.xyz[k] - pos))
            t += leg / 12.0
            t = max(t, float(s.earliest[k]))
            t += float(s.service[k])
            completion[k] = t
            total_distance += leg
            route_loads[u] += leg * 0.028 + s.service[k] * 0.11
            pos = s.xyz[k]
        end_times[u] = t

    makespan = end_times.max() / 1100.0
    energy = (total_distance * 0.028 + s.service.sum() * 0.11) / (480.0 + 8.0 * len(assign))
    risk_latency = np.average(completion / np.maximum(s.latest, 1), weights=s.priority) / 1.4
    imbalance = np.std(end_times) / (np.mean(end_times) + 1e-9)
    battery_penalty = np.maximum(route_loads - 290.0, 0).sum() / 120.0
    deadline_penalty = np.average(np.maximum(completion - s.latest, 0) / np.maximum(s.latest, 1), weights=s.priority)
    return float(0.40 * makespan + 0.20 * energy + 0.30 * risk_latency + 0.10 * imbalance
                 + 2.0 * battery_penalty + 1.5 * deadline_penalty)


def greedy_assignment(s: Scenario) -> np.ndarray:
    order = np.argsort(-s.priority + 1e-3 * s.latest)
    assign = np.zeros(len(order), dtype=float)
    load = np.zeros(N_UAV)
    last = s.bases.copy()
    for k in order:
        leg = np.linalg.norm(last - s.xyz[k], axis=1)
        score = leg / 12 + load + 0.05 * np.maximum(0, load - s.latest[k])
        u = int(np.argmin(score))
        assign[k] = u
        load[u] += leg[u] / 12 + s.service[k]
        last[u] = s.xyz[k]
    return assign


def pso(s: Scenario, rng: np.random.Generator, improved: bool) -> tuple[float, int]:
    n = len(s.xyz)
    x = rng.uniform(0, N_UAV - 1, (POP, n))
    if improved:
        g0 = greedy_assignment(s)
        for i in range(max(2, POP // 5)):
            x[i] = np.clip(g0 + rng.normal(0, 0.22 + 0.05 * i, n), 0, N_UAV - 1)
    v = rng.normal(0, 0.35, x.shape)
    fit = np.array([schedule_cost(row, s) for row in x])
    pbest, pfit = x.copy(), fit.copy()
    gidx = int(np.argmin(fit))
    g, gfit = x[gidx].copy(), float(fit[gidx])
    conv = ITER
    last_improve = 0
    for it in range(ITER):
        if improved:
            diversity = np.mean(np.std(x, axis=0)) / (N_UAV - 1)
            w = (0.90 - 0.50 * it / ITER) * (1.0 + 0.30 * (0.35 - diversity))
        else:
            w = 0.72
        r1, r2 = rng.random(x.shape), rng.random(x.shape)
        v = w * v + 1.55 * r1 * (pbest - x) + 1.55 * r2 * (g - x)
        x = np.clip(x + v, 0, N_UAV - 1)
        if improved and (it + 1) % 5 == 0:
            sigma = 0.50 - 0.45 * (it + 1) / ITER
            trial = np.clip(g + rng.normal(0, sigma, n), 0, N_UAV - 1)
            j = int(rng.integers(POP))
            x[j] = trial
        fit = np.array([schedule_cost(row, s) for row in x])
        mask = fit < pfit
        pfit[mask], pbest[mask] = fit[mask], x[mask]
        k = int(np.argmin(fit))
        if fit[k] < gfit - 1e-10:
            g, gfit, last_improve = x[k].copy(), float(fit[k]), it + 1
        if it - last_improve > 9 and conv == ITER:
            conv = it + 1
    return gfit, conv


def ga(s: Scenario, rng: np.random.Generator) -> tuple[float, int]:
    n = len(s.xyz)
    pop = rng.integers(0, N_UAV, (POP, n))
    fit = np.array([schedule_cost(row, s) for row in pop])
    best = float(fit.min())
    last_improve, conv = 0, ITER
    for it in range(ITER):
        rank = np.argsort(fit)
        elite = pop[rank[:2]].copy()
        children = [elite[0], elite[1]]
        while len(children) < POP:
            cand = rng.integers(0, POP, 6)
            p1 = pop[cand[:3][np.argmin(fit[cand[:3]])]]
            p2 = pop[cand[3:][np.argmin(fit[cand[3:]])]]
            mask = rng.random(n) < 0.5
            child = np.where(mask, p1, p2).copy()
            mut = rng.random(n) < 1.0 / n
            child[mut] = rng.integers(0, N_UAV, mut.sum())
            children.append(child)
        pop = np.asarray(children)
        fit = np.array([schedule_cost(row, s) for row in pop])
        if fit.min() < best - 1e-10:
            best, last_improve = float(fit.min()), it + 1
        if it - last_improve > 9 and conv == ITER:
            conv = it + 1
    return best, conv


def aco(s: Scenario, rng: np.random.Generator) -> tuple[float, int]:
    n = len(s.xyz)
    tau = np.ones((n, N_UAV))
    base_dist = np.linalg.norm(s.xyz[:, None, :] - s.bases[None, :, :], axis=2)
    eta = 1.0 / (base_dist + 50.0)
    best, best_a = np.inf, None
    last_improve, conv = 0, ITER
    for it in range(ITER + 1):
        ants = np.zeros((POP, n), dtype=int)
        for a in range(POP):
            load = np.zeros(N_UAV)
            for k in np.argsort(-s.priority):
                prob = tau[k] ** 1.1 * eta[k] ** 1.8 / (1 + load / 250)
                prob /= prob.sum()
                u = int(rng.choice(N_UAV, p=prob))
                ants[a, k] = u
                load[u] += s.service[k] + base_dist[k, u] / 12
        fit = np.array([schedule_cost(row, s) for row in ants])
        k = int(np.argmin(fit))
        if fit[k] < best - 1e-10:
            best, best_a, last_improve = float(fit[k]), ants[k].copy(), it
        tau *= 0.82
        if best_a is not None:
            tau[np.arange(n), best_a] += 1.0 / max(best, 1e-6)
        if it - last_improve > 9 and conv == ITER:
            conv = min(it, ITER)
    return best, conv


def differential_evolution(s: Scenario, rng: np.random.Generator) -> tuple[float, int]:
    n = len(s.xyz)
    pop = rng.uniform(0, N_UAV - 1, (POP, n))
    fit = np.array([schedule_cost(row, s) for row in pop])
    best, last_improve, conv = float(fit.min()), 0, ITER
    for it in range(ITER):
        for i in range(POP):
            pool = np.delete(np.arange(POP), i)
            a, b, c = rng.choice(pool, 3, replace=False)
            mutant = np.clip(pop[a] + 0.7 * (pop[b] - pop[c]), 0, N_UAV - 1)
            cross = rng.random(n) < 0.75
            cross[rng.integers(n)] = True
            trial = np.where(cross, mutant, pop[i])
            tf = schedule_cost(trial, s)
            if tf < fit[i]:
                pop[i], fit[i] = trial, tf
        if fit.min() < best - 1e-10:
            best, last_improve = float(fit.min()), it + 1
        if it - last_improve > 9 and conv == ITER:
            conv = it + 1
    return best, conv


ALGORITHMS = {
    "IPSO": lambda s, r: pso(s, r, True),
    "PSO": lambda s, r: pso(s, r, False),
    "GA": ga,
    "ACO": aco,
    "DE": differential_evolution,
}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adjusted = min(1.0, (m - rank) * p)
        running = max(running, adjusted)
        out[name] = running
    return out


def run():
    records = []
    for s in SCENARIOS:
        print(f"Scenario {s.scenario:02d}/{len(SCENARIOS)} (N={len(s.xyz)})", flush=True)
        for seed in SEEDS:
            for ai, (name, fn) in enumerate(ALGORITHMS.items()):
                rng = np.random.default_rng(seed + 100_000 * s.scenario + 1_000 * ai)
                t0 = time.perf_counter()
                fitness, convergence = fn(s, rng)
                records.append(
                    {
                        "scenario": s.scenario,
                        "tasks": len(s.xyz),
                        "seed": seed,
                        "algorithm": name,
                        "fitness": fitness,
                        "convergence_iteration": convergence,
                        "runtime_s": time.perf_counter() - t0,
                        "evaluation_budget": EVAL_BUDGET,
                    }
                )
    raw = pd.DataFrame(records)
    raw.to_csv(OUT / "scheduling_raw_30_seeds.csv", index=False)

    summary = (raw.groupby(["scenario", "tasks", "algorithm"], as_index=False)
               .agg(fitness_mean=("fitness", "mean"), fitness_sd=("fitness", "std"),
                    runtime_mean_s=("runtime_s", "mean"), runtime_sd_s=("runtime_s", "std"),
                    convergence_mean=("convergence_iteration", "mean")))
    summary.to_csv(OUT / "scheduling_summary_by_scenario.csv", index=False)

    overall = (raw.groupby("algorithm", as_index=False)
               .agg(fitness_mean=("fitness", "mean"), fitness_sd=("fitness", "std"),
                    runtime_mean_s=("runtime_s", "mean"), convergence_mean=("convergence_iteration", "mean")))
    overall.to_csv(OUT / "scheduling_overall_summary.csv", index=False)

    pivot = summary.pivot(index="scenario", columns="algorithm", values="fitness_mean")
    raw_p = {}
    tests = []
    for base in ["PSO", "GA", "ACO", "DE"]:
        stat, p = wilcoxon(pivot["IPSO"], pivot[base], alternative="less", zero_method="wilcox")
        raw_p[base] = float(p)
        diff = pivot[base] - pivot["IPSO"]
        wins = int((diff > 0).sum())
        losses = int((diff < 0).sum())
        tests.append({"baseline": base, "wilcoxon_W": float(stat), "p_raw": float(p),
                      "wins": wins, "ties": 19 - wins - losses, "losses": losses,
                      "median_relative_improvement_pct": float(np.median(diff / pivot[base]) * 100)})
    adjusted = holm_adjust(raw_p)
    for row in tests:
        row["p_holm"] = adjusted[row["baseline"]]
    pd.DataFrame(tests).to_csv(OUT / "scheduling_wilcoxon_holm.csv", index=False)

    scenario_rows = []
    for s in SCENARIOS:
        for k in range(len(s.xyz)):
            scenario_rows.append({"scenario": s.scenario, "task_id": k + 1,
                                  "x_m": s.xyz[k, 0], "y_m": s.xyz[k, 1], "z_m": s.xyz[k, 2],
                                  "priority": int(s.priority[k]), "task_type": int(s.task_type[k]),
                                  "service_s": s.service[k], "earliest_s": s.earliest[k], "latest_s": s.latest[k]})
    pd.DataFrame(scenario_rows).to_csv(DATA / "anonymized_synthetic_19_scenarios.csv", index=False)
    (META / "scheduling_benchmark_metadata.json").write_text(json.dumps({
        "task_counts": TASK_COUNTS, "total_tasks": sum(TASK_COUNTS), "seeds": SEEDS,
        "uavs": N_UAV, "population": POP, "iterations": ITER, "evaluation_budget": EVAL_BUDGET,
        "objective_weights": {"makespan": 0.40, "energy": 0.20, "risk_weighted_latency": 0.30, "load_imbalance": 0.10},
        "note": "Synthetic public benchmark; not the institutionally restricted raw coordinates."
    }, indent=2), encoding="utf-8")
    make_figure(raw, summary, pivot)
    print("\nOverall summary\n", overall.to_string(index=False))
    print("\nStatistical tests\n", pd.DataFrame(tests).to_string(index=False))


def make_figure(raw: pd.DataFrame, summary: pd.DataFrame, pivot: pd.DataFrame):
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    colors = {"IPSO": "#d93b48", "PSO": "#3386aa", "GA": "#239b6b", "ACO": "#f19a55", "DE": "#765fa6"}
    order = ["IPSO", "PSO", "GA", "ACO", "DE"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.3), constrained_layout=True)

    ax = axes[0, 0]
    for name in order:
        q = summary[summary.algorithm == name]
        ax.errorbar(q.scenario, q.fitness_mean, yerr=q.fitness_sd, color=colors[name], marker="o",
                    ms=3.2, lw=1.25, capsize=1.6, label=name)
    ax.set(xlabel="Scenario", ylabel="Fitness (mean ± SD)", xticks=range(1, 20))
    ax.set_title("(a) Thirty independent runs per scenario", loc="left", fontweight="bold")
    ax.grid(ls=":", alpha=0.6)
    ax.legend(ncol=3, frameon=False)

    ax = axes[0, 1]
    rel = pd.DataFrame({b: (pivot[b] - pivot["IPSO"]) / pivot[b] * 100 for b in order[1:]})
    positions = np.arange(1, 5)
    bp = ax.boxplot([rel[b] for b in order[1:]], positions=positions, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markersize=4, markerfacecolor="white", markeredgecolor="black"))
    for patch, b in zip(bp["boxes"], order[1:]):
        patch.set_facecolor(colors[b]); patch.set_alpha(0.70)
    ax.axhline(0, color="#555555", lw=0.9)
    ax.set(xticks=positions, xticklabels=order[1:], ylabel="IPSO relative improvement (%)")
    ax.set_title("(b) Scenario-level relative improvement", loc="left", fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.6)

    ax = axes[1, 0]
    rt = raw.groupby(["tasks", "algorithm"], as_index=False).runtime_s.mean()
    for name in order:
        q = rt[rt.algorithm == name]
        ax.plot(q.tasks, q.runtime_s, color=colors[name], marker="o", ms=3, lw=1.2, label=name)
    ax.set(xlabel="Number of tasks", ylabel="Runtime (s)")
    ax.set_title("(c) Computation time under equal budgets", loc="left", fontweight="bold")
    ax.grid(ls=":", alpha=0.6)

    ax = axes[1, 1]
    conv = raw.groupby("algorithm").convergence_iteration.agg(["mean", "std"]).loc[order]
    ax.bar(np.arange(5), conv["mean"], yerr=conv["std"], color=[colors[n] for n in order],
           alpha=0.82, capsize=3, edgecolor="black", linewidth=0.7)
    ax.set(xticks=np.arange(5), xticklabels=order, ylabel="Convergence iteration")
    ax.set_title("(d) Convergence iteration (mean ± SD)", loc="left", fontweight="bold")
    ax.grid(axis="y", ls=":", alpha=0.6)

    fig.suptitle("Reproducible scheduling benchmark on 19 public synthetic scenarios",
                 fontsize=12, fontweight="bold")
    fig.savefig(FIG / "Fig7_reproducible_scheduling_benchmark.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Fig7_reproducible_scheduling_benchmark.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run()
