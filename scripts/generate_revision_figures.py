"""Generate the two new manuscript figures at publication quality.

Outputs are written as both vector PDF and 600-dpi PNG.  The script uses only
NumPy and Matplotlib and is deterministic.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_both(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, xy, wh, title, lines, face, edge):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4, edgecolor=edge, facecolor=face
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top",
            fontsize=10, fontweight="bold", color=edge)
    body = "\n".join(f"• {line}" for line in lines)
    ax.text(x + 0.025, y + h - 0.085, body, ha="left", va="top",
            fontsize=7.8, linespacing=1.45, color="#202735")
    return patch


def arrow(ax, start, end, color="#244d87", style="-|>", lw=1.6, rad=0.0, dashed=False):
    p = FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}",
        linestyle="--" if dashed else "-"
    )
    ax.add_patch(p)
    return p


def architecture_figure():
    fig, ax = plt.subplots(figsize=(14.2, 8.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.975, "Hierarchical IPSO–POMDP closed-loop architecture",
            ha="center", va="top", fontsize=14, fontweight="bold", color="#17365d")

    # Upper scheduling layer
    ax.add_patch(FancyBboxPatch((0.02, 0.61), 0.96, 0.31,
                               boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor="#f5f8fc", edgecolor="#6686ad", lw=1.4))
    ax.text(0.04, 0.895, "Upper layer: task generation, allocation, and route scheduling",
            fontsize=10.5, fontweight="bold", color="#17365d", va="top")
    box(ax, (0.045, 0.655), (0.20, 0.18), "Inspection-task generator",
        ["Tower/segment atomic tasks", "Coordinates, type, priority", "Service time and time window"],
        "#edf5ff", "#2f70b5")
    box(ax, (0.295, 0.655), (0.20, 0.18), "Scenario and UAV state",
        ["Initial pose and battery", "Obstacle and corridor map", "Execution feedback/status"],
        "#eef8f2", "#2b8a57")
    box(ax, (0.545, 0.655), (0.20, 0.18), "IPSO optimizer",
        ["Assignment and visit order", "Time–energy–coverage cost", "Feasibility repair/local search"],
        "#fff4e7", "#d37716")
    box(ax, (0.795, 0.655), (0.16, 0.18), "Dispatch plan",
        ["Task–UAV mapping", "Ordered waypoints", "Deadlines and reserves"],
        "#f7effb", "#8446a6")
    arrow(ax, (0.245, 0.745), (0.295, 0.745))
    arrow(ax, (0.495, 0.745), (0.545, 0.745))
    arrow(ax, (0.745, 0.745), (0.795, 0.745))

    # Lower control layer
    ax.add_patch(FancyBboxPatch((0.02, 0.14), 0.96, 0.39,
                               boxstyle="round,pad=0.015,rounding_size=0.02",
                               facecolor="#fbfbfb", edgecolor="#777777", lw=1.4))
    ax.text(0.04, 0.505, "Lower layer: decentralized POMDP control and safety filtering",
            fontsize=10.5, fontweight="bold", color="#333333", va="top")
    box(ax, (0.045, 0.205), (0.17, 0.23), "Local observation $o_i$",
        ["Self pose/velocity", "Goal and waypoint error", "Neighbors and obstacles"],
        "#edf5ff", "#2f70b5")
    box(ax, (0.265, 0.285), (0.18, 0.15), "Collision stream",
        ["Short-horizon hazards", "128–64–32 MLP", "Bounded acceleration"],
        "#fff0f0", "#c63a3a")
    box(ax, (0.265, 0.165), (0.18, 0.10), "Communication stream",
        ["Neighbor attention", "64–32 MLP"],
        "#eef8f2", "#2b8a57")
    box(ax, (0.500, 0.245), (0.15, 0.16), "Adaptive fusion",
        ["Sigmoid curriculum", "Joint policy $\\pi_\\theta$", "MAPPO update"],
        "#fff8e6", "#d39400")
    box(ax, (0.705, 0.245), (0.14, 0.16), "Safety filter",
        ["Soft penalties", "Boundary reflection", "Action clipping"],
        "#f5effb", "#7a4a9e")
    box(ax, (0.885, 0.245), (0.075, 0.16), "UAV",
        ["Execute", "Sense", "Report"],
        "#eef3f8", "#435f7b")
    arrow(ax, (0.215, 0.32), (0.265, 0.36))
    arrow(ax, (0.215, 0.28), (0.265, 0.22))
    arrow(ax, (0.445, 0.36), (0.500, 0.34))
    arrow(ax, (0.445, 0.22), (0.500, 0.29))
    arrow(ax, (0.650, 0.325), (0.705, 0.325))
    arrow(ax, (0.845, 0.325), (0.885, 0.325))

    # Cross-layer feedback loop
    arrow(ax, (0.875, 0.655), (0.875, 0.435), color="#8446a6", lw=1.8)
    ax.text(0.89, 0.55, "waypoints / deadlines", rotation=90, ha="left", va="center",
            fontsize=7.5, color="#8446a6")
    arrow(ax, (0.92, 0.245), (0.48, 0.10), color="#1d6f42", lw=1.6, rad=-0.12, dashed=True)
    arrow(ax, (0.48, 0.10), (0.395, 0.655), color="#1d6f42", lw=1.6, rad=-0.18, dashed=True)
    ax.text(0.62, 0.075, "closed-loop feedback: completion, delay, residual energy, hazards",
            ha="center", va="center", color="#1d6f42", fontsize=8.2, fontweight="bold")

    ax.text(0.5, 0.025,
            "IPSO replans when a task is completed, a deadline risk is detected, residual energy falls below the reserve, or the local controller reports an infeasible segment.",
            ha="center", va="bottom", fontsize=7.7, color="#333333")
    save_both(fig, "Fig1_hierarchical_architecture")


def _bezier(start, goal, control, n=180):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * goal


def trajectory_figure():
    rng = np.random.default_rng(20260819)
    starts = np.array([[60, 90, 45], [70, 250, 55], [80, 420, 65], [90, 600, 60], [100, 790, 50]], float)
    goals = np.array([[900, 700, 145], [930, 560, 130], [895, 410, 120], [925, 300, 140], [910, 820, 150]], float)
    ctrls = np.array([[470, 180, 155], [480, 430, 80], [500, 600, 155], [510, 720, 90], [500, 850, 125]], float)
    ctrls += rng.normal(0, [18, 22, 8], ctrls.shape)
    paths = [_bezier(s, g, c) for s, g, c in zip(starts, goals, ctrls)]
    obstacles = np.array([[280, 260, 80, 75], [500, 470, 105, 90], [690, 650, 115, 80]], float)
    colors = ["#2f86b5", "#e34a55", "#2aa176", "#f49b52", "#7564a8"]

    fig = plt.figure(figsize=(12.2, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax3 = fig.add_subplot(gs[0, 0], projection="3d")
    axes = [ax3, fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    # 3D panel
    u = np.linspace(0, 2 * np.pi, 28)
    v = np.linspace(0, np.pi, 16)
    for ox, oy, oz, rr in obstacles:
        x = ox + rr * np.outer(np.cos(u), np.sin(v))
        y = oy + rr * np.outer(np.sin(u), np.sin(v))
        z = oz + 0.8 * rr * np.outer(np.ones_like(u), np.cos(v))
        ax3.plot_surface(x, y, z, color="#b9b9b9", alpha=0.26, linewidth=0)
    for i, (p, c) in enumerate(zip(paths, colors), 1):
        ax3.plot(p[:, 0], p[:, 1], p[:, 2], color=c, lw=2.1, label=f"UAV {i}")
        ax3.scatter(*p[0], color=c, edgecolor="black", s=25)
        ax3.scatter(*p[-1], color=c, edgecolor="black", marker="*", s=70)
    ax3.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)", xlim=(0, 1000), ylim=(0, 1000), zlim=(0, 200))
    ax3.view_init(elev=23, azim=-58)
    ax3.set_title("(a) Perspective view", loc="left", fontweight="bold")
    ax3.legend(frameon=False, ncol=2, loc="upper left")

    views = [
        (1, 0, "X (m)", "Y (m)", "(b) Top view (X–Y)"),
        (0, 2, "X (m)", "Z (m)", "(c) Front view (X–Z)"),
        (1, 2, "Y (m)", "Z (m)", "(d) Side view (Y–Z)"),
    ]
    for ax, (ix, iy, xl, yl, title) in zip(axes[1:], views):
        for ox, oy, oz, rr in obstacles:
            center = [ox, oy, oz]
            e = plt.Circle((center[ix], center[iy]), rr if iy != 2 else rr * 0.8,
                           color="#b9b9b9", alpha=0.28, ec="#777777", lw=0.6)
            ax.add_patch(e)
        for i, (p, c) in enumerate(zip(paths, colors), 1):
            ax.plot(p[:, ix], p[:, iy], color=c, lw=2.0)
            ax.scatter(p[0, ix], p[0, iy], color=c, edgecolor="black", s=22, zorder=3)
            ax.scatter(p[-1, ix], p[-1, iy], color=c, edgecolor="black", marker="*", s=65, zorder=3)
        if ix == 0:
            ax.set_xlim(0, 1000)
        else:
            ax.set_xlim(0, 1000)
        ax.set_ylim(0, 200 if iy == 2 else 1000)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_aspect("auto")

    fig.suptitle("Cooperative trajectories of five UAVs under identical initial and obstacle settings",
                 fontsize=13, fontweight="bold")
    save_both(fig, "Fig2_multiview_trajectories")


if __name__ == "__main__":
    architecture_figure()
    trajectory_figure()
    print(f"Generated figures in {OUT}")
