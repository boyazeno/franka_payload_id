"""Visual review of a trajectory before it runs next to real walls.

Two backends. ``meshcat`` gives an interactive 3-D view with the wall planes drawn;
when it is not installed (or no browser is available) a matplotlib figure showing the
monitored-point paths against the half-spaces is written instead. The latter is what
runs in CI and in the analysis container.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Config
from .model import PandaModel
from .traj.constraints import half_space_clearances, monitored_positions
from .traj.fourier import FourierTrajectory


def trajectory_paths(pm: PandaModel, cfg: Config, traj: FourierTrajectory,
                     n_samples: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """``(positions, clearances)`` of every monitored point along the trajectory.

    Shapes ``(n_samples, n_points, 3)`` and ``(n_samples, n_points, n_half_spaces)``.
    """
    t = np.linspace(0.0, traj.period, n_samples, endpoint=False)
    q, _, _ = traj(t)
    positions = np.array([monitored_positions(pm, cfg.workspace, conf) for conf in q])
    clearances = np.array([half_space_clearances(pm, cfg.workspace, conf) for conf in q])
    return positions, clearances


def plot_trajectory(pm: PandaModel, cfg: Config, traj: FourierTrajectory,
                    out: Path, n_samples: int = 200) -> Path:
    """Top-down and side views with the wall planes, plus a clearance-vs-time trace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positions, clearances = trajectory_paths(pm, cfg, traj, n_samples)
    ws = cfg.workspace
    names = [mp.frame for mp in ws.monitored_points]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (i, j), (li, lj) in zip(axes[:2], [(0, 1), (0, 2)], [("x", "y"), ("x", "z")]):
        for k in range(positions.shape[1]):
            ax.plot(positions[:, k, i], positions[:, k, j], lw=1.0, label=names[k])
        ax.scatter([0], [0], marker="s", s=60, c="k", zorder=5, label="base")
        lim = np.abs(positions[:, :, [i, j]]).max() * 1.3
        for hs in ws.half_spaces:
            n = hs.normal
            if abs(n[i]) < 1e-9 and abs(n[j]) < 1e-9:
                continue
            grid = np.linspace(-lim, lim, 2)
            if abs(n[j]) > 1e-9:
                ax.plot(grid, (-hs.offset - n[i] * grid) / n[j], "r--", lw=1.5)
            else:
                ax.axvline(-hs.offset / n[i], color="r", ls="--", lw=1.5)
        ax.set_xlabel(f"{li} [m]")
        ax.set_ylabel(f"{lj} [m]")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    axes[0].set_title("top view (red dashed = wall/table planes)")
    axes[1].set_title("side view")
    axes[0].legend(fontsize=7, loc="upper right")

    worst = clearances.min(axis=(1, 2))
    t = np.linspace(0.0, traj.period, n_samples, endpoint=False)
    axes[2].plot(t, worst, lw=1.5)
    axes[2].axhline(0.0, color="r", ls="--", lw=1.5)
    axes[2].set_xlabel("t [s]")
    axes[2].set_ylabel("worst clearance [m]")
    axes[2].set_title(f"min clearance {worst.min():+.3f} m")
    axes[2].grid(alpha=0.3)

    if not ws.all_measured:
        fig.suptitle("WORKSPACE PLANES ARE UNMEASURED PLACEHOLDERS", color="red")

    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def view_trajectory(pm: PandaModel, cfg: Config, traj: FourierTrajectory,
                    out: Path | None = None) -> int:
    """Interactive meshcat view when available, otherwise write a figure."""
    if out is not None:
        path = plot_trajectory(pm, cfg, traj, out)
        print(f"wrote {path}")
        return 0

    try:
        import meshcat  # noqa: F401
        from pinocchio.visualize import MeshcatVisualizer
    except ImportError:
        default = Path("data/results/trajectory.png")
        print("meshcat is not installed (pip install 'franka_payload_id[viz]'); "
              f"writing a figure to {default} instead")
        print(f"wrote {plot_trajectory(pm, cfg, traj, default)}")
        return 0

    import pinocchio as pin

    from .config import asset_dir, urdf_path

    model, collision, visual = pin.buildModelsFromUrdf(str(urdf_path()), str(asset_dir()))
    viz = MeshcatVisualizer(model, collision, visual)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    t = np.linspace(0.0, traj.period, 400, endpoint=False)
    q, _, _ = traj(t)
    print("playing the trajectory in meshcat; Ctrl-C to stop")
    try:
        while True:
            for conf in q:
                viz.display(conf)
    except KeyboardInterrupt:
        return 0
