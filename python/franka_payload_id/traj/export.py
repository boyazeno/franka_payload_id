"""Export trajectories and pose lists in the CSV forms the collector reads.

The excitation trajectory is sampled to the full 1 kHz here rather than on the robot,
so that the exact rows which will be executed are the ones that were safety-checked.
The collector then only replays them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import PandaLimits, Workspace
from ..model import PandaModel
from .constraints import ConstraintReport, check_configurations, check_trajectory
from .fourier import FourierTrajectory, StaticPoseSet


class UnsafeExportError(RuntimeError):
    """Raised when an export would put an unchecked trajectory on the robot."""


def _guard(report: ConstraintReport, force: bool, what: str) -> None:
    if report.ok and not report.placeholder_workspace:
        return
    if force:
        return
    raise UnsafeExportError(
        f"refusing to export {what}:\n{report.summary()}\n"
        "Fix the problems above, or pass force=True (CLI: --force) if you have "
        "independently verified that this is safe.")


def export_trajectory(path: Path | str, traj: FourierTrajectory, pm: PandaModel,
                      ws: Workspace, limits: PandaLimits, *,
                      sample_rate_hz: float = 1000.0, n_periods: int = 10,
                      force: bool = False, extra: dict | None = None,
                      payload_phi=None) -> Path:
    """Write the 1 kHz joint trajectory CSV.

    Format::

        # {json describing the trajectory}
        t,q0,q1,q2,q3,q4,q5,q6
        0.000,...

    Raises
    ------
    UnsafeExportError
        If the trajectory violates a limit or the workspace planes are still
        unmeasured placeholders.
    """
    from .constraints import make_torque_predictor
    report = check_trajectory(pm, ws, limits, traj, n_samples=1000,
                              torque_fn=make_torque_predictor(pm, payload_phi))
    _guard(report, force, "excitation trajectory")

    t, q, _, _ = traj.sample(sample_rate_hz, n_periods)
    header = traj.to_dict()
    header.update({
        "sample_rate_hz": float(sample_rate_hz),
        "n_periods": int(n_periods),
        "samples_per_period": int(traj.samples_per_period(sample_rate_hz)),
        "n_samples": int(t.size),
        "min_clearance_m": float(report.min_half_space_clearance_m),
        "max_velocity_ratio": float(report.max_velocity_ratio),
        "max_acceleration_ratio": float(report.max_acceleration_ratio),
        "max_torque_ratio": float(report.max_torque_ratio),
        "workspace_measured": bool(not report.placeholder_workspace),
    })
    if extra:
        header.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#" + json.dumps(header, separators=(",", ":")) + "\n")
        fh.write("t," + ",".join(f"q{i}" for i in range(7)) + "\n")
        for k in range(t.size):
            fh.write(f"{t[k]:.6f}," + ",".join(f"{v:.9f}" for v in q[k]) + "\n")
    return path


def export_static_poses(path: Path | str, poses: StaticPoseSet, pm: PandaModel,
                        ws: Workspace, limits: PandaLimits, *,
                        force: bool = False, extra: dict | None = None) -> Path:
    """Write the static-pose CSV.

    Format::

        # {json}
        direction,a0..a6,m0..m6

    One row per (pose, approach direction). The two directions of a pose are adjacent
    so the offline side can pair and average them.
    """
    waypoints = poses.waypoints()
    all_configs = np.array([w[0] for w in waypoints] + [w[1] for w in waypoints])
    report = check_configurations(pm, ws, limits, all_configs)
    _guard(report, force, "static pose set")

    header = poses.to_dict()
    header.update({
        "n_rows": len(waypoints),
        "min_clearance_m": float(report.min_half_space_clearance_m),
        "workspace_measured": bool(not report.placeholder_workspace),
    })
    if extra:
        header.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#" + json.dumps(header, separators=(",", ":")) + "\n")
        fh.write("direction," + ",".join(f"a{i}" for i in range(7)) + ","
                 + ",".join(f"m{i}" for i in range(7)) + "\n")
        for approach, measure, direction in waypoints:
            fh.write(f"{direction}," + ",".join(f"{v:.9f}" for v in approach) + ","
                     + ",".join(f"{v:.9f}" for v in measure) + "\n")
    return path


def load_trajectory_csv(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read back an exported trajectory: ``(t, q, header)``."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        first = fh.readline()
        header = json.loads(first[1:]) if first.startswith("#") else {}
        fh.readline()  # column names
        rows = np.loadtxt(fh, delimiter=",")
    rows = np.atleast_2d(rows)
    return rows[:, 0], rows[:, 1:], header
