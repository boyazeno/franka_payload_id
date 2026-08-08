"""Quality gating: decide whether a run may enter the identification set.

A contaminated run is worse than a missing one, because least squares will happily
absorb the contamination into whichever parameters correlate with it. The gates below
are the ones the FCI documentation and libfranka's own examples point at.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .robot_log import RunLog

# franka::RobotMode
MODE_IDLE, MODE_MOVE = 1, 2
_MODE_NAMES = {0: "Other", 1: "Idle", 2: "Move", 3: "Guiding",
               4: "Reflex", 5: "UserStopped", 6: "AutomaticErrorRecovery"}


@dataclass
class QualityReport:
    ok: bool
    n_samples: int
    min_success_rate: float
    mean_success_rate: float
    n_long_periods: int
    max_period_ms: float
    modes_seen: list = field(default_factory=list)
    n_error_samples: int = 0
    problems: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"run quality: {'PASS' if self.ok else 'REJECT'}",
                 f"  samples                     : {self.n_samples}",
                 f"  control_command_success_rate: min {self.min_success_rate:.3f}, "
                 f"mean {self.mean_success_rate:.3f}",
                 f"  control period              : max {self.max_period_ms:.2f} ms "
                 f"({self.n_long_periods} ticks over 1 ms)",
                 f"  robot modes seen            : {', '.join(self.modes_seen)}"]
        for p in self.problems:
            lines.append(f"  PROBLEM: {p}")
        return "\n".join(lines)


def assess_run(run: RunLog, *, min_success_rate: float = 0.99,
               max_long_period_fraction: float = 0.01,
               require_move_mode: bool = True) -> QualityReport:
    """Check a collector run against the acceptance gates.

    ``control_command_success_rate`` is the fraction of the last 100 control commands
    the robot actually received. Persistently below ~0.99 means commands were being
    extrapolated, so the logged state no longer corresponds to the intended trajectory.
    """
    success = run.success_rate
    dt_ms = run.column("dt_s") * 1e3
    modes = run.column("robot_mode").astype(int)
    errors = run.column("errors").astype(np.int64)

    problems: list[str] = []
    ok = True

    if run.n_samples == 0:
        return QualityReport(False, 0, 0.0, 0.0, 0, 0.0, [], 0, ["run contains no samples"])

    # A success rate of exactly 0 is what the robot reports when no control loop is
    # running, so it is only meaningful for motion runs.
    active = success[success > 0.0]
    min_sr = float(active.min()) if active.size else 0.0
    mean_sr = float(active.mean()) if active.size else 0.0
    if active.size and min_sr < min_success_rate:
        problems.append(
            f"control_command_success_rate dropped to {min_sr:.3f} (< {min_success_rate}); "
            "communication quality was insufficient and the logged states may be extrapolated")
        ok = False

    n_long = int(np.sum(dt_ms > 1.5))
    if run.n_samples and n_long / run.n_samples > max_long_period_fraction:
        problems.append(
            f"{n_long} of {run.n_samples} control periods exceeded 1.5 ms "
            f"({100.0 * n_long / run.n_samples:.2f}% > "
            f"{100.0 * max_long_period_fraction:.2f}%)")
        ok = False

    seen = sorted(set(modes.tolist()))
    names = [_MODE_NAMES.get(m, str(m)) for m in seen]
    if require_move_mode and any(m not in (MODE_IDLE, MODE_MOVE) for m in seen):
        problems.append(f"robot left Idle/Move during the run (modes seen: {names}); "
                        "a reflex or user stop contaminates the data")
        ok = False

    n_err = int(np.count_nonzero(errors))
    if n_err:
        problems.append(f"{n_err} samples carry a non-zero error bitmask")
        ok = False

    # A pair of runs must not differ in configured load, or the internal controller
    # tracked differently in the two and the difference is not purely the payload.
    total_mass, _, _ = run.meta.load_configuration()
    if total_mass != 0.0:
        problems.append(
            f"configured total load is {total_mass:.4f} kg, not zero. Both runs of a pair "
            "must be collected with the load zeroed so the controller behaves identically.")
        ok = False

    return QualityReport(ok=ok, n_samples=run.n_samples, min_success_rate=min_sr,
                         mean_success_rate=mean_sr, n_long_periods=n_long,
                         max_period_ms=float(dt_ms.max()), modes_seen=names,
                         n_error_samples=n_err, problems=problems)


def assert_pair_compatible(loaded: RunLog, bare: RunLog) -> None:
    """Raise unless two runs can legitimately be differenced."""
    if loaded.meta.loaded == bare.meta.loaded:
        raise ValueError(
            "both runs are marked "
            f"{'loaded' if loaded.meta.loaded else 'bare'}; a difference-of-torques pair "
            "needs one of each")
    if abs(loaded.meta.sample_rate_hz - bare.meta.sample_rate_hz) > 1e-9:
        raise ValueError("paired runs were recorded at different sample rates")
    if loaded.meta.samples_per_period != bare.meta.samples_per_period:
        raise ValueError("paired runs have different period lengths")

    lm, lc, li = loaded.meta.load_configuration()
    bm, bc, bi = bare.meta.load_configuration()
    if not (np.isclose(lm, bm) and np.allclose(lc, bc) and np.allclose(li, bi)):
        raise ValueError(
            "the two runs were collected with different configured end-effector/load "
            "parameters. The internal controller then tracks differently in each run, "
            "so their torque difference is not the payload alone.")
