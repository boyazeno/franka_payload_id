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

    # The configured load must match what was physically attached. Declaring it
    # truthfully is what keeps gravity compensation correct, and correct gravity
    # compensation is what makes both runs track the same reference -- which is the
    # property the difference method actually needs. A bare run must declare zero; a
    # loaded run declaring zero means the tool was unmodelled, which degrades tracking
    # and risks a tau_J_range_violation abort.
    total_mass, _, _ = run.meta.load_configuration()
    if not run.meta.loaded and total_mass != 0.0:
        problems.append(
            f"a BARE run declares a load of {total_mass:.4f} kg. Nothing was attached, so "
            "the configured load must be zero.")
        ok = False
    if run.meta.loaded and total_mass == 0.0:
        problems.append(
            "a LOADED run declares zero load: the tool was attached but unmodelled. "
            "Gravity compensation was therefore wrong, tracking differs from the bare "
            "run, and the robot may have aborted with tau_J_range_violation. Re-run with "
            "--load-mass / --load-com set to the tool's approximate values.")
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

    # The two runs are EXPECTED to declare different loads -- that is the point. What
    # must hold is that each declared what it was actually carrying. tau_J is a physical
    # link-side measurement, so the declaration cannot bias the difference directly; it
    # only acts through the achieved motion, and a truthful declaration is what keeps
    # both runs on the same reference trajectory.
    bare_mass, _, _ = bare.meta.load_configuration()
    if bare_mass != 0.0:
        raise ValueError(
            f"the bare run declares a load of {bare_mass:.4f} kg; it must declare zero.")

    loaded_mass, _, _ = loaded.meta.load_configuration()
    if loaded_mass == 0.0:
        raise ValueError(
            "the loaded run declares zero load, so the tool was carried unmodelled. "
            "Gravity compensation was wrong in that run only, so the two runs did not "
            "track the same trajectory and their torque difference contains an arm-"
            "dynamics term. Re-collect with --load-mass / --load-com.")
