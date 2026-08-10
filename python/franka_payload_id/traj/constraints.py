r"""Joint-limit and Cartesian half-space safety checking.

The robot stands in the corner of two walls facing outwards, so every monitored point
on the arm must satisfy, for every half-space :math:`w`,

.. math:: n_w^\top p_k(q) + d_w \;\ge\; \text{margin} + r_k

with :math:`n_w` the inward unit normal, :math:`r_k` the radius of the sphere bounding
that part of the robot, and the margin a fixed clearance.

The gradient with respect to the trajectory's free parameters is analytic:

.. math::
    \frac{\partial\,(n_w^\top p_k)}{\partial x}
      = n_w^\top J_{p_k}(q(t_i))\,\frac{\partial q(t_i)}{\partial x}

but the optimiser here uses SLSQP with finite differences on a modest parameter
vector, so only the *values* are needed; :func:`half_space_jacobian` is provided for
callers that want the exact gradient.

A hard joint box on :math:`q_1` is applied in addition. It is redundant with the
half-spaces when the optimiser succeeds, and it is the guarantee that survives when it
does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import PandaLimits, Workspace
from ..model import PandaModel
from .fourier import FourierTrajectory


@dataclass
class ConstraintReport:
    """Outcome of a safety check, with enough detail to tell the user what to fix."""

    ok: bool
    """Every geometric and kinematic constraint is satisfied."""

    placeholder_workspace: bool = False
    """True while ``config/workspace.yaml`` still holds unmeasured placeholder planes.

    Kept separate from :attr:`ok` on purpose: a trajectory can be perfectly feasible
    against the conservative defaults (so optimisation succeeds and the tests run
    hardware-free) while still being unfit to execute next to real walls. Export and
    the on-robot entry points require :attr:`ready_for_hardware`.
    """

    violations: list[str] = field(default_factory=list)
    # Worst-case margins; positive means satisfied.
    min_half_space_clearance_m: float = np.inf
    worst_half_space: str = ""
    max_position_excess_rad: float = 0.0
    max_velocity_ratio: float = 0.0
    max_acceleration_ratio: float = 0.0
    max_jerk_ratio: float = 0.0
    max_torque_ratio: float = 0.0

    @property
    def ready_for_hardware(self) -> bool:
        return self.ok and not self.placeholder_workspace

    def summary(self) -> str:
        lines = [f"safety check: {'PASS' if self.ok else 'FAIL'}"]
        if self.placeholder_workspace:
            lines.append("  NOT READY FOR HARDWARE: workspace planes are unmeasured placeholders")
        lines.append(f"  min clearance to any wall/table : {self.min_half_space_clearance_m:+.4f} m"
                     + (f"  (worst: {self.worst_half_space})" if self.worst_half_space else ""))
        lines.append(f"  max joint-position excess      : {self.max_position_excess_rad:.4f} rad")
        lines.append(f"  max |qd| / limit               : {self.max_velocity_ratio:.3f}")
        lines.append(f"  max |qdd| / limit              : {self.max_acceleration_ratio:.3f}")
        lines.append(f"  max |qddd| / limit             : {self.max_jerk_ratio:.3f}")
        if np.isfinite(self.max_torque_ratio) and self.max_torque_ratio > 0:
            lines.append(f"  max |tau| / limit              : {self.max_torque_ratio:.3f}")
        for v in self.violations:
            lines.append(f"  VIOLATION: {v}")
        return "\n".join(lines)


def make_torque_predictor(pm: PandaModel, payload_phi: np.ndarray | None = None):
    """Return ``f(q, qd, qdd) -> (K, 7)`` predicting joint torque [Nm] via RNEA.

    Includes the payload when given, since it only ever *increases* the requirement.
    The prediction is rigid-body only: it excludes joint friction and whatever
    corrective effort the internal impedance controller adds, so the derating factor in
    ``config/experiment.yaml`` has to cover those. Treat it as a budget, not a bound.
    """
    model = pm if payload_phi is None else pm.with_payload(payload_phi)

    def predict(q: np.ndarray, qd: np.ndarray, qdd: np.ndarray) -> np.ndarray:
        q = np.atleast_2d(q)
        qd = np.atleast_2d(qd)
        qdd = np.atleast_2d(qdd)
        return np.array([model.rnea(q[k], qd[k], qdd[k]) for k in range(q.shape[0])])

    return predict


def monitored_positions(pm: PandaModel, ws: Workspace, q: np.ndarray) -> np.ndarray:
    """Base-frame positions of every monitored point, shape ``(n_points, 3)``."""
    pm.forward(q)
    out = np.empty((len(ws.monitored_points), 3), dtype=float)
    for i, mp in enumerate(ws.monitored_points):
        placement = pm.data.oMf[pm.model.getFrameId(mp.frame)]
        out[i] = placement.act(mp.offset)
    return out


def half_space_clearances(pm: PandaModel, ws: Workspace, q: np.ndarray) -> np.ndarray:
    r"""Clearance of every (point, half-space) pair, shape ``(n_points, n_half_spaces)``.

    Entry ``(k, w)`` is :math:`n_w^\top p_k + d_w - r_k - \text{margin}`; it must be
    non-negative for the configuration to be safe.
    """
    pts = monitored_positions(pm, ws, q)
    radii = np.array([mp.radius for mp in ws.monitored_points])
    spaces = ws.half_spaces
    out = np.empty((pts.shape[0], len(spaces)), dtype=float)
    for w, hs in enumerate(spaces):
        out[:, w] = hs.signed_distance(pts) - radii - ws.margin
    return out


def half_space_jacobian(pm: PandaModel, ws: Workspace, q: np.ndarray) -> np.ndarray:
    r"""Gradient of the clearances w.r.t. ``q``, shape ``(n_points, n_half_spaces, 7)``.

    :math:`\partial(n_w^\top p_k)/\partial q = n_w^\top J_{p_k}(q)`.
    """
    spaces = ws.half_spaces
    out = np.empty((len(ws.monitored_points), len(spaces), pm.nv), dtype=float)
    for k, mp in enumerate(ws.monitored_points):
        jac = pm.frame_position_jacobian(q, mp.frame, mp.offset)   # (3, 7)
        for w, hs in enumerate(spaces):
            out[k, w, :] = hs.normal @ jac
    return out


def check_configurations(pm: PandaModel, ws: Workspace, limits: PandaLimits,
                         q: np.ndarray) -> ConstraintReport:
    """Check a set of configurations against joint limits and the half-spaces."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    report = ConstraintReport(ok=True)

    below = np.maximum(limits.q_min - q, 0.0)
    above = np.maximum(q - limits.q_max, 0.0)
    report.max_position_excess_rad = float(max(below.max(), above.max()))
    if report.max_position_excess_rad > 0.0:
        joints = sorted(set(np.argwhere((below + above) > 0.0)[:, 1].tolist()))
        report.violations.append(
            f"joint position limits exceeded on joints {[j + 1 for j in joints]} "
            f"by up to {report.max_position_excess_rad:.4f} rad")
        report.ok = False

    q1 = q[:, 0]
    if q1.min() < ws.q1_min - 1e-12 or q1.max() > ws.q1_max + 1e-12:
        report.violations.append(
            f"joint 1 leaves the hard safety box [{ws.q1_min:.3f}, {ws.q1_max:.3f}] rad "
            f"(range visited: [{q1.min():.3f}, {q1.max():.3f}])")
        report.ok = False

    worst = np.inf
    worst_name = ""
    spaces = ws.half_spaces
    for conf in q:
        clear = half_space_clearances(pm, ws, conf)
        idx = np.unravel_index(np.argmin(clear), clear.shape)
        if clear[idx] < worst:
            worst = float(clear[idx])
            worst_name = f"{ws.monitored_points[idx[0]].frame} vs {spaces[idx[1]].name}"
    report.min_half_space_clearance_m = worst
    report.worst_half_space = worst_name
    if worst < 0.0:
        report.violations.append(
            f"workspace half-space violated by {-worst:.4f} m ({worst_name})")
        report.ok = False

    if not ws.all_measured:
        report.placeholder_workspace = True
        report.violations.append(
            "workspace half-spaces are still the conservative placeholders "
            "(measured: false in config/workspace.yaml) -- measure the real walls "
            "before running on hardware")

    return report


def check_trajectory(pm: PandaModel, ws: Workspace, limits: PandaLimits,
                     traj: FourierTrajectory, n_samples: int = 500,
                     torque_fn=None) -> ConstraintReport:
    """Check a whole trajectory on a dense time grid.

    ``limits`` should be the *derated* limits. ``torque_fn(q, qd, qdd) -> (K, 7)``, if
    given, supplies a predicted torque for the torque-limit check.
    """
    t = np.linspace(0.0, traj.period, int(n_samples), endpoint=False)
    q, qd, qdd = traj(t)
    qddd = traj.jerk(t)

    report = check_configurations(pm, ws, limits, q)

    report.max_velocity_ratio = float(np.max(np.abs(qd) / limits.qd_max))
    report.max_acceleration_ratio = float(np.max(np.abs(qdd) / limits.qdd_max))
    report.max_jerk_ratio = float(np.max(np.abs(qddd) / limits.qddd_max))

    for name, ratio in (("velocity", report.max_velocity_ratio),
                        ("acceleration", report.max_acceleration_ratio),
                        ("jerk", report.max_jerk_ratio)):
        if ratio > 1.0:
            report.violations.append(f"{name} limit exceeded (ratio {ratio:.3f})")
            report.ok = False

    if torque_fn is not None:
        tau = np.asarray(torque_fn(q, qd, qdd))
        report.max_torque_ratio = float(np.max(np.abs(tau) / limits.tau_max))
        if report.max_torque_ratio > 1.0:
            report.violations.append(
                f"predicted torque exceeds the derated limit (ratio {report.max_torque_ratio:.3f})")
            report.ok = False

    rest_v, rest_a = traj.boundary_residuals()
    if rest_v > 1e-9 or rest_a > 1e-9:
        report.violations.append(
            f"trajectory does not start at rest (|qd(0)|={rest_v:.2e}, |qdd(0)|={rest_a:.2e}); "
            "the FCI requires qd_c = qdd_c = 0 at both ends")
        report.ok = False

    return report
