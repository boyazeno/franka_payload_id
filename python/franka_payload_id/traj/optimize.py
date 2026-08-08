r"""Excitation-trajectory optimisation.

Objective: maximise the information the trajectory carries about the **ten payload
parameters only**. Because there are only ten -- no base-parameter reduction is needed,
unlike full-robot identification -- this is a small, well-behaved problem.

Let :math:`\tilde W = W D` be the column-scaled stacked payload regressor, with
:math:`D = \operatorname{diag}(1, L, L, L, L^2 \ldots)`. Scaling is *not* cosmetic: the
columns carry units kg, kg m and kg m^2, so an unscaled condition number is a
meaningless mixture of units.

Criteria:

``d_optimal``
    :math:`\min -\log\det(\tilde W^\top \tilde W)` -- maximises the information volume,
    i.e. minimises the volume of the parameter confidence ellipsoid. Smooth and
    well-behaved, so this is the default.
``condition``
    :math:`\min \operatorname{cond}_2(\tilde W)` -- equalises the best- and
    worst-determined directions. Non-smooth, but easy to interpret.
``hybrid``
    A weighted sum of the two.

Constraints (joint position/velocity/acceleration/jerk, the wall half-spaces and the
hard joint-1 box) are handled by a penalty, then the result is *verified* with the
independent hard checker in :mod:`.constraints`. A trajectory that fails verification
is never returned as a success.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..config import PandaLimits, Workspace
from ..model import PandaModel, scaling_matrix, stack_regressor
from .constraints import ConstraintReport, check_trajectory, half_space_clearances
from .fourier import N_JOINTS, FourierTrajectory


@dataclass
class OptimizationResult:
    trajectory: FourierTrajectory
    condition: float
    log_det: float
    report: ConstraintReport
    n_restarts_tried: int
    objective_value: float

    @property
    def ok(self) -> bool:
        return self.report.ok


def regressor_condition(pm: PandaModel, traj: FourierTrajectory, n_samples: int,
                        length_scale: float) -> tuple[float, float]:
    """``(condition_number, log_det)`` of the column-scaled payload regressor."""
    t = np.linspace(0.0, traj.period, int(n_samples), endpoint=False)
    q, qd, qdd = traj(t)
    w = stack_regressor(pm, q, qd, qdd) @ scaling_matrix(length_scale)
    cond = float(np.linalg.cond(w))
    sign, logabsdet = np.linalg.slogdet(w.T @ w)
    log_det = float(logabsdet) if sign > 0 else -np.inf
    return cond, log_det


def _penalty(pm: PandaModel, ws: Workspace, limits: PandaLimits,
             traj: FourierTrajectory, n_collocation: int) -> float:
    """Smooth non-negative penalty; zero exactly when every constraint is satisfied."""
    t = np.linspace(0.0, traj.period, int(n_collocation), endpoint=False)
    q, qd, qdd = traj(t)
    qddd = traj.jerk(t)

    pen = 0.0
    pen += np.sum(np.maximum(limits.q_min - q, 0.0) ** 2)
    pen += np.sum(np.maximum(q - limits.q_max, 0.0) ** 2)
    pen += np.sum(np.maximum(np.abs(qd) / limits.qd_max - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(np.abs(qdd) / limits.qdd_max - 1.0, 0.0) ** 2)
    pen += np.sum(np.maximum(np.abs(qddd) / limits.qddd_max - 1.0, 0.0) ** 2)

    q1 = q[:, 0]
    pen += np.sum(np.maximum(ws.q1_min - q1, 0.0) ** 2)
    pen += np.sum(np.maximum(q1 - ws.q1_max, 0.0) ** 2)

    for conf in q:
        clear = half_space_clearances(pm, ws, conf)
        pen += float(np.sum(np.maximum(-clear, 0.0) ** 2))

    return float(pen)


def _information_term(cond: float, log_det: float, criterion: str) -> float:
    if criterion == "d_optimal":
        return -log_det
    if criterion == "condition":
        return float(np.log(cond))
    if criterion == "hybrid":
        return float(np.log(cond)) - 0.1 * log_det
    raise ValueError(f"unknown objective {criterion!r}")


def _objective(x: np.ndarray, pm: PandaModel, ws: Workspace, limits: PandaLimits,
               n_harmonics: int, base_frequency: float, n_collocation: int,
               length_scale: float, criterion: str, penalty_weight: float) -> float:
    """Single smooth expression: information term plus a quadratic constraint penalty.

    Branching on feasibility (as an earlier version did) makes the objective
    discontinuous at the feasible boundary, which defeats gradient-based optimisers.
    Since a penalty optimum generally sits slightly *outside* the feasible set,
    :func:`_restore_feasibility` shrinks the result afterwards until the independent
    hard checker passes.
    """
    try:
        traj = FourierTrajectory.from_free_parameters(x, n_harmonics, base_frequency)
    except ValueError:
        return 1e9

    cond, log_det = regressor_condition(pm, traj, n_collocation, length_scale)
    if not np.isfinite(log_det) or not np.isfinite(cond):
        return 1e9

    pen = _penalty(pm, ws, limits, traj, n_collocation)
    return _information_term(cond, log_det, criterion) + penalty_weight * pen


def _scaled(traj: FourierTrajectory, gamma: float) -> FourierTrajectory:
    """Shrink the oscillatory part about ``q0``.

    Uniform scaling of ``a`` and ``b`` preserves both rest conditions
    (:math:`\\sum a_l = 0` and :math:`\\sum l b_l = 0`), so the result is still a
    legal FCI trajectory.
    """
    return FourierTrajectory(traj.q0, traj.a * gamma, traj.b * gamma, traj.base_frequency)


def _restore_feasibility(pm: PandaModel, ws: Workspace, limits: PandaLimits,
                         traj: FourierTrajectory, n_samples: int,
                         tol: float = 1e-3,
                         backoff: float = 0.99) -> tuple[FourierTrajectory, float]:
    """Largest ``gamma`` in (0, 1] for which the hard checker passes, by bisection.

    Returns ``(trajectory, gamma)``. If even a nearly static trajectory is infeasible
    -- which means ``q0`` itself violates something -- ``gamma`` comes back as 0.0 and
    the caller reports failure rather than shipping an unsafe trajectory.

    Two details keep the answer robust rather than marginal. The check runs on a grid
    at least 1000 points dense, and the accepted scale is backed off by a further 1 %.
    Without both, the optimum sits exactly on a limit -- the D-optimal objective pushes
    there, since more excitation is always more information -- and a later check on a
    denser grid finds a peak the coarser one missed and declares the trajectory unsafe.
    """
    dense = max(int(n_samples), 1000)

    def feasible(g: float) -> bool:
        return check_trajectory(pm, ws, limits, _scaled(traj, g), n_samples=dense).ok

    if feasible(1.0):
        return traj, 1.0
    if not feasible(0.0):
        return _scaled(traj, 0.0), 0.0

    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid

    gamma = lo * backoff
    if not feasible(gamma):
        return _scaled(traj, 0.0), 0.0
    return _scaled(traj, gamma), gamma


def _seed(rng: np.random.Generator, limits: PandaLimits, ws: Workspace,
          n_harmonics: int, base_frequency: float, amplitude: float) -> np.ndarray:
    """A random but sane starting point: centred configuration, modest coefficients."""
    per_joint = FourierTrajectory.n_free_parameters(n_harmonics)
    x = np.zeros((N_JOINTS, per_joint))

    centre = 0.5 * (limits.q_min + limits.q_max)
    half = 0.5 * (limits.q_max - limits.q_min)
    x[:, 0] = centre + rng.uniform(-0.15, 0.15, N_JOINTS) * half
    x[0, 0] = np.clip(x[0, 0], ws.q1_min * 0.5, ws.q1_max * 0.5)

    # Velocity coefficients scaled so the resulting motion uses a fraction of the
    # velocity budget; the wrist joints get more because the inertia terms live there.
    budget = limits.qd_max * amplitude / n_harmonics
    weight = np.array([0.6, 0.6, 0.7, 0.7, 1.0, 1.0, 1.0])
    scale = (budget * weight)[:, None]
    x[:, 1:] = rng.uniform(-1.0, 1.0, (N_JOINTS, per_joint - 1)) * scale
    return x.ravel()


def optimize_trajectory(pm: PandaModel, ws: Workspace, limits: PandaLimits, *,
                        n_harmonics: int = 5, base_frequency: float = 0.2,
                        n_collocation: int = 60, length_scale: float = 0.1,
                        criterion: str = "d_optimal", n_restarts: int = 8,
                        max_iter: int = 300, seed: int = 0,
                        amplitude: float = 0.35,
                        verify_samples: int = 500) -> OptimizationResult:
    """Multi-start optimisation of a Fourier excitation trajectory.

    ``limits`` must already be derated. The returned result is only marked ``ok`` if
    the independent hard checker in :mod:`.constraints` also passes.
    """
    rng = np.random.default_rng(seed)
    args = (pm, ws, limits, n_harmonics, base_frequency, n_collocation,
            length_scale, criterion, 1e4)

    best: OptimizationResult | None = None
    for _ in range(int(n_restarts)):
        x0 = _seed(rng, limits, ws, n_harmonics, base_frequency, amplitude)
        res = minimize(_objective, x0, args=args, method="L-BFGS-B",
                       options={"maxiter": int(max_iter), "maxfun": int(max_iter) * 80,
                                "ftol": 1e-9, "gtol": 1e-7, "eps": 1e-6})

        candidate = FourierTrajectory.from_free_parameters(
            np.asarray(res.x), n_harmonics, base_frequency)
        # The penalty optimum typically sits marginally outside the feasible set;
        # shrink until the independent hard checker accepts it.
        candidate, gamma = _restore_feasibility(pm, ws, limits, candidate, verify_samples)
        if gamma <= 0.0:
            continue

        cond, log_det = regressor_condition(pm, candidate, max(n_collocation, 200),
                                            length_scale)
        if not np.isfinite(log_det):
            continue
        report = check_trajectory(pm, ws, limits, candidate, n_samples=verify_samples)
        value = _information_term(cond, log_det, criterion)

        if best is None or value < best.objective_value:
            best = OptimizationResult(trajectory=candidate, condition=cond,
                                      log_det=log_det, report=report,
                                      n_restarts_tried=int(n_restarts),
                                      objective_value=value)

    if best is None:
        raise RuntimeError(
            "no feasible excitation trajectory found. The workspace constraints in "
            "config/workspace.yaml may be too tight, or the derating too aggressive.")
    return best


# ---------------------------------------------------------------------------
# Static poses (Stage A)
# ---------------------------------------------------------------------------
def optimize_static_poses(pm: PandaModel, ws: Workspace, limits: PandaLimits, *,
                          n_poses: int = 50, length_scale: float = 0.1,
                          seed: int = 0, n_candidates: int = 4000) -> np.ndarray:
    r"""Greedily select static poses that condition the four-column gravity regressor.

    Candidates are sampled uniformly inside the derated joint box, filtered through the
    hard safety checks, then added one at a time by whichever candidate most increases
    :math:`\log\det(\tilde W^\top \tilde W)`. That is D-optimal design by greedy
    exchange -- cheap, deterministic given the seed, and quite sufficient because the
    static problem is well conditioned to begin with.

    Orientation spread is what makes the centre of mass observable, and this criterion
    produces it automatically: poses with similar flange orientations add nearly
    collinear rows and therefore little determinant.
    """
    from ..model import gravity_regressor  # local import keeps the module import light

    rng = np.random.default_rng(seed)
    scale = np.array([1.0, length_scale, length_scale, length_scale])

    feasible: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    for _ in range(int(n_candidates)):
        q = rng.uniform(limits.q_min, limits.q_max)
        q[0] = rng.uniform(ws.q1_min, ws.q1_max)
        if np.min(half_space_clearances(pm, ws, q)) < 0.0:
            continue
        feasible.append(q)
        blocks.append(gravity_regressor(pm, q) * scale)
        if len(feasible) >= 20 * n_poses:
            break

    if len(feasible) < n_poses:
        raise RuntimeError(
            f"only {len(feasible)} of {n_candidates} sampled configurations satisfy the "
            "workspace constraints; loosen config/workspace.yaml or raise n_candidates")

    chosen: list[int] = []
    # Seed with a small random subset so the information matrix starts invertible.
    info = np.eye(4) * 1e-9
    for _ in range(4):
        idx = int(rng.integers(len(feasible)))
        chosen.append(idx)
        info = info + blocks[idx].T @ blocks[idx]

    while len(chosen) < n_poses:
        best_idx, best_gain = -1, -np.inf
        remaining = set(range(len(feasible))) - set(chosen)
        for idx in remaining:
            sign, logdet = np.linalg.slogdet(info + blocks[idx].T @ blocks[idx])
            gain = logdet if sign > 0 else -np.inf
            if gain > best_gain:
                best_gain, best_idx = gain, idx
        chosen.append(best_idx)
        info = info + blocks[best_idx].T @ blocks[best_idx]

    return np.array([feasible[i] for i in chosen])
