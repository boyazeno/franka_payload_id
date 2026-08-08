r"""Stage B: physically-consistent dynamic identification of all ten parameters.

The estimator is a weighted least-squares problem subject to the pseudo-inertia LMI,

.. math::
    \min_{\phi}\;\; \lVert \Sigma^{-1/2}(W\phi - b)\rVert_2^2
        \;+\; \gamma\,\big[\operatorname{tr}(J_0^{-1}J(\phi)) - \log\det J(\phi)\big]
    \quad\text{s.t.}\quad J(\phi) \succeq \epsilon I,\;
        m \in [\underline m, \overline m],\;
        \operatorname{tr}(J(\phi)Q) \ge 0

Why this particular formulation:

* :math:`J(\phi) \succeq 0` is *the* condition for ``phi`` to be realisable by some
  non-negative mass density (Wensing, Kim & Slotine, RA-L 2018). Because
  :math:`J` is affine in :math:`\phi`, it is a linear matrix inequality, and it
  subsumes ``m > 0``, positive-definiteness of the inertia about the CoM, **and** the
  triangle inequalities on the principal moments in one constraint.
* The bracketed term is the Bregman divergence of :math:`-\log\det`, i.e. the KL
  divergence between zero-mean Gaussians with covariances :math:`J` and :math:`J_0`.
  It is convex, blows up as :math:`J` approaches the boundary (so it also acts as a
  barrier), and -- unlike a Euclidean penalty :math:`\lVert\phi-\phi_0\rVert^2` -- it
  is invariant to the choice of coordinates on the parameter space.
* For a small tool the inertia terms are barely observable (their torque signature sits
  well below the sensor noise floor, and the parallel-axis term ``m|c|^2`` dominates
  the tool's own ``I_C``). The prior means those directions degrade gracefully towards
  the bounding-box value instead of towards noise.

Numerics: the whole problem is solved in the non-dimensionalised variable
:math:`\tilde\phi = D^{-1}\phi`, and the LMI is written on the congruent matrix
:math:`S J S` with :math:`S = \operatorname{diag}(L^{-1}, L^{-1}, L^{-1}, 1)`.
Congruence preserves definiteness while removing the ~10^3 spread between the mass and
inertia blocks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.preprocess import DynamicDataset
from ..model import (
    InertialParams,
    PandaModel,
    bounding_ellipsoid_matrix,
    is_physically_consistent,
    pseudo_inertia,
    pseudo_inertia_basis,
    scaling_matrix,
    stack_regressor,
)

N_PARAMS = 10


@dataclass
class DynamicResult:
    phi: np.ndarray
    params: InertialParams
    condition: float
    residual_rms_nm: float
    covariance: np.ndarray
    sigma_pct: np.ndarray
    gamma: float
    solver: str
    n_equations: int
    physically_consistent: bool
    prior_dominated: list[str] = field(default_factory=list)
    prior_shift_sigma: np.ndarray | None = None
    """How far the regulariser moved each parameter, in units of its data-only sigma.

    ``%sigma`` measures *variance* and is blind to the *bias* a prior introduces, so a
    parameter can look well determined while actually being pulled most of the way to
    the prior. This is the missing half of the picture: it is computed by solving a
    second time with ``gamma = 0`` and comparing. A shift above ~1 sigma means the
    reported value owes more to the prior than to the data.
    """

    def summary(self) -> str:
        from ..model.params import PARAM_NAMES
        lines = [
            "Stage B (dynamic, physically-consistent)",
            f"  equations           : {self.n_equations}",
            f"  condition (scaled)  : {self.condition:.2f}",
            f"  residual RMS        : {self.residual_rms_nm:.4f} Nm",
            f"  gamma               : {self.gamma:g}   solver: {self.solver}",
            f"  physically consistent: {self.physically_consistent}",
            f"  mass                : {self.params.mass:.4f} kg",
            f"  centre of mass      : [{self.params.com[0]:+.4f}, {self.params.com[1]:+.4f},"
            f" {self.params.com[2]:+.4f}] m",
            "  relative std [%]    : "
            + ", ".join(f"{n}={v:.1f}" for n, v in zip(PARAM_NAMES, self.sigma_pct)),
        ]
        if self.prior_shift_sigma is not None:
            lines.append("  prior shift [sigma] : "
                         + ", ".join(f"{n}={v:.1f}" for n, v
                                     in zip(PARAM_NAMES, self.prior_shift_sigma)))
        if self.prior_dominated:
            lines.append("  PRIOR-DOMINATED (not determined by the data): "
                         + ", ".join(self.prior_dominated))
        return "\n".join(lines)


def build_normal_equations(pm: PandaModel, dataset: DynamicDataset,
                           length_scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Assemble the whitened, column-scaled system ``(W_tilde, b_tilde, cond)``.

    Rows where the mask is False (near-zero joint velocity) are dropped, and every row
    is divided by that joint's noise standard deviation.
    """
    w_full = stack_regressor(pm, dataset.q, dataset.qd, dataset.qdd)   # (7K, 10)
    b_full = dataset.dtau.reshape(-1)                                   # (7K,)
    weights = np.tile(1.0 / dataset.sigma, dataset.n_samples)
    mask = dataset.mask.reshape(-1)

    w = w_full[mask] * weights[mask, None]
    b = b_full[mask] * weights[mask]
    if w.shape[0] < N_PARAMS:
        raise ValueError(
            f"only {w.shape[0]} usable equations for {N_PARAMS} parameters; the "
            "zero-velocity threshold may be discarding too much data")

    w_tilde = w @ scaling_matrix(length_scale)
    return w_tilde, b, float(np.linalg.cond(w_tilde))


def _congruence(length_scale: float) -> np.ndarray:
    s = np.diag([1.0 / length_scale, 1.0 / length_scale, 1.0 / length_scale, 1.0])
    return s


def _scaled_basis(length_scale: float) -> np.ndarray:
    """Basis of ``S J(D phi_tilde) S`` as a function of the scaled parameters."""
    s = _congruence(length_scale)
    d = scaling_matrix(length_scale)
    basis = pseudo_inertia_basis()                     # (10, 4, 4), J = sum_k phi_k B_k
    # phi = D phi_tilde  =>  J = sum_k (D phi_tilde)_k B_k = sum_j phi_tilde_j (D_jj B_j)
    return np.stack([s @ (d[j, j] * basis[j]) @ s for j in range(N_PARAMS)])


def _huber_weights(residual: np.ndarray, delta_scale: float) -> np.ndarray:
    """IRLS weights for the Huber loss, with a robust scale estimate."""
    scale = 1.4826 * np.median(np.abs(residual - np.median(residual)))
    if scale <= 0.0:
        return np.ones_like(residual)
    delta = delta_scale * scale
    absr = np.abs(residual)
    return np.where(absr <= delta, 1.0, delta / np.maximum(absr, 1e-12))


def identify_dynamic_sdp(pm: PandaModel, dataset: DynamicDataset, *,
                         prior: InertialParams,
                         length_scale: float = 0.1,
                         gamma: float = 1e-2,
                         solver: str = "CLARABEL",
                         psd_epsilon: float = 1e-9,
                         use_entropic_prior: bool = True,
                         use_bounding_ellipsoid: bool = False,
                         bbox: tuple[np.ndarray, np.ndarray] | None = None,
                         mass_bounds: tuple[float, float] | None = None,
                         robust: str = "none",
                         huber_delta_scale: float = 1.345,
                         irls_iterations: int = 3,
                         sigma_threshold_pct: float = 30.0,
                         _measure_prior_shift: bool = True) -> DynamicResult:
    """Solve the constrained problem. Requires ``cvxpy``."""
    try:
        import cvxpy as cp
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError(
            "cvxpy is required for the SDP estimator. Install it, or use "
            "ident.logchol.identify_dynamic_logchol, which needs only scipy."
        ) from exc

    w_tilde, b, condition = build_normal_equations(pm, dataset, length_scale)
    d_mat = scaling_matrix(length_scale)
    basis_s = _scaled_basis(length_scale)
    s_mat = _congruence(length_scale)

    prior_phi = prior.to_phi()
    j0_scaled = s_mat @ pseudo_inertia(prior_phi) @ s_mat
    if np.linalg.eigvalsh(j0_scaled).min() <= 0.0:
        raise ValueError("the prior is not physically consistent; J_0 must be positive definite")
    j0_inv = np.linalg.inv(j0_scaled)

    q_ellipsoid = None
    if use_bounding_ellipsoid:
        if bbox is None:
            raise ValueError("use_bounding_ellipsoid requires the tool bounding box")
        # tr(J Q) = tr(S^-1 (S J S) S^-1 Q) = tr((S J S) (S^-1 Q S^-1))
        s_inv = np.linalg.inv(s_mat)
        q_ellipsoid = s_inv @ bounding_ellipsoid_matrix(*bbox) @ s_inv

    row_weights = np.ones(b.shape[0])
    last_solver_used = solver
    phi_tilde_value: np.ndarray | None = None

    n_irls = irls_iterations if robust == "huber" else 1
    for _ in range(max(1, int(n_irls))):
        phi_t = cp.Variable(N_PARAMS, name="phi_tilde")
        j_expr = sum(phi_t[k] * basis_s[k] for k in range(N_PARAMS))

        sqrt_w = np.sqrt(row_weights)
        objective = cp.sum_squares(cp.multiply(sqrt_w, w_tilde @ phi_t - b))
        if use_entropic_prior and gamma > 0.0:
            objective = objective + gamma * (cp.trace(j0_inv @ j_expr) - cp.log_det(j_expr))

        constraints = [j_expr >> psd_epsilon * np.eye(4)]
        if mass_bounds is not None:
            lo, hi = mass_bounds
            constraints += [phi_t[0] * d_mat[0, 0] >= lo, phi_t[0] * d_mat[0, 0] <= hi]
        if q_ellipsoid is not None:
            constraints.append(cp.trace(j_expr @ q_ellipsoid) >= 0)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(solver=getattr(cp, solver))
        except Exception:
            problem.solve(solver=cp.SCS)
            last_solver_used = "SCS"
        if phi_t.value is None:
            problem.solve(solver=cp.SCS)
            last_solver_used = "SCS"
        if phi_t.value is None:
            raise RuntimeError(f"SDP failed to solve (status {problem.status})")

        phi_tilde_value = np.asarray(phi_t.value).ravel()
        if robust == "huber":
            row_weights = _huber_weights(w_tilde @ phi_tilde_value - b, huber_delta_scale)

    assert phi_tilde_value is not None
    phi = d_mat @ phi_tilde_value

    residual = w_tilde @ phi_tilde_value - b
    dof = max(len(b) - N_PARAMS, 1)
    sigma_sq = float(residual @ residual) / dof
    # SVD-based, deliberately not pinv: a direction the data does not determine must
    # come back with a huge variance, not a truncated-to-zero one.
    from .validate import covariance_from_design
    cov_tilde = covariance_from_design(w_tilde, sigma_sq)
    covariance = d_mat @ cov_tilde @ d_mat

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pct = 100.0 * np.sqrt(np.abs(np.diag(covariance))) / np.abs(phi)
    sigma_pct = np.where(np.isfinite(sigma_pct), sigma_pct, np.inf)

    from ..model.params import PARAM_NAMES

    # Second solve with the regulariser switched off, to measure how far the prior
    # moved the answer. Variance alone cannot reveal that: a shrunk parameter has a
    # SMALL standard deviation precisely because the prior is holding it in place.
    prior_shift_sigma = None
    if _measure_prior_shift and use_entropic_prior and gamma > 0.0:
        try:
            unregularised = identify_dynamic_sdp(
                pm, dataset, prior=prior, length_scale=length_scale, gamma=0.0,
                solver=solver, psd_epsilon=psd_epsilon, use_entropic_prior=False,
                use_bounding_ellipsoid=use_bounding_ellipsoid, bbox=bbox,
                mass_bounds=mass_bounds, robust=robust,
                huber_delta_scale=huber_delta_scale, irls_iterations=irls_iterations,
                sigma_threshold_pct=sigma_threshold_pct, _measure_prior_shift=False)
            data_sigma = np.sqrt(np.abs(np.diag(covariance)))
            with np.errstate(divide="ignore", invalid="ignore"):
                prior_shift_sigma = np.abs(phi - unregularised.phi) / data_sigma
            prior_shift_sigma = np.where(np.isfinite(prior_shift_sigma),
                                         prior_shift_sigma, np.inf)
        except Exception:
            prior_shift_sigma = None

    prior_dominated = [name for name, s in zip(PARAM_NAMES, sigma_pct)
                       if s > sigma_threshold_pct]
    if prior_shift_sigma is not None:
        for name, shift in zip(PARAM_NAMES, prior_shift_sigma):
            if shift > 1.0 and name not in prior_dominated:
                prior_dominated.append(name)

    # Report the residual in physical units, undoing the whitening.
    phys_rms = float(np.sqrt(np.mean((residual * dataset.sigma.mean()) ** 2)))

    return DynamicResult(
        phi=phi,
        params=InertialParams.from_phi(phi),
        condition=condition,
        residual_rms_nm=phys_rms,
        covariance=covariance,
        sigma_pct=sigma_pct,
        gamma=float(gamma),
        solver=last_solver_used,
        n_equations=int(len(b)),
        physically_consistent=is_physically_consistent(phi),
        prior_dominated=prior_dominated,
        prior_shift_sigma=prior_shift_sigma,
    )


def select_gamma(pm: PandaModel, datasets: list[DynamicDataset], *,
                 prior: InertialParams, gamma_grid: list[float],
                 **kwargs) -> tuple[float, dict[float, float]]:
    """Leave-one-trajectory-out selection of the regularisation weight.

    Needs at least two datasets. The score is the mean prediction error on the
    held-out trajectory, so it measures generalisation rather than fit.
    """
    if len(datasets) < 2:
        raise ValueError("gamma selection needs at least two independent datasets")

    scores: dict[float, float] = {}
    for gamma in gamma_grid:
        errors = []
        for held_out in range(len(datasets)):
            train = [d for i, d in enumerate(datasets) if i != held_out]
            merged = _merge(train)
            result = identify_dynamic_sdp(pm, merged, prior=prior, gamma=gamma, **kwargs)
            errors.append(prediction_rmse(pm, datasets[held_out], result.phi))
        scores[gamma] = float(np.mean(errors))
    best = min(scores, key=scores.get)
    return best, scores


def _merge(datasets: list[DynamicDataset]) -> DynamicDataset:
    """Concatenate datasets that share a sampling rate."""
    if not datasets:
        raise ValueError("nothing to merge")
    rate = datasets[0].sample_rate_hz
    if any(abs(d.sample_rate_hz - rate) > 1e-9 for d in datasets):
        raise ValueError("cannot merge datasets recorded at different rates")
    return DynamicDataset(
        q=np.vstack([d.q for d in datasets]),
        qd=np.vstack([d.qd for d in datasets]),
        qdd=np.vstack([d.qdd for d in datasets]),
        dtau=np.vstack([d.dtau for d in datasets]),
        sigma=np.mean([d.sigma for d in datasets], axis=0),
        mask=np.vstack([d.mask for d in datasets]),
        period_index=np.concatenate([np.full(d.n_samples, i) for i, d in enumerate(datasets)]),
        sample_rate_hz=rate,
        meta={"merged_from": len(datasets)},
    )


def prediction_rmse(pm: PandaModel, dataset: DynamicDataset, phi: np.ndarray) -> float:
    """Torque-prediction RMSE [Nm] of ``phi`` on ``dataset`` (masked rows only)."""
    w = stack_regressor(pm, dataset.q, dataset.qd, dataset.qdd)
    b = dataset.dtau.reshape(-1)
    mask = dataset.mask.reshape(-1)
    residual = (w @ phi - b)[mask]
    return float(np.sqrt(np.mean(residual ** 2)))
