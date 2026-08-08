r"""Uncertainty quantification and cross-validation.

Three things are reported, and the third exists because the first two are easy to
overstate:

**Parameter covariance.** :math:`C_\phi = \hat\sigma^2 (W^\top W)^{-1}` with
:math:`\hat\sigma^2` the residual variance, giving Gautier's relative standard
deviation :math:`\%\sigma_i = 100\,\sigma_i/\lvert\hat\phi_i\rvert`. The conventional
reading is that :math:`\%\sigma_i > 30\%` means the parameter is **not identified by
the data**; the pipeline labels those "prior-dominated" rather than presenting them as
results.

**Bootstrap over periods.** Resampling whole periods with replacement gives intervals
that make no assumption about the noise being white or Gaussian. This is the honest
number when the residual is visibly structured.

**The over-sampling correction.** The covariance formula above assumes independent
rows. At 1 kHz with a 10 Hz cut-off, adjacent samples are nearly perfectly correlated,
so an un-decimated fit overstates the degrees of freedom by :math:`f_s/(2 f_c)` and the
reported :math:`\sigma` is optimistic by its square root -- roughly a factor of seven.
:func:`effective_sample_correction` computes that factor so it can be applied (or, as
the pipeline does by default, so decimation can be verified to have happened).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..data.preprocess import DynamicDataset
from ..model import PandaModel, stack_regressor
from ..model.params import PARAM_NAMES


@dataclass
class ParameterUncertainty:
    phi: np.ndarray
    sigma: np.ndarray
    sigma_pct: np.ndarray
    identified: np.ndarray            # bool per parameter
    threshold_pct: float

    def summary(self) -> str:
        lines = ["parameter uncertainty (Gautier %sigma)"]
        for name, value, s, pct, ok in zip(PARAM_NAMES, self.phi, self.sigma,
                                           self.sigma_pct, self.identified):
            flag = "" if ok else "   <-- prior-dominated"
            lines.append(f"  {name:>5s} = {value:+.6e}  +/- {s:.2e}  ({pct:6.1f} %){flag}")
        return "\n".join(lines)


def covariance_from_design(design: np.ndarray, sigma_sq: float,
                           rank_tol: float = 1e-12) -> np.ndarray:
    r"""``sigma^2 (W^T W)^{-1}`` computed through the SVD, without truncation.

    ``numpy.linalg.pinv`` discards singular values below its default cutoff, which for
    a covariance is exactly the wrong behaviour: a direction the data does **not**
    determine comes back with a *small* variance instead of an enormous one, so an
    unidentifiable parameter looks beautifully determined. Here the variance along a
    near-null direction is allowed to diverge, which is what makes the
    ``%sigma > 30 %`` rule able to catch it.

    With :math:`W = U S V^\top`, :math:`(W^\top W)^{-1} = V S^{-2} V^\top`.
    """
    design = np.asarray(design, dtype=float)
    _, singular, vt = np.linalg.svd(design, full_matrices=False)
    scale = np.max(singular) if singular.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_sq = np.where(singular > rank_tol * max(scale, 1e-300),
                          1.0 / np.maximum(singular, 1e-300) ** 2, np.inf)
    return sigma_sq * (vt.T * inv_sq) @ vt


def parameter_uncertainty(residual: np.ndarray, design: np.ndarray, phi: np.ndarray,
                          *, threshold_pct: float = 30.0,
                          scaling: np.ndarray | None = None) -> ParameterUncertainty:
    """Covariance-based uncertainty from a solved least-squares problem.

    ``design`` is the (possibly scaled and whitened) regressor actually used; if it was
    column-scaled by ``D``, pass ``scaling=D`` so the result is reported in physical
    units.
    """
    residual = np.asarray(residual, dtype=float).ravel()
    design = np.asarray(design, dtype=float)
    n_params = design.shape[1]
    dof = max(residual.size - n_params, 1)
    sigma_sq = float(residual @ residual) / dof
    cov = covariance_from_design(design, sigma_sq)
    if scaling is not None:
        cov = scaling @ cov @ scaling
    sigma = np.sqrt(np.abs(np.diag(cov)))

    with np.errstate(divide="ignore", invalid="ignore"):
        pct = 100.0 * sigma / np.abs(phi)
    pct = np.where(np.isfinite(pct), pct, np.inf)
    return ParameterUncertainty(phi=np.asarray(phi), sigma=sigma, sigma_pct=pct,
                                identified=pct <= threshold_pct,
                                threshold_pct=float(threshold_pct))


def effective_sample_correction(sample_rate_hz: float, cutoff_hz: float) -> float:
    r"""Factor by which an un-decimated fit understates :math:`\sigma`.

    Returns :math:`\sqrt{f_s / (2 f_c)}`, or 1.0 once the data has been decimated to
    twice the cut-off.
    """
    ratio = sample_rate_hz / (2.0 * cutoff_hz)
    return float(np.sqrt(max(ratio, 1.0)))


def bootstrap_parameters(estimator, dataset: DynamicDataset, *, n_samples: int = 200,
                         seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap over whole periods.

    ``estimator(dataset) -> phi``. Periods are resampled with replacement, which keeps
    the within-period correlation structure intact -- resampling individual rows would
    destroy it and give intervals that are far too tight.

    Returns ``(mean, std, percentile_2p5_97p5)`` with shapes ``(10,)``, ``(10,)`` and
    ``(2, 10)``.
    """
    rng = np.random.default_rng(seed)
    groups = np.unique(dataset.period_index)
    if groups.size < 2:
        raise ValueError(
            "bootstrap needs at least two independent blocks; merge several "
            "trajectories or keep the per-period structure in period_index")

    draws = []
    for _ in range(int(n_samples)):
        picked = rng.choice(groups, size=groups.size, replace=True)
        rows = np.concatenate([np.flatnonzero(dataset.period_index == g) for g in picked])
        resampled = DynamicDataset(
            q=dataset.q[rows], qd=dataset.qd[rows], qdd=dataset.qdd[rows],
            dtau=dataset.dtau[rows], sigma=dataset.sigma, mask=dataset.mask[rows],
            period_index=np.zeros(rows.size, dtype=int),
            sample_rate_hz=dataset.sample_rate_hz, meta=dict(dataset.meta))
        try:
            draws.append(np.asarray(estimator(resampled)).ravel())
        except Exception:
            continue

    if len(draws) < 2:
        raise RuntimeError("bootstrap produced too few successful fits")
    stacked = np.array(draws)
    return (stacked.mean(axis=0), stacked.std(axis=0, ddof=1),
            np.percentile(stacked, [2.5, 97.5], axis=0))


@dataclass
class ValidationReport:
    rmse_per_joint: np.ndarray
    relative_per_joint: np.ndarray
    r_squared_per_joint: np.ndarray
    overall_rmse: float
    n_equations: int
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(np.all(self.relative_per_joint <= 0.15))

    def summary(self) -> str:
        lines = [f"cross-validation on held-out data ({self.n_equations} equations)",
                 f"  overall RMSE : {self.overall_rmse:.4f} Nm"]
        for j in range(len(self.rmse_per_joint)):
            lines.append(f"  joint {j + 1}: RMSE {self.rmse_per_joint[j]:.4f} Nm  "
                         f"rel {self.relative_per_joint[j]:.3f}  "
                         f"R^2 {self.r_squared_per_joint[j]:.3f}")
        for n in self.notes:
            lines.append(f"  NOTE: {n}")
        return "\n".join(lines)


def cross_validate(pm: PandaModel, dataset: DynamicDataset, phi: np.ndarray) -> ValidationReport:
    """Predict a held-out trajectory's torque difference and score it per joint."""
    w = stack_regressor(pm, dataset.q, dataset.qd, dataset.qdd)
    predicted = (w @ np.asarray(phi)).reshape(-1, 7)
    measured = dataset.dtau
    mask = dataset.mask

    n_joints = measured.shape[1]
    rmse = np.zeros(n_joints)
    rel = np.zeros(n_joints)
    r2 = np.zeros(n_joints)
    notes: list[str] = []

    for j in range(n_joints):
        sel = mask[:, j]
        if sel.sum() < 2:
            rmse[j] = np.nan
            rel[j] = np.nan
            r2[j] = np.nan
            notes.append(f"joint {j + 1}: too few usable samples to score")
            continue
        residual = measured[sel, j] - predicted[sel, j]
        rmse[j] = float(np.sqrt(np.mean(residual ** 2)))
        spread = float(np.std(measured[sel, j]))
        rel[j] = rmse[j] / spread if spread > 0 else np.inf
        r2[j] = 1.0 - rel[j] ** 2 if np.isfinite(rel[j]) else -np.inf

    overall = float(np.sqrt(np.nanmean(rmse ** 2)))
    return ValidationReport(rmse_per_joint=rmse, relative_per_joint=rel,
                            r_squared_per_joint=r2, overall_rmse=overall,
                            n_equations=int(mask.sum()), notes=notes)


def friction_residual_diagnostic(dataset: DynamicDataset, pm: PandaModel,
                                 phi: np.ndarray) -> dict[str, np.ndarray]:
    r"""Test whether friction really cancelled in the difference.

    Regresses the fit residual of each joint onto :math:`[\operatorname{sign}(\dot q),
    \dot q]`. A significant ``coulomb`` term means the two runs did not traverse the
    trajectory identically (or the thermal state differed); a significant ``viscous``
    term means the same for the velocity-proportional part. Both should be small
    compared with the torque noise -- this is the check that makes a silent failure of
    the difference method visible.
    """
    w = stack_regressor(pm, dataset.q, dataset.qd, dataset.qdd)
    residual = (dataset.dtau.reshape(-1) - w @ np.asarray(phi)).reshape(-1, 7)

    coulomb = np.zeros(7)
    viscous = np.zeros(7)
    for j in range(7):
        sel = dataset.mask[:, j]
        if sel.sum() < 3:
            coulomb[j] = viscous[j] = np.nan
            continue
        design = np.column_stack([np.sign(dataset.qd[sel, j]), dataset.qd[sel, j]])
        sol, *_ = np.linalg.lstsq(design, residual[sel, j], rcond=None)
        coulomb[j], viscous[j] = sol
    return {"coulomb_nm": coulomb, "viscous_nms": viscous,
            "noise_nm": dataset.sigma.copy()}
