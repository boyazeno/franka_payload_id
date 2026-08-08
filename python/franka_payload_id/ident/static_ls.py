r"""Stage A: static gravity identification of mass and centre of mass.

At :math:`\dot q = \ddot q = 0` the six inertia columns of the payload regressor
vanish identically and the problem collapses to four parameters:

.. math:: \Delta\tau_{\text{static}} = Y_g(q)\,[m,\; m c_x,\; m c_y,\; m c_z]^\top

Four unknowns against roughly :math:`7 \times 50` equations, with a condition number
around 3-8. This is the workhorse: it typically pins the mass to better than 1% and
the centre of mass to 1-3 mm, and -- unlike the dynamic stage -- it is immune to
acceleration noise, errors-in-variables bias and (thanks to the bidirectional pose
approach) to Coulomb friction.

Since Franka's gravity compensation, collision thresholds and external-torque estimate
all depend only on ``m`` and the CoM, this stage alone delivers most of the practical
benefit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.preprocess import StaticDataset
from ..model import InertialParams, PandaModel, stack_gravity_regressor


@dataclass
class StaticResult:
    """Outcome of Stage A."""

    mass: float
    com: np.ndarray
    phi4: np.ndarray
    condition: float
    residual_rms_nm: float
    sigma_pct: np.ndarray
    covariance: np.ndarray
    n_poses: int
    mass_constrained: bool

    def summary(self) -> str:
        lines = [
            "Stage A (static gravity identification)",
            f"  poses               : {self.n_poses}",
            f"  mass                : {self.mass:.4f} kg"
            + ("  (constrained to the scale measurement)" if self.mass_constrained else ""),
            f"  centre of mass      : [{self.com[0]:+.4f}, {self.com[1]:+.4f}, "
            f"{self.com[2]:+.4f}] m  (flange frame)",
            f"  condition number    : {self.condition:.2f}",
            f"  residual RMS        : {self.residual_rms_nm:.4f} Nm",
            f"  relative std [%]    : "
            + ", ".join(f"{n}={v:.2f}" for n, v in
                        zip(("m", "m*cx", "m*cy", "m*cz"), self.sigma_pct)),
        ]
        return "\n".join(lines)

    def as_params(self, inertia_com: np.ndarray | None = None) -> InertialParams:
        """Promote to full :class:`InertialParams`, with a supplied inertia tensor."""
        if inertia_com is None:
            inertia_com = np.zeros((3, 3))
        return InertialParams(self.mass, self.com, inertia_com, frame="flange")


def identify_static(pm: PandaModel, dataset: StaticDataset, *,
                    mass_scale: float | None = None,
                    mass_tolerance: float = 0.02,
                    use_mass_constraint: bool = True,
                    length_scale: float = 0.1) -> StaticResult:
    r"""Weighted least squares for :math:`[m,\, m c]`.

    Parameters
    ----------
    mass_scale:
        Independently measured mass [kg]. When supplied and ``use_mass_constraint`` is
        set, the mass is fixed to it and only ``m*c`` is estimated. That barely changes
        ``m`` -- which is already the best-determined parameter -- but it removes the
        ``m`` / ``m c_z`` correlation that otherwise blurs the centre of mass when the
        CoM sits close to the flange axis.
    """
    w = stack_gravity_regressor(pm, dataset.q)                 # (7*P, 4)
    b = dataset.dtau.reshape(-1)                               # (7*P,)

    # Per-joint whitening: joints 5-7 are 12 Nm-rated and much quieter than 1-4, so
    # unweighted least squares would over-trust the big joints, which carry the least
    # information about the payload.
    weights = np.tile(1.0 / dataset.sigma, dataset.n_poses)
    wl = w * weights[:, None]
    bl = b * weights

    scale = np.array([1.0, length_scale, length_scale, length_scale])
    condition = float(np.linalg.cond(wl * scale))

    constrained = bool(use_mass_constraint and mass_scale is not None)
    if constrained:
        # Solve for m*c only, moving the known mass column to the right-hand side.
        rhs = bl - wl[:, 0] * float(mass_scale)
        sol, *_ = np.linalg.lstsq(wl[:, 1:], rhs, rcond=None)
        phi4 = np.concatenate([[float(mass_scale)], sol])
        design = wl[:, 1:]
        n_free = 3
    else:
        phi4, *_ = np.linalg.lstsq(wl, bl, rcond=None)
        design = wl
        n_free = 4

    residual = bl - wl @ phi4
    dof = max(len(bl) - n_free, 1)
    sigma_sq = float(residual @ residual) / dof
    from .validate import covariance_from_design
    cov_free = covariance_from_design(design, sigma_sq)

    covariance = np.zeros((4, 4))
    if constrained:
        covariance[1:, 1:] = cov_free
    else:
        covariance = cov_free

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pct = 100.0 * np.sqrt(np.diag(covariance)) / np.abs(phi4)
    sigma_pct = np.where(np.isfinite(sigma_pct), sigma_pct, np.inf)

    mass = float(phi4[0])
    if mass <= 0.0:
        raise ValueError(
            f"static identification returned a non-positive mass ({mass:.4f} kg). "
            "Check that the loaded and bare runs are not swapped and that the tool "
            "was actually attached for the loaded run.")

    # Residual RMS in physical units (undo the whitening) for a readable report.
    phys_residual = b - w @ phi4
    return StaticResult(
        mass=mass,
        com=phi4[1:] / mass,
        phi4=phi4,
        condition=condition,
        residual_rms_nm=float(np.sqrt(np.mean(phys_residual ** 2))),
        sigma_pct=sigma_pct,
        covariance=covariance,
        n_poses=dataset.n_poses,
        mass_constrained=constrained,
    )
