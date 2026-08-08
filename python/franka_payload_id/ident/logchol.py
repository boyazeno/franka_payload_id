r"""Stage B alternative: log-Cholesky parametrisation, no SDP solver needed.

Rucker & Wensing (RA-L 2022) parameterise the pseudo-inertia as :math:`J = U U^\top`
with :math:`U` upper triangular and positive diagonal,

.. math::
    U = e^{\alpha}\begin{bmatrix}
        e^{d_1} & s_{12} & s_{13} & t_1 \\
        0 & e^{d_2} & s_{23} & t_2 \\
        0 & 0 & e^{d_3} & t_3 \\
        0 & 0 & 0 & 1 \end{bmatrix}

so that an **unconstrained** :math:`\theta \in \mathbb{R}^{10}` maps *onto* exactly the
set of fully physically consistent inertial parameters. Physical consistency then costs
nothing: it holds by construction for every :math:`\theta`, and the problem becomes an
ordinary nonlinear least-squares one solvable with ``scipy``.

Trade-off against the SDP in :mod:`.dynamic_sdp`: no convexity guarantee, so a poor
start could in principle find a local minimum. With ten parameters and a warm start
from Stage A that is not a practical concern, and the two estimators agreeing is a
strong check that neither has a formulation bug -- which is why both are implemented.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares

from ..data.preprocess import DynamicDataset
from ..model import InertialParams, PandaModel, is_physically_consistent, pseudo_inertia, scaling_matrix
from .dynamic_sdp import DynamicResult, build_normal_equations

N_PARAMS = 10


def theta_to_phi(theta: np.ndarray) -> np.ndarray:
    """Log-Cholesky parameters to inertial parameters (always consistent)."""
    return np.asarray(
        pin.LogCholeskyParameters(np.asarray(theta, dtype=float)).toDynamicParameters()
    ).ravel()


def theta_jacobian(theta: np.ndarray) -> np.ndarray:
    """Analytic ``d phi / d theta``, shape ``(10, 10)``."""
    return np.asarray(
        pin.LogCholeskyParameters(np.asarray(theta, dtype=float)).calculateJacobian()
    )


def phi_to_theta(phi: np.ndarray) -> np.ndarray:
    r"""Inverse map, for warm starting.

    Requires ``phi`` to be physically consistent. Obtained from the *upper* Cholesky
    factor of :math:`J(\phi)`, which is the ordinary lower Cholesky factor of the
    reversed matrix, reversed back.
    """
    phi = np.asarray(phi, dtype=float).reshape(N_PARAMS)
    j = pseudo_inertia(phi)
    if np.linalg.eigvalsh(j).min() <= 0.0:
        raise ValueError(
            "cannot convert a physically inconsistent phi to log-Cholesky parameters; "
            "J(phi) must be positive definite")

    rev = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=float)
    lower = np.linalg.cholesky(rev @ j @ rev)
    upper = rev @ lower @ rev                      # upper triangular, U U^T = J

    alpha = np.log(upper[3, 3])
    v = upper / upper[3, 3]
    return np.array([
        alpha,
        np.log(v[0, 0]), np.log(v[1, 1]), np.log(v[2, 2]),
        v[0, 1], v[1, 2], v[0, 2],
        v[0, 3], v[1, 3], v[2, 3],
    ])


def identify_dynamic_logchol(pm: PandaModel, dataset: DynamicDataset, *,
                             prior: InertialParams,
                             warm_start: InertialParams | None = None,
                             length_scale: float = 0.1,
                             gamma: float = 0.0,
                             max_iter: int = 500,
                             x_tol: float = 1e-12,
                             sigma_threshold_pct: float = 30.0) -> DynamicResult:
    """Nonlinear least squares over the log-Cholesky parameters.

    ``gamma`` optionally adds a Euclidean pull towards the prior in the *scaled*
    parameters. It is not the entropic divergence used by the SDP -- physical
    consistency is already structural here, so the regulariser only needs to keep the
    unobservable directions near the prior.
    """
    w_tilde, b, condition = build_normal_equations(pm, dataset, length_scale)
    d_mat = scaling_matrix(length_scale)
    d_inv = np.linalg.inv(d_mat)

    start = (warm_start or prior).to_phi()
    if not is_physically_consistent(start):
        start = prior.to_phi()
    theta0 = phi_to_theta(start)
    prior_tilde = d_inv @ prior.to_phi()
    sqrt_gamma = np.sqrt(max(gamma, 0.0))

    def residual(theta: np.ndarray) -> np.ndarray:
        phi_tilde = d_inv @ theta_to_phi(theta)
        res = w_tilde @ phi_tilde - b
        if sqrt_gamma > 0.0:
            res = np.concatenate([res, sqrt_gamma * (phi_tilde - prior_tilde)])
        return res

    def jacobian(theta: np.ndarray) -> np.ndarray:
        dphi = theta_jacobian(theta)               # (10, 10)
        jac = w_tilde @ d_inv @ dphi
        if sqrt_gamma > 0.0:
            jac = np.vstack([jac, sqrt_gamma * (d_inv @ dphi)])
        return jac

    sol = least_squares(residual, theta0, jac=jacobian, method="lm",
                        max_nfev=int(max_iter), xtol=x_tol)
    phi = theta_to_phi(sol.x)

    res_fit = w_tilde @ (d_inv @ phi) - b
    dof = max(len(b) - N_PARAMS, 1)
    sigma_sq = float(res_fit @ res_fit) / dof
    from .validate import covariance_from_design
    cov_tilde = covariance_from_design(w_tilde, sigma_sq)
    covariance = d_mat @ cov_tilde @ d_mat

    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_pct = 100.0 * np.sqrt(np.abs(np.diag(covariance))) / np.abs(phi)
    sigma_pct = np.where(np.isfinite(sigma_pct), sigma_pct, np.inf)

    from ..model.params import PARAM_NAMES
    prior_dominated = [n for n, s in zip(PARAM_NAMES, sigma_pct) if s > sigma_threshold_pct]

    return DynamicResult(
        phi=phi,
        params=InertialParams.from_phi(phi),
        condition=condition,
        residual_rms_nm=float(np.sqrt(np.mean((res_fit * dataset.sigma.mean()) ** 2))),
        covariance=covariance,
        sigma_pct=sigma_pct,
        gamma=float(gamma),
        solver="scipy-lm/log-cholesky",
        n_equations=int(len(b)),
        physically_consistent=is_physically_consistent(phi),
        prior_dominated=prior_dominated,
    )
