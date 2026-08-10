r"""Algebra of the ten inertial parameters.

Parameter vector (Pinocchio's ``Inertia.toDynamicParameters`` convention)::

    phi = [ m, m*cx, m*cy, m*cz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz ]

Two things about this layout bite people, and both are pinned down by tests:

1. **The ordering of the six inertia entries is ``xx, xy, yy, xz, yz, zz``** -- the
   column-major lower-triangular layout of ``pinocchio::Symmetric3`` -- and *not* the
   more common ``xx, xy, xz, yy, yz, zz``. Getting it wrong silently swaps ``Iyy``
   with ``Ixz``.
2. **Those six entries are the inertia about the FRAME ORIGIN**, written
   :math:`\bar I`, not about the centre of mass. Franka's Desk fields want the inertia
   about the CoM. The two are related by the parallel-axis theorem

   .. math:: \bar I = I_C + m\,(\lVert c\rVert^2 I_3 - c c^\top)

See ``docs/THEORY.md`` section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_PARAMS = 10

# Index of (row, col) of the symmetric inertia matrix within phi[4:].
_TRIU_ORDER: tuple[tuple[int, int], ...] = (
    (0, 0),  # Ixx  -> phi[4]
    (0, 1),  # Ixy  -> phi[5]
    (1, 1),  # Iyy  -> phi[6]
    (0, 2),  # Ixz  -> phi[7]
    (1, 2),  # Iyz  -> phi[8]
    (2, 2),  # Izz  -> phi[9]
)

PARAM_NAMES: tuple[str, ...] = (
    "m", "m*cx", "m*cy", "m*cz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz",
)


def inertia_matrix_from_vec6(vec6: np.ndarray) -> np.ndarray:
    """Expand the six-vector ``[Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`` into a 3x3 tensor."""
    vec6 = np.asarray(vec6, dtype=float)
    out = np.zeros((3, 3), dtype=float)
    for value, (i, j) in zip(vec6, _TRIU_ORDER):
        out[i, j] = value
        out[j, i] = value
    return out


def vec6_from_inertia_matrix(mat: np.ndarray) -> np.ndarray:
    """Inverse of :func:`inertia_matrix_from_vec6`."""
    mat = np.asarray(mat, dtype=float)
    return np.array([mat[i, j] for (i, j) in _TRIU_ORDER], dtype=float)


def inertia_about_com(inertia_origin: np.ndarray, mass: float, com: np.ndarray) -> np.ndarray:
    r"""Parallel-axis shift from the frame origin to the centre of mass.

    :math:`I_C = \bar I - m(\lVert c\rVert^2 I_3 - c c^\top)`
    """
    com = np.asarray(com, dtype=float)
    steiner = mass * (float(com @ com) * np.eye(3) - np.outer(com, com))
    return np.asarray(inertia_origin, dtype=float) - steiner


def inertia_about_origin(inertia_com: np.ndarray, mass: float, com: np.ndarray) -> np.ndarray:
    r""":math:`\bar I = I_C + m(\lVert c\rVert^2 I_3 - c c^\top)`."""
    com = np.asarray(com, dtype=float)
    steiner = mass * (float(com @ com) * np.eye(3) - np.outer(com, com))
    return np.asarray(inertia_com, dtype=float) + steiner


@dataclass(frozen=True)
class InertialParams:
    """A payload's inertial parameters in a named frame.

    Attributes
    ----------
    mass:
        kg.
    com:
        Centre of mass w.r.t. the frame origin, in frame axes [m]. This is exactly
        Franka's ``F_x_Cload`` / ``F_x_Cee`` when ``frame == "flange"``.
    inertia_com:
        3x3 inertia tensor **about the centre of mass**, in frame axes [kg m^2].
        This is exactly Franka's ``I_load`` / ``I_ee``.
    """

    mass: float
    com: np.ndarray
    inertia_com: np.ndarray
    frame: str = "flange"

    def __post_init__(self) -> None:
        object.__setattr__(self, "com", np.asarray(self.com, dtype=float).reshape(3))
        object.__setattr__(self, "inertia_com",
                           np.asarray(self.inertia_com, dtype=float).reshape(3, 3))

    @property
    def inertia_origin(self) -> np.ndarray:
        """Inertia about the frame origin (what ``phi[4:]`` holds)."""
        return inertia_about_origin(self.inertia_com, self.mass, self.com)

    def to_phi(self) -> np.ndarray:
        return phi_from_mci(self.mass, self.com, self.inertia_com)

    @staticmethod
    def from_phi(phi: np.ndarray, frame: str = "flange") -> "InertialParams":
        m, c, ic = phi_to_mci(phi)
        return InertialParams(m, c, ic, frame=frame)

    def translated(self, translation: np.ndarray, frame: str) -> "InertialParams":
        """Express the same body in a frame offset by ``translation`` (no rotation).

        ``translation`` is the position of the NEW frame's origin expressed in the
        current frame. This is all that is needed to go from Pinocchio's joint-7 frame
        to Franka's flange frame, which differ by exactly ``(0, 0, 0.107)``.
        """
        t = np.asarray(translation, dtype=float).reshape(3)
        # The inertia about the CoM is invariant under a pure translation of the frame.
        return InertialParams(self.mass, self.com - t, self.inertia_com.copy(), frame=frame)

    def as_desk_fields(self) -> dict[str, object]:
        """Values in the exact shape Desk / ``setLoad`` expect.

        ``inertia`` is column-major flattened, about the CoM, in flange axes.
        """
        return {
            "mass": float(self.mass),
            "com": [float(v) for v in self.com],
            "inertia_column_major": [float(v) for v in self.inertia_com.flatten(order="F")],
            "inertia_matrix": [[float(v) for v in row] for row in self.inertia_com],
            "frame": self.frame,
        }


def phi_from_mci(mass: float, com: np.ndarray, inertia_com: np.ndarray) -> np.ndarray:
    """Pack ``(m, c, I_C)`` into the 10-vector, applying the parallel-axis shift."""
    com = np.asarray(com, dtype=float).reshape(3)
    ibar = inertia_about_origin(inertia_com, mass, com)
    phi = np.empty(N_PARAMS, dtype=float)
    phi[0] = mass
    phi[1:4] = mass * com
    phi[4:] = vec6_from_inertia_matrix(ibar)
    return phi


def phi_to_mci(phi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Unpack the 10-vector into ``(m, c, I_C)``.

    Raises
    ------
    ValueError
        If the mass is not strictly positive, since the CoM would be undefined.
    """
    phi = np.asarray(phi, dtype=float).reshape(N_PARAMS)
    mass = float(phi[0])
    if mass <= 0.0:
        raise ValueError(f"non-positive mass {mass!r}; cannot recover a centre of mass")
    com = phi[1:4] / mass
    ibar = inertia_matrix_from_vec6(phi[4:])
    return mass, com, inertia_about_com(ibar, mass, com)


# ---------------------------------------------------------------------------
# Frame changes (pure translation)
# ---------------------------------------------------------------------------
def translate_phi(phi: np.ndarray, translation: np.ndarray) -> np.ndarray:
    r"""Re-express ``phi`` in a frame whose origin is offset by ``-translation``.

    ``translation`` is the position of the *current* frame's origin as seen from the
    *new* frame, so ``c_new = c_old + translation``. Everything is linear in ``phi``,
    which lets this be applied to basis vectors with zero mass:

    .. math::
        m' &= m \\
        h' &= h + m\,t \\
        \bar I' &= \bar I + (2\,h\!\cdot\!t + m\lVert t\rVert^2) I_3
                   - (h t^\top + t h^\top + m\, t t^\top)

    Only valid for translations; the pipeline never needs a rotation because the
    joint-7 and flange frames share their orientation.
    """
    phi = np.asarray(phi, dtype=float).reshape(N_PARAMS)
    t = np.asarray(translation, dtype=float).reshape(3)
    m = phi[0]
    h = phi[1:4]
    ibar = inertia_matrix_from_vec6(phi[4:])

    h_new = h + m * t
    ibar_new = (ibar
                + (2.0 * float(h @ t) + m * float(t @ t)) * np.eye(3)
                - (np.outer(h, t) + np.outer(t, h) + m * np.outer(t, t)))

    out = np.empty(N_PARAMS, dtype=float)
    out[0] = m
    out[1:4] = h_new
    out[4:] = vec6_from_inertia_matrix(ibar_new)
    return out


def translation_map(translation: np.ndarray) -> np.ndarray:
    """10x10 matrix ``T`` with ``translate_phi(phi, t) == T @ phi``."""
    t = np.asarray(translation, dtype=float).reshape(3)
    cols = [translate_phi(np.eye(N_PARAMS)[k], t) for k in range(N_PARAMS)]
    return np.column_stack(cols)


# ---------------------------------------------------------------------------
# Pseudo-inertia and physical consistency
# ---------------------------------------------------------------------------
def pseudo_inertia(phi: np.ndarray) -> np.ndarray:
    r"""The 4x4 pseudo-inertia matrix :math:`J(\phi)`.

    .. math::
        J(\phi) = \begin{bmatrix} \Sigma & h \\ h^\top & m \end{bmatrix},
        \qquad \Sigma = \tfrac12 \operatorname{tr}(\bar I) I_3 - \bar I,
        \qquad h = m c

    :math:`J(\phi)` is the second-moment matrix of the mass density in homogeneous
    coordinates, so :math:`J \succ 0` holds **iff** ``phi`` is realisable by some
    non-negative mass density (Wensing, Kim & Slotine, RA-L 2018). Because it is
    affine in ``phi``, that condition is a linear matrix inequality, and it subsumes
    ``m > 0``, ``I_C > 0`` *and* the triangle inequalities on the principal moments.
    """
    phi = np.asarray(phi, dtype=float).reshape(N_PARAMS)
    ibar = inertia_matrix_from_vec6(phi[4:])
    sigma = 0.5 * np.trace(ibar) * np.eye(3) - ibar
    out = np.zeros((4, 4), dtype=float)
    out[:3, :3] = sigma
    out[:3, 3] = phi[1:4]
    out[3, :3] = phi[1:4]
    out[3, 3] = phi[0]
    return out


def pseudo_inertia_basis() -> np.ndarray:
    r"""Basis ``B`` with ``J(phi) == sum_k phi[k] * B[k]``, shape ``(10, 4, 4)``.

    Used to build ``J`` as an affine cvxpy expression without hand-writing the
    entries, so the numpy and cvxpy paths cannot drift apart.
    """
    basis = np.zeros((N_PARAMS, 4, 4), dtype=float)
    for k in range(N_PARAMS):
        e = np.zeros(N_PARAMS)
        e[k] = 1.0
        basis[k] = pseudo_inertia(e)
    return basis


def is_physically_consistent(phi: np.ndarray, rtol: float = 1e-12) -> bool:
    r"""True iff ``J(phi)`` is positive semi-definite to a **relative** tolerance.

    Two deliberate choices:

    *Semi*-definite, not strictly definite. :math:`J \succ 0` characterises bodies with
    a genuinely three-dimensional mass density; the closure :math:`J \succeq 0` also
    admits degenerate ones -- a point mass (rank 1), a thin rod, a flat plate. Those are
    perfectly realizable tools, and a fitted parameter vector legitimately lands on that
    boundary when the data prefers a nearly planar distribution.

    *Relative* tolerance. ``J`` mixes a mass of order 1 with second moments of order
    1e-4, so testing an eigenvalue against absolute zero is scale-dependent and fails at
    machine epsilon: a boundary solution produced by the log-Cholesky parameterisation
    -- which is consistent *by construction* -- comes back with a minimum eigenvalue of
    around -1e-17 and would be rejected. The threshold therefore scales with the largest
    eigenvalue. ``rtol`` sits far above machine epsilon and far below any physically
    meaningful violation.
    """
    try:
        eigvals = np.linalg.eigvalsh(pseudo_inertia(phi))
    except np.linalg.LinAlgError:
        return False
    scale = max(float(eigvals.max()), 0.0)
    return bool(eigvals.min() >= -rtol * max(scale, 1e-300))


def consistency_report(phi: np.ndarray) -> dict[str, object]:
    """Human-readable breakdown of *why* a parameter vector is or is not consistent."""
    j = pseudo_inertia(phi)
    eig_j = np.linalg.eigvalsh(j)
    out: dict[str, object] = {
        "mass_positive": bool(phi[0] > 0.0),
        "pseudo_inertia_min_eig": float(eig_j.min()),
        # Relative, for the reasons in is_physically_consistent.
        "pseudo_inertia_min_eig_relative": float(eig_j.min() / max(eig_j.max(), 1e-300)),
        "physically_consistent": is_physically_consistent(phi),
    }
    if phi[0] > 0.0:
        _, _, ic = phi_to_mci(phi)
        moments = np.linalg.eigvalsh(ic)
        out["principal_moments"] = [float(v) for v in moments]
        out["inertia_com_pd"] = bool(moments.min() > 0.0)
        a, b, c = sorted(moments)
        out["triangle_inequality"] = bool(a + b >= c)
    return out


# ---------------------------------------------------------------------------
# Column scaling
# ---------------------------------------------------------------------------
def scaling_matrix(length_scale: float) -> np.ndarray:
    r"""Diagonal non-dimensionalisation ``D = diag(1, L, L, L, L^2 x6)``.

    The ten columns of the regressor carry units kg, kg m and kg m^2. Any condition
    number, ridge penalty or "relative error" computed without first removing those
    units is meaningless. Estimation is done in the scaled variable
    :math:`\tilde\phi = D^{-1}\phi` with :math:`\tilde W = W D`.
    """
    if length_scale <= 0.0:
        raise ValueError("length_scale must be positive")
    d = np.ones(N_PARAMS)
    d[1:4] = length_scale
    d[4:] = length_scale ** 2
    return np.diag(d)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------
def bounding_box_prior(mass: float, bbox_min: np.ndarray, bbox_max: np.ndarray) -> InertialParams:
    r"""Uniform-density solid box prior.

    The CoM sits at the box centre and

    .. math:: I_C = \frac{m}{12}\operatorname{diag}(b^2+c^2,\; a^2+c^2,\; a^2+b^2)

    for side lengths :math:`a, b, c`. This is the default prior :math:`J_0` for the
    entropic regulariser, and it is also what should be typed into Desk for the
    inertia if the dynamic stage turns out to be prior-dominated.
    """
    bbox_min = np.asarray(bbox_min, dtype=float).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=float).reshape(3)
    if np.any(bbox_max <= bbox_min):
        raise ValueError("bounding box max must exceed min in every axis")
    if mass <= 0.0:
        raise ValueError("prior mass must be positive")
    a, b, c = bbox_max - bbox_min
    centre = 0.5 * (bbox_min + bbox_max)
    inertia = (mass / 12.0) * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    return InertialParams(mass, centre, inertia, frame="flange")


def bounding_ellipsoid_matrix(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    r"""Matrix ``Q`` for the density-realizability constraint ``tr(J(phi) Q) >= 0``.

    If all the mass lies inside the ellipsoid
    :math:`\{x : (x-x_0)^\top Q_e (x-x_0) \le 1\}` then
    :math:`\operatorname{tr}(J(\phi) Q) \ge 0` with

    .. math::
        Q = \begin{bmatrix} -Q_e & Q_e x_0 \\ x_0^\top Q_e & 1 - x_0^\top Q_e x_0
            \end{bmatrix}

    which is *linear* in ``phi``. Here the ellipsoid is the one circumscribing the
    tool's bounding box.
    """
    bbox_min = np.asarray(bbox_min, dtype=float).reshape(3)
    bbox_max = np.asarray(bbox_max, dtype=float).reshape(3)
    centre = 0.5 * (bbox_min + bbox_max)
    # Semi-axes of the ellipsoid circumscribing the box: half-extent * sqrt(3).
    semi = 0.5 * (bbox_max - bbox_min) * np.sqrt(3.0)
    qe = np.diag(1.0 / (semi ** 2))
    out = np.zeros((4, 4), dtype=float)
    out[:3, :3] = -qe
    out[:3, 3] = qe @ centre
    out[3, :3] = centre @ qe
    out[3, 3] = 1.0 - float(centre @ qe @ centre)
    return out
