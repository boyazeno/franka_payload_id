"""Pinocchio model of the Panda arm, plus the flange-frame bookkeeping.

The arm URDF carries the Gaz et al. (RA-L 2019) identified link inertias. Those are
used only for prior/torque-budget purposes: the difference-of-torques estimator does
not depend on them at all, only on the kinematics -- which come from the manufacturer
and are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

from ..config import urdf_path

FLANGE_FRAME = "panda_link8"
LAST_JOINT_ID = 7
FLANGE_OFFSET_IN_JOINT7 = np.array([0.0, 0.0, 0.107])
"""Franka's flange F relative to the joint-7 frame: a pure translation, no rotation.

Confirmed from the FCI DH table (flange row: a=0, d=0.107, alpha=0, theta=0) and from
``panda_joint8`` in the URDF (``xyz="0 0 0.107" rpy="0 0 0"``, fixed). Because there
is no rotation, converting inertial parameters between the two frames never involves
rotating an inertia tensor -- historically the main source of sign errors here.
"""

# Craig-convention DH parameters (a, d, alpha, theta_offset) from the FCI docs.
# Used only to verify that the URDF we ship describes the same robot.
PANDA_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -np.pi / 2),
    (0.0, 0.316, np.pi / 2),
    (0.0825, 0.0, np.pi / 2),
    (-0.0825, 0.384, -np.pi / 2),
    (0.0, 0.0, np.pi / 2),
    (0.088, 0.0, np.pi / 2),
)
PANDA_DH_FLANGE = (0.0, 0.107, 0.0)


def dh_forward_kinematics(q: np.ndarray) -> np.ndarray:
    """Base-to-flange transform from the official DH table (Craig convention).

    Independent of the URDF on purpose: :func:`PandaModel.check_against_dh` compares
    the two so a mismatched URDF cannot slip in unnoticed.
    """
    q = np.asarray(q, dtype=float).reshape(7)
    t = np.eye(4)
    rows = [*PANDA_DH, PANDA_DH_FLANGE]
    angles = [*q, 0.0]
    for (a, d, alpha), theta in zip(rows, angles):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(theta), np.sin(theta)
        # Craig: T = Rx(alpha) Tx(a) Rz(theta) Tz(d)
        step = np.array([
            [ct,      -st,     0.0,   a],
            [st * ca,  ct * ca, -sa, -d * sa],
            [st * sa,  ct * sa,  ca,  d * ca],
            [0.0,      0.0,     0.0,  1.0],
        ])
        t = t @ step
    return t


@dataclass
class PandaModel:
    """Thin wrapper around a Pinocchio model/data pair for the Panda arm."""

    model: pin.Model
    data: pin.Data
    flange_id: int

    # -- construction -------------------------------------------------------
    @staticmethod
    def load(urdf: Path | str | None = None) -> "PandaModel":
        path = Path(urdf) if urdf is not None else urdf_path()
        if not path.exists():
            raise FileNotFoundError(
                f"Panda URDF not found at {path}. It is generated from the local "
                "franka_description copy; see scripts/00_build_urdf.py."
            )
        model = pin.buildModelFromUrdf(str(path))
        if model.nv != 7:
            raise ValueError(f"expected a 7-DoF arm, got nv={model.nv} from {path}")
        if not model.existFrame(FLANGE_FRAME):
            raise ValueError(f"URDF {path} has no frame {FLANGE_FRAME!r}")
        return PandaModel(model, model.createData(), model.getFrameId(FLANGE_FRAME))

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    # -- kinematics ---------------------------------------------------------
    def forward(self, q: np.ndarray, v: np.ndarray | None = None,
                a: np.ndarray | None = None) -> None:
        """Populate joint and frame *placements* in ``data`` for the given state.

        Kinematics only. Note that :func:`pinocchio.rnea` does **not** fill
        ``data.oMi``, so placements must come from ``forwardKinematics``; using rnea
        here silently leaves every placement at the identity.

        The regressor functions in :mod:`.regressor` deliberately do not rely on this
        method: they call ``computeJointTorqueRegressor`` themselves, which runs the
        forward pass that populates ``data.v`` and ``data.a_gf`` (the classical
        acceleration *including* gravity) that the body regressors read.
        """
        q = np.asarray(q, dtype=float).reshape(self.nv)
        v = np.zeros(self.nv) if v is None else np.asarray(v, dtype=float).reshape(self.nv)
        a = np.zeros(self.nv) if a is None else np.asarray(a, dtype=float).reshape(self.nv)
        pin.forwardKinematics(self.model, self.data, q, v, a)
        pin.updateFramePlacements(self.model, self.data)

    def flange_placement(self, q: np.ndarray) -> pin.SE3:
        """Base-to-flange SE3 transform."""
        self.forward(q)
        return self.data.oMf[self.flange_id].copy()

    def frame_position(self, q: np.ndarray, frame: str,
                       offset: np.ndarray | None = None) -> np.ndarray:
        """Position in the base frame of a point rigidly attached to ``frame``."""
        self.forward(q)
        fid = self.model.getFrameId(frame)
        placement = self.data.oMf[fid]
        if offset is None:
            return placement.translation.copy()
        return placement.act(np.asarray(offset, dtype=float).reshape(3))

    def frame_position_jacobian(self, q: np.ndarray, frame: str,
                                offset: np.ndarray | None = None) -> np.ndarray:
        """3x7 translational Jacobian of that point, in base-frame axes.

        Obtained by shifting the frame Jacobian to the offset point:
        ``v_p = v_f + omega_f x (R offset)``, so
        ``J_p = J_v - skew(R offset) J_omega`` in LOCAL_WORLD_ALIGNED axes.
        """
        q = np.asarray(q, dtype=float).reshape(self.nv)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        fid = self.model.getFrameId(frame)
        jac = pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)
        if offset is None:
            return np.asarray(jac[:3, :])
        r_offset = self.data.oMf[fid].rotation @ np.asarray(offset, dtype=float).reshape(3)
        return np.asarray(jac[:3, :]) - pin.skew(r_offset) @ np.asarray(jac[3:, :])

    # -- dynamics -----------------------------------------------------------
    def rnea(self, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Inverse dynamics of the bare arm [Nm]."""
        return np.asarray(pin.rnea(
            self.model, self.data,
            np.asarray(q, dtype=float).reshape(self.nv),
            np.asarray(v, dtype=float).reshape(self.nv),
            np.asarray(a, dtype=float).reshape(self.nv),
        )).copy()

    def with_payload(self, phi_flange: np.ndarray) -> "PandaModel":
        """Copy of this model with a payload rigidly added at the flange.

        Only used to generate synthetic ground truth and to verify the regressor; the
        estimator never needs it.
        """
        from .params import phi_to_mci  # local import to avoid a cycle at import time

        mass, com, inertia_com = phi_to_mci(phi_flange)
        model = self.model.copy()
        payload = pin.Inertia(mass, com, inertia_com)
        # Express it in the joint-7 frame: pure translation by the flange offset.
        j_m_f = pin.SE3(np.eye(3), FLANGE_OFFSET_IN_JOINT7)
        model.inertias[LAST_JOINT_ID] = model.inertias[LAST_JOINT_ID] + j_m_f.act(payload)
        return PandaModel(model, model.createData(), model.getFrameId(FLANGE_FRAME))

    # -- verification -------------------------------------------------------
    def check_against_dh(self, q: np.ndarray) -> tuple[float, float]:
        """Compare URDF flange FK with the DH table.

        Returns ``(position_error_m, rotation_error_rad)``. Both should be ~1e-12.
        """
        urdf_t = self.flange_placement(q).homogeneous
        dh_t = dh_forward_kinematics(q)
        pos_err = float(np.linalg.norm(urdf_t[:3, 3] - dh_t[:3, 3]))
        r_rel = urdf_t[:3, :3].T @ dh_t[:3, :3]
        cos_angle = np.clip((np.trace(r_rel) - 1.0) / 2.0, -1.0, 1.0)
        return pos_err, float(np.arccos(cos_angle))

    def check_flange_offset(self) -> np.ndarray:
        """Joint-7-to-flange translation as measured from the loaded URDF."""
        self.forward(np.zeros(self.nv))
        rel = self.data.oMi[LAST_JOINT_ID].actInv(self.data.oMf[self.flange_id])
        if not np.allclose(rel.rotation, np.eye(3), atol=1e-12):
            raise ValueError(
                "joint-7 -> flange transform has a rotation; the whole pipeline assumes "
                "it is a pure translation"
            )
        return np.asarray(rel.translation).copy()

    def random_configuration(self, rng: np.random.Generator,
                             q_min: np.ndarray, q_max: np.ndarray) -> np.ndarray:
        return rng.uniform(np.asarray(q_min), np.asarray(q_max))
