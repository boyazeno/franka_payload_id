r"""The 7x10 payload regressor.

A payload rigidly bolted to the flange is a rigid body whose only kinematic ancestor
is joint 7. Its entire contribution to the joint torques is therefore

.. math:: \tau_{\text{load}} = Y_L(q,\dot q,\ddot q)\,\phi_L ,
          \qquad Y_L \in \mathbb{R}^{7\times 10}

and **all ten parameters are structurally identifiable** -- unlike full-robot
identification, where only about 40 of 70 base parameters are. That is what makes
payload identification tractable.

Two equivalent constructions are provided and cross-checked against each other in the
tests:

``payload_regressor``
    :math:`Y_L = J_F^\top A_F` with :math:`A_F` the 6x10 body regressor expressed in
    the flange frame and :math:`J_F` the LOCAL frame Jacobian. Gives ``phi``
    **directly in the flange frame**, which is what Franka wants, and does not require
    the payload to exist in the model at all.

``payload_regressor_joint7_block``
    The last ten columns of :func:`pinocchio.computeJointTorqueRegressor`, giving
    ``phi`` in the joint-7 frame, then mapped to the flange frame.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from .panda import FLANGE_OFFSET_IN_JOINT7, LAST_JOINT_ID, PandaModel
from .params import N_PARAMS, translation_map


def payload_regressor(pm: PandaModel, q: np.ndarray, v: np.ndarray,
                      a: np.ndarray) -> np.ndarray:
    r"""Payload regressor in the **flange frame**, shape ``(7, 10)``.

    Parameters
    ----------
    q, v, a:
        Joint position, velocity and acceleration, each length 7.

    Notes
    -----
    Call order matters. ``computeJointTorqueRegressor`` runs the forward pass that
    populates ``data.v`` and ``data.a_gf``, which ``frameBodyRegressor`` reads.
    ``computeFrameJacobian`` internally re-runs ``forwardKinematics(q)``, which zeroes
    those buffers -- so the body regressor is evaluated and copied *before* the
    Jacobian is requested.
    """
    n = pm.nv
    q = np.asarray(q, dtype=float).reshape(n)
    v = np.asarray(v, dtype=float).reshape(n)
    a = np.asarray(a, dtype=float).reshape(n)

    pin.computeJointTorqueRegressor(pm.model, pm.data, q, v, a)
    body = np.array(pin.frameBodyRegressor(pm.model, pm.data, pm.flange_id), dtype=float)

    jac = np.array(
        pin.computeFrameJacobian(pm.model, pm.data, q, pm.flange_id, pin.LOCAL),
        dtype=float,
    )
    return jac.T @ body


def payload_regressor_joint7(pm: PandaModel, q: np.ndarray, v: np.ndarray,
                             a: np.ndarray) -> np.ndarray:
    """Payload regressor with ``phi`` expressed in the **joint-7** frame, ``(7, 10)``."""
    n = pm.nv
    full = np.array(pin.computeJointTorqueRegressor(
        pm.model, pm.data,
        np.asarray(q, dtype=float).reshape(n),
        np.asarray(v, dtype=float).reshape(n),
        np.asarray(a, dtype=float).reshape(n),
    ), dtype=float)
    start = N_PARAMS * (LAST_JOINT_ID - 1)
    return full[:, start:start + N_PARAMS]


def payload_regressor_joint7_block(pm: PandaModel, q: np.ndarray, v: np.ndarray,
                                   a: np.ndarray) -> np.ndarray:
    """Same as :func:`payload_regressor`, via the joint-7 block plus a frame change.

    Kept as an independent implementation so the tests can cross-check the two.
    """
    y7 = payload_regressor_joint7(pm, q, v, a)
    # phi_joint7 = T phi_flange, so Y_flange = Y_joint7 @ T.
    t_map = translation_map(FLANGE_OFFSET_IN_JOINT7)
    return y7 @ t_map


def stack_regressor(pm: PandaModel, q: np.ndarray, v: np.ndarray,
                    a: np.ndarray) -> np.ndarray:
    """Stack the per-sample regressors of a trajectory into ``(7*K, 10)``.

    ``q``, ``v``, ``a`` are ``(K, 7)``.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    v = np.atleast_2d(np.asarray(v, dtype=float))
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if not (q.shape == v.shape == a.shape):
        raise ValueError(f"shape mismatch: q{q.shape} v{v.shape} a{a.shape}")
    k = q.shape[0]
    out = np.empty((k * pm.nv, N_PARAMS), dtype=float)
    for i in range(k):
        out[i * pm.nv:(i + 1) * pm.nv, :] = payload_regressor(pm, q[i], v[i], a[i])
    return out


def gravity_regressor(pm: PandaModel, q: np.ndarray) -> np.ndarray:
    r"""Static (gravity-only) payload regressor, shape ``(7, 4)``.

    Evaluated at :math:`\dot q = \ddot q = 0`, where the six inertia columns vanish
    identically, leaving only :math:`[m,\, m c_x,\, m c_y,\, m c_z]`.

    .. warning::
       Do **not** use :func:`pinocchio.computeStaticRegressor` for this. Despite the
       name it maps ``[m, m c]`` of every body to the *whole-system centre of mass
       position*, not to gravity torques.
    """
    zero = np.zeros(pm.nv)
    full = payload_regressor(pm, q, zero, zero)
    return full[:, :4]


def stack_gravity_regressor(pm: PandaModel, q: np.ndarray) -> np.ndarray:
    """Stack :func:`gravity_regressor` over ``K`` poses into ``(7*K, 4)``."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    out = np.empty((q.shape[0] * pm.nv, 4), dtype=float)
    for i in range(q.shape[0]):
        out[i * pm.nv:(i + 1) * pm.nv, :] = gravity_regressor(pm, q[i])
    return out
