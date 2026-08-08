"""The payload regressor and the Panda model.

The decisive test is :func:`test_regressor_reproduces_rnea_difference`: it pins the
central identity of the whole project,

    Y_L(q, qd, qdd) @ phi_L  ==  rnea(arm + payload) - rnea(arm)

which is exactly the quantity the difference-of-torques protocol measures.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

from franka_payload_id.model import (
    FLANGE_OFFSET_IN_JOINT7,
    PandaModel,
    gravity_regressor,
    payload_regressor,
    payload_regressor_joint7,
    payload_regressor_joint7_block,
    stack_gravity_regressor,
    stack_regressor,
    translate_phi,
)

from .conftest import sample_states


# ---------------------------------------------------------------- model
def test_flange_is_pure_translation(panda):
    """Everything downstream assumes joint 7 -> flange has no rotation."""
    offset = panda.check_flange_offset()
    np.testing.assert_allclose(offset, FLANGE_OFFSET_IN_JOINT7, atol=1e-12)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.107], atol=1e-12)


def test_urdf_matches_official_dh_table(panda, rng):
    """Guards against shipping a URDF whose frames are not the real robot's."""
    lo = np.asarray(panda.model.lowerPositionLimit)
    hi = np.asarray(panda.model.upperPositionLimit)
    for _ in range(50):
        pos_err, rot_err = panda.check_against_dh(rng.uniform(lo, hi))
        assert pos_err < 1e-9, f"flange position disagrees with the DH table by {pos_err} m"
        assert rot_err < 1e-6, f"flange rotation disagrees with the DH table by {rot_err} rad"


def test_model_is_seven_dof(panda):
    assert panda.nv == 7
    assert panda.model.njoints == 8  # universe + 7 revolute


def test_missing_urdf_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Panda URDF not found"):
        PandaModel.load(tmp_path / "nope.urdf")


# ---------------------------------------------------------------- regressor
def test_regressor_reproduces_rnea_difference(panda, rng, tool_phi):
    loaded = panda.with_payload(tool_phi)
    worst = 0.0
    for q, v, a in sample_states(panda, rng, 100):
        expected = loaded.rnea(q, v, a) - panda.rnea(q, v, a)
        worst = max(worst, float(np.abs(payload_regressor(panda, q, v, a) @ tool_phi
                                        - expected).max()))
    assert worst < 1e-9, f"payload regressor identity violated by {worst} Nm"


def test_two_regressor_constructions_agree(panda, rng):
    """Flange-frame construction vs joint-7 block plus frame change."""
    worst = 0.0
    for q, v, a in sample_states(panda, rng, 50):
        worst = max(worst, float(np.abs(payload_regressor(panda, q, v, a)
                                        - payload_regressor_joint7_block(panda, q, v, a)).max()))
    assert worst < 1e-10


def test_joint7_regressor_uses_joint7_frame(panda, rng, tool_phi):
    """The joint-7 block consumes phi expressed in the joint-7 frame."""
    phi_joint7 = translate_phi(tool_phi, FLANGE_OFFSET_IN_JOINT7)
    for q, v, a in sample_states(panda, rng, 20):
        np.testing.assert_allclose(
            payload_regressor_joint7(panda, q, v, a) @ phi_joint7,
            payload_regressor(panda, q, v, a) @ tool_phi,
            atol=1e-10,
        )


def test_regressor_is_linear_in_phi(panda, rng):
    """Sanity: the map really is linear, so superposition of two payloads holds."""
    q, v, a = next(iter(sample_states(panda, rng, 1)))
    y = payload_regressor(panda, q, v, a)
    p1 = rng.uniform(-1, 1, 10)
    p2 = rng.uniform(-1, 1, 10)
    np.testing.assert_allclose(y @ (2.5 * p1 - 0.5 * p2),
                               2.5 * (y @ p1) - 0.5 * (y @ p2), atol=1e-12)


def test_call_order_does_not_corrupt_the_body_regressor(panda, rng):
    """Regression guard for a subtle ordering bug.

    ``computeFrameJacobian`` re-runs ``forwardKinematics(q)``, which zeroes
    ``data.v``/``data.a_gf``. If the body regressor were evaluated after the Jacobian
    it would silently lose all velocity and acceleration terms, leaving only gravity.
    Calling the regressor twice in a row must give the same answer.
    """
    q, v, a = next(iter(sample_states(panda, rng, 1)))
    first = payload_regressor(panda, q, v, a)
    second = payload_regressor(panda, q, v, a)
    np.testing.assert_allclose(first, second, atol=1e-14)
    # And with a genuinely non-zero velocity it must differ from the static case.
    static = payload_regressor(panda, q, np.zeros(7), np.zeros(7))
    assert np.abs(first - static).max() > 1e-6


def test_gravity_regressor_is_the_static_limit(panda, rng, tool_phi):
    """At zero velocity and acceleration only the first four columns survive."""
    for q, _, _ in sample_states(panda, rng, 20):
        full = payload_regressor(panda, q, np.zeros(7), np.zeros(7))
        np.testing.assert_allclose(full[:, 4:], 0.0, atol=1e-12)
        np.testing.assert_allclose(gravity_regressor(panda, q), full[:, :4], atol=1e-15)


def test_gravity_regressor_predicts_static_torque(panda, rng, tool_phi):
    loaded = panda.with_payload(tool_phi)
    zero = np.zeros(7)
    for q, _, _ in sample_states(panda, rng, 20):
        expected = loaded.rnea(q, zero, zero) - panda.rnea(q, zero, zero)
        np.testing.assert_allclose(gravity_regressor(panda, q) @ tool_phi[:4],
                                   expected, atol=1e-10)


def test_static_regressor_is_well_conditioned(panda, rng):
    """The four-parameter gravity problem should be very well conditioned."""
    lo = np.asarray(panda.model.lowerPositionLimit)
    hi = np.asarray(panda.model.upperPositionLimit)
    poses = np.array([rng.uniform(lo, hi) for _ in range(40)])
    w = stack_gravity_regressor(panda, poses)
    scale = np.array([1.0, 0.1, 0.1, 0.1])
    assert np.linalg.cond(w * scale) < 20.0


def test_pinocchio_static_regressor_is_not_the_gravity_regressor(panda):
    """Documents the trap: computeStaticRegressor is a different object entirely."""
    q = np.zeros(7)
    static = pin.computeStaticRegressor(panda.model, panda.data, q)
    assert static.shape == (3, 4 * (panda.model.njoints - 1))
    assert gravity_regressor(panda, q).shape == (7, 4)


def test_stack_regressor_shapes_and_content(panda, rng, tool_phi):
    states = list(sample_states(panda, rng, 5))
    q = np.array([s[0] for s in states])
    v = np.array([s[1] for s in states])
    a = np.array([s[2] for s in states])
    stacked = stack_regressor(panda, q, v, a)
    assert stacked.shape == (5 * 7, 10)
    for i, (qi, vi, ai) in enumerate(states):
        np.testing.assert_allclose(stacked[i * 7:(i + 1) * 7],
                                   payload_regressor(panda, qi, vi, ai), atol=1e-14)

    with pytest.raises(ValueError, match="shape mismatch"):
        stack_regressor(panda, q, v, a[:-1])


def test_with_payload_adds_exactly_the_payload(panda, tool_phi):
    """model.with_payload must place the tool at the flange, not at joint 7."""
    loaded = panda.with_payload(tool_phi)
    added = loaded.model.inertias[7].toDynamicParameters() \
        - panda.model.inertias[7].toDynamicParameters()
    np.testing.assert_allclose(added, translate_phi(tool_phi, FLANGE_OFFSET_IN_JOINT7),
                               atol=1e-12)
