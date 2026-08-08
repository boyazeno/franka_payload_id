"""Excitation trajectory design, safety constraints and export gating."""

from __future__ import annotations

import json

import numpy as np
import pytest

from franka_payload_id.config import HalfSpace, asset_dir
from franka_payload_id.traj.constraints import (
    check_configurations,
    check_trajectory,
    half_space_clearances,
    half_space_jacobian,
    monitored_positions,
)
from franka_payload_id.traj.export import UnsafeExportError, export_trajectory, load_trajectory_csv
from franka_payload_id.traj.fourier import FourierTrajectory, StaticPoseSet
from franka_payload_id.traj.optimize import optimize_static_poses, regressor_condition


@pytest.fixture(scope="module")
def reference_traj() -> FourierTrajectory:
    path = asset_dir() / "excitation_reference.json"
    return FourierTrajectory.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------- Fourier
def test_derivatives_match_finite_differences(reference_traj):
    t = np.linspace(0.5, 3.5, 7)
    eps = 1e-6
    q_p, qd_p, _ = reference_traj(t + eps)
    q_m, qd_m, _ = reference_traj(t - eps)
    _, qd, qdd = reference_traj(t)
    np.testing.assert_allclose(qd, (q_p - q_m) / (2 * eps), atol=1e-6)
    np.testing.assert_allclose(qdd, (qd_p - qd_m) / (2 * eps), atol=1e-5)


def test_jerk_matches_finite_differences(reference_traj):
    t = np.linspace(0.5, 3.5, 7)
    eps = 1e-6
    _, _, a_p = reference_traj(t + eps)
    _, _, a_m = reference_traj(t - eps)
    np.testing.assert_allclose(reference_traj.jerk(t), (a_p - a_m) / (2 * eps), atol=1e-4)


def test_free_parameterisation_starts_and_ends_at_rest(rng):
    """The FCI rejects a commanded trajectory that does not start/end at rest."""
    for n_harmonics in (2, 3, 5, 7):
        x = rng.normal(size=7 * FourierTrajectory.n_free_parameters(n_harmonics)) * 0.2
        traj = FourierTrajectory.from_free_parameters(x, n_harmonics, 0.2)
        rest_v, rest_a = traj.boundary_residuals()
        assert rest_v < 1e-12
        assert rest_a < 1e-12
        # Periodicity means the end matches the start.
        q0, v0, a0 = traj(np.array([0.0]))
        q1, v1, a1 = traj(np.array([traj.period]))
        np.testing.assert_allclose(q0, q1, atol=1e-10)
        np.testing.assert_allclose(v0, v1, atol=1e-10)


def test_free_parameter_roundtrip(rng):
    x = rng.normal(size=7 * FourierTrajectory.n_free_parameters(5)) * 0.2
    traj = FourierTrajectory.from_free_parameters(x, 5, 0.2)
    np.testing.assert_allclose(traj.to_free_parameters(), x, atol=1e-12)


def test_too_few_harmonics_rejected():
    with pytest.raises(ValueError, match="at least 2 harmonics"):
        FourierTrajectory.from_free_parameters(np.zeros(7), 1, 0.2)


def test_sampling_requires_whole_periods(reference_traj):
    t, q, qd, qdd = reference_traj.sample(1000.0, 3)
    spp = reference_traj.samples_per_period(1000.0)
    assert t.size == 3 * spp
    assert q.shape == (3 * spp, 7)
    # Period averaging must be an exact reshape, so a rate that does not divide the
    # period has to be refused rather than silently resampled.
    assert reference_traj.period * 100.1 % 1.0 != 0.0
    with pytest.raises(ValueError, match="whole number of samples"):
        reference_traj.sample(100.1, 2)


def test_dict_roundtrip(reference_traj):
    clone = FourierTrajectory.from_dict(reference_traj.to_dict())
    t = np.linspace(0, reference_traj.period, 50)
    for a, b in zip(reference_traj(t), clone(t)):
        np.testing.assert_allclose(a, b, atol=1e-15)


def test_bandwidth_is_below_structural_modes(reference_traj):
    """Content must stay well under the ~10-20 Hz joint flexibility modes."""
    assert reference_traj.bandwidth_hz <= 2.0


# ---------------------------------------------------------------- constraints
def test_half_space_signed_distance():
    hs = HalfSpace("test", np.array([1.0, 0.0, 0.0]), -0.2)
    np.testing.assert_allclose(hs.signed_distance(np.array([[0.5, 0, 0], [0.1, 0, 0]])),
                               [0.3, -0.1])


def test_clearances_shape_and_sign(panda, cfg):
    q = np.zeros(7)
    q[3] = -1.5
    q[5] = 1.5
    clear = half_space_clearances(panda, cfg.workspace, q)
    assert clear.shape == (len(cfg.workspace.monitored_points), len(cfg.workspace.half_spaces))
    pts = monitored_positions(panda, cfg.workspace, q)
    assert pts.shape == (len(cfg.workspace.monitored_points), 3)


def test_half_space_jacobian_matches_finite_differences(panda, cfg, rng):
    q = np.array([0.2, -0.5, 0.1, -1.8, 0.3, 1.7, 0.4])
    jac = half_space_jacobian(panda, cfg.workspace, q)
    eps = 1e-6
    numeric = np.zeros_like(jac)
    for j in range(7):
        step = np.zeros(7)
        step[j] = eps
        plus = half_space_clearances(panda, cfg.workspace, q + step)
        minus = half_space_clearances(panda, cfg.workspace, q - step)
        numeric[:, :, j] = (plus - minus) / (2 * eps)
    np.testing.assert_allclose(jac, numeric, atol=1e-6)


def test_check_flags_joint_limit_violation(panda, cfg):
    bad = np.tile(cfg.limits.q_max + 0.5, (3, 1))
    report = check_configurations(panda, cfg.workspace, cfg.limits, bad)
    assert not report.ok
    assert any("joint position limits" in v for v in report.violations)


def test_check_flags_joint_one_box(panda, cfg):
    q = np.zeros((1, 7))
    q[0, 0] = 2.5          # outside the +-60 deg safety box
    q[0, 3] = -1.5
    q[0, 5] = 1.5
    report = check_configurations(panda, cfg.workspace, cfg.limits, q)
    assert not report.ok
    assert any("joint 1 leaves the hard safety box" in v for v in report.violations)


def test_placeholder_workspace_blocks_hardware_but_not_feasibility(panda, cfg):
    """Unmeasured walls must not look like a geometric violation."""
    q = np.zeros((1, 7))
    q[0, 3] = -1.5
    q[0, 5] = 1.5
    report = check_configurations(panda, cfg.workspace, cfg.limits, q)
    assert report.placeholder_workspace is True
    assert report.ready_for_hardware is False
    # ok reflects geometry only, so optimisation can still succeed against defaults.
    assert report.ok == (report.min_half_space_clearance_m >= 0.0)


def test_reference_trajectory_is_feasible(panda, cfg, reference_traj):
    report = check_trajectory(panda, cfg.workspace, cfg.derated_limits(), reference_traj,
                              n_samples=400)
    assert report.ok, report.summary()
    assert report.max_velocity_ratio <= 1.0 + 1e-6
    assert report.max_acceleration_ratio <= 1.0 + 1e-6
    assert report.max_jerk_ratio <= 1.0 + 1e-6
    assert report.min_half_space_clearance_m >= 0.0


# ---------------------------------------------------------------- export
def test_export_refuses_placeholder_workspace(panda, cfg, reference_traj, tmp_path):
    """The gate that keeps an unverified trajectory off the robot."""
    with pytest.raises(UnsafeExportError, match="placeholder"):
        export_trajectory(tmp_path / "t.csv", reference_traj, panda, cfg.workspace,
                          cfg.derated_limits(), n_periods=1)


def test_export_roundtrip_with_force(panda, cfg, reference_traj, tmp_path):
    path = export_trajectory(tmp_path / "t.csv", reference_traj, panda, cfg.workspace,
                             cfg.derated_limits(), n_periods=2, force=True)
    t, q, header = load_trajectory_csv(path)
    spp = reference_traj.samples_per_period(1000.0)
    assert t.size == 2 * spp
    assert header["samples_per_period"] == spp
    assert header["n_periods"] == 2
    assert header["workspace_measured"] is False
    _, q_expected, _, _ = reference_traj.sample(1000.0, 2)
    np.testing.assert_allclose(q, q_expected, atol=1e-9)


# ---------------------------------------------------------------- design
def test_static_pose_selection_is_well_conditioned(panda, cfg):
    from franka_payload_id.model import stack_gravity_regressor

    poses = optimize_static_poses(panda, cfg.workspace, cfg.derated_limits(),
                                  n_poses=25, seed=3, n_candidates=800)
    assert poses.shape == (25, 7)
    # Every selected pose must satisfy the workspace constraints.
    for q in poses:
        assert half_space_clearances(panda, cfg.workspace, q).min() >= 0.0
        assert cfg.workspace.q1_min <= q[0] <= cfg.workspace.q1_max
    scale = np.array([1.0, 0.1, 0.1, 0.1])
    assert np.linalg.cond(stack_gravity_regressor(panda, poses) * scale) < 20.0


def test_reported_condition_depends_on_the_length_scale(panda, reference_traj):
    """The condition number is only meaningful relative to a stated length scale.

    The ten columns carry units kg, kg m and kg m^2, so ``cond(W)`` is not a property
    of the trajectory alone -- it changes with ``L``. It is therefore comparable across
    runs only when they share ``L``, and it is *not* an absolute quality score. The
    scale-invariant quantity is the relative parameter uncertainty; see
    :func:`test_relative_uncertainty_is_scale_invariant`.
    """
    values = {L: regressor_condition(panda, reference_traj, 60, L)[0]
              for L in (0.01, 0.1, 1.0)}
    assert all(np.isfinite(v) for v in values.values())
    assert len(set(np.round(list(values.values()), 6))) > 1


def test_relative_uncertainty_is_scale_invariant(panda, reference_traj, rng):
    """%sigma must not depend on the arbitrary non-dimensionalisation constant."""
    from franka_payload_id.model import scaling_matrix, stack_regressor

    t = np.linspace(0.0, reference_traj.period, 80, endpoint=False)
    q, qd, qdd = reference_traj(t)
    w = stack_regressor(panda, q, qd, qdd)
    phi = rng.uniform(0.5, 1.5, 10) * np.array([0.5, 1e-2, 1e-2, 3e-2,
                                                3e-3, 1e-4, 3e-3, 1e-4, 1e-4, 8e-4])

    relative = []
    for length_scale in (0.05, 0.1, 0.5):
        d = scaling_matrix(length_scale)
        cov = np.linalg.pinv((w @ d).T @ (w @ d))
        sigma = np.sqrt(np.diag(d @ cov @ d))
        relative.append(sigma / np.abs(phi))
    np.testing.assert_allclose(relative[0], relative[1], rtol=1e-6)
    np.testing.assert_allclose(relative[0], relative[2], rtol=1e-6)


def test_static_pose_set_waypoints_are_bidirectional():
    poses = np.zeros((3, 7))
    offset = np.full(7, 0.08)
    ps = StaticPoseSet(poses, offset, bidirectional=True)
    wp = ps.waypoints()
    assert len(wp) == 6
    assert [w[2] for w in wp] == [1, -1, 1, -1, 1, -1]
    np.testing.assert_allclose(wp[0][0], -offset)
    np.testing.assert_allclose(wp[1][0], offset)

    single = StaticPoseSet(poses, offset, bidirectional=False)
    assert len(single.waypoints()) == 3
