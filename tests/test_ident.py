"""Estimator behaviour on synthetic data with known ground truth.

Several of these encode findings that are easy to get backwards, so they double as
executable documentation:

* the two Stage-B estimators (SDP and log-Cholesky) must agree -- they share no solver,
  so agreement is strong evidence neither has a formulation bug;
* the bidirectional static protocol must actually remove stiction;
* the ABBA block ordering must actually cancel thermal drift, and both the naive
  `L B L B` alternation and `L L B B` must visibly fail.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from franka_payload_id.config import asset_dir
from franka_payload_id.data.preprocess import (
    build_dynamic_dataset,
    build_static_dataset,
    combine_approaches,
)
from franka_payload_id.ident import identify_dynamic_sdp, identify_static
from franka_payload_id.ident.dynamic_sdp import prediction_rmse
from franka_payload_id.ident.logchol import identify_dynamic_logchol, phi_to_theta, theta_to_phi
from franka_payload_id.ident.validate import cross_validate, parameter_uncertainty
from franka_payload_id.model import (
    bounding_box_prior,
    is_physically_consistent,
    phi_to_mci,
    stack_regressor,
)
from franka_payload_id.selftest import reference_tool_phi
from franka_payload_id.synthetic import NoiseModel, simulate_dynamic_pair, simulate_static_pair
from franka_payload_id.traj.fourier import FourierTrajectory
from franka_payload_id.traj.optimize import optimize_static_poses


@pytest.fixture(scope="module")
def traj() -> FourierTrajectory:
    return FourierTrajectory.from_dict(
        json.loads((asset_dir() / "excitation_reference.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def truth(cfg) -> np.ndarray:
    return reference_tool_phi(cfg)


def _dataset(panda, cfg, traj, truth, *, noise, seed=0, n_periods=8, schedule="abba"):
    loaded, bare = simulate_dynamic_pair(panda, traj, truth, noise=noise,
                                         n_periods=n_periods, seed=seed,
                                         schedule=schedule)
    pp = cfg.experiment.preprocess
    return build_dynamic_dataset(
        loaded.q, loaded.tau_J, bare.q, bare.tau_J, sample_rate_hz=1000.0,
        samples_per_period=loaded.meta.samples_per_period,
        cutoff_hz=float(pp["cutoff_hz"]), filter_order=int(pp["filter_order"]),
        decimate_to_hz=float(pp["decimate_to_hz"]), drop_first_period=True,
        edge_trim_s=float(pp["edge_trim_s"]),
        zero_velocity_threshold=float(pp["zero_velocity_threshold"]),
        n_blocks=loaded.meta.n_blocks)


# ---------------------------------------------------------------- Stage A
def test_static_recovers_mass_and_com(panda, cfg, truth):
    poses = optimize_static_poses(panda, cfg.workspace, cfg.derated_limits(),
                                  n_poses=40, seed=0, n_candidates=1200)
    noise = NoiseModel.from_config(cfg.experiment.synthetic)
    q, tau_l, tau_b, direction = simulate_static_pair(panda, poses, truth, noise=noise, seed=1)

    q_c, tau_lc = combine_approaches(q, tau_l, direction)
    _, tau_bc = combine_approaches(q, tau_b, direction)
    dataset = build_static_dataset(q_c, tau_lc, q_c, tau_bc)

    result = identify_static(panda, dataset, use_mass_constraint=False)
    mass_true, com_true, _ = phi_to_mci(truth)

    assert abs(result.mass - mass_true) / mass_true < 0.01
    assert np.abs(result.com - com_true).max() < 2e-3
    assert result.condition < 25.0


def test_bidirectional_approach_cancels_stiction(panda, cfg, truth):
    """Without averaging the two approach directions, stiction biases the CoM."""
    poses = optimize_static_poses(panda, cfg.workspace, cfg.derated_limits(),
                                  n_poses=30, seed=0, n_candidates=1000)
    noise = NoiseModel.from_config(cfg.experiment.synthetic)
    q, tau_l, tau_b, direction = simulate_static_pair(
        panda, poses, truth, noise=noise, seed=1, bidirectional=True, stiction_nm=0.30)

    mass_true, com_true, _ = phi_to_mci(truth)

    q_c, tau_lc = combine_approaches(q, tau_l, direction)
    _, tau_bc = combine_approaches(q, tau_b, direction)
    combined = identify_static(panda, build_static_dataset(q_c, tau_lc, q_c, tau_bc),
                               use_mass_constraint=False)

    # Only the "+" approach direction: the stiction offset survives.
    one_way = direction > 0
    single = identify_static(
        panda, build_static_dataset(q[one_way], tau_l[one_way], q[one_way], tau_b[one_way]),
        use_mass_constraint=False)

    err_combined = np.abs(combined.com - com_true).max()
    err_single = np.abs(single.com - com_true).max()
    assert err_combined < err_single
    assert err_combined < 2e-3


def test_static_rejects_swapped_runs(panda, cfg, truth):
    """A negative mass means the loaded and bare runs were exchanged."""
    poses = optimize_static_poses(panda, cfg.workspace, cfg.derated_limits(),
                                  n_poses=20, seed=0, n_candidates=800)
    noise = NoiseModel.noiseless()
    q, tau_l, tau_b, _ = simulate_static_pair(panda, poses, truth, noise=noise,
                                              seed=1, bidirectional=False)
    swapped = build_static_dataset(q, tau_b, q, tau_l)   # deliberately the wrong way round
    with pytest.raises(ValueError, match="non-positive mass"):
        identify_static(panda, swapped, use_mass_constraint=False)


def test_static_dataset_rejects_mismatched_poses(panda):
    q_a = np.zeros((3, 7))
    q_b = q_a + 0.5
    with pytest.raises(ValueError, match="did not visit the same poses"):
        build_static_dataset(q_a, np.zeros((3, 7)), q_b, np.zeros((3, 7)))


# ---------------------------------------------------------------- Stage B
def test_dynamic_recovers_exactly_without_noise(panda, cfg, traj, truth):
    dataset = _dataset(panda, cfg, traj, truth, noise=NoiseModel.noiseless(), n_periods=4)
    prior = bounding_box_prior(cfg.tool.mass_scale, cfg.tool.bbox_min, cfg.tool.bbox_max)
    result = identify_dynamic_sdp(panda, dataset, prior=prior, gamma=0.0,
                                  use_entropic_prior=False, _measure_prior_shift=False)
    assert np.abs(result.phi - truth).max() < 1e-5
    assert result.physically_consistent


def test_sdp_and_logchol_agree(panda, cfg, traj, truth):
    """Two independent formulations, no shared solver."""
    dataset = _dataset(panda, cfg, traj, truth,
                       noise=NoiseModel.from_config(cfg.experiment.synthetic), seed=2)
    prior = bounding_box_prior(cfg.tool.mass_scale, cfg.tool.bbox_min, cfg.tool.bbox_max)

    sdp = identify_dynamic_sdp(panda, dataset, prior=prior, gamma=0.0,
                               use_entropic_prior=False, _measure_prior_shift=False)
    lch = identify_dynamic_logchol(panda, dataset, prior=prior, gamma=0.0)

    mass_s, com_s, _ = phi_to_mci(sdp.phi)
    mass_l, com_l, _ = phi_to_mci(lch.phi)
    assert abs(mass_s - mass_l) < 1e-4
    assert np.abs(com_s - com_l).max() < 1e-4
    assert lch.physically_consistent


def test_result_is_always_physically_consistent(panda, cfg, traj, truth):
    """Even at a noise level where unconstrained least squares goes non-physical."""
    raw = dict(cfg.experiment.synthetic)
    raw["torque_noise_nm"] = [0.6] * 7
    dataset = _dataset(panda, cfg, traj, truth, noise=NoiseModel.from_config(raw),
                       seed=5, n_periods=4)
    prior = bounding_box_prior(cfg.tool.mass_scale, cfg.tool.bbox_min, cfg.tool.bbox_max)

    w = stack_regressor(panda, dataset.q, dataset.qd, dataset.qdd)
    mask = dataset.mask.reshape(-1)
    ols, *_ = np.linalg.lstsq(w[mask], dataset.dtau.reshape(-1)[mask], rcond=None)

    result = identify_dynamic_sdp(panda, dataset, prior=prior, gamma=1e-2,
                                  _measure_prior_shift=False)
    assert result.physically_consistent
    if not is_physically_consistent(ols):
        # The point of the LMI: it repairs exactly this case.
        assert is_physically_consistent(result.phi)


def test_cross_validation_scores_a_held_out_trajectory(panda, cfg, traj, truth):
    train = _dataset(panda, cfg, traj, truth,
                     noise=NoiseModel.from_config(cfg.experiment.synthetic), seed=3)
    test = _dataset(panda, cfg, traj, truth,
                    noise=NoiseModel.from_config(cfg.experiment.synthetic), seed=4)
    prior = bounding_box_prior(cfg.tool.mass_scale, cfg.tool.bbox_min, cfg.tool.bbox_max)
    result = identify_dynamic_sdp(panda, train, prior=prior, gamma=1e-2,
                                  _measure_prior_shift=False)

    report = cross_validate(panda, test, result.phi)
    assert report.overall_rmse < 0.05
    assert np.nanmax(report.relative_per_joint) < 0.35
    assert prediction_rmse(panda, test, result.phi) == pytest.approx(
        np.sqrt(np.nanmean(report.rmse_per_joint ** 2)), rel=0.3)


# ---------------------------------------------------------------- protocol
def test_block_schedules_have_the_expected_drift_imbalance():
    """The ordering argument, at the level of arithmetic rather than estimation.

    ``drift_imbalance`` is the mean loaded collection time minus the mean bare one; a
    linear thermal drift multiplies it to give a constant bias on every sample of the
    torque difference. ABBA drives it to exactly zero, and does so *even when the tool
    swaps take time* -- which is what makes it usable in practice.
    """
    from franka_payload_id.synthetic import block_schedule, drift_imbalance

    block_seconds, swap_seconds = 50.0, 180.0

    assert "".join("L" if x else "B" for x in block_schedule(4, "abba")) == "LBBL"
    assert "".join("L" if x else "B" for x in block_schedule(4, "alternating")) == "LBLB"
    assert "".join("L" if x else "B" for x in block_schedule(4, "sequential")) == "LLBB"

    assert drift_imbalance(block_schedule(4, "abba"), block_seconds, swap_seconds) == \
        pytest.approx(0.0, abs=1e-9)
    assert drift_imbalance(block_schedule(8, "abba"), block_seconds, swap_seconds) == \
        pytest.approx(0.0, abs=1e-9)
    # Both alternatives leave a residual of order the campaign length.
    assert abs(drift_imbalance(block_schedule(4, "alternating"),
                               block_seconds, swap_seconds)) > 100.0
    assert abs(drift_imbalance(block_schedule(4, "sequential"),
                               block_seconds, swap_seconds)) > 100.0

    with pytest.raises(ValueError, match="multiple of 4"):
        block_schedule(6, "abba")
    with pytest.raises(ValueError, match="must be even"):
        block_schedule(3, "alternating")


def test_abba_scheduling_cancels_thermal_drift(panda, cfg, traj, truth):
    """The protocol detail that matters most, demonstrated end to end.

    Blocks -- not individual periods -- are the physical unit, because changing
    configuration means unbolting the tool. Ordering the blocks ``L B B L`` equalises
    the two configurations' mean collection times, so a linear thermal drift cancels.
    ``L B L B`` and ``L L B B`` both leave a constant offset on every sample of the
    difference, comparable to a small tool's whole inertia signature.
    """
    noise = NoiseModel.from_config(cfg.experiment.synthetic)
    prior = bounding_box_prior(cfg.tool.mass_scale, cfg.tool.bbox_min, cfg.tool.bbox_max)
    _, _, inertia_true = phi_to_mci(truth)

    def inertia_error(schedule: str) -> float:
        dataset = _dataset(panda, cfg, traj, truth, noise=noise, seed=7, schedule=schedule)
        result = identify_dynamic_sdp(panda, dataset, prior=prior, gamma=1e-2,
                                      _measure_prior_shift=False)
        _, _, inertia = phi_to_mci(result.phi)
        return float(np.abs(np.diag(inertia) - np.diag(inertia_true)).max()
                     / np.diag(inertia_true).max())

    abba = inertia_error("abba")
    alternating = inertia_error("alternating")
    sequential = inertia_error("sequential")

    assert abba < 0.5
    assert alternating > 3.0 * abba
    assert sequential > 3.0 * abba


def test_zero_velocity_rows_are_masked(panda, cfg, traj, truth):
    dataset = _dataset(panda, cfg, traj, truth, noise=NoiseModel.noiseless(), n_periods=4)
    threshold = float(cfg.experiment.preprocess["zero_velocity_threshold"])
    assert np.all(np.abs(dataset.qd[dataset.mask]) >= threshold)
    assert dataset.mask.mean() < 1.0     # something really was dropped


# ---------------------------------------------------------------- uncertainty
def test_parameter_uncertainty_flags_undetermined_parameters(rng):
    design = np.zeros((200, 10))
    design[:, :9] = rng.normal(size=(200, 9))
    design[:, 9] = 1e-9 * rng.normal(size=200)      # column carrying no information
    phi = np.ones(10)
    residual = rng.normal(scale=0.01, size=200)

    unc = parameter_uncertainty(residual, design, phi, threshold_pct=30.0)
    assert unc.identified[:9].all()
    assert not unc.identified[9]
    assert "prior-dominated" in unc.summary()


def test_logchol_parameterisation_is_always_consistent(rng):
    """Every theta maps to a physically consistent phi -- that is the whole point."""
    for _ in range(50):
        theta = rng.normal(scale=1.5, size=10)
        assert is_physically_consistent(theta_to_phi(theta))


def test_logchol_roundtrip(truth):
    np.testing.assert_allclose(theta_to_phi(phi_to_theta(truth)), truth, atol=1e-14)


def test_logchol_rejects_inconsistent_input():
    bad = np.zeros(10)
    bad[0] = -1.0
    with pytest.raises(ValueError, match="physically inconsistent"):
        phi_to_theta(bad)
