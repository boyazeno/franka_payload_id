"""Signal conditioning primitives.

These tests exist because each of these steps, done slightly wrong, produces a
plausible-looking but biased answer rather than an obvious failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sps

from franka_payload_id.data.preprocess import (
    average_periods,
    central_differences,
    combine_approaches,
    decimate_signal,
    zero_phase_lowpass,
)


def test_average_periods_reduces_noise_and_estimates_it(rng):
    spp, n_periods = 200, 16
    phase = np.linspace(0, 2 * np.pi, spp, endpoint=False)
    clean = np.column_stack([np.sin(phase), np.cos(2 * phase)])
    noise_std = 0.3
    noisy = np.vstack([clean + rng.normal(0, noise_std, clean.shape)
                       for _ in range(n_periods)])

    mean, std = average_periods(noisy, spp)
    assert mean.shape == clean.shape

    # The averaged signal must be ~sqrt(P) closer to the truth than a single period.
    err_single = np.sqrt(np.mean((noisy[:spp] - clean) ** 2))
    err_mean = np.sqrt(np.mean((mean - clean) ** 2))
    assert err_mean < err_single / np.sqrt(n_periods) * 2.0

    # The across-period spread recovers the injected noise level.
    assert std.mean() == pytest.approx(noise_std, rel=0.25)


def test_average_periods_rejects_short_input():
    with pytest.raises(ValueError, match="at least one full period"):
        average_periods(np.zeros((10, 3)), 50)


def test_zero_phase_filter_has_no_lag():
    """A causal filter would delay the signal; filtfilt must not."""
    fs, f0 = 1000.0, 3.0
    t = np.arange(0, 4.0, 1 / fs)
    x = np.sin(2 * np.pi * f0 * t)[:, None]
    filtered = zero_phase_lowpass(x, fs, cutoff=20.0, order=4)

    core = slice(500, -500)
    np.testing.assert_allclose(filtered[core, 0], x[core, 0], atol=2e-3)

    # For contrast, the causal version is visibly delayed.
    b, a = sps.butter(4, 20.0 / (0.5 * fs))
    causal = sps.lfilter(b, a, x[:, 0])
    assert np.abs(causal[core] - x[core, 0]).max() > 1e-2


def test_zero_phase_filter_removes_high_frequency_noise(rng):
    fs = 1000.0
    t = np.arange(0, 4.0, 1 / fs)
    clean = np.sin(2 * np.pi * 1.0 * t)[:, None]
    noisy = clean + rng.normal(0, 0.1, clean.shape)
    filtered = zero_phase_lowpass(noisy, fs, cutoff=10.0, order=4)
    core = slice(500, -500)
    assert (np.abs(filtered[core] - clean[core]).std()
            < np.abs(noisy[core] - clean[core]).std() / 5.0)


def test_filter_rejects_bad_parameters():
    x = np.zeros((1000, 2))
    with pytest.raises(ValueError, match="must lie in"):
        zero_phase_lowpass(x, 1000.0, cutoff=600.0)
    with pytest.raises(ValueError, match="too short"):
        zero_phase_lowpass(np.zeros((5, 2)), 1000.0, cutoff=10.0)


def test_central_differences_are_second_order_accurate():
    dt = 1e-3
    t = np.arange(0, 2.0, dt)
    x = np.column_stack([np.sin(3.0 * t), np.cos(2.0 * t)])
    d1, d2 = central_differences(x, dt)

    expected1 = np.column_stack([3.0 * np.cos(3.0 * t), -2.0 * np.sin(2.0 * t)])
    expected2 = np.column_stack([-9.0 * np.sin(3.0 * t), -4.0 * np.cos(2.0 * t)])
    core = slice(5, -5)
    # Truncation error of the central stencils is dt^2/6 * f''' and dt^2/12 * f'''',
    # i.e. 4.5e-6 and 6.8e-6 for this signal at dt = 1 ms. Tolerances sit just above.
    np.testing.assert_allclose(d1[core], expected1[core], atol=1e-5)
    np.testing.assert_allclose(d2[core], expected2[core], atol=1e-5)


def test_central_differences_validate_input():
    with pytest.raises(ValueError, match="dt must be positive"):
        central_differences(np.zeros((10, 2)), 0.0)
    with pytest.raises(ValueError, match="at least 3 samples"):
        central_differences(np.zeros((2, 2)), 1e-3)


def test_filtering_before_differentiating_is_essential(rng):
    """Differentiating unfiltered data twice is dominated by noise; filtering fixes it."""
    fs, dt = 1000.0, 1e-3
    t = np.arange(0, 4.0, dt)
    clean = np.sin(2 * np.pi * 1.0 * t)[:, None]
    truth = (-(2 * np.pi) ** 2 * np.sin(2 * np.pi * t))[:, None]
    noisy = clean + rng.normal(0, 1e-4, clean.shape)
    core = slice(1000, -1000)

    _, filtered_then_diff = central_differences(zero_phase_lowpass(noisy, fs, 10.0), dt)
    _, unfiltered = central_differences(noisy, dt)

    err_filtered = np.abs(filtered_then_diff[core] - truth[core]).max()
    err_unfiltered = np.abs(unfiltered[core] - truth[core]).max()

    assert err_filtered < 0.05 * np.abs(truth).max()
    # Without the low-pass the second derivative is useless -- orders of magnitude worse.
    assert err_unfiltered > 20.0 * err_filtered


def test_lowpass_and_central_differences_commute(rng):
    """Both operators are LTI, so the order genuinely does not matter.

    Documented explicitly because it is tempting to believe "filter first" is a
    mathematical requirement. It is not; it is a structural choice, made so that the
    regressor is built from the same filtered positions that produced qd and qdd.
    """
    fs, dt = 1000.0, 1e-3
    t = np.arange(0, 4.0, dt)
    x = (np.sin(2 * np.pi * 1.0 * t) + rng.normal(0, 1e-4, t.size))[:, None]
    core = slice(1000, -1000)

    _, a = central_differences(zero_phase_lowpass(x, fs, 10.0), dt)
    _, raw = central_differences(x, dt)
    b = zero_phase_lowpass(raw, fs, 10.0)

    # Not bit-exact: central_differences uses one-sided stencils at the two ends, and
    # those endpoints propagate differently through filtfilt's edge padding. In the
    # interior the two agree to ~1e-9.
    np.testing.assert_allclose(a[core], b[core], atol=1e-7)


def test_decimate_takes_every_nth():
    x = np.arange(100).reshape(-1, 1)
    np.testing.assert_array_equal(decimate_signal(x, 10)[:, 0], np.arange(0, 100, 10))
    np.testing.assert_array_equal(decimate_signal(x, 1), x)
    with pytest.raises(ValueError):
        decimate_signal(x, 0)


def test_combine_approaches_cancels_direction_dependent_offset():
    """The bidirectional static protocol must remove Coulomb hysteresis."""
    poses = np.array([[0.1] * 7, [0.2] * 7])
    gravity = np.array([[1.0, 2.0, 3.0, 4.0, 0.5, 0.6, 0.7],
                        [1.5, 2.5, 3.5, 4.5, 0.55, 0.65, 0.75]])
    stiction = np.full(7, 0.4)

    q = np.repeat(poses, 2, axis=0)
    tau = np.vstack([gravity[0] + stiction, gravity[0] - stiction,
                     gravity[1] + stiction, gravity[1] - stiction])
    direction = np.array([+1, -1, +1, -1])

    q_c, tau_c = combine_approaches(q, tau, direction)
    assert q_c.shape == (2, 7)
    np.testing.assert_allclose(tau_c, gravity, atol=1e-12)


def test_combine_approaches_passes_through_unpaired_rows():
    q = np.array([[0.1] * 7])
    tau = np.array([[1.0] * 7])
    q_c, tau_c = combine_approaches(q, tau, np.array([+1]))
    np.testing.assert_allclose(tau_c, tau)


def test_settling_period_is_dropped_from_every_block():
    """Dropping it only once would unbalance the ABBA drift cancellation."""
    from franka_payload_id.data.preprocess import build_dynamic_dataset

    spp, per_block, blocks = 100, 4, 2
    total = spp * per_block * blocks
    # A ramp in q makes it obvious which samples survived.
    q = np.tile(np.arange(total, dtype=float)[:, None], (1, 7)) * 1e-6
    tau = np.zeros((total, 7))

    ds = build_dynamic_dataset(q, tau, q, tau, sample_rate_hz=1000.0,
                               samples_per_period=spp, cutoff_hz=50.0,
                               decimate_to_hz=1000.0, drop_first_period=True,
                               edge_trim_s=0.0, zero_velocity_threshold=0.0,
                               n_blocks=blocks)
    # 2 blocks x (4 - 1) periods survive, averaged over the 3 remaining periods.
    assert ds.n_samples == spp

    with pytest.raises(ValueError, match="do not divide evenly"):
        build_dynamic_dataset(q, tau, q, tau, sample_rate_hz=1000.0,
                              samples_per_period=spp, cutoff_hz=50.0,
                              decimate_to_hz=1000.0, drop_first_period=True,
                              edge_trim_s=0.0, n_blocks=3)


def test_combine_approaches_validates_shapes():
    with pytest.raises(ValueError, match="same number of rows"):
        combine_approaches(np.zeros((2, 7)), np.zeros((3, 7)), np.array([1, -1]))
