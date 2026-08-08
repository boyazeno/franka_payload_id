r"""Signal conditioning, following Gautier's recipe.

Pipeline for the dynamic stage::

    average the P periods                 -> signal /sqrt(P), and a free noise estimate
    zero-phase Butterworth low-pass on q  -> no phase lag
    central differences on the FILTERED q -> qd, qdd
    the SAME filter applied to delta-tau  -> preserves tau = Y phi
    decimate to ~2 f_c                    -> independent rows, honest covariance
    drop transients and near-zero velocity

Each step has a reason, and getting any of them wrong biases the result in a way that
looks like a plausible answer:

* **Zero phase is mandatory.** A causal filter delays :math:`\dot q, \ddot q` relative
  to :math:`\tau`; the mismatch is proportional to velocity and therefore masquerades
  as extra viscous friction.
* **Never differentiate unfiltered data.** Differentiation amplifies noise by
  :math:`\omega` and double differentiation by :math:`\omega^2`, so a raw second
  derivative at 1 kHz is dominated by encoder noise. Note that the *order* of the
  low-pass and the central differences is immaterial -- both are linear
  time-invariant, so they commute exactly (up to edge effects); what matters is that
  the filter is applied at all. Filtering first is nonetheless the right structure
  here, because the same filtered ``q`` is what the regressor is built from, so
  positions, velocities and accelerations are guaranteed mutually consistent.
* **Filter both sides identically.** For constant :math:`\phi` and any linear operator
  :math:`F`, :math:`\tau = Y\phi \Rightarrow F\{\tau\} = F\{Y\}\phi`. Two different
  filters break the identity at every frequency where they differ.
* **Decimation is not an optimisation.** Adjacent 1 kHz samples of a 10 Hz band-limited
  signal are nearly perfectly correlated. Keeping them all inflates the apparent degrees
  of freedom, making the residual-based covariance optimistic by
  :math:`\sqrt{f_s/2f_c}\approx 7`.
* **Near-zero velocity rows are dropped.** Coulomb friction is discontinuous at a
  velocity zero crossing, so a tiny mismatch between the loaded and bare runs there
  produces a :math:`2F_c` spike that does not cancel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal as sps

N_JOINTS = 7


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------
def average_periods(x: np.ndarray, samples_per_period: int) -> tuple[np.ndarray, np.ndarray]:
    """Average a periodic signal over whole periods.

    Parameters
    ----------
    x:
        ``(P * samples_per_period, ...)``.

    Returns
    -------
    mean, std:
        Each ``(samples_per_period, ...)``. ``std`` is the sample standard deviation
        *across periods* at each phase, which is a direct estimate of the measurement
        noise and is what feeds the weighted-least-squares weights.
    """
    x = np.asarray(x, dtype=float)
    spp = int(samples_per_period)
    if spp <= 0:
        raise ValueError("samples_per_period must be positive")
    n_periods, remainder = divmod(x.shape[0], spp)
    if n_periods < 1:
        raise ValueError(f"need at least one full period, got {x.shape[0]} < {spp} samples")
    if remainder:
        x = x[:n_periods * spp]
    reshaped = x.reshape((n_periods, spp) + x.shape[1:])
    std = reshaped.std(axis=0, ddof=1) if n_periods > 1 else np.zeros_like(reshaped[0])
    return reshaped.mean(axis=0), std


def zero_phase_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (``scipy.signal.filtfilt``)."""
    x = np.asarray(x, dtype=float)
    nyquist = 0.5 * fs
    if not 0.0 < cutoff < nyquist:
        raise ValueError(f"cutoff {cutoff} Hz must lie in (0, {nyquist}) Hz")
    b, a = sps.butter(order, cutoff / nyquist, btype="low")
    padlen = 3 * max(len(a), len(b))
    if x.shape[0] <= padlen:
        raise ValueError(
            f"signal of {x.shape[0]} samples is too short for a zero-phase order-{order} "
            f"filter (needs > {padlen})")
    return sps.filtfilt(b, a, x, axis=0)


def central_differences(x: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    r"""First and second derivatives by second-order central differences.

    .. math::
        \dot x_k = \frac{x_{k+1}-x_{k-1}}{2\Delta t}, \qquad
        \ddot x_k = \frac{x_{k+1}-2x_k+x_{k-1}}{\Delta t^2}

    Edges use one-sided stencils; those samples are trimmed later anyway.
    """
    x = np.asarray(x, dtype=float)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if x.shape[0] < 3:
        raise ValueError("need at least 3 samples for central differences")

    first = np.empty_like(x)
    second = np.empty_like(x)

    first[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    first[0] = (x[1] - x[0]) / dt
    first[-1] = (x[-1] - x[-2]) / dt

    second[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (dt * dt)
    second[0] = second[1]
    second[-1] = second[-2]
    return first, second


def decimate_signal(x: np.ndarray, factor: int) -> np.ndarray:
    """Take every ``factor``-th sample.

    Plain subsampling is correct *here* because the caller has already applied the
    zero-phase low-pass, so the signal is band-limited well below the new Nyquist
    frequency. Using ``scipy.signal.decimate`` would apply a second, different filter
    to the signals but not to the regressor, breaking the "same filter on both sides"
    rule.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("decimation factor must be >= 1")
    return np.asarray(x)[::factor]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
@dataclass
class DynamicDataset:
    """Everything the dynamic estimators need, already conditioned.

    Attributes
    ----------
    q, qd, qdd:
        ``(K, 7)`` filtered kinematics used to build the regressor.
    dtau:
        ``(K, 7)`` difference of measured torques, loaded minus bare.
    sigma:
        ``(7,)`` per-joint noise standard deviation [Nm], from the across-period spread.
    mask:
        ``(K, 7)`` boolean; False where a row must be discarded (near-zero velocity).
    period_index:
        ``(K,)`` index of the source period, used by the bootstrap. All zeros when the
        periods have been averaged away.
    """

    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    dtau: np.ndarray
    sigma: np.ndarray
    mask: np.ndarray
    period_index: np.ndarray
    sample_rate_hz: float
    meta: dict = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.q.shape[0])

    @property
    def n_equations(self) -> int:
        return int(self.mask.sum())

    def summary(self) -> str:
        kept = self.mask.mean() * 100.0
        return (f"dynamic dataset: {self.n_samples} samples @ {self.sample_rate_hz:g} Hz, "
                f"{self.n_equations} usable equations ({kept:.1f}% of rows kept), "
                f"sigma={np.array2string(self.sigma, precision=3)} Nm")


@dataclass
class StaticDataset:
    """Averaged static poses for Stage A."""

    q: np.ndarray          # (n_poses, 7)
    dtau: np.ndarray       # (n_poses, 7)
    sigma: np.ndarray      # (7,)
    meta: dict = field(default_factory=dict)

    @property
    def n_poses(self) -> int:
        return int(self.q.shape[0])

    def summary(self) -> str:
        return (f"static dataset: {self.n_poses} poses, "
                f"sigma={np.array2string(self.sigma, precision=3)} Nm")


def build_dynamic_dataset(q_loaded: np.ndarray, tau_loaded: np.ndarray,
                          q_bare: np.ndarray, tau_bare: np.ndarray, *,
                          sample_rate_hz: float, samples_per_period: int,
                          cutoff_hz: float = 10.0, filter_order: int = 4,
                          decimate_to_hz: float = 100.0,
                          drop_first_period: bool = True,
                          edge_trim_s: float = 0.5,
                          zero_velocity_threshold: float = 0.05) -> DynamicDataset:
    """Turn a paired (loaded, bare) run into a conditioned dataset.

    Both runs must have been recorded on the *same* commanded trajectory at the same
    rate; that is what makes the arm dynamics and load-independent friction cancel.
    """
    q_loaded = np.asarray(q_loaded, dtype=float)
    q_bare = np.asarray(q_bare, dtype=float)
    tau_loaded = np.asarray(tau_loaded, dtype=float)
    tau_bare = np.asarray(tau_bare, dtype=float)

    n = min(q_loaded.shape[0], q_bare.shape[0])
    if n < samples_per_period:
        raise ValueError("runs are shorter than one period")
    q_loaded, q_bare = q_loaded[:n], q_bare[:n]
    tau_loaded, tau_bare = tau_loaded[:n], tau_bare[:n]

    spp = int(samples_per_period)
    if drop_first_period and n >= 2 * spp:
        q_loaded, q_bare = q_loaded[spp:], q_bare[spp:]
        tau_loaded, tau_bare = tau_loaded[spp:], tau_bare[spp:]

    # 1. Average over whole periods; the across-period spread gives the noise level.
    q_l_mean, _ = average_periods(q_loaded, spp)
    q_b_mean, _ = average_periods(q_bare, spp)
    tau_l_mean, tau_l_std = average_periods(tau_loaded, spp)
    tau_b_mean, tau_b_std = average_periods(tau_bare, spp)

    n_periods = q_loaded.shape[0] // spp
    # Std of the *difference of the two averages*: two independent means of n_periods.
    sigma = np.sqrt((tau_l_std ** 2 + tau_b_std ** 2).mean(axis=0) / max(n_periods, 1))

    dtau_raw = tau_l_mean - tau_b_mean

    # Floor sigma against the signal scale rather than against zero. On synthetic or
    # very quiet data the across-period spread comes out at ~1e-17 rather than exactly
    # zero, which passes a `> 0` test and then whitens the residuals by ~1e17 -- enough
    # to make the SDP numerically infeasible. One part per million of the largest
    # torque difference is far below any real sensor noise but keeps the scaling sane.
    floor = max(1e-9, 1e-6 * float(np.abs(dtau_raw).max()))
    sigma = np.maximum(sigma, floor)
    # Build the regressor from the mean of the two runs' measured positions: they track
    # the same reference, and averaging halves the encoder noise.
    q_raw = 0.5 * (q_l_mean + q_b_mean)

    # 2/3/4. Same zero-phase filter on positions and on the torque difference.
    q_filt = zero_phase_lowpass(q_raw, sample_rate_hz, cutoff_hz, filter_order)
    dtau_filt = zero_phase_lowpass(dtau_raw, sample_rate_hz, cutoff_hz, filter_order)
    qd, qdd = central_differences(q_filt, 1.0 / sample_rate_hz)

    # 5. Decimate to roughly twice the cut-off.
    factor = max(1, int(round(sample_rate_hz / decimate_to_hz)))
    q_d, qd_d, qdd_d = (decimate_signal(x, factor) for x in (q_filt, qd, qdd))
    dtau_d = decimate_signal(dtau_filt, factor)
    new_rate = sample_rate_hz / factor

    # 6. Trim filter edge transients.
    trim = int(round(edge_trim_s * new_rate))
    if trim > 0 and q_d.shape[0] > 2 * trim + 3:
        sl = slice(trim, q_d.shape[0] - trim)
        q_d, qd_d, qdd_d, dtau_d = q_d[sl], qd_d[sl], qdd_d[sl], dtau_d[sl]

    # 7. Drop rows at velocity zero crossings, per joint.
    mask = np.abs(qd_d) >= float(zero_velocity_threshold)

    return DynamicDataset(
        q=q_d, qd=qd_d, qdd=qdd_d, dtau=dtau_d, sigma=sigma, mask=mask,
        period_index=np.zeros(q_d.shape[0], dtype=int),
        sample_rate_hz=new_rate,
        meta={"n_periods": n_periods, "decimation_factor": factor,
              "cutoff_hz": cutoff_hz, "source_rate_hz": sample_rate_hz},
    )


def build_static_dataset(q_loaded: np.ndarray, tau_loaded: np.ndarray,
                         q_bare: np.ndarray, tau_bare: np.ndarray,
                         *, sigma: np.ndarray | None = None) -> StaticDataset:
    """Pair averaged static-pose measurements into ``(q, dtau)``.

    Inputs are already per-pose averages (the collector dwells and averages on the
    robot). When the poses were visited bidirectionally, the caller should have
    combined the two approach directions first -- see :func:`combine_approaches`.
    """
    q_loaded = np.atleast_2d(np.asarray(q_loaded, dtype=float))
    q_bare = np.atleast_2d(np.asarray(q_bare, dtype=float))
    tau_loaded = np.atleast_2d(np.asarray(tau_loaded, dtype=float))
    tau_bare = np.atleast_2d(np.asarray(tau_bare, dtype=float))

    if q_loaded.shape != q_bare.shape:
        raise ValueError(
            f"loaded and bare pose sets differ: {q_loaded.shape} vs {q_bare.shape}")
    if not np.allclose(q_loaded, q_bare, atol=5e-3):
        worst = float(np.abs(q_loaded - q_bare).max())
        raise ValueError(
            f"loaded and bare runs did not visit the same poses (max difference "
            f"{worst:.4f} rad); the difference method requires identical configurations")

    dtau = tau_loaded - tau_bare
    if sigma is None:
        # Fall back to the spread of the residual about a per-joint median.
        sigma = np.median(np.abs(dtau - np.median(dtau, axis=0)), axis=0) * 1.4826
        sigma = np.where(sigma > 0.0, sigma, 1.0)
    return StaticDataset(q=0.5 * (q_loaded + q_bare), dtau=dtau,
                         sigma=np.asarray(sigma, dtype=float))


def average_static_dwells(q: np.ndarray, tau: np.ndarray, *, n_rows: int,
                          samples_per_row: int, window_fraction: float = 0.5
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Reduce a static-pose log to one averaged sample per pose visit.

    The collector records every dwell sample of every pose row back to back, so the
    log reshapes exactly into ``(n_rows, samples_per_row, 7)``. Only the middle
    ``window_fraction`` of each dwell is averaged, discarding the settling transient at
    the start and any drift at the end.

    Returns ``(q_mean, tau_mean, tau_std)``.
    """
    q = np.asarray(q, dtype=float)
    tau = np.asarray(tau, dtype=float)
    needed = n_rows * samples_per_row
    if q.shape[0] < needed:
        raise ValueError(
            f"static log holds {q.shape[0]} samples but the metadata claims "
            f"{n_rows} poses x {samples_per_row} dwell samples = {needed}")

    q_r = q[:needed].reshape(n_rows, samples_per_row, -1)
    tau_r = tau[:needed].reshape(n_rows, samples_per_row, -1)

    keep = max(int(samples_per_row * window_fraction), 1)
    start = (samples_per_row - keep) // 2
    sl = slice(start, start + keep)
    return (q_r[:, sl].mean(axis=1),
            tau_r[:, sl].mean(axis=1),
            tau_r[:, sl].std(axis=1, ddof=1))


def combine_approaches(q: np.ndarray, tau: np.ndarray,
                       direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""Average the two approach directions of each static pose.

    Approaching a pose from opposite directions puts the Coulomb friction torque on
    opposite sides of the hysteresis loop, so the mean cancels it to first order:

    .. math:: \tfrac12\big[(\tau_g + F_c) + (\tau_g - F_c)\big] = \tau_g

    ``direction`` is ``+1``/``-1`` per row; rows are otherwise assumed to be in
    pose-major order with the two directions adjacent.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    tau = np.atleast_2d(np.asarray(tau, dtype=float))
    direction = np.asarray(direction).ravel()
    if not (q.shape[0] == tau.shape[0] == direction.shape[0]):
        raise ValueError("q, tau and direction must have the same number of rows")

    out_q, out_tau = [], []
    i = 0
    while i < q.shape[0]:
        if i + 1 < q.shape[0] and direction[i] != direction[i + 1] \
                and np.allclose(q[i], q[i + 1], atol=5e-3):
            out_q.append(0.5 * (q[i] + q[i + 1]))
            out_tau.append(0.5 * (tau[i] + tau[i + 1]))
            i += 2
        else:
            out_q.append(q[i])
            out_tau.append(tau[i])
            i += 1
    return np.array(out_q), np.array(out_tau)
