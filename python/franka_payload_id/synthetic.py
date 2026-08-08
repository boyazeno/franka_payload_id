r"""Hardware-free ground-truth generator.

The point of this module is to exercise the **production** code paths, not a
simplified stand-in for them. In particular the generated data is written through the
real record format, and the identification pipeline is then fed measured positions and
made to compute its own derivatives -- feeding it the analytic :math:`\dot q, \ddot q`
would hide exactly the errors-in-variables behaviour that matters on hardware.

Modelled effects, all switchable from ``config/experiment.yaml``:

* rigid-body torque from Pinocchio's RNEA, with and without the payload;
* joint friction, Coulomb plus viscous, with a **load-dependent** part -- this is the
  component the difference method does *not* cancel and it sets the floor on how well
  the inertia terms can be recovered;
* additive torque-sensor noise, per joint (joints 5-7 are quieter, being 12 Nm-rated);
* encoder quantisation noise on the logged positions;
* a slow thermal bias drift, which cancels only if loaded and bare runs are interleaved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data.robot_log import RunLog, RunMetadata, make_records
from .model import PandaModel
from .traj.fourier import FourierTrajectory

N_JOINTS = 7


@dataclass
class NoiseModel:
    torque_noise_nm: np.ndarray
    encoder_noise_rad: float
    friction_coulomb_nm: np.ndarray
    friction_viscous_nms: np.ndarray
    friction_load_sensitivity: float
    thermal_drift_nm_per_s: float

    @staticmethod
    def from_config(raw: dict) -> "NoiseModel":
        return NoiseModel(
            torque_noise_nm=np.asarray(raw["torque_noise_nm"], dtype=float),
            encoder_noise_rad=float(raw["encoder_noise_rad"]),
            friction_coulomb_nm=np.asarray(raw["friction_coulomb_nm"], dtype=float),
            friction_viscous_nms=np.asarray(raw["friction_viscous_nms"], dtype=float),
            friction_load_sensitivity=float(raw["friction_load_sensitivity"]),
            thermal_drift_nm_per_s=float(raw["thermal_drift_nm_per_s"]),
        )

    @staticmethod
    def noiseless() -> "NoiseModel":
        z = np.zeros(N_JOINTS)
        return NoiseModel(z.copy(), 0.0, z.copy(), z.copy(), 0.0, 0.0)

    def friction(self, qd: np.ndarray, loaded: bool, smooth_eps: float = 1e-3) -> np.ndarray:
        r"""Coulomb + viscous friction torque.

        ``tanh(qd/eps)`` stands in for ``sign(qd)`` so the model is differentiable at
        the zero crossing; the pipeline discards near-zero-velocity rows anyway, which
        is precisely why the discontinuity there is dangerous on real data.
        """
        gain = 1.0 + (self.friction_load_sensitivity if loaded else 0.0)
        return (gain * self.friction_coulomb_nm * np.tanh(qd / smooth_eps)
                + self.friction_viscous_nms * qd)


def simulate_torque(pm: PandaModel, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
                    *, phi_payload: np.ndarray | None, noise: NoiseModel,
                    rng: np.random.Generator, t: np.ndarray | None = None,
                    thermal_offset_s: float = 0.0) -> np.ndarray:
    """Measured ``tau_J`` for a trajectory, with or without the payload attached."""
    q = np.atleast_2d(q)
    qd = np.atleast_2d(qd)
    qdd = np.atleast_2d(qdd)
    model = pm.with_payload(phi_payload) if phi_payload is not None else pm

    tau = np.empty_like(q)
    for k in range(q.shape[0]):
        tau[k] = model.rnea(q[k], qd[k], qdd[k])

    tau = tau + np.array([noise.friction(row, phi_payload is not None) for row in qd])

    if noise.thermal_drift_nm_per_s and t is not None:
        tau = tau + noise.thermal_drift_nm_per_s * (np.asarray(t)[:, None] + thermal_offset_s)

    if np.any(noise.torque_noise_nm > 0.0):
        tau = tau + rng.normal(0.0, 1.0, tau.shape) * noise.torque_noise_nm
    return tau


def _records(t: np.ndarray, q: np.ndarray, qd: np.ndarray, qdd: np.ndarray,
             tau: np.ndarray, rng: np.random.Generator,
             encoder_noise: float, pm: PandaModel) -> np.ndarray:
    k = t.shape[0]
    q_meas = q + (rng.normal(0.0, encoder_noise, q.shape) if encoder_noise > 0 else 0.0)
    o_t_ee = np.tile(np.eye(4).flatten(order="F"), (k, 1))
    return make_records(
        seq=np.arange(k), time_s=t, dt_s=np.full(k, 1e-3),
        q=q_meas, dq=qd, q_d=q, dq_d=qd, ddq_d=qdd,
        tau_J=tau, tau_J_d=np.zeros_like(tau), dtau_J=np.zeros_like(tau),
        tau_ext=np.zeros_like(tau), o_t_ee=o_t_ee,
        success_rate=np.full(k, 1.0), robot_mode=np.full(k, 2.0), errors=np.zeros(k))


def wall_clock_times(t: np.ndarray, period: float, n_periods: int, *,
                     loaded: bool, interleave: bool) -> np.ndarray:
    r"""Map in-run sample times to the wall-clock times at which they were collected.

    Thermal drift is a function of wall-clock time, so what matters is *when* each
    period of each configuration was actually recorded.

    ``interleave=True`` models the recommended protocol: the two configurations are
    collected period by period, **alternating which one goes first in each pair**
    (L B, B L, L B, ...). Over an even number of periods the mean collection time of
    the loaded samples then equals that of the bare samples, so a linear drift cancels
    *exactly* in the difference.

    Naive alternation that always puts the same configuration first (L B, L B, ...)
    does **not** cancel: it leaves the bare run one slot later on average, i.e. a
    constant offset on every sample of the difference. That residual is comparable to
    a small tool's entire inertia signature, so the swap ordering is not a detail.

    ``interleave=False`` models "all loaded, then all bare", where the offset is the
    whole run duration.
    """
    t = np.asarray(t, dtype=float)
    if not interleave:
        return t if loaded else t + period * n_periods

    index = np.clip((t // period).astype(int), 0, max(n_periods - 1, 0))
    within = t - index * period
    # Two slots per period; swap which configuration occupies the first slot.
    first_is_loaded = (index % 2 == 0)
    takes_first = first_is_loaded if loaded else ~first_is_loaded
    return 2.0 * index * period + np.where(takes_first, 0.0, period) + within


def simulate_dynamic_pair(pm: PandaModel, traj: FourierTrajectory, phi_payload: np.ndarray,
                          *, noise: NoiseModel, n_periods: int = 10,
                          sample_rate_hz: float = 1000.0,
                          seed: int = 0,
                          interleave: bool = True) -> tuple[RunLog, RunLog]:
    """A (loaded, bare) run pair on the same commanded trajectory.

    See :func:`wall_clock_times` for what ``interleave`` models and why it matters.
    """
    rng = np.random.default_rng(seed)
    t, q, qd, qdd = traj.sample(sample_rate_hz, n_periods)

    t_loaded = wall_clock_times(t, traj.period, n_periods, loaded=True, interleave=interleave)
    t_bare = wall_clock_times(t, traj.period, n_periods, loaded=False, interleave=interleave)

    tau_loaded = simulate_torque(pm, q, qd, qdd, phi_payload=phi_payload, noise=noise,
                                 rng=rng, t=t_loaded)
    tau_bare = simulate_torque(pm, q, qd, qdd, phi_payload=None, noise=noise,
                               rng=rng, t=t_bare)

    spp = traj.samples_per_period(sample_rate_hz)

    def meta(loaded: bool) -> RunMetadata:
        return RunMetadata(
            run_id=f"synthetic_{'loaded' if loaded else 'bare'}", kind="trajectory",
            loaded=loaded, robot_ip="synthetic", libfranka_version="synthetic",
            sample_rate_hz=sample_rate_hz, samples_per_period=spp, n_periods=n_periods,
            trajectory=traj.to_dict(), notes="generated by franka_payload_id.synthetic")

    loaded_log = RunLog(_records(t, q, qd, qdd, tau_loaded, rng,
                                 noise.encoder_noise_rad, pm), meta(True))
    bare_log = RunLog(_records(t, q, qd, qdd, tau_bare, rng,
                               noise.encoder_noise_rad, pm), meta(False))
    return loaded_log, bare_log


def simulate_static_pair(pm: PandaModel, poses: np.ndarray, phi_payload: np.ndarray, *,
                         noise: NoiseModel, seed: int = 0,
                         bidirectional: bool = True,
                         stiction_nm: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Static-pose measurements, returning ``(q, tau_loaded, tau_bare, direction)``.

    Stiction is modelled as a torque offset whose **sign follows the approach
    direction**. That is exactly the hysteresis the bidirectional protocol is designed
    to remove: averaging the two directions cancels it to first order, and the tests
    check that the pipeline actually benefits.
    """
    rng = np.random.default_rng(seed)
    poses = np.atleast_2d(np.asarray(poses, dtype=float))
    zero = np.zeros(N_JOINTS)
    loaded_model = pm.with_payload(phi_payload)
    stiction = noise.friction_coulomb_nm if stiction_nm is None \
        else np.full(N_JOINTS, float(stiction_nm))

    directions = [+1, -1] if bidirectional else [+1]
    q_out, tau_l, tau_b, dir_out = [], [], [], []
    for pose in poses:
        for d in directions:
            base_l = loaded_model.rnea(pose, zero, zero) + d * stiction
            base_b = pm.rnea(pose, zero, zero) + d * stiction
            # Each pose is a dwell average of ~1000 samples, so the noise is reduced.
            scale = noise.torque_noise_nm / np.sqrt(1000.0)
            q_out.append(pose)
            tau_l.append(base_l + rng.normal(0.0, 1.0, N_JOINTS) * scale)
            tau_b.append(base_b + rng.normal(0.0, 1.0, N_JOINTS) * scale)
            dir_out.append(d)

    return (np.array(q_out), np.array(tau_l), np.array(tau_b), np.array(dir_out))
