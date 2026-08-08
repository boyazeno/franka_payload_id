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
* a slow thermal bias drift, which cancels only when the collection blocks are
  ordered ABBA (`L B B L`) -- see :func:`block_schedule` and :func:`drift_imbalance`.
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


def block_schedule(n_blocks: int, mode: str = "abba") -> list[bool]:
    """Order in which the two configurations are collected. ``True`` means loaded.

    A *block* is a contiguous run of several periods in one configuration. Blocks --
    not individual periods -- are the physical unit here, because changing
    configuration means unbolting the tool, which takes minutes.

    ``abba``
        ``L B B L`` repeated. The mean collection time of the loaded blocks equals that
        of the bare blocks, so a linear thermal drift cancels exactly (see
        :func:`drift_imbalance`). Requires ``n_blocks`` to be a multiple of 4.
    ``alternating``
        ``L B L B``. Intuitive, and wrong: it leaves the bare blocks one slot later on
        average, i.e. a constant offset on every sample of the difference.
    ``sequential``
        ``L L B B`` -- all of one, then all of the other. The worst case, with an offset
        of half the whole campaign.
    """
    if n_blocks % 2:
        raise ValueError("n_blocks must be even so each configuration gets the same count")

    if mode == "abba":
        if n_blocks % 4:
            raise ValueError("the abba schedule needs n_blocks to be a multiple of 4")
        pattern = [True, False, False, True]
        return [pattern[i % 4] for i in range(n_blocks)]
    if mode == "alternating":
        return [i % 2 == 0 for i in range(n_blocks)]
    if mode == "sequential":
        return [i < n_blocks // 2 for i in range(n_blocks)]
    raise ValueError(f"unknown schedule {mode!r}")


def block_start_times(schedule: list[bool], block_seconds: float,
                      swap_seconds: float) -> np.ndarray:
    """Wall-clock start time of each block, including the tool swaps between them.

    A swap is only needed where the configuration actually changes -- which is why the
    ``B B`` pair in the middle of an ABBA group costs nothing.
    """
    starts = np.empty(len(schedule), dtype=float)
    clock = 0.0
    for i, loaded in enumerate(schedule):
        if i > 0 and schedule[i - 1] != loaded:
            clock += swap_seconds
        starts[i] = clock
        clock += block_seconds
    return starts


def drift_imbalance(schedule: list[bool], block_seconds: float,
                    swap_seconds: float) -> float:
    r"""Mean loaded collection time minus mean bare collection time [s].

    This single number is what a linear thermal drift multiplies to produce a constant
    bias on every sample of :math:`\Delta\tau`. Zero means the drift cancels exactly.
    ABBA gives zero **even when the swaps take time**, provided the two swaps take
    similar time -- which is the practical reason to prefer it.
    """
    starts = block_start_times(schedule, block_seconds, swap_seconds)
    centres = starts + 0.5 * block_seconds
    loaded = np.array([c for c, is_loaded in zip(centres, schedule) if is_loaded])
    bare = np.array([c for c, is_loaded in zip(centres, schedule) if not is_loaded])
    return float(loaded.mean() - bare.mean())


def wall_clock_times(t: np.ndarray, period: float, n_periods: int, *,
                     loaded: bool, schedule: list[bool],
                     swap_seconds: float = 0.0) -> np.ndarray:
    """Map in-run sample times to the wall-clock times at which they were collected.

    Thermal drift is a function of wall-clock time, so what matters is *when* each
    period of each configuration was actually recorded. ``t`` runs from 0 to
    ``n_periods * period`` within one configuration's concatenated log; this spreads
    those periods across that configuration's blocks in the schedule.
    """
    t = np.asarray(t, dtype=float)
    my_blocks = [i for i, is_loaded in enumerate(schedule) if is_loaded == loaded]
    if not my_blocks:
        raise ValueError("the schedule contains no blocks for this configuration")
    if n_periods % len(my_blocks):
        raise ValueError(
            f"{n_periods} periods do not divide evenly into {len(my_blocks)} blocks")

    per_block = n_periods // len(my_blocks)
    block_seconds = per_block * period
    starts = block_start_times(schedule, block_seconds, swap_seconds)

    index = np.clip((t // period).astype(int), 0, max(n_periods - 1, 0))
    within_period = t - index * period
    block_of_period = index // per_block
    offset_in_block = (index % per_block) * period

    block_start = starts[np.asarray(my_blocks)[block_of_period]]
    return block_start + offset_in_block + within_period


def simulate_dynamic_pair(pm: PandaModel, traj: FourierTrajectory, phi_payload: np.ndarray,
                          *, noise: NoiseModel, n_periods: int = 10,
                          sample_rate_hz: float = 1000.0,
                          seed: int = 0,
                          schedule: str = "abba",
                          n_blocks: int = 4,
                          swap_seconds: float = 180.0) -> tuple[RunLog, RunLog]:
    """A (loaded, bare) run pair on the same commanded trajectory.

    ``n_periods`` is the total per configuration; it is split across that
    configuration's blocks in the schedule. ``swap_seconds`` is how long changing the
    tool takes -- three minutes by default, which is realistic and, with ABBA, harmless.

    See :func:`block_schedule` and :func:`drift_imbalance` for why the ordering matters.
    """
    rng = np.random.default_rng(seed)
    t, q, qd, qdd = traj.sample(sample_rate_hz, n_periods)

    order = block_schedule(n_blocks, schedule)
    t_loaded = wall_clock_times(t, traj.period, n_periods, loaded=True,
                                schedule=order, swap_seconds=swap_seconds)
    t_bare = wall_clock_times(t, traj.period, n_periods, loaded=False,
                              schedule=order, swap_seconds=swap_seconds)

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
            n_blocks=sum(1 for x in order if x == loaded),
            trajectory=traj.to_dict(),
            notes=f"generated by franka_payload_id.synthetic; schedule={schedule}")

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
