r"""Swevers finite-Fourier-series excitation trajectories.

Per joint :math:`j`,

.. math::
    q_j(t) &= q_{j,0} + \sum_{l=1}^{N}
        \Big[\tfrac{a_{jl}}{l\omega_f}\sin(l\omega_f t)
           - \tfrac{b_{jl}}{l\omega_f}\cos(l\omega_f t)\Big] \\
    \dot q_j(t) &= \sum_l \big[a_{jl}\cos(l\omega_f t) + b_{jl}\sin(l\omega_f t)\big] \\
    \ddot q_j(t) &= \sum_l l\omega_f\big[-a_{jl}\sin(l\omega_f t) + b_{jl}\cos(l\omega_f t)\big]

Two properties earn this parameterisation its place:

* **Periodicity.** Running :math:`P` periods and averaging sample-by-sample cuts the
  noise by :math:`\sqrt P` while leaving the (deterministic, periodic) signal intact,
  and the across-period variance is a free estimate of the measurement noise that
  feeds the weighted-least-squares weights.
* **Exact bandwidth control.** The content lives in :math:`[f_f,\, N f_f]`, so the
  excitation can be kept well below the ~10-20 Hz joint-flexibility modes.

Boundary conditions. The FCI requires a commanded trajectory to start and end at rest
(:math:`\dot q_c = \ddot q_c = 0`). Both are linear in the coefficients:
:math:`\sum_l a_{jl} = 0` gives :math:`\dot q_j(0)=0`, and :math:`\sum_l l\, b_{jl}=0`
gives :math:`\ddot q_j(0)=0`. Periodicity then makes the same true at :math:`t=T`.
:meth:`FourierTrajectory.from_free_parameters` enforces both by construction, so the
optimiser searches an unconstrained space.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_JOINTS = 7


@dataclass(frozen=True)
class FourierTrajectory:
    """A periodic joint-space excitation trajectory.

    Attributes
    ----------
    q0:
        Offsets, shape ``(7,)`` [rad].
    a, b:
        Cosine/sine velocity coefficients, shape ``(7, N)`` [rad/s].
    base_frequency:
        :math:`f_f` [Hz]; the period is ``1 / base_frequency``.
    """

    q0: np.ndarray
    a: np.ndarray
    b: np.ndarray
    base_frequency: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "q0", np.asarray(self.q0, dtype=float).reshape(N_JOINTS))
        a = np.atleast_2d(np.asarray(self.a, dtype=float))
        b = np.atleast_2d(np.asarray(self.b, dtype=float))
        if a.shape != b.shape or a.shape[0] != N_JOINTS:
            raise ValueError(f"a{a.shape} and b{b.shape} must both be (7, n_harmonics)")
        if self.base_frequency <= 0.0:
            raise ValueError("base_frequency must be positive")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)

    # -- basic properties ---------------------------------------------------
    @property
    def n_harmonics(self) -> int:
        return int(self.a.shape[1])

    @property
    def period(self) -> float:
        return 1.0 / self.base_frequency

    @property
    def omega(self) -> float:
        return 2.0 * np.pi * self.base_frequency

    @property
    def bandwidth_hz(self) -> float:
        """Highest frequency present in the trajectory."""
        return self.n_harmonics * self.base_frequency

    # -- evaluation ---------------------------------------------------------
    def __call__(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate ``(q, qd, qdd)`` at times ``t``; each result is ``(len(t), 7)``."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        harmonics = np.arange(1, self.n_harmonics + 1)          # (N,)
        w = self.omega * harmonics                              # (N,)
        phase = np.outer(t, w)                                  # (K, N)
        sin, cos = np.sin(phase), np.cos(phase)

        # einsum over harmonics: (K,N) x (7,N) -> (K,7)
        q = self.q0 + np.einsum("kn,jn->kj", sin, self.a / w) \
                    - np.einsum("kn,jn->kj", cos, self.b / w)
        qd = np.einsum("kn,jn->kj", cos, self.a) + np.einsum("kn,jn->kj", sin, self.b)
        qdd = np.einsum("kn,jn->kj", -sin, self.a * w) + np.einsum("kn,jn->kj", cos, self.b * w)
        return q, qd, qdd

    def jerk(self, t: np.ndarray) -> np.ndarray:
        """Third derivative, needed for the FCI jerk limit."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        harmonics = np.arange(1, self.n_harmonics + 1)
        w = self.omega * harmonics
        phase = np.outer(t, w)
        return (np.einsum("kn,jn->kj", -np.cos(phase), self.a * w**2)
                - np.einsum("kn,jn->kj", np.sin(phase), self.b * w**2))

    def sample(self, rate_hz: float, n_periods: int = 1) -> tuple[np.ndarray, ...]:
        """Uniformly sample ``n_periods`` periods at ``rate_hz``.

        Returns ``(t, q, qd, qdd)``. The number of samples per period is forced to be
        an integer so that period-averaging is an exact reshape.
        """
        per_period = int(round(rate_hz * self.period))
        if abs(per_period - rate_hz * self.period) > 1e-9:
            raise ValueError(
                f"rate {rate_hz} Hz does not divide the period {self.period} s into a "
                "whole number of samples; period averaging would need resampling"
            )
        n = per_period * int(n_periods)
        t = np.arange(n, dtype=float) / rate_hz
        q, qd, qdd = self(t)
        return t, q, qd, qdd

    def samples_per_period(self, rate_hz: float) -> int:
        return int(round(rate_hz * self.period))

    # -- parameterisation for the optimiser ---------------------------------
    @staticmethod
    def n_free_parameters(n_harmonics: int) -> int:
        """Free parameters per joint: ``q0`` plus ``a`` and ``b`` minus two constraints."""
        return 1 + 2 * n_harmonics - 2

    @staticmethod
    def from_free_parameters(x: np.ndarray, n_harmonics: int,
                             base_frequency: float) -> "FourierTrajectory":
        r"""Build a trajectory that starts and ends at rest, from unconstrained ``x``.

        Layout per joint: ``[q0, a_1..a_{N-1}, b_1..b_{N-1}]``. The withheld
        coefficients close the two boundary conditions:

        .. math::
            a_{N} = -\sum_{l<N} a_l, \qquad b_{N} = -\frac{1}{N}\sum_{l<N} l\, b_l
        """
        if n_harmonics < 2:
            raise ValueError("need at least 2 harmonics to satisfy both rest conditions")
        per_joint = FourierTrajectory.n_free_parameters(n_harmonics)
        x = np.asarray(x, dtype=float).reshape(N_JOINTS, per_joint)

        q0 = x[:, 0]
        a_free = x[:, 1:n_harmonics]                    # (7, N-1)
        b_free = x[:, n_harmonics:]                     # (7, N-1)

        a_last = -a_free.sum(axis=1, keepdims=True)
        weights = np.arange(1, n_harmonics)             # l = 1 .. N-1
        b_last = -(b_free * weights).sum(axis=1, keepdims=True) / n_harmonics

        return FourierTrajectory(q0,
                                 np.hstack([a_free, a_last]),
                                 np.hstack([b_free, b_last]),
                                 base_frequency)

    def to_free_parameters(self) -> np.ndarray:
        """Inverse of :meth:`from_free_parameters` (drops the dependent coefficients)."""
        n = self.n_harmonics
        return np.hstack([self.q0[:, None], self.a[:, :n - 1], self.b[:, :n - 1]]).ravel()

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, object]:
        return {
            "type": "fourier",
            "base_frequency_hz": float(self.base_frequency),
            "n_harmonics": int(self.n_harmonics),
            "q0": [float(v) for v in self.q0],
            "a": [[float(v) for v in row] for row in self.a],
            "b": [[float(v) for v in row] for row in self.b],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "FourierTrajectory":
        return FourierTrajectory(np.asarray(d["q0"], dtype=float),
                                 np.asarray(d["a"], dtype=float),
                                 np.asarray(d["b"], dtype=float),
                                 float(d["base_frequency_hz"]))

    def boundary_residuals(self) -> tuple[float, float]:
        """``(|qd(0)|_inf, |qdd(0)|_inf)`` -- both must be ~0 for the FCI to accept it."""
        _, qd, qdd = self(np.array([0.0]))
        return float(np.abs(qd).max()), float(np.abs(qdd).max())


@dataclass(frozen=True)
class StaticPoseSet:
    """Poses for Stage A, with the bidirectional approach baked in.

    ``poses`` are the configurations at which torque is averaged. When
    ``bidirectional`` is set, each pose is visited twice -- once approached from
    ``pose - offset`` and once from ``pose + offset`` -- and the two averages combined.
    That cancels Coulomb/stiction hysteresis to first order, which is the dominant
    error source in static identification.
    """

    poses: np.ndarray
    approach_offset: np.ndarray
    bidirectional: bool = True

    def __post_init__(self) -> None:
        poses = np.atleast_2d(np.asarray(self.poses, dtype=float))
        if poses.shape[1] != N_JOINTS:
            raise ValueError(f"poses must be (n, 7), got {poses.shape}")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "approach_offset",
                           np.asarray(self.approach_offset, dtype=float).reshape(N_JOINTS))

    @property
    def n_poses(self) -> int:
        return int(self.poses.shape[0])

    def waypoints(self) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """``(approach_from, measure_at, direction)`` triples in execution order."""
        out: list[tuple[np.ndarray, np.ndarray, int]] = []
        for pose in self.poses:
            out.append((pose - self.approach_offset, pose, +1))
            if self.bidirectional:
                out.append((pose + self.approach_offset, pose, -1))
        return out

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "static_poses",
            "bidirectional": bool(self.bidirectional),
            "approach_offset": [float(v) for v in self.approach_offset],
            "poses": [[float(v) for v in row] for row in self.poses],
        }

    @staticmethod
    def from_dict(d: dict[str, object]) -> "StaticPoseSet":
        return StaticPoseSet(np.asarray(d["poses"], dtype=float),
                             np.asarray(d["approach_offset"], dtype=float),
                             bool(d["bidirectional"]))
