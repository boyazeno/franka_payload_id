"""Shared fixtures. Everything here is hardware-free."""

from __future__ import annotations

import numpy as np
import pytest

from franka_payload_id.config import Config
from franka_payload_id.model import PandaModel, phi_from_mci


@pytest.fixture(scope="session")
def panda() -> PandaModel:
    return PandaModel.load()


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load()


@pytest.fixture(scope="session")
def tool_phi() -> np.ndarray:
    """A plausible tool: 0.73 kg, CoM 6 cm along +z of the flange, small inertia."""
    inertia_com = np.array([
        [1.10e-3, 5.0e-5, 3.0e-5],
        [5.0e-5, 1.40e-3, 2.0e-5],
        [3.0e-5, 2.0e-5, 0.90e-3],
    ])
    return phi_from_mci(0.73, np.array([-0.010, 0.020, 0.060]), inertia_com)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(20260808)


def sample_states(panda: PandaModel, rng: np.random.Generator, n: int,
                  vel: float = 1.0, acc: float = 2.0):
    """Random (q, v, a) triples inside the real joint position limits."""
    lo = np.asarray(panda.model.lowerPositionLimit)
    hi = np.asarray(panda.model.upperPositionLimit)
    for _ in range(n):
        yield (rng.uniform(lo, hi),
               rng.uniform(-vel, vel, panda.nv),
               rng.uniform(-acc, acc, panda.nv))
