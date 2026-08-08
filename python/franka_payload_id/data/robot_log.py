"""On-disk format shared by the C++ collector and this package.

A run is two files:

``<name>.bin``
    Little-endian ``float64`` records, row-major, ``len(SCHEMA)`` values per sample and
    nothing else. The collector writes it in one block after the motion has finished,
    so the real-time callback never touches the filesystem.

``<name>.meta.json``
    Run metadata *including the schema the collector actually used*. The schema is
    written by the C++ side and validated here on load, so a field added on one side
    and not the other fails loudly instead of silently shifting every column.

Everything is ``float64`` -- including the few integer-valued channels -- so the
record layout is trivially describable and the C++ writer needs no serialisation
library. The conversion to parquet happens on this side, keeping Arrow out of the
robot image.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

N_JOINTS = 7

SCHEMA: tuple[str, ...] = (
    "seq",
    "time_s",                 # robot-side clock, franka::Duration
    "dt_s",                   # control period reported for this tick; 1 ms unless packets were lost
    *[f"q_{i}" for i in range(N_JOINTS)],
    *[f"dq_{i}" for i in range(N_JOINTS)],
    *[f"q_d_{i}" for i in range(N_JOINTS)],
    *[f"dq_d_{i}" for i in range(N_JOINTS)],
    *[f"ddq_d_{i}" for i in range(N_JOINTS)],
    *[f"tau_J_{i}" for i in range(N_JOINTS)],          # measured link-side torque -- the signal
    *[f"tau_J_d_{i}" for i in range(N_JOINTS)],
    *[f"dtau_J_{i}" for i in range(N_JOINTS)],
    *[f"tau_ext_{i}" for i in range(N_JOINTS)],        # tau_ext_hat_filtered
    *[f"O_T_EE_{i}" for i in range(16)],               # column-major 4x4
    "control_command_success_rate",
    "robot_mode",
    "errors",
)

RECORD_SIZE = len(SCHEMA)


@dataclass
class RunMetadata:
    """Sidecar contents. Everything needed to reproduce or reject a run."""

    run_id: str = ""
    kind: str = ""                      # "trajectory" | "static" | "check"
    loaded: bool = True                 # tool attached?
    robot_ip: str = ""
    libfranka_version: str = ""
    robot_serial: str = ""
    system_version: str = ""
    collector_git_sha: str = ""
    started_at: str = ""
    finished_at: str = ""
    sample_rate_hz: float = 1000.0
    samples_per_period: int = 0
    n_periods: int = 0
    n_blocks: int = 1
    """How many separate collection blocks this log concatenates.

    Under the ABBA schedule the settling period must be dropped from each
    block, not just the first, or the drift cancellation is unbalanced.
    """
    trajectory: dict = field(default_factory=dict)
    # Configured end-effector / load state at collection time. These MUST be identical
    # (and normally zero) across a loaded/bare pair, or the internal controller behaves
    # differently in the two runs and the difference no longer isolates the payload.
    m_ee: float = 0.0
    F_x_Cee: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    I_ee: list = field(default_factory=lambda: [0.0] * 9)
    m_load: float = 0.0
    F_x_Cload: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    I_load: list = field(default_factory=lambda: [0.0] * 9)
    F_T_NE: list = field(default_factory=list)
    NE_T_EE: list = field(default_factory=list)
    schema: list = field(default_factory=lambda: list(SCHEMA))
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d: dict) -> "RunMetadata":
        known = {k: v for k, v in d.items() if k in RunMetadata().__dict__}
        meta = RunMetadata(**known)
        meta.notes = str(d.get("notes", ""))
        return meta

    def load_configuration(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Total configured load at collection time ``(mass, com, inertia_flat)``."""
        return (float(self.m_ee) + float(self.m_load),
                np.asarray(self.F_x_Cee, dtype=float) + np.asarray(self.F_x_Cload, dtype=float),
                np.asarray(self.I_ee, dtype=float) + np.asarray(self.I_load, dtype=float))


@dataclass
class RunLog:
    """A parsed collector run."""

    values: np.ndarray            # (K, RECORD_SIZE)
    meta: RunMetadata

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != RECORD_SIZE:
            raise ValueError(
                f"expected records of width {RECORD_SIZE}, got {self.values.shape}")

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    def column(self, name: str) -> np.ndarray:
        try:
            return self.values[:, SCHEMA.index(name)]
        except ValueError as exc:
            raise KeyError(f"unknown column {name!r}") from exc

    def block(self, prefix: str, width: int = N_JOINTS) -> np.ndarray:
        """Stack ``prefix_0 .. prefix_{width-1}`` into a ``(K, width)`` array."""
        idx = [SCHEMA.index(f"{prefix}_{i}") for i in range(width)]
        return self.values[:, idx]

    # Convenience accessors for the channels the pipeline actually consumes.
    @property
    def q(self) -> np.ndarray:
        return self.block("q")

    @property
    def dq(self) -> np.ndarray:
        return self.block("dq")

    @property
    def ddq_d(self) -> np.ndarray:
        return self.block("ddq_d")

    @property
    def tau_J(self) -> np.ndarray:
        """Measured link-side joint torque [Nm] -- NOT gravity compensated."""
        return self.block("tau_J")

    @property
    def tau_ext(self) -> np.ndarray:
        return self.block("tau_ext")

    @property
    def time_s(self) -> np.ndarray:
        return self.column("time_s")

    @property
    def success_rate(self) -> np.ndarray:
        return self.column("control_command_success_rate")

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.values, columns=list(SCHEMA))


def load_run(path: Path | str) -> RunLog:
    """Load a ``.bin`` + ``.meta.json`` pair. ``path`` may name either file."""
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in (".bin", ".json") else path
    if stem.name.endswith(".meta"):
        stem = stem.with_name(stem.name[: -len(".meta")])
    bin_path = stem.with_suffix(".bin")
    meta_path = stem.with_name(stem.name + ".meta.json")

    if not bin_path.exists():
        raise FileNotFoundError(f"no record file at {bin_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"no metadata sidecar at {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = RunMetadata.from_dict(json.load(fh))

    written = list(meta.schema) if meta.schema else list(SCHEMA)
    if written != list(SCHEMA):
        raise ValueError(
            f"{meta_path} was written with a different record schema.\n"
            f"  collector wrote {len(written)} fields, this build expects {len(SCHEMA)}.\n"
            "  Rebuild the collector and this package from the same commit."
        )

    raw = np.fromfile(bin_path, dtype="<f8")
    if raw.size % RECORD_SIZE:
        raise ValueError(
            f"{bin_path} holds {raw.size} values, not a multiple of the {RECORD_SIZE}-wide "
            "record; the file is truncated or corrupt")
    return RunLog(raw.reshape(-1, RECORD_SIZE), meta)


def save_run(path: Path | str, values: np.ndarray, meta: RunMetadata) -> tuple[Path, Path]:
    """Write a run in the collector's format. Used by tests and the simulator."""
    path = Path(path)
    stem = path.with_suffix("") if path.suffix in (".bin", ".json") else path
    bin_path = stem.with_suffix(".bin")
    meta_path = stem.with_name(stem.name + ".meta.json")
    bin_path.parent.mkdir(parents=True, exist_ok=True)

    values = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    if values.ndim != 2 or values.shape[1] != RECORD_SIZE:
        raise ValueError(f"expected (K, {RECORD_SIZE}) records, got {values.shape}")
    values.tofile(bin_path)

    meta.schema = list(SCHEMA)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta.to_dict(), fh, indent=2, sort_keys=True)
    return bin_path, meta_path


def to_parquet(run: RunLog, path: Path | str) -> Path:
    """Convert to parquet for archival and downstream tooling."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = run.to_dataframe()
    df.attrs.update(run.meta.to_dict())
    df.to_parquet(path, compression="zstd", index=False)
    return path


def make_records(*, seq: np.ndarray, time_s: np.ndarray, dt_s: np.ndarray,
                 q: np.ndarray, dq: np.ndarray, q_d: np.ndarray, dq_d: np.ndarray,
                 ddq_d: np.ndarray, tau_J: np.ndarray, tau_J_d: np.ndarray,
                 dtau_J: np.ndarray, tau_ext: np.ndarray, o_t_ee: np.ndarray,
                 success_rate: np.ndarray, robot_mode: np.ndarray,
                 errors: np.ndarray) -> np.ndarray:
    """Assemble a ``(K, RECORD_SIZE)`` block in schema order."""
    parts = [np.asarray(seq, dtype=float).reshape(-1, 1),
             np.asarray(time_s, dtype=float).reshape(-1, 1),
             np.asarray(dt_s, dtype=float).reshape(-1, 1),
             q, dq, q_d, dq_d, ddq_d, tau_J, tau_J_d, dtau_J, tau_ext, o_t_ee,
             np.asarray(success_rate, dtype=float).reshape(-1, 1),
             np.asarray(robot_mode, dtype=float).reshape(-1, 1),
             np.asarray(errors, dtype=float).reshape(-1, 1)]
    out = np.hstack([np.asarray(p, dtype=float) for p in parts])
    if out.shape[1] != RECORD_SIZE:
        raise ValueError(f"assembled width {out.shape[1]} != schema width {RECORD_SIZE}")
    return out
