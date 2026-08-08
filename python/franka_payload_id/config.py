"""Typed loading of the YAML files in ``config/``.

A single :class:`Config` object is threaded through the pipeline so that no module
reads a YAML file on its own and no magic numbers appear in the algorithms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_HERE = Path(__file__).resolve()
# .../<repo>/python/franka_payload_id/config.py -> parents[2] is the repo root,
# i.e. the directory that holds config/ and assets/.
_REPO_ROOT = _HERE.parents[2]


def _repo_root() -> Path:
    """Directory containing ``config/`` and ``assets/``."""
    return _REPO_ROOT


def config_dir() -> Path:
    env = os.environ.get("FPI_CONFIG_DIR")
    return Path(env).resolve() if env else _repo_root() / "config"


def asset_dir() -> Path:
    env = os.environ.get("FPI_ASSET_DIR")
    return Path(env).resolve() if env else _repo_root() / "assets"


def data_dir() -> Path:
    env = os.environ.get("FPI_DATA_DIR")
    return Path(env).resolve() if env else _repo_root() / "data"


def urdf_path() -> Path:
    return asset_dir() / "panda_arm.urdf"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Joint limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PandaLimits:
    """Official Panda (FER) joint limits. Arrays are length 7."""

    q_min: np.ndarray
    q_max: np.ndarray
    qd_max: np.ndarray
    qdd_max: np.ndarray
    qddd_max: np.ndarray
    tau_max: np.ndarray
    taud_max: np.ndarray
    q_elbow_flip: float
    rated_payload: float

    @staticmethod
    def load(path: Path | None = None) -> "PandaLimits":
        raw = _load_yaml(path or config_dir() / "panda_limits.yaml")
        arr = lambda k: np.asarray(raw[k], dtype=float)  # noqa: E731
        return PandaLimits(
            q_min=arr("q_min"),
            q_max=arr("q_max"),
            qd_max=arr("qd_max"),
            qdd_max=arr("qdd_max"),
            qddd_max=arr("qddd_max"),
            tau_max=arr("tau_max"),
            taud_max=arr("taud_max"),
            q_elbow_flip=float(raw["q_elbow_flip"]),
            rated_payload=float(raw["rated_payload"]),
        )

    def derated(self, position: float, velocity: float,
                acceleration: float, jerk: float, torque: float) -> "PandaLimits":
        """Shrink the limits by the given fractions.

        Position derating shrinks the range about its centre rather than scaling the
        raw bounds, because joint 4 is entirely negative and joint 6 almost entirely
        positive -- scaling those bounds directly would move them in the wrong
        direction.
        """
        centre = 0.5 * (self.q_min + self.q_max)
        half = 0.5 * (self.q_max - self.q_min) * position
        return PandaLimits(
            q_min=centre - half,
            q_max=centre + half,
            qd_max=self.qd_max * velocity,
            qdd_max=self.qdd_max * acceleration,
            qddd_max=self.qddd_max * jerk,
            tau_max=self.tau_max * torque,
            taud_max=self.taud_max,
            q_elbow_flip=self.q_elbow_flip,
            rated_payload=self.rated_payload,
        )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HalfSpace:
    """Allowed region ``normal . p + offset >= 0`` in the robot base frame."""

    name: str
    normal: np.ndarray
    offset: float
    measured: bool = False

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Signed distance of ``points`` (..., 3) into the allowed region [m]."""
        return np.asarray(points) @ self.normal + self.offset


@dataclass(frozen=True)
class MonitoredPoint:
    """A sphere rigidly attached to ``frame``, checked against every half-space."""

    frame: str
    offset: np.ndarray
    radius: float


@dataclass(frozen=True)
class Workspace:
    margin: float
    walls: list[HalfSpace]
    table: HalfSpace
    q1_min: float
    q1_max: float
    monitored_points: list[MonitoredPoint]

    @property
    def half_spaces(self) -> list[HalfSpace]:
        return [*self.walls, self.table]

    @property
    def all_measured(self) -> bool:
        return all(h.measured for h in self.half_spaces)

    @staticmethod
    def load(path: Path | None = None, tool: "ToolSpec | None" = None) -> "Workspace":
        raw = _load_yaml(path or config_dir() / "workspace.yaml")

        def _hs(d: dict[str, Any]) -> HalfSpace:
            n = np.asarray(d["normal"], dtype=float)
            norm = np.linalg.norm(n)
            if norm == 0.0:
                raise ValueError(f"half-space {d['name']!r} has a zero normal")
            return HalfSpace(str(d["name"]), n / norm, float(d["offset"]) / norm,
                             bool(d.get("measured", False)))

        pts = [
            MonitoredPoint(str(p["frame"]),
                           np.asarray(p.get("offset", [0.0, 0.0, 0.0]), dtype=float),
                           float(p["radius"]))
            for p in raw["monitored_points"]
        ]
        # The tool is a rigid extension of the flange, so its bounding sphere is
        # appended automatically rather than duplicated in workspace.yaml.
        if tool is not None:
            centre, radius = tool.bounding_sphere()
            pts.append(MonitoredPoint("panda_link8", centre, radius))

        box = raw.get("joint_box", {})
        return Workspace(
            margin=float(raw["margin"]),
            walls=[_hs(w) for w in raw["walls"]],
            table=_hs(raw["table"]),
            q1_min=float(box.get("q1_min", -np.inf)),
            q1_max=float(box.get("q1_max", np.inf)),
            monitored_points=pts,
        )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    mass_scale: float | None
    mass_scale_tolerance: float
    use_mass_constraint: bool
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    cad_mass: float | None = None
    cad_com: np.ndarray | None = None
    cad_inertia: np.ndarray | None = None

    @staticmethod
    def load(path: Path | None = None) -> "ToolSpec":
        raw = _load_yaml(path or config_dir() / "tool.yaml")
        bb = raw["bounding_box"]
        cad = raw.get("cad") or {}
        return ToolSpec(
            name=str(raw.get("name", "tool")),
            mass_scale=None if raw.get("mass_scale") is None else float(raw["mass_scale"]),
            mass_scale_tolerance=float(raw.get("mass_scale_tolerance", 0.02)),
            use_mass_constraint=bool(raw.get("use_mass_constraint", True)),
            bbox_min=np.asarray(bb["min"], dtype=float),
            bbox_max=np.asarray(bb["max"], dtype=float),
            cad_mass=None if cad.get("mass") is None else float(cad["mass"]),
            cad_com=None if cad.get("com") is None else np.asarray(cad["com"], dtype=float),
            cad_inertia=(None if cad.get("inertia") is None
                         else np.asarray(cad["inertia"], dtype=float)),
        )

    def bounding_sphere(self) -> tuple[np.ndarray, float]:
        """Centre (flange frame) and radius of a sphere bounding the tool."""
        centre = 0.5 * (self.bbox_min + self.bbox_max)
        radius = float(0.5 * np.linalg.norm(self.bbox_max - self.bbox_min))
        return centre, radius


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    """Raw nested dict from ``experiment.yaml`` plus convenience accessors."""

    raw: dict[str, Any] = field(repr=False)

    @staticmethod
    def load(path: Path | None = None) -> "Experiment":
        return Experiment(_load_yaml(path or config_dir() / "experiment.yaml"))

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def static(self) -> dict[str, Any]:
        return self.raw["static"]

    @property
    def trajectory(self) -> dict[str, Any]:
        return self.raw["trajectory"]

    @property
    def preprocess(self) -> dict[str, Any]:
        return self.raw["preprocess"]

    @property
    def estimator(self) -> dict[str, Any]:
        return self.raw["estimator"]

    @property
    def validation(self) -> dict[str, Any]:
        return self.raw["validation"]

    @property
    def synthetic(self) -> dict[str, Any]:
        return self.raw["synthetic"]


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RobotConfig:
    fci_ip: str
    log_size: int
    move_speed_factor: float
    collision: dict[str, list[float]]
    joint_impedance: list[float]
    zero_load_during_collection: bool

    @staticmethod
    def load(path: Path | None = None) -> "RobotConfig":
        raw = _load_yaml(path or config_dir() / "robot.yaml")
        return RobotConfig(
            fci_ip=str(raw["fci_ip"]),
            log_size=int(raw["log_size"]),
            move_speed_factor=float(raw["move_speed_factor"]),
            collision=raw["collision"],
            joint_impedance=list(raw["joint_impedance"]),
            zero_load_during_collection=bool(raw["zero_load_during_collection"]),
        )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    limits: PandaLimits
    workspace: Workspace
    tool: ToolSpec
    experiment: Experiment
    robot: RobotConfig

    @staticmethod
    def load(directory: Path | None = None) -> "Config":
        d = directory or config_dir()
        tool = ToolSpec.load(d / "tool.yaml")
        return Config(
            limits=PandaLimits.load(d / "panda_limits.yaml"),
            workspace=Workspace.load(d / "workspace.yaml", tool=tool),
            tool=tool,
            experiment=Experiment.load(d / "experiment.yaml"),
            robot=RobotConfig.load(d / "robot.yaml"),
        )

    def derated_limits(self) -> PandaLimits:
        d = self.experiment.trajectory["derate"]
        return self.limits.derated(
            position=float(d["position"]),
            velocity=float(d["velocity"]),
            acceleration=float(d["acceleration"]),
            jerk=float(d["jerk"]),
            torque=float(d["torque"]),
        )
