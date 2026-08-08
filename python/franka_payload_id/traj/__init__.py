"""Excitation trajectory design and safety checking."""

from .fourier import FourierTrajectory, StaticPoseSet  # noqa: F401
from .constraints import ConstraintReport, check_configurations, check_trajectory  # noqa: F401
from .optimize import optimize_trajectory, regressor_condition  # noqa: F401
