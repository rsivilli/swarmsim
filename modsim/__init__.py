"""modsim — 2D/3D sensor coverage modeling and simulation tool.

A tool for simulating sensor detection coverage with support for:
- Binary and probabilistic detection models
- 2D and 3D positioning with altitude
- Multiple sensor layouts (grid, hex, random, manual)
- Object motion (swarm and formation modes)
- Communication network emulation
- Visualization and data export
"""

from __future__ import annotations

import importlib.metadata as _metadata

try:
    __version__ = _metadata.version("modsim")
except _metadata.PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

__author__ = "Robert Sivilli"
__license__ = "MIT"

# Public API exports for backward compatibility
from .config import load_scenario
from .models import (
    Area,
    DetectionMode,
    DetectionRecord,
    GroupMode,
    ObjectGroupConfig,
    ScenarioConfig,
    SensorConfig,
    SensorLayout,
    SimObject,
    SimState,
    SpacingProfile,
    TimestepStats,
    Vec2,
    Vec3,
)
from .report import export_json, print_summary
from .sim import run, run_iter
from .viz import animate_scenario, plot_final

# Export custom errors
from .errors import ConfigError, ModsimError, SimulationError, ValidationError

__all__ = [
    # Version info
    "__version__",
    # Core functions
    "load_scenario",
    "run",
    "run_iter",
    "print_summary",
    "export_json",
    "plot_final",
    "animate_scenario",
    # Data models
    "Area",
    "DetectionMode",
    "DetectionRecord",
    "GroupMode",
    "ObjectGroupConfig",
    "ScenarioConfig",
    "SensorConfig",
    "SensorLayout",
    "SimObject",
    "SimState",
    "SpacingProfile",
    "TimestepStats",
    "Vec2",
    "Vec3",
    # Errors
    "ModsimError",
    "ConfigError",
    "ValidationError",
    "SimulationError",
]
