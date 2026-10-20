from pathlib import Path

import pytest
from click.testing import CliRunner

from modsim.models import (
    Area,
    DetectionMode,
    GroupMode,
    ObjectGroupConfig,
    ScenarioConfig,
    SensorConfig,
    SensorLayout,
    SpacingProfile,
    Vec2,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


# ===========================================================================
# Shared Test Fixtures
# ===========================================================================


@pytest.fixture
def cli_runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def example_scenarios():
    """Path to example scenario directory."""
    return EXAMPLES_DIR


@pytest.fixture
def basic_scenario(example_scenarios):
    """Path to basic_grid.yaml."""
    return example_scenarios / "basic_grid.yaml"


# ===========================================================================
# Shared Test Helpers
# ===========================================================================


def make_group(
    count: int = 10,
    mode: GroupMode = GroupMode.SWARM,
    heading: Vec2 = Vec2(1.0, 0.0),
    speed: float | None = None,
    speed_min: float = 0.05,
    speed_max: float = 0.08,
    spawn_center: Vec2 = Vec2(5.0, 5.0),
    spawn_radius: float = 1.0,
    spacing_profile: SpacingProfile = SpacingProfile.RANDOM,
) -> ObjectGroupConfig:
    """Create a test ObjectGroupConfig."""
    return ObjectGroupConfig(
        group_id="test",
        count=count,
        mode=mode,
        heading=heading,
        speed_min=speed_min,
        speed_max=speed_max,
        spawn_center=spawn_center,
        spawn_radius=spawn_radius,
        spacing_profile=spacing_profile,
        speed=speed,
    )


def make_area(w: float = 10.0, h: float = 10.0) -> Area:
    """Create a test Area."""
    return Area(width_km=w, height_km=h)


def small_scenario(**overrides) -> ScenarioConfig:
    """Create a small test scenario with sensible defaults."""
    defaults = dict(
        name="test",
        area=make_area(),
        dt=1.0,
        duration=10.0,
        detection_mode=DetectionMode.BINARY,
        groups=[make_group(count=5)],
        sensors=SensorConfig(
            layout=SensorLayout.GRID,
            detection_range=2.0,
            spacing=5.0,
        ),
        detection_range=2.0,
        pd_function="exponential",
        pd_half_range=1.0,
        seed=42,
    )
    defaults.update(overrides)
    return ScenarioConfig(**defaults)
