"""Tests for configuration loading and parsing."""

from __future__ import annotations

from modsim.models import DetectionMode, GroupMode, SensorLayout
from modsim.config import load_scenario
from conftest import EXAMPLES_DIR


class TestConfig:
    """Test YAML configuration loading."""

    def test_load_basic_grid(self):
        """Load basic_grid.yaml and verify structure."""
        cfg = load_scenario(EXAMPLES_DIR / "basic_grid.yaml")
        assert cfg.name == "Grid sensors vs loose swarm"
        assert cfg.area.width_km == 10.0
        assert cfg.area.height_km == 10.0
        assert len(cfg.groups) == 1
        assert cfg.groups[0].mode is GroupMode.SWARM
        assert cfg.sensors is not None
        assert cfg.sensors.layout is SensorLayout.GRID
        assert cfg.detection_mode is DetectionMode.BINARY

    def test_load_corridor(self):
        """Load corridor.yaml with probabilistic detection and manual sensors."""
        cfg = load_scenario(EXAMPLES_DIR / "corridor.yaml")
        assert cfg.detection_mode is DetectionMode.PROBABILISTIC
        assert cfg.sensors is not None
        assert cfg.sensors.layout is SensorLayout.MANUAL
        assert cfg.sensors.positions is not None
        assert len(cfg.sensors.positions) == 6
        assert len(cfg.groups) == 2

    def test_defaults(self, tmp_path):
        """Minimal YAML should use sensible defaults."""
        yaml_content = "name: minimal\narea:\n  width_km: 5.0\n  height_km: 5.0\nduration: 10\n"
        p = tmp_path / "minimal.yaml"
        p.write_text(yaml_content)
        cfg = load_scenario(p)
        assert cfg.dt == 1.0
        assert cfg.detection_mode is DetectionMode.BINARY
