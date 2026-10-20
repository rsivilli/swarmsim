"""Tests for simulation orchestration and reporting."""

from __future__ import annotations

import json
import pytest
from modsim.models import (
    DetectionMode,
    GroupMode,
    ScenarioConfig,
    SensorConfig,
    SensorLayout,
    SimState,
    TimestepStats,
    Vec2,
    Vec3,
)
from modsim.config import load_scenario
from modsim.sim import build_sensors, run, run_iter
from modsim.report import print_summary, export_json
from conftest import make_group, make_area, small_scenario, SCENARIOS_DIR


class TestSim:
    def test_build_sensors_grid(self):
        cfg = small_scenario(
            sensors=SensorConfig(layout=SensorLayout.GRID, detection_range=2.0, spacing=5.0)
        )
        positions, ranges = build_sensors(cfg)
        # 10km / 5km spacing → 2 columns, 2 rows = 4 sensors
        assert len(positions) == 4
        assert all(r == 2.0 for r in ranges)

    def test_build_sensors_manual(self):
        pts = [Vec2(1.0, 1.0), Vec2(3.0, 3.0)]
        cfg = small_scenario(
            sensors=SensorConfig(layout=SensorLayout.MANUAL, detection_range=1.5, positions=pts)
        )
        positions, ranges = build_sensors(cfg)
        assert len(positions) == 2
        assert positions[0].x == 1.0
        assert positions[1].y == 3.0

    def test_build_sensors_hex(self):
        cfg = small_scenario(
            sensors=SensorConfig(layout=SensorLayout.HEX, detection_range=2.0, spacing=3.0)
        )
        positions, _ = build_sensors(cfg)
        assert len(positions) > 0

    def test_build_sensors_random(self):
        cfg = small_scenario(
            sensors=SensorConfig(layout=SensorLayout.RANDOM, detection_range=2.0, count=20)
        )
        positions, ranges = build_sensors(cfg)
        assert len(positions) == 20
        assert len(ranges) == 20

    def test_run_basic(self):
        state = run(small_scenario())
        assert isinstance(state, SimState)
        assert len(state.timestep_stats) > 0
        assert isinstance(state.detection_log, list)

    def test_run_deterministic(self):
        cfg = small_scenario(
            detection_mode=DetectionMode.PROBABILISTIC,
            pd_function="exponential",
            pd_half_range=1.0,
        )
        s1 = run(cfg)
        s2 = run(cfg)
        assert len(s1.detection_log) == len(s2.detection_log)
        ids1 = [d.obj_id for d in s1.detection_log[:20]]
        ids2 = [d.obj_id for d in s2.detection_log[:20]]
        assert ids1 == ids2

    def test_run_iter_yields_frames(self):
        cfg = small_scenario(duration=5.0, dt=1.0)
        frames = list(run_iter(cfg))
        # Should be at most 5 frames (could be fewer if objects exit early)
        assert 0 < len(frames) <= 5
        step, time_sec, objs, dets, track_states, stats = frames[0]
        assert step == 0
        assert time_sec == 0.0
        assert isinstance(objs, list)
        assert isinstance(stats, TimestepStats)

    def test_early_termination(self):
        # Objects heading right at high speed from near the right edge → exit immediately
        cfg = small_scenario(
            duration=100.0,
            dt=1.0,
            groups=[
                make_group(
                    count=3,
                    speed=0.5,
                    mode=GroupMode.FORMATION,
                    heading=Vec2(1.0, 0.0),
                    spawn_center=Vec2(9.5, 5.0),
                    spawn_radius=0.1,
                )
            ],
        )
        state = run(cfg)
        assert len(state.timestep_stats) < 100


# ===========================================================================
# report
# ===========================================================================


class TestReport:
    def _make_state(self) -> SimState:
        return run(small_scenario())

    def test_print_summary_runs(self, capsys):
        print_summary(self._make_state())
        captured = capsys.readouterr()
        assert "Simulation Complete" in captured.out

    def test_export_json(self, tmp_path):
        path = str(tmp_path / "results.json")
        export_json(self._make_state(), path)
        with open(path) as f:
            data = json.load(f)
        assert "timestep_stats" in data
        assert "detection_log" in data
        assert "sensor_positions" in data


# ===========================================================================
# integration — run small bundled scenarios end to end
# ===========================================================================

# The full set includes 50k-object runs; CI validates all of them separately
_FAST_SCENARIOS = ["basic_grid", "corridor", "minimal_test", "needle_thread", "fast_dash"]


def _example_yamls():
    return [SCENARIOS_DIR / f"{name}.yaml" for name in _FAST_SCENARIOS]


@pytest.mark.parametrize("yaml_path", _example_yamls(), ids=lambda p: p.stem)
def test_example_scenario(yaml_path):
    cfg = load_scenario(yaml_path)
    state = run(cfg)
    assert isinstance(state.detection_log, list)
    assert len(state.timestep_stats) > 0


# ===========================================================================
# Algorithmic correctness
# ===========================================================================


class TestCoverageGeometry:
    """Verify coverage stats against analytically-known configurations."""

    def test_object_always_in_range_gives_100_percent(self):
        # Stationary object at sensor location, binary detection → 100% coverage every step
        cfg = ScenarioConfig(
            name="t",
            area=make_area(),
            dt=1.0,
            duration=10.0,
            detection_mode=DetectionMode.BINARY,
            groups=[
                make_group(
                    count=1,
                    mode=GroupMode.FORMATION,
                    speed=0.0,
                    heading=Vec2(1.0, 0.0),
                    spawn_center=Vec2(5.0, 5.0),
                    spawn_radius=0.0,
                )
            ],
            sensors=SensorConfig(
                layout=SensorLayout.MANUAL,
                detection_range=1.0,
                positions=[Vec3(5.0, 5.0, 0.0)],
            ),
            detection_range=1.0,
            seed=1,
        )
        state = run(cfg)
        for s in state.timestep_stats:
            assert s.coverage_pct == pytest.approx(100.0)

    def test_object_never_in_range_gives_0_percent(self):
        # Object far from any sensor → 0% coverage
        cfg = ScenarioConfig(
            name="t",
            area=make_area(100, 100),
            dt=1.0,
            duration=5.0,
            detection_mode=DetectionMode.BINARY,
            groups=[
                make_group(
                    count=1,
                    mode=GroupMode.FORMATION,
                    speed=0.01,
                    heading=Vec2(1.0, 0.0),
                    spawn_center=Vec2(50.0, 50.0),
                    spawn_radius=0.0,
                )
            ],
            sensors=SensorConfig(
                layout=SensorLayout.MANUAL,
                detection_range=1.0,
                positions=[Vec3(0.0, 0.0, 0.0)],
            ),
            detection_range=1.0,
            seed=1,
        )
        state = run(cfg)
        for s in state.timestep_stats:
            assert s.coverage_pct == pytest.approx(0.0)

    def test_grid_sensor_spacing_vs_range_full_coverage(self):
        # Grid sensors spaced at 2.0km with range 1.5km → overlapping circles
        # A slow-moving object in the center should be detected at every timestep
        cfg = ScenarioConfig(
            name="t",
            area=make_area(6, 6),
            dt=1.0,
            duration=20.0,
            detection_mode=DetectionMode.BINARY,
            groups=[
                make_group(
                    count=1,
                    mode=GroupMode.FORMATION,
                    speed=0.01,
                    heading=Vec2(1.0, 0.0),
                    spawn_center=Vec2(3.0, 3.0),
                    spawn_radius=0.0,
                )
            ],
            sensors=SensorConfig(
                layout=SensorLayout.GRID,
                detection_range=1.5,
                spacing=2.0,
            ),
            detection_range=1.5,
            seed=1,
        )
        state = run(cfg)
        # Object starts at center of a 6x6 area with 2km-spaced sensors with 1.5km range
        # It should be in range of at least one sensor for all/most timesteps
        covered_steps = sum(1 for s in state.timestep_stats if s.coverage_pct == 100.0)
        assert covered_steps == len(state.timestep_stats)
