"""Tests for object motion (spawning, stepping, formations)."""

from __future__ import annotations

import math
import random
import pytest
from modsim.models import GroupMode, SimObject, SpacingProfile, Vec2
from modsim.motion import spawn_group, step_objects, step_formation
from conftest import make_group, make_area


class TestMotion:
    def test_spawn_count(self):
        group = make_group(count=25)
        objs = spawn_group(group, make_area(), random.Random(42))
        assert len(objs) == 25

    def test_spawn_within_bounds(self):
        area = make_area()
        group = make_group(count=100, spawn_center=Vec2(5.0, 5.0), spawn_radius=2.0)
        objs = spawn_group(group, area, random.Random(42))
        for o in objs:
            assert 0 <= o.x <= area.width_km
            assert 0 <= o.y <= area.height_km

    def test_spawn_formation_shared_velocity(self):
        group = make_group(count=10, mode=GroupMode.FORMATION, speed=0.06)
        objs = spawn_group(group, make_area(), random.Random(42))
        vx0, vy0 = objs[0].vx, objs[0].vy
        for o in objs:
            assert o.vx == vx0
            assert o.vy == vy0

    def test_spawn_swarm_varied_velocity(self):
        group = make_group(count=20, mode=GroupMode.SWARM)
        objs = spawn_group(group, make_area(), random.Random(42))
        velocities = {(o.vx, o.vy) for o in objs}
        assert len(velocities) > 1

    def test_step_moves_objects(self):
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0.1, vy=-0.05, z=0.0, vz=0.0)
        result = step_objects([obj], dt=1.0, area=make_area())
        assert len(result) == 1
        assert result[0].x == pytest.approx(5.1)
        assert result[0].y == pytest.approx(4.95)

    def test_step_removes_out_of_bounds(self):
        obj = SimObject(obj_id="a", group_id="g", x=9.9, y=5.0, vx=0.2, vy=0.0, z=0.0, vz=0.0)
        result = step_objects([obj], dt=1.0, area=make_area())
        assert len(result) == 0

    def test_step_formation_rigid(self):
        objs = [
            SimObject(obj_id="a", group_id="g", x=3.0, y=5.0, vx=0.1, vy=0.0, z=0.0, vz=0.0),
            SimObject(obj_id="b", group_id="g", x=4.0, y=5.0, vx=0.1, vy=0.0, z=0.0, vz=0.0),
        ]
        result = step_formation(objs, dt=1.0, area=make_area())
        assert len(result) == 2
        # Both moved by same amount
        assert result[0].x == pytest.approx(3.1)
        assert result[1].x == pytest.approx(4.1)
        # Relative spacing preserved
        assert result[1].x - result[0].x == pytest.approx(1.0)

    @pytest.mark.parametrize("profile", list(SpacingProfile))
    def test_spawn_all_profiles(self, profile):
        group = make_group(count=15, spacing_profile=profile)
        objs = spawn_group(group, make_area(), random.Random(42))
        assert len(objs) == 15


# ===========================================================================
# detection
# ===========================================================================


class TestHeadingNormalization:
    """Verify that heading vectors are normalized and produce correct velocities."""

    def test_non_unit_heading_is_normalized(self):
        # heading (3, 4) should normalize to (0.6, 0.8)
        group = make_group(count=1, mode=GroupMode.FORMATION, speed=0.1, heading=Vec2(3.0, 4.0))
        objs = spawn_group(group, make_area(), random.Random(1))
        expected_vx = 0.6 * 0.1
        expected_vy = 0.8 * 0.1
        assert objs[0].vx == pytest.approx(expected_vx, abs=1e-10)
        assert objs[0].vy == pytest.approx(expected_vy, abs=1e-10)

    def test_speed_magnitude_matches_config(self):
        # Regardless of heading direction, speed magnitude should match
        for hx, hy in [(1, 0), (0, 1), (1, 1), (-3, 4), (0.1, -0.9)]:
            group = make_group(count=1, mode=GroupMode.FORMATION, speed=0.07, heading=Vec2(hx, hy))
            objs = spawn_group(group, make_area(), random.Random(1))
            actual_speed = math.hypot(objs[0].vx, objs[0].vy)
            assert actual_speed == pytest.approx(0.07, abs=1e-10)

    def test_swarm_mean_heading_near_configured(self):
        # With many objects, mean swarm heading should approximate the configured heading
        group = make_group(
            count=500,
            mode=GroupMode.SWARM,
            heading=Vec2(1.0, 0.0),
            speed_min=0.05,
            speed_max=0.05,  # fixed speed so we isolate heading
        )
        objs = spawn_group(group, make_area(100, 100), random.Random(42))
        mean_vx = sum(o.vx for o in objs) / len(objs)
        mean_vy = sum(o.vy for o in objs) / len(objs)
        # Mean heading should be ~(1, 0) * speed, so vy should be near 0
        assert mean_vx == pytest.approx(0.05, abs=0.005)
        assert mean_vy == pytest.approx(0.0, abs=0.005)


class TestMotionGeometry:
    """Verify exact positions over multi-step trajectories."""

    def test_position_after_n_steps(self):
        # Object at (2, 3) with velocity (0.1, -0.05), dt=2.0, after 5 steps:
        # expected position = (2 + 5*2*0.1, 3 + 5*2*(-0.05)) = (3.0, 2.5)
        area = make_area(20, 20)
        obj = SimObject(obj_id="a", group_id="g", x=2.0, y=3.0, vx=0.1, vy=-0.05, z=0.0, vz=0.0)
        for _ in range(5):
            result = step_objects([obj], dt=2.0, area=area)
            assert len(result) == 1
            obj = result[0]
        assert obj.x == pytest.approx(3.0, abs=1e-10)
        assert obj.y == pytest.approx(2.5, abs=1e-10)

    def test_formation_preserves_all_pairwise_distances(self):
        # Create a formation, step it 20 times, verify all pairwise distances are invariant
        group = make_group(
            count=8,
            mode=GroupMode.FORMATION,
            speed=0.04,
            heading=Vec2(1.0, 0.5),
            spacing_profile=SpacingProfile.UNIFORM,
        )
        objs = spawn_group(group, make_area(50, 50), random.Random(7))

        def pairwise_distances(objects):
            dists = []
            for i in range(len(objects)):
                for j in range(i + 1, len(objects)):
                    d = math.hypot(objects[i].x - objects[j].x, objects[i].y - objects[j].y)
                    dists.append(d)
            return dists

        initial_dists = pairwise_distances(objs)

        for _ in range(20):
            objs = step_formation(objs, dt=1.0, area=make_area(50, 50))
            if not objs:
                break

        if objs:
            final_dists = pairwise_distances(objs)
            assert len(final_dists) == len(initial_dists)
            for d0, d1 in zip(initial_dists, final_dists):
                assert d1 == pytest.approx(d0, abs=1e-10)

    def test_diagonal_motion_distance(self):
        # Object moving at 45 degrees at speed s for time t should travel s*t total distance
        speed = 0.1
        mag = math.hypot(1.0, 1.0)
        vx = (1.0 / mag) * speed
        vy = (1.0 / mag) * speed
        obj = SimObject(obj_id="a", group_id="g", x=1.0, y=1.0, vx=vx, vy=vy, z=0.0, vz=0.0)
        x0, y0 = obj.x, obj.y
        for _ in range(10):
            step_objects([obj], dt=1.0, area=make_area(50, 50))
        distance_traveled = math.hypot(obj.x - x0, obj.y - y0)
        assert distance_traveled == pytest.approx(speed * 10, abs=1e-10)
