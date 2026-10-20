"""Tests for detection logic (binary, probabilistic, Pd curves)."""

from __future__ import annotations

import math
import random
import pytest
from modsim.models import DetectionMode, SimObject, Vec3
from modsim.detection import detect_all, _compute_pd


class TestDetection:
    def test_binary_in_range(self):
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0, vy=0, z=0.0, vz=0.0)
        sensor = Vec3(5.5, 5.0, 0.0)  # 0.5 km away
        recs = detect_all([obj], [sensor], [1.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 1
        assert recs[0].detected is True
        assert recs[0].pd == 1.0

    def test_binary_out_of_range(self):
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0, vy=0, z=0.0, vz=0.0)
        sensor = Vec3(8.0, 5.0, 0.0)  # 3.0 km away
        recs = detect_all([obj], [sensor], [1.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 0

    def test_binary_exact_boundary(self):
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0, vy=0, z=0.0, vz=0.0)
        sensor = Vec3(6.0, 5.0, 0.0)  # exactly 1.0 km
        recs = detect_all([obj], [sensor], [1.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 1

    def test_probabilistic_pd_exponential_at_half_range(self):
        pd = _compute_pd(1.5, max_range=3.0, function="exponential", half_range=1.5)
        assert pd == pytest.approx(0.5, abs=0.01)

    def test_probabilistic_pd_sigmoid_at_half_range(self):
        pd = _compute_pd(1.5, max_range=3.0, function="sigmoid", half_range=1.5)
        assert pd == pytest.approx(0.5, abs=0.01)

    def test_probabilistic_pd_at_zero(self):
        assert _compute_pd(0.0, 3.0, "exponential", 1.5) == 1.0
        assert _compute_pd(0.0, 3.0, "sigmoid", 1.5) == 1.0

    def test_probabilistic_pd_at_max(self):
        assert _compute_pd(3.0, 3.0, "exponential", 1.5) == 0.0
        assert _compute_pd(5.0, 3.0, "sigmoid", 1.5) == 0.0

    def test_detect_all_empty_objects(self):
        assert detect_all([], [Vec3(0, 0, 0.0)], [1.0], DetectionMode.BINARY, 0, 0.0) == []

    def test_detect_all_empty_sensors(self):
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0, vy=0, z=0.0, vz=0.0)
        assert detect_all([obj], [], [], DetectionMode.BINARY, 0, 0.0) == []


# ===========================================================================
# sim
# ===========================================================================


class TestDetectionGeometry:
    """Verify detection results against analytically-known geometric setups."""

    def test_object_transit_detection_count(self):
        # Object moves at 0.1 km/s along y=5, sensor at (5, 5) with range 1.0
        # Object in range when |x - 5| <= 1.0, i.e., x in [4.0, 6.0]
        # With dt=1.0 and vx=0.1, object takes 20 steps to cross 2.0km of coverage
        # Starting from x=0, enters range at step 40 (x=4.0), exits at step 60 (x=6.0)
        # Expected detections: steps 40 through 60 inclusive = 21
        sensor = Vec3(5.0, 5.0, 0.0)
        detections = 0
        for step in range(100):
            x = 0.0 + step * 0.1
            if x > 10.0:
                break
            obj = SimObject(obj_id="a", group_id="g", x=x, y=5.0, vx=0.1, vy=0.0, z=0.0, vz=0.0)
            recs = detect_all([obj], [sensor], [1.0], DetectionMode.BINARY, step, float(step))
            detections += len(recs)
        assert detections == 21

    def test_object_near_miss(self):
        # Object passes at y=5.0, sensor at (5.0, 6.5) with range 1.0
        # Closest approach is 1.5 km (at x=5.0) — never in range
        sensor = Vec3(5.0, 6.5, 0.0)
        detections = 0
        for step in range(100):
            x = step * 0.1
            if x > 10.0:
                break
            obj = SimObject(obj_id="a", group_id="g", x=x, y=5.0, vx=0.1, vy=0.0, z=0.0, vz=0.0)
            recs = detect_all([obj], [sensor], [1.0], DetectionMode.BINARY, step, float(step))
            detections += len(recs)
        assert detections == 0

    def test_multi_sensor_overlap(self):
        # Object at (5, 5), two sensors at (5.3, 5) and (4.7, 5), both range 1.0
        # Object is within range of BOTH → 2 detection records
        obj = SimObject(obj_id="a", group_id="g", x=5.0, y=5.0, vx=0, vy=0, z=0.0, vz=0.0)
        sensors = [Vec3(5.3, 5.0, 0.0), Vec3(4.7, 5.0, 0.0)]
        recs = detect_all([obj], sensors, [1.0, 1.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 2
        assert {r.sensor_idx for r in recs} == {0, 1}

    def test_detection_range_records_correct_distance(self):
        # Known geometry: object at (3, 4), sensor at (0, 0)
        # Distance = 5.0
        obj = SimObject(obj_id="a", group_id="g", x=3.0, y=4.0, vx=0, vy=0, z=0.0, vz=0.0)
        sensor = Vec3(0.0, 0.0, 0.0)
        recs = detect_all([obj], [sensor], [6.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 1
        assert recs[0].range_km == pytest.approx(5.0, abs=1e-10)

    def test_many_objects_all_detected_inside_circle(self):
        # Place 50 objects uniformly within 0.5km of sensor, range=1.0 → all detected
        rng = random.Random(99)
        sensor = Vec3(5.0, 5.0, 0.0)
        objs = []
        for i in range(50):
            angle = rng.uniform(0, 2 * math.pi)
            r = 0.5 * math.sqrt(rng.random())
            objs.append(
                SimObject(
                    obj_id=f"o{i}",
                    group_id="g",
                    x=5.0 + r * math.cos(angle),
                    y=5.0 + r * math.sin(angle),
                    vx=0,
                    vy=0,
                    z=0.0,
                    vz=0.0,
                )
            )
        recs = detect_all(objs, [sensor], [1.0], DetectionMode.BINARY, 0, 0.0)
        assert len(recs) == 50


class TestPdCurveCorrectness:
    """Verify Pd curves are monotonically decreasing and statistically consistent."""

    @pytest.mark.parametrize("function", ["exponential", "sigmoid"])
    def test_pd_monotonically_decreasing(self, function):
        max_range = 5.0
        half_range = 2.5
        prev_pd = 1.0
        for d in [i * 0.1 for i in range(1, 50)]:
            if d >= max_range:
                break
            pd = _compute_pd(d, max_range, function, half_range)
            assert pd <= prev_pd, f"Pd increased at d={d}: {prev_pd} → {pd}"
            prev_pd = pd

    @pytest.mark.parametrize("function", ["exponential", "sigmoid"])
    def test_pd_range_bounds(self, function):
        # Pd should be in [0, 1] for all distances
        for d in [0, 0.5, 1.0, 2.0, 3.0, 4.9, 5.0, 10.0]:
            pd = _compute_pd(d, max_range=5.0, function=function, half_range=2.5)
            assert 0.0 <= pd <= 1.0

    def test_exponential_statistical_detection_rate(self):
        # At distance where Pd = 0.5, ~50% of 10,000 trials should detect
        # Use a generous confidence interval for test stability
        rng = random.Random(123)
        distance = 2.0
        max_range = 5.0
        half_range = 2.0  # Pd = 0.5 exactly at distance=2.0
        expected_pd = _compute_pd(distance, max_range, "exponential", half_range)
        assert expected_pd == pytest.approx(0.5, abs=0.01)

        n_trials = 10000
        detections = sum(1 for _ in range(n_trials) if rng.random() < expected_pd)
        detection_rate = detections / n_trials
        assert detection_rate == pytest.approx(0.5, abs=0.03)  # within 3%

    def test_probabilistic_detect_all_statistical(self):
        # Run probabilistic detection many times at a known distance,
        # verify aggregate detection rate matches the Pd curve
        sensor = Vec3(0.0, 0.0, 0.0)
        obj = SimObject(obj_id="a", group_id="g", x=1.5, y=0.0, vx=0, vy=0, z=0.0, vz=0.0)
        max_range = 3.0
        half_range = 1.5
        expected_pd = _compute_pd(1.5, max_range, "exponential", half_range)

        n_trials = 5000
        detected_count = 0
        for i in range(n_trials):
            rng = random.Random(i)
            recs = detect_all(
                [obj],
                [sensor],
                [max_range],
                DetectionMode.PROBABILISTIC,
                timestep=0,
                time_sec=0.0,
                pd_function="exponential",
                pd_half_range=half_range,
                rng=rng,
            )
            detected_count += len(recs)

        actual_rate = detected_count / n_trials
        assert actual_rate == pytest.approx(expected_pd, abs=0.03)
