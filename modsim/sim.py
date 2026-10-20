from __future__ import annotations

import math
import random as stdlib_random
from collections.abc import Generator

from .models import (
    DetectionRecord,
    GroupMode,
    ScenarioConfig,
    SensorLayout,
    SensorMetadata,
    SimObject,
    SimState,
    TimestepSnapshot,
    TimestepStats,
    TrackStateRecord,
    Vec2,
    Vec3,
)
from .motion import spawn_group, step_formation, step_objects
from .detection import detect_all, detect_all_with_state
from .comm.track_manager import TrackManager


def build_sensors(cfg: ScenarioConfig) -> tuple[list[Vec3], list[float]]:
    sc = cfg.sensors
    if sc is None:
        return [], []

    det_range = sc.detection_range or cfg.detection_range or 2.0

    if sc.layout == SensorLayout.MANUAL:
        positions = sc.positions or []
        return positions, [det_range] * len(positions)

    # For automatic layouts, use sensor_altitude
    sensor_z = sc.sensor_altitude

    if sc.layout == SensorLayout.GRID:
        spacing = sc.spacing or 2.0
        positions: list[Vec3] = []
        x = spacing / 2
        while x < cfg.area.width_km:
            y = spacing / 2
            while y < cfg.area.height_km:
                positions.append(Vec3(x, y, sensor_z))
                y += spacing
            x += spacing
        return positions, [det_range] * len(positions)

    if sc.layout == SensorLayout.HEX:
        spacing = sc.spacing or 2.0
        positions = []
        row = 0
        y = spacing / 2
        while y < cfg.area.height_km:
            x_offset = (spacing / 2) if row % 2 else 0.0
            x = spacing / 2 + x_offset
            while x < cfg.area.width_km:
                positions.append(Vec3(x, y, sensor_z))
                x += spacing
            y += spacing * math.sqrt(3) / 2
            row += 1
        return positions, [det_range] * len(positions)

    if sc.layout == SensorLayout.RANDOM:
        rng = stdlib_random.Random(cfg.seed)
        count = sc.count or 10
        positions = [
            Vec3(
                rng.uniform(0, cfg.area.width_km),
                rng.uniform(0, cfg.area.height_km),
                sensor_z,
            )
            for _ in range(count)
        ]
        return positions, [det_range] * len(positions)

    raise ValueError(f"Unknown sensor layout: {sc.layout}")


FrameData = tuple[
    int, float, list[SimObject], list[DetectionRecord], list[TrackStateRecord], TimestepStats
]


def run_iter(cfg: ScenarioConfig) -> Generator[FrameData, None, None]:
    rng = stdlib_random.Random(cfg.seed)

    sensor_positions, sensor_ranges = build_sensors(cfg)

    all_objects: list[SimObject] = []
    group_modes: dict[str, GroupMode] = {}
    for group_cfg in cfg.groups:
        group_modes[group_cfg.group_id] = group_cfg.mode
        all_objects.extend(spawn_group(group_cfg, cfg.area, rng))

    num_steps = int(cfg.duration / cfg.dt)

    # Create TrackManager if communication is enabled
    track_manager = TrackManager() if (cfg.communication and cfg.communication.enabled) else None

    for step in range(num_steps):
        time_sec = step * cfg.dt

        # Use new detection function if communication enabled
        if track_manager:
            detections, track_states = detect_all_with_state(
                objects=all_objects,
                sensors=sensor_positions,
                sensor_ranges=sensor_ranges,
                mode=cfg.detection_mode,
                timestep=step,
                time_sec=time_sec,
                track_manager=track_manager,
                pd_function=cfg.pd_function,
                pd_half_range=cfg.pd_half_range,
                rng=rng,
            )
        else:
            detections = detect_all(
                objects=all_objects,
                sensors=sensor_positions,
                sensor_ranges=sensor_ranges,
                mode=cfg.detection_mode,
                timestep=step,
                time_sec=time_sec,
                pd_function=cfg.pd_function,
                pd_half_range=cfg.pd_half_range,
                rng=rng,
            )
            track_states = []

        detected_ids = {d.obj_id for d in detections}
        per_sensor = [0] * len(sensor_positions)
        for d in detections:
            per_sensor[d.sensor_idx] += 1

        total_obj = len(all_objects)
        stats = TimestepStats(
            timestep=step,
            time_sec=time_sec,
            total_objects=total_obj,
            objects_detected=len(detected_ids),
            coverage_pct=(len(detected_ids) / total_obj * 100) if total_obj > 0 else 0.0,
            detections_per_sensor=per_sensor,
        )

        # Snapshot positions before stepping (for visualization)
        snapshot = [
            SimObject(
                obj_id=o.obj_id,
                group_id=o.group_id,
                x=o.x,
                y=o.y,
                z=o.z,
                vx=o.vx,
                vy=o.vy,
                vz=o.vz,
            )
            for o in all_objects
        ]

        yield step, time_sec, snapshot, detections, track_states, stats

        # Step motion
        by_group: dict[str, list[SimObject]] = {}
        for obj in all_objects:
            by_group.setdefault(obj.group_id, []).append(obj)

        next_objects: list[SimObject] = []
        for gid, group_objs in by_group.items():
            if group_modes.get(gid) == GroupMode.FORMATION:
                next_objects.extend(step_formation(group_objs, cfg.dt, cfg.area))
            else:
                next_objects.extend(step_objects(group_objs, cfg.dt, cfg.area))

        all_objects = next_objects
        if not all_objects:
            break


def _build_sensor_metadata(cfg: ScenarioConfig, sensors: list[Vec2]) -> list[SensorMetadata]:
    """Generate sensor metadata from config or defaults."""
    # If YAML provides metadata, use it
    if cfg.communication and cfg.communication.sensor_metadata:
        return [SensorMetadata(**meta) for meta in cfg.communication.sensor_metadata]

    # Otherwise generate defaults
    return [
        SensorMetadata(
            sensor_idx=i,
            sensor_id=f"SENSOR-{i:03d}",
            asset=f"PLATFORM-{i:03d}",
            asset_nat="US",
            sen_type="MASINT",
            network_tier=0,
            relay_parent=None,
        )
        for i in range(len(sensors))
    ]


def run(cfg: ScenarioConfig) -> SimState:
    sensor_positions, sensor_ranges = build_sensors(cfg)

    detection_log: list[DetectionRecord] = []
    track_state_log: list[TrackStateRecord] = []
    timestep_stats: list[TimestepStats] = []
    object_snapshots: list[TimestepSnapshot] = []
    last_objects: list[SimObject] = []

    for step, time_sec, objects, detections, track_states, stats in run_iter(cfg):
        detection_log.extend(detections)
        track_state_log.extend(track_states)
        timestep_stats.append(stats)
        object_snapshots.append(TimestepSnapshot(timestep=step, time_sec=time_sec, objects=objects))
        last_objects = objects

    # Generate sensor metadata if communication enabled
    sensor_metadata = []
    if cfg.communication and cfg.communication.enabled:
        sensor_metadata = _build_sensor_metadata(cfg, sensor_positions)

    return SimState(
        objects=last_objects,
        sensors=sensor_positions,
        sensor_range=sensor_ranges,
        timestep=timestep_stats[-1].timestep if timestep_stats else 0,
        time_sec=timestep_stats[-1].time_sec if timestep_stats else 0.0,
        detection_log=detection_log,
        timestep_stats=timestep_stats,
        track_state_log=track_state_log,
        sensor_metadata=sensor_metadata,
        object_snapshots=object_snapshots,
    )
