from __future__ import annotations

import math
import random as stdlib_random

import numpy as np

from .models import DetectionMode, DetectionRecord, SimObject, TrackStateRecord, Vec3
from .comm.track_manager import TrackManager


def detect_all(
    objects: list[SimObject],
    sensors: list[Vec3],
    sensor_ranges: list[float],
    mode: DetectionMode,
    timestep: int,
    time_sec: float,
    pd_function: str = "exponential",
    pd_half_range: float | None = None,
    rng: stdlib_random.Random | None = None,
) -> list[DetectionRecord]:
    if not objects or not sensors:
        return []

    # 3D positions
    obj_pos = np.array([[o.x, o.y, o.z] for o in objects])
    sen_pos = np.array([[s.x, s.y, s.z] for s in sensors])
    ranges_arr = np.array(sensor_ranges)

    # Pairwise 3D distances: (N, M)
    diff = obj_pos[:, np.newaxis, :] - sen_pos[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # Now includes z component

    in_range = distances <= ranges_arr[np.newaxis, :]

    records: list[DetectionRecord] = []
    obj_indices, sen_indices = np.nonzero(in_range)

    for idx in range(len(obj_indices)):
        i = int(obj_indices[idx])
        j = int(sen_indices[idx])
        dist = float(distances[i, j])
        obj = objects[i]

        if mode == DetectionMode.BINARY:
            records.append(
                DetectionRecord(
                    timestep=timestep,
                    time_sec=time_sec,
                    obj_id=obj.obj_id,
                    sensor_idx=j,
                    sensor_x=sensors[j].x,
                    sensor_y=sensors[j].y,
                    sensor_z=sensors[j].z,
                    range_km=dist,
                    detected=True,
                    pd=1.0,
                )
            )
        else:
            pd = _compute_pd(dist, sensor_ranges[j], pd_function, pd_half_range)
            detected = (rng or stdlib_random).random() < pd
            if detected:
                records.append(
                    DetectionRecord(
                        timestep=timestep,
                        time_sec=time_sec,
                        obj_id=obj.obj_id,
                        sensor_idx=j,
                        sensor_x=sensors[j].x,
                        sensor_y=sensors[j].y,
                        sensor_z=sensors[j].z,
                        range_km=dist,
                        detected=True,
                        pd=pd,
                    )
                )

    return records


def _compute_pd(
    distance: float, max_range: float, function: str, half_range: float | None
) -> float:
    if distance <= 0:
        return 1.0
    if distance >= max_range:
        return 0.0

    hr = half_range if half_range is not None else max_range * 0.6

    if function == "exponential":
        k = math.log(2) / hr
        return math.exp(-k * distance)
    elif function == "sigmoid":
        k = 10.0 / hr
        return 1.0 / (1.0 + math.exp(k * (distance - hr)))
    else:
        raise ValueError(f"Unknown pd_function: {function}")


def detect_all_with_state(
    objects: list[SimObject],
    sensors: list[Vec3],
    sensor_ranges: list[float],
    mode: DetectionMode,
    timestep: int,
    time_sec: float,
    track_manager: TrackManager,
    pd_function: str = "exponential",
    pd_half_range: float | None = None,
    rng: stdlib_random.Random | None = None,
) -> tuple[list[DetectionRecord], list[TrackStateRecord]]:
    """Extended detection that captures full object state.

    Returns both DetectionRecord (for backward compatibility) and
    TrackStateRecord (for communication emulation).
    """
    if not objects or not sensors:
        return [], []

    # 3D positions
    obj_pos = np.array([[o.x, o.y, o.z] for o in objects])
    sen_pos = np.array([[s.x, s.y, s.z] for s in sensors])
    ranges_arr = np.array(sensor_ranges)

    # Pairwise 3D distances: (N, M)
    diff = obj_pos[:, np.newaxis, :] - sen_pos[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))  # Now includes z component

    in_range = distances <= ranges_arr[np.newaxis, :]

    detection_records: list[DetectionRecord] = []
    track_state_records: list[TrackStateRecord] = []
    obj_indices, sen_indices = np.nonzero(in_range)

    for idx in range(len(obj_indices)):
        i = int(obj_indices[idx])
        j = int(sen_indices[idx])
        dist = float(distances[i, j])
        obj = objects[i]

        # Compute Pd
        if mode == DetectionMode.BINARY:
            pd = 1.0
            detected = True
        else:
            pd = _compute_pd(dist, sensor_ranges[j], pd_function, pd_half_range)
            detected = (rng or stdlib_random).random() < pd

        if detected:
            # Create DetectionRecord (backward compatible)
            detection_records.append(
                DetectionRecord(
                    timestep=timestep,
                    time_sec=time_sec,
                    obj_id=obj.obj_id,
                    sensor_idx=j,
                    sensor_x=sensors[j].x,
                    sensor_y=sensors[j].y,
                    sensor_z=sensors[j].z,
                    range_km=dist,
                    detected=True,
                    pd=pd,
                )
            )

            # Create TrackStateRecord with full object state
            trk_id = track_manager.get_or_create_track_id(obj.obj_id)
            track_manager.update_detection(obj.obj_id, timestep)
            trk_stat = track_manager.get_track_status(obj.obj_id, timestep)
            trk_num = track_manager.generate_track_num(obj.obj_id)

            # Compute heading and speed (including vertical component)
            speed_kmps = math.sqrt(obj.vx**2 + obj.vy**2 + obj.vz**2)
            heading_deg = (90 - math.degrees(math.atan2(obj.vy, obj.vx))) % 360

            track_state_records.append(
                TrackStateRecord(
                    timestep=timestep,
                    time_sec=time_sec,
                    obj_id=obj.obj_id,
                    sensor_idx=j,
                    trk_id=trk_id,
                    trk_num=trk_num,
                    trk_stat=trk_stat,
                    obj_x=obj.x,
                    obj_y=obj.y,
                    obj_z=obj.z,
                    obj_vx=obj.vx,
                    obj_vy=obj.vy,
                    obj_vz=obj.vz,
                    speed_kmps=speed_kmps,
                    heading_deg=heading_deg,
                    range_km=dist,
                    pd=pd,
                )
            )

    return detection_records, track_state_records
