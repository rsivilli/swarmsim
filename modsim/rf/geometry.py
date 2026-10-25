"""Trajectory interpolation and sensor-relative geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..models import TimestepSnapshot


@dataclass
class Trajectory:
    """Piecewise constant-velocity track of one object, sampled at the simulation timesteps.

    Motion in modsim is constant-velocity within each ``dt``, so linear extrapolation from the
    latest snapshot is exact.
    """

    group_id: str
    t: np.ndarray  # (K,) snapshot times
    pos_km: np.ndarray  # (K, 3)
    vel_kmps: np.ndarray  # (K, 3)
    t_end: float  # time the object leaves the scenario (last snapshot + dt)

    @property
    def t_start(self) -> float:
        return float(self.t[0])

    def state_at(self, times) -> tuple[np.ndarray, np.ndarray]:
        """Positions (km) and velocities (km/s) at the given times, shape (N, 3) each."""
        times = np.asarray(times, dtype=np.float64)
        k = np.clip(np.searchsorted(self.t, times, side="right") - 1, 0, len(self.t) - 1)
        dt = (times - self.t[k])[:, None]
        return self.pos_km[k] + self.vel_kmps[k] * dt, self.vel_kmps[k]


def build_trajectories(snapshots: list[TimestepSnapshot], dt: float) -> dict[str, Trajectory]:
    """Collect per-object trajectories from the simulation's object snapshots."""
    rows: dict[str, list[tuple]] = {}
    groups: dict[str, str] = {}
    for snap in snapshots:
        for o in snap.objects:
            groups[o.obj_id] = o.group_id
            rows.setdefault(o.obj_id, []).append((snap.time_sec, o.x, o.y, o.z, o.vx, o.vy, o.vz))
    trajectories = {}
    for obj_id, r in rows.items():
        a = np.asarray(r, dtype=np.float64)
        trajectories[obj_id] = Trajectory(
            group_id=groups[obj_id],
            t=a[:, 0],
            pos_km=a[:, 1:4],
            vel_kmps=a[:, 4:7],
            t_end=float(a[-1, 0] + dt),
        )
    return trajectories


def relative_geometry(
    emitter_pos_km: np.ndarray, emitter_vel_kmps: np.ndarray, sensor_pos_km: np.ndarray
) -> dict[str, np.ndarray]:
    """Range, ground range, radial velocity, azimuth and elevation of emitters seen from a sensor.

    Azimuth is degrees clockwise from North (+y), matching modsim's heading convention.
    Radial velocity is positive when the emitter moves away from the sensor.
    """
    # 1-D component arrays: much cheaper than (N, 3) temporaries for millions of rows
    dx = (emitter_pos_km[:, 0] - sensor_pos_km[0]) * 1000.0
    dy = (emitter_pos_km[:, 1] - sensor_pos_km[1]) * 1000.0
    dz = (emitter_pos_km[:, 2] - sensor_pos_km[2]) * 1000.0
    ground_m = np.hypot(dx, dy)
    range_m = np.hypot(ground_m, dz)
    safe = np.maximum(range_m, 1e-9)
    radial = (
        dx * emitter_vel_kmps[:, 0] + dy * emitter_vel_kmps[:, 1] + dz * emitter_vel_kmps[:, 2]
    ) * (1000.0 / safe)
    azimuth = np.degrees(np.arctan2(dx, dy)) % 360.0
    elevation = np.degrees(np.arctan2(dz, np.maximum(ground_m, 1e-9)))
    return {
        "range_m": range_m,
        "ground_range_m": ground_m,
        "radial_velocity_mps": radial,
        "azimuth_deg": azimuth,
        "elevation_deg": elevation,
    }
