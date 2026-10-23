from __future__ import annotations

from pathlib import Path

import yaml

from .rf.config import parse_rf

from .models import (
    Area,
    CommStrategy,
    CommunicationConfig,
    DetectionMode,
    GroupMode,
    NetworkTopology,
    ObjectGroupConfig,
    ScenarioConfig,
    SensorConfig,
    SensorLayout,
    SpacingProfile,
    Vec2,
    Vec3,
)


def load_scenario(path: str | Path) -> ScenarioConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    area = Area(width_km=raw["area"]["width_km"], height_km=raw["area"]["height_km"])

    det = raw.get("detection", {})
    detection_mode = DetectionMode(det.get("mode", "binary"))
    detection_range = det.get("range")
    pd_function = det.get("pd_function", "exponential")
    pd_half_range = det.get("pd_half_range")

    groups = []
    for g in raw.get("groups", []):
        heading = Vec2(*g["heading"])
        speed_fixed = g.get("speed")
        groups.append(
            ObjectGroupConfig(
                group_id=g["group_id"],
                count=g["count"],
                mode=GroupMode(g["mode"]),
                heading=heading,
                speed_min=g.get("speed_min", speed_fixed or 0.05),
                speed_max=g.get("speed_max", speed_fixed or 0.05),
                spawn_center=Vec2(*g["spawn_center"]),
                spawn_radius=g["spawn_radius"],
                spacing_profile=SpacingProfile(g.get("spacing_profile", "uniform")),
                speed=speed_fixed,
                spawn_altitude=g["spawn_altitude"],
                spawn_altitude_range=g.get("spawn_altitude_range", 0.0),
                vertical_speed=g.get("vertical_speed", None),
            )
        )

    sensors = None
    if "sensors" in raw:
        s = raw["sensors"]
        layout = SensorLayout(s["layout"])
        positions = None
        if layout == SensorLayout.MANUAL and "positions" in s:
            # Support both 2D [x, y] and 3D [x, y, z] position arrays
            positions = []
            for p in s["positions"]:
                if len(p) == 2:
                    positions.append(Vec3(p[0], p[1], 0.0))  # Default z=0 for ground level
                elif len(p) == 3:
                    positions.append(Vec3(p[0], p[1], p[2]))
                else:
                    raise ValueError(f"Position must be [x, y] or [x, y, z], got {p}")
        sensors = SensorConfig(
            layout=layout,
            detection_range=s.get("detection_range", detection_range or 2.0),
            positions=positions,
            count=s.get("count"),
            spacing=s.get("spacing"),
            sensor_altitude=s.get("altitude", 0.0),
        )

    # Parse communication configuration
    communication = None
    if "communication" in raw:
        comm = raw["communication"]

        # Parse network topology
        topo_raw = comm.get("topology", {})
        topology = NetworkTopology(
            bundling_mode=topo_raw.get("bundling_mode", "none"),
            relay_assignments={
                int(k): int(v) for k, v in topo_raw.get("relay_assignments", {}).items()
            },
            bundle_window_sec=topo_raw.get("bundle_window_sec", 1.0),
            transport=topo_raw.get("transport", "zenoh"),
        )

        # Parse communication strategy
        strat_raw = comm.get("strategy", {})
        strategy = CommStrategy(
            strategy_name=strat_raw.get("name", "direct"),
            transport_protocol=strat_raw.get("protocol", "zenoh_pubsub"),
            bundling_enabled=strat_raw.get("bundling", False),
            bundle_window_sec=strat_raw.get("window_sec"),
            max_bundle_size=strat_raw.get("max_size"),
        )

        communication = CommunicationConfig(
            enabled=comm.get("enabled", True),
            topology=topology,
            strategy=strategy,
            sensor_metadata=comm.get("sensor_metadata", []),
            reference_origin=tuple(comm.get("reference_origin", [0.0, 0.0])),
        )

    # Parse top-level reference_origin (optional)
    reference_origin = None
    if "reference_origin" in raw:
        ref_origin_raw = raw["reference_origin"]
        if isinstance(ref_origin_raw, (list, tuple)) and len(ref_origin_raw) == 2:
            reference_origin = tuple(ref_origin_raw)
    # Fall back to communication.reference_origin if present
    elif communication and communication.reference_origin != (0.0, 0.0):
        reference_origin = communication.reference_origin

    return ScenarioConfig(
        name=raw.get("name", "Unnamed"),
        area=area,
        dt=raw.get("dt", 1.0),
        duration=raw["duration"],
        detection_mode=detection_mode,
        groups=groups,
        sensors=sensors,
        detection_range=detection_range,
        pd_function=pd_function,
        pd_half_range=pd_half_range,
        seed=raw.get("seed"),
        communication=communication,
        reference_origin=reference_origin,
        rf=parse_rf(raw, raw["duration"]),
    )
