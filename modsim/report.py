from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING

from .geo import km_to_latlon
from .models import DetectionRecord, SimState

if TYPE_CHECKING:
    from .models import ScenarioConfig


def print_summary(state: SimState) -> None:
    total_detections = len(state.detection_log)
    total_steps = len(state.timestep_stats)

    print(f"\n{'=' * 50}")
    print("  Simulation Complete")
    print(f"  Timesteps:        {total_steps}")
    print(f"  Total Detections: {total_detections}")
    print(f"  Sensors:          {len(state.sensors)}")

    if state.timestep_stats:
        avg_coverage = sum(s.coverage_pct for s in state.timestep_stats) / len(state.timestep_stats)
        print(f"  Avg Coverage:     {avg_coverage:.1f}%")

    obj_tracks: dict[str, list[DetectionRecord]] = {}
    for d in state.detection_log:
        obj_tracks.setdefault(d.obj_id, []).append(d)

    print(f"\n  Object Tracks ({len(obj_tracks)} objects detected):")
    for obj_id, dets in sorted(obj_tracks.items()):
        sensors_seen = {d.sensor_idx for d in dets}
        first_t = min(d.time_sec for d in dets)
        last_t = max(d.time_sec for d in dets)
        print(
            f"    {obj_id}: {len(dets)} detections, "
            f"{len(sensors_seen)} sensors, t=[{first_t:.0f}s..{last_t:.0f}s]"
        )

    print(f"{'=' * 50}\n")


def export_json(
    state: SimState,
    path: str,
    export_mode: str = "default",
    config: "ScenarioConfig | None" = None,
) -> str:
    """Export simulation results.

    Args:
        state: Simulation state
        path: Output file path
        export_mode:
            - "default": Original format (backward compatible)
            - "comms": Communication timeline only (Locust format)
            - "states": Object states at every timestep
            - "both": Include both formats
        config: Optional scenario configuration (required for comms mode)

    Returns:
        The actual path written (may differ from input for JSONL auto-detection).
    """
    if export_mode == "default":
        # Original format (now with altitude)
        # Include geographic coordinates if reference_origin is provided
        sensor_positions = []
        for s in state.sensors:
            pos = {"x": s.x, "y": s.y, "z": s.z}
            if config and config.reference_origin:
                lat, lon = km_to_latlon(s.x, s.y, config.reference_origin)
                pos["lat"] = lat
                pos["lon"] = lon
            sensor_positions.append(pos)

        data = {
            "timestep_stats": [asdict(s) for s in state.timestep_stats],
            "detection_log": [asdict(d) for d in state.detection_log],
            "sensor_positions": sensor_positions,
        }

        if config and config.reference_origin:
            data["reference_origin"] = {
                "lat": config.reference_origin[0],
                "lon": config.reference_origin[1],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    elif export_mode == "comms":
        # Generate and export Locust timeline
        from .comm.export import (
            export_locust_format,
            export_locust_jsonl,
            generate_message_timeline,
        )
        from .models import CommStrategy, NetworkTopology

        if not state.track_state_log:
            raise ValueError("No track state data available. Enable communication in YAML.")

        # Get topology/strategy from config, or use defaults
        if config and config.communication:
            topology = config.communication.topology
            strategy = config.communication.strategy
            # Use top-level reference_origin if available, otherwise use communication.reference_origin
            reference_origin = config.reference_origin or config.communication.reference_origin
            scenario_name = config.name
        else:
            # Fallback to defaults if config not provided
            topology = NetworkTopology()
            strategy = CommStrategy(
                strategy_name="direct",
                transport_protocol="zenoh_pubsub",
                bundling_enabled=False,
            )
            reference_origin = config.reference_origin if config else (0.0, 0.0)
            scenario_name = config.name if config else "Unnamed Scenario"

        # Use streaming JSONL for large scenarios to avoid OOM
        if len(state.track_state_log) > 100_000:
            # Switch extension to .jsonl if path ends with .json
            if path.endswith(".json"):
                path = path[:-5] + ".jsonl"
            export_locust_jsonl(
                state.track_state_log,
                state.sensor_metadata,
                path,
                topology=topology,
                strategy=strategy,
                reference_origin=reference_origin,
                scenario_name=scenario_name,
                num_sensors=len(state.sensors),
            )
        else:
            events = generate_message_timeline(
                state.track_state_log,
                state.sensor_metadata,
                topology,
                strategy,
                reference_origin,
            )
            export_locust_format(
                events,
                path,
                scenario_name=scenario_name,
                topology=topology,
                strategy=strategy,
                reference_origin=reference_origin,
                num_sensors=len(state.sensors),
            )

    elif export_mode == "states":
        # Export object states at every timestep
        # Include geographic coordinates if reference_origin is provided
        timesteps = []
        for snapshot in state.object_snapshots:
            objects = []
            for obj in snapshot.objects:
                obj_dict = asdict(obj)
                if config and config.reference_origin:
                    lat, lon = km_to_latlon(obj.x, obj.y, config.reference_origin)
                    obj_dict["lat"] = lat
                    obj_dict["lon"] = lon
                objects.append(obj_dict)

            timesteps.append(
                {"timestep": snapshot.timestep, "time_sec": snapshot.time_sec, "objects": objects}
            )

        sensor_positions = []
        for s in state.sensors:
            pos = {"x": s.x, "y": s.y, "z": s.z}
            if config and config.reference_origin:
                lat, lon = km_to_latlon(s.x, s.y, config.reference_origin)
                pos["lat"] = lat
                pos["lon"] = lon
            sensor_positions.append(pos)

        data = {
            "timesteps": timesteps,
            "sensor_positions": sensor_positions,
        }

        if config and config.reference_origin:
            data["reference_origin"] = {
                "lat": config.reference_origin[0],
                "lon": config.reference_origin[1],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    else:  # "both"
        sensor_positions = []
        for s in state.sensors:
            pos = {"x": s.x, "y": s.y, "z": s.z}
            if config and config.reference_origin:
                lat, lon = km_to_latlon(s.x, s.y, config.reference_origin)
                pos["lat"] = lat
                pos["lon"] = lon
            sensor_positions.append(pos)

        data = {
            "timestep_stats": [asdict(s) for s in state.timestep_stats],
            "detection_log": [asdict(d) for d in state.detection_log],
            "sensor_positions": sensor_positions,
            "track_states": [asdict(ts) for ts in state.track_state_log],
            "sensor_metadata": [asdict(sm) for sm in state.sensor_metadata],
        }

        if config and config.reference_origin:
            data["reference_origin"] = {
                "lat": config.reference_origin[0],
                "lon": config.reference_origin[1],
            }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    return path
