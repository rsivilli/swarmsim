from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from ..models import CommStrategy, NetworkTopology, SensorMetadata, TrackStateRecord
from .track_messages import generate_track_message


@dataclass
class MessageEvent:
    """Single message transmission event for Locust replay."""

    timestamp_sec: float  # Simulation time when message is sent
    delay_sec: float  # Network delay added
    sender_id: str  # Sensor or relay ID
    receiver_id: str  # Relay, backend, or endpoint
    transport: str  # "zenoh_pub", "rest_post", etc.
    message: dict  # Track.json payload
    bundle_size: int = 1  # Number of messages in bundle


def generate_message_timeline(
    track_states: list[TrackStateRecord],
    sensor_metadata: list[SensorMetadata],
    topology: NetworkTopology,
    strategy: CommStrategy,
    reference_origin: tuple[float, float] = (0.0, 0.0),
) -> list[MessageEvent]:
    """Generate chronological message timeline with network effects.

    Applies bundling, relay hierarchy, and network delays based on strategy.

    Args:
        track_states: All track detections from simulation
        sensor_metadata: Sensor metadata list
        topology: Network topology configuration
        strategy: Communication strategy
        reference_origin: (lat, lon) for coordinate transforms

    Returns:
        Sorted list of message events ready for Locust replay
    """
    if strategy.bundling_enabled and strategy.bundle_window_sec:
        events = _generate_bundled_messages(
            track_states, sensor_metadata, topology, strategy, reference_origin
        )
    else:
        events = _generate_direct_messages(
            track_states, sensor_metadata, topology, strategy, reference_origin
        )

    # Sort by effective transmission time
    events.sort(key=lambda e: e.timestamp_sec + e.delay_sec)

    return events


def _generate_direct_messages(
    track_states: list[TrackStateRecord],
    sensor_metadata: list[SensorMetadata],
    topology: NetworkTopology,
    strategy: CommStrategy,
    reference_origin: tuple[float, float],
) -> list[MessageEvent]:
    """Generate one message per detection, sent immediately."""
    events = []
    sensor_meta_map = {s.sensor_idx: s for s in sensor_metadata}

    for ts in track_states:
        sensor_meta = sensor_meta_map.get(ts.sensor_idx)
        if not sensor_meta:
            continue

        # Generate Track.json message
        msg = generate_track_message(ts, sensor_meta, reference_origin)

        # Determine receiver based on topology
        if sensor_meta.relay_parent is not None and sensor_meta.relay_parent in sensor_meta_map:
            receiver_meta = sensor_meta_map[sensor_meta.relay_parent]
            receiver_id = receiver_meta.sensor_id
        else:
            receiver_id = "backend"

        # Simple network delay model: base delay + tier penalty
        delay = 0.01 + (0.04 * sensor_meta.network_tier)

        events.append(
            MessageEvent(
                timestamp_sec=ts.time_sec,
                delay_sec=delay,
                sender_id=sensor_meta.sensor_id,
                receiver_id=receiver_id,
                transport=strategy.transport_protocol,
                message=msg,
                bundle_size=1,
            )
        )

    return events


def _generate_bundled_messages(
    track_states: list[TrackStateRecord],
    sensor_metadata: list[SensorMetadata],
    topology: NetworkTopology,
    strategy: CommStrategy,
    reference_origin: tuple[float, float],
) -> list[MessageEvent]:
    """Bundle messages by sensor and time window."""
    events = []
    sensor_meta_map = {s.sensor_idx: s for s in sensor_metadata}

    # Group by (sensor_idx, time_window_index)
    window_sec = strategy.bundle_window_sec or 1.0
    bundles: dict[tuple[int, int], list[TrackStateRecord]] = defaultdict(list)

    for ts in track_states:
        window_idx = int(ts.time_sec / window_sec)
        bundles[(ts.sensor_idx, window_idx)].append(ts)

    # Create bundled message events
    for (sensor_idx, window_idx), bundle in bundles.items():
        sensor_meta = sensor_meta_map.get(sensor_idx)
        if not sensor_meta:
            continue

        # Generate Track.json messages for all detections in bundle
        messages = [generate_track_message(ts, sensor_meta, reference_origin) for ts in bundle]

        # Limit bundle size if configured
        if strategy.max_bundle_size and len(messages) > strategy.max_bundle_size:
            # Split into multiple bundles
            for i in range(0, len(messages), strategy.max_bundle_size):
                chunk = messages[i : i + strategy.max_bundle_size]
                _create_bundle_event(
                    events,
                    sensor_meta,
                    sensor_meta_map,
                    window_idx,
                    window_sec,
                    chunk,
                    strategy,
                )
        else:
            _create_bundle_event(
                events, sensor_meta, sensor_meta_map, window_idx, window_sec, messages, strategy
            )

    return events


def _create_bundle_event(
    events: list[MessageEvent],
    sensor_meta: SensorMetadata,
    sensor_meta_map: dict[int, SensorMetadata],
    window_idx: int,
    window_sec: float,
    messages: list[dict],
    strategy: CommStrategy,
) -> None:
    """Helper to create a single bundle event."""
    # Determine receiver
    if sensor_meta.relay_parent is not None and sensor_meta.relay_parent in sensor_meta_map:
        receiver_meta = sensor_meta_map[sensor_meta.relay_parent]
        receiver_id = receiver_meta.sensor_id
    else:
        receiver_id = "backend"

    # Bundle sent at end of time window
    bundle_time = (window_idx + 1) * window_sec
    delay = 0.01 + (0.04 * sensor_meta.network_tier)

    events.append(
        MessageEvent(
            timestamp_sec=bundle_time,
            delay_sec=delay,
            sender_id=sensor_meta.sensor_id,
            receiver_id=receiver_id,
            transport=strategy.transport_protocol,
            message={"tracks": messages},  # Bundled format
            bundle_size=len(messages),
        )
    )


def _calculate_deployment_from_stats(
    total_messages: int,
    duration_sec: float,
    unique_senders: int,
    peak_messages_per_sec: float,
    num_sensors: int,
    scenario_name: str,
    backend_host: str = "http://backend-service:8080",
) -> dict:
    """Calculate deployment recommendations from pre-computed statistics.

    Same logic as _calculate_deployment_recommendations but accepts scalars
    instead of requiring the full event list in memory.
    """
    if duration_sec <= 0:
        return {
            "backend_host": backend_host,
            "users": 10,
            "spawn_rate": 5,
            "run_time": "60s",
            "worker_replicas": 1,
        }

    avg_messages_per_sec = total_messages / duration_sec

    recommended_users = min(max(unique_senders * 3, 10), 500)

    ramp_up_time = max(duration_sec * 0.1, 5.0)
    recommended_spawn_rate = max(min(int(recommended_users / ramp_up_time), 50), 5)

    if avg_messages_per_sec < 50:
        worker_replicas = 1
    elif avg_messages_per_sec < 200:
        worker_replicas = 3
    elif avg_messages_per_sec < 500:
        worker_replicas = 5
    else:
        worker_replicas = min(int(avg_messages_per_sec / 100), 10)

    if duration_sec < 120:
        run_time = f"{int(duration_sec)}s"
    else:
        run_time = f"{int(duration_sec / 60)}m"

    return {
        "backend_host": backend_host,
        "users": recommended_users,
        "spawn_rate": recommended_spawn_rate,
        "run_time": run_time,
        "worker_replicas": worker_replicas,
        "statistics": {
            "total_messages": total_messages,
            "avg_messages_per_sec": round(avg_messages_per_sec, 2),
            "peak_messages_per_sec": peak_messages_per_sec,
            "unique_senders": unique_senders,
            "num_sensors": num_sensors,
        },
    }


def _calculate_deployment_recommendations(
    events: list[MessageEvent],
    num_sensors: int,
    scenario_name: str,
    backend_host: str = "http://backend-service:8080",
) -> dict:
    """Calculate recommended Locust deployment parameters based on simulation."""
    if not events:
        return _calculate_deployment_from_stats(
            total_messages=0,
            duration_sec=0,
            unique_senders=0,
            peak_messages_per_sec=0,
            num_sensors=num_sensors,
            scenario_name=scenario_name,
            backend_host=backend_host,
        )

    total_messages = sum(e.bundle_size for e in events)
    duration_sec = max((e.timestamp_sec + e.delay_sec for e in events), default=60.0)
    unique_senders = len({e.sender_id for e in events})
    peak = _calculate_peak_message_rate(events)

    return _calculate_deployment_from_stats(
        total_messages=total_messages,
        duration_sec=duration_sec,
        unique_senders=unique_senders,
        peak_messages_per_sec=peak,
        num_sensors=num_sensors,
        scenario_name=scenario_name,
        backend_host=backend_host,
    )


def _calculate_peak_message_rate(events: list[MessageEvent]) -> float:
    """Calculate peak message rate in 1-second windows."""
    if not events:
        return 0.0

    # Bucket messages by second
    message_counts: dict[int, int] = defaultdict(int)
    for e in events:
        time_bucket = int(e.timestamp_sec + e.delay_sec)
        message_counts[time_bucket] += e.bundle_size

    return max(message_counts.values()) if message_counts else 0.0


def export_locust_format(
    events: list[MessageEvent],
    output_path: str,
    scenario_name: str = "Unnamed Scenario",
    topology: NetworkTopology | None = None,
    strategy: CommStrategy | None = None,
    reference_origin: tuple[float, float] = (0.0, 0.0),
    num_sensors: int = 0,
    backend_host: str = "http://backend-service:8080",
) -> None:
    """Export message timeline in Locust-ready JSON format with deployment config.

    Args:
        events: List of message events (should be sorted)
        output_path: Path to write JSON file
        scenario_name: Name of the simulation scenario
        topology: Network topology configuration
        strategy: Communication strategy configuration
        reference_origin: (lat, lon) reference for coordinate transforms
        num_sensors: Total number of sensors in simulation
        backend_host: Target backend endpoint for deployment

    Output format:
        {
          "metadata": {
            "scenario": "...",
            "total_events": N,
            "duration_sec": T,
            "total_messages": M
          },
          "config": {
            "communication": {...},
            "reference_origin": [lat, lon],
            "recommended_deployment": {...}
          },
          "events": [...]
        }
    """
    # Calculate metadata
    total_messages = sum(e.bundle_size for e in events)
    duration_sec = max((e.timestamp_sec + e.delay_sec for e in events), default=0.0)

    # Build communication config section
    comm_config = {}
    if strategy:
        comm_config["strategy"] = {
            "name": strategy.strategy_name,
            "protocol": strategy.transport_protocol,
            "bundling_enabled": strategy.bundling_enabled,
        }
        if strategy.bundle_window_sec is not None:
            comm_config["strategy"]["bundle_window_sec"] = strategy.bundle_window_sec
        if strategy.max_bundle_size is not None:
            comm_config["strategy"]["max_bundle_size"] = strategy.max_bundle_size

    if topology:
        comm_config["topology"] = {
            "bundling_mode": topology.bundling_mode,
            "transport": topology.transport,
            "bundle_window_sec": topology.bundle_window_sec,
        }
        if topology.relay_assignments:
            comm_config["topology"]["relay_assignments"] = topology.relay_assignments

    # Calculate deployment recommendations
    deployment_config = _calculate_deployment_recommendations(
        events, num_sensors, scenario_name, backend_host
    )

    locust_data = {
        "metadata": {
            "scenario": scenario_name,
            "total_events": len(events),
            "duration_sec": duration_sec,
            "total_messages": total_messages,
        },
        "config": {
            "communication": comm_config,
            "reference_origin": list(reference_origin),
            "recommended_deployment": deployment_config,
        },
        "events": [
            {
                "time": e.timestamp_sec + e.delay_sec,
                "sender": e.sender_id,
                "receiver": e.receiver_id,
                "transport": e.transport,
                "payload": e.message,
                "bundle_size": e.bundle_size,
            }
            for e in events
        ],
    }

    with open(output_path, "w") as f:
        json.dump(locust_data, f, indent=2)


def export_locust_jsonl(
    track_states: list[TrackStateRecord],
    sensor_metadata: list[SensorMetadata],
    output_path: str,
    topology: NetworkTopology,
    strategy: CommStrategy,
    reference_origin: tuple[float, float] = (0.0, 0.0),
    scenario_name: str = "Unnamed Scenario",
    num_sensors: int = 0,
    backend_host: str = "http://backend-service:8080",
) -> None:
    """Export communication timeline in streaming JSONL format.

    Format:
        Line 1: Header JSON with metadata, config, deployment recs, sender list
        Lines 2+: One event per line (compact JSON, no indent)

    Memory is O(num_sensors) for direct mode. Track.json payloads are generated
    lazily and GC'd after each line write.

    For bundled mode, track_state references are grouped by (sensor_idx, window_idx)
    but payloads are generated one bundle at a time.
    """
    sensor_meta_map = {s.sensor_idx: s for s in sensor_metadata}
    bundling = strategy.bundling_enabled and strategy.bundle_window_sec
    window_sec = strategy.bundle_window_sec or 1.0

    # --- Stats pass: collect scalars only ---
    sender_ids: set[str] = set()
    rate_buckets: dict[int, int] = defaultdict(int)
    total_events = 0
    total_messages = 0
    max_time = 0.0

    if bundling:
        # Group by (sensor_idx, window_idx) — stores references, not payloads
        bundles: dict[tuple[int, int], list[TrackStateRecord]] = defaultdict(list)
        for ts in track_states:
            if ts.sensor_idx not in sensor_meta_map:
                continue
            window_idx = int(ts.time_sec / window_sec)
            bundles[(ts.sensor_idx, window_idx)].append(ts)

        for (sensor_idx, window_idx), bundle in bundles.items():
            sensor_meta = sensor_meta_map[sensor_idx]
            sender_ids.add(sensor_meta.sensor_id)
            delay = 0.01 + (0.04 * sensor_meta.network_tier)
            effective_time = (window_idx + 1) * window_sec + delay
            max_time = max(max_time, effective_time)

            max_bs = strategy.max_bundle_size
            if max_bs and len(bundle) > max_bs:
                n_chunks = (len(bundle) + max_bs - 1) // max_bs
                for _ in range(n_chunks):
                    chunk_size = min(max_bs, len(bundle) - _ * max_bs)
                    total_events += 1
                    total_messages += chunk_size
                    rate_buckets[int(effective_time)] += chunk_size
            else:
                total_events += 1
                total_messages += len(bundle)
                rate_buckets[int(effective_time)] += len(bundle)
    else:
        for ts in track_states:
            sensor_meta = sensor_meta_map.get(ts.sensor_idx)
            if not sensor_meta:
                continue
            sender_ids.add(sensor_meta.sensor_id)
            delay = 0.01 + (0.04 * sensor_meta.network_tier)
            effective_time = ts.time_sec + delay
            max_time = max(max_time, effective_time)
            total_events += 1
            total_messages += 1
            rate_buckets[int(effective_time)] += 1

    peak_rate = max(rate_buckets.values()) if rate_buckets else 0.0
    sorted_senders = sorted(sender_ids)

    # Build comm config (same structure as JSON format)
    comm_config: dict = {}
    if strategy:
        comm_config["strategy"] = {
            "name": strategy.strategy_name,
            "protocol": strategy.transport_protocol,
            "bundling_enabled": strategy.bundling_enabled,
        }
        if strategy.bundle_window_sec is not None:
            comm_config["strategy"]["bundle_window_sec"] = strategy.bundle_window_sec
        if strategy.max_bundle_size is not None:
            comm_config["strategy"]["max_bundle_size"] = strategy.max_bundle_size

    if topology:
        comm_config["topology"] = {
            "bundling_mode": topology.bundling_mode,
            "transport": topology.transport,
            "bundle_window_sec": topology.bundle_window_sec,
        }
        if topology.relay_assignments:
            comm_config["topology"]["relay_assignments"] = topology.relay_assignments

    deployment_config = _calculate_deployment_from_stats(
        total_messages=total_messages,
        duration_sec=max_time,
        unique_senders=len(sender_ids),
        peak_messages_per_sec=peak_rate,
        num_sensors=num_sensors,
        scenario_name=scenario_name,
        backend_host=backend_host,
    )

    # --- Write pass: stream events lazily ---
    with open(output_path, "w") as f:
        # Line 1: header
        header = {
            "metadata": {
                "scenario": scenario_name,
                "total_events": total_events,
                "duration_sec": max_time,
                "total_messages": total_messages,
            },
            "config": {
                "communication": comm_config,
                "reference_origin": list(reference_origin),
                "recommended_deployment": deployment_config,
            },
            "senders": sorted_senders,
        }
        f.write(json.dumps(header))
        f.write("\n")

        if bundling:
            # Sort bundle keys by effective time
            sorted_keys = sorted(
                bundles.keys(),
                key=lambda k: (
                    (k[1] + 1) * window_sec + 0.01 + 0.04 * sensor_meta_map[k[0]].network_tier
                ),
            )
            for sensor_idx, window_idx in sorted_keys:
                bundle = bundles[(sensor_idx, window_idx)]
                sensor_meta = sensor_meta_map[sensor_idx]

                if (
                    sensor_meta.relay_parent is not None
                    and sensor_meta.relay_parent in sensor_meta_map
                ):
                    receiver_id = sensor_meta_map[sensor_meta.relay_parent].sensor_id
                else:
                    receiver_id = "backend"

                delay = 0.01 + (0.04 * sensor_meta.network_tier)
                effective_time = (window_idx + 1) * window_sec + delay

                max_bs = strategy.max_bundle_size
                if max_bs and len(bundle) > max_bs:
                    for i in range(0, len(bundle), max_bs):
                        chunk = bundle[i : i + max_bs]
                        messages = [
                            generate_track_message(ts, sensor_meta, reference_origin)
                            for ts in chunk
                        ]
                        event_line = {
                            "time": effective_time,
                            "sender": sensor_meta.sensor_id,
                            "receiver": receiver_id,
                            "transport": strategy.transport_protocol,
                            "payload": {"tracks": messages},
                            "bundle_size": len(messages),
                        }
                        f.write(json.dumps(event_line))
                        f.write("\n")
                else:
                    messages = [
                        generate_track_message(ts, sensor_meta, reference_origin) for ts in bundle
                    ]
                    event_line = {
                        "time": effective_time,
                        "sender": sensor_meta.sensor_id,
                        "receiver": receiver_id,
                        "transport": strategy.transport_protocol,
                        "payload": {"tracks": messages},
                        "bundle_size": len(messages),
                    }
                    f.write(json.dumps(event_line))
                    f.write("\n")
        else:
            # Direct mode: one line per track_state, payload generated lazily
            for ts in track_states:
                sensor_meta = sensor_meta_map.get(ts.sensor_idx)
                if not sensor_meta:
                    continue

                msg = generate_track_message(ts, sensor_meta, reference_origin)

                if (
                    sensor_meta.relay_parent is not None
                    and sensor_meta.relay_parent in sensor_meta_map
                ):
                    receiver_id = sensor_meta_map[sensor_meta.relay_parent].sensor_id
                else:
                    receiver_id = "backend"

                delay = 0.01 + (0.04 * sensor_meta.network_tier)

                event_line = {
                    "time": ts.time_sec + delay,
                    "sender": sensor_meta.sensor_id,
                    "receiver": receiver_id,
                    "transport": strategy.transport_protocol,
                    "payload": msg,
                    "bundle_size": 1,
                }
                f.write(json.dumps(event_line))
                f.write("\n")
