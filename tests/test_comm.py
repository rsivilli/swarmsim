"""Tests for communication emulation module."""

from __future__ import annotations

import json
import tempfile


from modsim.models import (
    CommStrategy,
    NetworkTopology,
    SensorMetadata,
    TrackStateRecord,
)
from modsim.comm import TrackManager, generate_track_message
from modsim.comm.export import (
    export_locust_format,
    export_locust_jsonl,
    generate_message_timeline,
)


class TestCommunication:
    """Test communication emulation features."""

    def test_track_manager_creates_unique_ids(self):
        """TrackManager should create unique track IDs."""
        manager = TrackManager()

        track_id1 = manager.get_or_create_track_id("obj_1")
        track_num1 = manager.generate_track_num("obj_1")
        status1 = manager.get_track_status("obj_1", 0)

        track_id2 = manager.get_or_create_track_id("obj_2")
        track_num2 = manager.generate_track_num("obj_2")
        status2 = manager.get_track_status("obj_2", 0)

        assert track_id1 != track_id2
        assert track_num1 != track_num2
        assert status1 == "INITIATING"
        assert status2 == "INITIATING"

    def test_track_manager_maintains_persistent_id(self):
        """TrackManager should return same ID for repeated object."""
        manager = TrackManager()

        track_id1 = manager.get_or_create_track_id("obj_1")
        manager.update_detection("obj_1", 0)

        track_id2 = manager.get_or_create_track_id("obj_1")
        manager.update_detection("obj_1", 5)
        status2 = manager.get_track_status("obj_1", 5)

        assert track_id1 == track_id2
        assert status2 == "MAINTAINING"

    def test_track_message_generation(self):
        """generate_track_message should create Track.json format."""
        track_state = TrackStateRecord(
            timestep=0,
            time_sec=0.0,
            obj_id="test_obj",
            sensor_idx=0,
            trk_id="abc-123",
            trk_num="J0001",
            trk_stat="MAINTAINING",
            obj_x=5.0,
            obj_y=5.0,
            obj_z=1.0,  # 1km altitude
            obj_vx=0.1,
            obj_vy=0.0,
            obj_vz=0.0,
            speed_kmps=0.1,
            heading_deg=90.0,
            range_km=2.0,
            pd=0.95,
        )

        sensor_meta = SensorMetadata(
            sensor_idx=0,
            sensor_id="SENSOR-001",
            asset="TEST-PLATFORM",
            asset_nat="US",
            sen_type="RADAR",
            network_tier=0,
            relay_parent=None,
        )

        message = generate_track_message(track_state, sensor_meta)

        # Check required Track.json fields
        assert message["trkId"] == "abc-123"
        assert message["trkNum"] == "J0001"
        assert message["trkStat"] == "MAINTAINING"
        assert message["objId"] == "test_obj"
        assert message["asset"] == "TEST-PLATFORM"
        assert message["assetNat"] == "US"
        assert message["sen"] == "SENSOR-001"
        assert message["modType"] == "RADAR"

        # Check position fields
        assert "lat" in message
        assert "lon" in message
        assert message["alt"] == 1000.0  # 1km converted to meters

        # Check motion fields
        assert message["spd"] == 100.0  # 0.1 km/s = 100 m/s
        assert message["hdng"] == 90.0

        # Check quality fields
        assert message["trkConf"] == 0.95


# ---------------------------------------------------------------------------
# Shared fixtures for JSONL tests
# ---------------------------------------------------------------------------


def _make_test_data(n_tracks=10, n_sensors=2):
    """Build track_states, sensor_metadata, topology, strategy for testing."""
    sensors = [
        SensorMetadata(
            sensor_idx=i,
            sensor_id=f"SENSOR-{i:02d}",
            asset=f"PLATFORM-{i}",
            asset_nat="US",
            sen_type="RADAR",
            network_tier=0,
            relay_parent=None,
        )
        for i in range(n_sensors)
    ]

    tracks = []
    for t in range(n_tracks):
        tracks.append(
            TrackStateRecord(
                timestep=t,
                time_sec=float(t * 10),
                obj_id=f"obj_{t}",
                sensor_idx=t % n_sensors,
                trk_id=f"trk-{t:04d}",
                trk_num=f"J{t:04d}",
                trk_stat="MAINTAINING",
                obj_x=float(t),
                obj_y=5.0,
                obj_z=0.5,
                obj_vx=0.05,
                obj_vy=0.0,
                obj_vz=0.0,
                speed_kmps=0.05,
                heading_deg=90.0,
                range_km=3.0,
                pd=0.9,
            )
        )

    topology = NetworkTopology()
    strategy = CommStrategy(
        strategy_name="direct",
        transport_protocol="zenoh_pubsub",
        bundling_enabled=False,
    )

    return tracks, sensors, topology, strategy


class TestJSONLExport:
    """Test JSONL streaming export."""

    def test_jsonl_header_contains_sorted_senders(self):
        tracks, sensors, topology, strategy = _make_test_data()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        export_locust_jsonl(
            tracks,
            sensors,
            path,
            topology=topology,
            strategy=strategy,
            scenario_name="test",
        )

        with open(path) as f:
            header = json.loads(f.readline())

        assert "senders" in header
        assert header["senders"] == sorted(header["senders"])
        assert set(header["senders"]) == {"SENSOR-00", "SENSOR-01"}

    def test_jsonl_event_count_matches_header(self):
        tracks, sensors, topology, strategy = _make_test_data(n_tracks=20)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        export_locust_jsonl(
            tracks,
            sensors,
            path,
            topology=topology,
            strategy=strategy,
        )

        with open(path) as f:
            header = json.loads(f.readline())
            event_lines = [line for line in f if line.strip()]

        assert len(event_lines) == header["metadata"]["total_events"]

    def test_jsonl_events_match_json_roundtrip(self):
        """JSONL events should produce the same payloads as JSON export."""
        tracks, sensors, topology, strategy = _make_test_data(n_tracks=5)

        # JSON export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json_path = f.name
        events = generate_message_timeline(tracks, sensors, topology, strategy)
        export_locust_format(
            events,
            json_path,
            scenario_name="test",
            topology=topology,
            strategy=strategy,
            num_sensors=len(sensors),
        )

        # JSONL export
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            jsonl_path = f.name
        export_locust_jsonl(
            tracks,
            sensors,
            jsonl_path,
            topology=topology,
            strategy=strategy,
            scenario_name="test",
            num_sensors=len(sensors),
        )

        # Load both
        with open(json_path) as f:
            json_data = json.load(f)
        json_events = json_data["events"]

        with open(jsonl_path) as f:
            header = json.loads(f.readline())
            jsonl_events = [json.loads(line) for line in f if line.strip()]

        # Same event count
        assert len(jsonl_events) == len(json_events)

        # Same senders
        json_senders = sorted(set(e["sender"] for e in json_events))
        assert header["senders"] == json_senders

        # Same total_messages in metadata
        assert header["metadata"]["total_messages"] == json_data["metadata"]["total_messages"]

        # Each event has matching payload keys
        # (exact order may differ, so compare by sender groups)
        jsonl_by_sender = {}
        for e in jsonl_events:
            jsonl_by_sender.setdefault(e["sender"], []).append(e)
        json_by_sender = {}
        for e in json_events:
            json_by_sender.setdefault(e["sender"], []).append(e)

        assert set(jsonl_by_sender.keys()) == set(json_by_sender.keys())
        for sender in jsonl_by_sender:
            assert len(jsonl_by_sender[sender]) == len(json_by_sender[sender])

    def test_jsonl_bundled_mode(self):
        """Bundled strategy should produce bundled events in JSONL."""
        tracks, sensors, topology, _ = _make_test_data(n_tracks=10)
        strategy = CommStrategy(
            strategy_name="bundled",
            transport_protocol="zenoh_pubsub",
            bundling_enabled=True,
            bundle_window_sec=20.0,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        export_locust_jsonl(
            tracks,
            sensors,
            path,
            topology=topology,
            strategy=strategy,
        )

        with open(path) as f:
            header = json.loads(f.readline())
            events = [json.loads(line) for line in f if line.strip()]

        # Bundled events should have "tracks" key in payload
        for event in events:
            assert "tracks" in event["payload"]
            assert event["bundle_size"] == len(event["payload"]["tracks"])

        # Total messages should equal total tracks
        total_bundle_size = sum(e["bundle_size"] for e in events)
        assert total_bundle_size == header["metadata"]["total_messages"]
