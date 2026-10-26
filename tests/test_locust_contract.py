"""Contract tests between the comms export and locust/locustfile.py.

The locustfile runs in its own container and never imports modsim, so nothing else checks that
it can read what `modsim run --export-mode comms` writes. These tests export small timelines and
drive the locustfile's own readers on them. `locust` is stubbed so the file can be imported without
the load-testing dependencies; the zenoh and UDL SDKs are only imported inside the user classes.
"""

import collections
import dataclasses
import importlib.util
import itertools
import json
import sys
import types
from datetime import datetime

import pytest

from modsim.comm.export import export_locust_jsonl
from modsim.config import load_scenario
from modsim.report import export_json
from modsim.sim import run

from conftest import PROJECT_ROOT, SCENARIOS_DIR

LOCUSTFILE = PROJECT_ROOT / "locust" / "locustfile.py"
_load_count = itertools.count()


def _stub_locust() -> types.ModuleType:
    class _Hook:
        def add_listener(self, fn):
            return fn

    class User:
        pass

    class HttpUser(User):
        pass

    mod = types.ModuleType("locust")
    mod.User = User
    mod.HttpUser = HttpUser
    mod.between = lambda lo, hi: None
    mod.task = lambda fn: fn
    mod.events = types.SimpleNamespace(test_start=_Hook(), test_stop=_Hook())
    return mod


def load_locustfile(timeline_path, monkeypatch):
    """Import a fresh copy of the locustfile reading `timeline_path`."""
    monkeypatch.setitem(sys.modules, "locust", _stub_locust())
    monkeypatch.setenv("TIMELINE_PATH", str(timeline_path))
    name = f"_locustfile_under_test_{next(_load_count)}"
    spec = importlib.util.spec_from_file_location(name, LOCUSTFILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _export_pair(cfg, out_dir, stem):
    """Write the same simulation as .json (in-memory path) and .jsonl (streaming path)."""
    state = run(cfg)
    comm = cfg.communication
    json_path = out_dir / f"{stem}.json"
    jsonl_path = out_dir / f"{stem}.jsonl"
    export_json(state, str(json_path), "comms", cfg)
    export_locust_jsonl(
        state.track_state_log,
        state.sensor_metadata,
        str(jsonl_path),
        topology=comm.topology,
        strategy=comm.strategy,
        reference_origin=cfg.reference_origin or comm.reference_origin,
        scenario_name=cfg.name,
        num_sensors=len(state.sensors),
    )
    return json_path, jsonl_path


@pytest.fixture(scope="module")
def timelines(tmp_path_factory):
    """Bundled and direct timelines from comms_test.yaml, in both export formats."""
    out = tmp_path_factory.mktemp("timelines")
    bundled = dataclasses.replace(load_scenario(SCENARIOS_DIR / "comms_test.yaml"), duration=10)
    direct = dataclasses.replace(
        bundled,
        communication=dataclasses.replace(
            bundled.communication,
            strategy=dataclasses.replace(bundled.communication.strategy, bundling_enabled=False),
        ),
    )
    paths = {}
    for mode, cfg in (("bundled", bundled), ("direct", direct)):
        paths[mode, "json"], paths[mode, "jsonl"] = _export_pair(cfg, out, mode)
    return paths


TIMELINE_KINDS = [
    ("bundled", "json"),
    ("bundled", "jsonl"),
    ("direct", "json"),
    ("direct", "jsonl"),
]


def _all_events(path):
    if str(path).endswith(".jsonl"):
        with open(path) as f:
            f.readline()
            return [json.loads(line) for line in f]
    with open(path) as f:
        return json.load(f)["events"]


def _drain(source):
    batches = []
    while not source.done:
        batch = source.collect_batch()
        assert batch, "collect_batch() returned nothing while the source was not done"
        batches.append(batch)
    assert source.collect_batch() == []
    assert source.peek_time() is None
    source.close()
    return batches


def _event_key(event):
    return (round(event["time"], 9), event["sender"], json.dumps(event, sort_keys=True))


@pytest.mark.parametrize("kind", TIMELINE_KINDS, ids="-".join)
def test_header_has_what_locustfile_reads(timelines, kind, monkeypatch):
    lf = load_locustfile(timelines[kind], monkeypatch)
    for key in ("total_events", "duration_sec", "total_messages"):
        assert key in lf.METADATA
    events = _all_events(timelines[kind])
    assert events, "fixture scenario produced no comms events"
    assert lf.METADATA["total_events"] == len(events)
    assert lf._ALL_SENDERS == sorted({e["sender"] for e in events})
    # The hooks print these on test start/stop; they must not raise.
    lf.on_test_start(None)
    lf.on_test_stop(None)


@pytest.mark.parametrize("kind", TIMELINE_KINDS, ids="-".join)
def test_events_have_fields_transports_use(timelines, kind):
    for event in _all_events(timelines[kind]):
        assert isinstance(event["time"], (int, float))
        assert isinstance(event["sender"], str)
        assert isinstance(event["payload"], dict)
        assert event.get("transport", "rest_post")
        assert event.get("receiver", "backend")


@pytest.mark.parametrize("kind", TIMELINE_KINDS, ids="-".join)
def test_each_sender_replays_all_its_events_in_time_order(timelines, kind, monkeypatch):
    """Per user, batches are single-timestamp, increasing, and cover every assigned event once.

    For .jsonl this also checks the streaming reader's substring pre-filter, which depends on
    the exporter's JSON spacing (`"sender": "X"`).
    """
    path = timelines[kind]
    lf = load_locustfile(path, monkeypatch)
    expected = collections.defaultdict(list)
    for event in _all_events(path):
        expected[event["sender"]].append(event)

    seen = []
    for _ in lf._ALL_SENDERS:
        senders, source = lf._create_event_source()
        batches = _drain(source)
        times = [batch[0]["time"] for batch in batches]
        assert times == sorted(times) and len(set(times)) == len(times)
        for batch in batches:
            assert {e["time"] for e in batch} == {batch[0]["time"]}
            assert {e["sender"] for e in batch} <= set(senders)
        replayed = [e for batch in batches for e in batch]
        assert source.total_loaded == len(replayed)
        assert sorted(map(_event_key, replayed)) == sorted(
            _event_key(e) for s in senders for e in expected[s]
        )
        seen.extend(senders)

    # One user per sender: every sender is assigned exactly once.
    assert sorted(seen) == lf._ALL_SENDERS


@pytest.mark.parametrize("mode", ["bundled", "direct"])
def test_json_and_jsonl_exports_agree(timelines, mode):
    """Small runs export .json and large ones .jsonl; Locust must see the same timeline."""
    json_path, jsonl_path = timelines[mode, "json"], timelines[mode, "jsonl"]
    with open(json_path) as f:
        json_meta = json.load(f)["metadata"]
    with open(jsonl_path) as f:
        jsonl_meta = json.loads(f.readline())["metadata"]
    assert json_meta == jsonl_meta

    def key(e):
        tracks = e["payload"].get("tracks", [e["payload"]])
        ids = tuple(t["trkId"] for t in tracks)
        return (round(e["time"], 9), e["sender"], e.get("receiver"), e.get("transport"), ids)

    assert collections.Counter(map(key, _all_events(json_path))) == collections.Counter(
        map(key, _all_events(jsonl_path))
    )


def _tracks(payload):
    return payload.get("tracks", [payload])


@pytest.mark.parametrize("mode", ["bundled", "direct"])
def test_rebased_payloads_carry_wall_clock_timestamps(timelines, mode, monkeypatch):
    lf = load_locustfile(timelines[mode, "jsonl"], monkeypatch)
    event = _all_events(timelines[mode, "jsonl"])[-1]
    original = json.dumps(event["payload"], sort_keys=True)

    rebased = lf._rebase_payload_timestamps(event["payload"], event["time"])

    expected = lf._rebase_ts(event["time"])
    assert datetime.fromisoformat(expected).year >= 2026
    tracks = _tracks(rebased)
    assert tracks
    for track in tracks:
        assert track["ts"] == expected
        assert track["msgTs"] == expected
    # The source event is shared between batches and must not be modified.
    assert json.dumps(event["payload"], sort_keys=True) == original


def test_udl_mapping_reads_track_fields(timelines, monkeypatch):
    lf = load_locustfile(timelines["bundled", "json"], monkeypatch)
    event = _all_events(timelines["bundled", "json"])[0]
    track = _tracks(event["payload"])[0]
    for key in ("trkId", "trkNum", "lat", "lon", "alt", "spd", "hdng", "env", "dataMode"):
        assert key in track, f"Track payload lost {key!r}, which the UDL mapping reads"

    udl = lf._modsim_track_to_udl(track, event["time"])

    assert udl["track_id"] == track["trkId"]
    assert udl["track_name"] == track["trkNum"]
    assert (udl["lat"], udl["lon"], udl["altitude"]) == (track["lat"], track["lon"], track["alt"])
    assert udl["track_type"] == "AIRCRAFT"
    assert udl["velocity_x"] ** 2 + udl["velocity_y"] ** 2 == pytest.approx(track["spd"] ** 2)
    assert udl["ts"] == lf._rebase_ts(event["time"])
