"""Locust load testing script for modsim communication timeline replay.

Replays sensor communication events from a timeline JSON file against backend services.
Supports REST, Zenoh HTTP bridge, native Zenoh, and UDL SDK transports with
deterministic event assignment.

Timeline format auto-detection:
  - .jsonl : Streaming format — header-only load at startup, events streamed per-user
  - .json  : Legacy format — full load at startup (fine for small files)

Transport selection via ZENOH_TRANSPORT env var:
  - "rest"   : HTTP bridge through Zenoh REST plugin (default)
  - "native" : Native Zenoh protocol via eclipse-zenoh SDK
  - "udl"    : UDL SDK track publish (requires udl-sdk, UDL_AUTH_USERNAME,
               UDL_AUTH_PASSWORD)
"""

from __future__ import annotations

import json
import math
import os
import time

from locust import HttpUser, User, between, events, task


# ---------------------------------------------------------------------------
# Timeline loading: auto-detect format by extension
# ---------------------------------------------------------------------------
TIMELINE_PATH = os.getenv("TIMELINE_PATH", "/timeline.json")
_IS_JSONL = TIMELINE_PATH.endswith(".jsonl")

if _IS_JSONL:
    # JSONL: read only the header (line 1) — O(header) memory per worker
    with open(TIMELINE_PATH) as _f:
        _HEADER = json.loads(_f.readline())
    METADATA = _HEADER["metadata"]
    _ALL_SENDERS = _HEADER["senders"]
    EVENTS = None  # Streamed lazily per-user
else:
    # Legacy JSON: full load (fine for small files)
    with open(TIMELINE_PATH) as _f:
        _TIMELINE_DATA = json.load(_f)
    EVENTS = _TIMELINE_DATA["events"]
    METADATA = _TIMELINE_DATA["metadata"]
    _ALL_SENDERS = sorted(set(e["sender"] for e in EVENTS))

# Transport mode: "rest" | "native" | "udl"
ZENOH_TRANSPORT = os.getenv("ZENOH_TRANSPORT", "rest").lower()

# Native Zenoh connection config
ZENOH_ROUTER = os.getenv("ZENOH_ROUTER", "tcp/localhost:7447")

# UDL config
UDL_AUTH_USERNAME = os.getenv("UDL_AUTH_USERNAME", "")
UDL_AUTH_PASSWORD = os.getenv("UDL_AUTH_PASSWORD", "")
UDL_BASE_URL = os.getenv("UDL_BASE_URL", "")


# ---------------------------------------------------------------------------
# Event sources: unified interface for list-backed and streaming reads
# ---------------------------------------------------------------------------


class _ListEventSource:
    """Wraps a pre-loaded list of events (legacy JSON path)."""

    def __init__(self, events: list[dict]):
        self._events = events
        self._idx = 0

    def peek_time(self) -> float | None:
        if self._idx < len(self._events):
            return self._events[self._idx]["time"]
        return None

    def collect_batch(self) -> list[dict]:
        if self._idx >= len(self._events):
            return []
        batch_time = self._events[self._idx]["time"]
        batch = []
        while self._idx < len(self._events) and self._events[self._idx]["time"] == batch_time:
            batch.append(self._events[self._idx])
            self._idx += 1
        return batch

    @property
    def done(self) -> bool:
        return self._idx >= len(self._events)

    @property
    def total_loaded(self) -> int:
        return len(self._events)

    def close(self):
        pass


class _StreamEventSource:
    """Lazy JSONL reader — O(batch) memory instead of O(all_user_events).

    Streams the JSONL file line-by-line, pre-filters with substring match,
    and only parses lines matching assigned senders. Keeps a small read-ahead
    buffer (one timestep past the current batch) so collect_batch() can
    return all events at a given timestamp.
    """

    def __init__(self, path: str, sender_set: set[str]):
        self._sender_set = sender_set
        self._needles = [f'"sender": "{s}"' for s in sender_set]
        self._f = open(path)
        self._f.readline()  # skip header
        self._buf: list[dict] = []
        self._eof = False
        self._events_read = 0

    def _read_next_match(self) -> dict | None:
        """Read lines until we find one matching our senders, or EOF."""
        if self._eof:
            return None
        for line in self._f:
            if not any(n in line for n in self._needles):
                continue
            event = json.loads(line)
            if event["sender"] in self._sender_set:
                self._events_read += 1
                return event
        self._eof = True
        self._f.close()
        return None

    def _ensure_buffer(self):
        if not self._buf and not self._eof:
            ev = self._read_next_match()
            if ev:
                self._buf.append(ev)

    def peek_time(self) -> float | None:
        self._ensure_buffer()
        return self._buf[0]["time"] if self._buf else None

    def collect_batch(self) -> list[dict]:
        """Pop all events at the next timestamp from the stream."""
        self._ensure_buffer()
        if not self._buf:
            return []

        batch_time = self._buf[0]["time"]

        # Read ahead until we see a different timestamp (or EOF)
        while not self._eof:
            ev = self._read_next_match()
            if ev is None:
                break
            self._buf.append(ev)
            if ev["time"] != batch_time:
                break  # this event belongs to the next batch

        # Split: batch events vs read-ahead remainder
        batch = []
        remaining = []
        for ev in self._buf:
            if ev["time"] == batch_time:
                batch.append(ev)
            else:
                remaining.append(ev)
        self._buf = remaining
        return batch

    @property
    def done(self) -> bool:
        self._ensure_buffer()
        return not self._buf and self._eof

    @property
    def total_loaded(self) -> int:
        return self._events_read

    def close(self):
        if not self._eof:
            self._f.close()
            self._eof = True


# ---------------------------------------------------------------------------
# User/sender assignment
# ---------------------------------------------------------------------------

_user_counter = 0
_user_counter_lock = __import__("threading").Lock()


def _create_event_source() -> tuple[list[str], _ListEventSource | _StreamEventSource]:
    """Assign senders to this user and return an event source.

    JSONL: returns a streaming reader (O(batch) memory per user).
    JSON:  returns a list wrapper (events already in memory).
    """
    global _user_counter
    if not _ALL_SENDERS:
        return [], _ListEventSource([])

    with _user_counter_lock:
        idx = _user_counter
        _user_counter += 1

    my_senders = [
        s for i, s in enumerate(_ALL_SENDERS) if i % len(_ALL_SENDERS) == idx % len(_ALL_SENDERS)
    ]
    if not my_senders:
        my_senders = [_ALL_SENDERS[idx % len(_ALL_SENDERS)]]

    my_sender_set = set(my_senders)

    if _IS_JSONL:
        source = _StreamEventSource(TIMELINE_PATH, my_sender_set)
    else:
        my_events = [e for e in EVENTS if e["sender"] in my_sender_set]
        source = _ListEventSource(my_events)

    return my_senders, source


# ---------------------------------------------------------------------------
# Shared timestamp rebasing: sim-relative → wall-clock
# ---------------------------------------------------------------------------
UDL_CLASSIFICATION = os.getenv("UDL_CLASSIFICATION", "")

_time_anchor: float | None = None


def _rebase_ts(event_time_sec: float) -> str:
    """Convert simulation time to a wall-clock ISO8601 timestamp."""
    from datetime import datetime, timezone

    global _time_anchor
    if _time_anchor is None:
        _time_anchor = time.time()

    wall_time = datetime.fromtimestamp(_time_anchor + event_time_sec, tz=timezone.utc)
    return wall_time.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _rebase_payload_timestamps(payload: dict, event_time_sec: float) -> dict:
    """Return a copy of payload with ts/msgTs rebased to wall-clock.

    Bundled payloads ({"tracks": [...]}) have the timestamps on each track.
    """
    ts_iso = _rebase_ts(event_time_sec)
    payload = dict(payload)
    if "ts" in payload:
        payload["ts"] = ts_iso
    if "msgTs" in payload:
        payload["msgTs"] = ts_iso
    if isinstance(payload.get("tracks"), list):
        payload["tracks"] = [
            _rebase_payload_timestamps(t, event_time_sec) if isinstance(t, dict) else t
            for t in payload["tracks"]
        ]
    return payload


def _modsim_track_to_udl(track: dict, event_time_sec: float) -> dict:
    """Map a modsim Track.json payload to UDL track format."""
    ts_iso = _rebase_ts(event_time_sec)

    spd = track.get("spd", 0.0)
    hdng_deg = track.get("hdng", 0.0)
    hdng_rad = math.radians(hdng_deg)
    vx = spd * math.sin(hdng_rad)
    vy = spd * math.cos(hdng_rad)
    vz = 0.0

    env_to_track_type = {
        "AIR": "AIRCRAFT",
        "SURFACE": "SHIP",
        "LAND": "GROUND_VEHICLE",
        "SPACE": "SATELLITE",
        "SUBSURFACE": "UNKNOWN",
    }

    return {
        "classification_marking": UDL_CLASSIFICATION,
        "data_mode": track.get("dataMode", "SIMULATED"),
        "source": track.get("source", "modsim"),
        "track_id": track.get("trkId", ""),
        "track_type": env_to_track_type.get(track.get("env", ""), "UNKNOWN"),
        "track_name": track.get("trkNum", ""),
        "lat": track.get("lat", 0.0),
        "lon": track.get("lon", 0.0),
        "altitude": track.get("alt", 0.0),
        "velocity_x": vx,
        "velocity_y": vy,
        "velocity_z": vz,
        "epoch": ts_iso,
        "ts": ts_iso,
    }


# ---------------------------------------------------------------------------
# Transport User classes
# ---------------------------------------------------------------------------


class SensorReplayUser(HttpUser):
    """Replays sensor events via HTTP (REST API or Zenoh REST bridge)."""

    wait_time = between(0, 0.001)
    abstract = ZENOH_TRANSPORT not in ("rest",)

    def on_start(self):
        my_senders, self._source = _create_event_source()
        self.start_time = time.time()
        print(f"User {id(self)} [HTTP]: Assigned {len(my_senders)} sensors")

    def on_stop(self):
        self._source.close()

    @task
    def replay_event(self):
        next_time = self._source.peek_time()
        if next_time is None:
            time.sleep(1)
            return

        elapsed = time.time() - self.start_time
        if elapsed < next_time:
            time.sleep(next_time - elapsed)

        batch = self._source.collect_batch()
        if not batch:
            return

        # Group by sender
        by_sender: dict[str, list] = {}
        for ev in batch:
            by_sender.setdefault(ev["sender"], []).append(ev)

        transport = batch[0].get("transport", "rest_post")
        if "rest" in transport.lower():
            self._send_rest_batch(by_sender)
        elif "zenoh" in transport.lower():
            self._send_zenoh_bridge_batch(by_sender)

    def _send_rest_batch(self, by_sender: dict):
        """Send batched tracks per sender via REST POST."""
        for sender, evts in by_sender.items():
            event_time = evts[0].get("time", 0.0)
            payloads = [_rebase_payload_timestamps(e["payload"], event_time) for e in evts]
            body = payloads[0] if len(payloads) == 1 else {"tracks": payloads}
            receiver = evts[0].get("receiver", "backend")
            with self.client.post(
                "/api/tracks",
                json=body,
                catch_response=True,
                name=f"POST tracks [{sender}→{receiver}, {len(payloads)} msgs]",
            ) as response:
                if response.status_code in (200, 201):
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")

    def _send_zenoh_bridge_batch(self, by_sender: dict):
        """Send batched tracks per sender via Zenoh REST bridge PUT."""
        for sender, evts in by_sender.items():
            event_time = evts[0].get("time", 0.0)
            payloads = [_rebase_payload_timestamps(e["payload"], event_time) for e in evts]
            body = payloads[0] if len(payloads) == 1 else {"tracks": payloads}
            key_expr = f"tracks/{sender}"
            with self.client.put(
                f"/{key_expr}",
                json=body,
                catch_response=True,
                name=f"Zenoh PUT [{sender}, {len(payloads)} msgs]",
            ) as response:
                if response.status_code in (200, 204):
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")


class ZenohNativeUser(User):
    """Replays sensor events over native Zenoh protocol.

    Uses eclipse-zenoh SDK to publish directly to the Zenoh router,
    bypassing the REST HTTP bridge for lower latency and more realistic
    wire-level testing.

    Requires: pip install eclipse-zenoh
    Env vars:
      ZENOH_ROUTER  - Router endpoint (default: tcp/localhost:7447)
    """

    wait_time = between(0, 0.001)
    abstract = ZENOH_TRANSPORT != "native"

    def on_start(self):
        import zenoh

        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([ZENOH_ROUTER]))
        config.insert_json5("scouting/multicast/enabled", json.dumps(False))

        self._session = zenoh.open(config)
        self._publishers = {}

        my_senders, self._source = _create_event_source()
        self.start_time = time.time()
        print(
            f"User {id(self)} [Zenoh native → {ZENOH_ROUTER}]: Assigned {len(my_senders)} sensors"
        )

    def on_stop(self):
        self._source.close()
        for pub in self._publishers.values():
            pub.undeclare()
        self._publishers.clear()
        self._session.close()

    def _get_publisher(self, key_expr):
        """Get or create a declared publisher for a key expression."""
        import zenoh

        if key_expr not in self._publishers:
            self._publishers[key_expr] = self._session.declare_publisher(
                key_expr,
                encoding=zenoh.Encoding.APPLICATION_JSON,
            )
        return self._publishers[key_expr]

    @task
    def replay_event(self):
        next_time = self._source.peek_time()
        if next_time is None:
            time.sleep(1)
            return

        elapsed = time.time() - self.start_time
        if elapsed < next_time:
            time.sleep(next_time - elapsed)

        batch = self._source.collect_batch()
        if not batch:
            return

        self._send_zenoh_native_batch(batch)

    def _send_zenoh_native_batch(self, batch):
        """Publish a batch of events at the same timestamp via native Zenoh."""
        event_time = batch[0].get("time", 0.0)

        by_sender: dict[str, list] = {}
        for event in batch:
            sender = event["sender"]
            by_sender.setdefault(sender, []).append(
                _rebase_payload_timestamps(event["payload"], event_time)
            )

        total_len = 0
        start = time.perf_counter()
        exception = None
        msg_count = 0

        try:
            for sender, payloads in by_sender.items():
                key_expr = f"tracks/{sender}"
                pub = self._get_publisher(key_expr)
                if len(payloads) == 1:
                    data = json.dumps(payloads[0])
                else:
                    data = json.dumps({"tracks": payloads})
                pub.put(data)
                total_len += len(data)
                msg_count += len(payloads)
        except (OSError, ConnectionError, RuntimeError) as e:
            exception = e

        elapsed_ms = (time.perf_counter() - start) * 1000

        self.environment.events.request.fire(
            request_type="ZENOH",
            name=f"Zenoh batch [{msg_count} msgs, {len(by_sender)} keys]",
            response_time=elapsed_ms,
            response_length=total_len,
            exception=exception,
            context={},
        )


class UDLReplayUser(User):
    """Replays sensor events by publishing tracks to UDL.

    Uses the UDL SDK (unifieddatalibrary) to post track data, mapping
    modsim Track.json payloads to the UDL track format.

    Requires: pip install udl-sdk
    Env vars:
      UDL_AUTH_USERNAME - UDL username
      UDL_AUTH_PASSWORD - UDL password
      UDL_BASE_URL      - UDL base URL (required)
      UDL_CLASSIFICATION - classification marking applied to published tracks (required)
    """

    wait_time = between(0, 0.001)
    abstract = ZENOH_TRANSPORT != "udl"

    def on_start(self):
        missing = [
            name
            for name, value in (
                ("UDL_BASE_URL", UDL_BASE_URL),
                ("UDL_CLASSIFICATION", UDL_CLASSIFICATION),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"UDL transport requires: {', '.join(missing)}")

        from unifieddatalibrary import Unifieddatalibrary

        self._udl_client = Unifieddatalibrary(
            username=UDL_AUTH_USERNAME,
            password=UDL_AUTH_PASSWORD,
            base_url=UDL_BASE_URL,
        )

        my_senders, self._source = _create_event_source()
        self.start_time = time.time()
        print(f"User {id(self)} [UDL → {UDL_BASE_URL}]: Assigned {len(my_senders)} sensors")

    def on_stop(self):
        self._source.close()

    @task
    def replay_event(self):
        next_time = self._source.peek_time()
        if next_time is None:
            time.sleep(1)
            return

        elapsed = time.time() - self.start_time
        if elapsed < next_time:
            time.sleep(next_time - elapsed)

        batch = self._source.collect_batch()
        if not batch:
            return

        self._send_udl_batch(batch)

    def _send_udl_batch(self, batch):
        """Publish a batch of tracks to UDL in a single API call."""
        event_time = batch[0].get("time", 0.0)

        all_tracks = []
        for event in batch:
            payload = event["payload"]
            raw = payload.get("tracks", [payload]) if isinstance(payload, dict) else [payload]
            all_tracks.extend(raw)

        udl_tracks = [_modsim_track_to_udl(t, event_time) for t in all_tracks]
        senders = set(e["sender"] for e in batch)
        request_name = f"UDL batch [{len(all_tracks)} tracks, {len(senders)} sensors]"

        start = time.perf_counter()
        exception = None
        try:
            self._udl_client.track.unvalidated_publish(body=udl_tracks)
        except (OSError, ConnectionError, RuntimeError, ValueError) as e:
            exception = e

        elapsed_ms = (time.perf_counter() - start) * 1000

        self.environment.events.request.fire(
            request_type="UDL",
            name=request_name,
            response_time=elapsed_ms,
            response_length=len(all_tracks),
            exception=exception,
            context={},
        )

        if exception:
            raise exception


# ---------------------------------------------------------------------------
# Custom event handlers for metrics
# ---------------------------------------------------------------------------
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    fmt = "jsonl (streaming)" if _IS_JSONL else "json"
    print(
        f"Starting load test with {METADATA['total_events']} events "
        f"over {METADATA['duration_sec']}s "
        f"(transport: {ZENOH_TRANSPORT}, format: {fmt})"
    )


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    print(f"Load test complete. Total messages sent: {METADATA['total_messages']}")
