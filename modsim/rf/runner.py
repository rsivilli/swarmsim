"""Orchestration of the RF pass: emission schedule, link budget, truth tables and sizing."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..errors import ConfigError
from ..geo import km_to_latlon
from ..models import ScenarioConfig
from ..sim import run as run_sim
from .geometry import Trajectory, build_trajectories
from .links import EmitterStates, compute_links, emitter_states
from .models import EmissionTable, Emitter, RFSensorConfig
from .scheduler import build_emitters, schedule_all
from .sizing import RecordingEstimate, free_bytes, recordings_for_sensor
from .truth import resolve_format, write_table


@dataclass
class RFPlan:
    cfg: ScenarioConfig
    window: tuple[float, float]
    sensors: list[RFSensorConfig]
    trajectories: dict[str, Trajectory]
    emitters: list[Emitter]
    emissions: EmissionTable
    states: EmitterStates
    links: dict[str, np.ndarray]
    recordings: list[RecordingEstimate]
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(r.total_bytes for r in self.recordings)


def plan_rf(
    cfg: ScenarioConfig,
    window: tuple[float, float] | None = None,
    sensor_ids: list[str] | None = None,
) -> RFPlan:
    """Run the motion simulation and Stage A (emitters, emissions, link budget, sizing)."""
    rfs = cfg.rf
    if rfs is None:
        raise ConfigError("Scenario has no RF configuration (rf / rf_sensors / groups[].rf)")

    window = window or rfs.rf.window or (0.0, float(cfg.duration))
    if not 0 <= window[0] < window[1] <= cfg.duration:
        raise ConfigError(f"RF window {window} must lie within 0..{cfg.duration} s")

    sensors = rfs.sensors
    if sensor_ids:
        known = {s.sensor_id for s in sensors}
        unknown = [s for s in sensor_ids if s not in known]
        if unknown:
            raise ConfigError(f"Unknown RF sensor(s): {', '.join(unknown)}")
        sensors = [s for s in sensors if s.sensor_id in sensor_ids]

    timings = {}
    t = time.perf_counter()
    state = run_sim(cfg)
    trajectories = build_trajectories(state.object_snapshots, cfg.dt)
    timings["motion_s"] = time.perf_counter() - t

    t = time.perf_counter()
    emitters = build_emitters(cfg, rfs, trajectories, window)
    emissions = schedule_all(emitters, rfs.library, cfg.seed)
    timings["schedule_s"] = time.perf_counter() - t

    t = time.perf_counter()
    states = emitter_states(emissions, emitters, trajectories)
    links = compute_links(emissions, states, sensors, window, rfs.rf.cull_snr_db, rfs.rf.k_factor)
    timings["link_budget_s"] = time.perf_counter() - t

    duration = window[1] - window[0]
    recordings = [r for s in sensors for r in recordings_for_sensor(s, duration)]
    return RFPlan(
        cfg=cfg,
        window=window,
        sensors=sensors,
        trajectories=trajectories,
        emitters=emitters,
        emissions=emissions,
        states=states,
        links=links,
        recordings=recordings,
        timings=timings,
    )


def write_truth(plan: RFPlan, out_dir: Path, truth_format: str = "auto") -> list[Path]:
    """Write emitters / emissions / links truth tables; returns the written paths."""
    fmt = resolve_format(truth_format)
    truth_dir = out_dir / "truth"
    em = plan.emissions
    counts = np.bincount(em.emitter_idx, minlength=len(plan.emitters)) if len(em) else None
    origin = plan.cfg.reference_origin

    def static(e, attr):
        return getattr(e.position, attr) if e.position is not None else float("nan")

    emitters = {
        "emitter_idx": np.arange(len(plan.emitters), dtype=np.int32),
        "emitter_id": [e.emitter_id for e in plan.emitters],
        "kind": [e.kind.value for e in plan.emitters],
        "profile": [e.profile_id for e in plan.emitters],
        "object_id": [e.object_id or "" for e in plan.emitters],
        "group_id": [e.group_id or "" for e in plan.emitters],
        "link_key": [e.link_key for e in plan.emitters],
        "x_km": np.array([static(e, "x") for e in plan.emitters]),
        "y_km": np.array([static(e, "y") for e in plan.emitters]),
        "z_km": np.array([static(e, "z") for e in plan.emitters]),
        "active_start": np.array([e.active[0] for e in plan.emitters]),
        "active_end": np.array([e.active[1] for e in plan.emitters]),
        "eirp_dbm": np.array(
            [e.params["power_dbm"] + e.params["antenna_gain_dbi"] for e in plan.emitters]
        ),
        "num_emissions": counts if counts is not None else np.zeros(len(plan.emitters), int),
    }

    emissions = {"emission_idx": np.arange(len(em), dtype=np.int64), **em.columns()}
    emissions["x_km"] = plan.states.pos_km[:, 0]
    emissions["y_km"] = plan.states.pos_km[:, 1]
    emissions["z_km"] = plan.states.pos_km[:, 2]
    if origin is not None and len(em):
        lat, lon = km_to_latlon(plan.states.pos_km[:, 0], plan.states.pos_km[:, 1], origin)
        emissions["lat"] = np.asarray(lat, dtype=np.float64)
        emissions["lon"] = np.asarray(lon, dtype=np.float64)

    return [
        write_table(truth_dir / "emitters", emitters, fmt),
        write_table(truth_dir / "emissions", emissions, fmt),
        write_table(truth_dir / "links", plan.links, fmt),
    ]


def build_manifest(plan: RFPlan, scenario_path: str | Path | None, files: list[Path]) -> dict:
    cfg, rfs = plan.cfg, plan.cfg.rf
    profile_of = np.array([e.profile_id for e in plan.emitters], dtype=object)
    by_profile = Counter(profile_of[plan.emissions.emitter_idx]) if len(plan.emissions) else {}
    det = plan.links["detectable"]
    sha = None
    if scenario_path is not None:
        sha = hashlib.sha256(Path(scenario_path).read_bytes()).hexdigest()
    return {
        "modsim_version": __version__,
        "mode": "plan",  # becomes "iq" once recordings are synthesized
        "scenario": cfg.name,
        "scenario_file": str(scenario_path) if scenario_path else None,
        "scenario_sha256": sha,
        "seed": cfg.seed,
        "epoch": rfs.rf.epoch,
        "window_s": list(plan.window),
        "annotation_granularity": rfs.rf.annotation_granularity.value,
        "emitters": {
            "total": len(plan.emitters),
            "by_kind": dict(Counter(e.kind.value for e in plan.emitters)),
            "by_profile": dict(Counter(e.profile_id for e in plan.emitters)),
        },
        "emissions": {"total": len(plan.emissions), "by_profile": dict(by_profile)},
        "links": {"total": int(len(det)), "detectable": int(det.sum())},
        "sensors": [
            {
                "id": s.sensor_id,
                "type": s.sensor_type.value,
                "position_km": [s.position.x, s.position.y, s.position.z],
                "num_channels": s.num_channels,
                "tuning": s.tuning.mode.value,
                "bands": [
                    {"name": b.name, "center_hz": b.center_hz, "sample_rate": b.sample_rate}
                    for b in s.tuning.bands
                ],
            }
            for s in plan.sensors
        ],
        "recordings": [
            {
                "sensor_id": r.sensor_id,
                "recording": r.recording,
                "bands": r.bands,
                "sample_rate": r.sample_rate,
                "sample_format": r.sample_format,
                "num_channels": r.num_channels,
                "duration_s": r.duration_s,
                "bytes": r.total_bytes,
            }
            for r in plan.recordings
        ],
        "total_iq_bytes": plan.total_bytes,
        "truth_files": [str(p.name) for p in files],
        "timings_s": {k: round(v, 3) for k, v in plan.timings.items()},
    }


def write_outputs(
    plan: RFPlan, out_dir: Path, scenario_path: str | Path | None, truth_format: str = "auto"
) -> tuple[list[Path], Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = write_truth(plan, out_dir, truth_format)
    manifest = build_manifest(plan, scenario_path, files)
    manifest["free_bytes_at_output"] = free_bytes(out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return files, manifest_path, manifest
