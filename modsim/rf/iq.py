"""IQ synthesis orchestration: recording jobs, parallel chunk rendering, SigMF metadata."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .models import RFSensorConfig, TuningMode
from .propagation import C_MPS
from .receiver import build_receiver
from .runner import RFPlan
from .sigmf_writer import build_annotations, build_meta, write_collection, write_json
from .synth import EmitterInfo, RecordingJob, render_chunk


def unsupported_reason(sensor: RFSensorConfig) -> str | None:
    """Why a sensor can't be synthesized yet (None if it can)."""
    if sensor.array is not None:
        return "antenna arrays are synthesized from Phase 3"
    if sensor.tuning.mode == TuningMode.SCANNING:
        return "scanning tuning is synthesized from Phase 4"
    return None


@dataclass
class RecordingResult:
    name: str
    sensor_id: str
    data_path: Path
    meta_path: Path
    n_samples: int
    bytes: int
    rows: int
    annotations: int
    clipped: int


def _waveform_params(plan: RFPlan, emitter_idx: int) -> dict:
    em = plan.emitters[emitter_idx]
    params = {}
    for comp in em.params["activity"]:
        if "lora" in comp.params:
            params["lora"] = dict(comp.params["lora"])
    return params


def build_jobs(plan: RFPlan, out_dir: Path) -> tuple[list[RecordingJob], list[dict]]:
    """One job per (supported sensor, band); returns jobs and the per-job annotation rows."""
    rfs = plan.cfg.rf
    links = plan.links
    em = plan.emissions
    w0, w1 = plan.window
    jobs, job_rows = [], []

    for si, sensor in enumerate(plan.sensors):
        if unsupported_reason(sensor):
            continue
        for bi, band in enumerate(sensor.tuning.bands):
            sel = (links["sensor_idx"] == si) & (links["band_idx"] == bi)
            idx = links["emission_idx"][sel]
            emitter_idx = em.emitter_idx[idx]
            infos = {}
            trajectories = {}
            for ei in np.unique(emitter_idx).tolist():
                e = plan.emitters[ei]
                prof = rfs.library[e.profile_id]
                pos = (e.position.x, e.position.y, e.position.z) if e.position else None
                infos[ei] = EmitterInfo(prof.waveform, _waveform_params(plan, ei), pos, e.object_id)
                if pos is None:
                    trajectories[e.object_id] = plan.trajectories[e.object_id]

            name = f"{sensor.sensor_id}_{band.name}"
            path = out_dir / sensor.sensor_id / f"{name}.sigmf-data"
            receiver = build_receiver(sensor.receiver, band.sample_rate)
            n_samples = int(round((w1 - w0) * band.sample_rate))
            tau_mid = links["range_m"][sel] / C_MPS
            job = RecordingJob(
                name=name,
                path=str(path),
                sensor_id=sensor.sensor_id,
                band_name=band.name,
                sensor_pos_km=(sensor.position.x, sensor.position.y, sensor.position.z),
                rx_gain_dbi=sensor.rx_gain_dbi,
                center_hz=band.center_hz,
                sample_rate=band.sample_rate,
                t0=w0,
                n_samples=n_samples,
                chunk_samples=rfs.rf.chunk_samples,
                receiver=receiver,
                scenario_seed=plan.cfg.seed,
                emission_idx=idx,
                emitter_idx=emitter_idx,
                t_start=em.t_start[idx],
                duration=em.duration[idx],
                f_center_hz=em.f_center_hz[idx],
                bandwidth_hz=em.bandwidth_hz[idx],
                eirp_dbm=em.eirp_dbm[idx],
                seed=em.seed[idx],
                tau_mid=tau_mid,
                emitters=infos,
                trajectories=trajectories,
            )
            job.build_index()
            jobs.append(job)
            job_rows.append(
                {
                    "sensor": sensor,
                    "rows": {
                        "emission_idx": idx,
                        "emitter_idx": emitter_idx,
                        "t_start": job.t_start,
                        "duration": job.duration,
                        "f_center_hz": job.f_center_hz,
                        "bandwidth_hz": job.bandwidth_hz,
                        "eirp_dbm": job.eirp_dbm,
                        "tau_mid": tau_mid,
                        "x_km": plan.states.pos_km[idx, 0],
                        "y_km": plan.states.pos_km[idx, 1],
                        "z_km": plan.states.pos_km[idx, 2],
                        **{
                            k: links[k][sel]
                            for k in (
                                "snr_db",
                                "rx_power_dbm",
                                "range_m",
                                "doppler_hz",
                                "azimuth_deg",
                                "elevation_deg",
                                "detectable",
                            )
                        },
                    },
                }
            )
    return jobs, job_rows


_JOBS: list[RecordingJob] = []


def _init_worker(jobs: list[RecordingJob]) -> None:
    global _JOBS
    _JOBS = jobs


def _work(task: tuple[int, int]) -> tuple[int, dict]:
    j, c = task
    return j, render_chunk(_JOBS[j], c)


def synthesize(
    plan: RFPlan,
    out_dir: Path,
    workers: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[RecordingResult], dict]:
    """Render every supported recording and write its SigMF metadata and sensor collections."""
    rfs = plan.cfg.rf
    t_build = time.perf_counter()
    jobs, job_rows = build_jobs(plan, out_dir)
    for job in jobs:
        Path(job.path).parent.mkdir(parents=True, exist_ok=True)
        with open(job.path, "wb") as f:
            f.truncate(2 * job.n_samples * job.receiver.numpy_dtype.itemsize)
    build_s = time.perf_counter() - t_build

    tasks = [(j, c) for j, job in enumerate(jobs) for c in range(job.n_chunks)]
    stats = [{"rows": 0, "clipped": 0} for _ in jobs]
    workers = max(1, workers or os.cpu_count() or 1)
    t_render = time.perf_counter()

    def collect(j: int, res: dict) -> None:
        stats[j]["clipped"] += res["clipped"]
        if progress:
            job = jobs[j]
            n0 = res["chunk"] * job.chunk_samples
            progress(min(job.chunk_samples, job.n_samples - n0))

    if workers == 1 or len(tasks) <= 1:
        _init_worker(jobs)
        for task in tasks:
            collect(*_work(task))
    else:
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_init_worker, initargs=(jobs,)
        ) as pool:
            for j, res in pool.map(_work, tasks, chunksize=1):
                collect(j, res)
    render_s = time.perf_counter() - t_render

    results = []
    by_sensor: dict[str, list[Path]] = {}
    for job, info, st in zip(jobs, job_rows, stats):
        sensor = info["sensor"]
        annotations = build_annotations(
            info["rows"],
            granularity=rfs.rf.annotation_granularity,
            max_gap_s=rfs.rf.annotation_max_gap_s,
            sample_rate=job.sample_rate,
            t0=job.t0,
            n_samples=job.n_samples,
            emitters=plan.emitters,
            library=rfs.library,
            origin=plan.cfg.reference_origin,
        )
        data_path = Path(job.path)
        meta = build_meta(
            sensor=sensor,
            band_name=job.band_name,
            center_hz=job.center_hz,
            sample_rate=job.sample_rate,
            t0=job.t0,
            epoch=rfs.rf.epoch,
            receiver=job.receiver,
            annotations=annotations,
            clipped=st["clipped"],
            scenario_name=plan.cfg.name,
            seed=plan.cfg.seed,
            window=plan.window,
            origin=plan.cfg.reference_origin,
        )
        meta_path = data_path.with_suffix(".sigmf-meta")
        write_json(meta_path, meta)
        by_sensor.setdefault(sensor.sensor_id, []).append(meta_path)
        results.append(
            RecordingResult(
                name=job.name,
                sensor_id=sensor.sensor_id,
                data_path=data_path,
                meta_path=meta_path,
                n_samples=job.n_samples,
                bytes=data_path.stat().st_size,
                rows=len(job.t_start),
                annotations=len(annotations),
                clipped=st["clipped"],
            )
        )
    sensors = {s.sensor_id: s for s in plan.sensors}
    for sid, metas in by_sensor.items():
        write_collection(out_dir / sid / f"{sid}.sigmf-collection", sensors[sid], metas)

    total_samples = sum(job.n_samples for job in jobs)
    timing = {
        "build_s": build_s,
        "render_s": render_s,
        "workers": workers,
        "samples": total_samples,
        "msps": total_samples / render_s / 1e6 if render_s > 0 else 0.0,
    }
    return results, timing
