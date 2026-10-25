"""IQ synthesis orchestration: recording jobs, parallel chunk rendering, SigMF metadata."""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .array import calibration_errors, element_offsets_world
from .models import RFSensorConfig, TuningMode
from .propagation import C_MPS
from .channel import link_seed
from .links import scan_slots
from .receiver import build_receiver, phase_noise_boundaries
from .runner import RFPlan
from .scheduler import make_rng
from .sigmf_writer import build_annotations, build_meta, write_collection, write_json
from .synth import EmitterInfo, RecordingJob, render_chunk


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
    params = {"waveform": dict(plan.cfg.rf.library[em.profile_id].waveform_params)}
    for comp in em.params["activity"]:
        if "lora" in comp.params:
            params["lora"] = dict(comp.params["lora"])
    return params


def channel_calibration(plan: RFPlan, sensor: RFSensorConfig) -> np.ndarray:
    """Per-channel complex gain error of an array sensor (truth), drawn once per sensor."""
    cm = sensor.receiver.impairments.get("channel_mismatch") or {}
    rng = make_rng(plan.cfg.seed, "calibration", sensor.sensor_id)
    return calibration_errors(
        sensor.num_channels, cm.get("gain_db_sigma", 0.0), cm.get("phase_deg_sigma", 0.0), rng
    )


def _array_fields(plan: RFPlan, sensor: RFSensorConfig) -> dict:
    if sensor.array is None:
        return {}
    return {
        "element_offsets_m": element_offsets_world(sensor.array),
        "array": sensor.array,
        "channel_gain": channel_calibration(plan, sensor),
    }


def _impairment_fields(plan: RFPlan, sensor: RFSensorConfig) -> dict:
    """Per-sensor receiver impairment draws (deterministic from the scenario seed)."""
    imp = sensor.receiver.impairments
    seed, sid = plan.cfg.seed, sensor.sensor_id
    fields: dict = {}
    if imp.get("lo_offset_ppm"):
        rng = make_rng(seed, "lo_offset", sid)
        fields["lo_offset_frac"] = float(rng.uniform(-1.0, 1.0)) * imp["lo_offset_ppm"] * 1e-6
    if imp.get("phase_noise"):
        fields["pn_linewidth_hz"] = float(imp["phase_noise"]["linewidth_hz"])
    if imp.get("iq_imbalance"):
        fields["iq_gain_db"] = float(imp["iq_imbalance"].get("gain_db", 0.0))
        fields["iq_phase_deg"] = float(imp["iq_imbalance"].get("phase_deg", 0.0))
    if imp.get("dc_offset_dbfs") is not None:
        rng = make_rng(seed, "dc_offset", sid)
        phases = rng.uniform(0, 2 * np.pi, sensor.num_channels)
        fields["dc_offset"] = 10 ** (imp["dc_offset_dbfs"] / 20) * np.exp(1j * phases)
    if sensor.receiver.gain_mode == "agc":
        fields["agc"] = sensor.receiver.agc
    return fields


@dataclass
class _RecordingPlan:
    name: str
    band_indices: list[int]
    sample_rate: float
    center_hz: float  # nominal (first band) centre, used for metadata of fixed recordings
    segments: np.ndarray | None  # scanning dwell segments (start, end, band)
    captures: list[tuple[int, int]]  # (sample_start, band index)
    retunes: list[tuple[int, int]]  # zero-filled (start, end) sample ranges


def _recording_plans(sensor: RFSensorConfig, window: tuple[float, float]) -> list[_RecordingPlan]:
    bands = sensor.tuning.bands
    duration = window[1] - window[0]
    if sensor.tuning.mode != TuningMode.SCANNING:
        return [
            _RecordingPlan(
                f"{sensor.sensor_id}_{b.name}", [i], b.sample_rate, b.center_hz, None, [(0, i)], []
            )  # fmt: skip
            for i, b in enumerate(bands)
        ]
    fs = bands[0].sample_rate
    n_samples = int(round(duration * fs))
    slots, cycle = scan_slots(sensor)
    retune = sensor.tuning.retune_ms / 1000.0
    slot_len = cycle / len(slots)
    segs, captures, retunes = [], [], []
    for c in range(int(np.ceil(duration / cycle)) + 1):
        for j, (band, _) in enumerate(slots):
            t_slot = c * cycle + j * slot_len
            s_slot = min(int(round(t_slot * fs)), n_samples)
            s_dwell = min(int(round((t_slot + retune) * fs)), n_samples)
            s_end = min(int(round((t_slot + slot_len) * fs)), n_samples)
            if s_slot >= n_samples:
                break
            captures.append((s_slot, band))
            if s_dwell > s_slot:
                retunes.append((s_slot, s_dwell))
            if s_end > s_dwell:
                segs.append((s_dwell, s_end, band))
    return [
        _RecordingPlan(f"{sensor.sensor_id}_scan", list(range(len(bands))), fs, bands[0].center_hz,
                       np.asarray(segs, dtype=np.int64).reshape(-1, 3), captures, retunes)
    ]  # fmt: skip


def _dwell_rows(rows: dict, segments: np.ndarray, t0: float, fs: float) -> dict:
    """Split scanning rows into the parts that fall inside their band's dwell segments."""
    rx0 = rows["t_start"] + rows["tau_mid"] - t0
    rx1 = rx0 + rows["duration"]
    parts = {k: [] for k in rows}
    for band in np.unique(segments[:, 2]):
        seg = segments[segments[:, 2] == band] / fs  # dwell (start, end) in seconds
        sel = np.flatnonzero(rows["band_idx"] == band)
        if len(sel) == 0:
            continue
        # for each row, the dwells it overlaps: first index via searchsorted on dwell ends
        first = np.searchsorted(seg[:, 1], rx0[sel], side="right")
        last = np.searchsorted(seg[:, 0], rx1[sel], side="left") - 1
        count = np.maximum(last - first + 1, 0)
        rep = np.repeat(sel, count)
        dw = np.repeat(first, count) + (
            np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
        )
        lo = np.maximum(rx0[rep], seg[dw, 0])
        hi = np.minimum(rx1[rep], seg[dw, 1])
        keep = hi > lo
        rep, lo, hi = rep[keep], lo[keep], hi[keep]
        for k in rows:
            parts[k].append(rows[k][rep])
        parts["t_start"][-1] = lo + t0 - rows["tau_mid"][rep]
        parts["duration"][-1] = hi - lo
    return {k: (np.concatenate(v) if v else rows[k][:0]) for k, v in parts.items()}


def _impairment_truth(job: RecordingJob, agc_file: Path | None) -> dict:
    truth: dict = {}
    if job.lo_offset_frac:
        truth["swarmsim:lo_offset_ppm"] = round(job.lo_offset_frac * 1e6, 6)
        truth["swarmsim:lo_offset_hz"] = round(job.lo_offset_frac * job.center_hz, 3)
    if job.pn_linewidth_hz:
        truth["swarmsim:phase_noise_linewidth_hz"] = job.pn_linewidth_hz
    if job.iq_gain_db or job.iq_phase_deg:
        truth["swarmsim:iq_imbalance"] = {"gain_db": job.iq_gain_db, "phase_deg": job.iq_phase_deg}
    if job.dc_offset is not None:
        truth["swarmsim:dc_offset_dbfs"] = [
            round(float(v), 3) for v in 20 * np.log10(np.abs(job.dc_offset))
        ]
    if job.agc is not None and agc_file is not None:
        truth["swarmsim:agc_trace"] = agc_file.name
        truth["swarmsim:agc_block_samples"] = job.agc.block_samples
        truth["swarmsim:agc_target_dbfs"] = job.agc.target_dbfs
    return truth


def build_jobs(plan: RFPlan, out_dir: Path) -> tuple[list[RecordingJob], list[dict]]:
    """One job per recording; returns jobs and the per-job annotation inputs."""
    rfs = plan.cfg.rf
    links = plan.links
    em = plan.emissions
    w0, w1 = plan.window
    jobs, job_rows = [], []

    for si, sensor in enumerate(plan.sensors):
        impairments = _impairment_fields(plan, sensor)
        for rp in _recording_plans(sensor, plan.window):
            sel = (links["sensor_idx"] == si) & np.isin(links["band_idx"], rp.band_indices)
            idx = links["emission_idx"][sel]
            emitter_idx = em.emitter_idx[idx]
            infos = {}
            trajectories = {}
            for ei in np.unique(emitter_idx).tolist():
                e = plan.emitters[ei]
                prof = rfs.library[e.profile_id]
                pos = (e.position.x, e.position.y, e.position.z) if e.position else None
                infos[ei] = EmitterInfo(
                    prof.waveform,
                    _waveform_params(plan, ei),
                    pos,
                    e.object_id,
                    link_seed(plan.cfg.seed, e.emitter_id, sensor.sensor_id),
                )
                if pos is None:
                    trajectories[e.object_id] = plan.trajectories[e.object_id]

            receiver = build_receiver(sensor.receiver, rp.sample_rate)
            n_samples = int(round((w1 - w0) * rp.sample_rate))
            n_chunks = max(1, -(-n_samples // rfs.rf.chunk_samples))
            fields = dict(impairments)
            if fields.get("pn_linewidth_hz"):
                fields["pn_boundaries"] = phase_noise_boundaries(
                    n_chunks, rfs.rf.chunk_samples, n_samples, fields["pn_linewidth_hz"],
                    rp.sample_rate, make_rng(plan.cfg.seed, "phase_noise_walk", sensor.sensor_id, rp.name),
                )  # fmt: skip
            tau_mid = links["range_m"][sel] / C_MPS
            band_idx = links["band_idx"][sel]
            scanning = rp.segments is not None
            job = RecordingJob(
                name=rp.name,
                path=str(out_dir / sensor.sensor_id / f"{rp.name}.sigmf-data"),
                sensor_id=sensor.sensor_id,
                band_name=rp.name if scanning else sensor.tuning.bands[rp.band_indices[0]].name,
                sensor_pos_km=(sensor.position.x, sensor.position.y, sensor.position.z),
                rx_gain_dbi=sensor.rx_gain_dbi,
                center_hz=rp.center_hz,
                sample_rate=rp.sample_rate,
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
                prop=rfs.rf.propagation,
                segments=rp.segments,
                band_centers=[b.center_hz for b in sensor.tuning.bands] if scanning else None,
                row_band=band_idx if scanning else None,
                **_array_fields(plan, sensor),
                **fields,
            )
            job.build_index()
            jobs.append(job)
            rows = {
                "emission_idx": idx,
                "emitter_idx": emitter_idx,
                "band_idx": band_idx,
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
                        "radial_velocity_mps",
                        "two_ray_db",
                        "shadow_db",
                        "fade_db",
                        "detectable",
                    )
                },
            }
            if scanning:
                rows = _dwell_rows(rows, rp.segments, w0, rp.sample_rate)
            job_rows.append({"sensor": sensor, "rows": rows, "plan": rp})
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
            f.truncate(2 * job.n_channels * job.n_samples * job.receiver.numpy_dtype.itemsize)
    build_s = time.perf_counter() - t_build

    tasks = [(j, c) for j, job in enumerate(jobs) for c in range(job.n_chunks)]
    stats = [{"rows": 0, "clipped": 0, "agc": {}} for _ in jobs]
    workers = max(1, workers or os.cpu_count() or 1)
    t_render = time.perf_counter()

    def collect(j: int, res: dict) -> None:
        stats[j]["clipped"] += res["clipped"]
        if res.get("agc_db") is not None:
            stats[j]["agc"][res["chunk"]] = res["agc_db"]
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
            array=sensor.array,
        )
        data_path = Path(job.path)
        rp = info["plan"]
        agc_file = None
        if st["agc"]:
            trace = np.concatenate([st["agc"][c] for c in sorted(st["agc"])]).astype(np.float32)
            agc_file = data_path.with_suffix(".agc.npy")
            np.save(agc_file, trace)
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
            channel_gain=job.channel_gain if job.n_channels > 1 else None,
            captures=[(start, sensor.tuning.bands[b].center_hz) for start, b in rp.captures],
            retunes=rp.retunes,
            impairments=_impairment_truth(job, agc_file),
            propagation=rfs.rf.propagation,
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

    total_samples = sum(job.n_samples * job.n_channels for job in jobs)  # all channels
    timing = {
        "build_s": build_s,
        "render_s": render_s,
        "workers": workers,
        "samples": total_samples,
        "msps": total_samples / render_s / 1e6 if render_s > 0 else 0.0,
    }
    return results, timing
