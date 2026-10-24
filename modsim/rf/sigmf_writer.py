"""Stage C: SigMF metadata (``.sigmf-meta``) and per-sensor collections (``.sigmf-collection``)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .. import __version__
from ..geo import km_to_latlon
from .array import element_positions_local, relative_azimuth
from .models import AnnotationGranularity, ArrayConfig, Emitter, LinkProfile, RFSensorConfig
from .waveforms import fidelity

SIGMF_VERSION = "1.2.6"
EXTENSIONS = [
    {"name": "signal", "version": "1.0.0", "optional": True},
    {"name": "swarmsim", "version": "0.1.0", "optional": True},
]
SPATIAL_EXTENSION = {"name": "spatial", "version": "1.1.0", "optional": True}


def _element_geometry(arr: ArrayConfig) -> list[dict]:
    return [{"point": [round(float(v), 6) for v in p]} for p in element_positions_local(arr)]


def iso_utc(epoch: str, offset_s: float) -> str:
    base = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    return (base + timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _geojson(x_km, y_km, z_km, origin) -> dict | None:
    if origin is None:
        return None
    lat, lon = km_to_latlon(x_km, y_km, origin)
    return {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7), round(z_km * 1000, 2)]}


def build_annotations(
    rows: dict[str, np.ndarray],
    *,
    granularity: AnnotationGranularity,
    max_gap_s: float,
    sample_rate: float,
    t0: float,
    n_samples: int,
    emitters: list[Emitter],
    library: dict[str, LinkProfile],
    origin: tuple[float, float] | None,
    array: ArrayConfig | None = None,
) -> list[dict]:
    """Annotations for one recording, grouped per ``granularity`` and sorted by sample_start.

    ``rows`` holds one entry per rendered emission (link-table columns plus emission fields).
    For array sensors, ``spatial:signal_bearing`` gives the bearing truth relative to the
    aperture boresight (SigMF ``spatial`` convention) and ``spatial:emitter_location`` the
    emitter position.
    """
    n = len(rows["t_start"])
    if n == 0:
        return []
    em_idx = rows["emitter_idx"]
    order = np.lexsort((rows["t_start"], em_idx))
    r = {k: v[order] for k, v in rows.items()}
    em_idx = r["emitter_idx"]
    end_t = r["t_start"] + r["duration"]

    new = np.ones(n, dtype=bool)
    new[1:] = em_idx[1:] != em_idx[:-1]
    if granularity == AnnotationGranularity.EMISSION:
        new[:] = True
    elif granularity == AnnotationGranularity.DWELL:
        new[1:] |= r["f_center_hz"][1:] != r["f_center_hz"][:-1]
        new[1:] |= (r["t_start"][1:] - end_t[:-1]) > max_gap_s
    group = np.cumsum(new) - 1
    starts = np.flatnonzero(new)

    rx_start = (r["t_start"] + r["tau_mid"] - t0) * sample_rate
    rx_end = (end_t + r["tau_mid"] - t0) * sample_rate
    s0 = np.clip(np.rint(np.minimum.reduceat(rx_start, starts)), 0, n_samples).astype(np.int64)
    s1 = np.clip(np.rint(np.maximum.reduceat(rx_end, starts)), 0, n_samples).astype(np.int64)
    f_lo = np.minimum.reduceat(r["f_center_hz"] - r["bandwidth_hz"] / 2 + r["doppler_hz"], starts)
    f_hi = np.maximum.reduceat(r["f_center_hz"] + r["bandwidth_hz"] / 2 + r["doppler_hz"], starts)
    snr_max = np.maximum.reduceat(r["snr_db"], starts)
    detectable = np.logical_or.reduceat(r["detectable"], starts)
    count = np.diff(np.append(starts, n))
    # Representative row per group: the one with the highest SNR
    best = np.lexsort((-r["snr_db"], group))[starts]

    annotations = []
    for g in range(len(starts)):
        if s1[g] <= s0[g]:
            continue
        i = best[g]
        em = emitters[int(em_idx[i])]
        prof = library[em.profile_id]
        detail = {**prof.signal, "channel_bw": float(r["bandwidth_hz"][i])}
        emitter = {
            "seid": int(em_idx[i]),
            "label": em.emitter_id,
            "power_eirp": round(float(r["eirp_dbm"][i]), 2),
        }
        geo = _geojson(float(r["x_km"][i]), float(r["y_km"][i]), float(r["z_km"][i]), origin)
        if geo:
            emitter["geolocation"] = geo
        first, last = (
            int(r["emission_idx"][starts[g]]),
            int(r["emission_idx"][starts[g] + count[g] - 1]),
        )
        ann = {
            "core:sample_start": int(s0[g]),
            "core:sample_count": int(s1[g] - s0[g]),
            "core:freq_lower_edge": round(float(f_lo[g]), 1),
            "core:freq_upper_edge": round(float(f_hi[g]), 1),
            "core:label": em.profile_id,
            "signal:detail": detail,
            "signal:emitter": emitter,
            "swarmsim:role": em.kind.value,
            "swarmsim:object_id": em.object_id,
            "swarmsim:emission_count": int(count[g]),
            "swarmsim:emissions": [first, last],
            "swarmsim:snr_db": round(float(snr_max[g]), 2),
            "swarmsim:rx_power_dbm": round(float(r["rx_power_dbm"][i]), 2),
            "swarmsim:range_m": round(float(r["range_m"][i]), 2),
            "swarmsim:doppler_hz": round(float(r["doppler_hz"][i]), 3) + 0.0,
            "swarmsim:azimuth_deg": round(float(r["azimuth_deg"][i]), 3),
            "swarmsim:elevation_deg": round(float(r["elevation_deg"][i]), 3),
            "swarmsim:detectable": bool(detectable[g]),
            "swarmsim:waveform_fidelity": fidelity(prof.waveform),
        }
        for key in ("two_ray_db", "shadow_db", "fade_db"):
            if key in r and r[key][i] != 0:
                ann[f"swarmsim:{key}"] = round(float(r[key][i]), 2)
        if array is not None:
            ann["spatial:signal_bearing"] = {
                "azimuth": round(float(relative_azimuth(array, r["azimuth_deg"][i])), 3),
                "elevation": round(float(r["elevation_deg"][i]), 3),
                "range": round(float(r["range_m"][i]), 2),
                "range_rate": round(float(r["radial_velocity_mps"][i]), 3) + 0.0,
            }
            if geo:
                ann["spatial:emitter_location"] = geo
        annotations.append(ann)
    annotations.sort(key=lambda a: a["core:sample_start"])
    return annotations


def build_meta(
    *,
    sensor: RFSensorConfig,
    band_name: str,
    center_hz: float,
    sample_rate: float,
    t0: float,
    epoch: str,
    receiver,
    annotations: list[dict],
    clipped: int,
    scenario_name: str,
    seed: int | None,
    window: tuple[float, float],
    origin: tuple[float, float] | None,
    channel_gain: np.ndarray | None = None,
    captures: list[tuple[int, float]] | None = None,
    retunes: list[tuple[int, int]] | None = None,
    impairments: dict | None = None,
    propagation=None,
) -> dict:
    """Global, captures and annotations for one recording.

    ``captures`` lists (sample_start, frequency) per capture segment (one per dwell slot when
    scanning); ``retunes`` lists zero-filled retune spans, annotated as ``swarmsim:retune``.
    """
    p = sensor.position
    arr = sensor.array
    glob = {
        "core:datatype": sensor.receiver.sample_format,
        "core:sample_rate": float(sample_rate),
        "core:version": SIGMF_VERSION,
        "core:num_channels": sensor.num_channels,
        "core:recorder": f"swarmsim {__version__}",
        "core:hw": (
            f"Synthetic {sensor.sensor_type.value} RF sensor "
            f"(NF {sensor.receiver.noise_figure_db:g} dB, gain {sensor.rx_gain_dbi:g} dBi)"
        ),
        "core:description": f"{scenario_name}: {sensor.sensor_id} {band_name}",
        "core:extensions": EXTENSIONS + ([SPATIAL_EXTENSION] if arr else []),
        "swarmsim:sensor_id": sensor.sensor_id,
        "swarmsim:band": band_name,
        "swarmsim:scenario": scenario_name,
        "swarmsim:seed": seed,
        "swarmsim:window_s": list(window),
        "swarmsim:noise_power_dbm": round(receiver.noise_power_dbm, 3),
        "swarmsim:full_scale_dbm": round(receiver.full_scale_dbm, 3),
        "swarmsim:adc_bits": receiver.adc_bits,
        "swarmsim:clipped_components": clipped,
    }
    glob.update(impairments or {})
    if propagation is not None:
        glob["swarmsim:propagation"] = {
            "ground_reflection": propagation.ground_reflection,
            "shadowing_db": propagation.shadowing_db,
            "fading": propagation.fading,
            "rician_k_db": list(propagation.rician_k_db),
        }
    geo = _geojson(p.x, p.y, p.z, origin)
    if geo:
        glob["core:geolocation"] = geo
    capture_list = []
    for start, freq in captures or [(0, center_hz)]:
        capture_list.append(
            {
                "core:sample_start": int(start),
                "core:frequency": float(freq),
                "core:datetime": iso_utc(epoch, t0 + start / sample_rate),
            }
        )
    capture = capture_list[0]
    if retunes:
        annotations = sorted(
            annotations
            + [
                {
                    "core:sample_start": int(a),
                    "core:sample_count": int(b - a),
                    "core:label": "retune",
                    "swarmsim:retune": True,
                }
                for a, b in retunes
            ],
            key=lambda x: x["core:sample_start"],
        )
    if arr:
        glob["spatial:num_elements"] = arr.elements
        glob["spatial:channel_index"] = 0
        glob["swarmsim:element_pattern"] = arr.element_pattern
        if channel_gain is not None:
            # Calibration truth: the per-channel gain/phase error applied before the ADC
            glob["swarmsim:channel_gain_db"] = [
                round(float(v), 4) for v in 20 * np.log10(np.abs(channel_gain))
            ]
            glob["swarmsim:channel_phase_deg"] = [
                round(float(v), 4) for v in np.degrees(np.angle(channel_gain))
            ]
        for cap in capture_list:
            cap["spatial:aperture_azimuth"] = float(arr.boresight_deg)
        capture["spatial:element_geometry"] = _element_geometry(arr)
    return {"global": glob, "captures": capture_list, "annotations": annotations}


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=1))


def write_collection(path: Path, sensor: RFSensorConfig, meta_paths: list[Path]) -> None:
    streams = [
        {
            "name": m.name.removesuffix(".sigmf-meta"),
            "hash": hashlib.sha512(m.read_bytes()).hexdigest(),
        }
        for m in meta_paths
    ]
    collection = {
        "core:version": SIGMF_VERSION,
        "core:description": f"swarmsim RF sensor {sensor.sensor_id}",
        "core:streams": streams,
    }
    if sensor.array is not None:
        collection["core:extensions"] = [SPATIAL_EXTENSION]
        collection["spatial:element_geometry"] = _element_geometry(sensor.array)
    write_json(path, {"collection": collection})
