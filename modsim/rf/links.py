"""Link budget of every (emission, sensor, band) pair the synthesis stage will render.

Phase 0 model: free-space path loss, smooth-earth radio horizon, thermal noise from the
receiver noise figure, flat-spectrum band overlap, and dwell-time overlap for scanning sensors.
Two-ray, fading and receiver impairments arrive in later phases (see docs/design/rf-sigmf.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .array import mean_pattern_gain_db
from .geometry import Trajectory, relative_geometry
from .models import EmissionTable, Emitter, RFSensorConfig, TuningMode
from .propagation import doppler_hz, fspl_db, noise_floor_dbm, radio_horizon_m


@dataclass
class EmitterStates:
    """Emitter position and velocity at each emission's mid-time (aligned with the table)."""

    pos_km: np.ndarray  # (N, 3)
    vel_kmps: np.ndarray  # (N, 3)


def emitter_states(
    emissions: EmissionTable, emitters: list[Emitter], trajectories: dict[str, Trajectory]
) -> EmitterStates:
    n = len(emissions)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    if n == 0:
        return EmitterStates(pos, vel)
    mid = emissions.t_start + emissions.duration / 2
    idx = emissions.emitter_idx
    # The scheduler emits each emitter's rows contiguously; sort defensively anyway
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    bounds = np.flatnonzero(np.diff(sorted_idx)) + 1
    for rows in np.split(order, bounds):
        em = emitters[int(idx[rows[0]])]
        if em.position is not None:
            pos[rows] = (em.position.x, em.position.y, em.position.z)
        else:
            p, v = trajectories[em.object_id].state_at(mid[rows])
            pos[rows], vel[rows] = p, v
    return EmitterStates(pos, vel)


LINK_COLUMNS = (
    "emission_idx",
    "sensor_idx",
    "band_idx",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "radial_velocity_mps",
    "doppler_hz",
    "path_loss_db",
    "band_fraction",
    "observed_fraction",
    "rx_power_dbm",
    "noise_dbm",
    "snr_db",
    "detectable",
)


def _scan_observed(t_start, t_end, offset, dwell, cycle) -> np.ndarray:
    """Fraction of [t_start, t_end] that falls inside the periodic dwell windows
    [offset + n·cycle, offset + n·cycle + dwell)."""

    def covered(t):
        x = t - offset
        return np.floor(x / cycle) * dwell + np.clip(np.mod(x, cycle), 0.0, dwell)

    span = np.maximum(t_end - t_start, 1e-12)
    return np.clip((covered(t_end) - covered(t_start)) / span, 0.0, 1.0)


def scan_slots(sensor: RFSensorConfig) -> tuple[list[tuple[int, float]], float]:
    """(band_idx, dwell start offset) for each slot of a scanning cycle, plus the cycle length.

    Each slot is ``retune_ms`` of dead time (zero-filled in the recording) followed by
    ``dwell_ms`` on the band.
    """
    t = sensor.tuning
    names = [b.name for b in t.bands]
    order = t.schedule or names
    slot = (t.retune_ms + t.dwell_ms) / 1000.0
    return [
        (names.index(n), i * slot + t.retune_ms / 1000.0) for i, n in enumerate(order)
    ], slot * len(order)


def compute_links(
    emissions: EmissionTable,
    states: EmitterStates,
    sensors: list[RFSensorConfig],
    window: tuple[float, float],
    cull_snr_db: float,
    k_factor: float,
) -> dict[str, np.ndarray]:
    """Columnar link table, keeping only rows above ``cull_snr_db`` that the sensor observes."""
    out: dict[str, list[np.ndarray]] = {c: [] for c in LINK_COLUMNS}
    n = len(emissions)
    if n == 0 or not sensors:
        return {c: np.empty(0) for c in LINK_COLUMNS}

    f = emissions.f_center_hz
    bw = emissions.bandwidth_hz
    f_lo, f_hi = f - bw / 2, f + bw / 2
    t_s = emissions.t_start
    t_e = t_s + emissions.duration
    all_idx = np.arange(n)

    for si, sensor in enumerate(sensors):
        sp = np.array([sensor.position.x, sensor.position.y, sensor.position.z])
        geo = relative_geometry(states.pos_km, states.vel_kmps, sp)
        horizon = radio_horizon_m(states.pos_km[:, 2] * 1000.0, sp[2] * 1000.0, k_factor)
        visible = geo["ground_range_m"] <= horizon

        if sensor.tuning.mode == TuningMode.SCANNING:
            slots, cycle = scan_slots(sensor)
            dwell = sensor.tuning.dwell_ms / 1000.0
        for bi, band in enumerate(sensor.tuning.bands):
            overlap = np.minimum(f_hi, band.high_hz) - np.maximum(f_lo, band.low_hz)
            frac = np.where(bw > 0, np.clip(overlap / np.maximum(bw, 1e-9), 0.0, 1.0), 0.0)
            if sensor.tuning.mode == TuningMode.SCANNING:
                observed = np.zeros(n)
                for slot_band, offset in slots:
                    if slot_band == bi:
                        observed += _scan_observed(
                            t_s - window[0], t_e - window[0], offset, dwell, cycle
                        )
            else:
                observed = np.ones(n)

            cand = visible & (frac > 0) & (observed > 0)
            if not cand.any():
                continue
            idx = all_idx[cand]
            pl = fspl_db(geo["range_m"][idx], f[idx])
            rx = (
                emissions.eirp_dbm[idx].astype(np.float64)
                - pl
                + sensor.rx_gain_dbi
                + 10.0 * np.log10(frac[idx])
            )
            if sensor.array is not None and sensor.array.element_pattern != "isotropic":
                # Per-element SNR: element pattern averaged over the array toward the emitter
                rx = rx + mean_pattern_gain_db(
                    sensor.array, geo["azimuth_deg"][idx], geo["elevation_deg"][idx]
                )
            noise = noise_floor_dbm(np.maximum(overlap[idx], 1.0), sensor.receiver.noise_figure_db)
            snr = rx - noise
            keep = snr >= cull_snr_db
            idx, pl, rx, noise, snr = idx[keep], pl[keep], rx[keep], noise[keep], snr[keep]
            if len(idx) == 0:
                continue
            vr = geo["radial_velocity_mps"][idx]
            out["emission_idx"].append(idx)
            out["sensor_idx"].append(np.full(len(idx), si))
            out["band_idx"].append(np.full(len(idx), bi))
            out["range_m"].append(geo["range_m"][idx])
            out["azimuth_deg"].append(geo["azimuth_deg"][idx])
            out["elevation_deg"].append(geo["elevation_deg"][idx])
            out["radial_velocity_mps"].append(vr)
            out["doppler_hz"].append(doppler_hz(f[idx], vr))
            out["path_loss_db"].append(pl)
            out["band_fraction"].append(frac[idx])
            out["observed_fraction"].append(observed[idx])
            out["rx_power_dbm"].append(rx)
            out["noise_dbm"].append(noise)
            out["snr_db"].append(snr)
            out["detectable"].append(snr >= sensor.detector.threshold_snr_db)

    dtypes = {
        "emission_idx": np.int64,
        "sensor_idx": np.int32,
        "band_idx": np.int32,
        "detectable": bool,
    }
    return {
        c: np.concatenate(v).astype(dtypes.get(c, np.float64))
        if v
        else np.empty(0, dtype=dtypes.get(c, np.float64))
        for c, v in out.items()
    }
