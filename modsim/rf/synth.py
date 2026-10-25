"""Stage B: render IQ chunks for one recording.

A recording is one (sensor, band) pair. It is split into fixed-size chunks; each chunk sums
every emission in its link table that overlaps it, adds thermal noise, digitizes, and writes
its own slice of the memory-mapped ``.sigmf-data`` file. Each chunk depends only on
``(recording, chunk index)``, so output is byte-identical for any worker count.

Signal model for an emission at carrier ``f_e``, seen by a sensor tuned to ``f_c``:

    y(t) = A(t) · g(u) · s(u) · exp(j(2π(f_e − f_c)(t − t0) − 2π f_e τ(t) + φ0)),   u = t − τ(t) − t_s

where τ(t) = r(t)/c is the propagation delay, so delay and Doppler (−f_e·dτ/dt) are exact.
A(t) comes from the free-space link budget, g is a raised-cosine on/off ramp, and s is the
waveform. Range and amplitude are evaluated on a 1 ms grid and interpolated. Emissions that
reach past ±0.45·fs are rendered oversampled and decimated with an anti-alias filter, so only
their in-band part lands in the recording.

Array sensors: each element i has its own delay τᵢ(t), computed from its exact world position
(so near field is handled too), and an element-pattern amplitude. If the delay spread across the
aperture is negligible for the emission's bandwidth (2π·B/2·max|τᵢ − τ| < 0.01 rad), the
waveform is rendered once at the array centre and each element gets its exact carrier phase
−2π f_e (τᵢ − τ). Otherwise every element is rendered at its own delay. Channels are interleaved
per sample in the data file, as SigMF multichannel datasets are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

from .array import local_axes, pattern_power
from .channel import (
    doppler_spread_hz,
    image_paths,
    reflection_coefficient,
    rician_gain,
    rician_k_linear,
    shadowing_db,
)
from .geometry import Trajectory
from .models import AGCConfig, ArrayConfig, PropagationConfig
from .propagation import C_MPS, fspl_db
from .receiver import (
    ReceiverChain,
    agc_gains_db,
    apply_iq_imbalance,
    phase_noise_bridge,
)
from .scheduler import make_rng
from .waveforms import render

GEOMETRY_GRID_S = 1e-3
RAMP_S = 2e-6
ALIAS_LIMIT = 0.45  # fraction of fs an emission may reach before oversampling
DECIMATION_GUARD = 32  # output samples rendered beyond each side when decimating
NARROWBAND_RAD = 0.01  # max envelope phase error 2π·(B/2)·Δτ for single-render array steering


@dataclass
class EmitterInfo:
    waveform: str
    params: dict  # waveform params (e.g. {"lora": {...}})
    position_km: tuple[float, float, float] | None  # static emitters
    object_id: str | None  # mobile emitters
    link_seed: int = 0  # channel (shadowing / fading) seed for this emitter-sensor link


@dataclass
class RecordingJob:
    """Everything a worker needs to render one recording (picklable)."""

    name: str
    path: str  # .sigmf-data path
    sensor_id: str
    band_name: str
    sensor_pos_km: tuple[float, float, float]
    rx_gain_dbi: float
    center_hz: float
    sample_rate: float
    t0: float  # recording start (scenario seconds)
    n_samples: int
    chunk_samples: int
    receiver: ReceiverChain
    scenario_seed: int | None
    # Emission rows rendered in this recording (aligned arrays)
    emission_idx: np.ndarray
    emitter_idx: np.ndarray
    t_start: np.ndarray
    duration: np.ndarray
    f_center_hz: np.ndarray
    bandwidth_hz: np.ndarray
    eirp_dbm: np.ndarray
    seed: np.ndarray
    tau_mid: np.ndarray  # propagation delay at emission mid-time (s)
    emitters: dict[int, EmitterInfo]
    trajectories: dict[str, Trajectory]
    # Array sensors: element offsets from the sensor position (m, world ENU) and the array
    # config for element patterns; single-element sensors have one zero offset and no array
    element_offsets_m: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    array: ArrayConfig | None = None
    channel_gain: np.ndarray = field(default_factory=lambda: np.ones(1, dtype=np.complex128))
    prop: PropagationConfig = field(default_factory=PropagationConfig)
    # Tuning: dwell segments (start, end, band) in samples and each band's centre frequency.
    # Fixed/multi recordings have one segment over the whole recording; scanning recordings
    # have one per dwell, and the retune gaps between them stay zero.
    segments: np.ndarray | None = None
    band_centers: list[float] | None = None
    row_band: np.ndarray | None = None  # band index of each row (scanning)
    # Receiver impairments
    lo_offset_frac: float = 0.0  # LO error as a fraction of the tuned frequency
    pn_linewidth_hz: float = 0.0
    pn_boundaries: np.ndarray | None = None  # Wiener phase at chunk boundaries
    iq_gain_db: float = 0.0
    iq_phase_deg: float = 0.0
    dc_offset: np.ndarray | None = None  # per channel, full-scale units
    agc: AGCConfig | None = None
    # CSR index: rows overlapping chunk c are chunk_rows[chunk_ptr[c]:chunk_ptr[c + 1]]
    chunk_ptr: np.ndarray | None = None
    chunk_rows: np.ndarray | None = None
    row_start: np.ndarray | None = None
    row_end: np.ndarray | None = None

    @property
    def n_channels(self) -> int:
        return len(self.element_offsets_m)

    @property
    def n_chunks(self) -> int:
        return max(1, math.ceil(self.n_samples / self.chunk_samples))

    def sample_span(self) -> tuple[np.ndarray, np.ndarray]:
        """First and one-past-last sample index each row touches (clipped to the recording)."""
        start = np.floor((self.t_start + self.tau_mid - self.t0) * self.sample_rate) - 2
        end = (
            np.ceil((self.t_start + self.duration + self.tau_mid - self.t0) * self.sample_rate) + 2
        )
        return (
            np.clip(start, 0, self.n_samples).astype(np.int64),
            np.clip(end, 0, self.n_samples).astype(np.int64),
        )

    def segment_table(self) -> tuple[np.ndarray, list[float], np.ndarray]:
        segments = (
            self.segments
            if self.segments is not None
            else np.array([[0, self.n_samples, 0]], dtype=np.int64)
        )
        centers = self.band_centers or [self.center_hz]
        bands = self.row_band if self.row_band is not None else np.zeros(len(self.t_start), int)
        return segments, centers, bands

    def build_index(self) -> None:
        start, end = self.sample_span()
        self.row_start, self.row_end = start, end
        valid = end > start
        rows = np.flatnonzero(valid)
        c0 = start[rows] // self.chunk_samples
        c1 = (end[rows] - 1) // self.chunk_samples
        span = (c1 - c0 + 1).astype(np.int64)
        rep_rows = np.repeat(rows, span)
        offsets = np.arange(len(rep_rows)) - np.repeat(np.cumsum(span) - span, span)
        rep_chunks = np.repeat(c0, span) + offsets
        order = np.lexsort((rep_rows, rep_chunks))
        self.chunk_rows = rep_rows[order]
        counts = np.bincount(rep_chunks, minlength=self.n_chunks)
        self.chunk_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def _apply_ramp(out: np.ndarray, u: np.ndarray, duration: float, rate: float) -> None:
    """Raised-cosine on/off ramp, applied in place to the samples near either end."""
    ramp = min(RAMP_S, duration / 10)
    n = min(int(math.ceil(ramp * rate)) + 1, len(u))
    for sl in (slice(0, n), slice(len(u) - n, len(u))):
        x = np.clip(np.minimum(u[sl], duration - u[sl]) / ramp, 0.0, 1.0)
        out[sl] *= (0.5 - 0.5 * np.cos(np.pi * x)).astype(np.float32)


def _emission_signal(
    job: RecordingJob,
    row: int,
    k: np.ndarray,
    rate: float,
    chunk: int,
    center_hz: float | None = None,
):
    """Received baseband of one emission at recording sample indices ``k`` of a ``rate`` grid.

    Sums the direct and (optionally) ground-reflected paths, applies shadowing and Rician
    fading. Returns complex64 samples of shape (len(k), n_channels), zero outside the emission.
    """
    fe = float(job.f_center_hz[row])
    em = job.emitters[int(job.emitter_idx[row])]
    tau_mid = float(job.tau_mid[row])
    prop = job.prop
    center_hz = job.center_hz if center_hz is None else center_hz

    # Geometry on a coarse grid (sample-index axis). Delay, waveform time and carrier cycles
    # are all linear between grid points, so they are interpolated straight to the samples.
    n_grid = max(2, int(math.ceil((k[-1] - k[0]) / rate / GEOMETRY_GRID_S)) + 1)
    kg = np.linspace(float(k[0]), float(k[-1]), n_grid)
    t_emit = job.t0 + kg / rate - tau_mid
    if em.position_km is not None:
        pos = np.broadcast_to(np.asarray(em.position_km, dtype=np.float64), (n_grid, 3))
        speed = np.zeros(n_grid)
    else:
        pos, vel = job.trajectories[em.object_id].state_at(t_emit)
        speed = np.linalg.norm(vel, axis=1) * 1000.0
    sensor = np.asarray(job.sensor_pos_km)
    base_db = float(job.eirp_dbm[row]) + job.rx_gain_dbi
    if prop.shadowing_db > 0:
        base_db = base_db + shadowing_db(em.link_seed, speed * t_emit, prop)

    paths = [(pos, None)]
    if prop.ground_reflection:
        _, _, grazing = image_paths(pos, sensor)
        gamma = reflection_coefficient(grazing, fe, prop)
        paths.append((pos * np.array([1.0, 1.0, -1.0]), gamma))

    out = np.zeros((len(k), job.n_channels), dtype=np.complex64)
    for path_pos, gamma in paths:
        rel_m = (path_pos - sensor) * 1000.0  # sensor centre -> (image) emitter
        r_m = np.linalg.norm(rel_m, axis=1)
        amp_g = np.sqrt(10 ** ((base_db - fspl_db(r_m, fe)) / 10))
        extra = None
        if gamma is not None:
            amp_g = amp_g * np.abs(gamma)
            extra = np.unwrap(np.angle(gamma))
        out += _path_signal(job, row, k, rate, chunk, center_hz, kg, rel_m, r_m, amp_g, extra)

    if prop.fading == "rician":
        _apply_fading(job, row, k, rate, out, pos, speed, fe)
    return out


def _apply_fading(job, row, k, rate, out, pos, speed, fe) -> None:
    """Multiply the emission (all channels) by its link's Rician gain, in place."""
    nz = np.flatnonzero(np.any(out != 0, axis=1))
    if len(nz) == 0:
        return
    sl = slice(int(nz[0]), int(nz[-1]) + 1)
    em = job.emitters[int(job.emitter_idx[row])]
    prop = job.prop
    mid = len(pos) // 2
    rel = (pos[mid] - np.asarray(job.sensor_pos_km)) * 1000.0
    elevation = np.degrees(np.arctan2(rel[2], max(np.hypot(rel[0], rel[1]), 1e-9)))
    k_lin = float(rician_k_linear(elevation, prop))
    f_d = float(doppler_spread_hz(speed[mid], fe, prop))
    t = job.t0 + k[sl] / rate
    step = min(GEOMETRY_GRID_S, 1.0 / (20.0 * f_d))
    n_grid = max(2, int(math.ceil((t[-1] - t[0]) / step)) + 1)
    tg = np.linspace(t[0], t[-1], n_grid)
    offsets = job.element_offsets_m if job.n_channels > 1 else None
    h = rician_gain(em.link_seed, tg, f_d, fe, k_lin, prop, offsets)  # (grid, N)
    for i in range(job.n_channels):
        hi = np.empty(len(t), dtype=np.complex64)
        hi.real = np.interp(t, tg, h[:, i].real)
        hi.imag = np.interp(t, tg, h[:, i].imag)
        out[sl, i] *= hi


def _path_signal(job, row, k, rate, chunk, center_hz, kg, rel_m, r_m, amp_g, extra):
    """One propagation path to every channel (narrowband steering or per-element rendering)."""
    tau_g = r_m / C_MPS
    n_ch = job.n_channels
    if n_ch == 1:
        return _render_one(job, row, k, rate, chunk, center_hz, kg, tau_g, amp_g, extra)[:, None]

    out = np.zeros((len(k), n_ch), dtype=np.complex64)
    fe = float(job.f_center_hz[row])
    offsets = job.element_offsets_m
    tau_el = np.linalg.norm(rel_m[:, None, :] - offsets[None, :, :], axis=2) / C_MPS  # (grid, N)
    gain = np.ones(n_ch, dtype=np.float32)
    if job.array is not None and job.array.element_pattern != "isotropic":
        mid = rel_m[len(kg) // 2] / max(r_m[len(kg) // 2], 1e-9)
        d_local = local_axes(job.array.boresight_deg) @ mid
        gain = np.sqrt(pattern_power(job.array, d_local)).astype(np.float32)

    spread = float(np.abs(tau_el - tau_g[:, None]).max())
    if 2 * np.pi * float(job.bandwidth_hz[row]) / 2 * spread < NARROWBAND_RAD:
        ref = _render_one(job, row, k, rate, chunk, center_hz, kg, tau_g, amp_g, extra)
        nz = np.flatnonzero(ref)
        if len(nz) == 0:
            return out
        sl = slice(int(nz[0]), int(nz[-1]) + 1)
        ks = k[sl].astype(np.float64)
        for i in range(n_ch):
            # exact per-element carrier phase; fe·Δτ is small, so float64 is ample
            dphi = -2 * np.pi * fe * np.interp(ks, kg, tau_el[:, i] - tau_g)
            steer = np.empty(len(dphi), dtype=np.complex64)
            steer.real = np.cos(dphi)
            steer.imag = np.sin(dphi)
            out[sl, i] = ref[sl] * steer * gain[i]
    else:
        for i in range(n_ch):
            out[:, i] = (
                _render_one(job, row, k, rate, chunk, center_hz, kg, tau_el[:, i], amp_g, extra)
                * gain[i]
            )
    return out


def _render_one(job, row, k, rate, chunk, center_hz, kg, tau_g, amp_g, extra=None) -> np.ndarray:
    """One channel: the emission received with delay ``tau_g`` (on grid ``kg``).

    ``extra`` is an optional phase (rad) on the grid, e.g. a reflection coefficient's phase.
    """
    out = np.zeros(len(k), dtype=np.complex64)
    ts = float(job.t_start[row])
    dur = float(job.duration[row])
    fe = float(job.f_center_hz[row])
    em = job.emitters[int(job.emitter_idx[row])]
    off = fe - center_hz
    rel_g = kg / rate
    u_g = job.t0 + rel_g - tau_g - ts
    cyc_g = off * rel_g - fe * tau_g
    if extra is not None:
        cyc_g = cyc_g + extra / (2 * np.pi)
    cyc_g -= np.floor(cyc_g[0])  # only the fractional part matters

    # u increases monotonically with k: keep only samples with 0 <= u < duration
    lo = 0 if u_g[0] >= 0 else int(math.ceil(np.interp(0.0, u_g, kg) - k[0]))
    hi = len(k) if u_g[-1] < dur else int(math.ceil(np.interp(dur, u_g, kg) - k[0]))
    lo, hi = max(lo, 0), min(hi, len(k))
    if hi <= lo:
        return out
    sl = slice(lo, hi)
    ks = k[sl].astype(np.float64)
    u = np.interp(ks, kg, u_g)
    cycles = np.interp(ks, kg, cyc_g)
    amp = np.interp(ks, kg, amp_g).astype(np.float32)

    seed = int(job.seed[row])
    kind, s = render(
        em.waveform,
        u,
        bandwidth_hz=float(job.bandwidth_hz[row]),
        duration=dur,
        seed=seed,
        params=em.params,
        sample_rate=rate,
        noise_key=(int(job.emission_idx[row]), chunk),
    )
    phi0 = np.random.default_rng([seed, 0x9A5E]).uniform(0, 2 * np.pi)
    if kind == "phase":
        phase = np.mod(2 * np.pi * cycles + s + phi0, 2 * np.pi).astype(np.float32)
        seg = out[sl]
        seg.real = amp * np.cos(phase)
        seg.imag = amp * np.sin(phase)
    else:
        phase = np.mod(2 * np.pi * cycles + phi0, 2 * np.pi).astype(np.float32)
        carrier = np.empty(len(phase), dtype=np.complex64)
        carrier.real = np.cos(phase)
        carrier.imag = np.sin(phase)
        out[sl] = amp * carrier * s.astype(np.complex64)
    _apply_ramp(out[sl], u, dur, rate)
    return out


def render_chunk(job: RecordingJob, chunk: int) -> dict:
    """Render chunk ``chunk`` of ``job`` into its data file; returns per-chunk statistics."""
    fs = job.sample_rate
    n0 = chunk * job.chunk_samples
    n1 = min(n0 + job.chunk_samples, job.n_samples)
    n = n1 - n0
    n_ch = job.n_channels
    buf = np.zeros((n, n_ch), dtype=np.complex64)
    start, end = job.row_start, job.row_end
    segments, centers, row_band = job.segment_table()

    rows = job.chunk_rows[job.chunk_ptr[chunk] : job.chunk_ptr[chunk + 1]]
    dwell = np.zeros(n, dtype=bool)
    seg_ids = np.flatnonzero((segments[:, 0] < n1) & (segments[:, 1] > n0))
    for si in seg_ids:
        s0, s1, band = max(int(segments[si, 0]), n0), min(int(segments[si, 1]), n1), segments[si, 2]
        dwell[s0 - n0 : s1 - n0] = True
        center = centers[band]
        for row in rows[row_band[rows] == band]:
            a, b = max(s0, int(start[row])), min(s1, int(end[row]))
            if b <= a:
                continue
            reach = abs(job.f_center_hz[row] - center) + job.bandwidth_hz[row] / 2
            if reach <= ALIAS_LIMIT * fs:
                buf[a - n0 : b - n0] += _emission_signal(
                    job, row, np.arange(a, b), fs, chunk, center
                )
            else:
                m = int(math.ceil(reach / (ALIAS_LIMIT * fs)))
                g = DECIMATION_GUARD
                k = np.arange((a - g) * m, (b + g) * m)
                hi = _emission_signal(job, row, k, fs * m, chunk, center)
                dec = resample_poly(hi, 1, m, axis=0)[g : g + (b - a)]
                buf[a - n0 : b - n0] += dec.astype(np.complex64)
        # LO frequency error scales with the tuned frequency (common to all channels)
        if job.lo_offset_frac:
            idx = np.arange(s0, s1, dtype=np.float64)
            cyc = np.mod(-job.lo_offset_frac * center * idx / fs, 1.0)
            rot = np.exp(2j * np.pi * cyc).astype(np.complex64)
            buf[s0 - n0 : s1 - n0] *= rot[:, None]

    rng = make_rng(job.scenario_seed, "noise", job.sensor_id, job.band_name, chunk)
    job.receiver.add_noise(buf, rng)
    if job.pn_linewidth_hz > 0:
        pn_rng = make_rng(job.scenario_seed, "phase_noise", job.sensor_id, job.band_name, chunk)
        theta = phase_noise_bridge(
            job.pn_boundaries[chunk],
            job.pn_boundaries[chunk + 1],
            n,
            job.pn_linewidth_hz,
            fs,
            pn_rng,
        )
        buf *= np.exp(1j * theta).astype(np.complex64)[:, None]
    if n_ch > 1:
        buf *= job.channel_gain.astype(np.complex64)[None, :]  # per-channel calibration error
    if job.iq_gain_db or job.iq_phase_deg:
        apply_iq_imbalance(buf, job.iq_gain_db, job.iq_phase_deg)
    if job.dc_offset is not None:
        buf += (job.dc_offset * np.sqrt(job.receiver.full_scale_mw)).astype(np.complex64)[None, :]
    if not dwell.all():
        buf[~dwell] = 0  # scanning: retune gaps are zero-filled

    agc = None
    if job.agc is not None:
        scaled = buf * np.float32(1 / np.sqrt(job.receiver.full_scale_mw))
        agc = agc_gains_db(scaled, n0, job.agc.block_samples, job.agc)
        per_sample = np.repeat(10 ** (agc / 20), job.agc.block_samples)[:n].astype(np.float32)
        buf *= per_sample[:, None]
    iq, clipped = job.receiver.digitize(buf)

    width = 2 * n_ch
    data = np.memmap(
        job.path, dtype=job.receiver.numpy_dtype, mode="r+", shape=(width * job.n_samples,)
    )
    data[width * n0 : width * n1] = iq.reshape(-1)
    data.flush()
    del data
    return {
        "recording": job.name,
        "chunk": chunk,
        "rows": int(len(rows)),
        "clipped": clipped,
        "agc_db": agc,
    }
