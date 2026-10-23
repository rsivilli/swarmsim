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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.signal import resample_poly

from .geometry import Trajectory
from .propagation import C_MPS, fspl_db
from .receiver import ReceiverChain
from .scheduler import make_rng
from .waveforms import render

GEOMETRY_GRID_S = 1e-3
RAMP_S = 2e-6
ALIAS_LIMIT = 0.45  # fraction of fs an emission may reach before oversampling
DECIMATION_GUARD = 32  # output samples rendered beyond each side when decimating


@dataclass
class EmitterInfo:
    waveform: str
    params: dict  # waveform params (e.g. {"lora": {...}})
    position_km: tuple[float, float, float] | None  # static emitters
    object_id: str | None  # mobile emitters


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
    # CSR index: rows overlapping chunk c are chunk_rows[chunk_ptr[c]:chunk_ptr[c + 1]]
    chunk_ptr: np.ndarray | None = None
    chunk_rows: np.ndarray | None = None
    row_start: np.ndarray | None = None
    row_end: np.ndarray | None = None

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


def _emission_signal(job: RecordingJob, row: int, k: np.ndarray, rate: float, chunk: int):
    """Received baseband of one emission at recording sample indices ``k`` of a ``rate`` grid.

    Returns complex64 samples aligned with ``k`` (zero outside the emission).
    """
    out = np.zeros(len(k), dtype=np.complex64)
    ts = float(job.t_start[row])
    dur = float(job.duration[row])
    fe = float(job.f_center_hz[row])
    ei = int(job.emitter_idx[row])
    em = job.emitters[ei]
    tau_mid = float(job.tau_mid[row])
    off = fe - job.center_hz

    # Geometry on a coarse grid (sample-index axis). Delay, waveform time and carrier cycles
    # are all linear between grid points, so they are interpolated straight to the samples.
    n_grid = max(2, int(math.ceil((k[-1] - k[0]) / rate / GEOMETRY_GRID_S)) + 1)
    kg = np.linspace(float(k[0]), float(k[-1]), n_grid)
    rel_g = kg / rate
    tg = job.t0 + rel_g
    if em.position_km is not None:
        pos = np.broadcast_to(np.asarray(em.position_km), (n_grid, 3))
    else:
        pos, _ = job.trajectories[em.object_id].state_at(tg - tau_mid)
    r_m = np.linalg.norm((pos - np.asarray(job.sensor_pos_km)) * 1000.0, axis=1)
    tau_g = r_m / C_MPS
    u_g = tg - tau_g - ts
    cyc_g = off * rel_g - fe * tau_g
    cyc_g -= np.floor(cyc_g[0])  # only the fractional part matters
    p_dbm = float(job.eirp_dbm[row]) - fspl_db(r_m, fe) + job.rx_gain_dbi
    amp_g = np.sqrt(10 ** (p_dbm / 10))

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
    buf = np.zeros(n1 - n0, dtype=np.complex64)
    start, end = job.row_start, job.row_end

    rows = job.chunk_rows[job.chunk_ptr[chunk] : job.chunk_ptr[chunk + 1]]
    for row in rows:
        a, b = max(n0, int(start[row])), min(n1, int(end[row]))
        if b <= a:
            continue
        reach = abs(job.f_center_hz[row] - job.center_hz) + job.bandwidth_hz[row] / 2
        if reach <= ALIAS_LIMIT * fs:
            buf[a - n0 : b - n0] += _emission_signal(job, row, np.arange(a, b), fs, chunk)
        else:
            m = int(math.ceil(reach / (ALIAS_LIMIT * fs)))
            g = DECIMATION_GUARD
            k = np.arange((a - g) * m, (b + g) * m)
            hi = _emission_signal(job, row, k, fs * m, chunk)
            buf[a - n0 : b - n0] += resample_poly(hi, 1, m)[g : g + (b - a)].astype(np.complex64)

    rng = make_rng(job.scenario_seed, "noise", job.sensor_id, job.band_name, chunk)
    job.receiver.add_noise(buf, rng)
    iq, clipped = job.receiver.digitize(buf)

    data = np.memmap(
        job.path, dtype=job.receiver.numpy_dtype, mode="r+", shape=(2 * job.n_samples,)
    )
    data[2 * n0 : 2 * n1] = iq
    data.flush()
    del data
    return {"recording": job.name, "chunk": chunk, "rows": int(len(rows)), "clipped": clipped}
