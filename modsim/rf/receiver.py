"""Receiver model: thermal noise, gain to ADC full scale, quantization and clipping.

Sample values are carried in sqrt(mW) until quantization, so |x|^2 is power in mW at the
antenna port (after receive antenna gain). Full scale (|x| = 1 after scaling) corresponds to
``full_scale_dbm``. By default that is set ``headroom_db`` above the noise power in the
recorded bandwidth, as a fixed-gain front end would be configured. Integer samples are
LSB-justified: an ``adc_bits``-bit converter in a 16-bit container spans ±(2^(adc_bits-1) - 1).

Phase 1 implements thermal noise and the ADC. LO offset, phase noise, IQ imbalance, DC offset
and AGC are configured today but applied from Phase 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ReceiverConfig
from .propagation import noise_floor_dbm
from .sizing import parse_datatype


@dataclass
class ReceiverChain:
    sample_rate: float
    noise_power_dbm: float
    full_scale_dbm: float
    sample_format: str
    adc_bits: int | None  # None for unquantized float output
    numpy_dtype: np.dtype  # component dtype (I and Q interleaved)

    @property
    def noise_mw(self) -> float:
        return 10 ** (self.noise_power_dbm / 10)

    @property
    def full_scale_mw(self) -> float:
        return 10 ** (self.full_scale_dbm / 10)

    def add_noise(self, buf: np.ndarray, rng: np.random.Generator) -> None:
        """Add complex thermal noise in place to a complex64 buffer (samples × channels)."""
        sigma = np.float32(np.sqrt(self.noise_mw / 2))
        iq = buf.view(np.float32)
        iq += sigma * rng.standard_normal(iq.shape, dtype=np.float32)

    def digitize(self, x: np.ndarray) -> tuple[np.ndarray, int]:
        """Scale to full scale, quantize and clip; returns interleaved I/Q and clip count."""
        iq = (x * np.float32(1 / np.sqrt(self.full_scale_mw))).view(np.float32)
        if self.adc_bits is None:
            return iq.astype(self.numpy_dtype), 0

        kind = self.numpy_dtype.kind
        top = 2 ** (self.adc_bits - 1) - 1
        q = np.rint(iq * top)
        clipped = int(np.count_nonzero((q > top) | (q < -top)))
        q = np.clip(q, -top, top)
        if kind == "u":
            q = q + 2 ** (self.adc_bits - 1)
        elif kind == "f":
            q = q / top
        return q.astype(self.numpy_dtype), clipped


def build_receiver(rx: ReceiverConfig, sample_rate: float) -> ReceiverChain:
    _, kind, bits = parse_datatype(rx.sample_format)
    endian = "<" if rx.sample_format.endswith("_le") or bits == 8 else ">"
    dtype = np.dtype(f"{endian}{kind}{bits // 8}")
    noise_dbm = float(noise_floor_dbm(sample_rate, rx.noise_figure_db))
    full_scale = rx.full_scale_dbm if rx.full_scale_dbm is not None else noise_dbm + rx.headroom_db
    adc_bits = rx.adc_bits if rx.adc_bits is not None else (None if kind == "f" else bits)
    return ReceiverChain(
        sample_rate=sample_rate,
        noise_power_dbm=noise_dbm,
        full_scale_dbm=full_scale,
        sample_format=rx.sample_format,
        adc_bits=adc_bits,
        numpy_dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Impairments (Phase 4)
# ---------------------------------------------------------------------------


def phase_noise_boundaries(
    n_chunks: int, chunk_samples: int, n_samples: int, linewidth_hz: float, fs: float, rng
) -> np.ndarray:
    """Wiener phase (rad) at every chunk boundary, drawn sequentially (length n_chunks + 1)."""
    var = 2 * np.pi * linewidth_hz / fs  # per-sample phase increment variance
    lengths = np.minimum(chunk_samples, n_samples - np.arange(n_chunks) * chunk_samples)
    steps = rng.normal(0.0, 1.0, n_chunks) * np.sqrt(var * lengths)
    return np.concatenate([[0.0], np.cumsum(steps)])


def phase_noise_bridge(
    w0: float, w1: float, n: int, linewidth_hz: float, fs: float, rng
) -> np.ndarray:
    """Wiener phase over one chunk as a Brownian bridge from w0 (first sample) to w1 (next
    chunk's first sample), so chunks render independently with exact statistics."""
    sigma = np.sqrt(2 * np.pi * linewidth_hz / fs)
    steps = rng.normal(0.0, sigma, n)
    walk = np.concatenate([[0.0], np.cumsum(steps)])  # length n + 1
    j = np.arange(n + 1) / n
    bridge = w0 + walk - j * (walk[-1] - (w1 - w0))
    return bridge[:n]


def apply_iq_imbalance(buf: np.ndarray, gain_db: float, phase_deg: float) -> None:
    """In place: I' = I, Q' = g·(Q·cos φ − I·sin φ)."""
    g = 10 ** (gain_db / 20)
    phi = np.radians(phase_deg)
    i = buf.real.copy()
    q = buf.imag.copy()
    buf.imag = (g * (q * np.cos(phi) - i * np.sin(phi))).astype(buf.real.dtype)


def agc_gains_db(x_fs: np.ndarray, n0: int, block: int, agc) -> np.ndarray:
    """Per-block AGC gain (dB) for samples already scaled to full scale.

    Blocks are aligned to the absolute sample index (``n0`` must be a multiple of ``block``).
    """
    n = x_fs.shape[0]
    n_blocks = -(-n // block)
    power = np.abs(x_fs.reshape(n, -1)) ** 2
    pad = n_blocks * block - n
    if pad:
        power = np.concatenate([power, np.full((pad, power.shape[1]), np.nan)])
    rms_db = 10 * np.log10(np.maximum(np.nanmean(power.reshape(n_blocks, -1), axis=1), 1e-30))
    gain = agc.step_db * np.round((agc.target_dbfs - rms_db) / agc.step_db)
    return np.clip(gain, agc.min_gain_db, agc.max_gain_db)
