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
