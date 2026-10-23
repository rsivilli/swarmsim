"""Sampling of native-rate waveforms onto a sensor's sample grid.

OFDM is generated naturally by IFFT at its own sample rate (20 MS/s for Wi-Fi, N·15 kHz for
LTE-like numerologies). To place a burst on a sensor grid we apply the burst's fractional start
offset with a windowed-sinc FIR, then convert rates with an exact rational polyphase resampler,
which also band-limits the burst to the sensor's bandwidth.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

FRAC_HALF_TAPS = 32  # windowed-sinc fractional delay: 2·32+1 taps (≈ -50 dB error at 0.83·Nyquist)


def _rate_ratio(out_rate: float, native_rate: float) -> tuple[int, int]:
    r = Fraction(out_rate / native_rate).limit_denominator(100_000)
    if abs(float(r) - out_rate / native_rate) > 1e-9 * out_rate / native_rate:
        raise ValueError(f"No exact rational ratio for {out_rate} / {native_rate}")
    return r.numerator, r.denominator


def _frac_taps(frac: float) -> np.ndarray:
    n = np.arange(-FRAC_HALF_TAPS, FRAC_HALF_TAPS + 1)
    return np.sinc(n - frac) * np.kaiser(len(n), 8.0)


def sample_native(
    native: np.ndarray, native_rate: float, u0: float, n_out: int, out_rate: float
) -> np.ndarray:
    """Samples of a native-rate signal at times ``u0 + j / out_rate`` (j = 0 .. n_out-1).

    ``native[i]`` is the signal at time ``i / native_rate``; it is zero outside the array.
    """
    up, down = _rate_ratio(out_rate, native_rate)
    # Render beyond each end by more than resample_poly's filter half-length
    # (10·max(up, down) taps at the intermediate rate) so its edge transients are cropped off
    guard = math.ceil(10 * max(up, down) / down) + 4
    p0 = (u0 - guard / out_rate) * native_rate  # native position of the first output sample
    i0 = math.floor(p0)
    frac = p0 - i0
    h = FRAC_HALF_TAPS
    length = math.ceil((n_out + 2 * guard) * native_rate / out_rate) + 2
    start = i0 - h
    x = np.zeros(length + 2 * h, dtype=np.complex128)
    lo, hi = max(start, 0), min(start + len(x), len(native))
    if hi > lo:
        x[lo - start : hi - start] = native[lo:hi]
    # y[m] = signal at native position p0 + m
    y = _fir(x, frac)
    z = resample_poly(y, up, down)
    out = z[guard : guard + n_out]
    if len(out) < n_out:
        out = np.concatenate([out, np.zeros(n_out - len(out), dtype=out.dtype)])
    return out


def _fir(x: np.ndarray, frac: float) -> np.ndarray:
    """y[m] = Σ_n x[m + n] · taps[n], i.e. x interpolated at position m + h + frac."""
    taps = _frac_taps(frac)
    return np.convolve(x, taps[::-1], mode="valid")
