"""Sampling of native-rate waveforms onto a sensor's sample grid.

OFDM is generated naturally by IFFT at its own sample rate (20 MS/s for Wi-Fi, N·15 kHz for
LTE-like numerologies). To place a burst on a sensor grid we apply the burst's fractional start
offset with a windowed-sinc FIR, then convert rates with an exact rational polyphase resampler,
which also band-limits the burst to the sensor's bandwidth.

When the caller gives the signal's occupied half-bandwidth and the frequency shift it will apply
afterwards, a burst that the sensor's band edge cuts through is band-limited in a single step with
a complex bandpass (see ``sample_native``) instead of being rendered oversampled and decimated.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from scipy.signal import firwin, kaiserord, resample_poly

FRAC_HALF_TAPS = 32  # windowed-sinc fractional delay: 2·32+1 taps (≈ -50 dB error at 0.83·Nyquist)
REJECTION_DB = 60.0

_FILTERS: dict[tuple, np.ndarray | None] = {}
EDGE_TRANSITION = (
    0.1  # transition width (fraction of the output rate) where a band edge cuts a signal
)


def _rate_ratio(out_rate: float, native_rate: float) -> tuple[int, int]:
    r = Fraction(out_rate / native_rate).limit_denominator(100_000)
    if abs(float(r) - out_rate / native_rate) > 1e-9 * out_rate / native_rate:
        raise ValueError(f"No exact rational ratio for {out_rate} / {native_rate}")
    return r.numerator, r.denominator


def _edge_filter(
    up: int, native_rate: float, out_rate: float, lo: float, hi: float, cut_lo: bool, cut_hi: bool
):
    """Complex bandpass keeping [lo, hi] (Hz) of the native signal at the intermediate rate.

    On a side where the sensor's band edge cuts through the signal (``cut_lo``/``cut_hi``) the
    transition lies inside [lo, hi], so the stopband starts exactly at the cut and nothing beyond
    it aliases back in, as with a receiver's anti-alias filter. On a side where the signal ends
    naturally, the transition lies outside, where there is no signal.
    """
    key = ("edge", up, round(native_rate), round(out_rate), round(lo), round(hi), cut_lo, cut_hi)
    if key not in _FILTERS:
        inter = up * native_rate
        width = EDGE_TRANSITION * out_rate
        p_lo = lo + width if cut_lo else lo
        p_hi = hi - width if cut_hi else hi
        if p_hi < p_lo:
            p_lo = p_hi = (lo + hi) / 2
        numtaps, beta = kaiserord(REJECTION_DB, width / (inter / 2))
        numtaps |= 1
        cutoff = min((p_hi - p_lo) / 2 + width / 2, 0.999 * inter / 2)
        lp = firwin(numtaps, cutoff, window=("kaiser", beta), fs=inter)
        n = np.arange(numtaps) - (numtaps - 1) / 2
        _FILTERS[key] = lp * np.exp(2j * np.pi * (p_lo + p_hi) / 2 * n / inter)
    return _FILTERS[key]


def _frac_taps(frac: float, half: int = FRAC_HALF_TAPS) -> np.ndarray:
    n = np.arange(-half, half + 1)
    return np.sinc(n - frac) * np.kaiser(len(n), 8.0)


def sample_native(
    native: np.ndarray,
    native_rate: float,
    u0: float,
    n_out: int,
    out_rate: float,
    half_bw: float | None = None,
    shift_hz: float = 0.0,
) -> np.ndarray:
    """Samples of a native-rate signal at times ``u0 + j / out_rate`` (j = 0 .. n_out-1).

    ``native[i]`` is the signal at time ``i / native_rate``; it is zero outside the array.
    ``half_bw`` (Hz), if given, is the signal's occupied half-bandwidth.

    ``shift_hz`` is the frequency offset the caller will apply after sampling (the emission's
    offset from the sensor's centre). Mixing commutes with aliasing modulo the output rate, so
    keeping only the part of the signal that lands inside ±out_rate/2 after the shift, here
    with a complex bandpass, gives the exact in-band result without oversampling.
    """
    up, down = _rate_ratio(out_rate, native_rate)
    band = None
    if half_bw:
        lo = max(-half_bw, -out_rate / 2 - shift_hz)
        hi = min(half_bw, out_rate / 2 - shift_hz)
        if hi <= lo:
            return np.zeros(n_out, dtype=np.complex128)
        if lo > -half_bw or hi < half_bw:  # the sensor band edge cuts through the signal
            band = _edge_filter(up, native_rate, out_rate, lo, hi, lo > -half_bw, hi < half_bw)
    # Render beyond each end by more than the rate converter's filter half-length (at the
    # intermediate rate) so its edge transients are cropped off
    if up == down == 1:
        half_len = 0
    elif band is not None:
        half_len = (len(band) - 1) // 2
    else:
        half_len = 10 * max(up, down)
    guard = math.ceil(half_len / down) + 4
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
    y = _fir(x, frac, h)
    if up == down == 1:
        z = y  # same rate: the fractional delay is all that is needed
    elif band is not None:
        z = resample_poly(y, up, down, window=band)
    else:
        z = resample_poly(y, up, down)
    out = z[guard : guard + n_out]
    if len(out) < n_out:
        out = np.concatenate([out, np.zeros(n_out - len(out), dtype=out.dtype)])
    return out


def _fir(x: np.ndarray, frac: float, half: int = FRAC_HALF_TAPS) -> np.ndarray:
    """y[m] = Σ_n x[m + n] · taps[n], i.e. x interpolated at position m + h + frac."""
    taps = _frac_taps(frac, half)
    return np.convolve(x, taps[::-1], mode="valid")
