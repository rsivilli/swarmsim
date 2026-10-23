"""OFDM bursts (protocol-like): 802.11a/g/n legacy, LTE-like numerologies and DJI DroneID.

Bursts are generated at their native sample rate by IFFT. Structure follows the public
descriptions: the 802.11 legacy preamble (STF, LTF), then 52-subcarrier data symbols with four
pilots; LTE-like 15 kHz numerology with Zadoff-Chu sync symbols (as reported for DJI OcuSync);
and the DroneID burst layout (1024-point FFT at 15.36 MS/s, 600 subcarriers, nine symbols,
Zadoff-Chu roots 600 and 147 in symbols 4 and 6). Payload symbols are random. The output has
unit mean power.

Bursts are generated ``OVERSAMPLE``× oversampled (zero-padded IFFT), with ~100 ns raised-cosine
windowed overlap-add across symbol boundaries as 802.11 transmitters use. Out-of-band splatter
from hard symbol edges is suppressed, and the occupied band stays well inside the native Nyquist
range, so fractional-delay resampling onto any sensor grid is accurate.
"""

from __future__ import annotations

import math

import numpy as np

# 802.11 legacy short / long training sequences on subcarriers -26..26
_STF = np.zeros(53, dtype=np.complex128)
for _k, _v in {
    -24: 1,
    -20: -1,
    -16: 1,
    -12: -1,
    -8: -1,
    -4: 1,
    4: -1,
    8: -1,
    12: 1,
    16: 1,
    20: 1,
    24: 1,
}.items():
    _STF[_k + 26] = _v * (1 + 1j) * math.sqrt(13 / 6)
_LTF = np.array(
    [1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 0,
     1, -1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1],
    dtype=np.complex128,
)  # fmt: skip
_WIFI_PILOTS = (-21, -7, 7, 21)
OVERSAMPLE = 2
WINDOW_S = 100e-9  # symbol-boundary raised-cosine transition

# LTE channel bandwidth -> (FFT size, active subcarriers)
_LTE = {1.4e6: (128, 72), 3e6: (256, 180), 5e6: (512, 300), 10e6: (1024, 600),
        15e6: (1536, 900), 20e6: (2048, 1200), 40e6: (4096, 2400)}  # fmt: skip


def _constellation(name: str, n, rng: np.random.Generator) -> np.ndarray:
    if name == "bpsk":
        return rng.choice([-1.0, 1.0], size=n).astype(np.complex128)
    if name == "qpsk":
        return (rng.choice([-1.0, 1.0], size=n) + 1j * rng.choice([-1.0, 1.0], size=n)) / np.sqrt(2)
    levels = {"qam16": (np.array([-3, -1, 1, 3]), 10), "qam64": (np.arange(-7, 8, 2), 42)}
    if name not in levels:
        raise ValueError(f"Unknown OFDM modulation '{name}'")
    pts, norm = levels[name]
    return (rng.choice(pts, size=n) + 1j * rng.choice(pts, size=n)) / np.sqrt(norm)


def _zadoff_chu(root: int, length: int) -> np.ndarray:
    nzc = length + 1 if length % 2 == 0 else length
    n = np.arange(nzc)
    seq = np.exp(-1j * np.pi * root * n * (n + 1) / nzc)
    return np.delete(seq, nzc // 2)[:length] if nzc != length else seq


def _active_bins(n_fft: int, n_active: int) -> np.ndarray:
    half = n_active // 2
    k = np.concatenate([np.arange(-half, 0), np.arange(1, half + 1)])
    return k % n_fft


def _ifft_os(freq: np.ndarray, n_fft: int, n_active: int) -> np.ndarray:
    """Oversampled IFFT of rows of ``freq`` (… × n_fft bins), scaled to unit power."""
    m = n_fft * OVERSAMPLE
    big = np.zeros(freq.shape[:-1] + (m,), dtype=np.complex128)
    half = n_fft // 2
    big[..., :half] = freq[..., :half]
    big[..., m - half :] = freq[..., half:]
    return np.fft.ifft(big, axis=-1) * (m / np.sqrt(n_active))


def _wola(segments: list[np.ndarray], w: int) -> np.ndarray:
    """Overlap-add segments that each carry ``w`` extra samples of natural continuation.

    Each segment ramps up over its first ``w`` samples and down over its trailing ``w`` samples;
    consecutive segments overlap by ``w``.
    """
    ramp = 0.5 - 0.5 * np.cos(np.pi * (np.arange(w) + 0.5) / w)
    total = sum(len(s) - w for s in segments) + w
    out = np.zeros(total, dtype=np.complex128)
    pos = 0
    for seg in segments:
        seg = seg.copy()
        seg[:w] *= ramp
        seg[-w:] *= ramp[::-1]
        out[pos : pos + len(seg)] += seg
        pos += len(seg) - w
    return out


def _symbols(freq: np.ndarray, n_fft: int, cps: list[int], n_active: int, rate: float) -> list:
    """Time-domain OFDM symbols (cyclic prefix + body + cyclic suffix for windowing)."""
    w = _window_len(rate)
    t = _ifft_os(freq, n_fft, n_active)
    return [
        np.concatenate([row[len(row) - cp * OVERSAMPLE :], row, row[:w]]) for row, cp in zip(t, cps)
    ]


def _window_len(rate: float) -> int:
    return max(2, math.ceil(WINDOW_S * rate * OVERSAMPLE))


def wifi_burst(duration: float, seed: int, modulation: str = "qam16") -> tuple[np.ndarray, float]:
    rate, n_fft = 20e6, 64
    rng = np.random.default_rng([seed, 0x0FD])
    os_ = OVERSAMPLE
    w = _window_len(rate)
    k = np.arange(-26, 27)
    bins = k % n_fft
    stf_sym = np.zeros(n_fft, dtype=np.complex128)
    stf_sym[bins] = _STF
    stf = _ifft_os(stf_sym, n_fft, 52)
    ltf_sym = np.zeros(n_fft, dtype=np.complex128)
    ltf_sym[bins] = _LTF
    ltf = _ifft_os(ltf_sym, n_fft, 52)
    # STF: ten 0.8 µs periods (8 µs); LTF: 1.6 µs guard + two 3.2 µs symbols (8 µs)
    stf_seg = np.tile(stf[: 16 * os_], 10 + math.ceil(w / (16 * os_)))[: 160 * os_ + w]
    ltf_seg = np.concatenate([ltf[-32 * os_ :], ltf, ltf, ltf[:w]])

    n_data = max(1, math.ceil((duration * rate - 320) / 80))
    data_k = np.array([x for x in k if x != 0 and x not in _WIFI_PILOTS])
    freq = np.zeros((n_data, n_fft), dtype=np.complex128)
    freq[:, data_k % n_fft] = _constellation(modulation, (n_data, len(data_k)), rng)
    freq[:, np.array(_WIFI_PILOTS) % n_fft] = np.array([1, 1, 1, -1])
    freq[0, data_k % n_fft] = _constellation("bpsk", len(data_k), rng)  # SIGNAL field
    segments = [stf_seg, ltf_seg] + _symbols(freq, n_fft, [16] * n_data, 52, rate)
    return _wola(segments, w), rate * os_


def lte_burst(
    duration: float,
    seed: int,
    bandwidth_hz: float,
    modulation: str = "qpsk",
    sync_symbols: tuple[int, ...] = (3, 5),
    zc_roots: tuple[int, ...] = (600, 147),
) -> tuple[np.ndarray, float]:
    """LTE-like numerology (15 kHz, normal CP) sized to the channel bandwidth."""
    bw = min(_LTE, key=lambda b: abs(b - bandwidth_hz))
    n_fft, n_active = _LTE[bw]
    rate = n_fft * 15e3
    cp = round(n_fft * 144 / 2048)
    n_sym = max(1, math.ceil(duration * rate / (n_fft + cp)))
    return _sync_ofdm(n_fft, n_active, rate, [cp] * n_sym, seed, modulation, sync_symbols, zc_roots)


def droneid_burst(duration: float, seed: int) -> tuple[np.ndarray, float]:
    """DJI DroneID layout: 1024-FFT at 15.36 MS/s, 600 subcarriers, 9 symbols (CP 80, then 72)."""
    n_fft, n_active, rate = 1024, 600, 15.36e6
    n_sym = max(9, math.ceil((duration * rate - (80 + n_fft)) / (n_fft + 72)) + 1)
    cps = [80] + [72] * (n_sym - 1)
    return _sync_ofdm(n_fft, n_active, rate, cps, seed, "qpsk", (3, 5), (600, 147))


def _sync_ofdm(n_fft, n_active, rate, cps, seed, modulation, sync_symbols, zc_roots):
    rng = np.random.default_rng([seed, 0x0FD])
    bins = _active_bins(n_fft, n_active)
    freq = np.zeros((len(cps), n_fft), dtype=np.complex128)
    freq[:, bins] = _constellation(modulation, (len(cps), n_active), rng)
    for s, root in zip(sync_symbols, zc_roots):
        if s < len(cps):
            freq[s, bins] = _zadoff_chu(root, n_active)
    return _wola(_symbols(freq, n_fft, cps, n_active, rate), _window_len(rate)), rate * OVERSAMPLE


def ofdm_burst(
    duration: float, seed: int, bandwidth_hz: float, params: dict
) -> tuple[np.ndarray, float]:
    standard = params.get("standard", "wifi")
    if standard == "wifi":
        return wifi_burst(duration, seed, params.get("modulation", "qam16"))
    if standard == "lte":
        return lte_burst(
            duration,
            seed,
            bandwidth_hz,
            params.get("modulation", "qpsk"),
            tuple(params.get("sync_symbols", (3, 5))),
            tuple(params.get("zc_roots", (600, 147))),
        )
    if standard == "droneid":
        return droneid_burst(duration, seed)
    raise ValueError(f"Unknown OFDM standard '{standard}'")
