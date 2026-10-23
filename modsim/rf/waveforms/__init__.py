"""Baseband waveform generators.

All waveforms are protocol-like: correct modulation family, bandwidth, burst structure and
timing, with random payload bits.

* ``lora``, ``fm`` and ``cw`` are closed-form functions of the time since the emission started.
* ``fsk`` (GFSK) uses a phase table at 16 samples per bit, interpolated at the exact sample times.
* ``ofdm`` bursts are built by IFFT at their native rate and placed on the sensor grid with a
  fractional-delay FIR plus rational resampling (see ``native.py``).

Any chunk of a recording can be rendered independently, and propagation delay is applied by
evaluating the waveform at ``t - tau(t)``. ``noise`` is generic band-limited noise with the
right power, bandwidth and timing, for custom emitters with no specific modulation.

Generators return ``(kind, values)``: constant-envelope waveforms return ``("phase", radians)``,
so the caller can add the carrier phase and take a single sin/cos; others return
``("iq", complex samples)``. Both have unit mean power and are centred on the emission's carrier.
"""

from __future__ import annotations

import json

import numpy as np

from .fm import fm_video
from .fsk import gfsk_phase
from .lora import lora_css
from .native import sample_native
from .ofdm import ofdm_burst

PROTOCOL_LIKE = {"lora", "fm", "cw", "ofdm", "fsk"}

# Waveform parameters accepted in a profile's ``waveform_params``
WAVEFORM_PARAMS: dict[str, set[str]] = {
    "ofdm": {"standard", "modulation", "sync_symbols", "zc_roots"},
    "fsk": {"bit_rate", "h", "bt", "preamble", "sync"},
    "fm": {"deviation_hz", "audio_subcarrier_hz", "audio_level"},
    "lora": set(),  # LoRa parameters come from the activity's ``lora`` block
    "cw": set(),
    "noise": set(),  # generic band-limited noise (custom emitters; fidelity "placeholder")
    "none": set(),
}

OFDM_CHOICES = {
    "standard": {"wifi", "lte", "droneid"},
    "modulation": {"bpsk", "qpsk", "qam16", "qam64"},
}

_BURSTS: dict[tuple, tuple[np.ndarray, float]] = {}


def fidelity(waveform: str) -> str:
    return "protocol_like" if waveform in PROTOCOL_LIKE else "placeholder"


def _ofdm(duration: float, seed: int, bandwidth_hz: float, wp: dict) -> tuple[np.ndarray, float]:
    key = (seed, duration, bandwidth_hz, json.dumps(wp, sort_keys=True))
    if key not in _BURSTS:
        if len(_BURSTS) > 256:
            _BURSTS.clear()
        _BURSTS[key] = ofdm_burst(duration, seed, bandwidth_hz, wp)
    return _BURSTS[key]


def render(
    waveform: str,
    u: np.ndarray,
    *,
    bandwidth_hz: float,
    duration: float,
    seed: int,
    params: dict,
    sample_rate: float,
    noise_key: tuple,
) -> tuple[str, np.ndarray]:
    """Waveform at times ``u`` (seconds since emission start, spaced 1/sample_rate)."""
    wp = params.get("waveform") or {}
    if waveform == "cw":
        return "phase", np.zeros(len(u))
    if waveform == "lora":
        lora = params.get("lora") or {}
        return "phase", lora_css(
            u,
            sf=int(lora.get("sf", 7)),
            bw_hz=float(lora.get("bw_hz", bandwidth_hz)),
            preamble=int(lora.get("preamble", 8)),
            seed=seed,
        )
    if waveform == "fm":
        return "phase", fm_video(u, seed=seed, **wp)
    if waveform == "fsk":
        return "phase", gfsk_phase(u, duration, seed, wp)
    if waveform == "ofdm":
        native, rate = _ofdm(duration, seed, bandwidth_hz, wp)
        return "iq", sample_native(native, rate, float(u[0]), len(u), sample_rate)
    return "iq", bandlimited_noise(len(u), bandwidth_hz, sample_rate, seed, noise_key)


def bandlimited_noise(
    n: int, bandwidth_hz: float, sample_rate: float, seed: int, noise_key: tuple
) -> np.ndarray:
    """Unit-power complex Gaussian noise confined to |f| <= bandwidth/2 (FFT-domain shaping)."""
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    rng = np.random.default_rng([seed, *noise_key])
    white = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    spec = np.fft.fft(white)
    freqs = np.fft.fftfreq(n, d=1.0 / sample_rate)
    keep = np.abs(freqs) <= bandwidth_hz / 2
    kept = int(keep.sum())
    if kept == 0:
        keep[0] = True
        kept = 1
    spec[~keep] = 0
    # E|white|^2 = 2 per sample; after masking, power scales by kept/n
    return np.fft.ifft(spec) / np.sqrt(2.0 * kept / n)
