"""Baseband waveform generators.

Protocol-like generators (``lora``, ``fm``, ``cw``) are closed-form functions of the time since
the emission started, so any chunk of a recording can be rendered independently and the
propagation delay is applied exactly by evaluating them at ``t - tau(t)``. Waveform families
not implemented yet (``ofdm``, ``fsk``, until Phase 2) fall back to a band-limited noise
placeholder with the correct power, bandwidth and timing.

Generators return ``(kind, values)``: constant-envelope waveforms return ``("phase", radians)``,
so the caller can add the carrier phase and take a single sin/cos; others return
``("iq", complex samples)``. Both have unit mean power and are centred on the emission's carrier.
"""

from __future__ import annotations

import numpy as np

from .fm import fm_video
from .lora import lora_css

PROTOCOL_LIKE = {"lora", "fm", "cw"}


def fidelity(waveform: str) -> str:
    return "protocol_like" if waveform in PROTOCOL_LIKE else "placeholder"


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
    """Waveform at times ``u`` (seconds since emission start, uniformly spaced)."""
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
        return "phase", fm_video(u, seed=seed, **(params.get("fm") or {}))
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
