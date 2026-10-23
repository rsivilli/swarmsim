"""LoRa chirp spread spectrum (protocol-like: preamble, sync, SFD, random payload symbols)."""

from __future__ import annotations

import numpy as np

SYNC_SYMBOLS = (8, 16)  # public sync word 0x12 -> symbol values 1<<3 and 2<<3


def lora_css(u: np.ndarray, *, sf: int, bw_hz: float, preamble: int, seed: int) -> np.ndarray:
    """Phase (radians) of a LoRa frame at times ``u`` (s, ascending) after the frame start.

    Frame: ``preamble`` up-chirps, two sync symbols, 2.25 down-chirps (SFD), then random payload
    symbols until the emission ends. Each symbol spans an integer number of cycles
    (BW·T = 2^SF), so the phase is continuous across symbol boundaries.
    """
    n = 2**sf
    t_sym = n / bw_hz
    u = np.asarray(u, dtype=np.float64)
    phase = np.empty(len(u))

    n_head = preamble + len(SYNC_SYMBOLS)
    sfd_start = n_head * t_sym
    payload_start = sfd_start + 2.25 * t_sym
    i_sfd, i_pay = np.searchsorted(u, [sfd_start, payload_start])

    n_payload = int(max(0.0, (u[-1] - payload_start) / t_sym)) + 1 if len(u) else 0
    values = np.concatenate(
        [
            np.zeros(preamble, dtype=np.int64),
            np.asarray(SYNC_SYMBOLS, dtype=np.int64) % n,
            np.random.default_rng([seed, 0x10A]).integers(0, n, n_payload),
        ]
    )

    def up(sl, origin, first_symbol):
        x = u[sl] - origin
        sym = np.floor(x / t_sym)
        v = x - sym * t_sym
        k = values[first_symbol + sym.astype(np.int64)] / n
        wrap = np.maximum(0.0, v - (1.0 - k) * t_sym)
        phase[sl] = 2 * np.pi * bw_hz * (k * v + v * v / (2 * t_sym) - v / 2 - wrap)

    up(slice(0, i_sfd), 0.0, 0)
    up(slice(i_pay, len(u)), payload_start, n_head)
    vd = np.mod(u[i_sfd:i_pay] - sfd_start, t_sym)
    phase[i_sfd:i_pay] = 2 * np.pi * bw_hz * (vd / 2 - vd * vd / (2 * t_sym))
    return phase
