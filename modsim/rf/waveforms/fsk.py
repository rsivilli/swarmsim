"""GFSK bursts (protocol-like): preamble, sync word, random payload, Gaussian-filtered NRZ.

Used for Bluetooth LE advertising (1 Mb/s, h = 0.5, BT = 0.5, access address 0x8E89BED6) and
SiK-style telemetry radios. The phase is tabulated at 16 samples per bit and interpolated at the
exact (delayed) sample times; it is smooth, so the interpolation error is ~1e-3 rad.
"""

from __future__ import annotations

import math

import numpy as np

SPS = 16


def _hex_bits(hex_str: str) -> np.ndarray:
    if not hex_str:
        return np.zeros(0, dtype=np.int8)
    value = int(hex_str, 16)
    n = 4 * len(hex_str)
    # Transmitted LSB first, as BLE does
    return np.array([(value >> i) & 1 for i in range(n)], dtype=np.int8)


def gfsk_phase_table(
    duration: float,
    seed: int,
    *,
    bit_rate: float,
    h: float = 0.5,
    bt: float = 0.5,
    preamble: str = "aa",
    sync: str = "",
) -> tuple[np.ndarray, float]:
    """Phase (radians) at ``SPS`` samples per bit over the burst; returns (table, table rate)."""
    n_bits = math.ceil(duration * bit_rate) + 2
    head = np.concatenate([_hex_bits(preamble), _hex_bits(sync)])
    rng = np.random.default_rng([seed, 0xF5C])
    bits = np.concatenate([head, rng.integers(0, 2, max(n_bits - len(head), 0))])[:n_bits]
    nrz = np.repeat(2.0 * bits - 1.0, SPS)
    sigma = math.sqrt(math.log(2)) / (2 * math.pi * bt) * SPS
    t = np.arange(-3 * SPS, 3 * SPS + 1)
    g = np.exp(-(t**2) / (2 * sigma**2))
    freq = np.convolve(nrz, g / g.sum(), mode="same")
    phase = np.pi * h * np.concatenate([[0.0], np.cumsum(freq)[:-1]]) / SPS
    return phase, bit_rate * SPS


def gfsk_phase(u: np.ndarray, duration: float, seed: int, params: dict) -> np.ndarray:
    table, rate = gfsk_phase_table(
        duration,
        seed,
        bit_rate=float(params["bit_rate"]),
        h=float(params.get("h", 0.5)),
        bt=float(params.get("bt", 0.5)),
        preamble=str(params.get("preamble", "aa")),
        sync=str(params.get("sync", "")),
    )
    return np.interp(np.asarray(u) * rate, np.arange(len(table)), table)
