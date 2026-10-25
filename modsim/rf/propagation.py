"""Link-budget primitives: free-space path loss, radio horizon, noise floor, LoRa airtime."""

from __future__ import annotations

import math

import numpy as np

C_MPS = 299_792_458.0
EARTH_RADIUS_M = 6_371_000.0
THERMAL_NOISE_DBM_HZ = -174.0  # kT at 290 K


def fspl_db(distance_m, freq_hz):
    """Free-space path loss in dB: 20·log10(4πdf/c). Vectorized; distance is clamped to 1 m."""
    d = np.maximum(np.asarray(distance_m, dtype=np.float64), 1.0)
    f = np.asarray(freq_hz, dtype=np.float64)
    return 20.0 * np.log10(4.0 * math.pi * d * f / C_MPS)


def radio_horizon_m(h_tx_m, h_rx_m, k_factor: float = 4.0 / 3.0):
    """Maximum ground distance for line of sight over a smooth earth: √(2kRh₁) + √(2kRh₂)."""
    re = k_factor * EARTH_RADIUS_M
    h1 = np.maximum(np.asarray(h_tx_m, dtype=np.float64), 0.0)
    h2 = np.maximum(np.asarray(h_rx_m, dtype=np.float64), 0.0)
    return np.sqrt(2.0 * re * h1) + np.sqrt(2.0 * re * h2)


def noise_floor_dbm(bandwidth_hz, noise_figure_db: float):
    """Thermal noise power in a bandwidth: -174 dBm/Hz + 10·log10(B) + NF."""
    b = np.maximum(np.asarray(bandwidth_hz, dtype=np.float64), 1.0)
    return THERMAL_NOISE_DBM_HZ + 10.0 * np.log10(b) + noise_figure_db


def doppler_hz(freq_hz, radial_velocity_mps):
    """Doppler shift for a closing (negative) or opening (positive) radial velocity."""
    return -np.asarray(freq_hz) * np.asarray(radial_velocity_mps) / C_MPS


def lora_time_on_air(
    sf: int,
    bw_hz: float,
    payload_bytes: int,
    preamble: int = 8,
    cr: int = 1,
    implicit_header: bool = False,
    crc: bool = True,
    low_data_rate_optimize: bool | None = None,
) -> float:
    """LoRa packet time on air (Semtech SX127x datasheet formula).

    ``cr`` is the coding-rate index 1..4 (4/5 .. 4/8). Low-data-rate optimisation defaults to
    on when the symbol time exceeds 16 ms, as the datasheet recommends.
    """
    t_sym = (2**sf) / bw_hz
    de = (t_sym > 0.016) if low_data_rate_optimize is None else low_data_rate_optimize
    t_preamble = (preamble + 4.25) * t_sym
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * int(implicit_header)
    den = 4 * (sf - 2 * int(de))
    n_payload = 8 + max(math.ceil(num / den) * (cr + 4), 0)
    return t_preamble + n_payload * t_sym
