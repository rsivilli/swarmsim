"""Channel effects beyond free space: ground reflection, shadowing and Rician fading.

All processes are deterministic closed-form functions of time (and of a per-link seed), so the
link budget, the truth tables and every recording chunk evaluate exactly the same channel.

* **Ground reflection (two-ray):** the reflected path comes from the emitter's mirror image
  below a flat ground (z = 0). Its strength and phase are given by the Fresnel reflection
  coefficient for the grazing angle, frequency and ground constants. The synthesizer renders it
  as a real second path (own delay and Doppler); the link budget uses the narrowband two-ray
  factor at the carrier.
* **Shadowing:** log-normal, correlated along each emitter's track with an exponential
  (Gudmundson) autocorrelation exp(−Δs/d). It is generated as a sum of sinusoids in travelled
  distance with Cauchy-distributed spatial frequencies (whose spectrum gives exactly that
  autocorrelation).
* **Rician fading:** h = √(K/(K+1)) + √(1/(K+1))·g(t). The diffuse part g is a unit-power sum of
  sinusoids with Doppler spread f_D = v·f/c (Jakes-like). Each scattered ray also has an arrival
  azimuth, so array elements see a spatially correlated diffuse field. K rises linearly in dB
  with elevation (air-to-ground links are more line-of-sight at high elevation).
"""

from __future__ import annotations

import math

import numpy as np

from .models import PropagationConfig
from .propagation import C_MPS
from .scheduler import stable_seed

N_SHADOW_SINES = 32


def link_seed(scenario_seed: int | None, emitter_id: str, sensor_id: str) -> int:
    return stable_seed(scenario_seed or 0, "channel", emitter_id, sensor_id)


# ---------------------------------------------------------------------------
# Ground reflection
# ---------------------------------------------------------------------------


def reflection_coefficient(grazing_rad, freq_hz, prop: PropagationConfig):
    """Fresnel reflection coefficient of flat ground for the configured polarization."""
    lam = C_MPS / np.asarray(freq_hz, dtype=np.float64)
    eps = prop.ground_permittivity - 1j * 60.0 * lam * prop.ground_conductivity_s_m
    s = np.sin(grazing_rad)
    root = np.sqrt(eps - np.cos(grazing_rad) ** 2)
    if prop.polarization == "horizontal":
        return (s - root) / (s + root)
    return (eps * s - root) / (eps * s + root)


def image_paths(emitter_km: np.ndarray, sensor_km: np.ndarray):
    """Direct and ground-reflected path lengths (m) and grazing angle (rad).

    ``emitter_km`` is (..., 3); ``sensor_km`` is (3,) or broadcastable. Heights are clipped at
    ground level.
    """
    e = np.asarray(emitter_km, dtype=np.float64) * 1000.0
    s = np.asarray(sensor_km, dtype=np.float64) * 1000.0
    ground = np.hypot(e[..., 0] - s[..., 0], e[..., 1] - s[..., 1])
    h_e = np.maximum(e[..., 2], 0.0)
    h_s = np.maximum(s[..., 2], 0.0)
    direct = np.sqrt(ground**2 + (h_e - h_s) ** 2)
    reflected = np.sqrt(ground**2 + (h_e + h_s) ** 2)
    grazing = np.arctan2(h_e + h_s, np.maximum(ground, 1e-9))
    return direct, reflected, grazing


def two_ray_factor(emitter_km, sensor_km, freq_hz, prop: PropagationConfig):
    """Narrowband field factor 1 + Γ·(d/r)·exp(−j2πf(r−d)/c); |·|² is the power gain."""
    d, r, psi = image_paths(emitter_km, sensor_km)
    gamma = reflection_coefficient(psi, freq_hz, prop)
    return 1.0 + gamma * (d / r) * np.exp(-2j * np.pi * np.asarray(freq_hz) * (r - d) / C_MPS)


# ---------------------------------------------------------------------------
# Shadowing
# ---------------------------------------------------------------------------


def shadowing_db(seed: int, distance_m, prop: PropagationConfig):
    """Log-normal shadowing (dB) along a track at travelled distances ``distance_m``."""
    if prop.shadowing_db <= 0:
        return np.zeros_like(np.asarray(distance_m, dtype=np.float64))
    rng = np.random.default_rng([seed, 0x5AD])
    # Cauchy spatial frequencies (cycles/m) give the autocorrelation exp(−|Δs|/d)
    kappa = np.abs(rng.standard_cauchy(N_SHADOW_SINES)) / (
        2 * np.pi * prop.shadowing_decorrelation_m
    )
    phase = rng.uniform(0, 2 * np.pi, N_SHADOW_SINES)
    s = np.asarray(distance_m, dtype=np.float64)[..., None]
    x = np.sqrt(2.0 / N_SHADOW_SINES) * np.cos(2 * np.pi * kappa * s + phase).sum(axis=-1)
    return prop.shadowing_db * x


# ---------------------------------------------------------------------------
# Rician fading
# ---------------------------------------------------------------------------


def rician_k_linear(elevation_deg, prop: PropagationConfig):
    k0, k90 = prop.rician_k_db
    el = np.clip(np.asarray(elevation_deg, dtype=np.float64), 0.0, 90.0)
    return 10 ** ((k0 + (k90 - k0) * el / 90.0) / 10)


def doppler_spread_hz(speed_mps, freq_hz, prop: PropagationConfig):
    return np.maximum(
        np.asarray(speed_mps) * np.asarray(freq_hz) / C_MPS, prop.fading_min_doppler_hz
    )


def diffuse_gain(
    seed: int,
    t,
    f_d: float,
    freq_hz: float,
    prop: PropagationConfig,
    offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Unit-power diffuse component g(t) at absolute times ``t``, shape (len(t), N elements).

    Each of the M rays has a Doppler angle α (Doppler f_D·cos α), a random phase and a
    horizontal arrival azimuth β. Element i adds 2π f/c · (pᵢ · u_β).
    """
    m = prop.fading_paths
    rng = np.random.default_rng([seed, 0xFAD])
    alpha = rng.uniform(0, 2 * np.pi, m)
    phi = rng.uniform(0, 2 * np.pi, m)
    beta = rng.uniform(0, 2 * np.pi, m)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    fd = np.broadcast_to(np.asarray(f_d, dtype=np.float64), t.shape)[:, None]
    # Phase of each ray at each time; f_D·t is small enough for float64 at any scenario length
    ray = 2 * np.pi * np.mod(fd * np.cos(alpha)[None, :] * t[:, None], 1.0) + phi[None, :]
    if offsets_m is None:
        offsets_m = np.zeros((1, 3))
    u = np.column_stack([np.sin(beta), np.cos(beta), np.zeros(m)])  # (M, 3) east, north, up
    elem = 2 * np.pi * freq_hz / C_MPS * (offsets_m @ u.T)  # (N, M)
    return np.exp(1j * (ray[:, None, :] + elem[None, :, :])).sum(axis=-1) / math.sqrt(m)


def rician_gain(seed, t, f_d, freq_hz, k_lin, prop: PropagationConfig, offsets_m=None):
    """Complex Rician channel gain (unit mean power), shape (len(t), N elements).

    ``f_d`` and ``k_lin`` may be scalars or per-time arrays.
    """
    g = diffuse_gain(seed, t, f_d, freq_hz, prop, offsets_m)
    k = np.broadcast_to(np.asarray(k_lin, dtype=np.float64), (g.shape[0],))[:, None]
    return np.sqrt(k / (k + 1)) + np.sqrt(1 / (k + 1)) * g
