"""Antenna array geometry, element patterns and steering (SigMF ``spatial`` conventions).

Local array frame, as in the SigMF ``spatial`` extension: +X is the aperture boresight, +Y is
to its left, +Z is up. Bearings are azimuth clockwise (seen from above) and elevation above
the horizon. The boresight points ``boresight_deg`` east of true north, so the local axes in
world coordinates (east, north, up) are::

    X = (sin β, cos β, 0),  Y = (−cos β, sin β, 0),  Z = (0, 0, 1)
"""

from __future__ import annotations

import numpy as np

from .models import ArrayConfig, ArrayGeometry
from .propagation import C_MPS


def local_axes(boresight_deg: float) -> np.ndarray:
    """Rows are the local X, Y, Z axes expressed in world (east, north, up) coordinates."""
    b = np.radians(boresight_deg)
    return np.array([[np.sin(b), np.cos(b), 0.0], [-np.cos(b), np.sin(b), 0.0], [0.0, 0.0, 1.0]])


def element_positions_local(arr: ArrayConfig) -> np.ndarray:
    """(N, 3) element phase centres in metres, local frame."""
    n = arr.elements
    if arr.geometry == ArrayGeometry.ULA:
        y = (np.arange(n) - (n - 1) / 2) * arr.spacing_m
        return np.column_stack([np.zeros(n), y, np.zeros(n)])
    if arr.geometry == ArrayGeometry.UCA:
        a = 2 * np.pi * np.arange(n) / n  # counter-clockwise from boresight (+X)
        return np.column_stack([arr.radius_m * np.cos(a), arr.radius_m * np.sin(a), np.zeros(n)])
    return np.asarray(arr.element_positions_m, dtype=np.float64)


def element_offsets_world(arr: ArrayConfig) -> np.ndarray:
    """(N, 3) element offsets from the sensor position in metres, world (east, north, up)."""
    return element_positions_local(arr) @ local_axes(arr.boresight_deg)


def element_pointing_local(arr: ArrayConfig) -> np.ndarray:
    """(N, 3) unit pointing directions: radial for a UCA, boresight otherwise."""
    n = arr.elements
    if arr.geometry == ArrayGeometry.UCA:
        a = 2 * np.pi * np.arange(n) / n
        return np.column_stack([np.cos(a), np.sin(a), np.zeros(n)])
    return np.tile([1.0, 0.0, 0.0], (n, 1))


def direction_local(rel_azimuth_deg, elevation_deg) -> np.ndarray:
    """Unit vectors toward the given bearings (relative azimuth clockwise from boresight)."""
    az, el = np.broadcast_arrays(
        np.radians(np.asarray(rel_azimuth_deg, dtype=np.float64)),
        np.radians(np.asarray(elevation_deg, dtype=np.float64)),
    )
    return np.stack([np.cos(el) * np.cos(az), -np.cos(el) * np.sin(az), np.sin(el)], axis=-1)


def relative_azimuth(arr: ArrayConfig, azimuth_deg):
    """Azimuth relative to the aperture boresight, wrapped to [0, 360)."""
    return np.mod(np.asarray(azimuth_deg) - arr.boresight_deg, 360.0)


def pattern_power(arr: ArrayConfig, d_local: np.ndarray) -> np.ndarray:
    """Element power pattern (linear, peak 1) toward directions ``d_local`` (…, 3) → (…, N)."""
    shape = d_local.shape[:-1] + (arr.elements,)
    if arr.element_pattern == "isotropic":
        return np.ones(shape)
    cos_psi = d_local @ element_pointing_local(arr).T
    floor = 10 ** (arr.back_lobe_db / 10)
    return np.maximum(np.clip(cos_psi, 0.0, None) ** arr.pattern_exponent, floor)


def mean_pattern_gain_db(arr: ArrayConfig, azimuth_deg, elevation_deg) -> np.ndarray:
    """Element pattern averaged over the array, in dB, toward world bearings."""
    d = direction_local(relative_azimuth(arr, azimuth_deg), elevation_deg)
    return 10 * np.log10(pattern_power(arr, d).mean(axis=-1))


def steering_vector(arr: ArrayConfig, freq_hz: float, rel_azimuth_deg, elevation_deg) -> np.ndarray:
    """Plane-wave steering vectors exp(j2π f p·d / c), shape (…, N), isotropic elements.

    Matches the synthesizer's convention: an element closer to the emitter receives the wave
    earlier, i.e. with a phase advance relative to the array centre.
    """
    d = direction_local(rel_azimuth_deg, elevation_deg)
    delay_adv = d @ element_positions_local(arr).T / C_MPS
    return np.exp(2j * np.pi * freq_hz * delay_adv)


def calibration_errors(
    n: int, gain_db_sigma: float, phase_deg_sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-channel complex gain errors (unit mean), drawn once per sensor."""
    gain_db = rng.normal(0.0, gain_db_sigma, n) if gain_db_sigma > 0 else np.zeros(n)
    phase = rng.normal(0.0, phase_deg_sigma, n) if phase_deg_sigma > 0 else np.zeros(n)
    return 10 ** (gain_db / 20) * np.exp(1j * np.radians(phase))
