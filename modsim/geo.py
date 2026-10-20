"""Geographic coordinate conversion utilities."""

from __future__ import annotations

import math


def km_to_latlon(x_km: float, y_km: float, origin: tuple[float, float]) -> tuple[float, float]:
    """Convert km grid to lat/lon using flat earth approximation.

    Approximation:
    - 1 degree latitude ≈ 111 km
    - 1 degree longitude ≈ 111 km * cos(lat)

    Acceptable for areas <100km.

    Args:
        x_km: X coordinate in km (East-West)
        y_km: Y coordinate in km (North-South)
        origin: (lat, lon) reference origin

    Returns:
        (lat, lon) tuple in decimal degrees
    """
    origin_lat, origin_lon = origin

    # Convert y (km) to latitude degrees
    lat = origin_lat + (y_km / 111.0)

    # Convert x (km) to longitude degrees (accounting for latitude)
    lon = origin_lon + (x_km / (111.0 * math.cos(math.radians(origin_lat))))

    return lat, lon


def latlon_to_km(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    """Convert lat/lon to km grid using flat earth approximation.

    Inverse of km_to_latlon.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        origin: (lat, lon) reference origin

    Returns:
        (x_km, y_km) tuple in kilometers
    """
    origin_lat, origin_lon = origin

    # Convert latitude degrees to y (km)
    y_km = (lat - origin_lat) * 111.0

    # Convert longitude degrees to x (km) (accounting for latitude)
    x_km = (lon - origin_lon) * (111.0 * math.cos(math.radians(origin_lat)))

    return x_km, y_km
