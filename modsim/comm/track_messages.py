from __future__ import annotations

from datetime import datetime, timezone

from ..geo import km_to_latlon
from ..models import SensorMetadata, TrackStateRecord


def generate_track_message(
    track_state: TrackStateRecord,
    sensor_meta: SensorMetadata,
    reference_origin: tuple[float, float] = (0.0, 0.0),
    classification: str = "U",
    data_mode: str = "SIMULATED",
    source: str = "modsim",
) -> dict:
    """Generate Track.json compliant message from TrackStateRecord.

    Args:
        track_state: Detection with full object state
        sensor_meta: Sensor metadata
        reference_origin: (lat, lon) for coordinate transform
        classification: Classification marking
        data_mode: REAL, SIMULATED, EXERCISE, TEST
        source: Data source identifier

    Returns:
        Track.json formatted dictionary
    """
    # Convert simulation time to ISO8601
    ts_iso = datetime.fromtimestamp(track_state.time_sec, tz=timezone.utc).isoformat()

    # Convert km grid coordinates to lat/lon (flat earth approximation)
    lat, lon = km_to_latlon(track_state.obj_x, track_state.obj_y, reference_origin)

    # Convert speed from km/s to m/s
    spd_mps = track_state.speed_kmps * 1000.0

    return {
        "classificationMarking": classification,
        "ts": ts_iso,
        "msgTs": ts_iso,
        "source": source,
        "dataMode": data_mode,
        # Track identity
        "trkId": track_state.trk_id,
        "trkNum": track_state.trk_num,
        "trkStat": track_state.trk_stat,
        "trkQual": int(track_state.pd * 100),  # Use Pd as quality metric
        "trkPtType": "MEASURED",
        # Sensor/asset info
        "asset": sensor_meta.asset,
        "assetNat": sensor_meta.asset_nat,
        "sen": sensor_meta.sensor_id,
        "modType": sensor_meta.sen_type,
        # Object identity (generic, since simulation doesn't classify)
        "objId": track_state.obj_id,
        "objIdent": "UNKNOWN",
        "objType": "UNKNOWN",
        "env": "AIR",  # Default assumption
        # Position and motion
        "lat": lat,
        "lon": lon,
        "alt": track_state.obj_z * 1000.0,  # Convert km to meters
        "spd": spd_mps,
        "hdng": track_state.heading_deg,
        "trkConf": track_state.pd,
    }
