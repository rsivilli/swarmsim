"""Storage estimates for SigMF recordings (used by --dry-run and validate's preflight)."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from .models import RFSensorConfig, TuningMode

_DATATYPE = re.compile(r"^(?P<cplx>[cr])(?P<kind>[fiu])(?P<bits>8|16|32|64)(?P<endian>_le|_be)?$")
_VALID_BITS = {"f": {32, 64}, "i": {8, 16, 32}, "u": {8, 16, 32}}


def parse_datatype(fmt: str) -> tuple[bool, str, int]:
    """Parse a SigMF ``core:datatype`` string into (complex, kind, bits per component)."""
    m = _DATATYPE.match(fmt)
    if not m or int(m["bits"]) not in _VALID_BITS[m["kind"]]:
        raise ValueError(
            f"Invalid SigMF sample_format '{fmt}' (examples: ci16_le, cf32_le, ci8, cu8)"
        )
    bits = int(m["bits"])
    if bits > 8 and not m["endian"]:
        raise ValueError(f"sample_format '{fmt}' needs an endianness suffix (_le or _be)")
    if bits == 8 and m["endian"]:
        raise ValueError(f"sample_format '{fmt}': 8-bit types take no endianness suffix")
    return m["cplx"] == "c", m["kind"], bits


def bytes_per_sample(fmt: str) -> int:
    """Bytes per sample per channel (complex samples count both components)."""
    cplx, _, bits = parse_datatype(fmt)
    return (bits // 8) * (2 if cplx else 1)


@dataclass
class RecordingEstimate:
    sensor_id: str
    recording: str  # file stem, e.g. "RF-NORTH-01_ism2g4"
    bands: list[str]
    sample_rate: float
    sample_format: str
    num_channels: int
    duration_s: float

    @property
    def bytes_per_sec(self) -> float:
        return self.sample_rate * bytes_per_sample(self.sample_format) * self.num_channels

    @property
    def total_bytes(self) -> int:
        return int(self.bytes_per_sec * self.duration_s)


def recordings_for_sensor(sensor: RFSensorConfig, duration_s: float) -> list[RecordingEstimate]:
    """One recording per band for fixed/multi tuning; a single recording when scanning."""
    rx = sensor.receiver
    tuning = sensor.tuning
    if tuning.mode == TuningMode.SCANNING:
        return [
            RecordingEstimate(
                sensor_id=sensor.sensor_id,
                recording=f"{sensor.sensor_id}_scan",
                bands=[b.name for b in tuning.bands],
                sample_rate=tuning.bands[0].sample_rate,
                sample_format=rx.sample_format,
                num_channels=sensor.num_channels,
                duration_s=duration_s,
            )
        ]
    return [
        RecordingEstimate(
            sensor_id=sensor.sensor_id,
            recording=f"{sensor.sensor_id}_{b.name}",
            bands=[b.name],
            sample_rate=b.sample_rate,
            sample_format=rx.sample_format,
            num_channels=sensor.num_channels,
            duration_s=duration_s,
        )
        for b in tuning.bands
    ]


def free_bytes(path: str | Path) -> int:
    """Free space on the volume that holds ``path`` (or its nearest existing parent)."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} PB"
