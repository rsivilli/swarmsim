from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rf.models import RFScenarioConfig


class GroupMode(enum.Enum):
    FORMATION = "formation"
    SWARM = "swarm"


class DetectionMode(enum.Enum):
    BINARY = "binary"
    PROBABILISTIC = "probabilistic"


class SensorLayout(enum.Enum):
    MANUAL = "manual"
    GRID = "grid"
    HEX = "hex"
    RANDOM = "random"


class SpacingProfile(enum.Enum):
    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    RANDOM = "random"


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def to_vec3(self, z: float = 0.0) -> Vec3:
        """Convert Vec2 to Vec3 with specified z coordinate (default: 0.0 for ground level)."""
        return Vec3(self.x, self.y, z)


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float  # altitude in km

    def to_vec2(self) -> Vec2:
        """Convert Vec3 to Vec2 by dropping the z coordinate."""
        return Vec2(self.x, self.y)

    def at_ground(self) -> Vec3:
        """Return a new Vec3 at ground level (z=0)."""
        return Vec3(self.x, self.y, 0.0)


@dataclass(frozen=True)
class Area:
    width_km: float
    height_km: float


@dataclass
class ObjectGroupConfig:
    group_id: str
    count: int
    mode: GroupMode
    heading: Vec2
    speed_min: float
    speed_max: float
    spawn_center: Vec2
    spawn_radius: float
    spacing_profile: SpacingProfile
    speed: float | None = None
    spawn_altitude: float = 0.0  # Center altitude for spawning (km)
    spawn_altitude_range: float = 0.0  # Variation range (+/-) in km
    vertical_speed: float | None = None  # Vertical velocity component (km/s)


@dataclass
class SensorConfig:
    layout: SensorLayout
    detection_range: float
    positions: list[Vec3] | None = None
    count: int | None = None
    spacing: float | None = None
    sensor_altitude: float = 0.0  # Default altitude for grid/hex/random layouts (km)


@dataclass
class NetworkTopology:
    """Network hierarchy and bundling strategy."""

    bundling_mode: str = "none"  # "none" | "relay" | "time_window"
    relay_assignments: dict[int, int] = field(default_factory=dict)  # sensor_idx -> relay_idx
    bundle_window_sec: float = 1.0
    transport: str = "zenoh"


@dataclass
class CommStrategy:
    """Communication strategy definition."""

    strategy_name: str = "direct"  # "direct_zenoh", "bundled_relay", etc.
    transport_protocol: str = "zenoh_pubsub"  # "zenoh_pubsub", "rest_post"
    bundling_enabled: bool = False
    bundle_window_sec: float | None = None
    max_bundle_size: int | None = None


@dataclass
class CommunicationConfig:
    """Communication emulation configuration."""

    enabled: bool = False
    topology: NetworkTopology = field(default_factory=NetworkTopology)
    strategy: CommStrategy = field(default_factory=CommStrategy)
    sensor_metadata: list[dict] = field(default_factory=list)
    reference_origin: tuple[float, float] = (0.0, 0.0)  # (lat, lon)


@dataclass
class ScenarioConfig:
    name: str
    area: Area
    dt: float
    duration: float
    detection_mode: DetectionMode
    groups: list[ObjectGroupConfig] = field(default_factory=list)
    sensors: SensorConfig | None = None
    detection_range: float | None = None
    pd_function: str = "exponential"
    pd_half_range: float | None = None
    seed: int | None = None
    communication: CommunicationConfig | None = None
    reference_origin: tuple[float, float] | None = None  # (lat, lon) for geographic projection
    rf: RFScenarioConfig | None = None  # RF emitters / sensors (modsim rf)


@dataclass
class SimObject:
    obj_id: str
    group_id: str
    x: float
    y: float
    z: float  # altitude in km
    vx: float
    vy: float
    vz: float  # vertical velocity in km/s


@dataclass(frozen=True)
class DetectionRecord:
    timestep: int
    time_sec: float
    obj_id: str
    sensor_idx: int
    sensor_x: float
    sensor_y: float
    sensor_z: float  # sensor altitude in km
    range_km: float  # 3D distance
    detected: bool
    pd: float


@dataclass(frozen=True)
class TrackStateRecord:
    """Enriched detection with full object state for Track.json generation."""

    # Core linkage
    timestep: int
    time_sec: float
    obj_id: str
    sensor_idx: int

    # Track identity (persistent across detections)
    trk_id: str  # UUID
    trk_num: str  # J-series track number (5-char alphanumeric)
    trk_stat: str  # INITIATING | MAINTAINING | DROPPING

    # Object state at detection time (km, km/s)
    obj_x: float
    obj_y: float
    obj_z: float  # altitude in km
    obj_vx: float
    obj_vy: float
    obj_vz: float  # vertical velocity in km/s

    # Derived motion
    speed_kmps: float  # sqrt(vx² + vy² + vz²)
    heading_deg: float  # 0-360 degrees from North

    # Detection quality
    range_km: float  # 3D distance
    pd: float


@dataclass(frozen=True)
class SensorMetadata:
    """Sensor metadata for Track messages and network topology."""

    sensor_idx: int  # Links to sensors list position
    sensor_id: str  # "RADAR-01", "EO-TOWER-5"
    asset: str  # Platform/tower name
    asset_nat: str  # "US", "NATO"
    sen_type: str  # Sensor type for modType field
    network_tier: int  # 0=edge, 1=relay, 2=backbone
    relay_parent: int | None  # sensor_idx of parent (None if top-level)


@dataclass
class TimestepStats:
    timestep: int
    time_sec: float
    total_objects: int
    objects_detected: int
    coverage_pct: float
    detections_per_sensor: list[int]


@dataclass
class TimestepSnapshot:
    """Complete object state at a specific timestep."""

    timestep: int
    time_sec: float
    objects: list[SimObject]


@dataclass
class SimState:
    objects: list[SimObject]
    sensors: list[Vec3]  # Now 3D sensor positions
    sensor_range: list[float]
    timestep: int
    time_sec: float
    detection_log: list[DetectionRecord]
    timestep_stats: list[TimestepStats]

    # New fields for communication emulation (optional for backward compatibility)
    track_state_log: list[TrackStateRecord] = field(default_factory=list)
    sensor_metadata: list[SensorMetadata] = field(default_factory=list)

    # Complete object state history at every timestep
    object_snapshots: list[TimestepSnapshot] = field(default_factory=list)
