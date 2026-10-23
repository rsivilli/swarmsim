"""Data models for the RF emitter / SigMF layer.

Units: positions in km (same frame as the rest of modsim), frequencies in Hz,
times in seconds, powers in dBm, gains in dBi, unless a field name says otherwise.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np

from ..models import Vec2, Vec3


class EmitterKind(enum.Enum):
    DRONE = "drone"
    CONTROLLER = "controller"
    BACKGROUND = "background"


class RFSensorType(enum.Enum):
    DETECT = "detect"
    DF = "df"


class TuningMode(enum.Enum):
    FIXED = "fixed"
    MULTI = "multi"
    SCANNING = "scanning"


class AnnotationGranularity(enum.Enum):
    EMISSION = "emission"
    DWELL = "dwell"
    SESSION = "session"


class ArrayGeometry(enum.Enum):
    ULA = "ula"
    UCA = "uca"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Link profiles (library entries)
# ---------------------------------------------------------------------------


@dataclass
class ActivityComponent:
    """One activity model of a profile (a profile may combine several, e.g. beacons + traffic)."""

    model: (
        str  # continuous | periodic_hopping | framed_burst | beacon | csma_traffic | intermittent
    )
    params: dict = field(default_factory=dict)


@dataclass
class LinkProfile:
    """A commercial (or background) link type from the profile library."""

    profile_id: str
    description: str
    waveform: str  # ofdm | lora | fsk | fm | none
    bandwidth_hz: float
    power_dbm: float  # transmitter conducted power
    antenna_gain_dbi: float
    channels_mhz: list[float]  # channel centre frequencies
    channel_names: list[str] | None
    activity: list[ActivityComponent]
    roles: list[str]  # emitter kinds this profile may be attached to
    role_params: dict[str, dict] = field(default_factory=dict)  # per-role param overrides
    calibration: str = "provisional"
    sources: list[str] = field(default_factory=list)
    signal: dict = field(default_factory=dict)  # SigMF `signal:detail` fields for annotations
    waveform_params: dict = field(default_factory=dict)  # generator settings (see waveforms/)

    @property
    def silent(self) -> bool:
        return not self.activity

    def channel_hz(self, channel: str | float | int) -> float:
        """Resolve a channel given as a name (e.g. 'R3') or a frequency in MHz."""
        if isinstance(channel, str):
            if self.channel_names and channel in self.channel_names:
                return self.channels_mhz[self.channel_names.index(channel)] * 1e6
            raise ValueError(
                f"Profile '{self.profile_id}': unknown channel '{channel}' "
                f"(valid: {', '.join(self.channel_names or [])})"
            )
        return float(channel) * 1e6


@dataclass
class LinkConfig:
    """A link attached to an emitter in a scenario: profile + parameter overrides."""

    profile: str
    params: dict = field(default_factory=dict)


@dataclass
class ControllerConfig:
    """Ground controller(s) for a group.

    Every drone gets its own controller radio (one emitter per drone and link);
    ``placement`` only decides how many distinct operator positions they share.
    """

    placement: str | int = "per_group"  # "per_drone" | "per_group" | N operator sites
    position: Vec3 | None = None  # explicit operator position (km)
    offset_from_spawn: Vec2 | None = None  # operator offset from the group spawn centre (km)
    altitude_km: float = 0.0015
    links: list[LinkConfig] = field(default_factory=list)


@dataclass
class GroupRFConfig:
    links: list[LinkConfig] = field(default_factory=list)
    controller: ControllerConfig | None = None


@dataclass
class BackgroundConfig:
    profile: str
    density_per_km2: float
    altitude_m: tuple[float, float] = (2.0, 15.0)
    params: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RF sensors
# ---------------------------------------------------------------------------


@dataclass
class ArrayConfig:
    geometry: ArrayGeometry = ArrayGeometry.UCA
    elements: int = 4
    radius_m: float | None = None  # UCA
    spacing_m: float | None = None  # ULA
    element_positions_m: list[tuple[float, float, float]] | None = None  # custom
    boresight_deg: float = 0.0
    element_gain_dbi: float = 0.0


@dataclass
class ReceiverConfig:
    noise_figure_db: float = 7.0
    sample_format: str = "ci16_le"
    adc_bits: int | None = None
    full_scale_dbm: float | None = None  # None = auto: noise floor (over fs) + headroom_db
    headroom_db: float = 50.0
    gain_mode: str = "fixed"
    antenna_gain_dbi: float = 0.0  # single-element antenna gain (ignored when an array is set)
    impairments: dict = field(default_factory=dict)  # stored now, applied from Phase 4


@dataclass
class BandConfig:
    name: str
    center_hz: float
    sample_rate: float

    @property
    def low_hz(self) -> float:
        return self.center_hz - self.sample_rate / 2

    @property
    def high_hz(self) -> float:
        return self.center_hz + self.sample_rate / 2


@dataclass
class TuningConfig:
    mode: TuningMode
    bands: list[BandConfig]
    dwell_ms: float | None = None  # scanning
    retune_ms: float = 0.0  # scanning
    schedule: list[str] | None = None  # scanning: band order (default round robin)


@dataclass
class DetectorConfig:
    threshold_snr_db: float = 6.0


@dataclass
class RFSensorConfig:
    sensor_id: str
    position: Vec3
    sensor_type: RFSensorType
    receiver: ReceiverConfig
    tuning: TuningConfig
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    array: ArrayConfig | None = None

    @property
    def num_channels(self) -> int:
        return self.array.elements if self.array else 1

    @property
    def rx_gain_dbi(self) -> float:
        return self.array.element_gain_dbi if self.array else self.receiver.antenna_gain_dbi


# ---------------------------------------------------------------------------
# Scenario-level RF configuration
# ---------------------------------------------------------------------------


@dataclass
class RFConfig:
    epoch: str = "2026-01-01T00:00:00Z"
    window: tuple[float, float] | None = None  # (start, end) seconds; None = whole scenario
    chunk_samples: int = 4_194_304
    cull_snr_db: float = -15.0
    annotation_granularity: AnnotationGranularity = AnnotationGranularity.DWELL
    annotation_max_gap_s: float = 0.05  # dwell grouping: split when the gap exceeds this
    k_factor: float = 4.0 / 3.0  # effective earth radius factor for the radio horizon
    background: list[BackgroundConfig] = field(default_factory=list)
    profiles: dict[str, dict] = field(default_factory=dict)  # scenario-local profile overrides


@dataclass
class RFScenarioConfig:
    rf: RFConfig
    sensors: list[RFSensorConfig]
    groups: dict[str, GroupRFConfig]  # group_id -> RF config
    library: dict[str, LinkProfile]  # resolved profile library (built-ins + overrides)


# ---------------------------------------------------------------------------
# Scheduling outputs
# ---------------------------------------------------------------------------


@dataclass
class Emitter:
    emitter_id: str
    kind: EmitterKind
    profile_id: str
    params: dict  # resolved activity params (profile defaults + role + scenario overrides)
    link_key: str  # shared by both ends of a link (drone + its controller)
    object_id: str | None = None  # drone this emitter rides on / belongs to
    group_id: str | None = None
    position: Vec3 | None = None  # static position (controllers, background); None = on object
    active: tuple[float, float] = (0.0, 0.0)  # time interval the emitter can transmit


EMISSION_COLUMNS: dict[str, type] = {
    "emitter_idx": np.int32,
    "t_start": np.float64,
    "duration": np.float64,
    "f_center_hz": np.float64,
    "bandwidth_hz": np.float64,
    "eirp_dbm": np.float32,
    "seed": np.uint64,
}


@dataclass
class EmissionTable:
    """Columnar emission list (one row per burst / hop / continuous transmission).

    Columnar numpy storage keeps multi-million-row schedules cheap.
    """

    emitter_idx: np.ndarray
    t_start: np.ndarray
    duration: np.ndarray
    f_center_hz: np.ndarray
    bandwidth_hz: np.ndarray
    eirp_dbm: np.ndarray
    seed: np.ndarray

    def __len__(self) -> int:
        return len(self.t_start)

    @classmethod
    def empty(cls) -> EmissionTable:
        return cls(**{k: np.empty(0, dtype=t) for k, t in EMISSION_COLUMNS.items()})

    @classmethod
    def concat(cls, tables: list[EmissionTable]) -> EmissionTable:
        if not tables:
            return cls.empty()
        return cls(
            **{
                k: np.concatenate([getattr(t, k) for t in tables]).astype(dt, copy=False)
                for k, dt in EMISSION_COLUMNS.items()
            }
        )

    def take(self, idx: np.ndarray) -> EmissionTable:
        return EmissionTable(**{k: getattr(self, k)[idx] for k in EMISSION_COLUMNS})

    def columns(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in EMISSION_COLUMNS}
