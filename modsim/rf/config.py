"""Parsing and validation of the RF sections of a scenario (``rf``, ``rf_sensors``, ``groups[].rf``)."""

from __future__ import annotations

from datetime import datetime

from ..models import Vec2, Vec3
from ..validators import suggest_typo_fixes
from .library import build_library, resolve_params
from .models import (
    AnnotationGranularity,
    ArrayConfig,
    ArrayGeometry,
    BackgroundConfig,
    BandConfig,
    ControllerConfig,
    DetectorConfig,
    GroupRFConfig,
    LinkConfig,
    ReceiverConfig,
    RFConfig,
    RFScenarioConfig,
    RFSensorConfig,
    RFSensorType,
    TuningConfig,
    TuningMode,
)
from .scheduler import check_component
from .sizing import parse_datatype

RF_KEYS = {
    "epoch",
    "window",
    "chunk_samples",
    "cull_snr_db",
    "annotations",
    "k_factor",
    "background",
    "profiles",
}
SENSOR_KEYS = {"id", "position", "type", "array", "receiver", "tuning", "detector"}
ARRAY_KEYS = {
    "geometry",
    "elements",
    "radius_m",
    "spacing_m",
    "element_positions_m",
    "boresight_deg",
    "element_gain_dbi",
}
RECEIVER_KEYS = {
    "noise_figure_db",
    "sample_format",
    "adc_bits",
    "full_scale_dbm",
    "gain_mode",
    "antenna_gain_dbi",
    "impairments",
}
TUNING_KEYS = {"mode", "bands", "dwell_ms", "retune_ms", "schedule"}
BAND_KEYS = {"name", "center_hz", "sample_rate"}
DETECTOR_KEYS = {"threshold_snr_db", "integration"}
GROUP_RF_KEYS = {"links", "controller"}
CONTROLLER_KEYS = {"placement", "position", "offset_from_spawn", "altitude_m", "links"}
LINK_KEYS = {"profile", "params"}
BACKGROUND_KEYS = {"profile", "density_per_km2", "altitude_m", "params"}


def has_rf(raw: dict) -> bool:
    return "rf" in raw or "rf_sensors" in raw or any("rf" in g for g in raw.get("groups") or [])


def _check_keys(d, allowed: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(d).__name__}")
    for key in d:
        if key not in allowed:
            raise ValueError(f"{where}: {suggest_typo_fixes(str(key), sorted(allowed))}")


def _num(d: dict, key: str, where: str, default=None, *, minimum=None, exclusive=False):
    if key not in d or d[key] is None:
        if default is None:
            raise ValueError(f"{where}: missing '{key}'")
        return default
    v = d[key]
    if isinstance(v, str):
        # PyYAML (YAML 1.1) reads exponents without a sign, e.g. 915.0e6, as strings
        try:
            v = float(v)
        except ValueError:
            pass
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{where}: '{key}' must be a number, got {v!r}")
    if minimum is not None and (v <= minimum if exclusive else v < minimum):
        op = ">" if exclusive else ">="
        raise ValueError(f"{where}: '{key}' must be {op} {minimum}, got {v}")
    return float(v)


def _vec3(raw, where: str) -> Vec3:
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError(f"{where}: position must be [x, y] or [x, y, z] (km), got {raw!r}")
    return Vec3(float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) == 3 else 0.0)


def _parse_link(raw, where: str) -> LinkConfig:
    if isinstance(raw, str):
        return LinkConfig(profile=raw)
    _check_keys(raw, LINK_KEYS, where)
    if "profile" not in raw:
        raise ValueError(f"{where}: missing 'profile'")
    return LinkConfig(profile=str(raw["profile"]), params=dict(raw.get("params") or {}))


def _parse_rf_block(raw: dict, duration: float) -> RFConfig:
    _check_keys(raw, RF_KEYS, "rf")
    epoch = str(raw.get("epoch", RFConfig.epoch))
    try:
        datetime.fromisoformat(epoch.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"rf.epoch: not an ISO 8601 timestamp: {epoch!r}") from None

    window = None
    if raw.get("window") is not None:
        w = raw["window"]
        if not (isinstance(w, (list, tuple)) and len(w) == 2):
            raise ValueError(f"rf.window must be [start, end] seconds, got {w!r}")
        w0, w1 = float(w[0]), float(w[1])
        if not 0 <= w0 < w1 <= duration:
            raise ValueError(f"rf.window must satisfy 0 <= start < end <= duration ({duration})")
        window = (w0, w1)

    annotations = raw.get("annotations") or {}
    _check_keys(annotations, {"granularity"}, "rf.annotations")
    gran = annotations.get("granularity", AnnotationGranularity.DWELL.value)
    try:
        granularity = AnnotationGranularity(gran)
    except ValueError:
        valid = [g.value for g in AnnotationGranularity]
        raise ValueError(f"rf.annotations.granularity: {suggest_typo_fixes(gran, valid)}") from None

    background = []
    for i, b in enumerate(raw.get("background") or []):
        where = f"rf.background[{i}]"
        _check_keys(b, BACKGROUND_KEYS, where)
        if "profile" not in b:
            raise ValueError(f"{where}: missing 'profile'")
        alt = b.get("altitude_m", [2.0, 15.0])
        if not (isinstance(alt, (list, tuple)) and len(alt) == 2 and 0 <= alt[0] <= alt[1]):
            raise ValueError(f"{where}: altitude_m must be [min, max] metres")
        background.append(
            BackgroundConfig(
                profile=str(b["profile"]),
                density_per_km2=_num(b, "density_per_km2", where, minimum=0),
                altitude_m=(float(alt[0]), float(alt[1])),
                params=dict(b.get("params") or {}),
            )
        )

    chunk = raw.get("chunk_samples", RFConfig.chunk_samples)
    if not isinstance(chunk, int) or chunk <= 0:
        raise ValueError(f"rf.chunk_samples must be a positive integer, got {chunk!r}")

    return RFConfig(
        epoch=epoch,
        window=window,
        chunk_samples=chunk,
        cull_snr_db=_num(raw, "cull_snr_db", "rf", RFConfig.cull_snr_db),
        annotation_granularity=granularity,
        k_factor=_num(raw, "k_factor", "rf", RFConfig.k_factor, minimum=0, exclusive=True),
        background=background,
        profiles=dict(raw.get("profiles") or {}),
    )


def _parse_array(raw: dict, where: str) -> ArrayConfig:
    _check_keys(raw, ARRAY_KEYS, where)
    geom_raw = raw.get("geometry", "uca")
    try:
        geometry = ArrayGeometry(geom_raw)
    except ValueError:
        valid = [g.value for g in ArrayGeometry]
        raise ValueError(f"{where}.geometry: {suggest_typo_fixes(geom_raw, valid)}") from None
    elements = raw.get("elements", 4)
    if not isinstance(elements, int) or elements < 1:
        raise ValueError(f"{where}.elements must be a positive integer, got {elements!r}")
    arr = ArrayConfig(
        geometry=geometry,
        elements=elements,
        radius_m=raw.get("radius_m"),
        spacing_m=raw.get("spacing_m"),
        element_positions_m=raw.get("element_positions_m"),
        boresight_deg=float(raw.get("boresight_deg", 0.0)),
        element_gain_dbi=float(raw.get("element_gain_dbi", 0.0)),
    )
    if geometry == ArrayGeometry.UCA and not (arr.radius_m and arr.radius_m > 0):
        raise ValueError(f"{where}: uca arrays need radius_m > 0")
    if geometry == ArrayGeometry.ULA and not (arr.spacing_m and arr.spacing_m > 0):
        raise ValueError(f"{where}: ula arrays need spacing_m > 0")
    if geometry == ArrayGeometry.CUSTOM:
        pos = arr.element_positions_m
        if not isinstance(pos, list) or len(pos) != elements or any(len(p) != 3 for p in pos):
            raise ValueError(
                f"{where}: custom arrays need element_positions_m with {elements} [x, y, z] entries"
            )
    return arr


def _parse_receiver(raw: dict, where: str) -> ReceiverConfig:
    _check_keys(raw, RECEIVER_KEYS, where)
    fmt = str(raw.get("sample_format", ReceiverConfig.sample_format))
    cplx, kind, bits = parse_datatype(fmt)
    if not cplx:
        raise ValueError(f"{where}.sample_format: IQ recordings need a complex type, got '{fmt}'")
    adc_bits = raw.get("adc_bits")
    if adc_bits is not None:
        if not isinstance(adc_bits, int) or adc_bits < 1:
            raise ValueError(f"{where}.adc_bits must be a positive integer")
        if kind != "f" and adc_bits > bits:
            raise ValueError(f"{where}.adc_bits ({adc_bits}) exceeds the {bits}-bit sample format")
    gain_mode = raw.get("gain_mode", "fixed")
    if gain_mode not in ("fixed", "agc"):
        raise ValueError(f"{where}.gain_mode must be 'fixed' or 'agc', got {gain_mode!r}")
    return ReceiverConfig(
        noise_figure_db=_num(raw, "noise_figure_db", where, 7.0, minimum=0),
        sample_format=fmt,
        adc_bits=adc_bits,
        full_scale_dbm=_num(raw, "full_scale_dbm", where, -10.0),
        gain_mode=gain_mode,
        antenna_gain_dbi=_num(raw, "antenna_gain_dbi", where, 0.0),
        impairments=dict(raw.get("impairments") or {}),
    )


def _parse_tuning(raw: dict, where: str) -> TuningConfig:
    _check_keys(raw, TUNING_KEYS, where)
    mode_raw = raw.get("mode", "fixed")
    try:
        mode = TuningMode(mode_raw)
    except ValueError:
        valid = [m.value for m in TuningMode]
        raise ValueError(f"{where}.mode: {suggest_typo_fixes(mode_raw, valid)}") from None

    bands = []
    for i, b in enumerate(raw.get("bands") or []):
        bw = f"{where}.bands[{i}]"
        _check_keys(b, BAND_KEYS, bw)
        if "name" not in b:
            raise ValueError(f"{bw}: missing 'name'")
        bands.append(
            BandConfig(
                name=str(b["name"]),
                center_hz=_num(b, "center_hz", bw, minimum=0, exclusive=True),
                sample_rate=_num(b, "sample_rate", bw, minimum=0, exclusive=True),
            )
        )
    if not bands:
        raise ValueError(f"{where}: at least one band is required")
    names = [b.name for b in bands]
    if len(set(names)) != len(names):
        raise ValueError(f"{where}: band names must be unique")
    if mode == TuningMode.FIXED and len(bands) != 1:
        raise ValueError(f"{where}: fixed tuning takes exactly one band (use 'multi' for more)")

    tuning = TuningConfig(mode=mode, bands=bands)
    if mode == TuningMode.SCANNING:
        if len(bands) < 2:
            raise ValueError(f"{where}: scanning needs at least two bands")
        if len({b.sample_rate for b in bands}) != 1:
            raise ValueError(
                f"{where}: all scanning bands must share one sample_rate (SigMF allows one "
                "core:sample_rate per recording); use 'multi' tuning for mixed rates"
            )
        tuning.dwell_ms = _num(raw, "dwell_ms", where, minimum=0, exclusive=True)
        tuning.retune_ms = _num(raw, "retune_ms", where, 0.0, minimum=0)
        schedule = raw.get("schedule")
        if schedule is not None and schedule != "round_robin":
            if not isinstance(schedule, list) or not schedule:
                raise ValueError(f"{where}.schedule must be 'round_robin' or a list of band names")
            for name in schedule:
                if name not in names:
                    raise ValueError(f"{where}.schedule: {suggest_typo_fixes(str(name), names)}")
            tuning.schedule = [str(n) for n in schedule]
    else:
        for key in ("dwell_ms", "retune_ms", "schedule"):
            if key in raw:
                raise ValueError(f"{where}: '{key}' only applies to scanning tuning")
    return tuning


def _parse_sensor(raw: dict, i: int) -> RFSensorConfig:
    where = f"rf_sensors[{i}]"
    _check_keys(raw, SENSOR_KEYS, where)
    if "id" not in raw:
        raise ValueError(f"{where}: missing 'id'")
    where = f"rf_sensors[{raw['id']}]"
    if "position" not in raw:
        raise ValueError(f"{where}: missing 'position'")
    type_raw = raw.get("type", "detect")
    try:
        sensor_type = RFSensorType(type_raw)
    except ValueError:
        valid = [t.value for t in RFSensorType]
        raise ValueError(f"{where}.type: {suggest_typo_fixes(type_raw, valid)}") from None

    array = _parse_array(raw["array"], f"{where}.array") if raw.get("array") else None
    if sensor_type == RFSensorType.DF and (array is None or array.elements < 2):
        raise ValueError(f"{where}: df sensors need an 'array' with at least 2 elements")
    if "tuning" not in raw:
        raise ValueError(f"{where}: missing 'tuning'")

    det_raw = raw.get("detector") or {}
    _check_keys(det_raw, DETECTOR_KEYS, f"{where}.detector")
    integration = det_raw.get("integration", "per_emission")
    if integration != "per_emission":
        raise ValueError(f"{where}.detector.integration: only 'per_emission' is supported")

    return RFSensorConfig(
        sensor_id=str(raw["id"]),
        position=_vec3(raw["position"], where),
        sensor_type=sensor_type,
        receiver=_parse_receiver(raw.get("receiver") or {}, f"{where}.receiver"),
        tuning=_parse_tuning(raw["tuning"], f"{where}.tuning"),
        detector=DetectorConfig(
            threshold_snr_db=_num(det_raw, "threshold_snr_db", f"{where}.detector", 6.0)
        ),
        array=array,
    )


def _parse_controller(raw: dict, where: str) -> ControllerConfig:
    _check_keys(raw, CONTROLLER_KEYS, where)
    placement = raw.get("placement", "per_group")
    if isinstance(placement, dict) and set(placement) == {"count"}:
        placement = placement["count"]
    if placement not in ("per_drone", "per_group") and not (
        isinstance(placement, int) and not isinstance(placement, bool) and placement > 0
    ):
        raise ValueError(
            f"{where}.placement must be 'per_drone', 'per_group' or {{count: N}}, got {placement!r}"
        )
    position = _vec3(raw["position"], where) if raw.get("position") is not None else None
    if position is not None and placement not in ("per_group", 1):
        raise ValueError(f"{where}: an explicit position needs placement 'per_group'")
    offset = None
    if raw.get("offset_from_spawn") is not None:
        o = raw["offset_from_spawn"]
        if not (isinstance(o, (list, tuple)) and len(o) == 2):
            raise ValueError(f"{where}.offset_from_spawn must be [dx, dy] km")
        if position is not None:
            raise ValueError(f"{where}: use either 'position' or 'offset_from_spawn', not both")
        offset = Vec2(float(o[0]), float(o[1]))
    altitude_m = _num(raw, "altitude_m", where, 1.5, minimum=0)
    links = [_parse_link(lk, f"{where}.links[{j}]") for j, lk in enumerate(raw.get("links") or [])]
    if not links:
        raise ValueError(f"{where}: a controller needs at least one link")
    return ControllerConfig(
        placement=placement,
        position=Vec3(position.x, position.y, position.z or altitude_m / 1000.0)
        if position
        else None,
        offset_from_spawn=offset,
        altitude_km=altitude_m / 1000.0,
        links=links,
    )


def parse_rf(raw: dict, duration: float) -> RFScenarioConfig | None:
    """Parse all RF sections of a raw scenario dict; returns None when the scenario has none."""
    if not has_rf(raw):
        return None

    rf = _parse_rf_block(raw.get("rf") or {}, duration)
    library = build_library(rf.profiles)

    def check_link(link: LinkConfig, role: str, where: str) -> None:
        if link.profile not in library:
            raise ValueError(
                f"{where}: profile {suggest_typo_fixes(link.profile, sorted(library))}"
            )
        prof = library[link.profile]
        try:
            resolved = resolve_params(prof, role, link.params)
            for comp in resolved["activity"]:
                check_component(prof, comp)
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from None

    groups: dict[str, GroupRFConfig] = {}
    for gi, g in enumerate(raw.get("groups") or []):
        if "rf" not in g:
            continue
        gid = g.get("group_id", f"#{gi}")
        where = f"groups[{gid}].rf"
        _check_keys(g["rf"], GROUP_RF_KEYS, where)
        links = [
            _parse_link(lk, f"{where}.links[{j}]")
            for j, lk in enumerate(g["rf"].get("links") or [])
        ]
        for j, lk in enumerate(links):
            check_link(lk, "drone", f"{where}.links[{j}]")
        controller = None
        if g["rf"].get("controller") is not None:
            controller = _parse_controller(g["rf"]["controller"], f"{where}.controller")
            for j, lk in enumerate(controller.links):
                check_link(lk, "controller", f"{where}.controller.links[{j}]")
        groups[gid] = GroupRFConfig(links=links, controller=controller)

    for i, bg in enumerate(rf.background):
        check_link(LinkConfig(bg.profile, bg.params), "background", f"rf.background[{i}]")

    sensors = [_parse_sensor(s, i) for i, s in enumerate(raw.get("rf_sensors") or [])]
    ids = [s.sensor_id for s in sensors]
    if len(set(ids)) != len(ids):
        raise ValueError("rf_sensors: sensor ids must be unique")

    return RFScenarioConfig(rf=rf, sensors=sensors, groups=groups, library=library)
