"""Link profile library: loading built-in profiles and resolving per-emitter parameters."""

from __future__ import annotations

import copy
from importlib.resources import files

import yaml

from ..validators import suggest_typo_fixes
from .models import ActivityComponent, LinkProfile

# Allowed parameters per activity model (validated when profiles load and when
# scenarios override them)
ACTIVITY_PARAMS: dict[str, set[str]] = {
    "continuous": {"channel"},
    "periodic_hopping": {
        "packet_rate_hz",
        "hop_interval_packets",
        "tlm_ratio",
        "slot_owner",
        "airtime_s",
        "lora",
        "channels",
    },
    "framed_burst": {
        "frame_period_s",
        "duty_cycle",
        "bandwidth_mhz",
        "channel",
        "channel_dwell_s",
        "channel_plan",
    },
    "beacon": {"interval_s", "burst_s", "jitter_s", "sequence", "channel", "channel_gap_s"},
    "csma_traffic": {"offered_load", "frame_s", "channel"},
    "intermittent": {"interval_s", "burst_s", "jitter_s"},
}

# Profile-level fields a scenario may override through ``params``
PROFILE_LEVEL_PARAMS = {"power_dbm", "antenna_gain_dbi", "bandwidth_hz"}

ROLES = {"drone", "controller", "background"}


def _expand_channels(raw) -> list[float]:
    if isinstance(raw, dict):
        start, step, count = float(raw["start"]), float(raw["step"]), int(raw["count"])
        return [round(start + i * step, 6) for i in range(count)]
    return [float(c) for c in raw or []]


def parse_profile(raw: dict) -> LinkProfile:
    """Build a LinkProfile from its YAML dictionary, validating its structure."""
    pid = raw.get("id")
    if not pid:
        raise ValueError("Link profile is missing 'id'")

    activity = []
    for comp in raw.get("activity") or []:
        model = comp.get("model")
        if model not in ACTIVITY_PARAMS:
            raise ValueError(
                f"Profile '{pid}': {suggest_typo_fixes(str(model), sorted(ACTIVITY_PARAMS))} "
                "(activity model)"
            )
        params = dict(comp.get("params") or {})
        for key in params:
            if key not in ACTIVITY_PARAMS[model]:
                raise ValueError(
                    f"Profile '{pid}', model '{model}': "
                    f"{suggest_typo_fixes(key, sorted(ACTIVITY_PARAMS[model]))}"
                )
        activity.append(ActivityComponent(model=model, params=params))

    roles = list(raw.get("roles") or ["drone"])
    for role in roles:
        if role not in ROLES:
            raise ValueError(f"Profile '{pid}': unknown role '{role}'")

    channels = _expand_channels(raw.get("channels_mhz"))
    names = raw.get("channel_names")
    if names is not None:
        names = [str(n) for n in names]
        if len(names) != len(channels):
            raise ValueError(f"Profile '{pid}': channel_names and channels_mhz lengths differ")
    if activity and not channels:
        raise ValueError(f"Profile '{pid}': emitting profiles need channels_mhz")

    return LinkProfile(
        profile_id=pid,
        description=raw.get("description", ""),
        waveform=raw.get("waveform", "none"),
        bandwidth_hz=float(raw.get("bandwidth_hz", 0.0)),
        power_dbm=float(raw.get("power_dbm", 0.0)),
        antenna_gain_dbi=float(raw.get("antenna_gain_dbi", 0.0)),
        channels_mhz=channels,
        channel_names=names,
        activity=activity,
        roles=roles,
        role_params={k: dict(v or {}) for k, v in (raw.get("role_params") or {}).items()},
        calibration=raw.get("calibration", "provisional"),
        sources=list(raw.get("sources") or []),
        signal=dict(raw.get("signal") or {}),
    )


def load_builtin_profiles() -> dict[str, LinkProfile]:
    """Load every profile shipped in modsim/rf/profiles/*.yaml."""
    library: dict[str, LinkProfile] = {}
    for entry in sorted(files("modsim.rf").joinpath("profiles").iterdir(), key=lambda p: p.name):
        if not entry.name.endswith(".yaml"):
            continue
        profile = parse_profile(yaml.safe_load(entry.read_text()))
        library[profile.profile_id] = profile
    return library


def build_library(overrides: dict[str, dict] | None = None) -> dict[str, LinkProfile]:
    """Built-in profiles plus scenario-local profiles from ``rf.profiles``.

    A scenario profile with an existing id replaces the built-in one; a new id adds a profile.
    An override may set ``base: <profile id>`` to start from an existing profile and change
    only some fields.
    """
    library = load_builtin_profiles()
    for pid, raw in (overrides or {}).items():
        raw = dict(raw or {})
        base = raw.pop("base", None)
        if base is not None:
            if base not in library:
                raise ValueError(
                    f"rf.profiles.{pid}: base {suggest_typo_fixes(base, sorted(library))}"
                )
            merged = _profile_to_raw(library[base])
            merged.update(raw)
            raw = merged
        raw["id"] = pid
        library[pid] = parse_profile(raw)
    return library


def _profile_to_raw(p: LinkProfile) -> dict:
    return {
        "id": p.profile_id,
        "description": p.description,
        "waveform": p.waveform,
        "bandwidth_hz": p.bandwidth_hz,
        "power_dbm": p.power_dbm,
        "antenna_gain_dbi": p.antenna_gain_dbi,
        "channels_mhz": list(p.channels_mhz),
        "channel_names": list(p.channel_names) if p.channel_names else None,
        "activity": [{"model": c.model, "params": copy.deepcopy(c.params)} for c in p.activity],
        "roles": list(p.roles),
        "role_params": copy.deepcopy(p.role_params),
        "calibration": p.calibration,
        "sources": list(p.sources),
        "signal": dict(p.signal),
    }


def resolve_params(profile: LinkProfile, role: str, overrides: dict | None) -> dict:
    """Merge profile defaults, role defaults and scenario overrides.

    Returns ``{"power_dbm", "antenna_gain_dbi", "bandwidth_hz", "activity": [ActivityComponent]}``.
    Each override key must be a profile-level parameter or a parameter of at least one of the
    profile's activity components; ``lora`` dictionaries are merged key by key.
    """
    if role not in profile.roles:
        raise ValueError(
            f"Profile '{profile.profile_id}' cannot be used on a {role} "
            f"(allowed: {', '.join(profile.roles)})"
        )

    level = {
        "power_dbm": profile.power_dbm,
        "antenna_gain_dbi": profile.antenna_gain_dbi,
        "bandwidth_hz": profile.bandwidth_hz,
    }
    activity = [ActivityComponent(c.model, copy.deepcopy(c.params)) for c in profile.activity]

    for source in (profile.role_params.get(role, {}), overrides or {}):
        for key, value in source.items():
            if key in PROFILE_LEVEL_PARAMS:
                level[key] = float(value)
                continue
            targets = [c for c in activity if key in ACTIVITY_PARAMS[c.model]]
            if not targets:
                valid = sorted(
                    PROFILE_LEVEL_PARAMS | {k for c in activity for k in ACTIVITY_PARAMS[c.model]}
                )
                raise ValueError(
                    f"Profile '{profile.profile_id}': parameter {suggest_typo_fixes(key, valid)}"
                )
            for comp in targets:
                if key == "lora" and isinstance(value, dict):
                    comp.params["lora"] = {**comp.params.get("lora", {}), **value}
                else:
                    comp.params[key] = copy.deepcopy(value)

    return {**level, "activity": activity}
