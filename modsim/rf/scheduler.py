"""Stage A: expand emitters and schedule their emissions.

Every random draw comes from a generator seeded by ``(scenario seed, stable id)``, so schedules
are reproducible and independent of iteration order. Both ends of a link (a drone and its
controller) share a ``link_key``; hop sequences, slot phase and channel choices are drawn from
link-keyed generators, so the two sides agree on timing and frequency.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from ..models import ScenarioConfig, Vec3
from .geometry import Trajectory
from .library import resolve_params
from .models import (
    ActivityComponent,
    EmissionTable,
    Emitter,
    EmitterKind,
    LinkProfile,
    RFScenarioConfig,
)
from .propagation import lora_time_on_air

SLOT_OWNERS = {"all", "telemetry", "uplink"}
BEACON_SEQUENCES = {"one", "all", "random_one"}


def stable_seed(*parts) -> int:
    """64-bit seed from a stable hash of the given parts (independent of PYTHONHASHSEED)."""
    digest = hashlib.blake2b("\x1f".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def make_rng(scenario_seed: int | None, *parts) -> np.random.Generator:
    return np.random.default_rng([scenario_seed or 0, stable_seed(*parts)])


# ---------------------------------------------------------------------------
# Parameter checks (shared with config validation)
# ---------------------------------------------------------------------------


def _resolve_channel(profile: LinkProfile, channel, rng: np.random.Generator | None) -> float:
    if channel == "random":
        if rng is None:
            return profile.channels_mhz[0] * 1e6
        return float(rng.choice(profile.channels_mhz)) * 1e6
    return profile.channel_hz(channel)


def _channel_list_hz(profile: LinkProfile, subset) -> np.ndarray:
    if subset is None:
        return np.asarray(profile.channels_mhz, dtype=np.float64) * 1e6
    return np.asarray([profile.channel_hz(c) for c in subset], dtype=np.float64)


def _airtime(p: dict) -> float:
    if p.get("airtime_s") is not None:
        return float(p["airtime_s"])
    return lora_time_on_air(**p["lora"])


def check_component(profile: LinkProfile, comp: ActivityComponent) -> None:
    """Raise ValueError if a (resolved) activity component is missing or has invalid params."""
    p, model, pid = comp.params, comp.model, profile.profile_id

    def need(*keys):
        for k in keys:
            if p.get(k) is None:
                raise ValueError(f"Profile '{pid}' ({model}): missing parameter '{k}'")

    def positive(*keys):
        for k in keys:
            if not isinstance(p[k], (int, float)) or p[k] <= 0:
                raise ValueError(f"Profile '{pid}' ({model}): '{k}' must be > 0, got {p[k]!r}")

    if model == "continuous":
        need("channel")
        _resolve_channel(profile, p["channel"], None)
    elif model == "periodic_hopping":
        need("packet_rate_hz", "hop_interval_packets", "slot_owner")
        positive("packet_rate_hz", "hop_interval_packets")
        if p["slot_owner"] not in SLOT_OWNERS:
            raise ValueError(
                f"Profile '{pid}': slot_owner must be one of {sorted(SLOT_OWNERS)}, "
                f"got {p['slot_owner']!r}"
            )
        if p.get("airtime_s") is None and not p.get("lora"):
            raise ValueError(f"Profile '{pid}': periodic_hopping needs 'airtime_s' or 'lora'")
        try:
            airtime = _airtime(p)
        except TypeError as e:
            raise ValueError(f"Profile '{pid}': invalid lora parameters ({e})") from None
        if airtime > 1.0 / p["packet_rate_hz"]:
            raise ValueError(
                f"Profile '{pid}': packet airtime {airtime * 1e3:.2f} ms exceeds the "
                f"{1e3 / p['packet_rate_hz']:.2f} ms packet interval"
            )
        if int(p.get("tlm_ratio") or 0) < 0:
            raise ValueError(f"Profile '{pid}': tlm_ratio must be >= 0")
        _channel_list_hz(profile, p.get("channels"))
    elif model == "framed_burst":
        need("frame_period_s", "duty_cycle", "bandwidth_mhz", "channel_plan")
        positive("frame_period_s", "bandwidth_mhz")
        if not 0 < p["duty_cycle"] <= 1:
            raise ValueError(f"Profile '{pid}': duty_cycle must be in (0, 1]")
        key = _plan_key(p["bandwidth_mhz"])
        if key not in {str(k) for k in p["channel_plan"]}:
            raise ValueError(
                f"Profile '{pid}': no channel_plan entry for bandwidth_mhz={p['bandwidth_mhz']} "
                f"(available: {', '.join(str(k) for k in p['channel_plan'])})"
            )
        channel = p.get("channel", "random")
        if channel != "random":
            _resolve_channel(profile, channel, None)
    elif model == "beacon":
        need("interval_s", "burst_s", "sequence")
        positive("interval_s", "burst_s")
        if p["sequence"] not in BEACON_SEQUENCES:
            raise ValueError(
                f"Profile '{pid}': sequence must be one of {sorted(BEACON_SEQUENCES)}, "
                f"got {p['sequence']!r}"
            )
        if p["sequence"] == "one":
            need("channel")
            _resolve_channel(profile, p["channel"], None)
    elif model == "csma_traffic":
        need("offered_load", "frame_s", "channel")
        if not 0 < p["offered_load"] < 1:
            raise ValueError(f"Profile '{pid}': offered_load must be in (0, 1)")
        fs = p["frame_s"]
        if not (isinstance(fs, (list, tuple)) and len(fs) == 2 and 0 < fs[0] <= fs[1]):
            raise ValueError(f"Profile '{pid}': frame_s must be [min, max] with 0 < min <= max")
        _resolve_channel(profile, p["channel"], None)
    elif model == "intermittent":
        need("interval_s", "burst_s")
        positive("interval_s", "burst_s")
        if float(p.get("jitter_s") or 0.0) >= p["interval_s"]:
            raise ValueError(f"Profile '{pid}': jitter_s must be smaller than interval_s")


def _plan_key(bandwidth_mhz) -> str:
    v = float(bandwidth_mhz)
    return str(int(v)) if v.is_integer() else str(v)


# ---------------------------------------------------------------------------
# Activity models -> (t_start, duration, f_center_hz, bandwidth_hz)
# ---------------------------------------------------------------------------

Rows = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def _empty_rows() -> Rows:
    e = np.empty(0, dtype=np.float64)
    return e, e, e, e


def _continuous(t0, t1, p, profile, bw, rng, chan_rng) -> Rows:
    f = _resolve_channel(profile, p["channel"], chan_rng)
    return np.array([t0]), np.array([t1 - t0]), np.array([f]), np.array([bw])


def _periodic_hopping(t0, t1, p, profile, bw, rng, chan_rng) -> Rows:
    period = 1.0 / float(p["packet_rate_hz"])
    airtime = _airtime(p)
    phase = rng.uniform(0.0, period)
    # Slots are indexed from t=0 so both link ends agree, whatever their windows
    k0 = max(math.ceil((t0 - phase) / period), 0)
    k1 = math.floor((t1 - airtime - phase) / period)
    if k1 < k0:
        return _empty_rows()
    k = np.arange(k0, k1 + 1, dtype=np.int64)

    tlm = int(p.get("tlm_ratio") or 0)
    owner = p["slot_owner"]
    if owner != "all":
        is_tlm = (k % tlm == tlm - 1) if tlm > 0 else np.zeros(len(k), dtype=bool)
        k = k[is_tlm] if owner == "telemetry" else k[~is_tlm]
        if len(k) == 0:
            return _empty_rows()

    channels = _channel_list_hz(profile, p.get("channels"))
    hop = k // int(p["hop_interval_packets"])
    n_ch = len(channels)
    n_blocks = int(hop.max()) // n_ch + 1
    # Rows are drawn sequentially, so block b is identical however many blocks are drawn
    perms = np.argsort(rng.random((n_blocks, n_ch)), axis=1)
    f = channels[perms[hop // n_ch, hop % n_ch]]
    n = len(k)
    return phase + k * period, np.full(n, airtime), f, np.full(n, bw)


def _framed_burst(t0, t1, p, profile, bw_unused, rng, chan_rng) -> Rows:
    period = float(p["frame_period_s"])
    burst = float(p["duty_cycle"]) * period
    bw = float(p["bandwidth_mhz"]) * 1e6
    plan = {str(k): v for k, v in p["channel_plan"].items()}
    plan_hz = np.asarray(plan[_plan_key(p["bandwidth_mhz"])], dtype=np.float64) * 1e6

    phase = rng.uniform(0.0, period)
    k0 = max(math.ceil((t0 - phase) / period), 0)
    k1 = math.floor((t1 - burst - phase) / period)
    if k1 < k0:
        return _empty_rows()
    t = phase + np.arange(k0, k1 + 1) * period

    channel = p.get("channel", "random")
    first = (
        float(chan_rng.choice(plan_hz))
        if channel == "random"
        else _resolve_channel(profile, channel, None)
    )
    dwell = float(p.get("channel_dwell_s") or 0.0)
    if dwell > 0:
        n = int((t1 - t0) / dwell * 2) + 8
        changes = t0 + np.concatenate([[0.0], np.cumsum(rng.exponential(dwell, n))])
        while changes[-1] < t1:
            more = changes[-1] + np.cumsum(rng.exponential(dwell, n))
            changes = np.concatenate([changes, more])
        seg_ch = np.concatenate([[first], rng.choice(plan_hz, size=len(changes) - 1)])
        f = seg_ch[np.searchsorted(changes, t, side="right") - 1]
    else:
        f = np.full(len(t), first)
    return t, np.full(len(t), burst), f, np.full(len(t), bw)


def _beacon(t0, t1, p, profile, bw, rng, chan_rng) -> Rows:
    interval = float(p["interval_s"])
    burst = float(p["burst_s"])
    jitter = float(p.get("jitter_s") or 0.0)
    sequence = p["sequence"]
    gap = float(p.get("channel_gap_s") or 0.0)
    channels = _channel_list_hz(profile, None)
    n_sub = len(channels) if sequence == "all" else 1
    event_len = n_sub * burst + (n_sub - 1) * gap

    n_est = int((t1 - t0) / interval) + 2
    steps = interval + rng.uniform(0.0, jitter, n_est) if jitter > 0 else np.full(n_est, interval)
    events = t0 + rng.uniform(0.0, interval) + np.concatenate([[0.0], np.cumsum(steps[:-1])])
    events = events[events + event_len <= t1]
    if len(events) == 0:
        return _empty_rows()

    if sequence == "all":
        offsets = np.arange(n_sub) * (burst + gap)
        t = (events[:, None] + offsets[None, :]).ravel()
        f = np.tile(channels, len(events))
    elif sequence == "random_one":
        t = events
        f = rng.choice(channels, size=len(events))
    else:
        t = events
        f = np.full(len(events), _resolve_channel(profile, p["channel"], chan_rng))
    return t, np.full(len(t), burst), f, np.full(len(t), bw)


def _csma_traffic(t0, t1, p, profile, bw, rng, chan_rng) -> Rows:
    load = float(p["offered_load"])
    fmin, fmax = (float(v) for v in p["frame_s"])
    mean_gap = (fmin + fmax) / 2 * (1 - load) / load
    f = _resolve_channel(profile, p["channel"], chan_rng)

    starts, durs = [], []
    t = t0 + rng.exponential(mean_gap)
    batch = int((t1 - t0) / ((fmin + fmax) / 2 + mean_gap) * 1.1) + 16
    while t < t1:
        d = rng.uniform(fmin, fmax, batch)
        g = rng.exponential(mean_gap, batch)
        s = t + np.concatenate([[0.0], np.cumsum(d[:-1] + g[:-1])])
        starts.append(s)
        durs.append(d)
        t = s[-1] + d[-1] + g[-1]
    s, d = np.concatenate(starts), np.concatenate(durs)
    keep = s + d <= t1
    s, d = s[keep], d[keep]
    return s, d, np.full(len(s), f), np.full(len(s), bw)


def _intermittent(t0, t1, p, profile, bw, rng, chan_rng) -> Rows:
    interval = float(p["interval_s"])
    burst = float(p["burst_s"])
    jitter = float(p.get("jitter_s") or 0.0)
    n_est = int((t1 - t0) / max(interval - jitter, 1e-9)) + 2
    steps = interval + rng.uniform(-jitter, jitter, n_est)
    t = t0 + rng.uniform(0.0, interval) + np.concatenate([[0.0], np.cumsum(steps[:-1])])
    t = t[t + burst <= t1]
    f = rng.choice(_channel_list_hz(profile, None), size=len(t))
    return t, np.full(len(t), burst), f, np.full(len(t), bw)


ACTIVITY_MODELS = {
    "continuous": _continuous,
    "periodic_hopping": _periodic_hopping,
    "framed_burst": _framed_burst,
    "beacon": _beacon,
    "csma_traffic": _csma_traffic,
    "intermittent": _intermittent,
}


def schedule_emitter(
    emitter: Emitter, emitter_idx: int, profile: LinkProfile, scenario_seed: int | None
) -> EmissionTable:
    """All emissions of one emitter within its active interval."""
    t0, t1 = emitter.active
    resolved = emitter.params
    if t1 <= t0 or profile.silent:
        return EmissionTable.empty()

    eirp = resolved["power_dbm"] + resolved["antenna_gain_dbi"]
    parts = []
    for ci, comp in enumerate(resolved["activity"]):
        rng = make_rng(scenario_seed, "link", emitter.link_key, profile.profile_id, ci)
        # Re-seeded per component: a 'random' channel resolves identically for every
        # component of the link (e.g. an AP's beacons and its traffic)
        chan_rng = make_rng(scenario_seed, "channel", emitter.link_key, profile.profile_id)
        fn = ACTIVITY_MODELS[comp.model]
        t, d, f, bw = fn(t0, t1, comp.params, profile, resolved["bandwidth_hz"], rng, chan_rng)
        parts.append((t, d, f, bw))
    if not parts:
        return EmissionTable.empty()

    t = np.concatenate([x[0] for x in parts])
    order = np.argsort(t, kind="stable")
    n = len(t)
    seed_rng = make_rng(scenario_seed, "emission", emitter.emitter_id)
    return EmissionTable(
        emitter_idx=np.full(n, emitter_idx, dtype=np.int32),
        t_start=t[order],
        duration=np.concatenate([x[1] for x in parts])[order],
        f_center_hz=np.concatenate([x[2] for x in parts])[order],
        bandwidth_hz=np.concatenate([x[3] for x in parts])[order],
        eirp_dbm=np.full(n, eirp, dtype=np.float32),
        seed=seed_rng.integers(0, 2**63, size=n, dtype=np.uint64),
    )


# ---------------------------------------------------------------------------
# Emitter expansion
# ---------------------------------------------------------------------------


def build_emitters(
    cfg: ScenarioConfig,
    rfs: RFScenarioConfig,
    trajectories: dict[str, Trajectory],
    window: tuple[float, float],
) -> list[Emitter]:
    """Drone, controller and background emitters, with resolved params and active intervals."""
    w0, w1 = window
    emitters: list[Emitter] = []
    lib = rfs.library

    for group_id, grf in rfs.groups.items():
        objs = sorted(oid for oid, tr in trajectories.items() if tr.group_id == group_id)
        sites = _controller_sites(cfg, group_id, grf, objs, trajectories)

        for i, obj_id in enumerate(objs):
            tr = trajectories[obj_id]
            active = (max(tr.t_start, w0), min(tr.t_end, w1))
            for link in grf.links:
                prof = lib[link.profile]
                emitters.append(
                    Emitter(
                        emitter_id=f"{obj_id}/{link.profile}",
                        kind=EmitterKind.DRONE,
                        profile_id=link.profile,
                        params=resolve_params(prof, "drone", link.params),
                        link_key=f"{obj_id}/{link.profile}",
                        object_id=obj_id,
                        group_id=group_id,
                        active=active,
                    )
                )
            if grf.controller is not None:
                site = sites[i]
                for link in grf.controller.links:
                    prof = lib[link.profile]
                    emitters.append(
                        Emitter(
                            emitter_id=f"{obj_id}/ctl/{link.profile}",
                            kind=EmitterKind.CONTROLLER,
                            profile_id=link.profile,
                            params=resolve_params(prof, "controller", link.params),
                            link_key=f"{obj_id}/{link.profile}",
                            object_id=obj_id,
                            group_id=group_id,
                            position=site,
                            active=active,
                        )
                    )

    for bi, bg in enumerate(rfs.rf.background):
        prof = lib[bg.profile]
        rng = make_rng(cfg.seed, "background", bi, bg.profile)
        area = cfg.area.width_km * cfg.area.height_km
        count = int(rng.poisson(bg.density_per_km2 * area))
        xs = rng.uniform(0, cfg.area.width_km, count)
        ys = rng.uniform(0, cfg.area.height_km, count)
        zs = rng.uniform(bg.altitude_m[0], bg.altitude_m[1], count) / 1000.0
        params = resolve_params(prof, "background", bg.params)
        for j in range(count):
            eid = f"bg{bi}/{bg.profile}/{j:05d}"
            emitters.append(
                Emitter(
                    emitter_id=eid,
                    kind=EmitterKind.BACKGROUND,
                    profile_id=bg.profile,
                    params=params,
                    link_key=eid,
                    position=Vec3(float(xs[j]), float(ys[j]), float(zs[j])),
                    active=(w0, w1),
                )
            )
    return emitters


def _controller_sites(cfg, group_id, grf, objs, trajectories) -> list[Vec3]:
    """Operator position for each drone of a group (index-aligned with ``objs``)."""
    ctl = grf.controller
    if ctl is None or not objs:
        return []
    group = next(g for g in cfg.groups if g.group_id == group_id)
    z = ctl.altitude_km
    off = ctl.offset_from_spawn
    ox, oy = (off.x, off.y) if off else (0.0, 0.0)

    if ctl.placement == "per_drone":
        sites = []
        for obj_id in objs:
            p0 = trajectories[obj_id].pos_km[0]
            sites.append(Vec3(float(p0[0] + ox), float(p0[1] + oy), z))
        return sites

    n_sites = 1 if ctl.placement == "per_group" else int(ctl.placement)
    if ctl.position is not None:
        base = [ctl.position]
    else:
        rng = make_rng(cfg.seed, "controller_sites", group_id)
        cx, cy = group.spawn_center.x + ox, group.spawn_center.y + oy
        base = [Vec3(cx, cy, z)]
        for _ in range(n_sites - 1):
            ang = rng.uniform(0, 2 * math.pi)
            r = group.spawn_radius * math.sqrt(rng.random())
            base.append(Vec3(cx + r * math.cos(ang), cy + r * math.sin(ang), z))
    clipped = [
        Vec3(min(max(s.x, 0.0), cfg.area.width_km), min(max(s.y, 0.0), cfg.area.height_km), s.z)
        for s in base
    ]
    return [clipped[i % len(clipped)] for i in range(len(objs))]


def schedule_all(
    emitters: list[Emitter], library: dict[str, LinkProfile], scenario_seed: int | None
) -> EmissionTable:
    return EmissionTable.concat(
        [
            schedule_emitter(em, i, library[em.profile_id], scenario_seed)
            for i, em in enumerate(emitters)
        ]
    )
