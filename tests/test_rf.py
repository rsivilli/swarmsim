"""Tests for the RF layer (Phase 0: profiles, config, scheduling, link budget, sizing, CLI)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import yaml

from modsim.cli import cli
from modsim.config import load_scenario
from modsim.models import Vec3
from modsim.rf.geometry import relative_geometry
from modsim.rf.library import load_builtin_profiles, resolve_params
from modsim.rf.links import _scan_observed, compute_links, emitter_states
from modsim.rf.models import Emitter, EmitterKind
from modsim.rf.propagation import (
    doppler_hz,
    fspl_db,
    lora_time_on_air,
    noise_floor_dbm,
    radio_horizon_m,
)
from modsim.rf.runner import plan_rf
from modsim.rf.scheduler import check_component, schedule_emitter
from modsim.rf.sizing import bytes_per_sample, parse_datatype, recordings_for_sensor
from conftest import PROJECT_ROOT

LIB = load_builtin_profiles()

BASE = {
    "name": "rf test",
    "area": {"width_km": 10.0, "height_km": 10.0},
    "dt": 1.0,
    "duration": 20,
    "seed": 3,
    "groups": [
        {
            "group_id": "fpv",
            "count": 2,
            "mode": "formation",
            "heading": [1.0, 0.0],
            "speed": 0.02,
            "spawn_center": [2.0, 5.0],
            "spawn_radius": 0.2,
            "spacing_profile": "uniform",
            "spawn_altitude": 0.1,
            "rf": {
                "links": [{"profile": "elrs_900"}, {"profile": "analog_fpv"}],
                "controller": {"placement": "per_group", "links": [{"profile": "elrs_900"}]},
            },
        }
    ],
    "rf_sensors": [
        {
            "id": "RF-A",
            "position": [5.0, 5.0, 0.01],
            "type": "detect",
            "tuning": {
                "mode": "fixed",
                "bands": [{"name": "ism915", "center_hz": 915.0e6, "sample_rate": 30.0e6}],
            },
        }
    ],
}


def write_scenario(tmp_path, raw, name="rf.yaml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw))
    return p


def load_raw(tmp_path, raw):
    return load_scenario(write_scenario(tmp_path, raw))


def emitter(profile_id, role="drone", active=(0.0, 10.0), link_key="x/link", eid=None, **params):
    return Emitter(
        emitter_id=eid or f"{link_key}/{role}",
        kind=EmitterKind(role),
        profile_id=profile_id,
        params=resolve_params(LIB[profile_id], role, params),
        link_key=link_key,
        position=Vec3(0.0, 0.0, 0.0),
        active=active,
    )


# ---------------------------------------------------------------------------
# Propagation primitives
# ---------------------------------------------------------------------------


class TestPropagation:
    def test_fspl_reference_value(self):
        # 1 km at 1 GHz: 20log10(1000) + 20log10(1e9) - 147.55 = 92.45 dB
        assert fspl_db(1000.0, 1e9) == pytest.approx(92.45, abs=0.01)

    def test_fspl_scales_6db_per_doubling(self):
        assert fspl_db(2000.0, 2.4e9) - fspl_db(1000.0, 2.4e9) == pytest.approx(6.02, abs=0.01)

    def test_radio_horizon(self):
        # 4/3 earth: sqrt(2 kR h) for 100 m and 10 m antennas
        expected = np.sqrt(2 * 4 / 3 * 6_371_000 * 100) + np.sqrt(2 * 4 / 3 * 6_371_000 * 10)
        assert radio_horizon_m(100.0, 10.0) == pytest.approx(expected)
        assert radio_horizon_m(100.0, 10.0) == pytest.approx(54_260, rel=0.01)

    def test_noise_floor(self):
        assert noise_floor_dbm(1e6, 0.0) == pytest.approx(-114.0)
        assert noise_floor_dbm(20e6, 7.0) == pytest.approx(-174 + 73.01 + 7, abs=0.01)

    def test_doppler_sign(self):
        # Closing (negative radial velocity) raises the received frequency
        assert doppler_hz(5.8e9, -30.0) == pytest.approx(580.4, abs=0.1)
        assert doppler_hz(5.8e9, 30.0) < 0

    def test_lora_time_on_air_matches_semtech_calculator(self):
        # SF7 / 125 kHz / 10 bytes / CR 4/5 / explicit header / CRC on = 41.216 ms
        assert lora_time_on_air(7, 125e3, 10, 8, 1, False, True) == pytest.approx(0.041216)

    def test_relative_geometry(self):
        geo = relative_geometry(
            np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]),
            np.array([[0.01, 0.0, 0.0], [0.0, -0.01, 0.0]]),
            np.array([0.0, 0.0, 0.0]),
        )
        assert geo["azimuth_deg"][0] == pytest.approx(90.0)  # due East
        assert geo["azimuth_deg"][1] == pytest.approx(0.0)  # due North
        assert geo["elevation_deg"][1] == pytest.approx(45.0)
        assert geo["radial_velocity_mps"][0] == pytest.approx(10.0)  # moving away
        assert geo["radial_velocity_mps"][1] == pytest.approx(-10.0 / np.sqrt(2))


# ---------------------------------------------------------------------------
# Profile library
# ---------------------------------------------------------------------------


class TestLibrary:
    def test_builtin_profiles_present(self):
        for pid in (
            "dji_ocusync_dl",
            "dji_ocusync_ul",
            "elrs_900",
            "elrs_2g4",
            "crossfire",
            "analog_fpv",
            "wifi_drone",
            "remote_id_ble",
            "remote_id_wifi",
            "silent",
            "wifi_ap",
            "ble_device",
        ):
            assert pid in LIB

    @pytest.mark.parametrize("pid", sorted(LIB))
    def test_profile_defaults_are_valid_for_every_role(self, pid):
        prof = LIB[pid]
        assert prof.sources or prof.silent
        for role in prof.roles:
            for comp in resolve_params(prof, role, None)["activity"]:
                check_component(prof, comp)

    def test_unknown_override_suggests_fix(self):
        with pytest.raises(ValueError, match="packet_rate_hz"):
            resolve_params(LIB["elrs_900"], "drone", {"packet_rate": 100})

    def test_role_not_allowed(self):
        with pytest.raises(ValueError, match="cannot be used on a drone"):
            resolve_params(LIB["dji_ocusync_ul"], "drone", None)

    def test_role_params_and_overrides(self):
        ctl = resolve_params(LIB["elrs_900"], "controller", {"packet_rate_hz": 50})
        assert ctl["power_dbm"] == 27
        assert ctl["activity"][0].params["slot_owner"] == "uplink"
        assert ctl["activity"][0].params["packet_rate_hz"] == 50

    def test_lora_override_merges(self):
        r = resolve_params(LIB["elrs_900"], "drone", {"lora": {"sf": 7}})
        lora = r["activity"][0].params["lora"]
        assert lora["sf"] == 7 and lora["bw_hz"] == 500000

    def test_elrs_airtime_fits_packet_interval(self):
        p = LIB["elrs_900"].activity[0].params
        assert lora_time_on_air(**p["lora"]) < 1 / p["packet_rate_hz"]


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


class TestScheduler:
    def test_hopping_link_sides_share_slots_without_collisions(self):
        drone = emitter("elrs_900", "drone", eid="d")
        ctl = emitter("elrs_900", "controller", eid="c")
        d = schedule_emitter(drone, 0, LIB["elrs_900"], 1)
        c = schedule_emitter(ctl, 1, LIB["elrs_900"], 1)
        period = 1 / 200
        kd = np.round(d.t_start / period).astype(int)
        kc = np.round(c.t_start / period).astype(int)
        assert len(np.intersect1d(kd, kc)) == 0
        assert np.all(kd % 8 == 7)  # 1:8 telemetry ratio
        assert len(kd) + len(kc) == pytest.approx(10 * 200, abs=2)
        # Same phase on both sides
        assert (d.t_start[0] % period) == pytest.approx(c.t_start[0] % period)

    def test_hop_sequence_visits_all_channels_per_cycle(self):
        e = schedule_emitter(emitter("elrs_900", "drone", slot_owner="all"), 0, LIB["elrs_900"], 1)
        hops = e.f_center_hz[::4][:40]  # hop every 4 packets; first 40 hops = one block
        assert len(np.unique(hops)) == 40

    def test_both_sides_agree_on_channel_per_slot(self):
        all_slots = schedule_emitter(
            emitter("elrs_900", "drone", eid="a", slot_owner="all"), 0, LIB["elrs_900"], 1
        )
        tlm = schedule_emitter(emitter("elrs_900", "drone", eid="d"), 0, LIB["elrs_900"], 1)
        lookup = dict(zip(np.round(all_slots.t_start, 9), all_slots.f_center_hz))
        for t, f in zip(np.round(tlm.t_start, 9), tlm.f_center_hz):
            assert lookup[t] == f

    def test_deterministic_and_seed_dependent(self):
        em = emitter("elrs_900", "drone")
        a = schedule_emitter(em, 0, LIB["elrs_900"], 5)
        b = schedule_emitter(em, 0, LIB["elrs_900"], 5)
        c = schedule_emitter(em, 0, LIB["elrs_900"], 6)
        np.testing.assert_array_equal(a.t_start, b.t_start)
        np.testing.assert_array_equal(a.seed, b.seed)
        assert not np.array_equal(a.f_center_hz, c.f_center_hz)

    def test_emissions_stay_inside_active_window(self):
        for pid in ("elrs_900", "dji_ocusync_dl", "remote_id_ble", "wifi_drone"):
            e = schedule_emitter(emitter(pid, active=(2.0, 7.0)), 0, LIB[pid], 1)
            assert len(e) > 0
            assert e.t_start.min() >= 2.0
            assert (e.t_start + e.duration).max() <= 7.0 + 1e-9

    def test_continuous_and_silent(self):
        fpv = schedule_emitter(emitter("analog_fpv", channel="R5"), 0, LIB["analog_fpv"], 1)
        assert len(fpv) == 1
        assert fpv.f_center_hz[0] == 5806e6
        assert fpv.duration[0] == pytest.approx(10.0)
        assert len(schedule_emitter(emitter("silent"), 0, LIB["silent"], 1)) == 0

    def test_ble_beacon_uses_three_channels_with_adv_delay(self):
        e = schedule_emitter(
            emitter("remote_id_ble", active=(0.0, 30.0)), 0, LIB["remote_id_ble"], 1
        )
        assert set(e.f_center_hz) == {2402e6, 2426e6, 2480e6}
        starts = e.t_start[::3]
        gaps = np.diff(starts)
        assert gaps.min() >= 0.25 and gaps.max() <= 0.26 + 1e-9
        assert len(starts) == pytest.approx(30 / 0.255, rel=0.05)

    def test_csma_offered_load(self):
        prof = LIB["wifi_ap"]
        traffic = resolve_params(prof, "background", None)
        traffic["activity"] = [traffic["activity"][1]]  # traffic only
        em = emitter("wifi_ap", "background", active=(0.0, 60.0))
        em.params = traffic
        e = schedule_emitter(em, 0, prof, 1)
        assert e.duration.sum() / 60.0 == pytest.approx(0.05, rel=0.1)
        # frames do not overlap
        assert np.all(e.t_start[1:] >= e.t_start[:-1] + e.duration[:-1] - 1e-12)

    def test_csma_window_shorter_than_first_gap(self):
        # 5% load with ~0.8 ms frames: mean gap ~15 ms, far longer than a 0.1 ms window
        traffic = resolve_params(LIB["wifi_ap"], "background", None)
        traffic["activity"] = [traffic["activity"][1]]
        for seed in range(20):
            em = emitter("wifi_ap", "background", active=(0.0, 1e-4), link_key=f"ap{seed}")
            em.params = traffic
            e = schedule_emitter(em, 0, LIB["wifi_ap"], seed)
            assert (e.t_start + e.duration).max(initial=0.0) <= 1e-4

    def test_wifi_ap_components_share_random_channel(self):
        e = schedule_emitter(
            emitter("wifi_ap", "background", link_key="bg/ap"), 0, LIB["wifi_ap"], 1
        )
        assert len(np.unique(e.f_center_hz)) == 1

    def test_framed_burst_duty_cycle(self):
        prof = LIB["dji_ocusync_dl"]
        e = schedule_emitter(emitter("dji_ocusync_dl", active=(0.0, 20.0)), 0, prof, 1)
        assert e.duration.sum() / 20.0 == pytest.approx(0.6, rel=0.01)
        assert set(e.bandwidth_hz) == {20e6}
        e40 = schedule_emitter(
            emitter("dji_ocusync_dl", bandwidth_mhz=40, channel_dwell_s=1.0), 0, prof, 1
        )
        assert set(e40.bandwidth_hz) == {40e6}
        assert set(e40.f_center_hz) <= {f * 1e6 for f in [2420, 2440, 2460, 5755, 5775, 5795, 5815]}
        assert len(set(e40.f_center_hz)) > 1  # channel changes with a 1 s mean dwell

    def test_intermittent(self):
        prof = LIB["sik_915_intermittent"]
        e = schedule_emitter(emitter("sik_915_intermittent", active=(0.0, 300.0)), 0, prof, 1)
        gaps = np.diff(e.t_start)
        assert gaps.min() >= 8.0 and gaps.max() <= 12.0
        assert len(e) == pytest.approx(30, abs=2)


# ---------------------------------------------------------------------------
# Link budget
# ---------------------------------------------------------------------------


class TestLinks:
    def test_scan_observed_fraction(self):
        # Dwell 50 ms after 1 ms retune, 3 bands -> 153 ms cycle
        obs = _scan_observed(np.array([0.0]), np.array([30.6]), 0.001, 0.05, 0.153)
        assert obs[0] == pytest.approx(50 / 153, rel=1e-3)
        inside = _scan_observed(np.array([0.010]), np.array([0.020]), 0.001, 0.05, 0.153)
        outside = _scan_observed(np.array([0.060]), np.array([0.070]), 0.001, 0.05, 0.153)
        assert inside[0] == pytest.approx(1.0)
        assert outside[0] == 0.0

    def test_single_link_snr_matches_hand_calculation(self, tmp_path):
        cfg = load_raw(tmp_path, BASE)
        sensor = cfg.rf.sensors[0]
        em = emitter("analog_fpv", channel="R5")
        em.position = Vec3(5.0, 6.0, 0.01)  # 1 km north of the sensor, same height
        e = schedule_emitter(em, 0, LIB["analog_fpv"], 1)
        sensor.tuning.bands[0].center_hz = 5.806e9
        links = compute_links(e, emitter_states(e, [em], {}), [sensor], (0, 10), -15, 4 / 3)
        expected = 25.0 - fspl_db(1000.0, 5.806e9) - noise_floor_dbm(18e6, 7.0)
        assert links["snr_db"][0] == pytest.approx(expected)
        assert links["azimuth_deg"][0] == pytest.approx(0.0)
        assert bool(links["detectable"][0])

    def test_out_of_band_and_beyond_horizon_are_dropped(self, tmp_path):
        cfg = load_raw(tmp_path, BASE)
        sensor = cfg.rf.sensors[0]  # 915 MHz band
        fpv = emitter("analog_fpv")  # 5.8 GHz: out of band
        elrs = emitter("elrs_900", "controller", eid="far")
        elrs.position = Vec3(5.0, 80.0, 0.0)  # 75 km away at ground level: beyond horizon
        near = emitter("elrs_900", "controller", eid="near")
        near.position = Vec3(5.0, 6.0, 0.0015)
        ems = [fpv, elrs, near]
        tables = [schedule_emitter(m, i, LIB[m.profile_id], 1) for i, m in enumerate(ems)]
        from modsim.rf.models import EmissionTable

        e = EmissionTable.concat(tables)
        links = compute_links(e, emitter_states(e, ems, {}), [sensor], (0, 10), -15, 4 / 3)
        seen = set(e.emitter_idx[links["emission_idx"]])
        assert seen == {2}

    def test_cull_floor(self, tmp_path):
        cfg = load_raw(tmp_path, BASE)
        sensor = cfg.rf.sensors[0]
        em = emitter("elrs_900", "controller", eid="c")
        em.position = Vec3(5.0, 15.0, 0.05)  # 10 km
        e = schedule_emitter(em, 0, LIB["elrs_900"], 1)
        lo = compute_links(e, emitter_states(e, [em], {}), [sensor], (0, 10), -15, 4 / 3)
        hi = compute_links(e, emitter_states(e, [em], {}), [sensor], (0, 10), 200, 4 / 3)
        assert len(lo["snr_db"]) == len(e)
        assert len(hi["snr_db"]) == 0


# ---------------------------------------------------------------------------
# Config parsing and validation
# ---------------------------------------------------------------------------


def _mutate(fn):
    raw = copy.deepcopy(BASE)
    fn(raw)
    return raw


class TestConfig:
    def test_scenarios_without_rf(self):
        cfg = load_scenario(PROJECT_ROOT / "scenarios" / "minimal_test.yaml")
        assert cfg.rf is None

    def test_parses_base(self, tmp_path):
        cfg = load_raw(tmp_path, BASE)
        assert cfg.rf.sensors[0].tuning.bands[0].center_hz == 915e6  # "915.0e6" string accepted
        assert [lk.profile for lk in cfg.rf.groups["fpv"].links] == ["elrs_900", "analog_fpv"]

    @pytest.mark.parametrize(
        "mutation, match",
        [
            (lambda r: r["rf_sensors"][0].update(tpye="df"), "type"),
            (lambda r: r["rf_sensors"][0].update(type="df"), "array"),
            (
                lambda r: r["groups"][0]["rf"]["links"].append({"profile": "elrs900"}),
                "elrs_900",
            ),
            (
                lambda r: r["groups"][0]["rf"]["links"].append({"profile": "dji_ocusync_ul"}),
                "cannot be used on a drone",
            ),
            (
                lambda r: r["groups"][0]["rf"]["links"].append(
                    {"profile": "analog_fpv", "params": {"channel": "Z9"}}
                ),
                "unknown channel",
            ),
            (
                lambda r: r["groups"][0]["rf"]["links"].append(
                    {"profile": "elrs_900", "params": {"packet_rate_hz": 1000}}
                ),
                "exceeds",
            ),
            (lambda r: r.update(rf={"window": [5, 50]}), "window"),
            (lambda r: r.update(rf={"epoch": "yesterday"}), "ISO 8601"),
            (
                lambda r: r["rf_sensors"][0]["tuning"]["bands"].append(
                    {"name": "b2", "center_hz": 2.4e9, "sample_rate": 1e8}
                ),
                "exactly one band",
            ),
            (
                lambda r: r["rf_sensors"][0].update(
                    tuning={
                        "mode": "scanning",
                        "dwell_ms": 10,
                        "bands": [
                            {"name": "a", "center_hz": 9.15e8, "sample_rate": 3e7},
                            {"name": "b", "center_hz": 2.44e9, "sample_rate": 1e8},
                        ],
                    }
                ),
                "share one sample_rate",
            ),
            (
                lambda r: r["rf_sensors"][0].update(receiver={"sample_format": "ri16_le"}),
                "complex",
            ),
            (
                lambda r: r["rf_sensors"][0].update(receiver={"sample_format": "ci16"}),
                "endianness",
            ),
            (lambda r: r["rf_sensors"].append(copy.deepcopy(r["rf_sensors"][0])), "unique"),
            (
                lambda r: r["groups"][0]["rf"]["controller"].update(placement="everywhere"),
                "placement",
            ),
            (
                lambda r: r.update(
                    rf={"background": [{"profile": "elrs_900", "density_per_km2": 1}]}
                ),
                "cannot be used on a background",
            ),
        ],
    )
    def test_invalid_configs(self, tmp_path, mutation, match):
        with pytest.raises(ValueError, match=match):
            load_raw(tmp_path, _mutate(mutation))

    def test_scenario_profile_override_with_base(self, tmp_path):
        raw = _mutate(
            lambda r: r.update(
                rf={"profiles": {"loud_fpv": {"base": "analog_fpv", "power_dbm": 30}}}
            )
        )
        raw["groups"][0]["rf"]["links"] = [{"profile": "loud_fpv"}]
        cfg = load_raw(tmp_path, raw)
        assert cfg.rf.library["loud_fpv"].power_dbm == 30
        assert cfg.rf.library["loud_fpv"].channel_names == LIB["analog_fpv"].channel_names


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


class TestSizing:
    def test_datatypes(self):
        assert bytes_per_sample("ci16_le") == 4
        assert bytes_per_sample("cf32_le") == 8
        assert bytes_per_sample("ci8") == 2
        assert bytes_per_sample("cf64_be") == 16
        assert parse_datatype("cu8") == (True, "u", 8)
        with pytest.raises(ValueError):
            parse_datatype("cf16_le")

    def test_recording_estimates(self, tmp_path):
        raw = _mutate(
            lambda r: r["rf_sensors"].append(
                {
                    "id": "DF",
                    "position": [1, 1, 0.01],
                    "type": "df",
                    "array": {"geometry": "uca", "elements": 4, "radius_m": 0.08},
                    "tuning": {
                        "mode": "scanning",
                        "dwell_ms": 50,
                        "retune_ms": 1,
                        "bands": [
                            {"name": "a", "center_hz": 2.44e9, "sample_rate": 1e8},
                            {"name": "b", "center_hz": 5.8e9, "sample_rate": 1e8},
                        ],
                    },
                }
            )
        )
        cfg = load_raw(tmp_path, raw)
        fixed, scan = cfg.rf.sensors
        r = recordings_for_sensor(fixed, 10.0)
        assert len(r) == 1 and r[0].total_bytes == 30e6 * 4 * 10
        r = recordings_for_sensor(scan, 300.0)
        assert len(r) == 1 and r[0].recording == "DF_scan"
        # Design doc table: 4-element DF, 100 MS/s, ci16 -> 1.6 GB/s, 480 GB for 5 minutes
        assert r[0].bytes_per_sec == 1.6e9
        assert r[0].total_bytes == 480e9


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_plan_includes_drone_controller_and_background(self, tmp_path):
        raw = _mutate(
            lambda r: r.update(
                rf={"background": [{"profile": "ble_device", "density_per_km2": 0.2}]}
            )
        )
        plan = plan_rf(load_raw(tmp_path, raw))
        kinds = {e.kind for e in plan.emitters}
        assert kinds == {EmitterKind.DRONE, EmitterKind.CONTROLLER, EmitterKind.BACKGROUND}
        # Two drones share one operator site with per_group placement
        ctl = [e for e in plan.emitters if e.kind == EmitterKind.CONTROLLER]
        assert len(ctl) == 2 and ctl[0].position == ctl[1].position
        assert len(plan.links["snr_db"]) > 0

    def test_moving_drone_positions_follow_trajectory(self, tmp_path):
        plan = plan_rf(load_raw(tmp_path, BASE))
        idx = next(i for i, e in enumerate(plan.emitters) if e.emitter_id == "fpv-0000/elrs_900")
        rows = plan.emissions.emitter_idx == idx
        x = plan.states.pos_km[rows, 0]
        t = plan.emissions.t_start[rows] + plan.emissions.duration[rows] / 2
        slope = np.polyfit(t, x, 1)[0]
        assert slope == pytest.approx(0.02, rel=1e-6)

    def test_window_and_sensor_filters(self, tmp_path):
        cfg = load_raw(tmp_path, BASE)
        plan = plan_rf(cfg, window=(5.0, 10.0), sensor_ids=["RF-A"])
        assert plan.emissions.t_start.min() >= 5.0
        assert plan.recordings[0].duration_s == 5.0

    def test_cli_dry_run_writes_truth_and_manifest(self, tmp_path, cli_runner):
        path = write_scenario(tmp_path, BASE)
        out = tmp_path / "out"
        result = cli_runner.invoke(
            cli, ["rf", str(path), "--dry-run", "--out", str(out), "--truth-format", "csv"]
        )
        assert result.exit_code == 0, result.output
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["emitters"]["by_kind"] == {"drone": 4, "controller": 2}
        assert manifest["recordings"][0]["bytes"] == 30e6 * 4 * 20
        for name in ("emitters", "emissions", "links"):
            assert (out / "truth" / f"{name}.csv").exists()

    def test_cli_parquet_truth(self, tmp_path, cli_runner):
        pq = pytest.importorskip("pyarrow.parquet")
        path = write_scenario(tmp_path, BASE)
        out = tmp_path / "out"
        result = cli_runner.invoke(cli, ["rf", str(path), "--dry-run", "--out", str(out)])
        assert result.exit_code == 0, result.output
        table = pq.read_table(out / "truth" / "links.parquet")
        assert "snr_db" in table.column_names

    def test_cli_validate_prints_rf_summary(self, tmp_path, cli_runner):
        result = cli_runner.invoke(cli, ["validate", str(write_scenario(tmp_path, BASE))])
        assert result.exit_code == 0, result.output
        assert "RF: 1 sensor(s)" in result.output
        assert "RF IQ estimate: 2.4 GB" in result.output

    def test_bundled_rf_scenario_plans(self):
        cfg = load_scenario(PROJECT_ROOT / "scenarios" / "rf_fpv_mix.yaml")
        plan = plan_rf(cfg, window=(0.0, 5.0))
        profiles = {plan.emitters[i].profile_id for i in np.unique(plan.emissions.emitter_idx)}
        assert {"elrs_900", "analog_fpv", "dji_ocusync_dl", "remote_id_ble"} <= profiles
        assert int(plan.links["detectable"].sum()) > 0
