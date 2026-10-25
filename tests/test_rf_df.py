"""Tests for RF Phase 3: antenna arrays, direction finding, multichannel SigMF."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import yaml

import modsim.rf.synth as synth
from modsim.config import load_scenario
from modsim.rf.array import (
    direction_local,
    element_offsets_world,
    element_positions_local,
    local_axes,
    mean_pattern_gain_db,
    pattern_power,
    relative_azimuth,
    steering_vector,
)
from modsim.rf.iq import build_jobs, synthesize
from modsim.rf.models import ArrayConfig, ArrayGeometry
from modsim.rf.runner import plan_rf

CW_PROFILE = {
    "description": "test carrier",
    "waveform": "cw",
    "bandwidth_hz": 10000,
    "power_dbm": 10,
    "antenna_gain_dbi": 0,
    "roles": ["drone"],
    "channels_mhz": [2440.2],
    "activity": [{"model": "continuous", "params": {"channel": 2440.2}}],
}
UCA8 = {"geometry": "uca", "elements": 8, "radius_m": 0.06, "boresight_deg": 30.0}
SENSOR_KM = (5.0, 5.0, 0.01)


def scenario(bearing_deg: float, *, dist_km=1.0, alt_km=0.1, link="cw_test", array=None,
             fs=1e6, center=2.44e9, window=0.05, impairments=None, speed=0.0):  # fmt: skip
    b = math.radians(bearing_deg)
    spawn = [SENSOR_KM[0] + dist_km * math.sin(b), SENSOR_KM[1] + dist_km * math.cos(b)]
    receiver = {"noise_figure_db": 6, "sample_format": "cf32_le"}
    if impairments:
        receiver["impairments"] = impairments
    return {
        "name": "df test",
        "area": {"width_km": 10.0, "height_km": 10.0},
        "dt": 0.5,
        "duration": 2,
        "seed": 21,
        "reference_origin": [38.0, -104.0],
        "groups": [
            {
                "group_id": "g",
                "count": 1,
                "mode": "formation",
                "heading": [1.0, 0.3],
                "speed": speed,
                "spawn_center": spawn,
                "spawn_radius": 0.0,
                "spacing_profile": "uniform",
                "spawn_altitude": alt_km,
                "rf": {"links": [link]},
            }
        ],
        "rf": {"window": [0.0, window], "profiles": {"cw_test": copy.deepcopy(CW_PROFILE)}},
        "rf_sensors": [
            {
                "id": "DF",
                "position": list(SENSOR_KM),
                "type": "df",
                "array": copy.deepcopy(array or UCA8),
                "receiver": receiver,
                "tuning": {
                    "mode": "fixed",
                    "bands": [{"name": "b", "center_hz": center, "sample_rate": fs}],
                },
            }
        ],
    }


def run(tmp_path, raw, workers=1, name="out"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_scenario(p)
    plan = plan_rf(cfg)
    results, _ = synthesize(plan, tmp_path / name, workers=workers)
    return cfg, plan, results[0]


def read_mc(res, n_ch: int) -> np.ndarray:
    raw = np.fromfile(res.data_path, dtype="<f4").astype(np.float64)
    iq = raw.reshape(-1, n_ch, 2)
    return iq[..., 0] + 1j * iq[..., 1]


def arr_cfg(raw) -> ArrayConfig:
    a = raw["rf_sensors"][0]["array"]
    return ArrayConfig(
        geometry=ArrayGeometry(a["geometry"]),
        elements=a["elements"],
        radius_m=a.get("radius_m"),
        spacing_m=a.get("spacing_m"),
        boresight_deg=a.get("boresight_deg", 0.0),
        element_pattern=a.get("element_pattern", "isotropic"),
        pattern_exponent=a.get("pattern_exponent", 1.0),
    )


def truth(plan) -> tuple[float, float]:
    """True (world azimuth, elevation) from the link table (first emission)."""
    return float(plan.links["azimuth_deg"][0]), float(plan.links["elevation_deg"][0])


def bartlett(x, arr, freq, el, step=0.05):
    r = x.T @ x.conj() / len(x)  # (N, N) covariance, R = E[x x^H]
    az = np.arange(0, 360, step)
    a = steering_vector(arr, freq, az, el)  # (A, N)
    p = np.real(np.einsum("an,nm,am->a", a.conj(), r, a))
    return az[np.argmax(p)]


def music(x, arr, freq, n_src=1, step=0.05, el_grid=None):
    r = x.T @ x.conj() / len(x)
    _, v = np.linalg.eigh(r)
    en = v[:, : arr.elements - n_src]  # noise subspace
    best, best_val = (0.0, 0.0), -np.inf
    for el in el_grid if el_grid is not None else [0.0]:
        az = np.arange(0, 360, step)
        a = steering_vector(arr, freq, az, el)
        p = 1 / np.sum(np.abs(a.conj() @ en) ** 2, axis=1)
        i = int(np.argmax(p))
        if p[i] > best_val:
            best, best_val = (az[i], el), p[i]
    return best


def ang_err(a, b):
    return abs((a - b + 180) % 360 - 180)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_local_axes_follow_boresight(self):
        x, y, z = local_axes(90.0)
        np.testing.assert_allclose(x, [1, 0, 0], atol=1e-12)  # boresight due east
        np.testing.assert_allclose(y, [0, 1, 0], atol=1e-12)  # left of east is north
        np.testing.assert_allclose(z, [0, 0, 1])

    def test_element_layouts(self):
        ula = ArrayConfig(geometry=ArrayGeometry.ULA, elements=4, spacing_m=0.1)
        np.testing.assert_allclose(element_positions_local(ula)[:, 1], [-0.15, -0.05, 0.05, 0.15])
        uca = ArrayConfig(geometry=ArrayGeometry.UCA, elements=4, radius_m=0.2)
        np.testing.assert_allclose(
            element_positions_local(uca), [[0.2, 0, 0], [0, 0.2, 0], [-0.2, 0, 0], [0, -0.2, 0]],
            atol=1e-12,
        )  # fmt: skip
        world = element_offsets_world(ArrayConfig(**{**uca.__dict__, "boresight_deg": 90.0}))
        np.testing.assert_allclose(world[0], [0.2, 0, 0], atol=1e-12)  # element 0 east

    def test_relative_bearing_convention(self):
        uca = ArrayConfig(geometry=ArrayGeometry.UCA, elements=4, radius_m=0.2, boresight_deg=30)
        assert relative_azimuth(uca, 120.0) == pytest.approx(90.0)
        # Relative azimuth 90° (clockwise) is to the right of boresight: local −Y
        np.testing.assert_allclose(direction_local(90.0, 0.0), [0, -1, 0], atol=1e-12)

    def test_steering_phase_advances_toward_emitter(self):
        ula = ArrayConfig(geometry=ArrayGeometry.ULA, elements=2, spacing_m=0.1)
        a = steering_vector(ula, 1e9, 270.0, 0.0)  # emitter to the left (+Y): element 1 nearer
        assert np.angle(a[1] / a[0]) == pytest.approx(2 * np.pi * 1e9 * 0.1 / 299_792_458.0)

    def test_cosine_pattern(self):
        uca = ArrayConfig(
            geometry=ArrayGeometry.UCA, elements=4, radius_m=0.1, element_pattern="cosine"
        )
        p = pattern_power(uca, direction_local(0.0, 0.0))
        np.testing.assert_allclose(p, [1.0, 0.01, 0.01, 0.01], atol=1e-9)  # back lobe floor
        assert float(mean_pattern_gain_db(uca, 0.0, 0.0)) == pytest.approx(
            10 * np.log10(np.mean(p))
        )


# ---------------------------------------------------------------------------
# Direction finding on synthesized IQ (Phase 3 exit criterion)
# ---------------------------------------------------------------------------


class TestDirectionFinding:
    @pytest.mark.parametrize("bearing", [17.0, 130.0, 245.0, 333.0])
    def test_bartlett_and_music_recover_cw_bearing(self, tmp_path, bearing):
        raw = scenario(bearing)
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        az_true, el_true = truth(plan)
        rel_true = float(relative_azimuth(arr, az_true))
        assert ang_err(az_true, bearing) < 0.01
        assert ang_err(bartlett(x, arr, 2440.2e6, el_true), rel_true) < 0.2
        assert ang_err(music(x, arr, 2440.2e6, el_grid=[el_true])[0], rel_true) < 0.2

    def test_azimuth_and_elevation_from_uca(self, tmp_path):
        raw = scenario(75.0, dist_km=0.8, alt_km=0.5)
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        az_true, el_true = truth(plan)
        assert el_true > 25
        az, el = music(x, arr, 2440.2e6, step=0.1, el_grid=np.arange(0, 90, 0.5))
        assert ang_err(az, float(relative_azimuth(arr, az_true))) < 0.5
        assert el == pytest.approx(el_true, abs=1.0)

    def test_wideband_wifi_bearing(self, tmp_path):
        raw = scenario(
            200.0, link={"profile": "wifi_drone", "params": {"channel": "6"}},
            fs=40e6, center=2.437e9, window=0.02,
        )  # fmt: skip
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        active = np.abs(x[:, 0]) > 3 * np.median(np.abs(x[:, 0]))
        az_true, el_true = truth(plan)
        est = bartlett(x[active], arr, 2.437e9, el_true)
        assert ang_err(est, float(relative_azimuth(arr, az_true))) < 1.0

    def test_moving_emitter_bearing_tracks_truth(self, tmp_path):
        raw = scenario(300.0, speed=0.03, window=0.5)
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        az_true, el_true = truth(plan)  # the continuous emission's mid-time bearing
        assert (
            ang_err(bartlett(x, arr, 2440.2e6, el_true), float(relative_azimuth(arr, az_true)))
            < 0.5
        )


class TestCalibration:
    def test_channel_phases_match_recorded_calibration_truth(self, tmp_path):
        raw = scenario(
            100.0, impairments={"channel_mismatch": {"gain_db_sigma": 1.0, "phase_deg_sigma": 20}}
        )
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        g = json.loads(res.meta_path.read_text())["global"]
        cal = 10 ** (np.array(g["swarmsim:channel_gain_db"]) / 20) * np.exp(
            1j * np.radians(g["swarmsim:channel_phase_deg"])
        )
        assert np.std(g["swarmsim:channel_phase_deg"]) > 5  # mismatch actually applied
        az_true, el_true = truth(plan)
        a = steering_vector(arr, 2440.2e6, float(relative_azimuth(arr, az_true)), el_true)
        r = x.T @ x.conj() / len(x)
        measured = r[:, 0] / (a * a[0].conj())  # ∝ c_i · conj(c_0)
        expected = cal * cal[0].conj()
        np.testing.assert_allclose(np.degrees(np.angle(measured / expected)), 0.0, atol=0.5)
        # Correcting with the recorded truth restores an accurate bearing
        est = bartlett(x / cal[None, :], arr, 2440.2e6, el_true)
        assert ang_err(est, float(relative_azimuth(arr, az_true))) < 0.2


class TestPatternsAndPaths:
    def test_cosine_pattern_shapes_channel_power(self, tmp_path):
        array = {**UCA8, "element_pattern": "cosine", "pattern_exponent": 2.0}
        raw = scenario(30.0, array=array, alt_km=0.01)  # due boresight, element 0 faces it
        _, plan, res = run(tmp_path, raw)
        arr = arr_cfg(raw)
        x = read_mc(res, arr.elements)
        meta = json.loads(res.meta_path.read_text())["global"]
        fs_mw = 10 ** (meta["swarmsim:full_scale_dbm"] / 10)
        noise_mw = 10 ** (meta["swarmsim:noise_power_dbm"] / 10)
        sig = np.mean(np.abs(x) ** 2, axis=0) * fs_mw - noise_mw
        az_true, el_true = truth(plan)
        d = direction_local(float(relative_azimuth(arr, az_true)), el_true)
        expected = pattern_power(arr, d)
        front = expected > 0.1
        np.testing.assert_allclose(
            10 * np.log10(sig[front] / sig[0]), 10 * np.log10(expected[front] / expected[0]),
            atol=0.3,
        )  # fmt: skip

    def test_narrowband_path_matches_exact_rendering(self, tmp_path, monkeypatch):
        raw = scenario(140.0, speed=0.02)
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        plan = plan_rf(load_scenario(p))
        (job,), _ = build_jobs(plan, tmp_path / "o")
        k = np.arange(1000, 6000)
        fast = synth._emission_signal(job, 0, k, job.sample_rate, 0)
        monkeypatch.setattr(synth, "NARROWBAND_RAD", -1.0)  # force per-element rendering
        exact = synth._emission_signal(job, 0, k, job.sample_rate, 0)
        err = np.mean(np.abs(fast - exact) ** 2) / np.mean(np.abs(exact) ** 2)
        assert 10 * np.log10(err) < -60

    def test_byte_identical_across_workers(self, tmp_path):
        raw = scenario(50.0, speed=0.02, window=0.1)
        raw["rf"]["chunk_samples"] = 16384
        _, _, r1 = run(tmp_path, raw, workers=1, name="w1")
        _, _, r2 = run(tmp_path, raw, workers=2, name="w2")
        assert r1.data_path.read_bytes() == r2.data_path.read_bytes()


class TestSigMFSpatial:
    def test_metadata(self, tmp_path):
        raw = scenario(10.0)
        raw["rf"]["annotations"] = {"granularity": "emission"}
        _, plan, res = run(tmp_path, raw)
        meta = json.loads(res.meta_path.read_text())
        g, cap, ann = meta["global"], meta["captures"][0], meta["annotations"][0]
        assert g["core:num_channels"] == 8
        assert g["spatial:num_elements"] == 8 and g["spatial:channel_index"] == 0
        assert {"name": "spatial", "version": "1.1.0", "optional": True} in g["core:extensions"]
        assert cap["spatial:aperture_azimuth"] == 30.0
        assert len(cap["spatial:element_geometry"]) == 8
        assert cap["spatial:element_geometry"][0]["point"] == [0.06, 0.0, 0.0]
        bearing = ann["spatial:signal_bearing"]
        assert bearing["azimuth"] == pytest.approx((10.0 - 30.0) % 360, abs=0.01)
        assert bearing["range"] == pytest.approx(plan.links["range_m"][0], abs=0.01)
        assert ann["spatial:emitter_location"]["type"] == "Point"
        assert res.data_path.stat().st_size == 50_000 * 8 * 8  # samples × channels × cf32

        coll = json.loads((res.meta_path.parent / "DF.sigmf-collection").read_text())["collection"]
        assert coll["spatial:element_geometry"] == cap["spatial:element_geometry"]

    def test_validates_with_sigmf(self, tmp_path):
        sigmf = pytest.importorskip("sigmf")
        _, _, res = run(tmp_path, scenario(10.0))
        handle = sigmf.fromfile(str(res.meta_path))
        handle.validate()
        assert handle.get_global_field("core:num_channels") == 8
