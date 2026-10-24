"""Tests for RF Phase 4: ground reflection, shadowing, Rician fading, receiver impairments,
AGC and scanning tuning."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import yaml

from modsim.config import load_scenario
from modsim.rf.channel import (
    diffuse_gain,
    reflection_coefficient,
    rician_gain,
    shadowing_db,
    two_ray_factor,
)
from modsim.rf.iq import synthesize
from modsim.rf.models import PropagationConfig
from modsim.rf.propagation import C_MPS
from modsim.rf.runner import plan_rf

CW = {
    "description": "test carrier",
    "waveform": "cw",
    "bandwidth_hz": 10000,
    "power_dbm": 30,
    "antenna_gain_dbi": 0,
    "roles": ["drone"],
    "channels_mhz": [915.2],
    "activity": [{"model": "continuous", "params": {"channel": 915.2}}],
}


def scenario(*, spawn=(3.0, 5.0), alt_km=0.1, speed=0.0, heading=(1.0, 0.0), duration=2,
             window=None, rf_extra=None, receiver=None, tuning=None, sensor_alt_km=0.01):  # fmt: skip
    rx = {"noise_figure_db": 6, "sample_format": "cf32_le"}
    rx.update(receiver or {})
    rf = {"profiles": {"cw_test": copy.deepcopy(CW)}}
    if window:
        rf["window"] = list(window)
    rf.update(rf_extra or {})
    return {
        "name": "channel test",
        "area": {"width_km": 20.0, "height_km": 10.0},
        "dt": 0.5,
        "duration": duration,
        "seed": 5,
        "groups": [
            {
                "group_id": "g",
                "count": 1,
                "mode": "formation",
                "heading": list(heading),
                "speed": speed,
                "spawn_center": list(spawn),
                "spawn_radius": 0.0,
                "spacing_profile": "uniform",
                "spawn_altitude": alt_km,
                "rf": {"links": ["cw_test"]},
            }
        ],
        "rf": rf,
        "rf_sensors": [
            {
                "id": "S",
                "position": [1.0, 5.0, sensor_alt_km],
                "type": "detect",
                "receiver": rx,
                "tuning": tuning
                or {
                    "mode": "fixed",
                    "bands": [{"name": "b", "center_hz": 915.0e6, "sample_rate": 1e6}],
                },
            }
        ],
    }


def run(tmp_path, raw, workers=1, name="out"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(raw))
    plan = plan_rf(load_scenario(p))
    results, _ = synthesize(plan, tmp_path / name, workers=workers)
    return plan, results[0]


def read(res) -> np.ndarray:
    raw = np.fromfile(res.data_path, dtype="<f4").astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def meta(res) -> dict:
    return json.loads(res.meta_path.read_text())


def fs_mw(m) -> float:
    return 10 ** (m["global"]["swarmsim:full_scale_dbm"] / 10)


def tone_power_dbm(x, fs, f0, m, half_width=20.0):
    spec = np.abs(np.fft.fft(x)) ** 2 / len(x) ** 2
    f = np.fft.fftfreq(len(x), 1 / fs)
    return 10 * np.log10(spec[np.abs(f - f0) < half_width].sum() * fs_mw(m))


def baseband_envelope(x, fs, f0, block):
    """Mix the tone at f0 to DC and average blocks (removes noise, keeps slow fading)."""
    y = x * np.exp(-2j * np.pi * f0 * np.arange(len(x)) / fs)
    n = len(y) // block * block
    return y[:n].reshape(-1, block).mean(axis=1)


# ---------------------------------------------------------------------------
# Channel primitives
# ---------------------------------------------------------------------------


class TestPrimitives:
    def test_reflection_coefficient_limits(self):
        prop = PropagationConfig()
        for pol in ("vertical", "horizontal"):
            prop.polarization = pol
            g = reflection_coefficient(np.radians(0.01), 915e6, prop)
            assert abs(g + 1) < 0.01  # grazing incidence: Γ → −1
        prop.polarization = "vertical"
        mags = np.abs(reflection_coefficient(np.radians(np.linspace(0.1, 89, 400)), 915e6, prop))
        assert mags.min() < 0.3  # pseudo-Brewster dip for vertical polarization

    def test_two_ray_matches_two_path_geometry(self):
        prop = PropagationConfig(ground_permittivity=1e9, polarization="horizontal")  # Γ ≈ −1
        h1, h2, f = 100.0, 10.0, 915e6
        d = np.array([2000.0, 5000.0, 20000.0])
        e = np.column_stack([d / 1000, np.zeros(3), np.full(3, h1 / 1000)])
        power = np.abs(two_ray_factor(e, np.array([0.0, 0.0, h2 / 1000]), f, prop)) ** 2
        direct = np.hypot(d, h1 - h2)
        refl = np.hypot(d, h1 + h2)
        exact = np.abs(1 - direct / refl * np.exp(-2j * np.pi * f * (refl - direct) / C_MPS)) ** 2
        np.testing.assert_allclose(power, exact, rtol=1e-3)
        # Far field: the familiar 4·sin²(2π h1 h2 / (λ d))
        far = 4 * np.sin(2 * np.pi * h1 * h2 * f / (C_MPS * d[1:])) ** 2
        np.testing.assert_allclose(power[1:], far, rtol=0.02)

    def test_shadowing_statistics(self):
        prop = PropagationConfig(shadowing_db=6.0, shadowing_decorrelation_m=50.0)
        a = np.array([shadowing_db(s, 0.0, prop) for s in range(2000)])
        b = np.array([shadowing_db(s, 50.0, prop) for s in range(2000)])
        assert np.std(a) == pytest.approx(6.0, rel=0.08)
        assert np.corrcoef(a, b)[0, 1] == pytest.approx(math.exp(-1), abs=0.08)

    def test_diffuse_component_unit_power_and_doppler_limited(self):
        prop = PropagationConfig(fading_paths=32)
        t = np.arange(0, 20, 1e-3)
        g = diffuse_gain(3, t, 50.0, 2.4e9, prop)[:, 0]
        assert np.mean(np.abs(g) ** 2) == pytest.approx(1.0, rel=0.15)
        spec = np.abs(np.fft.fft(g)) ** 2
        f = np.fft.fftfreq(len(g), 1e-3)
        assert spec[np.abs(f) > 51].sum() / spec.sum() < 1e-3

    def test_rician_mean_power(self):
        prop = PropagationConfig(fading_paths=32)
        h = rician_gain(7, np.arange(0, 50, 2e-3), 40.0, 2.4e9, 10**0.6, prop)
        assert np.mean(np.abs(h) ** 2) == pytest.approx(1.0, rel=0.1)


# ---------------------------------------------------------------------------
# Propagation in synthesized recordings
# ---------------------------------------------------------------------------


class TestPropagationIQ:
    @pytest.mark.parametrize("dist_km", [1.0, 2.0, 3.3])
    def test_two_ray_power_matches_link_budget(self, tmp_path, dist_km):
        raw = scenario(
            spawn=(1.0 + dist_km, 5.0), window=(0.0, 0.2),
            rf_extra={"propagation": {"ground_reflection": True}},
        )  # fmt: skip
        plan, res = run(tmp_path, raw)
        m = meta(res)
        assert plan.links["two_ray_db"][0] != 0
        measured = tone_power_dbm(read(res), 1e6, 200e3, m)
        assert measured == pytest.approx(plan.links["rx_power_dbm"][0], abs=0.5)

    def test_shadowing_applied_consistently(self, tmp_path):
        raw = scenario(window=(0.0, 0.2), rf_extra={"propagation": {"shadowing_db": 8.0}})
        plan, res = run(tmp_path, raw)
        shadow = plan.links["shadow_db"][0]
        assert abs(shadow) > 0.1
        measured = tone_power_dbm(read(res), 1e6, 200e3, meta(res))
        assert measured == pytest.approx(plan.links["rx_power_dbm"][0], abs=0.5)

    def test_rician_k_factor_recovered_from_envelope(self, tmp_path):
        raw = scenario(
            speed=0.05, duration=12, window=(0.0, 10.0),
            rf_extra={"propagation": {"fading": "rician", "rician_k_db": 6.0, "fading_paths": 32}},
        )  # fmt: skip
        _, res = run(tmp_path, raw, workers=2)
        x = read(res)
        f_d = 50.0 * 915.2e6 / C_MPS  # ≈ 153 Hz
        env = baseband_envelope(x, 1e6, 200e3 - 915.2e6 * 50.0 / C_MPS, 200)  # 0.2 ms blocks
        r2 = np.abs(env) ** 2
        gamma = np.var(r2) / np.mean(r2) ** 2
        k_est = math.sqrt(1 - gamma) / (1 - math.sqrt(1 - gamma))
        assert 10 * math.log10(k_est) == pytest.approx(6.0, abs=1.5)
        # Fading spectrum stays within the Doppler spread around the (Doppler-shifted) tone
        spec = np.abs(np.fft.fft(env - env.mean())) ** 2
        f = np.fft.fftfreq(len(env), 200 / 1e6)
        assert spec[np.abs(f) > 1.2 * f_d].sum() / spec.sum() < 0.02

    def test_fade_truth_matches_recording(self, tmp_path):
        raw = scenario(
            speed=0.03, duration=3, window=(0.0, 2.0),
            rf_extra={"propagation": {"fading": "rician", "rician_k_db": 0.0}},
        )  # fmt: skip
        plan, res = run(tmp_path, raw)
        x = read(res)
        fade = plan.links["fade_db"][0]  # at the continuous emission's mid-time (t = 1 s)
        env = baseband_envelope(x, 1e6, 200e3 + plan.links["doppler_hz"][0], 100)
        mid = len(env) // 2
        local = 10 * np.log10(np.mean(np.abs(env[mid - 2 : mid + 2]) ** 2) * fs_mw(meta(res)))
        assert local - plan.links["rx_power_dbm"][0] == pytest.approx(fade, abs=1.0)


# ---------------------------------------------------------------------------
# Receiver impairments and AGC
# ---------------------------------------------------------------------------


class TestReceiverImpairments:
    def test_lo_offset_shifts_tone_by_recorded_amount(self, tmp_path):
        raw = scenario(window=(0.0, 1.0), receiver={"impairments": {"lo_offset_ppm": 2.0}})
        _, res = run(tmp_path, raw)
        m = meta(res)
        shift = m["global"]["swarmsim:lo_offset_hz"]
        assert 0 < abs(shift) <= 2e-6 * 915e6
        x = read(res)
        n = len(x) * 4
        spec = np.abs(np.fft.fft(x, n)) ** 2
        f = np.fft.fftfreq(n, 1e-6)
        band = np.abs(f - 200e3) < 3000
        peak = f[band][np.argmax(spec[band])]
        assert peak - 200e3 == pytest.approx(-shift, abs=1.0)

    def test_phase_noise_variance_and_continuity(self, tmp_path):
        raw = scenario(
            window=(0.0, 0.5), receiver={"impairments": {"phase_noise": {"linewidth_hz": 500}}},
            rf_extra={"chunk_samples": 50_000},
        )  # fmt: skip
        _, res = run(tmp_path, raw, workers=2)
        x = read(res)
        y = x * np.exp(-2j * np.pi * 200e3 * np.arange(len(x)) / 1e6)
        block = 50
        z = y[: len(y) // block * block].reshape(-1, block).mean(axis=1)  # 50 µs steps
        dphi = np.angle(z[1:] / z[:-1])
        # Difference of consecutive L-sample means of a Wiener phase: σ²·(2L² + 1)/(3L)
        expected = 2 * np.pi * 500 / 1e6 * (2 * block**2 + 1) / (3 * block)
        assert np.var(dphi) == pytest.approx(expected, rel=0.2)
        # No extra jump at chunk boundaries (every 1000 steps)
        at_boundary = dphi[999::1000]
        assert np.max(np.abs(at_boundary)) < 5 * math.sqrt(expected)

    def test_iq_imbalance_image(self, tmp_path):
        g_db, phi_deg = 0.5, 3.0
        raw = scenario(
            window=(0.0, 0.2),
            receiver={"impairments": {"iq_imbalance": {"gain_db": g_db, "phase_deg": phi_deg}}},
        )  # fmt: skip
        _, res = run(tmp_path, raw)
        x = read(res)
        spec = np.abs(np.fft.fft(x)) ** 2
        f = np.fft.fftfreq(len(x), 1e-6)
        main = spec[np.abs(f - 200e3) < 20].sum()
        image = spec[np.abs(f + 200e3) < 20].sum()
        g, phi = 10 ** (g_db / 20), np.radians(phi_deg)
        expected = abs((1 - g * np.exp(1j * phi)) / (1 + g * np.exp(-1j * phi))) ** 2
        assert 10 * np.log10(image / main) == pytest.approx(10 * np.log10(expected), abs=0.5)

    def test_dc_offset(self, tmp_path):
        raw = scenario(window=(0.0, 0.2), receiver={"impairments": {"dc_offset_dbfs": -30}})
        _, res = run(tmp_path, raw)
        x = read(res)
        assert 20 * np.log10(np.abs(x.mean())) == pytest.approx(-30.0, abs=0.2)
        assert meta(res)["global"]["swarmsim:dc_offset_dbfs"] == [-30.0]

    def test_agc_levels_blocks_and_records_trace(self, tmp_path):
        raw = scenario(
            window=(0.0, 0.5),
            receiver={"gain_mode": "agc", "agc": {"target_dbfs": -12, "block_samples": 4096}},
            rf_extra={"chunk_samples": 65536},
        )  # fmt: skip
        _, res = run(tmp_path, raw, workers=2)
        m = meta(res)["global"]
        trace = np.load(res.data_path.parent / m["swarmsim:agc_trace"])
        assert len(trace) == math.ceil(500_000 / 4096)
        x = read(res)
        rms = 10 * np.log10(np.mean(np.abs(x[: 4096 * 100].reshape(100, 4096)) ** 2, axis=1))
        np.testing.assert_allclose(rms, -12.0, atol=0.6)  # 1 dB gain steps


# ---------------------------------------------------------------------------
# Scanning tuning
# ---------------------------------------------------------------------------

SCAN = {
    "mode": "scanning",
    "dwell_ms": 20,
    "retune_ms": 2,
    "bands": [
        {"name": "a", "center_hz": 915.0e6, "sample_rate": 1e6},
        {"name": "c", "center_hz": 925.0e6, "sample_rate": 1e6},
    ],
}


class TestScanning:
    def test_captures_retunes_and_dwells(self, tmp_path):
        raw = scenario(window=(0.0, 0.2), tuning=SCAN)
        raw["rf"]["annotations"] = {"granularity": "emission"}
        plan, res = run(tmp_path, raw)
        m = meta(res)
        caps = m["captures"]
        slot = 22_000  # samples per slot (2 ms retune + 20 ms dwell)
        assert [c["core:sample_start"] for c in caps[:4]] == [0, slot, 2 * slot, 3 * slot]
        assert [c["core:frequency"] for c in caps[:4]] == [915e6, 925e6, 915e6, 925e6]
        assert caps[1]["core:datetime"] == "2026-01-01T00:00:00.022000Z"

        x = read(res)
        retunes = [a for a in m["annotations"] if a.get("swarmsim:retune")]
        assert len(retunes) == len(caps)
        for r in retunes:
            s = r["core:sample_start"]
            assert r["core:sample_count"] == 2000
            assert not np.any(x[s : s + 2000])  # zero-filled

        # The 915.2 MHz carrier is present only during band-a dwells, at +200 kHz
        a_dwell = x[2000:22000]
        c_dwell = x[slot + 2000 : 2 * slot]
        assert np.mean(np.abs(a_dwell) ** 2) > 1000 * np.mean(np.abs(c_dwell) ** 2)
        spec = np.abs(np.fft.fft(a_dwell)) ** 2
        f = np.fft.fftfreq(len(a_dwell), 1e-6)
        assert abs(f[np.argmax(spec)] - 200e3) < 100

        # Emission annotations are clipped to band-a dwells
        emis = [a for a in m["annotations"] if a.get("core:label") == "cw_test"]
        assert emis
        for a in emis:
            s0 = a["core:sample_start"] % (2 * slot)
            assert 2000 <= s0 and s0 + a["core:sample_count"] <= slot

    def test_scanning_validates_and_is_deterministic(self, tmp_path):
        sigmf = pytest.importorskip("sigmf")
        raw = scenario(window=(0.0, 0.2), tuning=SCAN)
        raw["rf"]["chunk_samples"] = 30_000
        _, r1 = run(tmp_path, raw, workers=1, name="w1")
        _, r2 = run(tmp_path, raw, workers=3, name="w3")
        assert r1.data_path.read_bytes() == r2.data_path.read_bytes()
        handle = sigmf.fromfile(str(r1.meta_path))
        handle.validate()


class TestConfig:
    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda r: r["rf"].update(propagation={"fading": "rayleigh"}), "rician"),
            (lambda r: r["rf"].update(propagation={"groundreflection": True}), "ground_reflection"),
            (
                lambda r: r["rf_sensors"][0]["receiver"].update(
                    impairments={"phase_noise": {"model": "flicker", "linewidth_hz": 1}}
                ),
                "wiener",
            ),
            (
                lambda r: r["rf_sensors"][0]["receiver"].update(
                    gain_mode="agc", agc={"block_samples": 3000}
                ),
                "divide",
            ),
            (
                lambda r: r["rf_sensors"][0]["receiver"].update(impairments={"dc_offset_dbfs": 3}),
                "dBFS",
            ),
        ],
    )
    def test_invalid(self, tmp_path, mutate, match):
        raw = scenario()
        mutate(raw)
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match=match):
            load_scenario(p)
