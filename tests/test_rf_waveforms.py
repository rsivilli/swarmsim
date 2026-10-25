"""Tests for RF Phase 2: OFDM and GFSK waveforms, native-rate resampling, profile parameters."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from modsim.config import load_scenario
from modsim.rf.iq import build_jobs, synthesize
from modsim.rf.library import load_builtin_profiles, parse_profile
from modsim.rf.propagation import C_MPS
from modsim.rf.runner import plan_rf
from modsim.rf.synth import _emission_signal
from modsim.rf.waveforms import fidelity, render
from modsim.rf.waveforms.fsk import SPS, gfsk_phase_table
from modsim.rf.waveforms.native import sample_native
from modsim.rf.waveforms.ofdm import (
    OVERSAMPLE,
    _active_bins,
    _window_len,
    _zadoff_chu,
    droneid_burst,
    lte_burst,
    wifi_burst,
)

LIB = load_builtin_profiles()


def obw(x: np.ndarray, fs: float, frac: float = 0.99) -> float:
    p = np.abs(np.fft.fftshift(np.fft.fft(x))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1 / fs))
    c = np.cumsum(p) / p.sum()
    return f[np.searchsorted(c, 1 - (1 - frac) / 2)] - f[np.searchsorted(c, (1 - frac) / 2)]


def waveform(pid: str, duration: float, fs: float, bandwidth_hz: float | None = None):
    prof = LIB[pid]
    u = np.arange(int(duration * fs)) / fs
    kind, s = render(
        prof.waveform,
        u,
        bandwidth_hz=bandwidth_hz or prof.bandwidth_hz,
        duration=duration,
        seed=5,
        params={"waveform": prof.waveform_params},
        sample_rate=fs,
        noise_key=(0,),
    )
    return np.exp(1j * s) if kind == "phase" else s


class TestProfiles:
    def test_every_emitting_profile_is_protocol_like(self):
        for pid, prof in LIB.items():
            if not prof.silent:
                assert fidelity(prof.waveform) == "protocol_like", pid

    @pytest.mark.parametrize(
        "raw, match",
        [
            ({"waveform": "ofdm", "waveform_params": {"standrd": "wifi"}}, "standard"),
            ({"waveform": "fsk"}, "bit_rate"),
            ({"waveform": "qpsk"}, "Unknown key 'qpsk'"),
            ({"waveform": "ofdm", "waveform_params": {"standard": "wifii"}}, "'wifi'"),
            ({"waveform": "ofdm", "waveform_params": {"modulation": "qam256"}}, "modulation"),
        ],
    )
    def test_waveform_params_validated(self, raw, match):
        with pytest.raises(ValueError, match=match):
            parse_profile({"id": "x", "channels_mhz": [1.0], **raw})


class TestOFDM:
    @pytest.mark.parametrize(
        "pid, duration, fs, bandwidth, expected_mhz",
        [
            ("wifi_drone", 0.0015, 100e6, None, 16.6),
            ("dji_ocusync_dl", 0.003, 40e6, 20e6, 18.0),
            ("dji_ocusync_dl", 0.003, 40e6, 10e6, 9.0),
            ("dji_ocusync_ul", 0.0006, 10e6, None, 1.08),
            ("dji_droneid", 0.00064, 30e6, None, 9.0),
        ],
    )
    def test_occupied_bandwidth_and_power(self, pid, duration, fs, bandwidth, expected_mhz):
        x = waveform(pid, duration, fs, bandwidth)
        body = x[int(0.05 * len(x)) : int(0.95 * len(x))]
        assert 10 * np.log10(np.mean(np.abs(body) ** 2)) == pytest.approx(0.0, abs=0.3)
        assert obw(x, fs) / 1e6 == pytest.approx(expected_mhz, rel=0.05)

    def test_wifi_preamble_structure(self):
        x, rate = wifi_burst(0.0005, seed=1)
        os_, w = OVERSAMPLE, _window_len(20e6)
        assert rate == 20e6 * os_
        # STF repeats every 0.8 µs (outside the windowed edges)
        p = 16 * os_
        stf = x[w : 160 * os_ - w]
        assert np.allclose(stf[p:], stf[:-p])
        # LTF: 1.6 µs guard interval then two identical 3.2 µs symbols
        g0 = 160 * os_
        ltf1 = x[g0 + 32 * os_ : g0 + 96 * os_]
        ltf2 = x[g0 + 96 * os_ : g0 + 160 * os_]
        assert np.allclose(ltf1, ltf2)
        assert np.allclose(x[g0 + w : g0 + 32 * os_], ltf1[-(32 * os_ - w) :])

    def test_wifi_cyclic_prefix(self):
        x, _ = wifi_burst(0.0005, seed=1)
        os_, w = OVERSAMPLE, _window_len(20e6)
        s0 = 320 * os_  # first data symbol (after the 16 µs preamble)
        cp, body = 16 * os_, 64 * os_
        assert np.allclose(x[s0 + w : s0 + cp], x[s0 + body + w : s0 + body + cp])

    def test_droneid_zadoff_chu_symbols(self):
        x, rate = droneid_burst(0.00064, seed=3)
        os_ = OVERSAMPLE
        assert rate == 15.36e6 * os_
        n_fft = 1024 * os_
        k = np.concatenate([np.arange(-300, 0), np.arange(1, 301)])
        cps = [80 * os_] + [72 * os_] * 8
        starts = np.concatenate([[0], np.cumsum([c + n_fft for c in cps])])
        for sym, root in ((3, 600), (5, 147)):
            body = x[starts[sym] + cps[sym] : starts[sym] + cps[sym] + n_fft]
            freq = np.fft.fft(body)[k % n_fft]
            ref = _zadoff_chu(root, 600)
            rho = np.abs(np.vdot(ref, freq)) / (np.linalg.norm(ref) * np.linalg.norm(freq))
            assert rho > 0.999
        assert len(x) == starts[-1] + _window_len(15.36e6)

    def test_lte_numerology_follows_bandwidth(self):
        for bw, n_fft in ((1.4e6, 128), (10e6, 1024), (20e6, 2048), (40e6, 4096)):
            _, rate = lte_burst(0.001, 1, bw)
            assert rate == n_fft * 15e3 * OVERSAMPLE


class TestGFSK:
    def test_ble_bits_and_deviation(self):
        table, rate = gfsk_phase_table(
            0.000376, 9, bit_rate=1e6, h=0.5, bt=0.5, preamble="aa", sync="8e89bed6"
        )
        freq = np.diff(table) * rate / (2 * np.pi)
        assert np.abs(freq).max() == pytest.approx(250e3, rel=0.02)
        # Demodulate at bit centres: preamble 0xAA then access address 0x8E89BED6, LSB first
        centres = freq[np.arange(40) * SPS + SPS // 2]
        bits = (centres > 0).astype(int)
        expected = [(0xAA >> i) & 1 for i in range(8)] + [(0x8E89BED6 >> i) & 1 for i in range(32)]
        assert bits.tolist() == expected

    def test_ble_bandwidth(self):
        x = waveform("ble_device", 0.000376, 10e6)
        assert obw(x, 10e6) / 1e6 == pytest.approx(1.05, abs=0.15)

    def test_sik_bandwidth(self):
        x = waveform("sik_915_intermittent", 0.05, 1e6)
        assert 0.1 < obw(x, 1e6) / 1e6 < 0.2  # within the profile's 200 kHz channel


class TestNativeResampling:
    @pytest.mark.parametrize("fs", [30.72e6, 40e6, 100e6, 1e6])
    def test_tone_accuracy(self, fs):
        rate, f0 = 20e6, (8.3e6 if fs > 20e6 else 0.3e6)
        native = np.exp(2j * np.pi * f0 * np.arange(100_000) / rate)
        u0, m = 1.2345e-4, 2000
        y = sample_native(native, rate, u0, m, fs)
        ref = np.exp(2j * np.pi * f0 * (u0 + np.arange(m) / fs))
        assert 10 * np.log10(np.mean(np.abs(y - ref) ** 2)) < -45

    def test_out_of_band_content_is_filtered(self):
        native = np.exp(2j * np.pi * 5e6 * np.arange(40000) / 20e6)  # 5 MHz tone
        y = sample_native(native, 20e6, 1e-4, 4000, 2e6)  # 2 MS/s grid: tone is out of band
        assert np.mean(np.abs(y) ** 2) < 1e-4


# ---------------------------------------------------------------------------
# End to end through the synthesizer
# ---------------------------------------------------------------------------


def scenario(link: str, fs: float, center: float, **extra) -> dict:
    raw = {
        "name": "phase 2",
        "area": {"width_km": 10.0, "height_km": 10.0},
        "dt": 0.5,
        "duration": 2,
        "seed": 4,
        "groups": [
            {
                "group_id": "g",
                "count": 1,
                "mode": "formation",
                "heading": [1.0, 0.0],
                "speed": 0.02,
                "spawn_center": [3.0, 5.0],
                "spawn_radius": 0.0,
                "spacing_profile": "uniform",
                "spawn_altitude": 0.1,
                "rf": {"links": [link]},
            }
        ],
        "rf": {"annotations": {"granularity": "emission"}},
        "rf_sensors": [
            {
                "id": "S",
                "position": [1.0, 5.0, 0.1],
                "receiver": {"noise_figure_db": 6, "sample_format": "cf32_le"},
                "tuning": {
                    "mode": "fixed",
                    "bands": [{"name": "b", "center_hz": center, "sample_rate": fs}],
                },
            }
        ],
    }
    raw.update(extra)
    return raw


def run(tmp_path, raw):
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(raw))
    plan = plan_rf(load_scenario(p))
    results, _ = synthesize(plan, tmp_path / "out", workers=1)
    return plan, results[0]


def read_iq(res) -> np.ndarray:
    raw = np.fromfile(res.data_path, dtype="<f4").astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


class TestEndToEnd:
    def test_wifi_energy_matches_link_budget(self, tmp_path):
        raw = scenario({"profile": "wifi_drone", "params": {"channel": "6"}}, 40e6, 2.437e9)
        raw["rf"]["window"] = [0.0, 0.2]
        plan, res = run(tmp_path, raw)
        x = read_iq(res)
        import json

        meta = json.loads(res.meta_path.read_text())["global"]
        fs_mw = 10 ** (meta["swarmsim:full_scale_dbm"] / 10)
        noise_mw = 10 ** (meta["swarmsim:noise_power_dbm"] / 10)
        signal = np.sum(np.abs(x) ** 2) * fs_mw - noise_mw * len(x)
        predicted = np.sum(
            10 ** (plan.links["rx_power_dbm"] / 10)
            * plan.emissions.duration[plan.links["emission_idx"]]
            * 40e6
        )
        assert 10 * np.log10(signal / predicted) == pytest.approx(0.0, abs=0.5)

    def test_droneid_burst_found_where_annotated(self, tmp_path):
        raw = scenario("dji_droneid", 15.36e6, 2.4445e9)
        raw["rf_sensors"][0]["tuning"]["bands"][0]["center_hz"] = 2.4445e9
        raw["rf"]["profiles"] = {
            "droneid_24": {"base": "dji_droneid", "channels_mhz": [2444.5]},
        }
        raw["groups"][0]["rf"]["links"] = ["droneid_24"]
        plan, res = run(tmp_path, raw)
        x = read_iq(res)
        import json

        anns = json.loads(res.meta_path.read_text())["annotations"]
        assert anns, "expected DroneID bursts"
        # Reference: symbol 3 (ZC root 600) of a DroneID burst, time domain at 15.36 MS/s
        n_fft, bins = 1024, _active_bins(1024, 600)
        freq = np.zeros(n_fft, dtype=complex)
        freq[bins] = _zadoff_chu(600, 600)
        ref = np.fft.ifft(freq)
        offset = 80 + n_fft + 2 * (72 + n_fft) + 72  # start of symbol 3's body
        for ann in anns[:3]:
            s0 = ann["core:sample_start"]
            seg = x[s0 : s0 + 9 * (n_fft + 80)]
            corr = np.abs(np.correlate(seg, ref, mode="valid"))
            assert abs(int(np.argmax(corr)) - offset) <= 1

    def test_ofdm_chunks_join_seamlessly(self, tmp_path):
        raw = scenario({"profile": "wifi_drone", "params": {"channel": "6"}}, 40e6, 2.437e9)
        raw["rf"]["window"] = [0.0, 0.05]
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        plan = plan_rf(load_scenario(p))
        (job,), _ = build_jobs(plan, tmp_path / "o")
        job_rows = np.argsort(job.duration)[::-1]
        row = int(job_rows[0])  # longest burst
        start, end = job.sample_span()
        a, b = int(start[row]), int(end[row])
        mid = (a + b) // 2
        whole = _emission_signal(job, row, np.arange(a, b), job.sample_rate, 0)
        left = _emission_signal(job, row, np.arange(a, mid), job.sample_rate, 0)
        right = _emission_signal(job, row, np.arange(mid, b), job.sample_rate, 1)
        joined = np.concatenate([left, right])
        err = np.mean(np.abs(joined - whole) ** 2) / np.mean(np.abs(whole) ** 2)
        # Residual: native bursts are sampled on a uniform grid, ignoring the v/c time
        # compression across one burst (~5e-11 s here)
        assert 10 * np.log10(err) < -50

    def test_gfsk_propagation_delay(self, tmp_path):
        prof = copy.deepcopy(LIB["remote_id_ble"])
        assert prof.waveform == "fsk"
        # 20 dBm so a single burst stands well clear of the noise in a 4 MS/s recording
        raw = scenario({"profile": "remote_id_ble", "params": {"power_dbm": 20}}, 4e6, 2.426e9)
        plan, res = run(tmp_path, raw)
        import json

        anns = json.loads(res.meta_path.read_text())["annotations"]
        e = anns[0]["swarmsim:emissions"][0]
        tau = plan.links["range_m"][plan.links["emission_idx"] == e][0] / C_MPS
        assert anns[0]["core:sample_start"] == round((plan.emissions.t_start[e] + tau) * 4e6)
        x = read_iq(res)
        s0 = anns[0]["core:sample_start"]
        n = anns[0]["core:sample_count"]
        assert np.mean(np.abs(x[s0 : s0 + n]) ** 2) > 10 * np.mean(
            np.abs(x[s0 - 2000 : s0 - 100]) ** 2
        )


class TestBandEdgeNative:
    def test_single_rate_edge_path_matches_oversampled_path(self, tmp_path, monkeypatch):
        """OFDM cut by the band edge: the single-rate bandpass path (Phase 5) must agree with
        the oversample-and-decimate path in the band interior and leak no more beyond the cut."""
        import modsim.rf.synth as synth

        raw = scenario({"profile": "wifi_drone", "params": {"channel": "6", "power_dbm": 30}},
                       20e6, 2.447e9)  # fmt: skip  # 2437 MHz: half of the burst is out of band
        raw["rf"]["window"] = [0.0, 0.05]
        raw["rf_sensors"][0]["receiver"]["noise_figure_db"] = 0
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        plan = plan_rf(load_scenario(p))
        (new,), _ = synthesize(plan, tmp_path / "new", workers=1)
        monkeypatch.setattr(synth, "NATIVE_RATE", set())
        (old,), _ = synthesize(plan, tmp_path / "old", workers=1)
        a, b = read_iq(new), read_iq(old)
        f = np.fft.fftfreq(len(a), 1 / 20e6)
        spec_a, spec_b = np.abs(np.fft.fft(a)) ** 2, np.abs(np.fft.fft(b)) ** 2
        diff = np.abs(np.fft.fft(a - b)) ** 2
        interior = (f > -8e6) & (f < -1e6)
        assert 10 * np.log10(diff[interior].sum() / spec_b[interior].sum()) < -35
        beyond = (f > 1e6) & (f < 9e6)  # where content past the cut would alias
        assert spec_a[beyond].mean() <= 1.26 * spec_b[beyond].mean()


class TestPhase5Equivalence:
    @pytest.mark.parametrize("fs", [100e6, 30.72e6])
    def test_native_resampling_against_exact_interpolation(self, fs):
        """Guard on OFDM placement accuracy: compare with exact trigonometric interpolation of
        the band-limited burst (zero-padded, so periodic extension is harmless)."""
        native, rate = wifi_burst(0.0002, seed=4)
        pad = np.zeros(3 * len(native), dtype=complex)
        pad[len(native) : 2 * len(native)] = native
        spec, m = np.fft.fft(pad), len(pad)
        k = np.fft.fftfreq(m, 1 / m)
        n = int(1.5e-4 * fs)
        t = 2.5e-7 + np.arange(n) / fs
        exact = np.exp(2j * np.pi * np.outer(t * rate + len(native), k) / m) @ spec / m
        f = np.fft.fftfreq(n, 1 / fs)
        band = np.abs(f) < 8.5e6  # the Wi-Fi channel
        for got in (
            sample_native(native, rate, 2.5e-7, n, fs),
            sample_native(native, rate, 2.5e-7, n, fs, half_bw=10e6),  # unclipped: same path
        ):
            err = (np.abs(np.fft.fft(got - exact))[band] ** 2).sum()
            assert 10 * np.log10(err / (np.abs(np.fft.fft(exact))[band] ** 2).sum()) < -45

    def test_threaded_link_table_is_deterministic(self, tmp_path, monkeypatch):
        import modsim.rf.links as links

        raw = scenario({"profile": "wifi_drone", "params": {"channel": "6"}}, 40e6, 2.437e9)
        raw["rf_sensors"] += [
            dict(copy.deepcopy(raw["rf_sensors"][0]), id=f"S{i}", position=[1.0 + i, 4.0, 0.01])
            for i in range(5)
        ]
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        cfg = load_scenario(p)
        many = plan_rf(cfg).links
        monkeypatch.setattr(links.os, "cpu_count", lambda: 1)
        one = plan_rf(cfg).links
        for col in many:
            np.testing.assert_array_equal(many[col], one[col])
