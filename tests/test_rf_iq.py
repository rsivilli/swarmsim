"""Tests for RF Phase 1: IQ synthesis, receiver model and SigMF output."""

from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest
import yaml

from modsim.cli import cli
from modsim.config import load_scenario
from modsim.rf.iq import build_jobs, synthesize
from modsim.rf.models import ReceiverConfig
from modsim.rf.propagation import C_MPS
from modsim.rf.receiver import build_receiver
from modsim.rf.runner import plan_rf
from modsim.rf.synth import _emission_signal
from modsim.rf.waveforms import bandlimited_noise, render
from modsim.rf.waveforms.lora import lora_css

CW_PROFILE = {
    "description": "test carrier",
    "waveform": "cw",
    "bandwidth_hz": 10000,
    "power_dbm": 0,
    "antenna_gain_dbi": 0,
    "roles": ["drone"],
    "channels_mhz": [915.2],
    "activity": [{"model": "continuous", "params": {"channel": 915.2}}],
}


def base(link="cw_test", speed=0.0, heading=(1.0, 0.0), spawn=(3.0, 5.0), **sensor):
    sensor_cfg = {
        "id": "S",
        "position": [1.0, 5.0, 0.1],
        "type": "detect",
        "receiver": {"noise_figure_db": 6, "sample_format": "cf32_le"},
        "tuning": {
            "mode": "fixed",
            "bands": [{"name": "b", "center_hz": 915.0e6, "sample_rate": 1.0e6}],
        },
    }
    sensor_cfg.update(sensor)
    return {
        "name": "iq test",
        "area": {"width_km": 10.0, "height_km": 10.0},
        "dt": 0.5,
        "duration": 2,
        "seed": 11,
        "reference_origin": [38.0, -104.0],
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
                "spawn_altitude": 0.1,
                "rf": {"links": [link] if isinstance(link, (str, dict)) else link},
            }
        ],
        "rf": {"profiles": {"cw_test": copy.deepcopy(CW_PROFILE)}},
        "rf_sensors": [sensor_cfg],
    }


def run(tmp_path, raw, workers=1, name="out"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(raw))
    plan = plan_rf(load_scenario(p))
    out = tmp_path / name
    results, timing = synthesize(plan, out, workers=workers)
    return plan, results, out


def read_iq(result, fmt="cf32_le"):
    dtype = {"cf32_le": "<f4", "ci16_le": "<i2"}[fmt]
    raw = np.fromfile(result.data_path, dtype=dtype).astype(np.float64)
    return raw[0::2] + 1j * raw[1::2]


def meta(result):
    return json.loads(result.meta_path.read_text())


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------


class TestWaveforms:
    def test_lora_frequency_sweeps_the_bandwidth(self):
        fs, bw = 4e6, 500e3
        u = np.arange(int(0.004 * fs)) / fs
        phase = lora_css(u, sf=6, bw_hz=bw, preamble=6, seed=1)
        inst = np.diff(np.unwrap(phase)) * fs / (2 * np.pi)
        assert inst.max() <= bw / 2 + 1e3 and inst.min() >= -bw / 2 - 1e3
        # Preamble up-chirps: frequency rises linearly at BW/T within a symbol
        t_sym = 64 / bw
        n = int(t_sym * fs)
        slope = np.polyfit(u[1 : n // 2], inst[: n // 2 - 1], 1)[0]
        assert slope == pytest.approx(bw / t_sym, rel=0.01)

    def test_lora_phase_is_continuous(self):
        fs = 8e6
        u = np.arange(int(0.004 * fs)) / fs
        phase = lora_css(u, sf=6, bw_hz=500e3, preamble=6, seed=2)
        jumps = np.abs(np.angle(np.exp(1j * np.diff(phase))))
        assert jumps.max() < 2 * np.pi * 250e3 / fs * 1.01

    def test_fm_occupies_carson_bandwidth(self):
        fs = 60e6
        u = np.arange(int(0.005 * fs)) / fs
        kind, ph = render(
            "fm",
            u,
            bandwidth_hz=18e6,
            duration=1.0,
            seed=3,
            params={},
            sample_rate=fs,
            noise_key=(0,),
        )
        assert kind == "phase"
        spec = np.abs(np.fft.fft(np.exp(1j * ph))) ** 2
        f = np.fft.fftfreq(len(u), 1 / fs)
        assert spec[np.abs(f) <= 9e6].sum() / spec.sum() > 0.99

    def test_placeholder_noise_is_bandlimited_unit_power(self):
        x = bandlimited_noise(200_000, 2e6, 10e6, 5, (1,))
        assert np.mean(np.abs(x) ** 2) == pytest.approx(1.0, rel=0.03)
        spec = np.abs(np.fft.fft(x)) ** 2
        f = np.fft.fftfreq(len(x), 1 / 10e6)
        assert spec[np.abs(f) > 1.01e6].sum() == pytest.approx(0.0, abs=1e-6 * spec.sum())


# ---------------------------------------------------------------------------
# Receiver
# ---------------------------------------------------------------------------


class TestReceiver:
    def test_auto_full_scale_puts_noise_headroom_below_full_scale(self):
        rx = build_receiver(ReceiverConfig(sample_format="ci16_le", noise_figure_db=6), 1e6)
        assert rx.full_scale_dbm == pytest.approx(rx.noise_power_dbm + 50.0)
        buf = np.zeros(200_000, dtype=np.complex64)
        rx.add_noise(buf, np.random.default_rng(0))
        iq, clipped = rx.digitize(buf)
        assert iq.dtype == np.dtype("<i2") and clipped == 0
        rms = np.sqrt(np.mean(iq.astype(float) ** 2) * 2)  # complex RMS in LSB
        assert rms == pytest.approx(32767 * 10 ** (-50 / 20), rel=0.03)

    def test_clipping_is_counted(self):
        rx = build_receiver(ReceiverConfig(sample_format="ci16_le", adc_bits=12), 1e6)
        buf = np.full(1000, 10 * np.sqrt(rx.full_scale_mw), dtype=np.complex64)
        iq, clipped = rx.digitize(buf)
        assert clipped == 1000  # every I component clips (Q is zero)
        assert iq.max() == 2047

    @pytest.mark.parametrize(
        "fmt, dtype", [("cf32_le", "<f4"), ("ci8", "i1"), ("cu8", "u1"), ("ci16_be", ">i2")]
    )
    def test_formats(self, fmt, dtype):
        rx = build_receiver(ReceiverConfig(sample_format=fmt), 1e6)
        iq, _ = rx.digitize(np.zeros(4, dtype=np.complex64))
        assert iq.dtype == np.dtype(dtype) and len(iq) == 8
        if fmt == "cu8":
            assert set(iq.tolist()) == {128}


# ---------------------------------------------------------------------------
# Synthesis accuracy (Phase 1 exit criteria)
# ---------------------------------------------------------------------------


class TestSynthesis:
    def test_cw_power_matches_link_budget(self, tmp_path):
        plan, (res,), _ = run(tmp_path, base())
        x = read_iq(res)
        fs, off = 1e6, 200e3
        spec = np.abs(np.fft.fft(x)) ** 2 / len(x) ** 2
        f = np.fft.fftfreq(len(x), 1 / fs)
        tone = np.abs(f - off) < 20
        fs_mw = 10 ** (meta(res)["global"]["swarmsim:full_scale_dbm"] / 10)
        measured_dbm = 10 * np.log10(spec[tone].sum() * fs_mw)
        predicted = plan.links["rx_power_dbm"][0]
        assert measured_dbm == pytest.approx(predicted, abs=0.5)
        # Noise floor matches kTB·NF over the recorded bandwidth
        noise_dbm = 10 * np.log10(spec[~tone].sum() * len(x) / (~tone).sum() * fs_mw)
        assert noise_dbm == pytest.approx(-174 + 60 + 6, abs=0.3)

    def test_lora_snr_matches_link_budget(self, tmp_path):
        raw = base(link={"profile": "elrs_900", "params": {"slot_owner": "all"}})
        raw["rf_sensors"][0]["tuning"]["bands"][0]["sample_rate"] = 4.0e6
        plan, (res,), _ = run(tmp_path, raw)
        x = read_iq(res)
        m = meta(res)
        fs_mw = 10 ** (m["global"]["swarmsim:full_scale_dbm"] / 10)
        noise_mw = 10 ** (m["global"]["swarmsim:noise_power_dbm"] / 10)
        signal_energy = np.sum(np.abs(x) ** 2) * fs_mw - noise_mw * len(x)
        predicted = np.sum(
            10 ** (plan.links["rx_power_dbm"] / 10)
            * plan.emissions.duration[plan.links["emission_idx"]]
            * 4e6
        )
        assert 10 * np.log10(signal_energy / predicted) == pytest.approx(0.0, abs=0.5)

    def test_doppler_tone_within_1hz(self, tmp_path):
        # Drone flies straight away from the sensor at 30 m/s (same altitude)
        plan, (res,), _ = run(tmp_path, base(speed=0.03))
        x = read_iq(res)
        fs, off = 1e6, 200e3
        n = len(x) * 8
        spec = np.abs(np.fft.fft(x, n)) ** 2
        f = np.fft.fftfreq(n, 1 / fs)
        band = np.abs(f - off) < 500
        peak = f[band][np.argmax(spec[band])]
        expected = -915.2e6 * 30.0 / C_MPS
        assert expected == pytest.approx(-91.58, abs=0.01)
        assert peak - off == pytest.approx(expected, abs=1.0)
        assert plan.links["doppler_hz"][0] == pytest.approx(expected, abs=1e-3)

    def test_propagation_delay_shifts_burst_start(self, tmp_path):
        raw = base(link={"profile": "elrs_900", "params": {"slot_owner": "all"}})
        raw["rf_sensors"][0]["tuning"]["bands"][0]["sample_rate"] = 4.0e6
        raw["rf"]["annotations"] = {"granularity": "emission"}
        plan, (res,), _ = run(tmp_path, raw)
        ann = meta(res)["annotations"][0]
        e = int(ann["swarmsim:emissions"][0])
        tau = plan.links["range_m"][0] / C_MPS
        assert ann["core:sample_start"] == round((plan.emissions.t_start[e] + tau) * 4e6)

    def test_chunks_join_seamlessly(self, tmp_path):
        raw = base(speed=0.03)
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        plan = plan_rf(load_scenario(p))
        (job,), _ = build_jobs(plan, tmp_path / "o")
        whole = _emission_signal(job, 0, np.arange(1000, 3000), job.sample_rate, 0)
        a = _emission_signal(job, 0, np.arange(1000, 2000), job.sample_rate, 0)
        b = _emission_signal(job, 0, np.arange(2000, 3000), job.sample_rate, 1)
        np.testing.assert_allclose(np.concatenate([a, b]), whole, rtol=1e-4, atol=1e-12)

    def test_band_edge_emission_is_filtered_not_aliased(self, tmp_path):
        prof = copy.deepcopy(CW_PROFILE)
        prof.update(waveform="noise", bandwidth_hz=200000, channels_mhz=[915.45], power_dbm=30)
        prof["activity"] = [{"model": "continuous", "params": {"channel": 915.45}}]
        raw = base()
        raw["rf"]["profiles"]["edge"] = prof
        raw["groups"][0]["rf"]["links"] = ["edge"]
        plan, (res,), _ = run(tmp_path, raw)
        x = read_iq(res)
        spec = np.abs(np.fft.fft(x)) ** 2
        f = np.fft.fftfreq(len(x), 1 / 1e6)
        upper = spec[(f > 360e3) & (f < 440e3)].mean()
        mirror = spec[(f < -360e3) & (f > -440e3)].mean()  # where aliases would land
        empty = spec[(f < -100e3) & (f > -200e3)].mean()  # thermal noise only
        assert 10 * np.log10(upper / mirror) > 40
        assert 10 * np.log10(mirror / empty) == pytest.approx(0.0, abs=0.5)

    def test_byte_identical_across_worker_counts(self, tmp_path):
        raw = base(
            link=[{"profile": "elrs_900", "params": {"slot_owner": "all"}}, "cw_test"], speed=0.02
        )
        raw["rf"]["chunk_samples"] = 65536
        _, (r1,), _ = run(tmp_path, raw, workers=1, name="w1")
        _, (r2,), _ = run(tmp_path, raw, workers=3, name="w3")
        assert r1.data_path.read_bytes() == r2.data_path.read_bytes()
        assert meta(r1) == meta(r2)

    def test_partial_window(self, tmp_path):
        raw = base()
        raw["rf"]["window"] = [0.5, 1.5]
        plan, (res,), _ = run(tmp_path, raw)
        assert res.n_samples == 1_000_000
        assert meta(res)["captures"][0]["core:datetime"] == "2026-01-01T00:00:00.500000Z"


# ---------------------------------------------------------------------------
# SigMF output
# ---------------------------------------------------------------------------


class TestSigMF:
    def test_metadata_validates_and_reads(self, tmp_path):
        sigmf = pytest.importorskip("sigmf")
        raw = base(link=[{"profile": "elrs_900", "params": {"slot_owner": "all"}}, "cw_test"])
        _, (res,), out = run(tmp_path, raw)
        handle = sigmf.fromfile(str(res.meta_path))
        handle.validate()
        assert handle.read_samples(0, 10).shape == (10,)
        assert handle.get_global_field("core:sample_rate") == 1e6

    def test_annotations(self, tmp_path):
        raw = base(link={"profile": "elrs_900", "params": {"slot_owner": "all"}})
        raw["rf_sensors"][0]["tuning"]["bands"][0]["sample_rate"] = 4.0e6
        raw["rf"]["annotations"] = {"granularity": "emission"}
        plan, (res,), _ = run(tmp_path, raw)
        m = meta(res)
        anns = m["annotations"]
        assert len(anns) == len(plan.links["snr_db"])
        starts = [a["core:sample_start"] for a in anns]
        assert starts == sorted(starts)
        a = anns[0]
        assert a["core:label"] == "elrs_900"
        assert a["signal:detail"]["standard"] == "ExpressLRS"
        assert a["signal:detail"]["channel_bw"] == 500000.0
        assert a["signal:emitter"]["label"] == "g-0000/elrs_900"
        assert a["signal:emitter"]["geolocation"]["type"] == "Point"
        assert a["swarmsim:waveform_fidelity"] == "protocol_like"
        assert {e["name"] for e in m["global"]["core:extensions"]} == {"signal", "swarmsim"}
        # Energy sits inside the annotated spans
        x = read_iq(res)
        inside = np.zeros(len(x), dtype=bool)
        for ann in anns:
            inside[
                ann["core:sample_start"] : ann["core:sample_start"] + ann["core:sample_count"]
            ] = True
        assert np.mean(np.abs(x[inside]) ** 2) > 100 * np.mean(np.abs(x[~inside]) ** 2)

    def test_dwell_and_session_granularity(self, tmp_path):
        counts = {}
        for g in ("emission", "dwell", "session"):
            raw = base(link={"profile": "elrs_900", "params": {"slot_owner": "all"}})
            raw["rf"]["annotations"] = {"granularity": g}
            _, (res,), _ = run(tmp_path, raw, name=g)
            counts[g] = len(meta(res)["annotations"])
        assert counts["emission"] > counts["dwell"] > counts["session"] == 1
        # 200 Hz packets hopping every 4 packets -> about one dwell per hop
        assert counts["dwell"] == pytest.approx(counts["emission"] / 4, rel=0.05)

    def test_collection_hashes_meta(self, tmp_path):
        raw = base()
        raw["rf_sensors"][0]["tuning"] = {
            "mode": "multi",
            "bands": [
                {"name": "a", "center_hz": 915.0e6, "sample_rate": 1.0e6},
                {"name": "b", "center_hz": 2.44e9, "sample_rate": 1.0e6},
            ],
        }
        _, results, out = run(tmp_path, raw)
        assert len(results) == 2
        coll = json.loads((out / "S" / "S.sigmf-collection").read_text())["collection"]
        streams = {s["name"]: s["hash"] for s in coll["core:streams"]}
        for r in results:
            assert streams[r.name] == hashlib.sha512(r.meta_path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_synthesizes_and_records_manifest(self, tmp_path, cli_runner):
        raw = base()
        raw["rf_sensors"].append(
            {
                "id": "DF",
                "position": [2, 2, 0.01],
                "type": "df",
                "array": {"geometry": "uca", "elements": 4, "radius_m": 0.08},
                "tuning": {
                    "mode": "fixed",
                    "bands": [{"name": "b", "center_hz": 9.15e8, "sample_rate": 1e6}],
                },
            }
        )
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        out = tmp_path / "o"
        result = cli_runner.invoke(
            cli, ["rf", str(p), "--out", str(out), "--workers", "1", "--truth-format", "csv"]
        )
        assert result.exit_code == 0, result.output
        assert "not synthesized: antenna arrays" in result.output
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["mode"] == "iq"
        assert manifest["synthesis"]["skipped_sensors"] == {
            "DF": "antenna arrays are synthesized from Phase 3"
        }
        assert (out / "S" / "S_b.sigmf-data").stat().st_size == 2_000_000 * 8
        assert not (out / "DF").exists()

    def test_errors_when_nothing_can_be_synthesized(self, tmp_path, cli_runner):
        raw = base()
        raw["rf_sensors"][0]["array"] = {"geometry": "uca", "elements": 2, "radius_m": 0.05}
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump(raw))
        result = cli_runner.invoke(cli, ["rf", str(p), "--out", str(tmp_path / "o")])
        assert result.exit_code == 1
        assert "No RF sensor can be synthesized" in result.output
