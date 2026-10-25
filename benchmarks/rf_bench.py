"""RF synthesis benchmarks (Phase 5).

Runs representative workloads from scenarios/rf_fpv_mix.yaml and reports throughput in
complex samples per second (all channels). Usage::

    uv run python benchmarks/rf_bench.py [--workers 1 4] [--only NAME ...] [--profile NAME]

Each workload synthesizes a short window of one recording into a temporary directory.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import pstats
import tempfile
import time
from pathlib import Path

import yaml

from modsim.config import load_scenario
from modsim.rf.iq import synthesize
from modsim.rf.runner import plan_rf

ROOT = Path(__file__).resolve().parent.parent
SCENARIO = ROOT / "scenarios" / "rf_fpv_mix.yaml"

# name: (sensor id, band name or None for all, window seconds, description)
WORKLOADS = {
    "elrs_915": ("RF-01", "ism915", 1.0, "915 MHz, 30 MS/s, 12 ELRS radios"),
    "fpv_5g8": ("RF-04", "fpv_r5", 0.5, "5.8 GHz, 40 MS/s, 6 analog FPV + DJI downlink"),
    "busy_2g4": ("RF-04", "ism2g4", 0.2, "2.4 GHz, 100 MS/s, ~560 Wi-Fi/BLE + DJI + Remote ID"),
    "df_2g4": ("RF-02", "ism2g4", 0.1, "2.4 GHz DF, 4-element UCA, 100 MS/s/ch"),
    "scan_3band": ("RF-03", None, 0.3, "3-band scanner, 100 MS/s"),
}


def scenario_for(sensor_id: str, band: str | None, tmp: Path, chunk: int | None = None) -> Path:
    raw = yaml.safe_load(SCENARIO.read_text())
    if chunk:
        raw.setdefault("rf", {})["chunk_samples"] = chunk
    sensor = copy.deepcopy(next(s for s in raw["rf_sensors"] if s["id"] == sensor_id))
    if band is not None:
        sensor["tuning"] = {
            "mode": "fixed",
            "bands": [b for b in sensor["tuning"]["bands"] if b["name"] == band],
        }
    raw["rf_sensors"] = [sensor]
    path = tmp / f"{sensor_id}_{band}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def run(
    name: str, workers: int, tmp: Path, profile: bool = False, chunk: int | None = None
) -> dict:
    sensor_id, band, window, desc = WORKLOADS[name]
    cfg = load_scenario(scenario_for(sensor_id, band, tmp, chunk))
    plan = plan_rf(cfg, window=(0.0, window))
    out = tmp / f"{name}_{workers}"
    if profile:
        prof = cProfile.Profile()
        prof.enable()
    t = time.perf_counter()
    _, timing = synthesize(plan, out, workers=workers)
    wall = time.perf_counter() - t
    if profile:
        prof.disable()
        pstats.Stats(prof).sort_stats("tottime").print_stats(18)
    return {"name": name, "desc": desc, "workers": workers, "samples": timing["samples"],
            "wall_s": wall, "msps": timing["samples"] / wall / 1e6}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--only", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--profile", default=None, help="profile one workload with 1 worker")
    ap.add_argument("--chunk", type=int, default=None, help="override rf.chunk_samples")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        if args.profile:
            run(args.profile, 1, tmp, profile=True, chunk=args.chunk)
            return
        print(
            f"{'workload':12s} {'workers':>7s} {'MS':>8s} {'wall s':>8s} {'MS/s':>8s}  description"
        )
        for name in args.only:
            for w in args.workers:
                r = run(name, w, tmp, chunk=args.chunk)
                print(
                    f"{name:12s} {w:7d} {r['samples'] / 1e6:8.1f} {r['wall_s']:8.1f} "
                    f"{r['msps']:8.2f}  {r['desc']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
