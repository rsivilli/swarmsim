# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

swarmsim (Python package and CLI: `modsim`) simulates swarms moving through a sensor field, computes
per-timestep detections, and exports the resulting comms traffic as a timeline that Locust replays
against a backend. A separate opt-in RF pass (`modsim rf`) synthesizes SigMF IQ recordings from the
same trajectories. README.md has the full CLI reference, scenario YAML schema and output formats.

## Commands

Uses [uv](https://docs.astral.sh/uv/); Python 3.11+ (CI tests 3.11 and 3.14).

```bash
uv sync                                    # dev group includes pytest, ruff, pyarrow, sigmf
uv run ruff check .
uv run ruff format --check .               # CI checks formatting; run `uv run ruff format .` to fix
MPLBACKEND=Agg uv run pytest               # Agg avoids GUI windows from viz tests
uv run pytest tests/test_rf_iq.py::test_name -q   # single test
for f in scenarios/*.yaml; do uv run modsim validate "$f"; done   # CI validates every scenario

uv run modsim run scenarios/minimal_test.yaml --no-viz            # fastest end-to-end check
uv run modsim run scenarios/mothership_salvo.yaml --no-viz --export-mode comms --helm-values
uv run modsim rf scenarios/rf_fpv_mix.yaml --dry-run              # RF truth + sizing, no IQ
uv run python benchmarks/rf_bench.py                              # RF synthesis throughput
```

Changes under `locust/` are also checked in CI with `docker build -t modsim-locust:ci locust/` and
`helm lint locust/helm-chart --set timeline.existingConfigMap=locust-timeline`.

Add dependencies with `uv add` — CI runs `uv sync --locked`, so `uv.lock` must stay in sync. Ruff
selects only `E4,E7,E9,F` with line length 100.

## Architecture

Pipeline: scenario YAML → `config.load_scenario` (→ `ScenarioConfig` dataclasses in `models.py`;
`validators.py` does structural checks with typo suggestions for `modsim validate`) → `sim.run_iter`
→ report / viz / comms export.

- **Simulation loop** (`sim.py`): `run_iter` is a generator yielding one frame per timestep. Order
  per step is **detect → record stats → yield snapshot → step motion → drop out-of-bounds**. Sensor
  positions come from `build_sensors` (manual/grid/hex/random). Swarm vs formation groups step via
  different functions in `motion.py`.
- **Units**: km, km/s, seconds throughout. `geo.py` projects km ↔ lat/lon around
  `reference_origin` only for map display and Track payloads.
- **Comms** (`comm/`): when `communication.enabled`, detection uses `detect_all_with_state` with a
  `TrackManager` (track IDs + state machine); `track_messages.py` builds Track.json payloads and
  `export.py` writes the Locust timeline (`.json`, or streaming `.jsonl` chosen by extension).
  `helm_values.py` derives Locust deployment sizing from message rate/duration.
- **Locust** (`locust/`): standalone container — `locustfile.py` does **not** import `modsim`; it
  only reads the exported timeline file (`TIMELINE_PATH`). Transport (`rest`/`native` Zenoh/`udl`)
  is chosen by `ZENOH_TRANSPORT`. Changing the export format means updating the locustfile parser too.
- **RF** (`rf/`, design in `docs/design/rf-sigmf.md`): `rf/config.py` parses `rf`, `rf_sensors` and
  `groups[].rf` (called from `config.load_scenario`). `runner.plan_rf` builds emitters and emissions
  (`scheduler.py`), link budgets (`links.py`), and writes truth tables + `manifest.json`. `iq.py`
  turns the plan into per-recording jobs rendered in worker processes by `synth.py` (waveforms from
  `rf/waveforms/`, channel effects from `channel.py`, receiver/ADC/impairments from `receiver.py`),
  then `sigmf_writer.py` writes SigMF. Link profiles are YAML files in `rf/profiles/`.
- **RF determinism invariant**: every random draw uses `scheduler.make_rng(scenario_seed, *stable_id
  parts)` (hash independent of `PYTHONHASHSEED`), never a shared sequential RNG. This is what makes
  IQ output byte-identical for any `--workers` count — preserve it when adding randomness to RF code.
  The non-RF sim instead uses a single `random.Random(cfg.seed)` threaded through spawning/detection.

## Tests and contributions

Shared fixtures and builders (`small_scenario`, `make_group`, `make_area`, `cli_runner`) are in
`tests/conftest.py`. New scenarios in `scenarios/` must pass `modsim validate`, stay fast, and note
what real-world event they're based on. User-facing changes get a line under "Unreleased" in
CHANGELOG.md.
