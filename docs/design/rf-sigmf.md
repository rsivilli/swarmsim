# Design: RF emitter simulation and SigMF IQ generation

Status: **Phase 0 implemented**; phases 1–5 not started. The open questions from the first draft
have been resolved (§14).

## 1. Summary

Add an RF layer to swarmsim. Drones, their ground controllers and background devices are modelled as RF
emitters with realistic commercial link profiles. RF sensors placed in the scenario record **continuous,
synthesized baseband IQ**, which is written to disk as **SigMF** recordings with full ground-truth
annotations.

The RF layer is a **separate, opt-in pass** (`modsim rf`). It runs on the trajectories produced by the
existing motion model. `modsim run`, the comms timeline and the Locust pipeline are unchanged.

### Decisions from the requirements interview

| Topic | Decision |
|---|---|
| SigMF content | Synthesized IQ (`.sigmf-data`) plus metadata (`.sigmf-meta`) |
| Delivery | Files on disk (no Locust/timeline integration) |
| Derived tracks | None. RF sensors produce SigMF only; geolocation and tracking happen downstream |
| Sensor types | Detect/classify, plus direction finding (DF) |
| Waveform fidelity | Protocol-like: correct modulation family, bandwidth, burst timing and hopping, but not bit-exact framing |
| Capture mode | Continuous recording for the whole scenario window |
| DF representation | Multi-channel array IQ **and** bearing truth in annotations |
| Link profiles (first cut) | DJI OcuSync/O3/O4, ExpressLRS/Crossfire, analog FPV video, Wi-Fi and Remote ID |
| Scale | Large; accept the storage and compute cost |
| Tuning | Fixed single band, multiple fixed bands, **and** scanning |
| Sample format | Configurable per sensor (`ci16_le`, `cf32_le`, ...) |
| Channel realism | Path loss + noise, Doppler, multipath/fading, receiver impairments |
| Compute | Local multiprocess (single large machine) |
| Ground truth | Emission truth, geometry truth, detectability flags |
| Extra emitters | Ground controllers, silent/intermittent drones, background RF |
| Annotation default | `dwell` granularity; per-burst truth in Parquet |
| Timestamps | Fixed, configurable scenario epoch (`rf.epoch`) |
| Profile calibration | Public sources only (FCC filings, datasheets, open-source firmware, published captures) |
| Array file layout | One interleaved multichannel dataset per recording (`core:num_channels`) |
| Scanning retune gaps | Zero-filled samples |
| Dependencies | `sigmf` and `pyarrow` in an optional `rf` extra |

### Non-goals (for now)

- Bit-exact, decodable protocol framing. This could be added later for Remote ID and Wi-Fi, which have
  public specifications.
- Derived detections, bearings-only reports or tracks in the comms timeline or Locust.
- GPU acceleration or Kubernetes batch execution. The chunked design leaves room for both later.
- Terrain/DEM line-of-sight masking (§11 lists it as a later extension).
- Jamming or electronic attack.

## 2. Fit with the current codebase

| Existing piece | How the RF layer uses it |
|---|---|
| `sim.run()` → `SimState.object_snapshots` | Source of truth for drone trajectories. Motion is constant-velocity within each `dt`, so linear interpolation between snapshots is exact. |
| `ScenarioConfig` / `config.py` / `validators.py` | Extended with new optional top-level keys (`rf`, `rf_sensors`, per-group `rf`). Existing scenarios are unaffected. |
| `SensorConfig` (one homogeneous sensor field) | **Left as is.** RF sensors are a separate list (`rf_sensors`), so the radar/EO detection model and the comms export don't change. |
| `geo.km_to_latlon` | Gives `core:geolocation` and the truth positions in lat/lon. |
| `cli.py` | Gets a new `modsim rf` command. `modsim validate` learns to validate and size the RF sections. |

A separate top-level `rf_sensors` list avoids a breaking refactor of `SensorConfig`. The two sensor
models can be unified later if needed.

## 3. Architecture

The pass runs in two stages: a cheap, global **emission schedule**, then a heavy synthesis stage that
runs in parallel per **(sensor, band, time-chunk)**.

```
scenario.yaml
    │
    ▼
sim.run()  ──►  trajectories (object_snapshots, interpolated)
    │
    ▼
Stage A: emission scheduling (single process, deterministic)
    • expand emitters: drones, controllers, background devices
    • per emitter: activity model → list of Emissions
      (t0, dur, f_center, bw, eirp, waveform, waveform params, seed)
    • write truth/emissions.parquet (full burst-level truth)
    │
    ▼
Stage B: synthesis (multiprocessing pool)
    work unit = (sensor, band/recording, chunk[k])
    for each unit:
      1. select emissions overlapping the chunk in time and frequency
      2. cull emissions whose predicted SNR is below the culling floor
      3. per emission: geometry over the chunk → delay, path loss, Doppler, AoA
      4. generate baseband waveform segment (seeded per emission, so it is chunk-independent)
      5. apply channel: two-ray / Rician fading, Doppler, fractional delay
      6. resample + frequency-shift into the sensor's band
      7. apply array steering vector (per element)
      8. accumulate; add thermal noise + background
      9. receiver impairments → quantize → write to memmap at the chunk offset
    │
    ▼
Stage C: metadata
    • .sigmf-meta per recording: global, captures, annotations
    • .sigmf-collection per sensor (array elements / bands)
    • run manifest (config hash, seeds, sizing, timing)
```

### Determinism

- Every emission gets its own seed, derived with `numpy.random.SeedSequence` from
  `(scenario seed, emitter id, emission index)`.
- Noise is seeded per `(sensor, channel, chunk index)`.
- Output is therefore **bit-identical regardless of worker count or chunk scheduling**. Tests enforce
  this.

### Chunking

- Chunks are a fixed number of samples (default 2^22). Each worker opens the output file as a `numpy.memmap`
  and writes only its own slice, so no merge step is needed.
- An emission that straddles a chunk boundary is rendered by both chunks. Each chunk regenerates only
  its overlapping segment from the emission's seed, so the two halves join seamlessly.
- Filters (resampling, the fading-channel FIR) use the overlap-save method with a `filter_len` guard
  region on each side.

## 4. Emitters

### 4.1 Emitter entities

| Entity | Position | Created from |
|---|---|---|
| Drone transmitter(s) | Interpolated trajectory | Each object in a group whose `rf.profile` is not `silent` |
| Ground controller | Static (or slow) ground point | `groups[].rf.controller`: one per drone, one per group, or N per group |
| Background device | Static, random within an area/density | `rf.background[]` |

A drone can carry **several links** (e.g. DJI video downlink plus Remote ID). Each link is one emitter
with its own profile. A controller emits the uplink half of a link profile (e.g. DJI control uplink, or
the ELRS TX module).

### 4.2 Activity models

Each link profile defines an **activity model** that produces a list of emissions:

| Model | Used for | Parameters |
|---|---|---|
| `continuous` | Analog FPV video | Channel; on for the whole time the drone is active |
| `periodic_hopping` | ELRS / Crossfire / DJI uplink | Packet rate, burst duration, hop sequence (seeded), channel list, telemetry ratio |
| `framed_burst` | DJI OcuSync downlink | Frame period, duty cycle, bandwidth mode, channel-selection behaviour |
| `beacon` | Remote ID | Interval (≥ 1 Hz), advertising channels, jitter |
| `csma_traffic` | Wi-Fi drones, background Wi-Fi | Offered load, burst size distribution, inter-frame gaps |
| `silent` / `intermittent` | Autonomous drones | On/off schedule (e.g. a burst of telemetry every N s, or nothing at all) |

### 4.3 Profile library (first cut)

Profiles live in `modsim/rf/profiles/*.yaml` and can be overridden per scenario. Parameters come from
**public sources only**: FCC ID test reports, manufacturer datasheets and open-source firmware
(ExpressLRS, SiK). No private reference captures are available, so the values below are **starting
points to be calibrated** during implementation against those sources and against published research
captures. Each profile file records its sources.

| Profile id | Band(s) | Waveform family | Bandwidth | Timing | Typical EIRP |
|---|---|---|---|---|---|
| `dji_ocusync_dl` | 2.4 / 5.8 GHz | OFDM bursts | 10 / 20 / 40 MHz modes | Framed, high duty | ≈ 20–30 dBm |
| `dji_ocusync_ul` | 2.4 / 5.8 GHz | Narrowband hopping bursts | ≈ 1–2 MHz | Periodic hopping | ≈ 20–30 dBm |
| `elrs_900` | 902–928 / 868 MHz | LoRa CSS | 500 kHz (SF6–SF9) | 25–200 Hz packets, FHSS | 10 mW – 1 W |
| `elrs_2g4` | 2.4 GHz | LoRa CSS / FLRC | 812 kHz / ~1.3 MHz | 50–1000 Hz packets, FHSS | 10 mW – 1 W |
| `crossfire` | 868 / 915 MHz | LoRa / FSK | ≈ 0.5 MHz | 50 / 150 Hz, FHSS | up to 2 W |
| `analog_fpv` | 5.65–5.95 GHz | FM video | ≈ 15–20 MHz occupied | Continuous | 25 mW – 1 W |
| `wifi_drone` | 2.4 / 5 GHz | 802.11 OFDM (20 MHz) | 20 MHz | CSMA traffic | ≈ 20 dBm |
| `remote_id_ble` | 2.4 GHz (adv. ch 37/38/39) | GFSK 1 Mbps / coded PHY | 1–2 MHz | Beacon ≥ 1 Hz | ≈ 0–10 dBm |
| `remote_id_wifi` | 2.4 / 5 GHz | 802.11 beacon/NAN | 20 MHz | Beacon ≥ 1 Hz | ≈ 15–20 dBm |
| `silent` | — | — | — | — | — |

"Protocol-like" means the waveform has the correct **modulation family, occupied bandwidth, spectral
shape, burst structure and hop behaviour**. That makes it a realistic input for energy detectors,
spectrogram classifiers and DF. The payload bits are random and the framing is approximate.

### 4.4 Waveform generators (`modsim/rf/waveforms/`)

Each generator takes `(params, n_samples, fs_native, rng)` and returns complex baseband at its native rate:

- `ofdm.py`: configurable FFT size, subcarrier spacing, cyclic prefix, pilots, QPSK/16QAM/64QAM
  symbols, guard bands. Used by DJI and Wi-Fi.
- `lora.py`: chirp spread spectrum (up-chirp preamble, sync word, random symbols) at a given SF/BW.
- `fsk.py`: GFSK (BLE, FLRC-like, FSK Crossfire modes).
- `fm.py`: FM-modulated synthetic video baseband (line/frame sync structure plus random luminance),
  giving the characteristic analog FPV spectrum.

Generators are cached per `(profile, params)` as template bursts with randomized payload, so a high-rate
hopper doesn't regenerate identical preambles thousands of times.

## 5. RF sensors

### 5.1 Configuration model

```yaml
rf_sensors:
  - id: RF-NORTH-01
    position: [5.0, 8.0, 0.012]        # km, same frame as other sensors
    type: df                            # "detect" | "df"
    array:                              # required for df; omitted => single element
      geometry: uca                     # "ula" | "uca" | "custom"
      elements: 4
      radius_m: 0.08
      boresight_deg: 0
      element_gain_dbi: 3
    receiver:
      noise_figure_db: 7
      sample_format: ci16_le            # any SigMF real/complex int/float type
      adc_bits: 14                      # quantization model (ci16 container)
      full_scale_dbm: -10               # ADC clipping point
      gain_mode: fixed                  # "fixed" | "agc"
      impairments:
        lo_offset_ppm: 0.5
        phase_noise: {model: wiener, linewidth_hz: 50}
        iq_imbalance: {gain_db: 0.2, phase_deg: 1.0}
        dc_offset_dbfs: -50
        channel_mismatch: {gain_db_sigma: 0.3, phase_deg_sigma: 5}   # array calibration error
    tuning:
      mode: scanning                    # "fixed" | "multi" | "scanning"
      bands:
        - {name: ism915, center_hz: 915e6,  sample_rate: 30e6}
        - {name: ism2g4, center_hz: 2.442e9, sample_rate: 100e6}
        - {name: ism5g8, center_hz: 5.8e9,  sample_rate: 160e6}
      dwell_ms: 50                      # scanning only
      retune_ms: 1                      # scanning only; dead time between dwells
      schedule: round_robin             # scanning only; or explicit list
    detector:                           # used only for the detectability truth flag
      threshold_snr_db: 6
      integration: per_emission         # SNR measured over the emission's bandwidth/duration
```

### 5.2 Tuning modes and their SigMF layout

| Mode | Behaviour | SigMF output per sensor |
|---|---|---|
| `fixed` | One band for the whole run | One recording (one dataset) |
| `multi` | Several bands recorded **simultaneously** (parallel receivers) | One recording per band |
| `scanning` | One receiver hops through bands on a dwell schedule | One recording; a new `captures` segment at every retune (`core:sample_start`, `core:frequency`, `core:datetime`). Retune dead time is written as **zero-filled samples**, so the sample index maps linearly to time. Each gap is marked with a `swarmsim:retune` annotation. |

In `scanning` mode, all bands must share a sample rate, because SigMF has one `core:sample_rate` per
recording. The validator enforces this. Modes that need different rates should use `multi` instead.

### 5.3 Arrays and DF

- Each array element is a separate receive channel. Narrowband steering is applied per emission and
  chunk from the emitter's azimuth/elevation relative to the array (`a(θ, φ) · s(t)`).
- **Wideband caveat:** for apertures much smaller than `c / bandwidth` (10 cm vs 7.5 m at 40 MHz), the
  narrowband model is accurate. Custom large apertures fall back to per-element fractional delays,
  which is slower but correct.
- Element gain patterns are configurable (isotropic, cosine, or a tabulated pattern).
- Per-element gain/phase calibration error is drawn once per sensor and recorded in the metadata as
  truth, so DF algorithms can be scored with and without calibration.

## 6. Propagation and channel model (`modsim/rf/channel.py`)

Per (emitter, sensor) link, evaluated on a time grid fine enough for the dynamics (default 1 ms) and
interpolated in between:

1. **Geometry:** 3D slant range, radial velocity, azimuth/elevation at the sensor, and grazing angle.
2. **Large-scale path loss:** free-space path loss, plus an optional **two-ray ground reflection**
   (configurable ground permittivity/conductivity), plus log-normal shadowing (σ configurable,
   correlated in time).
3. **Radio horizon:** links beyond `√(2kR·h_tx) + √(2kR·h_rx)` (k = 4/3) are culled.
4. **Small-scale fading:** an air-to-ground **Rician** channel whose K-factor rises with elevation
   angle, falling back to Rayleigh near the ground; optional tapped-delay-line multipath with a
   configurable delay spread. Fading is generated with a sum-of-sinusoids (Jakes-like) process whose
   Doppler spread is set by relative speed.
5. **Doppler:** a time-varying frequency shift `f_d(t) = −f_c · v_r(t) / c`, integrated into phase so it
   stays continuous across chunks (e.g. 580 Hz at 5.8 GHz for 30 m/s).
6. **Propagation delay:** absolute fractional delay `r(t)/c` (needed for consistency between sensors,
   and for future TDOA work).

Received power and SNR follow the link budget:
`P_rx = EIRP_tx(θ) + G_rx(θ) − L_path − L_fading`, and `SNR = P_rx − (−174 dBm/Hz + 10·log10(B) + NF)`.

## 7. Receiver model (`modsim/rf/receiver.py`)

Applied after accumulating all emissions and before quantization:

1. Thermal noise at `kTB·NF`, independent per element (seeded per chunk).
2. Background RF contributions (they go through the same emitter → channel path, so they are just more
   emissions).
3. LO offset (ppm → Hz at each band centre) and phase noise. These are shared across array elements
   (common LO), which is what DF algorithms see on real hardware.
4. IQ gain/phase imbalance and DC offset.
5. Per-element gain/phase mismatch (array calibration error).
6. Gain: fixed, or a simple AGC with configurable attack and decay.
7. ADC: clip at full scale, quantize to `adc_bits`, cast to `sample_format`. Clipping events are counted
   in the metadata.

## 8. SigMF output

### 8.1 File layout

```
output/rf/<scenario>/
  manifest.json                         # config hash, seeds, sizing, timing, software version
  truth/
    emissions.parquet                   # every emission (burst/hop), full parameters
    links.parquet                       # per (emission, sensor): SNR, range, AoA, Doppler, detectable
    emitters.parquet                    # emitter registry: id, kind, profile, parent drone/controller
  RF-NORTH-01/
    RF-NORTH-01.sigmf-collection
    RF-NORTH-01_ism2g4.sigmf-meta
    RF-NORTH-01_ism2g4.sigmf-data       # multichannel: interleaved elements (core:num_channels)
    ...
```

Multichannel array data goes **in one dataset per recording, using `core:num_channels`**, with the
samples interleaved per element. This is the most widely supported way to store multichannel data in
SigMF. Per-element files are not supported.

### 8.2 Metadata contents

- **global:** `core:datatype`, `core:sample_rate`, `core:num_channels`, `core:version`,
  `core:recorder: "swarmsim"`, `core:description`, `core:geolocation` (sensor position, GeoJSON), and
  `core:extensions`, which declares the extensions used (below).
- **captures:** one entry per tuning segment, with `core:frequency`, `core:datetime` (scenario epoch +
  offset) and `core:sample_start`.
- **annotations:** emission truth (§8.3).
- **Extensions:**
  - `antenna` for element type and gain.
  - `spatial` for array geometry, element positions and **bearing truth** (azimuth/elevation and the
    associated error fields on annotations).
  - `signal` for modulation class and standard (e.g. `"LoRa"`, `"802.11"`).
  - A small custom namespace, **`swarmsim:`**, for fields with no standard home: emitter/drone id, SNR,
    Doppler, range, detectability, and array calibration truth.
  - Exact field names will be checked against the current published extension schemas, using
    `sigmf-python` for validation.

### 8.3 Annotation granularity

A 500 Hz ELRS link seen by 10 sensors for 5 minutes produces 1.5 M bursts from **one** drone. SigMF
metadata is a single JSON document, so per-burst annotations for large swarms would produce
multi-gigabyte `.sigmf-meta` files that most tools can't load. Granularity is therefore configurable:

| `annotations.granularity` | One annotation per | Notes |
|---|---|---|
| `emission` | burst / hop | Exact; for small scenarios and ML dataset generation |
| `dwell` (default) | contiguous activity of one emitter on one channel | Collapses repeated bursts on the same frequency |
| `session` | emitter active period within a capture | Coarsest; frequency edges cover the whole hop set |

Whatever the granularity, the **full burst-level truth always goes to `truth/*.parquet`**, keyed by the
same ids, so nothing is lost. Each annotation carries:

- `core:sample_start`, `core:sample_count`, `core:freq_lower_edge`, `core:freq_upper_edge`
- `core:label` (profile id) and `core:comment`
- `signal:detail` (modulation, standard)
- `swarmsim:emitter_id`, `swarmsim:object_id`, `swarmsim:role` (`drone` / `controller` / `background`)
- `swarmsim:snr_db`, `swarmsim:rx_power_dbm`, `swarmsim:range_m`, `swarmsim:doppler_hz`
- The spatial bearing truth: azimuth and elevation, plus emitter lat/lon/alt
- `swarmsim:detectable` (bool), `swarmsim:clipped` (bool)

## 9. Scenario schema additions

```yaml
rf:
  epoch: "2026-01-01T00:00:00Z"     # core:datetime base
  window: [0, 60]                   # seconds of the scenario to synthesize (default: all)
  chunk_samples: 4194304
  cull_snr_db: -15                  # skip emissions below this SNR at a sensor
  annotations:
    granularity: dwell
  background:
    - profile: wifi_ap
      density_per_km2: 5
      altitude_m: [2, 15]
      traffic: {offered_load: 0.1}
    - profile: ble_device
      density_per_km2: 50

groups:
  - group_id: fpv_swarm
    # ... existing motion fields ...
    rf:
      links:
        - profile: elrs_900           # drone side: telemetry downlink
          params: {packet_rate_hz: 150, tlm_ratio: 8, power_dbm: 20}
        - profile: analog_fpv
          params: {channel: R3, power_dbm: 25}
      controller:
        placement: per_group          # "per_drone" | "per_group" | "count: N"
        position: [1.0, 1.0, 0.0015]  # km; or "offset_from_spawn: [dx, dy]"
        links:
          - profile: elrs_900         # uplink from the controller
            params: {packet_rate_hz: 150, power_dbm: 27}
  - group_id: shahed_wave
    rf:
      links:
        - profile: silent

rf_sensors:
  - { ... see §5.1 ... }
```

`modsim validate` gains checks for:
- unknown profiles and parameters;
- bands outside the sensor's tuning;
- scanning bands with different sample rates;
- arrays required for `df` sensors;
- a **sizing preflight** (§10) against free disk space.

## 10. Sizing and performance

### 10.1 Storage

Bytes/s per recording = `sample_rate × bytes_per_complex_sample × num_channels`
(`ci16` = 4 B, `cf32` = 8 B).

| Configuration | Rate | 5-minute run |
|---|---|---|
| 1 element, 915 MHz band, 30 MS/s, ci16 | 120 MB/s | 36 GB |
| 1 element, 2.4 GHz ISM, 100 MS/s, ci16 | 400 MB/s | 120 GB |
| 4-element DF, 2.4 GHz, 100 MS/s, ci16 | 1.6 GB/s | 480 GB |
| 4-element DF, 5.8 GHz, 160 MS/s, ci16 | 2.56 GB/s | 768 GB |
| Same as the row above, cf32 | 5.12 GB/s | 1.5 TB |

The CLI prints this estimate in `--dry-run` and `modsim validate`, and refuses to start if the estimate
exceeds free space on the output volume (unless `--force`).

### 10.2 Compute (single machine, multiprocess)

- Expected cost is roughly **10–40 complex MS/s per core per channel**, depending on how many emissions
  overlap a chunk and how much channel realism is enabled. This is an estimate to be benchmarked in
  Phase 1. One 100 MS/s, 4-element sensor needs about 400 MS/s, i.e. about 10–40 cores for real time.
- **Disk write bandwidth will often be the bottleneck before CPU**: 1.6–2.5 GB/s per DF sensor exceeds
  most single NVMe drives sustained.
- `--window`, `--sensors` and `--bands` let users synthesize subsets. `--workers` sets the pool size.
- Optimizations, in order of expected payoff:
  1. SNR culling;
  2. template caching for repeated bursts;
  3. computing the channel on a coarse grid and interpolating;
  4. `float32` math throughout;
  5. `scipy.signal.resample_poly` with cached filters;
  6. pyFFTW, if available.
- The worker interface is a pure function of `(config, unit) → bytes at offset`. Kubernetes or GPU
  backends could be added later without redesign.

### 10.3 Memory

Trajectories are needed only for the objects that emit, and only within `rf.window`. Stage A extracts
compact NumPy arrays of `(t, x, y, z, vx, vy, vz)` per emitter instead of passing `SimObject` lists to
workers. Workers receive them through shared memory.

## 11. Proposed module layout

```
modsim/rf/
  __init__.py
  models.py          # RFSensorConfig, BandConfig, ArrayConfig, LinkProfile, Emitter, Emission, ...
  config.py          # parsing of rf / rf_sensors / groups[].rf
  validators.py      # schema checks + sizing preflight
  profiles/          # *.yaml profile library + loader with per-scenario overrides
  scheduler.py       # Stage A: emitters → emissions (activity models)
  geometry.py        # trajectory interpolation, range/AoA/Doppler
  channel.py         # path loss, two-ray, fading, delay
  waveforms/         # ofdm.py, lora.py, fsk.py, fm.py
  array.py           # steering vectors, element patterns, calibration error
  receiver.py        # noise, impairments, AGC, ADC
  synth.py           # Stage B worker: (sensor, band, chunk) → samples
  runner.py          # orchestration, multiprocessing pool, memmaps, progress
  sigmf_writer.py    # Stage C: .sigmf-meta / .sigmf-collection
  truth.py           # parquet truth tables
```

New CLI command:

```
modsim rf <scenario.yaml> [--out output/rf] [--workers N] [--window START:END]
                          [--sensors ID,...] [--bands NAME,...] [--dry-run] [--force]
```

New optional dependency group `rf = ["sigmf", "pyarrow"]` (plus an optional `pyfftw`). The core install
is unchanged.

Later extensions, out of scope for this design: terrain/DEM line-of-sight, bit-exact Remote ID/Wi-Fi
framing, TDOA network sensors, a GPU backend, and a Kubernetes job runner.

## 12. Phasing

| Phase | Scope | Exit criteria |
|---|---|---|
| **0: Plumbing and truth** ✅ | Models, YAML parsing and validation, profile library, controllers, silent drones, Stage A scheduler, link budget, detectability, parquet truth, `--dry-run` sizing | `modsim rf --dry-run` produces truth tables and sizing for a sample scenario; unit tests for path loss, horizon, scheduling |
| **1: Single-channel IQ** | Fixed tuning, noise + FSPL + Doppler, `lora` + `fm` waveforms (ELRS, analog FPV), chunked multiprocess writer, SigMF writer, determinism | Measured in-band SNR matches the link budget within ±0.5 dB; Doppler tone within ±1 Hz; output byte-identical for 1 vs N workers; `sigmf` validation passes; throughput benchmark recorded |
| **2: Remaining profiles + clutter** | `ofdm` + `fsk` waveforms (DJI, Wi-Fi, Remote ID BLE/Wi-Fi), background RF, CSMA traffic, annotation granularity modes | Spectrogram/occupancy checks per profile against reference shapes; annotations line up with energy in the spectrogram |
| **3: Arrays and DF** | Array steering, element patterns, calibration error, spatial annotations, multichannel SigMF | A MUSIC/Bartlett estimator in the test suite recovers synthetic bearings within tolerance at high SNR |
| **4: Realism and tuning modes** | Two-ray, Rician/TDL fading, receiver impairments, AGC/ADC clipping, multi-band and scanning tuning | Fading statistics match the target K-factor; scanning recordings have correct `captures` segments; clipping is counted |
| **5: Performance** | Template caching, coarse-grid channel, profiling, optional pyFFTW | Benchmarks published in the README next to the existing scaling table |

### Phase 0 implementation notes

Where the Phase 0 code differs from, or adds detail to, the design above:

- **Controllers:** every drone gets its own controller radio (`<obj>/ctl/<profile>`), because real
  controllers bind 1:1. `placement` (`per_drone`, `per_group` or `{count: N}`) only sets how many
  operator **sites** those radios share. Both ends of a link share a `link_key`, so hop sequence,
  slot phase and "random" channel choice agree. With `tlm_ratio: N`, the drone owns every Nth slot
  (telemetry) and the controller owns the rest.
- **Profiles** are YAML files in `modsim/rf/profiles/`; each lists its public sources and is marked
  `calibration: provisional`. A scenario can add or patch profiles under `rf.profiles` (use
  `base: <id>` to inherit from an existing profile).
- **Truth tables** are written as `truth/{emitters,emissions,links}.parquet`, falling back to CSV when
  pyarrow (the `rf` extra) isn't installed. `links` holds only rows above `rf.cull_snr_db`; for a
  scanning sensor, a row also needs a non-zero `observed_fraction` (the share of the emission that
  falls inside that band's dwells).
- **Link budget:** free-space loss, smooth-earth horizon, receiver gain, flat-spectrum band overlap,
  and noise over the overlapping bandwidth. Two-ray, shadowing and fading remain Phase 4.
- **CLI:** `modsim rf` currently requires `--dry-run`. `--workers`, `--bands` and the free-space
  `--force` override arrive with synthesis in Phase 1. `modsim validate` prints the RF summary and the
  IQ size preflight (as a warning).
- **Scale check:** 1,000 ELRS + FPV drones (2,000 radios with controllers) against 10 sensors for 30 s
  gives 4.5 M emissions and 45 M link rows in about 70 s on one core. Most of that time goes to
  allocation and truth writing, which Phase 5 can optimize.
- **Example:** `scenarios/rf_fpv_mix.yaml` (FPV/ELRS, DJI, silent drones, background Wi-Fi/BLE; fixed,
  multi-band DF and scanning sensors).

## 13. Testing strategy

- **Physics unit tests:** FSPL and two-ray against closed form; horizon cutoff; Doppler sign and
  magnitude; SNR measured from synthesized samples vs link-budget prediction.
- **Waveform tests:** occupied bandwidth (99% power) and spectral shape per generator; LoRa chirp slope;
  OFDM subcarrier spacing; FM deviation.
- **Determinism:** identical bytes across worker counts and chunk sizes (where filters allow), and
  identical truth for the same seed.
- **SigMF conformance:** validate every produced `.sigmf-meta` with `sigmf-python`.
- **DF:** a bearing estimator on synthesized array data recovers truth; calibration error degrades it as
  expected.
- **CI:** small, fast scenarios only (≤ 1 s of IQ at low sample rates). Large benchmarks run manually.

## 14. Resolved questions

| Question | Decision |
|---|---|
| Default annotation granularity | `dwell`. Use `emission` explicitly for small ML-dataset runs. |
| `core:datetime` base | Fixed epoch from `rf.epoch`, so reruns with the same seed produce identical metadata |
| Profile calibration | Public sources only; sources cited in each profile file |
| Array layout | Interleaved multichannel in one dataset per recording |
| Scanning retune gaps | Zero-filled samples, each marked by a `swarmsim:retune` annotation |
| Dependencies | `sigmf` + `pyarrow` in an optional `rf` extra |
