"""Plot the SigMF recordings from a ``modsim rf`` run against their ground truth.

Reads only what the run wrote to disk: ``manifest.json``, the ``truth/`` tables and the SigMF
recordings. It produces three kinds of figure:

- ``rf_overview.png``: a spectrogram of every recording (channel 0) with the truth emissions boxed.
  Solid boxes are emissions the sensor can detect, dashed boxes fall below its threshold. Scanning
  recordings label each dwell and hatch the retune gaps.
- ``rf_zoom.png`` (with ``--zoom``): burst-level views of time slices, every detectable emission
  boxed, background included.
- ``rf_df_<recording>.png`` (multichannel recordings): MUSIC bearing estimates computed from the
  SigMF file alone (``spatial:element_geometry`` and ``spatial:aperture_azimuth``), compared with
  the ``spatial:signal_bearing`` truth. The table also gives each burst's margin over the strongest
  emission that overlaps it in time and frequency. A negative margin usually explains a miss.

Usage::

    uv run modsim rf scenarios/rf_fpv_mix.yaml --window 20:20.5
    uv run python examples/rf_plot_recordings.py output/rf/rf_fpv_mix \\
        --zoom RF-01_ism915:0:60 --zoom RF-04_ism2g4:200:8 --df RF-02_ism2g4

Truth tables can be Parquet (needs pyarrow, from the ``rf`` extra) or CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

C_MPS = 299_792_458.0

# Categorical colours in a fixed order. Profiles not listed fall back to a neutral grey.
COLORS = {
    "elrs_900": "#2a78d6",
    "analog_fpv": "#eb6834",
    "dji_ocusync_dl": "#1baf7a",
    "dji_ocusync_ul": "#eda100",
    "dji_droneid": "#e87ba4",
    "remote_id_ble": "#008300",
    "wifi_ap": "#6250d6",
    "ble_device": "#e34948",
}
OTHER = "#8a8984"
TEXT2 = "#52514e"
SURFACE = "#fcfcfb"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.edgecolor": "#b5b4ae",
        "axes.labelcolor": TEXT2,
        "xtick.color": TEXT2,
        "ytick.color": TEXT2,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    }
)


def color(profile: str) -> str:
    return COLORS.get(profile, OTHER)


def profile_order(profiles) -> list[str]:
    known = [p for p in COLORS if p in profiles]
    return known + sorted(set(profiles) - set(known))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def read_table(truth_dir: Path, name: str) -> dict[str, np.ndarray]:
    """A truth table as numpy columns, from Parquet or CSV."""
    pq_path = truth_dir / f"{name}.parquet"
    if pq_path.exists():
        import pyarrow.parquet as pq

        cols = pq.read_table(pq_path).to_pydict()
        return {k: np.asarray(v) for k, v in cols.items()}
    with open(truth_dir / f"{name}.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0] if rows else []:
        vals = [r[k] for r in rows]
        try:
            out[k] = np.asarray(vals, dtype=np.float64)
        except ValueError:
            out[k] = np.asarray(vals)
    for k in ("emission_idx", "emitter_idx", "sensor_idx", "band_idx"):
        if k in out and out[k].dtype.kind == "f":
            out[k] = out[k].astype(np.int64)
    if "detectable" in out and out["detectable"].dtype.kind != "b":
        out["detectable"] = np.isin(out["detectable"].astype(str), ["True", "true", "1", "1.0"])
    return out


class Truth:
    """The run's manifest and truth tables, with per-emission profile and kind."""

    def __init__(self, run_dir: Path):
        self.manifest = json.loads((run_dir / "manifest.json").read_text())
        self.sensor_ids = [s["id"] for s in self.manifest["sensors"]]
        emitters = read_table(run_dir / "truth", "emitters")
        self.em = read_table(run_dir / "truth", "emissions")
        self.lk = read_table(run_dir / "truth", "links")
        self.profile = np.asarray(emitters["profile"])[self.em["emitter_idx"]]
        self.kind = np.asarray(emitters["kind"])[self.em["emitter_idx"]]
        self.t0 = float(self.manifest["window_s"][0])

    def sensor(self, sensor_id: str) -> dict:
        return self.manifest["sensors"][self.sensor_ids.index(sensor_id)]

    def band_index(self, sensor_id: str, center_hz: float) -> int:
        centers = [b["center_hz"] for b in self.sensor(sensor_id)["bands"]]
        return int(np.argmin(np.abs(np.asarray(centers) - center_hz)))

    def link_mask(self, sensor_id: str, band_idx: int) -> np.ndarray:
        return (self.lk["sensor_idx"] == self.sensor_ids.index(sensor_id)) & (
            self.lk["band_idx"] == band_idx
        )

    def rows(self, sensor_id, band_idx, t0, t1, roles, detectable=True) -> np.ndarray:
        """Emission indices heard by a sensor band in [t0, t1) seconds of scenario time."""
        m = self.link_mask(sensor_id, band_idx) & (self.lk["detectable"] == detectable)
        e = self.lk["emission_idx"][m]
        em = self.em
        keep = np.isin(self.kind[e], roles)
        keep &= (em["t_start"][e] < t1) & (em["t_start"][e] + em["duration"][e] > t0)
        return e[keep]


_DATATYPE = re.compile(r"^(?P<cplx>[cr])(?P<kind>[fiu])(?P<bits>\d+)(?:_(?P<endian>le|be))?$")


class Recording:
    """A memory-mapped SigMF recording, shape (samples, channels)."""

    def __init__(self, data_path: Path):
        self.path = data_path
        self.name = data_path.name.removesuffix(".sigmf-data")
        self.meta = json.loads(data_path.with_name(f"{self.name}.sigmf-meta").read_text())
        g = self.meta["global"]
        self.fs = float(g["core:sample_rate"])
        self.channels = int(g.get("core:num_channels", 1))
        self.sensor_id = g["swarmsim:sensor_id"]
        self.captures = self.meta["captures"]
        self.annotations = self.meta["annotations"]
        m = _DATATYPE.match(g["core:datatype"])
        if not m or m["cplx"] != "c":
            raise ValueError(f"{self.name}: unsupported datatype {g['core:datatype']}")
        bits = int(m["bits"])
        order = {"le": "<", "be": ">"}.get(m["endian"] or "", "|")
        self._dtype = np.dtype(f"{order}{m['kind']}{bits // 8}")
        self._offset = 2.0 ** (bits - 1) if m["kind"] == "u" else 0.0
        raw = np.memmap(data_path, dtype=self._dtype, mode="r")
        self._raw = raw.reshape(-1, self.channels, 2)
        self.samples = self._raw.shape[0]

    @property
    def duration(self) -> float:
        return self.samples / self.fs

    def iq(self, start: int, n: int, channel: int | None = 0) -> np.ndarray:
        """Complex samples [start, start + n): (n,) for one channel, (n, channels) for None."""
        sl = (
            self._raw[start : start + n]
            if channel is None
            else self._raw[start : start + n, channel]
        )
        x = np.asarray(sl, dtype=np.float32) - self._offset
        return x[..., 0] + 1j * x[..., 1]


def find_recordings(run_dir: Path) -> list[Recording]:
    return [Recording(p) for p in sorted(run_dir.glob("*/*.sigmf-data"))]


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def spectrogram(rec: Recording, start: int, n: int, ncols: int, nfft=1024, navg=8) -> np.ndarray:
    """Power (dB), shape (nfft, ncols). Each column averages navg FFTs spread across its span."""
    win = np.hanning(nfft).astype(np.float32)
    step = max(n // ncols, 1)
    cols = []
    for c in range(ncols):
        s0 = start + c * step
        p = 0.0
        for o in np.linspace(0, max(step - nfft, 0), navg).astype(int):
            p = p + np.abs(np.fft.fftshift(np.fft.fft(rec.iq(s0 + o, nfft) * win))) ** 2
        cols.append(10 * np.log10(p / navg + 1e-12))
    return np.array(cols).T


def show(ax, spec: np.ndarray, extent) -> None:
    lo = np.median(spec) - 2
    hi = max(np.percentile(spec, 99.9), lo + 20)
    ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="Greys",
        vmin=lo,
        vmax=hi,
        interpolation="nearest",
    )


def draw_boxes(ax, truth: Truth, rows, center_hz, tclip=None, lw=1.2, ls="-") -> set[str]:
    """Outline emissions in (ms since window start, MHz offset from center_hz)."""
    em, seen = truth.em, set()
    for e in rows:
        ts = em["t_start"][e] - truth.t0
        te = ts + em["duration"][e]
        if tclip:
            ts, te = max(ts, tclip[0]), min(te, tclip[1])
            if te <= ts:
                continue
        f0 = (em["f_center_hz"][e] - em["bandwidth_hz"][e] / 2 - center_hz) / 1e6
        p = str(truth.profile[e])
        ax.add_patch(
            Rectangle(
                (ts * 1e3, f0),
                (te - ts) * 1e3,
                em["bandwidth_hz"][e] / 1e6,
                fill=False,
                ec=color(p),
                lw=lw,
                ls=ls,
            )
        )
        seen.add(p)
    return seen


def legend(fig, profiles, dashed=False) -> None:
    order = profile_order(profiles)
    handles = [Rectangle((0, 0), 1, 1, fill=False, ec=color(p), lw=1.6) for p in order]
    if dashed:
        handles.append(Rectangle((0, 0), 1, 1, fill=False, ec=TEXT2, lw=1.2, ls="--"))
        order.append("dashed = below sensor's detection threshold")
    fig.legend(
        handles,
        order,
        loc="lower center",
        ncol=min(len(handles), 7),
        frameon=False,
        title="Truth emission boxes",
    )


def corner_note(ax, text: str) -> None:
    ax.text(
        0.99,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        fontsize=7,
        color=TEXT2,
        bbox=dict(fc=SURFACE, ec="none", alpha=0.8),
    )


def grid(n: int, ncols: int, panel=(7.0, 4.2)):
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel[0] * ncols, panel[1] * nrows + 0.8))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)
    return fig, axes


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

DRONE_ROLES = ("drone", "controller")
ALL_ROLES = ("drone", "controller", "background")


def retunes(rec: Recording) -> list[tuple[float, float]]:
    """Zero-filled retune spans (s since recording start) of a scanning recording."""
    return [
        (
            a["core:sample_start"] / rec.fs,
            (a["core:sample_start"] + a["core:sample_count"]) / rec.fs,
        )
        for a in rec.annotations
        if a.get("core:label") == "retune"
    ]


def dwells(rec: Recording) -> list[tuple[float, float, float]]:
    """(start s, end s, centre Hz) per capture segment, excluding retune spans at its edges."""
    edges = [c["core:sample_start"] / rec.fs for c in rec.captures] + [rec.duration]
    out = []
    for i, cap in enumerate(rec.captures):
        lo, hi = edges[i], edges[i + 1]
        for r0, r1 in retunes(rec):
            if r0 <= lo < r1:
                lo = r1
            elif r0 < hi <= r1:
                hi = r0
        if hi > lo:
            out.append((lo, hi, cap["core:frequency"]))
    return out


def overlay_truth(ax, truth: Truth, rec: Recording, t_lo, t_hi, roles, weak, lw=1.2) -> set[str]:
    """Box the truth emissions heard in [t_lo, t_hi) s, per dwell; hatch retune gaps."""
    used: set[str] = set()
    for d_lo, d_hi, fc in dwells(rec):
        seg = (max(d_lo, t_lo), min(d_hi, t_hi))
        if seg[1] <= seg[0]:
            continue
        bi = truth.band_index(rec.sensor_id, fc)
        span = (truth.t0 + seg[0], truth.t0 + seg[1])
        if weak:
            rows = truth.rows(rec.sensor_id, bi, *span, roles, detectable=False)
            used |= draw_boxes(ax, truth, rows, fc, tclip=seg, lw=0.9, ls="--")
        rows = truth.rows(rec.sensor_id, bi, *span, roles)
        used |= draw_boxes(ax, truth, rows, fc, tclip=seg, lw=lw)
    for r0, r1 in retunes(rec):
        if r0 < t_hi and r1 > t_lo:
            ax.axvspan(r0 * 1e3, r1 * 1e3, color="#d9d8d3", hatch="////", ec="#b5b4ae", lw=0)
    return used


def plot_overview(truth: Truth, recs: list[Recording], out: Path) -> None:
    fig, axes = grid(len(recs), 2)
    used: set[str] = set()
    for ax, rec in zip(axes, recs):
        show(
            ax,
            spectrogram(rec, 0, rec.samples, 600),
            [0, rec.duration * 1e3, -rec.fs / 2e6, rec.fs / 2e6],
        )
        sensor = truth.sensor(rec.sensor_id)
        used |= overlay_truth(ax, truth, rec, 0.0, rec.duration, DRONE_ROLES, weak=True)
        for t_lo, _, fc in dwells(rec) if len(rec.captures) > 1 else []:
            name = sensor["bands"][truth.band_index(rec.sensor_id, fc)]["name"]
            ax.text(t_lo * 1e3 + 1, rec.fs / 2e6 * 0.92, name, fontsize=7, color=TEXT2)
        if len(rec.captures) > 1:
            bands = "/".join(b["name"] for b in sensor["bands"])
            title = f"{rec.name}  —  scanning {bands}, {rec.fs / 1e6:g} MS/s"
            ax.set_ylabel("Offset from dwell centre (MHz)")
        else:
            fc = rec.captures[0]["core:frequency"]
            array = f", {rec.channels}-ch array (ch 0)" if rec.channels > 1 else ""
            title = f"{rec.name}  —  {fc / 1e9:.4g} GHz, {rec.fs / 1e6:g} MS/s{array}"
            ax.set_ylabel(f"Offset from {fc / 1e6:g} MHz (MHz)")
        n_ann = sum(1 for a in rec.annotations if a.get("core:label") != "retune")
        ax.set_title(title, loc="left")
        ax.set_xlabel(f"Time since t = {truth.t0:g} s (ms)")
        corner_note(ax, f"{n_ann} annotations")
    legend(fig, used, dashed=True)
    span = truth.manifest["window_s"][1] - truth.t0
    fig.suptitle(
        f"{truth.manifest['scenario']} — {span:g} s of synthesized IQ per SigMF recording "
        "(darker = more power)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_zooms(truth: Truth, recs: dict[str, Recording], zooms, out: Path) -> None:
    fig, axes = grid(len(zooms), 2, panel=(7.0, 4.6))
    used: set[str] = set()
    for ax, (name, start_ms, length_ms) in zip(axes, zooms):
        rec = recs[name]
        z0, zlen = start_ms / 1e3, length_ms / 1e3
        s0 = int(z0 * rec.fs)
        nfft = 1024 if rec.fs >= 50e6 else 512
        spec = spectrogram(rec, s0, int(zlen * rec.fs), 500, nfft=nfft, navg=2)
        show(ax, spec, [z0 * 1e3, (z0 + zlen) * 1e3, -rec.fs / 2e6, rec.fs / 2e6])
        used |= overlay_truth(ax, truth, rec, z0, z0 + zlen, ALL_ROLES, weak=False, lw=1.0)
        ax.set_xlim(z0 * 1e3, (z0 + zlen) * 1e3)
        ax.set_title(f"{rec.name}  —  {length_ms:g} ms zoom", loc="left")
        ax.set_xlabel(f"Time since t = {truth.t0:g} s (ms)")
        segs = [d for d in dwells(rec) if d[0] < z0 + zlen and d[1] > z0]
        fcs = {fc for _, _, fc in segs} or {rec.captures[0]["core:frequency"]}
        if len(fcs) == 1:
            ax.set_ylabel(f"Offset from {fcs.pop() / 1e6:g} MHz (MHz)")
        else:
            ax.set_ylabel("Offset from dwell centre (MHz)")
        corner_note(ax, "detectable emissions only")
    legend(fig, used)
    fig.suptitle(
        "Burst-level detail with ground-truth emission boxes",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.94])
    fig.savefig(out, dpi=110)
    plt.close(fig)


def music_bearings(truth: Truth, rec: Recording, per_profile=3, max_samples=2_000_000):
    """MUSIC azimuth for the strongest detectable annotation of each emitter.

    Uses only the SigMF file: element positions from ``spatial:element_geometry`` (array frame:
    +X boresight, +Y left, +Z up) and ``spatial:aperture_azimuth``. Azimuth is clockwise from
    North. Matches ``modsim.rf.array.steering_vector``: an element nearer the emitter sees a
    phase advance.
    """
    cap = rec.captures[0]
    fc = cap["core:frequency"]
    pos = np.array([e["point"] for e in cap["spatial:element_geometry"]])
    boresight = cap.get("spatial:aperture_azimuth", 0.0)
    az_grid = np.arange(0, 360, 0.25)
    rel = np.radians(az_grid - boresight)
    d = np.stack([np.cos(rel), -np.sin(rel), np.zeros_like(rel)], -1)
    band_mask = truth.link_mask(rec.sensor_id, truth.band_index(rec.sensor_id, fc))
    em, lk = truth.em, truth.lk

    cands = [a for a in rec.annotations if a.get("swarmsim:detectable") and "signal:emitter" in a]
    cands.sort(key=lambda a: -a["swarmsim:snr_db"])
    picked, per_label = {}, {}
    for a in cands:
        label, emitter = a["core:label"], a["signal:emitter"]["label"]
        if emitter not in picked and per_label.get(label, 0) < per_profile:
            picked[emitter] = a
            per_label[label] = per_label.get(label, 0) + 1

    order = profile_order({a["core:label"] for a in picked.values()})
    results = []
    for a in sorted(picked.values(), key=lambda a: order.index(a["core:label"])):
        e = a["swarmsim:emissions"][0]  # first burst of this annotation
        s0 = max(int((em["t_start"][e] - truth.t0) * rec.fs), a["core:sample_start"])
        n = min(int(em["duration"][e] * rec.fs), max_samples, rec.samples - s0)
        x = rec.iq(s0, n, channel=None)  # (n, channels)
        lo, hi = a["core:freq_lower_edge"], a["core:freq_upper_edge"]
        spec = np.fft.fft(x, axis=0)
        f = np.fft.fftfreq(n, 1 / rec.fs) + fc
        spec[(f < lo) | (f > hi)] = 0  # keep only this emitter's band
        x = np.fft.ifft(spec, axis=0)
        r = x.T @ x.conj() / n  # R = E[x x^H]
        _, v = np.linalg.eigh(r)
        noise = v[:, :-1]
        a_grid = np.exp(2j * np.pi * (lo + hi) / 2 * (d @ pos.T) / C_MPS)
        p = 1 / np.sum(np.abs(a_grid.conj() @ noise) ** 2, axis=1)
        est = float(az_grid[np.argmax(p)])
        true_az = a["spatial:signal_bearing"]["azimuth"]

        # Margin over the strongest other emission overlapping this burst in time and frequency
        t0, t1 = em["t_start"][e], em["t_start"][e] + em["duration"][e]
        sel = band_mask & (lk["emission_idx"] != e)
        o = lk["emission_idx"][sel]
        of0 = em["f_center_hz"][o] - em["bandwidth_hz"][o] / 2
        hit = (em["t_start"][o] < t1) & (em["t_start"][o] + em["duration"][o] > t0)
        hit &= (of0 < hi) & (of0 + em["bandwidth_hz"][o] > lo)
        margin = a["swarmsim:snr_db"] - lk["snr_db"][sel][hit].max() if hit.any() else np.inf

        results.append(
            {
                "profile": a["core:label"],
                "emitter": a["signal:emitter"]["label"],
                "snr_db": a["swarmsim:snr_db"],
                "margin_db": margin,
                "truth_az": true_az,
                "est_az": est,
                "error": (est - true_az + 180) % 360 - 180,
                "spectrum": p / p.max(),
            }
        )
    return az_grid, results


def plot_df(truth: Truth, rec: Recording, out: Path) -> list[dict]:
    az_grid, results = music_bearings(truth, rec)
    fig = plt.figure(figsize=(14, 1.6 + 0.3 * max(len(results), 12)))
    axp = fig.add_subplot(1, 2, 1, projection="polar")
    axp.set_theta_zero_location("N")
    axp.set_theta_direction(-1)
    for r in results:
        c = color(r["profile"])
        axp.plot(np.radians(az_grid), 10 * np.log10(r["spectrum"]) + 30, color=c, lw=1, alpha=0.8)
        axp.plot([np.radians(r["truth_az"])] * 2, [0, 32], color=c, lw=1.6, ls="--")
    axp.set_ylim(0, 32)
    axp.set_yticklabels([])
    axp.set_title("MUSIC spectra (solid) vs truth bearing (dashed)", loc="left", pad=18)

    axt = fig.add_subplot(1, 2, 2)
    axt.axis("off")
    if results:
        cells = [
            [
                r["profile"],
                r["emitter"],
                f"{r['snr_db']:.1f}",
                "clear" if np.isinf(r["margin_db"]) else f"{r['margin_db']:+.1f}",
                f"{r['truth_az']:.1f}°",
                f"{r['est_az']:.1f}°",
                f"{r['error']:+.1f}°",
            ]
            for r in results
        ]
        header = ["profile", "emitter", "SNR dB", "vs strongest\nco-channel dB"]
        header += ["truth az", "MUSIC az", "error"]
        tbl = axt.table(
            cellText=cells, colLabels=header, loc="center", cellLoc="left", colLoc="left"
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        tbl.auto_set_column_width(list(range(len(header))))
        for (row, col), cell in tbl.get_celld().items():
            cell.set_edgecolor("#e4e3de")
            if row == 0:
                cell.set_text_props(fontweight="bold", color=TEXT2)
            elif col == 0:
                cell.get_text().set_color(color(results[row - 1]["profile"]))
    else:
        axt.text(0.5, 0.5, "no detectable emitters", ha="center", color=TEXT2)

    fig.suptitle(
        f"{rec.name} direction finding — bearing estimated from the {rec.channels}-ch SigMF file "
        "alone (element geometry from spatial:element_geometry)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_zoom(text: str) -> tuple[str, float, float]:
    try:
        name, start, length = text.rsplit(":", 2)
        return name, float(start), float(length)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected RECORDING:START_MS:LENGTH_MS, got {text!r}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, help="modsim rf output directory (has manifest.json)")
    ap.add_argument("--out", type=Path, help="Where to write PNGs (default: RUN_DIR/plots)")
    ap.add_argument(
        "--zoom",
        type=parse_zoom,
        action="append",
        default=[],
        metavar="REC:START_MS:LEN_MS",
        help="Zoomed panel, e.g. RF-01_ism915:0:60 (ms from the window start); repeatable",
    )
    ap.add_argument(
        "--df",
        action="append",
        metavar="REC",
        help="Multichannel recording to run MUSIC on; repeatable (default: all multichannel)",
    )
    args = ap.parse_args(argv)

    truth = Truth(args.run_dir)
    recs = find_recordings(args.run_dir)
    if not recs:
        raise SystemExit(f"No .sigmf-data under {args.run_dir}: run modsim rf without --dry-run")
    by_name = {r.name: r for r in recs}
    for name in [z[0] for z in args.zoom] + (args.df or []):
        if name not in by_name:
            raise SystemExit(f"Unknown recording {name!r}; have {', '.join(by_name)}")
    out = args.out or args.run_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)

    plot_overview(truth, recs, out / "rf_overview.png")
    print(f"✓ {out / 'rf_overview.png'}")
    if args.zoom:
        plot_zooms(truth, by_name, args.zoom, out / "rf_zoom.png")
        print(f"✓ {out / 'rf_zoom.png'}")
    df_names = args.df or [
        r.name for r in recs if r.channels > 1 and "spatial:element_geometry" in r.captures[0]
    ]
    for name in df_names:
        path = out / f"rf_df_{name}.png"
        results = plot_df(truth, by_name[name], path)
        print(f"✓ {path}")
        for r in results:
            margin = "clear" if np.isinf(r["margin_db"]) else f"{r['margin_db']:+.1f} dB"
            print(
                f"    {r['profile']:15s} {r['emitter']:30s} SNR {r['snr_db']:5.1f} dB  "
                f"truth {r['truth_az']:6.1f}°  MUSIC {r['est_az']:6.1f}°  "
                f"error {r['error']:+6.1f}°  co-channel {margin}"
            )


if __name__ == "__main__":
    main()
