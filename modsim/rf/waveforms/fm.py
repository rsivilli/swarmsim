"""Analog FPV video: FM-modulated composite video with an FM audio subcarrier (protocol-like)."""

from __future__ import annotations

import numpy as np

LINE_S = 63.5556e-6  # NTSC line period
LINES = 525
SYNC_S = 4.7e-6
BACK_PORCH_S = 4.7e-6
FRONT_PORCH_S = 1.5e-6
SEGMENTS = 64  # luminance segments per active line

SYNC_LEVEL = -1.0
BLANK_LEVEL = -0.6
BLACK, WHITE = -0.4, 1.0


def _frame_table(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment start times, levels and cumulative integral of the modulating signal (one frame)."""
    rng = np.random.default_rng([seed, 0xF1])
    active = LINE_S - SYNC_S - BACK_PORCH_S - FRONT_PORCH_S
    seg = active / SEGMENTS
    # Smooth-ish random picture: per-line random walk of luminance
    lum = np.clip(
        np.cumsum(rng.normal(0, 0.15, (LINES, SEGMENTS)), axis=1) + rng.uniform(0, 1, (LINES, 1)),
        0,
        1,
    )
    levels_line = [SYNC_LEVEL, BLANK_LEVEL]
    durs_line = [SYNC_S, BACK_PORCH_S]
    levels, durs = [], []
    for line in range(LINES):
        levels += levels_line + list(BLACK + (WHITE - BLACK) * lum[line]) + [BLANK_LEVEL]
        durs += durs_line + [seg] * SEGMENTS + [FRONT_PORCH_S]
    levels = np.asarray(levels)
    durs = np.asarray(durs)
    starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
    cum = np.concatenate([[0.0], np.cumsum(levels * durs)])
    return starts, levels, cum


_CACHE: dict[int, tuple] = {}


def fm_video(
    u: np.ndarray,
    *,
    seed: int,
    deviation_hz: float = 4.5e6,
    audio_subcarrier_hz: float = 6.5e6,
    audio_level: float = 0.15,
) -> np.ndarray:
    """Phase of FM video at times ``u`` after transmitter start (peak deviation ``deviation_hz``)."""
    if seed not in _CACHE:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[seed] = _frame_table(seed)
    starts, levels, cum = _CACHE[seed]
    frame_s = LINES * LINE_S
    u = np.asarray(u, dtype=np.float64)

    if len(u) == 0:
        return np.zeros(0)
    # Within one segment the modulating level L is constant, so the integral is A + L·u.
    # Enumerate the segments spanned by u (u ascends) and expand per-segment coefficients.
    n_seg = len(levels)
    g0 = _segment_index(u[0], starts, frame_s, n_seg)
    g1 = _segment_index(u[-1], starts, frame_s, n_seg)
    g = np.arange(g0, g1 + 1)
    frame, i = np.divmod(g, n_seg)
    seg_start = frame * frame_s + starts[i]
    lev = levels[i]
    coef = frame * cum[-1] + cum[i] - lev * seg_start
    counts = np.diff(
        np.concatenate([[0], np.searchsorted(u, seg_start[1:], side="left"), [len(u)]])
    )
    integral = np.repeat(coef, counts) + np.repeat(lev, counts) * u
    if audio_level:
        w = 2 * np.pi * audio_subcarrier_hz
        cyc = np.mod(audio_subcarrier_hz * u, 1.0).astype(np.float32)
        integral = integral + audio_level * (1.0 - np.cos(2 * np.pi * cyc)) / w
    return 2 * np.pi * deviation_hz * integral


def _segment_index(t: float, starts: np.ndarray, frame_s: float, n_seg: int) -> int:
    frame = int(np.floor(t / frame_s))
    r = t - frame * frame_s
    i = int(np.clip(np.searchsorted(starts, r, side="right") - 1, 0, n_seg - 1))
    return frame * n_seg + i
