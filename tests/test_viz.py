"""Tests for animation frame selection."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest

from modsim import viz
from tests.conftest import small_scenario


@pytest.fixture
def captured_frames(monkeypatch):
    """Capture the frame count passed to FuncAnimation instead of rendering."""
    captured = {}

    class FakeAnimation:
        def __init__(self, fig, func, frames, **kwargs):
            captured["frames"] = frames

    monkeypatch.setattr(viz.animation, "FuncAnimation", FakeAnimation)
    monkeypatch.setattr(viz.plt, "show", lambda: None)
    return captured


@pytest.mark.parametrize(("frame_step", "expected"), [(1, 10), (3, 4), (10, 1)])
def test_frame_step_subsamples_timesteps(captured_frames, frame_step, expected):
    viz.animate_scenario(small_scenario(), map_style="basic", frame_step=frame_step)
    assert captured_frames["frames"] == expected


def test_frame_step_must_be_positive():
    with pytest.raises(ValueError):
        viz.animate_scenario(small_scenario(), frame_step=0)
