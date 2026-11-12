"""Checks on synthetic trajectories where the right answer is known."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from footperf import Thresholds, add_speed, detect_efforts, player_summary  # noqa: E402

FPS = 10


def make_track(speed_ms: np.ndarray, noise_m: float = 0.0, seed: int = 0, frame0: int = 0) -> pd.DataFrame:
    """Player running along x with a given speed profile (one value per frame)."""
    rng = np.random.default_rng(seed)
    x = np.concatenate(([0.0], np.cumsum(speed_ms[1:] / FPS)))
    x = x + rng.normal(0, noise_m, len(x))
    n = len(x)
    return pd.DataFrame({
        "frame": np.arange(frame0, frame0 + n), "period": 1, "player_id": 7,
        "x": x, "y": rng.normal(0, noise_m, n), "detected": True,
    })


def test_constant_speed_is_recovered():
    tr = add_speed(make_track(np.full(600, 5.0)))
    assert np.nanmedian(tr.speed) == pytest.approx(5.0, abs=1e-6)


def test_distance_with_noise_within_3_percent():
    # 60 s at 4 m/s = 240 m, with 10 cm positional jitter
    tr = add_speed(make_track(np.full(600, 4.0), noise_m=0.10, seed=1))
    s = player_summary(tr).iloc[0]
    assert s.total_distance_m == pytest.approx(240, rel=0.03)


def test_raw_differences_overestimate_distance_with_noise():
    """The reason smoothing exists: naive frame differences inflate distance."""
    tr = make_track(np.full(600, 1.0), noise_m=0.10, seed=2)
    naive = np.hypot(np.diff(tr.x), np.diff(tr.y)).sum()
    smooth = player_summary(add_speed(tr)).iloc[0].total_distance_m
    assert naive > 1.3 * 60  # true distance is 60 m
    assert smooth == pytest.approx(60, rel=0.10)


def test_one_sprint_detected():
    # walk 10 s, sprint 3 s at 8 m/s (28.8 km/h), walk 10 s
    v = np.concatenate([np.full(100, 1.5), np.full(30, 8.0), np.full(100, 1.5)])
    e = detect_efforts(add_speed(make_track(v), window=5))
    assert len(e) == 1
    assert e.iloc[0].kind == "sprint"
    assert e.iloc[0].peak_kmh == pytest.approx(28.8, abs=0.5)


def test_short_burst_not_counted():
    # 0.4 s above 25 km/h is below the 0.7 s minimum
    v = np.concatenate([np.full(100, 1.5), np.full(4, 8.0), np.full(100, 1.5)])
    e = detect_efforts(add_speed(make_track(v), window=3, poly=1), Thresholds())
    assert (e.kind == "sprint").sum() == 0


def test_hsr_without_sprint():
    v = np.concatenate([np.full(100, 2.0), np.full(30, 6.3), np.full(100, 2.0)])  # 22.7 km/h
    e = detect_efforts(add_speed(make_track(v), window=5))
    assert list(e.kind) == ["hsr"]


def test_no_derivative_across_gap():
    """A player missing for 5 s then reappearing 30 m away must not produce a fake sprint."""
    a = make_track(np.full(50, 1.0))
    b = make_track(np.full(50, 1.0), frame0=100)
    b["x"] += 30
    tr = add_speed(pd.concat([a, b], ignore_index=True))
    assert np.nanmax(tr.speed) < 1.5
    assert len(detect_efforts(tr)) == 0
