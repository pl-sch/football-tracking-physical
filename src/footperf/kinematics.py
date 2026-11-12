"""Positions -> speed and acceleration.

Raw broadcast tracking is noisy: differentiating positions frame by frame amplifies
that noise (a 10 cm jitter at 10 Hz is already 1 m/s, i.e. 3.6 km/h of fake speed).
We fit a local polynomial (Savitzky-Golay) and take its analytic derivative instead.
Derivatives are only computed on contiguous runs of frames, never across a gap.

SkillCorner positions are already smoothed upstream, so on this dataset the window
barely moves distances. It does move acceleration counts a lot (second derivative):
see scripts/sensitivity.py. The default (9 frames) is the one that matches
SkillCorner's own acceleration counts on match 1886347.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .data import FPS

MAX_HUMAN_SPEED = 12.5  # m/s (45 km/h). Anything above is a tracking artefact.


def _contiguous_segments(frames: np.ndarray) -> list[slice]:
    """Split a sorted frame index into runs of consecutive frames."""
    if len(frames) == 0:
        return []
    breaks = np.flatnonzero(np.diff(frames) != 1) + 1
    bounds = np.concatenate(([0], breaks, [len(frames)]))
    return [slice(a, b) for a, b in zip(bounds[:-1], bounds[1:])]


def _derive_segment(x: np.ndarray, y: np.ndarray, window: int, poly: int, dt: float):
    n = len(x)
    if n < window:
        nan = np.full(n, np.nan)
        return nan, nan.copy()
    vx = savgol_filter(x, window, poly, deriv=1, delta=dt)
    vy = savgol_filter(y, window, poly, deriv=1, delta=dt)
    speed = np.hypot(vx, vy)
    accel = savgol_filter(speed, window, poly, deriv=1, delta=dt)
    return speed, accel


def add_speed(track: pd.DataFrame, window: int = 9, poly: int = 2, fps: float = FPS) -> pd.DataFrame:
    """Add `speed` (m/s) and `accel` (m/s^2) columns.

    window: Savitzky-Golay window in frames (odd). 9 frames = 0.9 s at 10 Hz.
    """
    if window % 2 == 0:
        raise ValueError("window must be odd")
    dt = 1.0 / fps
    out = track.sort_values(["player_id", "period", "frame"]).reset_index(drop=True)
    speed = np.full(len(out), np.nan)
    accel = np.full(len(out), np.nan)

    for _, idx in out.groupby(["player_id", "period"], sort=False).indices.items():
        frames = out["frame"].to_numpy()[idx]
        x = out["x"].to_numpy()[idx]
        y = out["y"].to_numpy()[idx]
        for seg in _contiguous_segments(frames):
            s, a = _derive_segment(x[seg], y[seg], window, poly, dt)
            speed[idx[seg]] = s
            accel[idx[seg]] = a

    bad = speed > MAX_HUMAN_SPEED
    speed[bad] = np.nan
    accel[bad] = np.nan
    out["speed"] = speed
    out["accel"] = accel
    return out
