"""Physical metrics per player: distance by speed zone, high-intensity efforts, accelerations.

Thresholds follow SkillCorner's published definitions so the output can be checked
against their aggregates:
  running 15-20 km/h, HSR 20-25 km/h, sprint > 25 km/h,
  an effort counts if sustained for at least 0.7 s,
  high acceleration / deceleration beyond +/- 3 m/s^2 for at least 0.7 s.

Effort classification is our own convention (SkillCorner does not publish it):
an effort is a continuous stretch above 20 km/h. It is a sprint if it contains
at least 0.7 s above 25 km/h, otherwise an HSR. One run = one effort, so a player
accelerating through the HSR band into a sprint is not counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FPS


@dataclass(frozen=True)
class Thresholds:
    running_kmh: float = 15.0
    hsr_kmh: float = 20.0
    sprint_kmh: float = 25.0
    accel_ms2: float = 3.0
    min_duration_s: float = 0.7


def _runs(mask: np.ndarray, frames: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) index pairs where mask is True on consecutive frames."""
    if len(mask) == 0:
        return []
    cont = np.ones(len(mask), dtype=bool)
    cont[1:] = np.diff(frames) == 1
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is not None and not cont[i]:
            runs.append((start, i))
            start = i
        elif m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def detect_efforts(df: pd.DataFrame, th: Thresholds = Thresholds(), fps: float = FPS) -> pd.DataFrame:
    """One row per high-intensity effort. `df` needs player_id, period, frame, x, y, speed."""
    min_frames = int(round(th.min_duration_s * fps))
    rows = []
    for (pid, period), g in df.groupby(["player_id", "period"], sort=False):
        g = g.sort_values("frame")
        frames = g["frame"].to_numpy()
        kmh = np.nan_to_num(g["speed"].to_numpy() * 3.6, nan=0.0)
        x, y = g["x"].to_numpy(), g["y"].to_numpy()
        det = g["detected"].to_numpy()
        for a, b in _runs(kmh >= th.hsr_kmh, frames):
            if b - a < min_frames:
                continue
            seg = kmh[a:b]
            is_sprint = _longest_true_run(seg >= th.sprint_kmh) >= min_frames
            # Top speed: held for at least min_duration AND on frames where the player is
            # actually visible. Extrapolated positions are smooth but can drift to
            # impossible speeds (44 km/h observed), so they are excluded here.
            seen = np.where(det[a:b], seg, np.nan)
            sustained = pd.Series(seen).rolling(min_frames).min().max()
            rows.append({
                "player_id": pid,
                "period": period,
                "start_frame": int(frames[a]),
                "end_frame": int(frames[b - 1]),
                "duration_s": (b - a) / fps,
                "distance_m": float(seg.sum() / 3.6 / fps),
                "peak_kmh": float(seg.max()),
                "peak_sustained_kmh": float(sustained),
                "detected_share": float(det[a:b].mean()),
                "x_start": x[a], "y_start": y[a], "x_end": x[b - 1], "y_end": y[b - 1],
                "kind": "sprint" if is_sprint else "hsr",
            })
    cols = ["player_id", "period", "start_frame", "end_frame", "duration_s", "distance_m",
            "peak_kmh", "peak_sustained_kmh", "detected_share", "x_start", "y_start", "x_end", "y_end", "kind"]
    return pd.DataFrame(rows, columns=cols)


def _count_accel_events(g: pd.DataFrame, sign: int, th: Thresholds, fps: float) -> int:
    min_frames = int(round(th.min_duration_s * fps))
    acc = np.nan_to_num(g["accel"].to_numpy(), nan=0.0) * sign
    return sum(1 for a, b in _runs(acc >= th.accel_ms2, g["frame"].to_numpy()) if b - a >= min_frames)


def player_summary(df: pd.DataFrame, th: Thresholds = Thresholds(), fps: float = FPS) -> pd.DataFrame:
    """One row per player with match totals. `df` is the output of kinematics.add_speed."""
    dt = 1.0 / fps
    efforts = detect_efforts(df, th, fps)
    counts = efforts.pivot_table(index="player_id", columns="kind", values="start_frame",
                                 aggfunc="count", fill_value=0)

    rows = []
    for pid, g in df.groupby("player_id", sort=False):
        g = g.sort_values(["period", "frame"])
        kmh = g["speed"].to_numpy() * 3.6
        ok = ~np.isnan(kmh)
        k = kmh[ok]
        rows.append({
            "player_id": pid,
            "minutes_tracked": len(g) / fps / 60,
            "detected_share": g["detected"].mean(),
            "total_distance_m": k.sum() / 3.6 * dt,
            "running_distance_m": k[(k >= th.running_kmh) & (k < th.hsr_kmh)].sum() / 3.6 * dt,
            "hsr_distance_m": k[(k >= th.hsr_kmh) & (k < th.sprint_kmh)].sum() / 3.6 * dt,
            "sprint_distance_m": k[k >= th.sprint_kmh].sum() / 3.6 * dt,
            "max_speed_kmh": k.max() if len(k) else np.nan,
            "high_accel_count": sum(_count_accel_events(gp, +1, th, fps) for _, gp in g.groupby("period")),
            "high_decel_count": sum(_count_accel_events(gp, -1, th, fps) for _, gp in g.groupby("period")),
        })
    out = pd.DataFrame(rows).set_index("player_id")
    for kind in ("hsr", "sprint"):
        out[f"{kind}_count"] = counts[kind] if kind in counts else 0
    out[["hsr_count", "sprint_count"]] = out[["hsr_count", "sprint_count"]].fillna(0).astype(int)
    out["top_speed_kmh"] = efforts.groupby("player_id")["peak_sustained_kmh"].max()
    return out.reset_index()


def add_rates(summary: pd.DataFrame, minutes_col: str) -> pd.DataFrame:
    """Per-minute distance using the chosen denominator.

    Broadcast tracking loses the players during replays and close-ups, which mostly
    happen in stoppages. Dividing by tracked minutes therefore over-weights live play
    and inflates intensity. Dividing by minutes played (match sheet) is the fair rate,
    at the cost of missing the (low-speed) distance covered off-camera.
    """
    out = summary.copy()
    out["minutes"] = out[minutes_col]
    out["meters_per_minute"] = out["total_distance_m"] / out["minutes"]
    return out
