"""One call per match: tracking + events -> per-player table and efforts, cached as CSV.

Two sources share the same processing:
  A-League (SkillCorner, broadcast tracking, 10 Hz, dynamic events)   -> process_match
  Bundesliga (DFL / IDSSE, optical tracking, 25 Hz, DFL event data)   -> process_match_sportec
Results go to outputs/<match_id>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import sportec
from .data import FPS as SKILLCORNER_FPS
from .data import attacking_sign, fetch_match, load_players, load_tracking
from .events import fetch_dynamic_events, label_efforts, load_dynamic_events, technical_summary
from .kinematics import add_speed
from .physical import detect_efforts, player_summary, possession_split, speed_histogram

# Savitzky-Golay window in seconds, per source. Broadcast tracking (A-League) needs 0.9 s to
# match SkillCorner's acceleration counts. Optical tracking (Bundesliga) is cleaner: 0.44 s
# matches the DFL's own speed within 1%, 0.9 s would shave 4-5% off sprint distance
# (scripts/bundesliga_check.py --window-scan).
WINDOW_S = 0.9
WINDOW_S_OPTICAL = 0.44


def window_frames(fps: float, seconds: float = WINDOW_S) -> int:
    n = int(round(seconds * fps))
    return n if n % 2 else n + 1


def _cached(out: Path, match_id: str, force: bool) -> pd.DataFrame | None:
    f = out / "players.csv"
    if f.exists() and not force:
        return pd.read_csv(f).assign(match_id=str(match_id))
    return None


def _finish(track, players, tech, efforts, sign, fps, out, match_id, extra=None) -> pd.DataFrame:
    """Shared part: physical summary, possession split, effort phase, attacking coordinates."""
    side = players.set_index("player_id")["side"]
    summary = (players.merge(player_summary(track, fps=fps), on="player_id")
               .merge(possession_split(track, side, fps=fps), on="player_id", how="left")
               .merge(tech, on="player_id", how="left"))
    if extra is not None:
        summary = summary.merge(extra, on="player_id", how="left")
    tech_cols = [c for c in tech.columns if c != "player_id"]
    summary[tech_cols] = summary[tech_cols].fillna(0)

    efforts = efforts.merge(players[["player_id", "name", "team", "position", "side"]], on="player_id")
    key = track.set_index(["player_id", "frame"])["possession"].astype(object)
    key = key[~key.index.duplicated()]
    at_start = key.reindex(pd.MultiIndex.from_arrays([efforts.player_id, efforts.start_frame])).to_numpy()
    efforts["team_phase"] = np.where(pd.isna(at_start), "no possession",
                                     np.where(at_start == efforts.side, "in possession",
                                              "out of possession"))
    s = np.array([sign.get((sd, int(p)), 1) for sd, p in zip(efforts.side, efforts.period)])
    for c in ("x_start", "y_start", "x_end", "y_end"):
        efforts[c + "_att"] = efforts[c] * s

    out.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(out / "players.csv", index=False)
    efforts.round(2).to_csv(out / "efforts.csv", index=False)
    speed_histogram(track, fps=fps).round(2).to_csv(out / "speed_histogram.csv", index=False)
    return summary.assign(match_id=str(match_id))


def process_match(match_id: int, data_dir: Path, out_dir: Path, window: int | None = None,
                  force: bool = False) -> pd.DataFrame:
    """A-League match (SkillCorner). Cached unless force=True."""
    out = out_dir / str(match_id)
    if (hit := _cached(out, match_id, force)) is not None:
        return hit
    fps = SKILLCORNER_FPS
    paths = fetch_match(match_id, data_dir)
    players = load_players(paths["match"])
    de = load_dynamic_events(fetch_dynamic_events(match_id, data_dir))
    track = add_speed(load_tracking(paths["tracking"]), window=window or window_frames(fps), fps=fps)
    tech = technical_summary(de)
    efforts = label_efforts(detect_efforts(track, fps=fps), de)
    return _finish(track, players, tech, efforts, attacking_sign(paths["match"]), fps, out, match_id)


def process_match_sportec(match_id: str, data_dir: Path, out_dir: Path, force: bool = False) -> pd.DataFrame:
    """Bundesliga match (DFL / IDSSE). Cached unless force=True."""
    out = out_dir / match_id
    if (hit := _cached(out, match_id, force)) is not None:
        return hit
    fps = sportec.FPS
    p = sportec.paths(match_id, data_dir)
    players = sportec.load_players(p["meta"])
    track = add_speed(sportec.load_tracking(p["positions"], p["ball"]),
                      window=window_frames(fps, WINDOW_S_OPTICAL), fps=fps)
    players = players[players.player_id.isin(track.player_id.unique())]
    players = players.assign(minutes_official=players.player_id.map(
        track.groupby("player_id").size() / fps / 60))  # optical tracking: on pitch = tracked
    tech = sportec.technical_summary(p["events"])
    efforts = detect_efforts(track, fps=fps).assign(context="not available (no tactical events)",
                                                     event_subtype=None, run_targeted=False)
    agree = sportec.provider_agreement(track, fps)
    sign = sportec.attacking_sign(p["events"], p["meta"])
    return _finish(track, players, tech, efforts, sign, fps, out, match_id, extra=agree)


SUM_COLS = ["minutes_official", "minutes_tracked", "total_distance_m", "running_distance_m",
            "hsr_distance_m", "sprint_distance_m", "hsr_count", "sprint_count",
            "hsr_effort_count", "sprint_effort_count", "high_accel_count", "high_decel_count",
            "possessions", "passes", "passes_successful", "shots", "possession_losses",
            "possessions_leading_to_shot", "off_ball_runs", "runs_targeted", "runs_received",
            "runs_leading_to_shot", "defensive_engagements", "regains", "disruptions",
            "fouls_committed", "distance_tip_m", "distance_otip_m", "hi_distance_tip_m",
            "hi_distance_otip_m", "minutes_tip", "minutes_otip",
            "shots_on_target", "goals", "xg", "duels", "duels_won", "aerial_duels",
            "aerial_duels_won", "tackles_won"]


def pool_players(by_match: pd.DataFrame) -> pd.DataFrame:
    """Season view: sum volumes and minutes over matches, median top speed per player.
    Stats a league does not record stay empty (NaN), not zero."""
    g = by_match.groupby("player_id")
    cols = [c for c in SUM_COLS if c in by_match]
    pooled = g[cols].sum(min_count=1)
    pooled["top_speed_kmh"] = g["top_speed_kmh"].median()
    pooled["n_matches"] = g.size()
    for c in ("name", "team", "league"):
        if c in by_match:
            pooled[c] = g[c].first()
    pooled["position"] = g["position"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    return pooled.reset_index()
