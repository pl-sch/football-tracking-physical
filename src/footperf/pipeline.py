"""One call per match: tracking + dynamic events -> per-player table and labelled efforts.

Results are cached as CSV in outputs/<match_id>/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import attacking_sign, fetch_match, load_players, load_tracking
from .events import fetch_dynamic_events, label_efforts, load_dynamic_events, technical_summary
from .kinematics import add_speed
from .physical import detect_efforts, player_summary, possession_split, speed_histogram


def process_match(match_id: int, data_dir: Path, out_dir: Path, window: int = 9,
                  force: bool = False) -> pd.DataFrame:
    """Per-player physical + technical totals for one match. Cached unless force=True."""
    out = out_dir / str(match_id)
    cached = out / "players.csv"
    if cached.exists() and not force:
        return pd.read_csv(cached).assign(match_id=match_id)

    paths = fetch_match(match_id, data_dir)
    players = load_players(paths["match"])
    de = load_dynamic_events(fetch_dynamic_events(match_id, data_dir))
    track = add_speed(load_tracking(paths["tracking"]), window=window)

    side = players.set_index("player_id")["side"]
    summary = (players.merge(player_summary(track), on="player_id")
               .merge(possession_split(track, side), on="player_id", how="left")
               .merge(technical_summary(de), on="player_id", how="left"))
    tech_cols = [c for c in technical_summary(de.head(0)).columns if c != "player_id"]
    summary[tech_cols] = summary[tech_cols].fillna(0).astype(int)

    efforts = label_efforts(detect_efforts(track), de)
    efforts = efforts.merge(players[["player_id", "name", "team", "position", "side"]], on="player_id")
    # Ball possession when the effort starts
    key = track.set_index(["player_id", "frame"])["possession"].astype(object)
    at_start = key.reindex(pd.MultiIndex.from_arrays([efforts.player_id, efforts.start_frame])).to_numpy()
    efforts["team_phase"] = np.where(pd.isna(at_start), "no possession",
                                     np.where(at_start == efforts.side, "in possession",
                                              "out of possession"))
    # Coordinates with the player's team always attacking towards +x
    sign = attacking_sign(paths["match"])
    s = np.array([sign[(sd, int(p))] for sd, p in zip(efforts.side, efforts.period)])
    for c in ("x_start", "y_start", "x_end", "y_end"):
        efforts[c + "_att"] = efforts[c] * s

    out.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(cached, index=False)
    efforts.round(2).to_csv(out / "efforts.csv", index=False)
    speed_histogram(track).to_csv(out / "speed_histogram.csv", index=False)
    return summary.assign(match_id=match_id)


SUM_COLS = ["minutes_official", "minutes_tracked", "total_distance_m", "running_distance_m",
            "hsr_distance_m", "sprint_distance_m", "hsr_count", "sprint_count",
            "hsr_effort_count", "sprint_effort_count", "high_accel_count", "high_decel_count",
            "possessions", "passes", "passes_successful", "shots", "possession_losses",
            "possessions_leading_to_shot", "off_ball_runs", "runs_targeted", "runs_received",
            "runs_leading_to_shot", "defensive_engagements", "regains", "disruptions",
            "fouls_committed", "distance_tip_m", "distance_otip_m", "hi_distance_tip_m",
            "hi_distance_otip_m", "minutes_tip", "minutes_otip"]


def pool_players(by_match: pd.DataFrame) -> pd.DataFrame:
    """Season view: sum volumes and minutes over matches, median top speed per player."""
    g = by_match.groupby("player_id")
    pooled = g[[c for c in SUM_COLS if c in by_match]].sum()
    pooled["top_speed_kmh"] = g["top_speed_kmh"].median()
    pooled["n_matches"] = g.size()
    pooled["name"] = g["name"].first()
    pooled["team"] = g["team"].first()
    pooled["position"] = g["position"].agg(lambda s: s.mode().iat[0] if s.notna().any() else None)
    return pooled.reset_index()
