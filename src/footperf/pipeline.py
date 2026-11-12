"""One call per match: tracking -> per-player physical table + efforts, cached as CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import fetch_match, load_players, load_tracking
from .kinematics import add_speed
from .physical import detect_efforts, player_summary


def process_match(match_id: int, data_dir: Path, out_dir: Path, window: int = 9,
                  force: bool = False) -> pd.DataFrame:
    """Per-player physical totals for one match. Reuses cached results unless force=True."""
    out = out_dir / str(match_id)
    cached = out / "players_physical.csv"
    if cached.exists() and not force:
        return pd.read_csv(cached).assign(match_id=match_id)

    paths = fetch_match(match_id, data_dir)
    players = load_players(paths["match"])
    track = add_speed(load_tracking(paths["tracking"]), window=window)
    summary = players.merge(player_summary(track), on="player_id")
    efforts = detect_efforts(track).merge(players[["player_id", "name", "team"]], on="player_id")

    out.mkdir(parents=True, exist_ok=True)
    summary.round(3).to_csv(cached, index=False)
    efforts.round(2).to_csv(out / "efforts.csv", index=False)
    return summary.assign(match_id=match_id)


SUM_COLS = ["minutes_official", "minutes_tracked", "total_distance_m", "running_distance_m",
            "hsr_distance_m", "sprint_distance_m", "hsr_count", "sprint_count",
            "high_accel_count", "high_decel_count"]


def pool_players(by_match: pd.DataFrame) -> pd.DataFrame:
    """Season view: sum volumes and minutes over matches, median top speed per player."""
    g = by_match.groupby("player_id")
    pooled = g[SUM_COLS].sum()
    pooled["top_speed_kmh"] = g["top_speed_kmh"].median()
    pooled["n_matches"] = g.size()
    pooled["name"] = g["name"].first()
    pooled["team"] = g["team"].first()
    pooled["position"] = g["position"].first()
    return pooled.reset_index()
