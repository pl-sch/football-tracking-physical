"""Download and load SkillCorner open data (A-League 2024/25, broadcast tracking at 10 Hz)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/master/data"
# Tracking files are stored with Git LFS, so they are served from the media host.
MEDIA = "https://media.githubusercontent.com/media/SkillCorner/opendata/master/data"

FPS = 10.0


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def fetch_match(match_id: int, data_dir: Path | str = "data") -> dict[str, Path]:
    """Download tracking + metadata for one match. Returns local paths."""
    d = Path(data_dir) / str(match_id)
    base = f"matches/{match_id}/{match_id}"
    return {
        "tracking": _download(f"{MEDIA}/{base}_tracking_extrapolated.jsonl", d / "tracking.jsonl"),
        "match": _download(f"{RAW}/{base}_match.json", d / "match.json"),
    }


def list_matches(data_dir: Path | str = "data") -> pd.DataFrame:
    """All matches available in the open data repo."""
    path = _download(f"{RAW}/matches.json", Path(data_dir) / "matches.json")
    return pd.DataFrame([{
        "match_id": m["id"],
        "date": m["date_time"][:10],
        "home": m["home_team"]["short_name"],
        "away": m["away_team"]["short_name"],
    } for m in json.loads(path.read_text())]).sort_values("date").reset_index(drop=True)


def fetch_physical_aggregates(data_dir: Path | str = "data") -> Path:
    name = "aus1league_physicalaggregates_20242025.csv"
    return _download(f"{RAW}/aggregates/{name}", Path(data_dir) / name)


def load_players(match_path: Path | str) -> pd.DataFrame:
    """One row per player listed in the match sheet."""
    meta = json.loads(Path(match_path).read_text())
    home_id = meta["home_team"]["id"]
    teams = {home_id: meta["home_team"]["short_name"],
             meta["away_team"]["id"]: meta["away_team"]["short_name"]}
    rows = []
    for p in meta["players"]:
        rows.append({
            "player_id": p["id"],
            "name": p["short_name"],
            "team": teams.get(p["team_id"]),
            "side": "home" if p["team_id"] == home_id else "away",
            "position": p["player_role"]["position_group"] if p["player_role"] else None,
            "minutes_official": (p["playing_time"]["total"] or {}).get("minutes_played")
            if p.get("playing_time") else None,
        })
    return pd.DataFrame(rows)


def attacking_sign(match_path: Path | str) -> dict[tuple[str, int], int]:
    """(side, period) -> +1 if that team attacks towards +x, else -1."""
    meta = json.loads(Path(match_path).read_text())
    out = {}
    for period, home_dir in enumerate(meta["home_team_side"], start=1):
        home = 1 if home_dir == "left_to_right" else -1
        out[("home", period)] = home
        out[("away", period)] = -home
    return out


def load_tracking(tracking_path: Path | str) -> pd.DataFrame:
    """Long format: one row per (frame, player).

    Columns: frame, period, player_id, x, y, detected, possession ("home", "away" or NaN
    when SkillCorner assigns the ball to no team, e.g. dead ball or contested).

    Coordinates are in metres, origin at the centre spot.
    Frames before kick-off and between periods (period is null) are dropped.
    """
    frames, periods, pids, xs, ys, det, poss = [], [], [], [], [], [], []
    with open(tracking_path) as f:
        for line in f:
            fr = json.loads(line)
            if fr["period"] is None:
                continue
            group = {"home team": "home", "away team": "away"}.get(fr["possession"]["group"])
            for p in fr["player_data"]:
                frames.append(fr["frame"])
                periods.append(fr["period"])
                pids.append(p["player_id"])
                xs.append(p["x"])
                ys.append(p["y"])
                det.append(bool(p["is_detected"]))
                poss.append(group)
    return pd.DataFrame({
        "frame": np.asarray(frames, dtype=np.int32),
        "period": np.asarray(periods, dtype=np.int8),
        "player_id": np.asarray(pids, dtype=np.int64),
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "detected": np.asarray(det, dtype=bool),
        "possession": pd.Categorical(poss, categories=["home", "away"]),
    })
