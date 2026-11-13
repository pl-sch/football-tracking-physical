"""Bundesliga 2022/23: the IDSSE open dataset (DFL / Sportec Solutions, CC-BY 4.0).

Optical tracking (TRACAB, cameras in the stadium) at 25 Hz: every player is tracked all the
time, nothing is extrapolated. The provider also publishes its own speed per frame, which
gives a direct check of our speed computation.

Files are fetched with scripts/fetch_idsse.sh and the position XML is converted to compact
CSV with src/footperf/sportec_convert.py (both run on a machine that can reach Hugging Face).

Reference: Bassek et al. (2025), Scientific Data, https://doi.org/10.1038/s41597-025-04505-y
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 25.0

MATCHES = {  # match id -> competition id
    "J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001", "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002", "J03WPY": "DFL-COM-000002", "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}

# DFL position codes -> the five groups used for the A-League data
POSITIONS = {
    "TW": "Other",
    "IVL": "Central Defender", "IVR": "Central Defender", "IVZ": "Central Defender",
    "LV": "Full Back", "RV": "Full Back",
    "DML": "Midfield", "DMR": "Midfield", "DMZ": "Midfield", "DLM": "Midfield", "DRM": "Midfield",
    "ZO": "Midfield", "HL": "Midfield", "HR": "Midfield",
    "LM": "Wide Attacker", "RM": "Wide Attacker", "OLM": "Wide Attacker", "ORM": "Wide Attacker",
    "LA": "Wide Attacker", "RA": "Wide Attacker",
    "STL": "Center Forward", "STR": "Center Forward", "STZ": "Center Forward",
}


def player_int_id(dfl_id: str) -> int:
    """Stable integer id from a DFL person id, far above SkillCorner's id range."""
    return int(hashlib.md5(dfl_id.encode()).hexdigest()[:12], 16)


def paths(match_id: str, data_dir: Path | str = "data") -> dict[str, Path]:
    d = Path(data_dir) / "idsse"
    c = MATCHES[match_id]
    p = {
        "meta": d / f"DFL_02_01_matchinformation_{c}_DFL-MAT-{match_id}.xml",
        "events": d / f"DFL_03_02_events_raw_{c}_DFL-MAT-{match_id}.xml",
        "positions": d / f"{match_id}_positions.csv.gz",
        "ball": d / f"{match_id}_ball.csv.gz",
    }
    missing = [str(v) for v in p.values() if not v.exists()]
    if missing:
        raise FileNotFoundError("run scripts/fetch_idsse.sh then "
                                "python src/footperf/sportec_convert.py data/idsse. Missing: "
                                + ", ".join(missing))
    return p


def match_info(meta_path: Path) -> dict:
    g = ET.parse(meta_path).getroot().find(".//General").attrib
    return {"date": g["KickoffTime"][:10], "home": g["HomeTeamName"], "away": g["GuestTeamName"],
            "competition": g["CompetitionName"], "home_id": g["HomeTeamId"], "away_id": g["GuestTeamId"]}


def load_players(meta_path: Path) -> pd.DataFrame:
    root = ET.parse(meta_path).getroot()
    rows = []
    for team in root.iter("Team"):
        side = "home" if team.get("Role") == "home" else "away"
        for p in team.iter("Player"):
            rows.append({
                "player_id": player_int_id(p.get("PersonId")),
                "source_id": p.get("PersonId"),
                "name": p.get("Shortname"),
                "jersey": p.get("ShirtNumber"),
                "team": team.get("TeamName"),
                "side": side,
                "position": POSITIONS.get(p.get("PlayingPosition"), "Other"),
            })
    return pd.DataFrame(rows)


def load_tracking(positions_csv: Path, ball_csv: Path) -> pd.DataFrame:
    """Same schema as data.load_tracking, plus speed_provider (m/s) and alive (bool)."""
    pos = pd.read_csv(positions_csv, dtype={"half": np.int8, "frame": np.int32, "x": float,
                                            "y": float, "speed_kmh": float})
    ball = pd.read_csv(ball_csv)
    ball["possession"] = ball.possession.map({1: "home", 2: "away"})
    ball["alive"] = ball.alive == 1
    ball.loc[~ball.alive, "possession"] = None  # dead ball: nobody has it (as in SkillCorner)
    pos = pos.merge(ball[["half", "frame", "possession", "alive"]], on=["half", "frame"], how="left")
    ids = {s: player_int_id(s) for s in pos.person.unique()}
    return pd.DataFrame({
        "frame": pos.frame.to_numpy(),
        "period": pos.half.to_numpy(),
        "player_id": pos.person.map(ids).to_numpy(np.int64),
        "x": pos.x.to_numpy(),
        "y": pos.y.to_numpy(),
        "detected": True,
        "possession": pd.Categorical(pos.possession, categories=["home", "away"]),
        "alive": pos.alive.fillna(False).to_numpy(bool),
        "speed_provider": pos.speed_kmh.to_numpy() / 3.6,
    })


def attacking_sign(events_path: Path, meta_path: Path) -> dict[tuple[str, int], int]:
    """(side, period) -> +1 if that team attacks towards +x. From the kick-off events:
    TeamLeft is the team defending the left (-x) goal, so it attacks towards +x."""
    info = match_info(meta_path)
    side_of = {info["home_id"]: "home", info["away_id"]: "away"}
    out = {}
    for ko in ET.parse(events_path).getroot().iter("KickOff"):
        period = {"firstHalf": 1, "secondHalf": 2}.get(ko.get("GameSection"))
        if period and ko.get("TeamLeft") in side_of:
            left = side_of[ko.get("TeamLeft")]
            out[(left, period)] = 1
            out[("away" if left == "home" else "home", period)] = -1
    return out


def technical_summary(events_path: Path) -> pd.DataFrame:
    """Per-player counts from DFL event data. Unlike SkillCorner, DFL events include duels:
    TacklingGame records a winner and a loser, Type 'air' is an aerial duel (headers)."""
    root = ET.parse(events_path).getroot()
    c: dict[str, Counter] = {}

    def add(pid, key, v=1):
        if pid:
            c.setdefault(pid, Counter())[key] += v

    for play in root.iter("Play"):
        add(play.get("Player"), "passes")
        if play.get("Evaluation") in ("successfullyCompleted", "successful"):
            add(play.get("Player"), "passes_successful")
    for shot in root.iter("ShotAtGoal"):
        pid = shot.get("Player")
        add(pid, "shots")
        add(pid, "xg", float(shot.get("xG", 0) or 0))
        kinds = {ch.tag for ch in shot}
        if kinds & {"SavedShot", "SuccessfulShot"}:
            add(pid, "shots_on_target")
        if "SuccessfulShot" in kinds:
            add(pid, "goals")
    for duel in root.iter("TacklingGame"):
        aerial = duel.get("Type") == "air"
        for pid, won in ((duel.get("Winner"), 1), (duel.get("Loser"), 0)):
            add(pid, "duels")
            add(pid, "duels_won", won)
            if aerial:
                add(pid, "aerial_duels")
                add(pid, "aerial_duels_won", won)
        if duel.get("WinnerRole") == "withoutBallControl" and duel.get("Type") != "air":
            add(duel.get("Winner"), "tackles_won")
    for claim in root.iter("BallClaiming"):
        add(claim.get("Player"), "regains")
    for foul in root.iter("Foul"):
        add(foul.get("Fouler"), "fouls_committed")

    cols = ["passes", "passes_successful", "shots", "shots_on_target", "goals", "xg", "duels",
            "duels_won", "aerial_duels", "aerial_duels_won", "tackles_won", "regains",
            "fouls_committed"]
    df = pd.DataFrame.from_dict(c, orient="index").reindex(columns=cols).fillna(0)
    df[[k for k in cols if k != "xg"]] = df[[k for k in cols if k != "xg"]].astype(int)
    df.index = [player_int_id(i) for i in df.index]
    df.index.name = "player_id"
    return df.reset_index()


def provider_agreement(track: pd.DataFrame, fps: float = FPS) -> pd.DataFrame:
    """Per player: distances from our speed vs from the provider's speed, same thresholds."""
    t = track[["player_id", "speed", "speed_provider"]].dropna()
    rows = {}
    for pid, g in t.groupby("player_id"):
        ours, prov = g.speed.to_numpy(), g.speed_provider.to_numpy()
        r = {}
        for name, s in (("ours", ours), ("provider", prov)):
            kmh = s * 3.6
            r[f"total_{name}"] = s.sum() / fps
            r[f"hsr_{name}"] = s[(kmh >= 20) & (kmh < 25)].sum() / fps
            r[f"sprint_{name}"] = s[kmh >= 25].sum() / fps
        r["speed_rmse_kmh"] = float(np.sqrt(np.mean((ours - prov) ** 2)) * 3.6)
        rows[pid] = r
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "player_id"
    return out.reset_index()
