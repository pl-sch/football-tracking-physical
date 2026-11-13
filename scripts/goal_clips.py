"""Short tracking clips around every Bundesliga goal, for the replay in the app.

For each goal in the DFL events: estimate the frame from the event time, then refine it to
the frame where the ball actually crosses the goal line in the tracking data. Keep 10 s
before and 2 s after, at 10 frames per second, with the scoring team attacking left to right.

    python scripts/goal_clips.py   # needs data/idsse with ball coordinates (sportec_convert.py)
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footperf import sportec  # noqa: E402

OUT = ROOT / "outputs" / "all_matches"
BEFORE_S, AFTER_S, CLIP_FPS = 10.0, 2.0, 10


def goal_events(events_path: Path) -> list[dict]:
    root = ET.parse(events_path).getroot()
    kickoff = {}
    goals = []
    for ev in root.iter("Event"):
        t = datetime.fromisoformat(ev.get("EventTime"))
        for ch in ev.iter():  # shots can sit inside a Penalty or FreeKick element
            if ch.tag == "KickOff" and ch.get("GameSection") in ("firstHalf", "secondHalf"):
                kickoff[1 if ch.get("GameSection") == "firstHalf" else 2] = t
            if ch.tag == "ShotAtGoal" and ch.find("SuccessfulShot") is not None:
                goals.append({"time": t, "scorer": ch.get("Player"), "team": ch.get("Team"),
                              "xg": float(ch.get("xG") or 0), "body": ch.get("TypeOfShot")})
    for g in goals:
        half = 2 if 2 in kickoff and g["time"] >= kickoff[2] else 1
        g["half"] = half
        g["seconds"] = (g["time"] - kickoff[half]).total_seconds()
        g["minute"] = int(g["seconds"] // 60) + (1 if half == 1 else 46)
    return goals


def crossing_frame(ball: pd.DataFrame, half: int, guess: int) -> tuple[int, int]:
    """First frame near `guess` where the ball is inside a goal. Returns (frame, side +1/-1)."""
    b = ball[(ball.half == half) & ball.frame.between(guess - 25 * 12, guess + 25 * 12)]
    inside = b[(b.x.abs() > 52.4) & (b.y.abs() < 4.0) & (b.z < 3.0)]
    if inside.empty:  # ball tracking misses the line: take the closest approach to a goal mouth
        d = np.hypot(b.x.abs() - 52.5, b.y)
        j = d.idxmin()
        return int(b.loc[j, "frame"]), int(np.sign(b.loc[j, "x"]))
    i = (inside.frame - guess).abs().idxmin()
    # earliest frame of that crossing
    run = inside[inside.frame <= inside.loc[i, "frame"]].frame.to_numpy()
    start = run[-1]
    for f in run[::-1]:
        if f < start - 1:
            break
        start = f
    return int(start), int(np.sign(inside.loc[i, "x"]))


def main() -> None:
    clips, meta = [], []
    for mid in sportec.MATCHES:
        p = sportec.paths(mid, ROOT / "data")
        info = sportec.match_info(p["meta"])
        players = sportec.load_players(p["meta"]).set_index("source_id")
        pos = pd.read_csv(p["positions"])
        ball = pd.read_csv(p["ball"])
        start = ball.groupby("half").frame.min()
        teams = {info["home_id"]: info["home"], info["away_id"]: info["away"]}
        score = {info["home_id"]: 0, info["away_id"]: 0}
        for k, g in enumerate(sorted(goal_events(p["events"]), key=lambda g: g["time"])):
            guess = int(start[g["half"]] + g["seconds"] * sportec.FPS)
            frame, side = crossing_frame(ball, g["half"], guess)
            score[g["team"]] = score.get(g["team"], 0) + 1
            gid = f"{mid}-{k + 1}"
            scorer = players.loc[g["scorer"], "name"] if g["scorer"] in players.index else "?"
            meta.append({
                "goal_id": gid, "match_id": mid, "date": info["date"],
                "match": f"{info['home']} - {info['away']}", "minute": g["minute"],
                "scorer": scorer, "team": teams.get(g["team"], "?"),
                "score": f"{score[info['home_id']]}-{score[info['away_id']]}",
                "xg": round(g["xg"], 2), "body": g["body"],
                "line_crossing_found": bool(((ball.half == g["half"]) & (ball.frame == frame)
                                             & (ball.x.abs() > 52.4)).any()),
            })
            f0, f1 = frame - int(BEFORE_S * sportec.FPS), frame + int(AFTER_S * sportec.FPS)
            step = int(sportec.FPS // CLIP_FPS)  # 25 Hz -> every 2nd frame = 12.5 Hz, close enough
            keep = np.arange(f0, f1 + 1, step)
            pp = pos[(pos.half == g["half"]) & pos.frame.isin(keep)].copy()
            pp["kind"] = np.where(pp.team == g["team"], "scoring team", "defending team")
            pp["label"] = pp.person.map(players["jersey"]).fillna("").astype(str)
            pp["name"] = pp.person.map(players["name"])
            bb = ball[(ball.half == g["half"]) & ball.frame.isin(keep)].assign(
                kind="ball", label="", name="ball")
            part = pd.concat([pp[["frame", "x", "y", "kind", "label", "name"]],
                              bb[["frame", "x", "y", "kind", "label", "name"]]])
            part["t"] = ((part.frame - frame) / sportec.FPS).round(2)
            part["x"] = (part.x * side).round(2)  # scoring team attacks towards +x
            part["y"] = (part.y * side).round(2)
            clips.append(part.drop(columns="frame").assign(goal_id=gid))
        print(mid, info["home"], "-", info["away"], "goals:", sum(m["match_id"] == mid for m in meta))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(meta).to_csv(OUT / "goals.csv", index=False)
    pd.concat(clips).to_csv(OUT / "goal_clips.csv.gz", index=False)
    print(pd.DataFrame(meta)[["goal_id", "minute", "scorer", "team", "score", "xg",
                              "line_crossing_found"]].to_string(index=False))


if __name__ == "__main__":
    main()
