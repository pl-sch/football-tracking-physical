"""SkillCorner dynamic events: technical stats per player, and the football context of each sprint.

Dynamic events come with the open data and use the same frame numbering as the tracking:
  player_possession   every touch sequence, ending in a pass, shot, loss, clearance...
  off_ball_run        runs without the ball (in behind, overlap, coming short...)
  on_ball_engagement  defensive actions on the ball carrier (pressing, pressure...)
  passing_option      teammates available for a pass (not used here)
There are no tackles or headers in this data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import RAW, _download

# Off-ball run subtypes grouped by intent
RUN_GROUPS = {
    "behind": "run in behind",
    "run_ahead_of_the_ball": "run in behind",
    "cross_receiver": "attacking the box",
    "overlap": "width / overlap",
    "underlap": "width / overlap",
    "pulling_wide": "width / overlap",
    "pulling_half_space": "width / overlap",
    "support": "support / coming short",
    "coming_short": "support / coming short",
    "dropping_off": "support / coming short",
}

CONTEXT_ORDER = ["with the ball", "run in behind", "attacking the box", "width / overlap",
                 "support / coming short", "pressing", "no linked event"]


def fetch_dynamic_events(match_id: int, data_dir: Path | str = "data") -> Path:
    return _download(f"{RAW}/matches/{match_id}/{match_id}_dynamic_events.csv",
                     Path(data_dir) / str(match_id) / "dynamic_events.csv")


def load_dynamic_events(path: Path | str) -> pd.DataFrame:
    keep = ["event_type", "event_subtype", "player_id", "frame_start", "frame_end", "period",
            "end_type", "pass_outcome", "lead_to_shot", "lead_to_goal", "targeted", "received"]
    de = pd.read_csv(path, low_memory=False, usecols=lambda c: c in keep)
    de = de.dropna(subset=["player_id"])
    de["player_id"] = de["player_id"].astype(np.int64)
    return de


def technical_summary(de: pd.DataFrame) -> pd.DataFrame:
    """Per-player counts of on-ball, off-ball and defensive actions."""
    pp = de[de.event_type == "player_possession"]
    ob = de[de.event_type == "off_ball_run"]
    en = de[de.event_type == "on_ball_engagement"]

    def count(df, mask=None):
        df = df if mask is None else df[mask]
        return df.groupby("player_id").size()

    out = pd.DataFrame({
        "possessions": count(pp),
        "passes": count(pp, pp.end_type == "pass"),
        "passes_successful": count(pp, (pp.end_type == "pass") & (pp.pass_outcome == "successful")),
        "shots": count(pp, pp.end_type == "shot"),
        "possession_losses": count(pp, pp.end_type == "possession_loss"),
        "possessions_leading_to_shot": count(pp, pp.lead_to_shot == True),  # noqa: E712
        "off_ball_runs": count(ob),
        "runs_targeted": count(ob, ob.targeted == True),  # noqa: E712
        "runs_received": count(ob, ob.received == True),  # noqa: E712
        "runs_leading_to_shot": count(ob, ob.lead_to_shot == True),  # noqa: E712
        "defensive_engagements": count(en),
        "regains": count(en, en.end_type.isin(["direct_regain", "indirect_regain"])),
        "disruptions": count(en, en.end_type.isin(["direct_disruption", "indirect_disruption"])),
        "fouls_committed": count(en, en.end_type == "foul_committed"),
    }).fillna(0).astype(int)
    out.index.name = "player_id"
    return out.reset_index()


def label_efforts(efforts: pd.DataFrame, de: pd.DataFrame) -> pd.DataFrame:
    """Attach a football context to each high-intensity effort.

    An effort takes the label of the dynamic event of the same player that overlaps it in
    time, with priority: possession > off-ball run > defensive engagement. Efforts with no
    overlapping event are mostly out of possession (58% across 20 matches): recovery runs, tracking back.
    """
    priority = {"player_possession": 0, "off_ball_run": 1, "on_ball_engagement": 2}
    ev = de[de.event_type.isin(priority)].copy()
    ev["prio"] = ev.event_type.map(priority)

    context = np.full(len(efforts), CONTEXT_ORDER[-1], dtype=object)
    subtype = np.full(len(efforts), None, dtype=object)
    targeted = np.zeros(len(efforts), dtype=bool)
    for pid, idx in efforts.groupby("player_id").indices.items():
        e = ev[ev.player_id == pid]
        if e.empty:
            continue
        s0 = efforts["start_frame"].to_numpy()[idx][:, None]
        s1 = efforts["end_frame"].to_numpy()[idx][:, None]
        overlap = (e.frame_start.to_numpy()[None, :] <= s1) & (e.frame_end.to_numpy()[None, :] >= s0)
        prio = np.where(overlap, e.prio.to_numpy()[None, :], 99)
        best = prio.argmin(axis=1)
        for k, (i, j) in enumerate(zip(idx, best)):
            if prio[k, j] == 99:
                continue
            row = e.iloc[j]
            if row.event_type == "player_possession":
                context[i] = "with the ball"
            elif row.event_type == "off_ball_run":
                context[i] = RUN_GROUPS.get(row.event_subtype, "support / coming short")
                subtype[i] = row.event_subtype
                targeted[i] = row.targeted == True  # noqa: E712
            else:
                context[i] = "pressing"
                subtype[i] = row.event_subtype
    out = efforts.copy()
    out["context"] = context
    out["event_subtype"] = subtype
    out["run_targeted"] = targeted
    return out
