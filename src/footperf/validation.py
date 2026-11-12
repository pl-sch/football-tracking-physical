"""Compare our single-match metrics with SkillCorner's season aggregates.

This is not an exact check: SkillCorner publishes per-player season averages, not
match-level values, and computes them on its own internal tracking. A single match
deviates from a season average for real reasons (opponent, game state, tactics).
What we expect if the pipeline is sound:
  - a ratio ours/theirs centred near 1 (no systematic bias),
  - a strong correlation across players (we rank players the way they do).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (our column, their column, normalisation)
# "per90": our value / our minutes * 90  vs  their value / minutes_full_all * 90
# "raw": compared as is
PAIRS = [
    ("meters_per_minute", "total_metersperminute_full_all", "raw"),
    ("running_distance_m", "running_distance_full_all", "per90"),
    ("hsr_distance_m", "hsr_distance_full_all", "per90"),
    ("sprint_distance_m", "sprint_distance_full_all", "per90"),
    ("hsr_count", "hsr_count_full_all", "per90"),
    ("sprint_count", "sprint_count_full_all", "per90"),
    ("hsr_effort_count", "hsr_count_full_all", "per90"),
    ("sprint_effort_count", "sprint_count_full_all", "per90"),
    ("high_accel_count", "highaccel_count_full_all", "per90"),
    ("high_decel_count", "highdecel_count_full_all", "per90"),
    ("meters_per_minute_tip", "total_metersperminute_full_tip", "raw"),
    ("meters_per_minute_otip", "total_metersperminute_full_otip", "raw"),
    ("top_speed_kmh", "psv99", "raw"),
]


def collapse_aggregates(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per player.

    The file splits some players over several rows (e.g. by position group). Every metric
    is a per-match average, so rows are merged with a mean weighted by match count.
    """
    if agg["player_id"].is_unique:
        return agg
    cols = list(dict.fromkeys([t for _, t, _ in PAIRS] + ["minutes_full_all"]))
    w = agg["count_match"].clip(lower=1)
    weighted = agg[cols].mul(w, axis=0).assign(player_id=agg["player_id"], _w=w)
    g = weighted.groupby("player_id")
    out = g[cols].sum().div(g["_w"].sum(), axis=0)
    out["count_match"] = agg.groupby("player_id")["count_match"].sum()
    return out.reset_index()


def compare(ours: pd.DataFrame, agg: pd.DataFrame, min_minutes: float = 45.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-player comparison table, per-metric summary)."""
    m = ours.merge(collapse_aggregates(agg), on="player_id", how="inner", suffixes=("", "_sc"))
    m = m[(m["minutes"] >= min_minutes) & (m["minutes_full_all"] > 0)].copy()

    rows = []
    for o, t, norm in PAIRS:
        if norm == "per90":
            a = m[o] / m["minutes"] * 90
            b = m[t] / m["minutes_full_all"] * 90
        else:
            a, b = m[o], m[t]
        m[f"{o}__ours"] = a
        m[f"{o}__sc"] = b
        valid = (b > 0) & a.notna()
        ratio = (a[valid] / b[valid])
        rows.append({
            "metric": o,
            "n_players": int(valid.sum()),
            "median_ratio": ratio.median(),
            "pearson_r": np.corrcoef(a[valid], b[valid])[0, 1] if valid.sum() > 2 else np.nan,
            "spearman_r": a[valid].rank().corr(b[valid].rank()) if valid.sum() > 2 else np.nan,
        })
    return m, pd.DataFrame(rows)
