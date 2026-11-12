"""Fixed sprint threshold (25 km/h) vs a threshold relative to each player's top speed.

25 km/h is 81% of the top speed of a 31 km/h winger, but 89% for a 28 km/h centre-back.
A relative threshold measures how often a player works close to his own maximum.
Uses the speed histograms written by run_all.py, so no tracking is reprocessed.

    python scripts/relative_thresholds.py --fraction 0.85
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from footperf.data import fetch_physical_aggregates  # noqa: E402
from footperf.validation import collapse_aggregates  # noqa: E402

OUT = ROOT / "outputs"
BIN = 0.5  # km/h, must match physical.speed_histogram


def distance_above(hist: pd.DataFrame, threshold: pd.Series | float) -> pd.Series:
    """Metres covered above a threshold (scalar or per player), from speed histograms."""
    h = hist.copy()
    thr = h.player_id.map(threshold) if isinstance(threshold, pd.Series) else threshold
    h["d"] = (h.bin_kmh + BIN / 2) / 3.6 * h.frames / 10
    return h[h.bin_kmh >= thr].groupby("player_id").d.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=0.85)
    ap.add_argument("--absolute", type=float, default=25.0)
    ap.add_argument("--min-matches", type=int, default=3)
    args = ap.parse_args()

    files = [f for f in OUT.glob("*/speed_histogram.csv") if f.parent.name.isdigit()]
    if not files:
        sys.exit("no histograms: run scripts/run_all.py first")
    hist = (pd.concat([pd.read_csv(f) for f in files])
            .groupby(["player_id", "bin_kmh"]).frames.sum().reset_index())
    p = pd.read_csv(OUT / "all_matches" / "players_pooled.csv")
    p = p[(p.n_matches >= args.min_matches) & (p.position != "Other")].set_index("player_id")
    # Reference top speed: SkillCorner's season PSV-99 when available. Ours is computed on
    # few matches and correlates only ~0.6 with it; a relative threshold inherits that error
    # directly (a 24 km/h "top speed" turns ordinary running into "sprinting").
    agg = collapse_aggregates(pd.read_csv(fetch_physical_aggregates(ROOT / "data")))
    psv = agg.set_index("player_id")["psv99"].reindex(p.index)
    p["top_speed_ref"] = psv.fillna(p.top_speed_kmh)
    hist = hist[hist.player_id.isin(p.index)]

    per90 = 90 / p.minutes_official
    t = pd.DataFrame({
        "absolute": distance_above(hist, args.absolute).reindex(p.index).fillna(0) * per90,
        "relative": distance_above(hist, p.top_speed_ref * args.fraction).reindex(p.index).fillna(0) * per90,
    })
    t["position"] = p.position
    t["name"] = p.name
    t["top_speed_ref"] = p.top_speed_ref
    rho = t.absolute.rank().corr(t.relative.rank())

    med = t.groupby("position")[["absolute", "relative"]].median().sort_values("absolute")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    y = np.arange(len(med))
    ax.barh(y - 0.18, med.absolute, height=0.36, color="#9aa5b1",
            label=f"above {args.absolute:.0f} km/h (fixed)")
    ax.barh(y + 0.18, med.relative, height=0.36, color="#D55E00",
            label=f"above {args.fraction:.0%} of own top speed (SkillCorner PSV-99)")
    ax.set_yticks(y, med.index)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("sprint distance per 90 min (m), median per position")
    ax.set_title(f"Fixed vs relative sprint threshold, {len(t)} players ({args.min_matches}+ matches). "
                 f"Rank correlation between the two: {rho:.2f}", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    out = OUT / "all_matches"
    fig.savefig(out / "relative_thresholds.png", dpi=150)
    t.round(1).to_csv(out / "relative_thresholds.csv")

    t["rank_abs"] = t.absolute.rank(ascending=False)
    t["rank_rel"] = t.relative.rank(ascending=False)
    t["move"] = t.rank_abs - t.rank_rel
    pd.set_option("display.width", 200)
    print(f"rank correlation fixed vs relative: {rho:.2f}")
    print(med.round(0).to_string())
    print("\nbiggest climbers with a relative threshold:")
    print(t.sort_values("move", ascending=False).head(5)[
        ["name", "position", "top_speed_ref", "absolute", "relative", "rank_abs", "rank_rel"]].round(1).to_string())


if __name__ == "__main__":
    main()
