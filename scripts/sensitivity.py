"""How much do the results depend on processing choices?

Grid: smoothing window x minutes denominator. For each cell, median ratio and
correlation against SkillCorner's season aggregates.

    python scripts/sensitivity.py --match 1886347
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footperf import (add_rates, add_speed, compare, fetch_match,  # noqa: E402
                      fetch_physical_aggregates, load_players, load_tracking, player_summary)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=1886347)
    ap.add_argument("--windows", type=int, nargs="+", default=[5, 9, 13, 17, 21])
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    paths = fetch_match(args.match, args.data)
    agg = pd.read_csv(fetch_physical_aggregates(args.data))
    players = load_players(paths["match"])
    raw = load_tracking(paths["tracking"])

    rows = []
    for w in args.windows:
        base = players.merge(player_summary(add_speed(raw, window=w)), on="player_id")
        for denom in ("minutes_tracked", "minutes_official"):
            _, val = compare(add_rates(base, denom), agg)
            val.insert(0, "denominator", denom)
            val.insert(0, "window", w)
            rows.append(val)
    res = pd.concat(rows, ignore_index=True)
    out = args.out / str(args.match)
    out.mkdir(parents=True, exist_ok=True)
    res.round(3).to_csv(out / "sensitivity.csv", index=False)

    pd.set_option("display.width", 200)
    for col in ("median_ratio", "pearson_r"):
        print(f"\n{col}")
        print(res.pivot_table(index="metric", columns=["denominator", "window"], values=col)
              .round(2).to_string())


if __name__ == "__main__":
    main()
