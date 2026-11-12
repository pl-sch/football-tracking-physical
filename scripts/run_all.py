"""Every available match, then validation pooled per player.

Question: is the weak sprint correlation on one match sampling noise or a method problem?
If it is noise, pooling several matches per player should raise it.

    python scripts/run_all.py              # all matches (~1.8 GB download on first run)
    python scripts/run_all.py --limit 5    # first 5 matches only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footperf import add_rates, compare, fetch_physical_aggregates, list_matches  # noqa: E402
from footperf import viz  # noqa: E402
from footperf import sportec  # noqa: E402
from footperf.pipeline import pool_players, process_match, process_match_sportec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--min-matches", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--plot-min-matches", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="recompute cached matches")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    matches = list_matches(args.data).assign(league="A-League 2024/25")
    if args.limit:
        matches = matches.head(args.limit)
    agg = pd.read_csv(fetch_physical_aggregates(args.data))

    parts, single = [], []
    for i, m in matches.iterrows():
        print(f"[{i + 1}/{len(matches)}] {m.match_id} {m.date} {m.home} - {m.away}")
        s = process_match(int(m.match_id), args.data, args.out, force=args.force)
        parts.append(s.assign(league=m.league))
        _, val = compare(add_rates(s, "minutes_official"), agg)
        single.append(val.assign(match_id=m.match_id))

    # Bundesliga (IDSSE): only if the files were fetched and converted, see scripts/fetch_idsse.sh
    bl = []
    for mid in sportec.MATCHES:
        try:
            p = sportec.paths(mid, args.data)
        except FileNotFoundError:
            continue
        info = sportec.match_info(p["meta"])
        league = f"{info['competition']} 2022/23"
        print(f"[Bundesliga] {mid} {info['date']} {info['home']} - {info['away']}")
        parts.append(process_match_sportec(mid, args.data, args.out, args.force).assign(league=league))
        bl.append({"match_id": mid, "date": info["date"], "home": info["home"], "away": info["away"],
                   "league": league})
    if bl:
        matches = pd.concat([matches, pd.DataFrame(bl)], ignore_index=True)
    else:
        print("no Bundesliga data found (run scripts/fetch_idsse.sh): A-League only")
    by_match = pd.concat(parts, ignore_index=True)

    out = args.out / "all_matches"
    out.mkdir(parents=True, exist_ok=True)
    by_match.round(3).to_csv(out / "players_by_match.csv", index=False)
    matches.to_csv(out / "matches.csv", index=False)

    # Baseline: average correlation when each match is compared on its own
    single = pd.concat(single)
    table = single.groupby("metric", sort=False)[["pearson_r", "median_ratio"]].mean()
    table.columns = ["r_single_match", "ratio_single_match"]

    pooled = add_rates(pool_players(by_match), "minutes_official")
    pooled.round(3).to_csv(out / "players_pooled.csv", index=False)
    pooled_al = pooled[pooled.league.str.startswith("A-League")]
    for k in args.min_matches:
        sub = pooled_al[pooled_al.n_matches >= k]
        players, val = compare(sub, agg)
        v = val.set_index("metric")
        table[f"r_pooled_{k}+"] = v["pearson_r"]
        table[f"n_{k}+"] = v["n_players"]
        if k == args.plot_min_matches:
            table["ratio_pooled"] = v["median_ratio"]
            viz.validation_grid(players, val, out / "validation_pooled.png",
                                ylabel=f"ours, pooled ({k}+ matches)")
            players.round(2).to_csv(out / "validation_pooled_players.csv", index=False)

    table = table.reset_index()
    table.round(3).to_csv(out / "validation_pooled.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"\n{len(matches)} matches, {by_match.player_id.nunique()} players")
    print("validation against SkillCorner aggregates (A-League only):")
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
