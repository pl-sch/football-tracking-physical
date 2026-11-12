"""Full pipeline for one match: download, speeds, physical metrics, validation, figures.

    python scripts/run_match.py --match 1886347
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footperf import (add_rates, add_speed, compare, fetch_match,  # noqa: E402
                      fetch_physical_aggregates, load_tracking)
from footperf import viz  # noqa: E402
from footperf.pipeline import process_match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", type=int, default=1886347)
    ap.add_argument("--window", type=int, default=9, help="Savitzky-Golay window, frames (odd)")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    out = args.out / str(args.match)
    out.mkdir(parents=True, exist_ok=True)

    paths = fetch_match(args.match, args.data)
    agg = pd.read_csv(fetch_physical_aggregates(args.data))
    summary = process_match(args.match, args.data, args.out, args.window, force=True)
    summary = add_rates(summary, "minutes_official").sort_values("total_distance_m", ascending=False)
    efforts = pd.read_csv(out / "efforts.csv")
    track = add_speed(load_tracking(paths["tracking"]), window=args.window)  # for figures

    table, val = compare(summary, agg)
    val.round(3).to_csv(out / "validation_summary.csv", index=False)
    table.round(2).to_csv(out / "validation_players.csv", index=False)

    # Figures
    viz.validation_grid(table, val, out / "validation.png")

    top = summary.sort_values("sprint_count", ascending=False).iloc[0]
    meta = json.loads(Path(paths["match"]).read_text())
    home_id = meta["home_team"]["id"]
    team_id = next(p["team_id"] for p in meta["players"] if p["id"] == top.player_id)
    sides = meta["home_team_side"]  # home team direction per period
    sign = {}
    for i, side in enumerate(sides, start=1):
        home_ltr = side == "left_to_right"
        attacking_ltr = home_ltr if team_id == home_id else not home_ltr
        sign[i] = 1 if attacking_ltr else -1
    viz.sprint_map(efforts, sign, int(top.player_id), top["name"], out=out / "sprint_map.png")

    first_sprint = efforts[(efforts.player_id == top.player_id) & (efforts.kind == "sprint")].iloc[0]
    start = int(first_sprint.start_frame) - 600
    viz.speed_profile(track, int(top.player_id), top["name"], int(first_sprint.period),
                      out / "speed_profile.png", raw_window=(start, start + 1200))

    cols = ["name", "team", "position", "minutes", "total_distance_m", "meters_per_minute",
            "hsr_distance_m", "sprint_distance_m", "hsr_count", "sprint_count", "top_speed_kmh"]
    pd.set_option("display.width", 200)
    print(summary[cols].round(1).head(10).to_string(index=False))
    print("\nValidation vs SkillCorner season aggregates")
    print(val.round(2).to_string(index=False))
    sp = efforts[efforts.kind == "sprint"]
    print(f"\nframes with the player visible: {track.detected.mean():.0%} overall, "
          f"{sp.detected_share.mean():.0%} during sprints; "
          f"{(sp.detected_share < 0.5).mean():.0%} of sprints are mostly extrapolated")
    print(f"\noutputs written to {out}")


if __name__ == "__main__":
    main()
