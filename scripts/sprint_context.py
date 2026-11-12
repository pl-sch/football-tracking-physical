"""What are sprints for? Links every effort to SkillCorner's dynamic events.

Needs scripts/run_all.py to have run first (reads outputs/*/efforts.csv).

    python scripts/sprint_context.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from footperf import viz  # noqa: E402


def main() -> None:
    outputs = ROOT / "outputs"
    files = [f for f in outputs.glob("*/efforts.csv") if f.parent.name.isdigit()]
    if not files:
        sys.exit("no efforts found: run scripts/run_all.py first")
    efforts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pooled = pd.read_csv(outputs / "all_matches" / "players_pooled.csv")

    out = outputs / "all_matches"
    by_pos = viz.sprint_context_by_position(efforts, out / "sprint_context_by_position.png")
    top = viz.top_sprinters(efforts, pooled, out / "top_sprinters.png")
    by_pos.round(3).to_csv(out / "sprint_context_by_position.csv")
    top.round(2).to_csv(out / "top_sprinters.csv")

    s = efforts[efforts.kind == "sprint"]
    behind = s[s.context == "run in behind"]
    pd.set_option("display.width", 200)
    print(f"{len(files)} matches, {len(s)} sprints")
    print(s.context.value_counts(normalize=True).round(2).to_string())
    print(f"\nruns in behind at sprint speed targeted by a pass: {behind.run_targeted.mean():.0%}")
    print("\n" + by_pos.round(2).to_string())


if __name__ == "__main__":
    main()
