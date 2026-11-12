"""Bundesliga (optical tracking): our speed vs the provider's own speed.

The A-League check compares season averages, which is indirect. Here the DFL publishes a
speed value for every player and every frame, so the comparison is exact: same player,
same match, same thresholds.

    python scripts/bundesliga_check.py      # needs scripts/run_all.py to have run with the IDSSE data
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "all_matches"


def window_scan(match_id: str = "J03WPY") -> None:
    """Smoothing window vs agreement with the provider, on one match."""
    sys.path.insert(0, str(ROOT / "src"))
    from footperf import sportec
    from footperf.kinematics import add_speed

    p = sportec.paths(match_id, ROOT / "data")
    raw = sportec.load_tracking(p["positions"], p["ball"])
    rows = []
    for w in (7, 11, 15, 23, 31):
        a = sportec.provider_agreement(add_speed(raw, window=w, fps=sportec.FPS), sportec.FPS)
        row = {"window_frames": w, "window_s": round(w / sportec.FPS, 2)}
        for k in ("total", "hsr", "sprint"):
            row[f"{k}_ratio"] = (a[f"{k}_ours"] / a[f"{k}_provider"]).median()
        row["speed_rmse_kmh"] = a.speed_rmse_kmh.median()
        rows.append(row)
    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT / "bundesliga_window_scan.csv", index=False)
    print(res.to_string(index=False))


def main() -> None:
    if "--window-scan" in sys.argv:
        window_scan()
        return
    b = pd.read_csv(OUT / "players_by_match.csv", low_memory=False)
    b = b[b.league.str.contains("Bundesliga") & (b.minutes_official >= 20)]
    if b.empty:
        sys.exit("no Bundesliga rows: fetch the IDSSE data and run scripts/run_all.py")

    panels = [("total", "total distance (m)"), ("hsr", "HSR distance, 20-25 km/h (m)"),
              ("sprint", "sprint distance, >25 km/h (m)")]
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (k, label) in zip(axes, panels):
        x, y = b[f"{k}_provider"], b[f"{k}_ours"]
        r = np.corrcoef(x, y)[0, 1]
        ratio = (y / x.replace(0, np.nan)).median()
        rows.append({"metric": k, "pearson_r": r, "median_ratio": ratio, "n": len(b)})
        hi = max(x.max(), y.max()) * 1.05
        ax.plot([0, hi], [0, hi], color="#9aa5b1", lw=0.8, ls="--")
        ax.scatter(x, y, s=14, color="#2f6fb0", alpha=0.7)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_title(f"{label}\nr={r:.3f}  median ratio={ratio:.3f}", loc="left", fontsize=9)
        ax.set_xlabel("from DFL speed", fontsize=8)
        ax.set_ylabel("from our Savitzky-Golay speed", fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Bundesliga 2022/23, optical tracking at 25 Hz: {len(b)} player-matches",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "bundesliga_provider_check.png", dpi=150)

    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "bundesliga_provider_check.csv", index=False)
    print(res.round(3).to_string(index=False))
    print(f"\nspeed RMSE vs provider, median over player-matches: {b.speed_rmse_kmh.median():.2f} km/h")


if __name__ == "__main__":
    main()
