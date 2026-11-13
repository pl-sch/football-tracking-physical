"""How much error does broadcast-quality tracking cause by itself?

Takes the clean Bundesliga optical tracking, degrades it one defect at a time (frame rate,
position noise, missing frames), runs the broadcast pipeline settings on each version, and
compares with the clean result. The clean reference is our 25 Hz pipeline, which matches the
DFL's own speed within 1% (scripts/bundesliga_check.py).

    python scripts/degrade_optical.py              # all 7 matches, ~10 min
    python scripts/degrade_optical.py --matches J03WPY
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

from footperf import sportec  # noqa: E402
from footperf.degrade import add_noise, drop_dead_ball, resample  # noqa: E402
from footperf.kinematics import add_speed  # noqa: E402
from footperf.physical import player_summary  # noqa: E402
from footperf.pipeline import WINDOW_S_OPTICAL, window_frames  # noqa: E402

OUT = ROOT / "outputs" / "all_matches"
METRICS = ["total_distance_m", "hsr_distance_m", "sprint_distance_m", "hsr_count",
           "sprint_count", "high_accel_count", "top_speed_kmh"]
LABELS = {"total_distance_m": "total distance", "hsr_distance_m": "HSR distance",
          "sprint_distance_m": "sprint distance", "hsr_count": "HSR count",
          "sprint_count": "sprint count", "high_accel_count": "high accelerations",
          "top_speed_kmh": "top speed"}

# name -> (noise in metres, drop dead-ball frames?)
SCENARIOS = {
    "10 Hz": (0.0, False),
    "10 Hz + 10 cm noise": (0.10, False),
    "10 Hz + 30 cm noise": (0.30, False),
    "10 Hz + 50 cm noise": (0.50, False),
    "10 Hz + 30 cm noise + 26% frames missing": (0.30, True),
}
WINDOWS = [5, 9, 13, 17]  # frames at 10 Hz, scanned on the 30 cm scenario


def summary(track: pd.DataFrame, fps: float, window: int) -> pd.DataFrame:
    s = player_summary(add_speed(track, window=window, fps=fps), fps=fps)
    return s.set_index("player_id")[METRICS + ["minutes_tracked"]]


def compare(clean: pd.DataFrame, test: pd.DataFrame) -> dict:
    m = clean.join(test, rsuffix="_t", how="inner")
    m = m[m.minutes_tracked >= 45]
    row = {"n": len(m)}
    for c in METRICS:
        a, b = m[f"{c}_t"], m[c]
        ok = b > 0
        row[f"{c}__ratio"] = (a[ok] / b[ok]).median()
        row[f"{c}__r"] = np.corrcoef(a[ok], b[ok])[0, 1] if ok.sum() > 2 else np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=list(sportec.MATCHES))
    ap.add_argument("--plot-only", action="store_true", help="redraw from degradation_by_match.csv")
    args = ap.parse_args()

    if args.plot_only:
        report(pd.read_csv(OUT / "degradation_by_match.csv"))
        return
    rows = []
    for mid in args.matches:
        p = sportec.paths(mid, ROOT / "data")
        raw = sportec.load_tracking(p["positions"], p["ball"])
        clean = summary(raw, sportec.FPS, window_frames(sportec.FPS, WINDOW_S_OPTICAL))
        ten = resample(raw, sportec.FPS, 10.0)
        print(f"{mid}: clean done, {len(ten):,} rows at 10 Hz", flush=True)
        for name, (sigma, drop) in SCENARIOS.items():
            t = add_noise(ten, sigma, seed=hash(mid) % 2**32)
            if drop:
                t = drop_dead_ball(t, 0.26, seed=1)
            windows = WINDOWS if name == "10 Hz + 30 cm noise" else [9]
            for w in windows:
                rows.append({"match_id": mid, "scenario": name, "window": w,
                             **compare(clean, summary(t, 10.0, w))})
        print(f"{mid}: scenarios done", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "degradation_by_match.csv", index=False)
    report(res)


def report(res: pd.DataFrame) -> None:
    cols = [c for c in res.columns if "__" in c]
    agg = res.groupby(["scenario", "window"], sort=False)[cols].median().reset_index()
    agg.round(3).to_csv(OUT / "degradation.csv", index=False)

    main9 = agg[agg.window == 9].set_index("scenario")
    ratio = main9[[f"{c}__ratio" for c in METRICS]]
    ratio.columns = [LABELS[c] for c in METRICS]
    corr = main9[[f"{c}__r" for c in METRICS]]
    corr.columns = ratio.columns

    # Figure: ratio to clean truth, per metric and scenario
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors = ["#9aa5b1", "#56B4E9", "#0072B2", "#003f6b", "#D55E00"]
    width = 0.16
    x = np.arange(len(ratio.columns))
    for i, (name, r) in enumerate(ratio.iterrows()):
        ax.bar(x + (i - 2) * width, r.to_numpy(), width, label=name, color=colors[i])
    ax.axhline(1, color="#1f2933", lw=0.8)
    ax.set_xticks(x, ratio.columns, fontsize=9)
    ax.set_ylabel("degraded / clean (median over players)")
    ax.set_ylim(0, max(1.5, np.nanmax(ratio.to_numpy()) * 1.08))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("Bundesliga optical tracking degraded to broadcast quality, analysed with the "
                 "broadcast settings (0.9 s smoothing)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "degradation.png", dpi=150)

    scan = agg[agg.scenario == "10 Hz + 30 cm noise"].set_index("window")
    scan = scan[[f"{c}__ratio" for c in METRICS]]
    scan.columns = [LABELS[c] for c in METRICS]

    pd.set_option("display.width", 220)
    print("\nmedian ratio degraded / clean (window 9 frames = 0.9 s)")
    print(ratio.round(2).to_string())
    print("\ncorrelation with clean values")
    print(corr.round(2).to_string())
    print("\nsmoothing window on '10 Hz + 30 cm noise' (frames at 10 Hz)")
    print(scan.round(2).to_string())


if __name__ == "__main__":
    main()
