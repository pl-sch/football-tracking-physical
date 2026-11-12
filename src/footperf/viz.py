"""Figures. Matplotlib only, no pitch library needed."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import FPS

INK = "#1f2933"
MUTED = "#9aa5b1"
ACCENT = "#d64545"
BLUE = "#2f6fb0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(alpha=0.25, linewidth=0.6)


def speed_profile(track: pd.DataFrame, player_id: int, name: str, period: int, out: Path,
                  raw_window: tuple[int, int] | None = None) -> None:
    """Speed and acceleration: raw finite differences vs Savitzky-Golay, over ~2 minutes.

    On SkillCorner data the positions are already smoothed upstream, so speed barely
    changes. Acceleration is a second derivative and still shows the difference.
    """
    g = track[(track.player_id == player_id) & (track.period == period)].sort_values("frame")
    if raw_window:
        g = g[(g.frame >= raw_window[0]) & (g.frame < raw_window[1])]
    t = ((g["frame"] - g["frame"].iloc[0]) / FPS).to_numpy()
    gap = np.diff(g["frame"]) != 1
    raw_v = np.hypot(np.diff(g["x"]), np.diff(g["y"])) * FPS
    raw_v[gap] = np.nan
    raw_a = np.diff(raw_v) * FPS
    raw_a[gap[1:]] = np.nan

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(t[1:], raw_v * 3.6, color=MUTED, lw=0.8, label="raw (frame-to-frame difference)")
    ax1.plot(t, g["speed"] * 3.6, color=BLUE, lw=1.6, label="Savitzky-Golay")
    for lvl, lab in ((20, "HSR 20 km/h"), (25, "sprint 25 km/h")):
        ax1.axhline(lvl, color=ACCENT, lw=0.8, ls="--")
        ax1.text(t[-1], lvl + 0.5, lab, color=ACCENT, ha="right", fontsize=8)
    ax1.set_ylabel("speed (km/h)")
    ax1.set_ylim(0, 40)
    ax1.set_title(f"{name}: speed and acceleration, period {period}", loc="left", color=INK)
    ax1.legend(frameon=False, loc="upper left", fontsize=8)

    ax2.plot(t[2:], raw_a, color=MUTED, lw=0.8)
    ax2.plot(t, g["accel"], color=BLUE, lw=1.6)
    for lvl in (3, -3):
        ax2.axhline(lvl, color=ACCENT, lw=0.8, ls="--")
    ax2.text(t[-1], 3.3, "+/- 3 m/s² (high accel / decel)", color=ACCENT, ha="right", fontsize=8)
    ax2.set_ylabel("acceleration (m/s²)")
    ax2.set_ylim(-10, 10)
    ax2.set_xlabel("time (s)")
    for ax in (ax1, ax2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def validation_grid(table: pd.DataFrame, summary: pd.DataFrame, out: Path,
                    ylabel: str = "ours, this match") -> None:
    """Our match value vs SkillCorner season average, one panel per metric."""
    metrics = summary["metric"].tolist()
    ncol = 3
    nrow = int(np.ceil(len(metrics) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.6 * nrow))
    for ax, (_, s) in zip(axes.flat, summary.iterrows()):
        m = s["metric"]
        a, b = table[f"{m}__ours"], table[f"{m}__sc"]
        hi = np.nanmax([a.max(), b.max()]) * 1.05
        lo = np.nanmin([a.min(), b.min()])
        lo = lo * 0.95 if lo > 0.5 * hi else 0  # zoom when all values sit far from zero
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls="--")
        ax.scatter(b, a, s=18, color=BLUE, alpha=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{m}\nr={s['pearson_r']:.2f}  median ratio={s['median_ratio']:.2f}",
                     fontsize=9, loc="left", color=INK)
        ax.set_xlabel("SkillCorner season avg", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        _style(ax)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def sprint_map(efforts: pd.DataFrame, track_x_sign: dict[int, int], player_id: int, name: str,
               pitch=(104, 68), out: Path | None = None) -> None:
    """Arrows from start to end of each sprint/HSR, always attacking left to right."""
    e = efforts[efforts.player_id == player_id]
    L, W = pitch
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.add_patch(plt.Rectangle((-L / 2, -W / 2), L, W, fill=False, color=MUTED))
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color=MUTED, lw=0.8))
    for sx in (-1, 1):
        ax.add_patch(plt.Rectangle((sx * L / 2 - (16.5 if sx > 0 else 0), -20.16), 16.5, 40.32,
                                   fill=False, color=MUTED, lw=0.8))
    for _, r in e.iterrows():
        s = track_x_sign.get(int(r["period"]), 1)
        color = ACCENT if r["kind"] == "sprint" else BLUE
        ax.annotate("", xy=(s * r.x_end, s * r.y_end), xytext=(s * r.x_start, s * r.y_start),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.85))
    n_s = (e.kind == "sprint").sum()
    n_h = (e.kind == "hsr").sum()
    ax.set_title(f"{name}: {n_s} sprints (red), {n_h} HSR (blue). Attacking left to right",
                 loc="left", color=INK, fontsize=10)
    ax.set_xlim(-L / 2 - 2, L / 2 + 2)
    ax.set_ylim(-W / 2 - 2, W / 2 + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    plt.close(fig)


# Okabe-Ito, colour-blind safe. Grey for "no event".
CONTEXT_COLORS = {
    "with the ball": "#E69F00",
    "run in behind": "#D55E00",
    "attacking the box": "#CC79A7",
    "width / overlap": "#F0E442",
    "support / coming short": "#56B4E9",
    "pressing": "#0072B2",
    "no linked event": "#BBBBBB",
}


def _stacked(ax, table: pd.DataFrame, value_fmt: str | None = None) -> None:
    left = np.zeros(len(table))
    for ctx, color in CONTEXT_COLORS.items():
        if ctx not in table:
            continue
        vals = table[ctx].to_numpy()
        ax.barh(table.index, vals, left=left, color=color, label=ctx, height=0.7)
        if value_fmt:
            for y, (l, v) in enumerate(zip(left, vals)):
                if v >= 0.06:
                    ax.text(l + v / 2, y, format(v, value_fmt), ha="center", va="center",
                            fontsize=7.5, color=INK)
        left += vals
    ax.invert_yaxis()
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(left=False, colors=INK)


def sprint_context_by_position(efforts: pd.DataFrame, out: Path) -> pd.DataFrame:
    s = efforts[(efforts.kind == "sprint") & efforts.position.notna() & (efforts.position != "Other")]
    t = pd.crosstab(s.position, s.context, normalize="index")
    t = t.loc[t["no linked event"].sort_values().index]
    n = s.groupby("position").size()
    t.index = [f"{p} ({n[p]} sprints)" for p in t.index]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    _stacked(ax, t, ".0%")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("What are sprints for? Share of sprints by football context, 20 A-League matches",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return t


def top_sprinters(efforts: pd.DataFrame, pooled: pd.DataFrame, out: Path, n: int = 15,
                  min_matches: int = 3) -> pd.DataFrame:
    p = pooled[(pooled.n_matches >= min_matches) & (pooled.position != "Other")].set_index("player_id")
    s = efforts[(efforts.kind == "sprint") & efforts.player_id.isin(p.index)]
    t = pd.crosstab(s.player_id, s.context).div(p["minutes_official"], axis=0).mul(90).dropna()
    t = t.loc[t.sum(axis=1).sort_values(ascending=False).index[:n]]
    t.index = [f"{p.loc[i, 'name']} ({p.loc[i, 'position']}, {p.loc[i, 'team']})" for i in t.index]
    fig, ax = plt.subplots(figsize=(13, 0.38 * n + 1.6))
    _stacked(ax, t)
    ax.set_xlabel("sprints per 90 minutes", color=INK)
    ax.set_title(f"Top {n} sprinters ({min_matches}+ matches), split by what the sprint was for",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return t
