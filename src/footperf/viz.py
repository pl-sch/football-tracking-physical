"""Figures. Matplotlib only, no pitch library needed."""

from __future__ import annotations

from pathlib import Path

import matplotlib

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


def validation_grid(table: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
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
        ax.set_ylabel("ours, this match", fontsize=8)
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
