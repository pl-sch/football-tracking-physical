"""Turn clean optical tracking into broadcast-like tracking, one defect at a time.

The Bundesliga data (25 Hz, every player always tracked) has a known answer: the DFL's own
speed. Degrading it step by step and running the broadcast pipeline on each version shows
how much error each defect of broadcast tracking causes on its own:

  resample_10hz   10 frames per second instead of 25
  add_noise       random position error (metres), as a camera-based estimate would have
  drop_dead_ball  whole frames missing during stoppages, like broadcast replays and close-ups
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample(track: pd.DataFrame, fps_in: float, fps_out: float = 10.0) -> pd.DataFrame:
    """Linear interpolation of x, y onto a regular grid at fps_out, per player and period.

    Frame numbers are rebuilt on the new grid so that consecutive output frames differ by 1.
    Possession and alive flags take the value of the nearest earlier input frame.
    """
    out = []
    for (pid, period), g in track.groupby(["player_id", "period"], sort=False):
        g = g.sort_values("frame")
        t_in = g["frame"].to_numpy() / fps_in
        t_out = np.arange(np.ceil(t_in[0] * fps_out), np.floor(t_in[-1] * fps_out) + 1) / fps_out
        if len(t_out) < 2:
            continue
        idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, len(g) - 1)
        part = pd.DataFrame({
            "frame": np.round(t_out * fps_out).astype(np.int64),
            "period": period,
            "player_id": pid,
            "x": np.interp(t_out, t_in, g["x"].to_numpy()),
            "y": np.interp(t_out, t_in, g["y"].to_numpy()),
            "detected": True,
            "possession": g["possession"].to_numpy()[idx],
            "alive": g["alive"].to_numpy()[idx],
        })
        out.append(part)
    res = pd.concat(out, ignore_index=True)
    res["possession"] = pd.Categorical(res["possession"], categories=["home", "away"])
    return res


def add_noise(track: pd.DataFrame, sigma_m: float, seed: int = 0) -> pd.DataFrame:
    """Independent Gaussian error on x and y, standard deviation sigma_m metres."""
    if sigma_m <= 0:
        return track
    rng = np.random.default_rng(seed)
    out = track.copy()
    out["x"] = out["x"] + rng.normal(0, sigma_m, len(out))
    out["y"] = out["y"] + rng.normal(0, sigma_m, len(out))
    return out


def drop_dead_ball(track: pd.DataFrame, share: float = 0.26, seed: int = 0) -> pd.DataFrame:
    """Remove whole frames (all players) in dead-ball stretches until `share` of frames is gone.

    Broadcast video cuts to replays and close-ups mostly when the ball is out of play, so the
    gaps are taken from dead-ball stretches, in random order. 26% is what was measured on the
    A-League broadcast data.
    """
    frames = track[["period", "frame", "alive"]].drop_duplicates(["period", "frame"]).sort_values(
        ["period", "frame"])
    total = len(frames)
    dead = frames[~frames["alive"].astype(bool)]
    # contiguous dead-ball blocks
    brk = (dead["frame"].diff() != 1) | (dead["period"] != dead["period"].shift())
    dead = dead.assign(block=brk.cumsum())
    blocks = dead.groupby("block").size()
    rng = np.random.default_rng(seed)
    order = rng.permutation(blocks.index.to_numpy())
    chosen, removed = [], 0
    for b in order:
        if removed >= share * total:
            break
        chosen.append(b)
        removed += blocks[b]
    gone = dead[dead["block"].isin(chosen)][["period", "frame"]]
    key = pd.MultiIndex.from_frame(track[["period", "frame"]])
    drop = key.isin(pd.MultiIndex.from_frame(gone))
    return track[~drop].reset_index(drop=True)
