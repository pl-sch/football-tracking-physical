"""Interactive explorer for the precomputed results (no tracking download needed).

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ALL = OUT / "all_matches"

# column -> (label, kind). kind: "volume" is shown per 90 min, "rate" as is.
METRICS = {
    "total_distance_m": ("Distance (m)", "volume"),
    "running_distance_m": ("Running 15-20 km/h (m)", "volume"),
    "hsr_distance_m": ("HSR 20-25 km/h (m)", "volume"),
    "sprint_distance_m": ("Sprint >25 km/h (m)", "volume"),
    "hsr_count": ("HSR actions", "volume"),
    "sprint_count": ("Sprints", "volume"),
    "high_accel_count": ("High accelerations", "volume"),
    "high_decel_count": ("High decelerations", "volume"),
    "top_speed_kmh": ("Top speed (km/h)", "rate"),
    "meters_per_minute_tip": ("m/min, team in possession", "rate"),
    "meters_per_minute_otip": ("m/min, team out of possession", "rate"),
    "passes": ("Passes", "volume"),
    "pass_accuracy": ("Pass accuracy (%)", "rate"),
    "shots": ("Shots", "volume"),
    "possession_losses": ("Possessions lost", "volume"),
    "off_ball_runs": ("Off-ball runs", "volume"),
    "runs_targeted": ("Runs targeted by a pass", "volume"),
    "defensive_engagements": ("Defensive engagements", "volume"),
    "regains": ("Regains", "volume"),
}
DEFAULT = ["total_distance_m", "hsr_distance_m", "sprint_count", "top_speed_kmh", "passes",
           "pass_accuracy", "regains"]
CONTEXT_COLORS = {
    "with the ball": "#E69F00", "run in behind": "#D55E00", "attacking the box": "#CC79A7",
    "width / overlap": "#F0E442", "support / coming short": "#56B4E9", "pressing": "#0072B2",
    "no linked event": "#BBBBBB",
}


@st.cache_data
def load():
    by_match = pd.read_csv(ALL / "players_by_match.csv")
    pooled = pd.read_csv(ALL / "players_pooled.csv")
    matches = pd.read_csv(ALL / "matches.csv")
    matches["label"] = matches.date + "  " + matches.home + " - " + matches.away
    efforts = pd.concat([pd.read_csv(f).assign(match_id=int(f.parent.name))
                         for f in OUT.glob("*/efforts.csv") if f.parent.name.isdigit()],
                        ignore_index=True)
    for df in (by_match, pooled):
        df["pass_accuracy"] = 100 * df.passes_successful / df.passes.replace(0, np.nan)
        df["meters_per_minute_tip"] = df.distance_tip_m / df.minutes_tip.replace(0, np.nan)
        df["meters_per_minute_otip"] = df.distance_otip_m / df.minutes_otip.replace(0, np.nan)
    return by_match, pooled, matches, efforts


def per90(df: pd.DataFrame, cols: list[str], on: bool = True) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if on and METRICS[c][1] == "volume":
            out[c] = out[c] / out.minutes_official * 90
    return out


def labels(cols, p90=True):
    return {c: METRICS[c][0] + (" /90" if p90 and METRICS[c][1] == "volume" else "") for c in cols}


def pitch(efforts: pd.DataFrame):
    L, W = 104, 68
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.add_patch(plt.Rectangle((-L / 2, -W / 2), L, W, fill=False, color="#9aa5b1"))
    ax.axvline(0, color="#9aa5b1", lw=0.8)
    ax.add_patch(plt.Circle((0, 0), 9.15, fill=False, color="#9aa5b1", lw=0.8))
    for x0 in (-L / 2, L / 2 - 16.5):
        ax.add_patch(plt.Rectangle((x0, -20.16), 16.5, 40.32, fill=False, color="#9aa5b1", lw=0.8))
    for _, r in efforts.iterrows():
        ax.annotate("", xy=(r.x_end_att, r.y_end_att), xytext=(r.x_start_att, r.y_start_att),
                    arrowprops=dict(arrowstyle="->", lw=1.3, alpha=0.85,
                                    color=CONTEXT_COLORS.get(r.context, "#888")))
    ax.text(L / 2, W / 2 + 3, "attacking →", ha="right", color="#52606d", fontsize=9)
    ax.set_xlim(-L / 2 - 2, L / 2 + 2)
    ax.set_ylim(-W / 2 - 2, W / 2 + 5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    return fig


st.set_page_config(page_title="Football tracking explorer", layout="wide")
by_match, pooled, matches, efforts = load()

st.title("Football tracking explorer")
st.caption("A-League 2024/25, 20 matches of SkillCorner broadcast tracking. Physical metrics computed "
           "from raw positions, technical stats from SkillCorner dynamic events. "
           "[Code and method](https://github.com/pl-sch/football-tracking-physical)")

tab_player, tab_match, tab_compare = st.tabs(["Player", "Match", "Compare players"])

# ---------------------------------------------------------------- Player
with tab_player:
    p = pooled[pooled.position != "Other"].sort_values("name")
    opts = {int(k): v for k, v in zip(p.player_id, p.name + " (" + p.team + ", "
                                      + p.position.fillna("?") + ")")}
    default = int(p.sort_values("sprint_count").player_id.iloc[-1])
    pid = st.selectbox("Search a player (type a name)", list(opts), format_func=lambda i: opts[int(i)],
                       index=list(opts).index(default),
                       help="Click the box and start typing: the list filters as you type.")
    me = pooled[pooled.player_id == pid].iloc[0]
    peers = per90(pooled[(pooled.position == me.position) & (pooled.minutes_official >= 180)],
                  list(METRICS))
    mine = per90(pooled[pooled.player_id == pid], list(METRICS)).iloc[0]

    c = st.columns(5)
    c[0].metric("Matches", int(me.n_matches))
    c[1].metric("Minutes", f"{me.minutes_official:.0f}")
    c[2].metric("Distance /90", f"{mine.total_distance_m / 1000:.1f} km")
    c[3].metric("Sprints /90", f"{mine.sprint_count:.1f}")
    c[4].metric("Top speed", f"{me.top_speed_kmh:.1f} km/h")

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Percentile among {me.position}s")
        st.caption(f"{len(peers)} players with 180+ minutes. Volumes per 90 minutes.")
        rows = []
        for col in METRICS:
            v = mine[col]
            if pd.isna(v) or peers[col].notna().sum() < 5:
                continue
            rows.append({"metric": labels([col])[col], "value": round(v, 1),
                         "percentile": round((peers[col] < v).mean() * 100)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                     column_config={"percentile": st.column_config.ProgressColumn(
                         "percentile", min_value=0, max_value=100, format="%d")})
    with right:
        st.subheader("Sprints and HSR on the pitch")
        e = efforts[efforts.player_id == pid]
        kinds = st.multiselect("Effort type", ["sprint", "hsr"], default=["sprint"])
        ctx = st.multiselect("Context", list(CONTEXT_COLORS), default=list(CONTEXT_COLORS))
        sel = e[e.kind.isin(kinds) & e.context.isin(ctx)]
        st.pyplot(pitch(sel))
        st.caption(f"{len(sel)} efforts. Colour = context: " +
                   ", ".join(f"{k}" for k in CONTEXT_COLORS))
        share = sel.context.value_counts(normalize=True).mul(100).round(0)
        st.bar_chart(share.rename("% of efforts"), horizontal=True)

    st.subheader("Match by match")
    cols = st.multiselect("Stats", list(METRICS), default=DEFAULT, format_func=lambda c: METRICS[c][0],
                          key="player_stats")
    p90 = st.toggle("Per 90 minutes", value=False, key="player_p90")
    mm = per90(by_match[by_match.player_id == pid], cols, p90).merge(
        matches[["match_id", "label"]], on="match_id").sort_values("label")
    st.dataframe(mm[["label", "minutes_official"] + cols].rename(columns=labels(cols, p90))
                 .round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Match
with tab_match:
    mlabels = {int(k): v for k, v in zip(matches.match_id, matches.label)}
    mid = st.selectbox("Match", list(mlabels), format_func=lambda i: mlabels[int(i)],
                       index=len(mlabels) - 1)
    cols = st.multiselect("Stats", list(METRICS), default=DEFAULT, format_func=lambda c: METRICS[c][0],
                          key="match_stats")
    p90 = st.toggle("Per 90 minutes", value=False, key="match_p90")
    m = per90(by_match[(by_match.match_id == mid) & (by_match.position != "Other")], cols, p90)
    for team, g in m.groupby("team"):
        st.subheader(team)
        st.dataframe(g[["name", "position", "minutes_official"] + cols]
                     .sort_values(cols[0] if cols else "minutes_official", ascending=False)
                     .rename(columns=labels(cols, p90)).round(1),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- Compare
with tab_compare:
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", list(METRICS), index=list(METRICS).index("sprint_count"),
                     format_func=lambda c: METRICS[c][0])
    y = c2.selectbox("Y axis", list(METRICS), index=list(METRICS).index("runs_targeted"),
                     format_func=lambda c: METRICS[c][0])
    k = c3.slider("Minimum matches", 1, int(pooled.n_matches.max()), 3)
    d = per90(pooled[(pooled.n_matches >= k) & (pooled.position != "Other")], list({x, y}))
    lab = labels([x, y])
    shown = list(dict.fromkeys([x, y]))
    d = d[["name", "team", "position", "n_matches"] + shown].rename(columns=lab)
    st.scatter_chart(d, x=lab[x], y=lab[y], color="position", height=520)
    st.caption(f"{len(d)} players. Volumes per 90 minutes. Hover a point for its values; "
               "the table below names them.")
    st.dataframe(d.round(2).sort_values(lab[y], ascending=False), hide_index=True, width="stretch")
