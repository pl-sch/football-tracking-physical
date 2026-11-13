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
import plotly.graph_objects as go
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
    "shots_on_target": ("Shots on target", "volume"),
    "xg": ("Expected goals (xG)", "volume"),
    "duels": ("Duels", "volume"),
    "duel_win": ("Duels won (%)", "rate"),
    "aerial_duels": ("Aerial duels (headers)", "volume"),
    "aerial_win": ("Aerial duels won (%)", "rate"),
    "tackles_won": ("Tackles won", "volume"),
}
# Which stats each data source provides (others are left empty, not zero)
SOURCE_ONLY = {
    "A-League": {"off_ball_runs", "runs_targeted", "defensive_engagements", "possession_losses"},
    "Bundesliga": {"shots_on_target", "xg", "duels", "duel_win", "aerial_duels", "aerial_win",
                   "tackles_won"},
}
DEFAULT = ["total_distance_m", "hsr_distance_m", "sprint_count", "top_speed_kmh", "passes",
           "pass_accuracy", "regains"]
CONTEXT_COLORS = {
    "with the ball": "#E69F00", "run in behind": "#D55E00", "attacking the box": "#CC79A7",
    "width / overlap": "#F0E442", "support / coming short": "#56B4E9", "pressing": "#0072B2",
    "no linked event": "#BBBBBB",
}
PHASE_COLORS = {"in possession": "#D55E00", "out of possession": "#0072B2", "no possession": "#BBBBBB"}


@st.cache_data
def load():
    by_match = pd.read_csv(ALL / "players_by_match.csv", dtype={"match_id": str}, low_memory=False)
    pooled = pd.read_csv(ALL / "players_pooled.csv", low_memory=False)
    matches = pd.read_csv(ALL / "matches.csv", dtype={"match_id": str})
    matches["label"] = matches.date + "  " + matches.home + " - " + matches.away
    efforts = pd.concat([pd.read_csv(f).assign(match_id=f.parent.name)
                         for f in OUT.glob("*/efforts.csv") if f.parent.name != "all_matches"],
                        ignore_index=True)
    for df in (by_match, pooled):
        df["source"] = np.where(df.league.str.startswith("A-League"), "A-League", "Bundesliga")
        if "duels" in df:
            df["duel_win"] = 100 * df.duels_won / df.duels.replace(0, np.nan)
            df["aerial_win"] = 100 * df.aerial_duels_won / df.aerial_duels.replace(0, np.nan)
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


MIN_MINUTES = {"A-League": 180, "Bundesliga": 90}  # fewer Bundesliga matches, lower bar
SIM_BASE = ["total_distance_m", "hsr_distance_m", "sprint_distance_m", "sprint_count",
            "high_accel_count", "top_speed_kmh", "passes", "pass_accuracy", "shots", "regains"]
SIM_EXTRA = {"A-League": ["off_ball_runs", "runs_targeted", "defensive_engagements"],
             "Bundesliga": ["duels", "aerial_duels", "tackles_won", "xg"]}


def similar_players(pool: pd.DataFrame, pid: int, feats: list[str], min_minutes: float,
                    same_position: bool, n: int = 5) -> pd.DataFrame:
    """Cosine similarity on per-90 stats standardised within the league (z-scores).
    100 = identical profile, 50 = unrelated, 0 = opposite."""
    me = pool[pool.player_id == pid].iloc[0]
    cand = pool[(pool.position != "Other") &
                ((pool.minutes_official >= min_minutes) | (pool.player_id == pid))]
    if same_position:
        cand = cand[cand.position == me.position]
    X = per90(cand, feats).set_index("player_id")[feats].astype(float)
    X = X.fillna(X.median())
    Z = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    z = Z.loc[pid]
    cos = Z.drop(index=pid).dot(z) / (np.linalg.norm(Z.drop(index=pid), axis=1) * np.linalg.norm(z) + 1e-9)
    out = cand.set_index("player_id").loc[cos.index, ["name", "team", "position", "n_matches"]]
    out["similarity"] = (50 * (1 + cos)).round(0)
    return out.sort_values("similarity", ascending=False).head(n).reset_index()


def pitch_shapes() -> list[dict]:
    line = dict(color="white", width=1.5)
    L, W = 52.5, 34
    shapes = [dict(type="rect", x0=-L, y0=-W, x1=L, y1=W, line=line),
              dict(type="line", x0=0, y0=-W, x1=0, y1=W, line=line),
              dict(type="circle", x0=-9.15, y0=-9.15, x1=9.15, y1=9.15, line=line)]
    for s in (-1, 1):
        shapes += [dict(type="rect", x0=s * L, y0=-20.16, x1=s * (L - 16.5), y1=20.16, line=line),
                   dict(type="rect", x0=s * L, y0=-9.16, x1=s * (L - 5.5), y1=9.16, line=line),
                   dict(type="rect", x0=s * L, y0=-3.66, x1=s * (L + 2), y1=3.66, line=line)]
    return shapes


def replay_figure(clip: pd.DataFrame) -> go.Figure:
    style = {"scoring team": dict(color="#D55E00", size=17),
             "defending team": dict(color="#0072B2", size=17),
             "ball": dict(color="white", size=10, line=dict(color="black", width=1.5))}
    times = sorted(clip.t.unique())

    def traces(t):
        f = clip[clip.t == t]
        out = []
        for kind, st_ in style.items():
            g = f[f.kind == kind]
            out.append(go.Scatter(x=g.x, y=g.y, mode="markers+text" if kind != "ball" else "markers",
                                  text=g.label, textfont=dict(color="white", size=9),
                                  hovertext=g.name, hoverinfo="text", marker=st_, name=kind))
        return out

    fig = go.Figure(data=traces(times[0]),
                    frames=[go.Frame(data=traces(t), name=f"{t:.1f}") for t in times])
    fig.update_layout(
        shapes=pitch_shapes(), plot_bgcolor="#3a7d44", paper_bgcolor="rgba(0,0,0,0)", height=560,
        xaxis=dict(range=[-56, 56], visible=False),
        yaxis=dict(range=[-36, 36], visible=False, scaleanchor="x"),
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.02),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.08, showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=False), fromcurrent=True,
                                  transition=dict(duration=0))]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(currentvalue=dict(prefix="seconds to goal: "), pad=dict(t=30),
                      steps=[dict(method="animate", label=f"{t:.1f}",
                                  args=[[f"{t:.1f}"], dict(mode="immediate",
                                                          frame=dict(duration=0, redraw=False))])
                             for t in times])])
    return fig


def available(source: str) -> list[str]:
    """Stats recorded by this data source."""
    other = [k for k in SOURCE_ONLY if k != source]
    banned = set().union(*(SOURCE_ONLY[k] for k in other))
    return [c for c in METRICS if c not in banned]


def pitch(efforts: pd.DataFrame, colors: dict, column: str):
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
                                    color=colors.get(r[column], "#888")))
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
st.caption("Physical metrics computed from raw player positions. "
           "A-League 2024/25: 20 matches, SkillCorner broadcast tracking, tactical events. "
           "Bundesliga 2022/23: 7 matches, DFL optical tracking, duels and xG. "
           "[Code and method](https://github.com/pl-sch/football-tracking-physical)")

source = st.radio("Data", ["A-League", "Bundesliga"], horizontal=True,
                  help="The two leagues come from different providers and record different stats: "
                       "they are kept apart rather than mixed in one ranking.")
STATS = available(source)
DEF = [c for c in DEFAULT + (["duels", "aerial_duels"] if source == "Bundesliga" else []) if c in STATS]
pool_s = pooled[pooled.source == source]
bym_s = by_match[by_match.source == source]
matches_s = matches[matches.league.str.startswith(source) | (
    (source == "Bundesliga") & matches.league.str.contains("Bundesliga"))]

names = ["Player", "Match", "Compare players"] + (["Goal replays"] if source == "Bundesliga" else [])
tabs = st.tabs(names)
tab_player, tab_match, tab_compare = tabs[:3]

# ---------------------------------------------------------------- Player
with tab_player:
    p = pool_s[pool_s.position != "Other"].sort_values("name")
    opts = {int(k): v for k, v in zip(p.player_id, p.name + " (" + p.team + ", "
                                      + p.position.fillna("?") + ")")}
    default = int(p.sort_values("sprint_count").player_id.iloc[-1])
    pid = st.selectbox("Search a player (type a name)", list(opts), format_func=lambda i: opts[int(i)],
                       index=list(opts).index(default), key=f"player_{source}",
                       help="Click the box and start typing: the list filters as you type.")
    me = pool_s[pool_s.player_id == pid].iloc[0]
    outfield = pool_s[(pool_s.position != "Other") & (pool_s.minutes_official >= MIN_MINUTES[source])]
    same_pos = outfield[outfield.position == me.position]
    if len(same_pos) >= 8:
        peers, who = same_pos, f"{me.position}s"
        peer_note = ""
    else:
        peers, who = outfield, "all outfield players"
        peer_note = (f"Only {len(same_pos)} {me.position}s in this sample, too few for percentiles: "
                     "compared with all outfield players instead.")
    peers = per90(peers, STATS)
    mine = per90(pool_s[pool_s.player_id == pid], STATS).iloc[0]

    c = st.columns(5)
    c[0].metric("Matches", int(me.n_matches))
    c[1].metric("Minutes", f"{me.minutes_official:.0f}")
    c[2].metric("Distance /90", f"{mine.total_distance_m / 1000:.1f} km")
    c[3].metric("Sprints /90", f"{mine.sprint_count:.1f}")
    c[4].metric("Top speed", f"{me.top_speed_kmh:.1f} km/h")

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Percentile among {who} ({source})")
        st.caption(f"{len(peers)} players with {MIN_MINUTES[source]}+ minutes. Volumes per 90 minutes. "
                   + peer_note)
        rows = []
        for col in STATS:
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
        if source == "A-League":
            colors, column, what = CONTEXT_COLORS, "context", "what the sprint was for"
        else:
            colors, column, what = PHASE_COLORS, "team_phase", "ball possession at the start"
        pick = st.multiselect(f"Colour = {what}", list(colors), default=list(colors))
        sel = e[e.kind.isin(kinds) & e[column].isin(pick)]
        st.pyplot(pitch(sel, colors, column))
        st.caption(f"{len(sel)} efforts, all matches, team always attacking left to right.")
        share = sel[column].value_counts(normalize=True).mul(100).round(0)
        st.bar_chart(share.rename("% of efforts"), horizontal=True)

    st.subheader("Similar profiles")
    feats = SIM_BASE + SIM_EXTRA[source]
    same = st.toggle("Same position only", value=len(same_pos) >= 8, key=f"same_{source}")
    sim = similar_players(pool_s, pid, feats, MIN_MINUTES[source], same)
    st.caption("Closest players on " + ", ".join(METRICS[f][0].lower() for f in feats)
               + " (per 90, standardised within the league). 100 = same profile, 50 = unrelated.")
    show = per90(pool_s[pool_s.player_id.isin([pid] + sim.player_id.tolist())], feats)
    show = show.set_index("player_id").loc[[pid] + sim.player_id.tolist()]
    show.insert(0, "similarity", [None] + sim.similarity.tolist())
    show = show[["name", "team", "position", "similarity"] + feats].rename(columns=labels(feats))
    st.dataframe(show.round(1), hide_index=True, width="stretch",
                 column_config={"similarity": st.column_config.ProgressColumn(
                     "similarity", min_value=0, max_value=100, format="%d")})

    st.subheader("Match by match")
    cols = st.multiselect("Stats", STATS, default=DEF, format_func=lambda c: METRICS[c][0],
                          key=f"player_stats_{source}")
    p90 = st.toggle("Per 90 minutes", value=False, key="player_p90")
    mm = per90(bym_s[bym_s.player_id == pid], cols, p90).merge(
        matches_s[["match_id", "label"]], on="match_id").sort_values("label")
    st.dataframe(mm[["label", "minutes_official"] + cols].rename(columns=labels(cols, p90))
                 .round(1), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Match
with tab_match:
    mlabels = dict(zip(matches_s.match_id, matches_s.label))
    mid = st.selectbox("Match", list(mlabels), format_func=lambda i: mlabels[str(i)],
                       index=len(mlabels) - 1, key=f"match_{source}")
    cols = st.multiselect("Stats", STATS, default=DEF, format_func=lambda c: METRICS[c][0],
                          key=f"match_stats_{source}")
    p90 = st.toggle("Per 90 minutes", value=False, key="match_p90")
    m = per90(bym_s[(bym_s.match_id == mid) & (bym_s.position != "Other")], cols, p90)
    for team, g in m.groupby("team"):
        st.subheader(team)
        st.dataframe(g[["name", "position", "minutes_official"] + cols]
                     .sort_values(cols[0] if cols else "minutes_official", ascending=False)
                     .rename(columns=labels(cols, p90)).round(1),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- Compare
with tab_compare:
    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X axis", STATS, index=STATS.index("sprint_count"),
                     format_func=lambda c: METRICS[c][0], key=f"x_{source}")
    y_default = "runs_targeted" if source == "A-League" else "aerial_duels"
    y = c2.selectbox("Y axis", STATS, index=STATS.index(y_default),
                     format_func=lambda c: METRICS[c][0], key=f"y_{source}")
    k = c3.slider("Minimum matches", 1, int(pool_s.n_matches.max()), 2 if source == "Bundesliga" else 3,
                  key=f"k_{source}")
    d = per90(pool_s[(pool_s.n_matches >= k) & (pool_s.position != "Other")], list({x, y}))
    lab = labels([x, y])
    shown = list(dict.fromkeys([x, y]))
    d = d[["name", "team", "position", "n_matches"] + shown].rename(columns=lab)
    st.scatter_chart(d, x=lab[x], y=lab[y], color="position", height=520)
    st.caption(f"{len(d)} players. Volumes per 90 minutes. Hover a point for its values; "
               "the table below names them.")
    st.dataframe(d.round(2).sort_values(lab[y], ascending=False), hide_index=True, width="stretch")

# ---------------------------------------------------------------- Goal replays (Bundesliga)
if source == "Bundesliga":
    with tabs[3]:
        goals = pd.read_csv(ALL / "goals.csv")
        clips = pd.read_csv(ALL / "goal_clips.csv.gz")
        glab = {r.goal_id: f"{r.date}  {r.match}  ·  {r.minute}'  {r.scorer} ({r.team})  {r.score}"
                for r in goals.itertuples()}
        gid = st.selectbox("Goal", list(glab), format_func=lambda g: glab[g], key="goal")
        g = goals.set_index("goal_id").loc[gid]
        st.plotly_chart(replay_figure(clips[clips.goal_id == gid]), width="stretch")
        note = "" if g.line_crossing_found else " The ball is not tracked over the line here: timing is approximate."
        st.caption(f"Optical tracking, 10 s before the goal to 2 s after. Red: {g.team} (attacking "
                   f"left to right), blue: defenders, white: ball. Numbers are shirt numbers. "
                   f"Shot xG: {g.xg:.2f}.{note}")
