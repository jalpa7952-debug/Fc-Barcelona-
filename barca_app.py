import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===========================================================================
# SCOUTING ENGINE (merged from scouting_engine.py)
# ===========================================================================
POSITION_WEIGHTS = {
    "GK": {
        "gk_diving": 0.25, "gk_handling": 0.20, "gk_reflexes": 0.25,
        "gk_positioning": 0.15, "gk_kicking": 0.15,
    },
    "CB": {
        "standing_tackle": 0.20, "sliding_tackle": 0.10, "interceptions": 0.15,
        "def_awareness": 0.15, "heading_accuracy": 0.15, "strength": 0.15,
        "jumping": 0.10,
    },
    "LB": {
        "sprint_speed": 0.15, "stamina": 0.15, "crossing": 0.15,
        "standing_tackle": 0.15, "interceptions": 0.15, "def_awareness": 0.15,
        "acceleration": 0.10,
    },
    "RB": {
        "sprint_speed": 0.15, "stamina": 0.15, "crossing": 0.15,
        "standing_tackle": 0.15, "interceptions": 0.15, "def_awareness": 0.15,
        "acceleration": 0.10,
    },
    "CDM": {
        "interceptions": 0.20, "def_awareness": 0.15, "standing_tackle": 0.15,
        "short_passing": 0.15, "stamina": 0.15, "strength": 0.10, "vision": 0.10,
    },
    "CM": {
        "short_passing": 0.20, "vision": 0.15, "long_passing": 0.10,
        "stamina": 0.15, "ball_control": 0.15, "dribbling": 0.10, "composure": 0.15,
    },
    "CAM": {
        "vision": 0.20, "dribbling": 0.15, "short_passing": 0.15,
        "composure": 0.15, "ball_control": 0.15, "finishing": 0.10, "agility": 0.10,
    },
    "LW": {
        "sprint_speed": 0.15, "acceleration": 0.15, "dribbling": 0.20,
        "agility": 0.15, "crossing": 0.10, "finishing": 0.15, "curve": 0.10,
    },
    "RW": {
        "sprint_speed": 0.15, "acceleration": 0.15, "dribbling": 0.20,
        "agility": 0.15, "crossing": 0.10, "finishing": 0.15, "curve": 0.10,
    },
    "ST": {
        "finishing": 0.25, "positioning": 0.20, "shot_power": 0.15,
        "heading_accuracy": 0.10, "composure": 0.15, "volleys": 0.10, "reactions": 0.05,
    },
}

ALL_POSITIONS = list(POSITION_WEIGHTS.keys())


def load_data(csv_path="players.csv"):
    return pd.read_csv(csv_path)


def _get_weighted_matrix(df, position):
    """Return scaled, weighted attribute matrix for a given position template."""
    weights = POSITION_WEIGHTS[position]
    cols = list(weights.keys())
    cols = [c for c in cols if c in df.columns]

    sub = df[cols].fillna(0)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(sub)

    weight_arr = np.array([weights[c] for c in cols])
    weighted = scaled * weight_arr
    return weighted, cols


def role_fit_ranking(df, position, top_n=10):
    """Mode B: rank ALL players at a position by weighted attribute score (no reference)."""
    pool = df[df["position"] == position].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame()

    weighted, cols = _get_weighted_matrix(pool, position)
    pool["fit_score"] = weighted.sum(axis=1) / np.array(list(POSITION_WEIGHTS[position].values())).sum()
    pool["fit_score"] = (pool["fit_score"] * 100).round(1)

    pool = pool.sort_values("fit_score", ascending=False).head(top_n)
    return pool


def style_match(df, reference_name, position=None, top_n=10):
    """Mode A: find players most similar to a reference player using weighted cosine similarity."""
    ref_rows = df[df["name"].str.lower() == reference_name.lower()]
    if ref_rows.empty:
        return pd.DataFrame(), None

    ref_player = ref_rows.iloc[0]
    pos = position or ref_player["position"]

    pool = df[df["position"] == pos].reset_index(drop=True)
    if pool.empty:
        return pd.DataFrame(), ref_player

    weighted, cols = _get_weighted_matrix(pool, pos)

    ref_idx = pool[pool["name"].str.lower() == reference_name.lower()].index
    if len(ref_idx) == 0:
        pool = pd.concat([pool, ref_rows], ignore_index=True)
        weighted, cols = _get_weighted_matrix(pool, pos)
        ref_idx = pool[pool["name"].str.lower() == reference_name.lower()].index

    ref_vec = weighted[ref_idx[0]].reshape(1, -1)
    sims = cosine_similarity(ref_vec, weighted)[0]

    pool["similarity"] = (sims * 100).round(1)
    pool = pool[pool["name"].str.lower() != reference_name.lower()]
    pool = pool.sort_values("similarity", ascending=False).head(top_n)

    return pool, ref_player


# ===========================================================================
# RADAR CHART HELPER
# ===========================================================================
RADAR_ATTRS = ["pac", "sho", "pas", "dri", "def", "phy"]
RADAR_LABELS = ["Pace", "Shooting", "Passing", "Dribbling", "Defending", "Physical"]


def plot_radar_comparison(players_df, names):
    """Draw a matplotlib radar chart comparing 2+ players on PAC/SHO/PAS/DRI/DEF/PHY."""
    angles = np.linspace(0, 2 * np.pi, len(RADAR_ATTRS), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#04124a")
    ax.set_facecolor("#0d1b3e")

    colors = ["#ffd1dc", "#a50044", "#5dcaa5", "#f0997b"]

    for i, name in enumerate(names):
        row = players_df[players_df["name"] == name].iloc[0]
        values = [row[a] for a in RADAR_ATTRS]
        values += values[:1]
        color = colors[i % len(colors)]
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, color="white", size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#9aa9d6", size=8)
    ax.tick_params(colors="white")
    ax.spines["polar"].set_color("#2a3a6b")
    ax.grid(color="#2a3a6b", alpha=0.5)

    legend = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                        facecolor="#0d1b3e", edgecolor="#2a3a6b", labelcolor="white", fontsize=9)
    fig.tight_layout()
    return fig

st.set_page_config(page_title="FC Barcelona Scouting Tool", page_icon="🔵🔴", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS — navy/maroon Barça theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #04124a 0%, #a50044 100%);
    padding: 28px 32px;
    border-radius: 14px;
    margin-bottom: 24px;
}
.main-header h1 { color: white; margin: 0; font-size: 30px; }
.main-header p { color: #ffd1dc; margin: 4px 0 0 0; font-size: 14px; }

.metric-card {
    background: #0d1b3e;
    border: 1px solid #2a3a6b;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
    color: white;
}
.metric-card .value { font-size: 22px; font-weight: 700; color: #ffd1dc; }
.metric-card .label { font-size: 12px; color: #9aa9d6; }

.player-card {
    background: #0d1b3e;
    border: 1px solid #2a3a6b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: #a50044; color: white;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 16px; flex-shrink: 0;
}
.player-info { flex-grow: 1; }
.player-name { color: white; font-weight: 600; font-size: 16px; }
.player-meta { color: #9aa9d6; font-size: 13px; }
.fit-bar-bg {
    background: #1c2c5c; border-radius: 6px; height: 8px; width: 100%; margin-top: 6px;
}
.fit-bar-fill {
    background: linear-gradient(90deg, #a50044, #ffd1dc);
    height: 8px; border-radius: 6px;
}
.fit-score { color: #ffd1dc; font-weight: 700; font-size: 15px; min-width: 60px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
  <h1>FC Barcelona Scouting Tool</h1>
  <p>Position-weighted player recommendation engine — style matching & role-fit ranking</p>
</div>
""", unsafe_allow_html=True)

df = load_data("players.csv")

if "selected_player" not in st.session_state:
    st.session_state.selected_player = None

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔵🔴 Navigation")
    page = st.radio("Go to", ["🔍 Scout", "🎯 Find Replacement", "⚔️ Compare", "📋 Advanced Explorer"], label_visibility="collapsed")

if page == "📋 Advanced Explorer":
    st.markdown('<div class="main-header"><h1>Advanced Player Explorer</h1>'
                '<p>Filter and inspect every attribute in the dataset</p></div>',
                unsafe_allow_html=True)

    # --- Filter controls covering ALL attribute groups ---
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        club_filter = st.multiselect("Club", sorted(df["club"].unique()))
    with f2:
        pos_filter = st.multiselect("Position", sorted(df["position"].unique()))
    with f3:
        foot_filter = st.multiselect("Preferred foot", sorted(df["preferred_foot"].unique()))
    with f4:
        league_filter = st.multiselect("League", sorted(df["league"].unique()))

    f5, f6 = st.columns(2)
    with f5:
        age_range = st.slider("Age", int(df["age"].min()), int(df["age"].max()),
                               (int(df["age"].min()), int(df["age"].max())))
    with f6:
        overall_range = st.slider("Overall rating", int(df["overall"].min()), int(df["overall"].max()),
                                   (int(df["overall"].min()), int(df["overall"].max())))

    st.markdown("**Filter by any sub-attribute**")
    sub_attrs = ["pac", "sho", "pas", "dri", "def", "phy",
                 "acceleration", "sprint_speed", "positioning", "finishing", "shot_power",
                 "long_shots", "volleys", "penalties", "vision", "crossing",
                 "free_kick_accuracy", "short_passing", "long_passing", "curve", "agility",
                 "balance", "reactions", "ball_control", "dribbling", "composure",
                 "interceptions", "heading_accuracy", "def_awareness", "standing_tackle",
                 "sliding_tackle", "jumping", "stamina", "strength", "aggression", "skill_moves"]
    chosen_attr = st.selectbox("Attribute", ["(none)"] + sub_attrs)
    attr_min = 0
    if chosen_attr != "(none)":
        attr_min = st.slider(f"Minimum {chosen_attr.replace('_',' ').title()}", 0, 99, 0)

    # --- Apply filters ---
    filtered = df.copy()
    if club_filter:
        filtered = filtered[filtered["club"].isin(club_filter)]
    if pos_filter:
        filtered = filtered[filtered["position"].isin(pos_filter)]
    if foot_filter:
        filtered = filtered[filtered["preferred_foot"].isin(foot_filter)]
    if league_filter:
        filtered = filtered[filtered["league"].isin(league_filter)]
    filtered = filtered[filtered["age"].between(*age_range)]
    filtered = filtered[filtered["overall"].between(*overall_range)]
    if chosen_attr != "(none)" and attr_min > 0:
        filtered = filtered[filtered[chosen_attr] >= attr_min]

    search_text = st.text_input("🔎 Search by player name")
    if search_text:
        filtered = filtered[filtered["name"].str.contains(search_text, case=False, na=False)]

    st.caption(f"{len(filtered)} players match your filters")

    display_cols = ["name", "club", "position", "alternative_positions", "age", "overall",
                     "pac", "sho", "pas", "dri", "def", "phy",
                     "height", "weight", "preferred_foot", "weak_foot",
                     "att_work_rate", "def_work_rate", "skill_moves", "league"]
    st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True, height=450)

    with st.expander("📊 Show ALL attributes (full sub-attribute table)"):
        full_cols = ["name"] + sub_attrs + ["height", "weight", "preferred_foot", "weak_foot"]
        st.dataframe(filtered[full_cols].reset_index(drop=True), use_container_width=True, height=450)

    csv = filtered[display_cols].to_csv(index=False)
    st.download_button("⬇️ Download filtered CSV", data=csv,
                        file_name="filtered_players.csv", mime="text/csv")

    st.stop()  # don't render the Scout page below

if page == "🎯 Find Replacement":
    st.markdown('<div class="main-header"><h1>Find a Replacement</h1>'
                '<p>Pick the exact attribute(s) your team is weak on — get the best-fitting players</p></div>',
                unsafe_allow_html=True)

    ALL_ATTRIBUTES = [
        "pac", "sho", "pas", "dri", "def", "phy",
        "acceleration", "sprint_speed", "positioning", "finishing", "shot_power",
        "long_shots", "volleys", "penalties", "vision", "crossing",
        "free_kick_accuracy", "short_passing", "long_passing", "curve", "agility",
        "balance", "reactions", "ball_control", "dribbling", "composure",
        "interceptions", "heading_accuracy", "def_awareness", "standing_tackle",
        "sliding_tackle", "jumping", "stamina", "strength", "aggression", "skill_moves",
        "overall", "age",
    ]
    ATTR_LABELS = {a: a.replace("_", " ").title() for a in ALL_ATTRIBUTES}

    st.markdown("**Select the attribute(s) your team needs** (e.g. select only *Penalties* if your penalty taking is weak)")
    chosen = st.multiselect(
        "Attributes",
        options=ALL_ATTRIBUTES,
        format_func=lambda a: ATTR_LABELS[a],
        label_visibility="collapsed",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        pos_filter = st.multiselect("Optional: limit to position(s)", ALL_POSITIONS)
    with col_b:
        top_n = st.slider("Show top N players", 3, 20, 8)

    if not chosen:
        st.info("Tick at least one attribute above to see rankings.")
    else:
        pool = df.copy()
        if pos_filter:
            pool = pool[pool["position"].isin(pos_filter)]

        pool["match_score"] = pool[chosen].mean(axis=1).round(1)
        pool = pool.sort_values("match_score", ascending=False).head(top_n)

        st.markdown(f"### Top players for: {', '.join(ATTR_LABELS[c] for c in chosen)}")

        # --- Matplotlib bar chart: overall match score per player ---
        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(pool))))
        fig.patch.set_facecolor("#04124a")
        ax.set_facecolor("#0d1b3e")
        bars = ax.barh(pool["name"], pool["match_score"], color="#a50044")
        ax.invert_yaxis()
        ax.set_xlabel("Average score on selected attributes", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#2a3a6b")
        ax.bar_label(bars, fmt="%.1f", color="white", padding=3)
        fig.tight_layout()
        st.pyplot(fig)

        # --- Matplotlib grouped bar chart: per-attribute breakdown for top 5 ---
        if len(chosen) > 1:
            top5 = pool.head(5)
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            fig2.patch.set_facecolor("#04124a")
            ax2.set_facecolor("#0d1b3e")
            x = np.arange(len(chosen))
            width = 0.8 / max(len(top5), 1)
            palette = ["#ffd1dc", "#a50044", "#5dcaa5", "#f0997b", "#85b7eb"]
            for i, (_, row) in enumerate(top5.iterrows()):
                values = [row[c] for c in chosen]
                ax2.bar(x + i * width, values, width=width, label=row["name"], color=palette[i % len(palette)])
            ax2.set_xticks(x + width * (len(top5) - 1) / 2)
            ax2.set_xticklabels([ATTR_LABELS[c] for c in chosen], color="white", rotation=20, ha="right")
            ax2.tick_params(colors="white")
            for spine in ax2.spines.values():
                spine.set_color("#2a3a6b")
            ax2.set_ylabel("Attribute value", color="white")
            legend = ax2.legend(facecolor="#0d1b3e", edgecolor="#2a3a6b", labelcolor="white", fontsize=9)
            fig2.tight_layout()
            st.pyplot(fig2)

        for _, row in pool.iterrows():
            initials = "".join([w[0] for w in row["name"].split()[:2]]).upper()
            score = row["match_score"]
            attr_chips = " ".join(
                f"<span style='background:#1c2c5c;color:#ffd1dc;border-radius:6px;padding:2px 8px;"
                f"font-size:11px;margin-right:4px;'>{ATTR_LABELS[c]}: {row[c]}</span>"
                for c in chosen
            )
            st.markdown(f"""
            <div class="player-card">
                <div class="avatar">{initials}</div>
                <div class="player-info">
                    <div class="player-name">{row['name']}</div>
                    <div class="player-meta">{row['club']} · {row['position']} · Age {row['age']} · OVR {row['overall']}</div>
                    <div style="margin-top:6px;">{attr_chips}</div>
                </div>
                <div class="fit-score">{score}</div>
            </div>
            """, unsafe_allow_html=True)

    st.stop()

if page == "⚔️ Compare":
    st.markdown('<div class="main-header"><h1>Compare Players</h1>'
                '<p>Side-by-side radar chart + full attribute comparison</p></div>',
                unsafe_allow_html=True)

    compare_names = st.multiselect(
        "Select 2-4 players to compare",
        sorted(df["name"].unique()),
        max_selections=4,
        default=sorted(df["name"].unique())[:2] if len(df) >= 2 else None,
    )

    if len(compare_names) >= 2:
        fig = plot_radar_comparison(df, compare_names)
        st.pyplot(fig)

        st.markdown("### Full attribute comparison")
        compare_rows = df[df["name"].isin(compare_names)].set_index("name")

        all_attr_groups = {
            "Core ratings": ["overall", "pac", "sho", "pas", "dri", "def", "phy"],
            "Pace & physical": ["acceleration", "sprint_speed", "stamina", "strength", "jumping", "aggression"],
            "Shooting": ["positioning", "finishing", "shot_power", "long_shots", "volleys", "penalties"],
            "Passing & vision": ["vision", "crossing", "free_kick_accuracy", "short_passing", "long_passing", "curve"],
            "Dribbling": ["agility", "balance", "reactions", "ball_control", "dribbling", "composure"],
            "Defending": ["interceptions", "heading_accuracy", "def_awareness", "standing_tackle", "sliding_tackle"],
            "Bio & profile": ["age", "height", "weight", "preferred_foot", "weak_foot", "skill_moves",
                               "att_work_rate", "def_work_rate", "club", "league", "position"],
        }

        tabs = st.tabs(list(all_attr_groups.keys()))
        for tab, (group_name, cols) in zip(tabs, all_attr_groups.items()):
            with tab:
                cols = [c for c in cols if c in compare_rows.columns]
                st.table(compare_rows[cols].T)
    else:
        st.info("Select at least 2 players to compare.")

    st.stop()

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
for col, (label, value) in zip(
    [c1, c2, c3, c4],
    [
        ("Players in DB", len(df)),
        ("Clubs covered", df["club"].nunique()),
        ("Positions", df["position"].nunique()),
        ("Avg. Overall", round(df["overall"].mean(), 1)),
    ],
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{value}</div>
            <div class="label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
mode = st.radio("Scouting mode", ["Style Match", "Role-Fit Ranking"], horizontal=True)

st.write("")

results = pd.DataFrame()
ref_player = None

if mode == "Style Match":
    colA, colB, colC = st.columns([2, 1, 1])
    with colA:
        ref_name = st.selectbox("Reference player", sorted(df["name"].unique()))
    with colB:
        pos_override = st.selectbox("Search position pool", ["(same as reference)"] + ALL_POSITIONS)
    with colC:
        top_n = st.slider("Results", 3, 15, 5)

    if st.button("Find similar players", type="primary"):
        pos = None if pos_override == "(same as reference)" else pos_override
        results, ref_player = style_match(df, ref_name, position=pos, top_n=top_n)

else:
    colA, colB = st.columns([2, 1])
    with colA:
        position = st.selectbox("Position", ALL_POSITIONS)
    with colB:
        top_n = st.slider("Results", 3, 15, 5)

    if st.button("Rank players", type="primary"):
        results = role_fit_ranking(df, position, top_n=top_n)
        results = results.rename(columns={"fit_score": "similarity"})  # unify display column

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
score_col = "similarity" if "similarity" in results.columns else "fit_score"

if not results.empty:
    if ref_player is not None:
        st.caption(f"Reference: **{ref_player['name']}** ({ref_player['club']}, {ref_player['position']})")

    for _, row in results.iterrows():
        initials = "".join([w[0] for w in row["name"].split()[:2]]).upper()
        score = row[score_col]

        card_col, btn_col = st.columns([5, 1])
        with card_col:
            st.markdown(f"""
            <div class="player-card">
                <div class="avatar">{initials}</div>
                <div class="player-info">
                    <div class="player-name">{row['name']}</div>
                    <div class="player-meta">{row['club']} · {row['position']} · Age {row['age']} · OVR {row['overall']}</div>
                    <div class="fit-bar-bg"><div class="fit-bar-fill" style="width:{min(score,100)}%;"></div></div>
                </div>
                <div class="fit-score">{score}%</div>
            </div>
            """, unsafe_allow_html=True)
        with btn_col:
            if st.button("View profile", key=f"view_{row['name']}"):
                st.session_state.selected_player = row["name"]

elif st.session_state.get("_searched"):
    st.info("No results found.")

# ---------------------------------------------------------------------------
# Player profile detail panel
# ---------------------------------------------------------------------------
if st.session_state.selected_player:
    p = df[df["name"] == st.session_state.selected_player]
    if not p.empty:
        p = p.iloc[0]
        st.divider()
        st.subheader(f"📋 {p['name']} — Full Profile")

        close_col, _ = st.columns([1, 6])
        with close_col:
            if st.button("✕ Close profile"):
                st.session_state.selected_player = None
                st.rerun()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        for col, stat in zip([m1, m2, m3, m4, m5, m6], ["pac", "sho", "pas", "dri", "def", "phy"]):
            with col:
                st.metric(stat.upper(), int(p[stat]))

        bio_col, club_col = st.columns(2)
        with bio_col:
            st.markdown("**Bio**")
            st.write(f"Nation: {p['nation']}")
            st.write(f"Age: {p['age']}  |  Height: {p['height']}  |  Weight: {p['weight']}")
            st.write(f"Preferred foot: {p['preferred_foot']}  |  Weak foot: {p['weak_foot']}★")
            st.write(f"Skill moves: {p['skill_moves']}★")
        with club_col:
            st.markdown("**Club / League**")
            st.write(f"Club: {p['club']}")
            st.write(f"League: {p['league']}")
            st.write(f"Position: {p['position']} (alt: {p['alternative_positions']})")
            st.write(f"Work rates: {p['att_work_rate']} / {p['def_work_rate']}")
            if pd.notna(p.get("url")):
                st.markdown(f"[EA Ratings page]({p['url']})")

        if p["position"] == "GK":
            st.markdown("**Goalkeeping attributes**")
            gk_cols = ["gk_diving", "gk_handling", "gk_kicking", "gk_positioning", "gk_reflexes"]
            st.table(p[gk_cols].rename(lambda c: c.replace("gk_", "").replace("_", " ").title()))
        else:
            groups = {
                "Pace & physical": ["acceleration", "sprint_speed", "stamina", "strength", "jumping", "aggression"],
                "Shooting": ["positioning", "finishing", "shot_power", "long_shots", "volleys", "penalties"],
                "Passing & vision": ["vision", "crossing", "free_kick_accuracy", "short_passing", "long_passing", "curve"],
                "Dribbling": ["agility", "balance", "reactions", "ball_control", "dribbling", "composure"],
                "Defending": ["interceptions", "heading_accuracy", "def_awareness", "standing_tackle", "sliding_tackle"],
            }
            tabs = st.tabs(list(groups.keys()))
            for tab, (group_name, cols) in zip(tabs, groups.items()):
                with tab:
                    cols = [c for c in cols if c in p.index]
                    table = pd.DataFrame({"Attribute": [c.replace("_", " ").title() for c in cols],
                                           "Value": [int(p[c]) for c in cols]})
                    st.table(table.set_index("Attribute"))
