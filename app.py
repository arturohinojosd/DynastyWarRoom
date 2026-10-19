import re
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Birdgang Dynasty War Room",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_LEAGUE_ID = "1313348860042571776"
DEFAULT_GM = "arturohd"


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================
def get_json(url):
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def clean_name(name_str):
    if not isinstance(name_str, str):
        return ""
    s = name_str.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_gm(name):
    if not name:
        return "Unknown"
    if name.strip().lower() == "tittykisser":
        return "omarlmussot"
    return name.strip()


# ==============================================================================
# CACHED DATA PIPELINE (SLEEPER + NFLVERSE + DYNASTYPROCESS)
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_war_room_data(league_id, target_gm, lookback_weeks=3):
    # 1. Detect Current NFL State (Season & Active Week)
    nfl_state = get_json("https://api.sleeper.app/v1/state/nfl") or {}
    curr_season = int(nfl_state.get("season", 2026))
    curr_week = max(1, int(nfl_state.get("display_week") or nfl_state.get("week") or 6))

    # 2. Load Master NFL Players & Global Trending Waiver Adds (24h)
    players_db = get_json("https://api.sleeper.app/v1/players/nfl") or {}
    trending_adds_raw = (
        get_json("https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100")
        or []
    )
    trending_map = {str(item["player_id"]): int(item["count"]) for item in trending_adds_raw}

    # 3. Load League Info, Users, Rosters & Current Matchups
    league_info = get_json(f"https://api.sleeper.app/v1/league/{league_id}") or {}
    roster_positions = league_info.get("roster_positions") or []
    rec_pt = (league_info.get("scoring_settings") or {}).get("rec", 1.0)
    pts_key = "pts_ppr" if rec_pt >= 0.8 else ("pts_half_ppr" if rec_pt >= 0.3 else "pts_std")

    users = get_json(f"https://api.sleeper.app/v1/league/{league_id}/users") or []
    rosters = get_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters") or []
    matchups = get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{curr_week}") or []

    u_map = {u["user_id"]: normalize_gm(u.get("display_name", "User")) for u in users if u.get("user_id")}

    # 4. Load Live DynastyProcess Values
    val_map, rank_map = {}, {}
    try:
        df_val = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv")
        val_col = "value_2qb" if "value_2qb" in df_val.columns else "value_1qb"
        ecr_col = "ecr_2qb" if "ecr_2qb" in df_val.columns else "ecr_1qb"
        df_val["clean"] = df_val["player"].apply(clean_name)
        for _, r in df_val.iterrows():
            cn = r["clean"]
            v = int(r[val_col]) if pd.notnull(r[val_col]) else 0
            rk = int(r[ecr_col]) if pd.notnull(r[ecr_col]) else 999
            if cn not in val_map or v > val_map[cn]:
                val_map[cn] = v
                rank_map[cn] = rk
    except Exception:
        pass

    # 5. Pull Recent Weekly Usage Stats (Snaps, Targets, Carries, Red-Zone, Air Yards)
    # Combines Sleeper Weekly Stats API (exact player_id match) + nflverse WOPR/NGS
    completed_weeks = [w for w in range(max(1, curr_week - lookback_weeks), curr_week)]
    if not completed_weeks:
        completed_weeks = [1]

    weekly_usage = defaultdict(lambda: {
        "wks": 0, "snap_pcts": [], "tgts": [], "carries": [], "rz_touches": [], "air_yds": [], "pts": []
    })

    for w in completed_weeks:
        w_stats = get_json(f"https://api.sleeper.app/v1/stats/nfl/regular/{curr_season}/{w}") or {}
        if isinstance(w_stats, dict):
            for pid, st_d in w_stats.items():
                if not isinstance(st_d, dict):
                    continue
                off_snp = float(st_d.get("off_snp", 0.0) or 0.0)
                tm_snp = float(st_d.get("tm_off_snp", 0.0) or 0.0)
                tgt = float(st_d.get("rec_tgt", 0.0) or 0.0)
                car = float(st_d.get("rush_att", 0.0) or 0.0)
                rz_t = float(st_d.get("rec_rz_tgt", 0.0) or 0.0) + float(st_d.get("rush_rz_att", 0.0) or 0.0)
                air = float(st_d.get("rec_air_yd", 0.0) or 0.0)
                pts = float(st_d.get(pts_key, st_d.get("pts_ppr", 0.0)) or 0.0)

                if off_snp > 0 or tgt > 0 or car > 0 or pts > 0:
                    snap_pct = (off_snp / tm_snp * 100.0) if tm_snp > 0 else 0.0
                    u_rec = weekly_usage[str(pid)]
                    u_rec["wks"] += 1
                    u_rec["snap_pcts"].append(snap_pct)
                    u_rec["tgts"].append(tgt)
                    u_rec["carries"].append(car)
                    u_rec["rz_touches"].append(rz_t)
                    u_rec["air_yds"].append(air)
                    u_rec["pts"].append(pts)

    # 6. Enrich with nflverse WOPR, Target Share & Next Gen Separation
    nflverse_metrics = {}
    for yr in [curr_season, curr_season - 1]:
        try:
            url_ps = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{yr}.csv"
            df_ps = pd.read_csv(url_ps)
            df_ps["clean"] = df_ps["player_display_name"].apply(clean_name)
            max_w = df_ps["week"].max()
            rec_ps = df_ps[df_ps["week"] >= max(1, max_w - 2)]
            grp_ps = rec_ps.groupby("clean").agg(
                tgt_share=("target_share", "mean"),
                wopr=("wopr", "mean"),
            ).reset_index()
            for _, r in grp_ps.iterrows():
                nflverse_metrics[r["clean"]] = {
                    "tgt_share": round(float(r["tgt_share"] or 0.0) * 100, 1),
                    "wopr": round(float(r["wopr"] or 0.0), 2),
                }
            break
        except Exception:
            continue

    ngs_sep_map = {}
    for yr in [curr_season, curr_season - 1]:
        try:
            url_ngs = f"https://github.com/nflverse/nflverse-data/releases/download/nextgen_stats/ngs_{yr}_receiving.csv"
            df_ngs = pd.read_csv(url_ngs)
            df_ngs["clean"] = df_ngs["player_display_name"].apply(clean_name)
            rec_ngs = df_ngs[df_ngs["week"] == 0] if 0 in df_ngs["week"].values else df_ngs
            for _, r in rec_ngs.iterrows():
                ngs_sep_map[r["clean"]] = round(float(r.get("avg_separation", 0.0) or 0.0), 2)
            break
        except Exception:
            continue

    return (
        curr_season,
        curr_week,
        roster_positions,
        players_db,
        u_map,
        rosters,
        matchups,
        trending_map,
        val_map,
        rank_map,
        dict(weekly_usage),
        nflverse_metrics,
        ngs_sep_map,
    )


def build_player_profile(pid, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map):
    p = players_db.get(str(pid), {})
    full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or str(pid)
    pos = p.get("position", "N/A")
    team = p.get("team") or "FA"
    inj = p.get("injury_status") or "Healthy"
    cn = clean_name(full)

    val = val_map.get(cn, 0)
    rnk = rank_map.get(cn, 999)
    trend_adds = trending_map.get(str(pid), 0)

    u = weekly_usage.get(str(pid), {})
    snap_list = u.get("snap_pcts", [])
    avg_snap = round(float(np.mean(snap_list)), 1) if snap_list else 0.0
    snap_delta = round(snap_list[-1] - snap_list[0], 1) if len(snap_list) >= 2 else 0.0

    tgt_g = round(float(np.mean(u.get("tgts", [0.0]))), 1) if u.get("tgts") else 0.0
    car_g = round(float(np.mean(u.get("carries", [0.0]))), 1) if u.get("carries") else 0.0
    rz_g = round(float(np.mean(u.get("rz_touches", [0.0]))), 1) if u.get("rz_touches") else 0.0
    ppg = round(float(np.mean(u.get("pts", [0.0]))), 1) if u.get("pts") else 0.0

    nv = nflverse_metrics.get(cn, {})
    tgt_share = nv.get("tgt_share", round(tgt_g * 3.1, 1))
    wopr = nv.get("wopr", round((tgt_g * 0.045) + (avg_snap * 0.003), 2))
    sep = ngs_sep_map.get(cn, 0.0)

    # Composite Opportunity / Volume Score (0-100 scale for Start/Sit & Waiver ranking)
    if pos in ["WR", "TE"]:
        vol_score = round((avg_snap * 0.35) + (tgt_g * 4.2) + (rz_g * 6.0) + (wopr * 22.0), 1)
    elif pos == "RB":
        vol_score = round((avg_snap * 0.35) + (car_g * 2.2) + (tgt_g * 4.5) + (rz_g * 6.5), 1)
    elif pos == "QB":
        vol_score = round((avg_snap * 0.50) + (ppg * 2.0), 1)
    else:
        vol_score = round(ppg * 3.0, 1)

    return {
        "PID": str(pid),
        "Player": full,
        "Pos": pos,
        "Team": team,
        "Status": inj,
        "Snap_%": avg_snap,
        "Snap_Trend": f"{snap_delta:+.1f}%",
        "Tgt/G": tgt_g,
        "Car/G": car_g,
        "RZ_Touch/G": rz_g,
        "Tgt_Share_%": tgt_share,
        "WOPR": wopr,
        "NGS_Sep": sep,
        "Recent_PPG": ppg,
        "Volume_Score": vol_score,
        "Dynasty_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
        "Dynasty_Val": val,
        "24h_Adds": trend_adds,
    }


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("🦅 Birdgang War Room")
    st.caption("Live Sleeper + nflverse + Gemini AI")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else "",
        type="password",
        help="Get a free API key at https://aistudio.google.com/apikey",
    )
    league_id_input = st.text_input("🏆 Sleeper League ID", value=DEFAULT_LEAGUE_ID)
    gm_input = st.text_input("👤 Your Sleeper Username", value=DEFAULT_GM)
    lookback_wks = st.slider("📊 Usage Lookback Window (Weeks)", min_value=1, max_value=5, value=3)

    if st.button("🔄 Refresh Live Sleeper Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Load Data
with st.spinner("Syncing live Sleeper rosters, nflverse usage & DynastyProcess values..."):
    (
        curr_season,
        curr_week,
        roster_positions,
        players_db,
        u_map,
        rosters,
        matchups,
        trending_map,
        val_map,
        rank_map,
        weekly_usage,
        nflverse_metrics,
        ngs_sep_map,
    ) = fetch_all_war_room_data(league_id_input.strip(), gm_input.strip(), lookback_wks)

# Parse User Roster, Opponent Roster, and All 12 League Teams
rostered_pids = set()
my_starters_pids, my_bench_pids, my_ir_pids, my_taxi_pids = [], [], [], []
my_roster_id = None
all_gm_rosters = {}

for r in rosters:
    rid = r.get("roster_id")
    gm = u_map.get(r.get("owner_id"), f"Roster_{rid}")
    pids = [str(x) for x in (r.get("players") or []) if x and str(x) != "0"]
    starters = [str(x) for x in (r.get("starters") or []) if x and str(x) != "0"]
    ir_list = [str(x) for x in (r.get("reserve") or []) if x and str(x) != "0"]
    taxi_list = [str(x) for x in (r.get("taxi") or []) if x and str(x) != "0"]

    rostered_pids.update(pids)
    all_gm_rosters[gm] = {
        "roster_id": rid,
        "players": pids,
        "starters": starters,
        "wins": (r.get("settings") or {}).get("wins", 0),
        "losses": (r.get("settings") or {}).get("losses", 0),
        "fpts": (r.get("settings") or {}).get("fpts", 0),
    }

    if gm.lower() == gm_input.strip().lower():
        my_roster_id = rid
        my_starters_pids = starters
        my_ir_pids = ir_list
        my_taxi_pids = taxi_list
        my_bench_pids = [p for p in pids if p not in starters and p not in ir_list and p not in taxi_list]

# Find Current Week Opponent
opp_gm_name = "Bye / TBD"
opp_starters_profiles = []
if my_roster_id and matchups:
    my_m = next((m for m in matchups if m.get("roster_id") == my_roster_id), None)
    if my_m and my_m.get("matchup_id"):
        opp_m = next(
            (m for m in matchups if m.get("matchup_id") == my_m.get("matchup_id") and m.get("roster_id") != my_roster_id),
            None,
        )
        if opp_m:
            opp_rid = opp_m.get("roster_id")
            opp_gm_name = next((g for g, d in all_gm_rosters.items() if d["roster_id"] == opp_rid), f"Roster {opp_rid}")
            opp_starters_profiles = [
                build_player_profile(pid, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
                for pid in (opp_m.get("starters") or [])
                if pid and str(pid) != "0"
            ]

my_starters_df = pd.DataFrame([
    build_player_profile(p, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
    for p in my_starters_pids
])
my_bench_df = pd.DataFrame([
    build_player_profile(p, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
    for p in my_bench_pids
])

# Build Top Unrostered Free Agents Pool
unrostered_list = []
for pid, p in players_db.items():
    if str(pid) not in rostered_pids and p.get("position") in ["QB", "RB", "WR", "TE"] and p.get("team"):
        prof = build_player_profile(pid, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
        if prof["Volume_Score"] >= 15.0 or prof["Dynasty_Val"] >= 150 or prof["24h_Adds"] >= 25:
            unrostered_list.append(prof)
df_waivers = pd.DataFrame(unrostered_list)
if not df_waivers.empty:
    df_waivers = df_waivers.sort_values(by=["Volume_Score", "Dynasty_Val"], ascending=False)

# ==============================================================================
# AUTOMATED ALERTS ENGINE (INJURIES, IR SLOTS, & START/SIT VOLUME SWAPS)
# ==============================================================================
injury_alerts = []
volume_swap_alerts = []

if not my_starters_df.empty:
    for _, st_row in my_starters_df.iterrows():
        if st_row["Status"] in ["Out", "Doubtful", "IR", "Questionable"]:
            injury_alerts.append(
                f"🚨 **Starter Injury Alert:** **{st_row['Player']} ({st_row['Pos']})** is currently listed as **{st_row['Status']}**."
            )

if not my_bench_df.empty:
    for _, b_row in my_bench_df.iterrows():
        if b_row["Status"] in ["Out", "IR"]:
            injury_alerts.append(
                f"💡 **Free Roster Spot Alert:** Bench player **{b_row['Player']}** has status **{b_row['Status']}**. Move them to an open **IR slot** to pick up a free agent for $0!"
            )

# Compare Starters vs Bench by Volume_Score (RB, WR, TE)
if not my_starters_df.empty and not my_bench_df.empty:
    for pos in ["RB", "WR", "TE"]:
        st_pos = my_starters_df[my_starters_df["Pos"] == pos]
        bn_pos = my_bench_df[(my_bench_df["Pos"] == pos) & (my_bench_df["Status"] == "Healthy")]
        for _, s_r in st_pos.iterrows():
            for _, b_r in bn_pos.iterrows():
                if b_r["Volume_Score"] >= s_r["Volume_Score"] + 8.0 and b_r["Snap_%"] >= s_r["Snap_%"]:
                    volume_swap_alerts.append(
                        f"⚡ **Volume Swap Recommendation ({pos}):** Bench **{b_r['Player']}** has a higher Opportunity Score (**{b_r['Volume_Score']}** | {b_r['Snap_%']}% snaps, {b_r['Tgt/G']} tgt/g, {b_r['Car/G']} car/g) than starter **{s_r['Player']}** (**{s_r['Volume_Score']}** | {s_r['Snap_%']}% snaps)."
                    )

# ==============================================================================
# TOP HEADER METRICS
# ==============================================================================
st.markdown(f"## 🦅 {gm_input}'s Dynasty War Room — {curr_season} Week {curr_week} (vs. `{opp_gm_name}`)")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Active Season / Week", f"{curr_season} Wk {curr_week}")
m2.metric("Week Opponent", opp_gm_name)
m3.metric("Lineup Injury Flags", len(injury_alerts))
m4.metric("Start/Sit Volume Alerts", len(volume_swap_alerts))

# ==============================================================================
# 4 MAIN APPLICATION TABS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 AI Dynasty GM (Chat)",
    "⚡ Start / Sit Volume Comparator",
    "📡 Live Waiver & Free Agent Radar",
    "🔄 Trade Surplus & Rival Scanner",
])

# ------------------------------------------------------------------------------
# TAB 1: CONVERSATIONAL AI DYNASTY ANALYST
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("💬 Ask Your AI Dynasty Analyst")
    st.caption(
        "Grounded in your live Sleeper lineup, bench volume (Snap %, Targets, Carries, Red-Zone, WOPR), opponent lineup, and unrostered waiver pool."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hey Carlos! I'm synced with **{gm_input}**'s live Week {curr_week} roster against **{opp_gm_name}**.\n\n"
                    "Ask me anything about **who to start/sit based on snap & target volume**, **which waiver wire free agents to claim**, or **which rival GM to target in a trade**!"
                ),
            }
        ]

    # Quick-Action Prompt Buttons
    q1, q2, q3 = st.columns(3)
    quick_prompt = None
    if q1.button("⚡ Audit My Week Start/Sit Lineup", use_container_width=True):
        quick_prompt = f"Analyze my Week {curr_week} starting lineup vs my bench using Snap %, Targets/G, Carries/G, WOPR, and Red-Zone touches. Am I leaving points on the bench?"
    if q2.button("📡 Top 5 High-Volume Waiver Pickups", use_container_width=True):
        quick_prompt = "Scan the unrostered free agents in my league. Which 5 players have the best underlying Snap %, Target/Touch volume, and dynasty upside to pick up right now?"
    if q3.button("🤝 Suggest 2 Win-Now Trades", use_container_width=True):
        quick_prompt = "Look at my roster and the positional strengths/weaknesses of my leaguemates. Propose 2 realistic trades that boost my starting PPG."

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about start/sit volume, waiver stashes, or trade targets...") or quick_prompt

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            with st.chat_message("assistant"):
                st.warning("Please enter your Gemini API Key in the left sidebar to enable live AI responses.")
        else:
            # Build Rich RAG System Context
            starters_ctx = my_starters_df.to_dict(orient="records") if not my_starters_df.empty else []
            bench_ctx = my_bench_df.to_dict(orient="records") if not my_bench_df.empty else []
            waivers_ctx = df_waivers.head(20).to_dict(orient="records") if not df_waivers.empty else []

            system_instruction = f"""
            You are an elite NFL Fantasy Football Data Scientist and Dynasty GM Advisor for Carlos (Sleeper username: '{gm_input}', team: 'Birdgang') in League {league_id_input}.
            Current NFL Context: {curr_season} Season, Week {curr_week}. Current Opponent: {opp_gm_name}.

            LIVE ROSTER & VOLUME DATA (Last {lookback_wks} Weeks Average):
            - My Active Starters: {starters_ctx}
            - My Bench Players: {bench_ctx}
            - Automated Injury Alerts: {injury_alerts}
            - Automated Volume Swap Alerts: {volume_swap_alerts}
            - Opponent ({opp_gm_name}) Starters: {opp_starters_profiles}
            - Top 20 Unrostered Waiver Wire Players Available: {waivers_ctx}

            ANALYTICAL RULES:
            1. Prioritize predictive volume metrics over fluke touchdowns: Snap %, Snap Trend, Targets/G, Target Share %, WOPR, Carries/G, and Red-Zone Touches/G.
            2. Always cite the exact numbers (Snap %, Tgt/G, Volume Score, Dynasty Value) when comparing two players.
            3. Be direct, sharp, and actionable.
            """

            try:
                client = genai.Client(api_key=api_key)
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing snaps, target shares & matchup data..."):
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=user_query,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.3,
                            ),
                        )
                        ans = response.text
                        st.markdown(ans)
                        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

# ------------------------------------------------------------------------------
# TAB 2: START / SIT VOLUME COMPARATOR & INJURY ALERTS
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"⚡ Week {curr_week} Start / Sit Volume Comparator")

    if injury_alerts or volume_swap_alerts:
        for al in injury_alerts:
            st.error(al)
        for val_al in volume_swap_alerts:
            st.warning(val_al)
    else:
        st.success("✅ Optimal Lineup: All starters are healthy and no bench player exceeds your starters' Volume Scores!")

    display_cols = [
        "Player", "Pos", "Team", "Status", "Volume_Score", "Snap_%", "Snap_Trend",
        "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "Recent_PPG", "Dynasty_Val"
    ]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Current Starting Lineup")
        if not my_starters_df.empty:
            st.dataframe(my_starters_df[display_cols], use_container_width=True, hide_index=True)
    with c2:
        st.markdown("#### 🪑 Bench Alternatives (Sorted by Volume Score)")
        if not my_bench_df.empty:
            st.dataframe(
                my_bench_df.sort_values("Volume_Score", ascending=False)[display_cols],
                use_container_width=True,
                hide_index=True,
            )

# ------------------------------------------------------------------------------
# TAB 3: LIVE WAIVER & FREE AGENT RADAR
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📡 Unrostered Waiver Wire & Free Agent Radar")
    st.caption(
        "Filtered strictly for players owned by 0 teams in your league. Ranked by Opportunity Volume Score, Snap %, and Global 24h Sleeper Adds."
    )

    f_col1, f_col2 = st.columns([1, 2])
    pos_filter = f_col1.selectbox("Filter by Position", ["ALL", "RB", "WR", "TE", "QB"])
    sort_metric = f_col2.selectbox(
        "Sort Free Agents By",
        ["Volume_Score", "24h_Adds", "Snap_%", "Dynasty_Val", "Recent_PPG"],
    )

    if not df_waivers.empty:
        view_waivers = df_waivers.copy()
        if pos_filter != "ALL":
            view_waivers = view_waivers[view_waivers["Pos"] == pos_filter]
        view_waivers = view_waivers.sort_values(by=sort_metric, ascending=False)

        waiver_cols = [
            "Player", "Pos", "Team", "Status", "Volume_Score", "24h_Adds",
            "Snap_%", "Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "Recent_PPG", "Dynasty_Val"
        ]
        st.dataframe(view_waivers[waiver_cols].head(35), use_container_width=True, hide_index=True)
    else:
        st.info("No unrostered players matched the filter criteria.")

# ------------------------------------------------------------------------------
# TAB 4: TRADE SURPLUS, RIVAL SCANNER & INTERACTIVE TRADE CALCULATOR
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("🔄 League Positional Value Matrix & Trade Machine")
    st.caption("Identify which GMs have a surplus at a position you need (or a deficit where you have depth).")

    matrix_rows = []
    for gm_u, rdata in all_gm_rosters.items():
        pos_vals = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        for pid in rdata["players"]:
            p = players_db.get(str(pid), {})
            pos = p.get("position")
            if pos in pos_vals:
                cn = clean_name(f"{p.get('first_name', '')} {p.get('last_name', '')}")
                pos_vals[pos] += val_map.get(cn, 0)
        tot_v = sum(pos_vals.values())
        matrix_rows.append({
            "GM": gm_u,
            "Record": f"{rdata['wins']}-{rdata['losses']}",
            "Total_Roster_Val": tot_v,
            "QB_Val": pos_vals["QB"],
            "RB_Val": pos_vals["RB"],
            "WR_Val": pos_vals["WR"],
            "TE_Val": pos_vals["TE"],
        })

    df_matrix = pd.DataFrame(matrix_rows).sort_values("Total_Roster_Val", ascending=False)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Live Trade Calculator (Dynasty Value + Weekly Volume PPG)")
    tc1, tc2 = st.columns(2)

    my_all_pids = my_starters_pids + my_bench_pids + my_ir_pids + my_taxi_pids
    my_all_profiles = {
        f"{p['Player']} ({p['Pos']} - Val: {p['Dynasty_Val']} | {p['Recent_PPG']} PPG)": p
        for p in [
            build_player_profile(pid, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
            for pid in my_all_pids
        ]
    }

    with tc1:
        partner_gm = st.selectbox(
            "Select Trade Partner GM",
            [g for g in sorted(all_gm_rosters.keys()) if g.lower() != gm_input.strip().lower()],
        )
        send_sel = st.multiselect("📤 Players You Send", list(my_all_profiles.keys()))

    with tc2:
        partner_pids = all_gm_rosters.get(partner_gm, {}).get("players", [])
        partner_profiles = {
            f"{p['Player']} ({p['Pos']} - Val: {p['Dynasty_Val']} | {p['Recent_PPG']} PPG)": p
            for p in [
                build_player_profile(pid, players_db, val_map, rank_map, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map)
                for pid in partner_pids
            ]
        }
        get_sel = st.multiselect(f"📥 Players You Receive from {partner_gm}", list(partner_profiles.keys()))

    if send_sel or get_sel:
        send_val = sum(my_all_profiles[k]["Dynasty_Val"] for k in send_sel)
        get_val = sum(partner_profiles[k]["Dynasty_Val"] for k in get_sel)
        send_ppg = sum(my_all_profiles[k]["Recent_PPG"] for k in send_sel)
        get_ppg = sum(partner_profiles[k]["Recent_PPG"] for k in get_sel)

        val_diff = get_val - send_val
        ppg_diff = round(get_ppg - send_ppg, 1)

        r1, r2, r3 = st.columns(3)
        r1.metric("You Receive (Total Val)", f"{get_val:,}", delta=f"{val_diff:+,} Net Dynasty Surplus")
        r2.metric("You Send (Total Val)", f"{send_val:,}")
        r3.metric("Weekly Scoring Impact", f"{get_ppg:.1f} PPG", delta=f"{ppg_diff:+.1f} PPG vs Sent")
