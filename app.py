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

ALL_32_NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

TEAM_NORM = {
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX"
}


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


def norm_nfl_team(t):
    if not t:
        return "FA"
    t = str(t).upper().strip()
    return TEAM_NORM.get(t, t)


def calc_custom_fantasy_pts(st_d, scoring, pos):
    """Calculates exact fantasy points for a weekly stat dict using granular Sleeper scoring_settings."""
    pts = 0.0
    for k, mult in scoring.items():
        if k in st_d and st_d[k] is not None:
            try:
                pts += float(st_d[k]) * float(mult)
            except Exception:
                pass
    # Apply TE Reception Premium if not already in st_d['bonus_rec_te']
    if pos == "TE" and "bonus_rec_te" in scoring and "bonus_rec_te" not in st_d:
        pts += float(st_d.get("rec", 0.0) or 0.0) * float(scoring.get("bonus_rec_te", 0.0))
    return round(pts, 2)


def calc_expected_fp(st_d, scoring, pos):
    """Calculates volume-based Expected Fantasy Points (xFP) adjusted for league scoring settings."""
    rec_val = float(scoring.get("rec", 1.0))
    tep = float(scoring.get("bonus_rec_te", 0.0)) if pos == "TE" else 0.0
    pass_td_val = float(scoring.get("pass_td", 4.0))
    rec_fd = float(scoring.get("bonus_rec_fd", 0.0))
    rush_fd = float(scoring.get("bonus_rush_fd", 0.0))

    tgt = float(st_d.get("rec_tgt", 0.0) or 0.0)
    rz_tgt = float(st_d.get("rec_rz_tgt", 0.0) or 0.0)
    air_yd = max(0.0, float(st_d.get("rec_air_yd", 0.0) or 0.0))
    car = float(st_d.get("rush_att", 0.0) or 0.0)
    rz_car = float(st_d.get("rush_rz_att", 0.0) or 0.0)
    pass_att = float(st_d.get("pass_att", 0.0) or 0.0)

    if pos in ["WR", "TE"]:
        catch_rate = 0.63 if pos == "WR" else 0.68
        xfp = (
            tgt * (catch_rate * (rec_val + tep) + 0.72 + (0.28 * rec_fd))
            + (air_yd * 0.045)
            + (rz_tgt * 1.45)
            + (car * 0.48)
            + (rz_car * 1.65)
        )
    elif pos == "RB":
        xfp = (
            car * (0.45 + (0.22 * rush_fd))
            + (rz_car * 1.75)
            + tgt * (0.75 * (rec_val + tep) + 0.54 + (0.20 * rec_fd))
            + (rz_tgt * 1.40)
        )
    elif pos == "QB":
        xfp = (
            pass_att * (0.29 + (0.043 * pass_td_val))
            + (car * 0.58)
            + (rz_car * 1.85)
        )
    else:
        xfp = 0.0
    return round(xfp, 2)


# ==============================================================================
# CACHED MASTER DATA PIPELINE
# ==============================================================================
@st.cache_data(ttl=900, show_spinner=False)
def fetch_all_war_room_data(league_id, lookback_weeks=3):
    # 1. Detect Current NFL State
    nfl_state = get_json("https://api.sleeper.app/v1/state/nfl") or {}
    curr_season = int(nfl_state.get("season", 2026))
    curr_week = max(1, int(nfl_state.get("display_week") or nfl_state.get("week") or 6))

    # 2. Master NFL Database & Global Trending Adds
    players_db = get_json("https://api.sleeper.app/v1/players/nfl") or {}
    trending_adds_raw = (
        get_json("https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100")
        or []
    )
    trending_map = {str(item["player_id"]): int(item["count"]) for item in trending_adds_raw}

    # 3. League Settings, Scoring, Users, Rosters, Matchups & Traded Picks
    league_info = get_json(f"https://api.sleeper.app/v1/league/{league_id}") or {}
    roster_positions = league_info.get("roster_positions") or []
    scoring_settings = league_info.get("scoring_settings") or {}
    league_settings = league_info.get("settings") or {}

    is_superflex = ("SUPER_FLEX" in roster_positions) or (roster_positions.count("QB") >= 2)

    users = get_json(f"https://api.sleeper.app/v1/league/{league_id}/users") or []
    rosters = get_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters") or []
    matchups = get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{curr_week}") or []
    traded_picks = get_json(f"https://api.sleeper.app/v1/league/{league_id}/traded_picks") or []

    u_map = {
        u["user_id"]: normalize_gm(u.get("display_name", "User"))
        for u in users
        if u.get("user_id")
    }

    # 4. Live DynastyProcess Market Values (Auto-Selecting Superflex vs 1QB)
    val_map, rank_map = {}, {}
    try:
        df_val = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv")
        val_col = "value_2qb" if (is_superflex and "value_2qb" in df_val.columns) else "value_1qb"
        ecr_col = "ecr_2qb" if (is_superflex and "ecr_2qb" in df_val.columns) else "ecr_1qb"
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

    # 5. Fetch NFL Schedule, Bye Weeks & Vegas Implied Team Totals
    # Maps: (week, team) -> {'opp': opp_team, 'implied': implied_pts, 'spread': spread, 'total': total}
    schedule_map = {}
    playing_teams_curr_wk = set()
    try:
        df_sched = pd.read_csv("https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv")
        df_s_yr = df_sched[df_sched["season"] == curr_season]
        for _, r in df_s_yr.iterrows():
            wk = int(r.get("week", 0) or 0)
            ht = norm_nfl_team(r.get("home_team"))
            at = norm_nfl_team(r.get("away_team"))
            tot_line = float(r["total_line"]) if pd.notnull(r.get("total_line")) else 44.5
            spr_line = float(r["spread_line"]) if pd.notnull(r.get("spread_line")) else 0.0
            # In nflverse, positive spread_line typically favors home_team
            h_imp = round((tot_line / 2.0) + (spr_line / 2.0), 1)
            a_imp = round((tot_line / 2.0) - (spr_line / 2.0), 1)
            schedule_map[(wk, ht)] = {"opp": at, "implied": h_imp, "total": tot_line, "is_home": True}
            schedule_map[(wk, at)] = {"opp": ht, "implied": a_imp, "total": tot_line, "is_home": False}
            if wk == curr_week:
                playing_teams_curr_wk.update([ht, at])
    except Exception:
        pass

    # Fallback Bye-Week check using Sleeper current week projections if schedule CSV didn't have curr_season
    if not playing_teams_curr_wk:
        wk_proj = get_json(f"https://api.sleeper.app/v1/projections/nfl/regular/{curr_season}/{curr_week}") or {}
        if isinstance(wk_proj, dict):
            for pid, p_proj in wk_proj.items():
                if isinstance(p_proj, dict):
                    pts_p = p_proj.get("pts_ppr") or p_proj.get("pts_std") or p_proj.get("gp") or 0
                    if float(pts_p or 0) > 0:
                        t_abbr = norm_nfl_team((players_db.get(str(pid)) or {}).get("team"))
                        if t_abbr in ALL_32_NFL_TEAMS:
                            playing_teams_curr_wk.add(t_abbr)

    bye_teams_curr_wk = sorted(list(ALL_32_NFL_TEAMS - playing_teams_curr_wk)) if playing_teams_curr_wk else []

    # 6. Pull Weekly Usage, Granular Fantasy Points, xFP, FPOE & DvP (Defense vs. Position)
    completed_weeks = [w for w in range(max(1, curr_week - lookback_weeks), curr_week)]
    if not completed_weeks:
        completed_weeks = [1]

    all_completed_season_wks = [w for w in range(1, curr_week)] or [1]
    dvp_pts_allowed = defaultdict(lambda: defaultdict(list))  # dvp_pts_allowed[def_team][pos] -> list of weekly pts

    weekly_usage = defaultdict(lambda: {
        "wks": 0, "snap_pcts": [], "tgts": [], "carries": [], "rz_touches": [],
        "air_yds": [], "rec_yds": [], "fds": [], "pts": [], "xfp": []
    })

    for w in all_completed_season_wks:
        w_stats = get_json(f"https://api.sleeper.app/v1/stats/nfl/regular/{curr_season}/{w}") or {}
        if not isinstance(w_stats, dict):
            continue

        team_pos_pts_wk = defaultdict(float)  # (off_team, pos) -> pts in week w

        for pid, st_d in w_stats.items():
            if not isinstance(st_d, dict):
                continue
            p_meta = players_db.get(str(pid), {})
            pos = p_meta.get("position", "N/A")
            if pos not in ["QB", "RB", "WR", "TE"]:
                continue
            tm = norm_nfl_team(p_meta.get("team"))

            off_snp = float(st_d.get("off_snp", 0.0) or 0.0)
            tm_snp = float(st_d.get("tm_off_snp", 0.0) or 0.0)
            tgt = float(st_d.get("rec_tgt", 0.0) or 0.0)
            car = float(st_d.get("rush_att", 0.0) or 0.0)
            rz_t = float(st_d.get("rec_rz_tgt", 0.0) or 0.0) + float(st_d.get("rush_rz_att", 0.0) or 0.0)
            air = float(st_d.get("rec_air_yd", 0.0) or 0.0)
            rec_yd = float(st_d.get("rec_yd", 0.0) or 0.0)
            fd = float(st_d.get("rec_fd", 0.0) or 0.0) + float(st_d.get("rush_fd", 0.0) or 0.0)

            custom_pts = calc_custom_fantasy_pts(st_d, scoring_settings, pos)
            xfp_val = calc_expected_fp(st_d, scoring_settings, pos)

            if custom_pts > 0 and tm in ALL_32_NFL_TEAMS:
                team_pos_pts_wk[(tm, pos)] += custom_pts

            if w in completed_weeks and (off_snp > 0 or tgt > 0 or car > 0 or custom_pts > 0):
                snap_pct = (off_snp / tm_snp * 100.0) if tm_snp > 0 else 0.0
                u_rec = weekly_usage[str(pid)]
                u_rec["wks"] += 1
                u_rec["snap_pcts"].append(snap_pct)
                u_rec["tgts"].append(tgt)
                u_rec["carries"].append(car)
                u_rec["rz_touches"].append(rz_t)
                u_rec["air_yds"].append(air)
                u_rec["rec_yds"].append(rec_yd)
                u_rec["fds"].append(fd)
                u_rec["pts"].append(custom_pts)
                u_rec["xfp"].append(xfp_val)

        # Attribute offensive positional points to the opposing NFL defense for DvP ranking
        for (off_tm, pos), tot_p in team_pos_pts_wk.items():
            opp_def = (schedule_map.get((w, off_tm)) or {}).get("opp")
            if opp_def and opp_def in ALL_32_NFL_TEAMS:
                dvp_pts_allowed[opp_def][pos].append(tot_p)

    # Compute DvP Ranks (1 = Toughest Defense / Fewest Pts Allowed, 32 = Easiest / Most Pts Allowed)
    dvp_ranks = {}  # (def_team, pos) -> rank 1..32
    for pos in ["QB", "RB", "WR", "TE"]:
        avg_allowed = []
        for dt in ALL_32_NFL_TEAMS:
            vals = dvp_pts_allowed[dt][pos]
            avg_allowed.append((dt, float(np.mean(vals)) if vals else 15.0))
        avg_allowed.sort(key=lambda x: x[1])  # Ascending: 1 = toughest, 32 = easiest
        for idx, (dt, _) in enumerate(avg_allowed, 1):
            dvp_ranks[(dt, pos)] = idx

    # 7. Enrich with nflverse WOPR, PACR, Target Share & Next Gen Separation
    nflverse_metrics = {}
    for yr in [curr_season, curr_season - 1]:
        try:
            url_ps = f"https://github.com/nflverse/nflverse-data/releases/download/player_stats/stats_player_week_{yr}.csv"
            df_ps = pd.read_csv(url_ps)
            df_ps["clean"] = df_ps["player_display_name"].apply(clean_name)
            max_w = df_ps["week"].max()
            rec_ps = df_ps[df_ps["week"] >= max(1, max_w - lookback_weeks)]
            grp_ps = rec_ps.groupby("clean").agg(
                tgt_share=("target_share", "mean"),
                wopr=("wopr", "mean"),
                pacr=("pacr", "mean"),
            ).reset_index()
            for _, r in grp_ps.iterrows():
                nflverse_metrics[r["clean"]] = {
                    "tgt_share": round(float(r["tgt_share"] or 0.0) * 100, 1),
                    "wopr": round(float(r["wopr"] or 0.0), 2),
                    "pacr": round(float(r["pacr"] or 0.0), 2),
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
        is_superflex,
        roster_positions,
        scoring_settings,
        league_settings,
        players_db,
        u_map,
        rosters,
        matchups,
        traded_picks,
        trending_map,
        val_map,
        rank_map,
        schedule_map,
        bye_teams_curr_wk,
        dvp_ranks,
        dict(weekly_usage),
        nflverse_metrics,
        ngs_sep_map,
    )


def build_player_profile(
    pid,
    curr_week,
    players_db,
    val_map,
    rank_map,
    schedule_map,
    bye_teams_curr_wk,
    dvp_ranks,
    weekly_usage,
    nflverse_metrics,
    ngs_sep_map,
    trending_map,
):
    p = players_db.get(str(pid), {})
    full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or str(pid)
    pos = p.get("position", "N/A")
    team = norm_nfl_team(p.get("team"))
    age = float(p.get("age") or 0.0)
    depth_order = int(p.get("depth_chart_order") or 9)
    raw_inj = p.get("injury_status") or ""
    cn = clean_name(full)

    # Check Bye Week Status First!
    is_on_bye = team in bye_teams_curr_wk
    if is_on_bye:
        status_str = f"BYE (Wk {curr_week})" if not raw_inj else f"BYE / {raw_inj}"
    else:
        status_str = raw_inj if raw_inj else "Healthy"

    # Matchup, Vegas Implied Total & DvP Rank
    sched_info = schedule_map.get((curr_week, team))
    if is_on_bye:
        opp_str = "BYE"
        vegas_imp = 0.0
        dvp_rk = 0
        dvp_str = "BYE"
    elif sched_info:
        opp_tm = sched_info["opp"]
        prefix = "vs" if sched_info["is_home"] else "@"
        opp_str = f"{prefix} {opp_tm}"
        vegas_imp = sched_info["implied"]
        dvp_rk = dvp_ranks.get((opp_tm, pos), 16)
        dvp_tier = "🟢 Easy" if dvp_rk >= 22 else ("🔴 Tough" if dvp_rk <= 10 else "🟡 Neutral")
        dvp_str = f"#{dvp_rk} ({dvp_tier})"
    else:
        opp_str = "TBD"
        vegas_imp = 21.5
        dvp_rk = 16
        dvp_str = "#16 (🟡 Neutral)"

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
    fd_g = round(float(np.mean(u.get("fds", [0.0]))), 1) if u.get("fds") else 0.0
    ppg = round(float(np.mean(u.get("pts", [0.0]))), 1) if u.get("pts") else 0.0
    xfp_g = round(float(np.mean(u.get("xfp", [0.0]))), 1) if u.get("xfp") else 0.0
    fpoe = round(ppg - xfp_g, 1)

    tot_air = sum(u.get("air_yds", [0.0]))
    tot_rec_yd = sum(u.get("rec_yds", [0.0]))
    calc_pacr = round(tot_rec_yd / tot_air, 2) if tot_air >= 10 else 0.0

    nv = nflverse_metrics.get(cn, {})
    tgt_share = nv.get("tgt_share", round(tgt_g * 3.1, 1))
    wopr = nv.get("wopr", round((tgt_g * 0.045) + (avg_snap * 0.003), 2))
    pacr = nv.get("pacr", calc_pacr)
    sep = ngs_sep_map.get(cn, 0.0)

    # Composite Opportunity / Volume Score (Adjusted for Game Environment & Bye Week)
    env_boost = max(0.0, (vegas_imp - 20.0) * 0.8) + ((dvp_rk - 16) * 0.25) if not is_on_bye else 0.0
    if pos in ["WR", "TE"]:
        base_vol = (avg_snap * 0.30) + (tgt_g * 3.8) + (rz_g * 5.5) + (wopr * 20.0) + (xfp_g * 1.2) + env_boost
    elif pos == "RB":
        base_vol = (avg_snap * 0.30) + (car_g * 2.0) + (tgt_g * 4.2) + (rz_g * 6.0) + (xfp_g * 1.2) + env_boost
    elif pos == "QB":
        base_vol = (avg_snap * 0.40) + (xfp_g * 1.8) + (rz_g * 4.0) + env_boost
    else:
        base_vol = ppg * 2.5

    vol_score = 0.0 if is_on_bye else round(base_vol, 1)

    # Backfield Contingency Index (Specifically for RB2/RB3 handcuffs)
    contingency_score = 0.0
    if pos == "RB":
        tm_off_bonus = max(0.0, vegas_imp - 18.0) * 3.5
        youth_bonus = max(0.0, (26.0 - age) * 6.0) if age > 0 else 10.0
        role_factor = 25.0 if depth_order == 2 else (15.0 if depth_order >= 3 else 10.0)
        contingency_score = round(role_factor + tm_off_bonus + youth_bonus + (avg_snap * 0.4) + (val * 0.015), 1)

    # Dynasty Value Cliff Warning
    cliff_warning = "✅ Prime Window"
    if pos == "RB" and age >= 26.0:
        cliff_warning = f"⚠️ RB Age Cliff ({age:.0f}yo)"
    elif pos == "WR" and age >= 28.5:
        cliff_warning = f"⚠️ WR Age Cliff ({age:.0f}yo)"
    elif pos == "TE" and age >= 29.5:
        cliff_warning = f"⚠️ TE Age Cliff ({age:.0f}yo)"
    elif pos == "QB" and age >= 33.5:
        cliff_warning = f"⚠️ QB Age Cliff ({age:.0f}yo)"

    return {
        "PID": str(pid),
        "Player": full,
        "Pos": pos,
        "Team": team,
        "Age": age if age > 0 else None,
        "Status": status_str,
        "Raw_Injury": raw_inj,
        "Is_Bye": is_on_bye,
        "Matchup": opp_str,
        "Vegas_Implied": vegas_imp,
        "DvP_Matchup": dvp_str,
        "DvP_Rank": dvp_rk,
        "Volume_Score": vol_score,
        "xFP/G": xfp_g,
        "Recent_PPG": ppg,
        "FPOE": fpoe,
        "Snap_%": avg_snap,
        "Snap_Trend": f"{snap_delta:+.1f}%",
        "Tgt/G": tgt_g,
        "Car/G": car_g,
        "RZ_Touch/G": rz_g,
        "1st_Down/G": fd_g,
        "Tgt_Share_%": tgt_share,
        "WOPR": wopr,
        "PACR": pacr,
        "NGS_Sep": sep,
        "Contingency_Score": contingency_score,
        "Age_Cliff": cliff_warning,
        "Dynasty_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
        "Dynasty_Val": val,
        "24h_Adds": trend_adds,
    }


# ==============================================================================
# SIDEBAR CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("🦅 Birdgang War Room")
    st.caption("Granular Sleeper + nflverse + Vegas/DvP + Gemini AI")

    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else "",
        type="password",
    )
    model_choice = st.selectbox(
        "🤖 Gemini Model",
        ["gemini-3.8-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        index=0,
    )
    league_id_input = st.text_input("🏆 Sleeper League ID", value=DEFAULT_LEAGUE_ID)
    gm_input = st.text_input("👤 Your Sleeper Username", value=DEFAULT_GM)
    lookback_wks = st.slider("📊 Usage Lookback Window (Weeks)", min_value=1, max_value=6, value=3)

    if st.button("🔄 Refresh Live Sleeper Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch Data
with st.spinner("Syncing Sleeper settings, NFL schedule, xFP/FPOE, DvP & Max PF..."):
    (
        curr_season,
        curr_week,
        is_superflex,
        roster_positions,
        scoring_settings,
        league_settings,
        players_db,
        u_map,
        rosters,
        matchups,
        traded_picks,
        trending_map,
        val_map,
        rank_map,
        schedule_map,
        bye_teams_curr_wk,
        dvp_ranks,
        weekly_usage,
        nflverse_metrics,
        ngs_sep_map,
    ) = fetch_all_war_room_data(league_id_input.strip(), lookback_wks)

# ==============================================================================
# PARSE LEAGUE RULES, FAAB, MAX PF & SEGMENT RIVAL ROSTERS
# ==============================================================================
total_faab_budget = int(league_settings.get("waiver_budget", 100) or 100)
max_ir_slots = int(league_settings.get("reserve_slots", 2) or 2)

# Strict IR Eligibility Rules Based on Exact Sleeper League Toggles
allowed_ir_tags = {"IR", "PUP", "PUP-R", "NFI", "NFI-R"}
if int(league_settings.get("reserve_allow_out", 0) or 0) == 1:
    allowed_ir_tags.add("Out")
if int(league_settings.get("reserve_allow_doubtful", 0) or 0) == 1:
    allowed_ir_tags.add("Doubtful")
if int(league_settings.get("reserve_allow_sus", 0) or 0) == 1:
    allowed_ir_tags.add("Sus")
if int(league_settings.get("reserve_allow_na", 0) or 0) == 1:
    allowed_ir_tags.update(["NA", "DNR"])
if int(league_settings.get("reserve_allow_cov", 0) or 0) == 1:
    allowed_ir_tags.add("COV")

rostered_pids = set()
my_starters_pids, my_bench_pids, my_ir_pids, my_taxi_pids = [], [], [], []
my_roster_id = None
my_faab_rem = total_faab_budget
all_gm_rosters = {}
rid_to_gm = {}

for r in rosters:
    rid = r.get("roster_id")
    gm = u_map.get(r.get("owner_id"), f"Roster_{rid}")
    rid_to_gm[rid] = gm

    pids = [str(x) for x in (r.get("players") or []) if x and str(x) != "0"]
    starters = [str(x) for x in (r.get("starters") or []) if x and str(x) != "0"]
    ir_list = [str(x) for x in (r.get("reserve") or []) if x and str(x) != "0"]
    taxi_list = [str(x) for x in (r.get("taxi") or []) if x and str(x) != "0"]
    active_bench = [p for p in pids if p not in starters and p not in ir_list and p not in taxi_list]

    rostered_pids.update(pids)
    st_cfg = r.get("settings") or {}
    wins = int(st_cfg.get("wins", 0) or 0)
    losses = int(st_cfg.get("losses", 0) or 0)
    fpts = float(st_cfg.get("fpts", 0) or 0) + (float(st_cfg.get("fpts_decimal", 0) or 0) / 100.0)
    ppts = float(st_cfg.get("ppts", 0) or 0) + (float(st_cfg.get("ppts_decimal", 0) or 0) / 100.0)
    faab_used = int(st_cfg.get("waiver_budget_used", 0) or 0)
    faab_rem = max(0, total_faab_budget - faab_used)
    coach_eff = round((fpts / ppts) * 100.0, 1) if ppts > 0 else 100.0

    all_gm_rosters[gm] = {
        "roster_id": rid,
        "players": pids,
        "starters": starters,
        "bench": active_bench,
        "ir": ir_list,
        "taxi": taxi_list,
        "wins": wins,
        "losses": losses,
        "fpts": round(fpts, 2),
        "max_pf": round(ppts, 2) if ppts > 0 else round(fpts * 1.12, 2),
        "coach_eff_%": coach_eff,
        "faab_rem": faab_rem,
    }

    if gm.lower() == gm_input.strip().lower():
        my_roster_id = rid
        my_starters_pids = starters
        my_bench_pids = active_bench
        my_ir_pids = ir_list
        my_taxi_pids = taxi_list
        my_faab_rem = faab_rem

# Build DataFrames for Starters, Active Bench, Taxi Squad, and IR
def build_df_from_pids(pid_list):
    if not pid_list:
        return pd.DataFrame()
    return pd.DataFrame([
        build_player_profile(
            p, curr_week, players_db, val_map, rank_map, schedule_map,
            bye_teams_curr_wk, dvp_ranks, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map
        )
        for p in pid_list
    ])

my_starters_df = build_df_from_pids(my_starters_pids)
my_bench_df = build_df_from_pids(my_bench_pids)
my_taxi_df = build_df_from_pids(my_taxi_pids)
my_ir_df = build_df_from_pids(my_ir_pids)
my_full_roster_df = build_df_from_pids(my_starters_pids + my_bench_pids + my_taxi_pids + my_ir_pids)

# Current Week Opponent
opp_gm_name = "Bye / TBD"
opp_starters_df = pd.DataFrame()
if my_roster_id and matchups:
    my_m = next((m for m in matchups if m.get("roster_id") == my_roster_id), None)
    if my_m and my_m.get("matchup_id"):
        opp_m = next(
            (m for m in matchups if m.get("matchup_id") == my_m.get("matchup_id") and m.get("roster_id") != my_roster_id),
            None,
        )
        if opp_m:
            opp_rid = opp_m.get("roster_id")
            opp_gm_name = rid_to_gm.get(opp_rid, f"Roster {opp_rid}")
            opp_starters_df = build_df_from_pids(
                [str(x) for x in (opp_m.get("starters") or []) if x and str(x) != "0"]
            )

# Build Unrostered Waiver Wire Pool
unrostered_list = []
for pid, p in players_db.items():
    if str(pid) not in rostered_pids and p.get("position") in ["QB", "RB", "WR", "TE"] and p.get("team"):
        prof = build_player_profile(
            pid, curr_week, players_db, val_map, rank_map, schedule_map,
            bye_teams_curr_wk, dvp_ranks, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map
        )
        if prof["Volume_Score"] >= 12.0 or prof["Dynasty_Val"] >= 120 or prof["24h_Adds"] >= 20 or prof["Contingency_Score"] >= 45.0:
            unrostered_list.append(prof)

df_waivers = pd.DataFrame(unrostered_list)
if not df_waivers.empty:
    df_waivers = df_waivers.sort_values(by=["Volume_Score", "xFP/G", "Dynasty_Val"], ascending=False)

# ==============================================================================
# AUTOMATED ALERTS ENGINE (BYE WEEKS, STRICT IR RULES, & VOLUME/xFP SWAPS)
# ==============================================================================
lineup_alerts = []
ir_alerts = []
volume_swap_alerts = []

# 1. Starter Bye Week or Injury Alerts
if not my_starters_df.empty:
    for _, s_row in my_starters_df.iterrows():
        if s_row["Is_Bye"]:
            lineup_alerts.append(
                f"🛑 **BYE WEEK ALERT:** Starter **{s_row['Player']} ({s_row['Pos']} - {s_row['Team']})** is on a **BYE** in Week {curr_week}! Swap them out immediately."
            )
        elif s_row["Raw_Injury"] in ["Out", "Doubtful", "IR", "PUP", "Questionable", "Sus"]:
            lineup_alerts.append(
                f"🚨 **Starter Injury Alert:** **{s_row['Player']} ({s_row['Pos']})** carries designation **{s_row['Raw_Injury']}** ({s_row['Matchup']})."
            )

# 2. Strict League-Synced IR Eligibility Alerts (Excludes 'Out'/'NA' unless allowed by league_settings!)
open_ir_spots = max(0, max_ir_slots - len(my_ir_pids))
if open_ir_spots > 0:
    active_non_ir_df = pd.concat([my_starters_df, my_bench_df], ignore_index=True)
    if not active_non_ir_df.empty:
        for _, r_row in active_non_ir_df.iterrows():
            if r_row["Raw_Injury"] in allowed_ir_tags:
                ir_alerts.append(
                    f"🏥 **Verified IR Stash Eligible:** **{r_row['Player']}** has official designation **`{r_row['Raw_Injury']}`** (Allowed by your league's IR rules: `{', '.join(sorted(allowed_ir_tags))}`). Move them to one of your **{open_ir_spots} open IR slot(s)** to claim a free agent!"
                )

# 3. Start/Sit Volume & xFP Swap Alerts (Compares Starters strictly against Healthy, Playing Active Bench — Excludes Taxi & Bye!)
if not my_starters_df.empty and not my_bench_df.empty:
    eligible_bench = my_bench_df[
        (~my_bench_df["Is_Bye"]) & (~my_bench_df["Raw_Injury"].isin(["Out", "Doubtful", "IR", "PUP", "NA", "Sus"]))
    ]
    for pos in ["QB", "RB", "WR", "TE"]:
        st_pos = my_starters_df[my_starters_df["Pos"] == pos]
        bn_pos = eligible_bench[eligible_bench["Pos"] == pos]
        for _, s_r in st_pos.iterrows():
            for _, b_r in bn_pos.iterrows():
                if s_r["Is_Bye"] or (
                    b_r["Volume_Score"] >= s_r["Volume_Score"] + 7.5 and b_r["xFP/G"] >= s_r["xFP/G"]
                ):
                    reason = (
                        f"Starter {s_r['Player']} is on BYE"
                        if s_r["Is_Bye"]
                        else f"Volume Score {b_r['Volume_Score']} vs {s_r['Volume_Score']} & {b_r['xFP/G']} xFP/G vs {s_r['xFP/G']} xFP/G"
                    )
                    volume_swap_alerts.append(
                        f"⚡ **Start/Sit Swap ({pos}):** Start Active Bench **{b_r['Player']}** ({b_r['Matchup']}, DvP {b_r['DvP_Matchup']}, {b_r['Snap_%']}% snaps) over **{s_r['Player']}** ({reason})."
                    )

# ==============================================================================
# TOP HEADER & KPI METRICS
# ==============================================================================
fmt_tag = "Superflex" if is_superflex else "1QB"
tep_val = float(scoring_settings.get("bonus_rec_te", 0.0) or 0.0)
ppr_val = float(scoring_settings.get("rec", 1.0) or 1.0)

st.markdown(
    f"## 🦅 {gm_input}'s Dynasty War Room — {curr_season} Week {curr_week} (vs. `{opp_gm_name}`)"
)
st.caption(
    f"**Synced League Settings:** {fmt_tag} | {ppr_val} PPR | +{tep_val} TE Premium | "
    f"Allowed IR Tags: `{', '.join(sorted(allowed_ir_tags))}` | "
    f"**Teams on Bye (Wk {curr_week}):** `{', '.join(bye_teams_curr_wk) if bye_teams_curr_wk else 'None'}`"
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Week Matchup", f"vs. {opp_gm_name}")
k2.metric("Your FAAB Remaining", f"${my_faab_rem} /${total_faab_budget}")
k3.metric("Bye / Injury Flags", len(lineup_alerts))
k4.metric("Start/Sit Volume Swaps", len(volume_swap_alerts))
my_eff = all_gm_rosters.get(gm_input, {}).get("coach_eff_%", 100.0)
k5.metric("Coaching Efficiency", f"{my_eff}%", help="Actual Points Scored divided by Optimal Max PF")

# ==============================================================================
# 6 DEEP-ANALYSIS TABS
# ==============================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "💬 AI Dynasty GM",
    "⚡ Start/Sit & Game Env",
    "📈 xFP Regression & Age Cliffs",
    "📡 Waiver Radar & RB Contingency",
    "🔮 Max PF Pick Forecaster",
    "🔄 Rival Segments & Trade Machine",
])

# ------------------------------------------------------------------------------
# TAB 1: CONVERSATIONAL AI DYNASTY GM (GEMINI 3.8 FLASH)
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("💬 Conversational AI Dynasty Scientist")
    st.caption(
        f"Powered by `{model_choice}` — Synced with Bye Weeks, Strict IR rules, Taxi separation, Vegas Implied Totals, DvP, xFP/FPOE, and Max PF."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    f"Hey Carlos! I'm synced with **{gm_input}**'s Week {curr_week} matchup against **{opp_gm_name}**.\n\n"
                    f"• **Teams on Bye this week:** `{', '.join(bye_teams_curr_wk) if bye_teams_curr_wk else 'None'}`\n"
                    f"• **Your Remaining FAAB:** `${my_faab_rem}`\n\n"
                    "Ask me about **Start/Sit decisions**, **xFP buy-low candidates**, **waiver wire contingency RBs**, or **trade packages**!"
                ),
            }
        ]

    b1, b2, b3, b4 = st.columns(4)
    quick_prompt = None
    if b1.button("⚡ Optimize Week Lineup", use_container_width=True):
        quick_prompt = (
            f"Audit my Week {curr_week} starting lineup vs my Active Bench (excluding Taxi and players on BYE). "
            "Use xFP/G, Volume Score, Snap %, Vegas Implied Totals, and DvP Matchup ranks to tell me who to start or sit."
        )
    if b2.button("📈 Find Buy-Low / Sell-Highs", use_container_width=True):
        quick_prompt = (
            "Analyze my roster using Expected Fantasy Points (xFP/G) vs Actual PPG (FPOE) and Age Cliffs. "
            "Who on my roster is due for positive regression, and who should I sell high?"
        )
    if b3.button("📡 Top Waiver & Handcuff Stashes", use_container_width=True):
        quick_prompt = (
            f"I have ${my_faab_rem} FAAB remaining. Scan the unrostered free agents and recommend the top 3 high-volume pickups "
            "and top 2 RB contingency handcuffs, including how much FAAB to bid."
        )
    if b4.button("🤝 Propose 2 Contender Trades", use_container_width=True):
        quick_prompt = (
            "Review the Contender vs Rebuilder segmentation of my leaguemates and propose 2 win-now trades that upgrade my starting lineup."
        )

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask about start/sit volume, xFP regression, waivers, or trades...") or quick_prompt

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if not api_key:
            with st.chat_message("assistant"):
                st.warning("Please enter your Gemini API Key in the sidebar (or in Streamlit Secrets) to enable AI analysis.")
        else:
            system_instruction = f"""
            You are an elite NFL Fantasy Football Data Scientist and Dynasty GM Advisor for Carlos (Sleeper username: '{gm_input}', team: 'Birdgang') in League {league_id_input}.
            Current NFL Context: {curr_season} Season, Week {curr_week}. Opponent: {opp_gm_name}.
            League Format: {fmt_tag}, {ppr_val} PPR, +{tep_val} TE Premium.
            Strict IR Eligible Designations: {sorted(list(allowed_ir_tags))}. (Players marked 'Out' or 'NA' CANNOT go on IR unless listed here).
            NFL Teams on BYE in Week {curr_week}: {bye_teams_curr_wk}.

            CRITICAL RULES:
            1. NEVER recommend starting any player whose Status contains 'BYE' or whose team is in {bye_teams_curr_wk}. Acknowledge they are on a bye week!
            2. NEVER recommend starting a Taxi Squad player unless explicitly discussing promoting them from Taxi.
            3. NEVER tell the user to move an 'Out', 'Doubtful', or 'NA' player to IR unless that designation is in {sorted(list(allowed_ir_tags))}.
            4. Base Start/Sit and Waiver advice on xFP/G, FPOE (regression), Snap %, Target Share %, WOPR, PACR, Red-Zone Touches/G, Vegas Implied Team Totals, and DvP Matchup Rank (#1 = toughest defense, #32 = easiest defense).

            LIVE DATA CONTEXT:
            - Active Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
            - Active Bench (Eligible to Start): {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
            - Taxi Squad (Developmental Stashes): {my_taxi_df.to_dict(orient='records') if not my_taxi_df.empty else []}
            - IR Slot Players: {my_ir_df.to_dict(orient='records') if not my_ir_df.empty else []}
            - Opponent ({opp_gm_name}) Starters: {opp_starters_df.to_dict(orient='records') if not opp_starters_df.empty else []}
            - Automated Alerts: {lineup_alerts + ir_alerts + volume_swap_alerts}
            - Top 20 Unrostered Free Agents: {df_waivers.head(20).to_dict(orient='records') if not df_waivers.empty else []}
            """

            try:
                client = genai.Client(api_key=api_key)
                with st.chat_message("assistant"):
                    with st.spinner(f"Analyzing with {model_choice}..."):
                        response = client.models.generate_content(
                            model=model_choice,
                            contents=user_query,
                            config=types.GenerateContentConfig(
                                system_instruction=system_instruction,
                                temperature=0.25,
                            ),
                        )
                        ans = response.text
                        st.markdown(ans)
                        st.session_state.chat_messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Gemini API Error: {e}")

# ------------------------------------------------------------------------------
# TAB 2: START / SIT VOLUME COMPARATOR, GAME ENVIRONMENT & TAXI/IR SEPARATION
# ------------------------------------------------------------------------------
with tab2:
    st.subheader(f"⚡ Week {curr_week} Start / Sit Volume & Game Environment Comparator")

    for al in lineup_alerts:
        st.error(al)
    for ir_al in ir_alerts:
        st.info(ir_al)
    for sw_al in volume_swap_alerts:
        st.warning(sw_al)
    if not (lineup_alerts or ir_alerts or volume_swap_alerts):
        st.success("✅ Lineup Clean: No starters on Bye/Out, IR slots optimized, and highest-volume players are starting!")

    start_sit_cols = [
        "Player", "Pos", "Team", "Status", "Matchup", "Vegas_Implied", "DvP_Matchup",
        "Volume_Score", "xFP/G", "Recent_PPG", "FPOE", "Snap_%", "Snap_Trend",
        "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "PACR", "NGS_Sep", "Dynasty_Val"
    ]

    st.markdown("#### 🏆 Active Starting Lineup")
    if not my_starters_df.empty:
        st.dataframe(my_starters_df[start_sit_cols], use_container_width=True, hide_index=True)

    st.markdown("#### 🪑 Active Bench (Eligible for Weekly Lineup — Sorted by Volume Score)")
    if not my_bench_df.empty:
        st.dataframe(
            my_bench_df.sort_values(by=["Is_Bye", "Volume_Score"], ascending=[True, False])[start_sit_cols],
            use_container_width=True,
            hide_index=True,
        )

    col_tx, col_ir = st.columns(2)
    with col_tx:
        st.markdown("#### 🚕 Taxi Squad (Excluded from Weekly Start/Sit Alerts)")
        if not my_taxi_df.empty:
            st.dataframe(my_taxi_df[start_sit_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No players currently on Taxi Squad.")
    with col_ir:
        st.markdown(f"#### 🏥 Injured Reserve / IR ({len(my_ir_pids)} / {max_ir_slots} Slots Used)")
        if not my_ir_df.empty:
            st.dataframe(my_ir_df[start_sit_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No players currently in IR slots.")

# ------------------------------------------------------------------------------
# TAB 3: EXPECTED FANTASY POINTS (xFP), FPOE REGRESSION & AGE CLIFFS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("📈 Expected Fantasy Points (xFP), FPOE Regression & Dynasty Value Cliffs")
    st.caption(
        "**FPOE (Fantasy Points Over Expected)** = `Actual PPG - xFP/G`. Negative FPOE with high xFP/G signals an imminent **Buy-Low / Positive Regression Breakout**. High positive FPOE with low xFP/G signals **Unsustainable TD Luck (Sell High)**."
    )

    if not my_full_roster_df.empty:
        reg_df = my_full_roster_df.copy()
        reg_df["Regression_Signal"] = np.where(
            (reg_df["xFP/G"] >= 10.0) & (reg_df["FPOE"] <= -2.0),
            "🟢 BUY LOW / MUST START (Due for +TDs)",
            np.where(
                (reg_df["FPOE"] >= 3.5) & (reg_df["xFP/G"] < 12.0),
                "🟠 SELL HIGH (TD Dependent / Overperforming)",
                "⚪ Performing to Volume",
            ),
        )
        reg_cols = [
            "Player", "Pos", "Team", "Age", "Age_Cliff", "xFP/G", "Recent_PPG",
            "FPOE", "Regression_Signal", "Snap_%", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "PACR", "Dynasty_Val"
        ]
        st.dataframe(
            reg_df.sort_values("xFP/G", ascending=False)[reg_cols],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### 🧠 League-Wide Coaching Lineup Efficiency (`Actual PF / Optimal Max PF`)")
    eff_rows = []
    for gm_u, d in all_gm_rosters.items():
        pts_left_on_bench = round(max(0.0, d["max_pf"] - d["fpts"]), 2)
        eff_rows.append({
            "GM": gm_u,
            "Record": f"{d['wins']}-{d['losses']}",
            "Actual_PF": d["fpts"],
            "Optimal_Max_PF": d["max_pf"],
            "Points_Left_On_Bench": pts_left_on_bench,
            "Coaching_Efficiency_%": f"{d['coach_eff_%']}%",
        })
    st.dataframe(
        pd.DataFrame(eff_rows).sort_values("Actual_PF", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------------------
# TAB 4: WAIVER RADAR, FAAB TRACKER & RB CONTINGENCY INDEX
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("📡 Unrostered Waiver Radar, FAAB Tracker & Backfield Contingency Index")

    w_c1, w_c2, w_c3 = st.columns([1, 1, 2])
    pos_filter = w_c1.selectbox("Filter Position", ["ALL", "RB", "WR", "TE", "QB"])
    sort_waiver = w_c2.selectbox(
        "Rank Free Agents By",
        ["Volume_Score", "xFP/G", "Contingency_Score", "24h_Adds", "Snap_%", "Dynasty_Val"],
    )
    w_c3.info(
        f"💰 **Your Remaining FAAB:** `${my_faab_rem}` (League Avg Remaining: `${int(np.mean([d['faab_rem'] for d in all_gm_rosters.values()]))}`)"
    )

    if not df_waivers.empty:
        vw = df_waivers.copy()
        if pos_filter != "ALL":
            vw = vw[vw["Pos"] == pos_filter]
        vw = vw.sort_values(by=sort_waiver, ascending=False)
        w_cols = [
            "Player", "Pos", "Team", "Age", "Status", "Matchup", "Volume_Score",
            "xFP/G", "Recent_PPG", "Contingency_Score", "24h_Adds", "Snap_%",
            "Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "Dynasty_Val"
        ]
        st.dataframe(vw[w_cols].head(35), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 💵 League-Wide Remaining FAAB Balances")
    faab_df = pd.DataFrame([
        {"GM": g, "Remaining_FAAB": f"${d['faab_rem']}", "FAAB_Raw": d["faab_rem"], "Record": f"{d['wins']}-{d['losses']}"}
        for g, d in all_gm_rosters.items()
    ]).sort_values("FAAB_Raw", ascending=False)
    st.dataframe(faab_df[["GM", "Remaining_FAAB", "Record"]], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 5: MAX PF ROOKIE DRAFT PICK FORECASTER (2027-2029)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader(f"🔮 Live {curr_season + 1} Rookie Draft Pick Forecaster (Sorted by Inverse Max PF)")
    st.caption(
        "Projects the exact 1.01 to 1.12 rookie draft slots using inverse **Max PF (Optimal Potential Points)** and maps traded pick ownership."
    )

    next_draft_yr = str(curr_season + 1)
    # Map (season, round, orig_roster_id) -> current_owner_rid
    pick_owner_map = {}
    for tp in traded_picks:
        if str(tp.get("season")) == next_draft_yr:
            pick_owner_map[(int(tp.get("round", 1)), int(tp.get("roster_id")))] = int(tp.get("owner_id"))

    # Sort rosters by ascending Max PF (lowest Max PF = 1.01)
    sorted_by_max_pf = sorted(all_gm_rosters.items(), key=lambda x: (x[1]["max_pf"], x[1]["wins"]))

    pick_forecast_rows = []
    for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
        orig_rid = rdata["roster_id"]
        r1_owner_rid = pick_owner_map.get((1, orig_rid), orig_rid)
        r2_owner_rid = pick_owner_map.get((2, orig_rid), orig_rid)
        r3_owner_rid = pick_owner_map.get((3, orig_rid), orig_rid)

        r1_owner_gm = rid_to_gm.get(r1_owner_rid, orig_gm)
        r2_owner_gm = rid_to_gm.get(r2_owner_rid, orig_gm)
        r3_owner_gm = rid_to_gm.get(r3_owner_rid, orig_gm)

        pick_forecast_rows.append({
            "Projected_Slot": f"1.{slot_idx:02d}",
            "Original_Team_GM": orig_gm,
            "Current_Record": f"{rdata['wins']}-{rdata['losses']}",
            "Max_PF (Optimal Pts)": rdata["max_pf"],
            "Actual_PF": rdata["fpts"],
            f"{next_draft_yr}_1st_Owner": f"🦅 {r1_owner_gm}" if r1_owner_gm.lower() == gm_input.lower() else r1_owner_gm,
            f"{next_draft_yr}_2nd_Owner": f"🦅 {r2_owner_gm}" if r2_owner_gm.lower() == gm_input.lower() else r2_owner_gm,
            f"{next_draft_yr}_3rd_Owner": f"🦅 {r3_owner_gm}" if r3_owner_gm.lower() == gm_input.lower() else r3_owner_gm,
        })

    st.dataframe(pd.DataFrame(pick_forecast_rows), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# TAB 6: RIVAL CONTENDER/REBUILDER SEGMENTATION & TRADE MACHINE
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("🔄 Contender vs. Rebuilder Segmentation & Live Trade Machine")

    matrix_rows = []
    all_max_pfs = [d["max_pf"] for d in all_gm_rosters.values()]
    median_max_pf = float(np.median(all_max_pfs)) if all_max_pfs else 500.0

    for gm_u, rdata in all_gm_rosters.items():
        pos_vals = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        for pid in rdata["players"]:
            p = players_db.get(str(pid), {})
            pos = p.get("position")
            if pos in pos_vals:
                cn = clean_name(f"{p.get('first_name', '')} {p.get('last_name', '')}")
                pos_vals[pos] += val_map.get(cn, 0)
        tot_v = sum(pos_vals.values())

        # Classify Team Tier
        if rdata["wins"] > rdata["losses"] and rdata["max_pf"] >= median_max_pf:
            tier = "🏆 True Contender"
        elif rdata["wins"] < rdata["losses"] and rdata["max_pf"] < median_max_pf:
            tier = "🔨 Rebuilder"
        else:
            tier = "⚖️ Middle Tier / Purgatory"

        matrix_rows.append({
            "GM": gm_u,
            "Strategic_Tier": tier,
            "Record": f"{rdata['wins']}-{rdata['losses']}",
            "Max_PF": rdata["max_pf"],
            "FAAB": f"${rdata['faab_rem']}",
            "Total_Dynasty_Val": tot_v,
            "QB_Val": pos_vals["QB"],
            "RB_Val": pos_vals["RB"],
            "WR_Val": pos_vals["WR"],
            "TE_Val": pos_vals["TE"],
        })

    df_matrix = pd.DataFrame(matrix_rows).sort_values("Total_Dynasty_Val", ascending=False)
    st.dataframe(df_matrix, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### ⚖️ Multi-Metric Trade Machine (Dynasty Value + xFP/G + Actual PPG)")
    tc1, tc2 = st.columns(2)

    my_all_pids = my_starters_pids + my_bench_pids + my_ir_pids + my_taxi_pids
    my_all_profiles = {
        f"{p['Player']} ({p['Pos']} | Val: {p['Dynasty_Val']} | {p['xFP/G']} xFP/G)": p
        for p in [
            build_player_profile(
                pid, curr_week, players_db, val_map, rank_map, schedule_map,
                bye_teams_curr_wk, dvp_ranks, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map
            )
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
            f"{p['Player']} ({p['Pos']} | Val: {p['Dynasty_Val']} | {p['xFP/G']} xFP/G)": p
            for p in [
                build_player_profile(
                    pid, curr_week, players_db, val_map, rank_map, schedule_map,
                    bye_teams_curr_wk, dvp_ranks, weekly_usage, nflverse_metrics, ngs_sep_map, trending_map
                )
                for pid in partner_pids
            ]
        }
        get_sel = st.multiselect(f"📥 Players You Receive from {partner_gm}", list(partner_profiles.keys()))

    if send_sel or get_sel:
        send_val = sum(my_all_profiles[k]["Dynasty_Val"] for k in send_sel)
        get_val = sum(partner_profiles[k]["Dynasty_Val"] for k in get_sel)
        send_xfp = sum(my_all_profiles[k]["xFP/G"] for k in send_sel)
        get_xfp = sum(partner_profiles[k]["xFP/G"] for k in get_sel)
        send_ppg = sum(my_all_profiles[k]["Recent_PPG"] for k in send_sel)
        get_ppg = sum(partner_profiles[k]["Recent_PPG"] for k in get_sel)

        r1, r2, r3 = st.columns(3)
        r1.metric("Net Dynasty Value", f"{get_val:,}", delta=f"{get_val - send_val:+,} vs Sent ({send_val:,})")
        r2.metric("Expected Volume Impact (xFP/G)", f"{get_xfp:.1f} xFP/G", delta=f"{get_xfp - send_xfp:+.1f} xFP/G")
        r3.metric("Actual Scoring Impact (PPG)", f"{get_ppg:.1f} PPG", delta=f"{get_ppg - send_ppg:+.1f} PPG")
