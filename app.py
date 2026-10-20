import re
from collections import defaultdict
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# PAGE CONFIGURATION & LEAGUE PRESETS
# ==============================================================================
st.set_page_config(
    page_title="Dynasty & Chop War Room",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

LEAGUE_PRESETS = {
    "🦅 Birdgang (12-Team Dynasty)": {
        "id": "1313348860042571776",
        "mode": "DYNASTY",
        "default_gm": "arturohd",
    },
    "🪓 The Chop Block (18-Team Guillotine)": {
        "id": "1383889769309691904",
        "mode": "CHOP",
        "default_gm": "arturohd",
    },
    "⚙️ Custom Sleeper League ID": {
        "id": "",
        "mode": "DYNASTY",
        "default_gm": "arturohd",
    },
}

ALL_32_NFL_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB", "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV", "MIA", "MIN", "NE", "NO", "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
}

TEAM_NORM = {
    "LA": "LAR", "STL": "LAR", "SD": "LAC", "OAK": "LV", "WSH": "WAS", "JAC": "JAX"
}

WAIVER_DAY_MAP = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
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


def generate_ai_response_with_fallback(client, primary_model, prompt_text, sys_instruction):
    """Calls Gemini API with automatic fallback across models if 503 UNAVAILABLE or 404 occurs."""
    candidate_models = []
    for m in [primary_model, "gemini-3.8-flash", "gemini-2.0-flash", "gemini-2.5-pro"]:
        if m not in candidate_models:
            candidate_models.append(m)

    last_err = None
    for m_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=m_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.25,
                ),
            )
            return response.text, m_name
        except Exception as e:
            last_err = e
            err_str = str(e).upper()
            if "503" in err_str or "UNAVAILABLE" in err_str or "404" in err_str or "OVERLOADED" in err_str:
                continue
            else:
                break
    raise last_err


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

    # 3. League Settings, Scoring, Users, Rosters, Matchups, Traded Picks & ALL Waiver Bids
    league_info = get_json(f"https://api.sleeper.app/v1/league/{league_id}") or {}
    roster_positions = league_info.get("roster_positions") or []
    scoring_settings = league_info.get("scoring_settings") or {}
    league_settings = league_info.get("settings") or {}

    is_superflex = ("SUPER_FLEX" in roster_positions) or (roster_positions.count("QB") >= 2)

    users = get_json(f"https://api.sleeper.app/v1/league/{league_id}/users") or []
    rosters = get_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters") or []
    matchups = get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{curr_week}") or []
    prev_matchups = (
        get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{max(1, curr_week - 1)}")
        or []
    )
    traded_picks = get_json(f"https://api.sleeper.app/v1/league/{league_id}/traded_picks") or []

    u_map = {
        u["user_id"]: normalize_gm(u.get("display_name", "User"))
        for u in users
        if u.get("user_id")
    }

    rid_to_gm_local = {}
    for r in rosters:
        rid_to_gm_local[int(r.get("roster_id"))] = u_map.get(r.get("owner_id"), f"Roster_{r.get('roster_id')}")

    # Crawl all weeks to extract recently dropped players & complete waiver bid history (complete + failed)
    recently_dropped_pids = {}
    raw_waiver_bids = []

    for w_tx in range(1, curr_week + 2):
        tx_list = get_json(f"https://api.sleeper.app/v1/league/{league_id}/transactions/{w_tx}") or []
        for tx in tx_list:
            tx_status = tx.get("status")
            tx_type = tx.get("type")

            if tx_status == "complete" and w_tx >= max(1, curr_week - 1):
                for d_pid, d_rid in (tx.get("drops") or {}).items():
                    recently_dropped_pids[str(d_pid)] = {"week": w_tx, "rid": d_rid}

            if tx_type == "waiver" and tx_status in ["complete", "failed"]:
                adds_dict = tx.get("adds") or {}
                settings_dict = tx.get("settings") or {}
                bid_amt = int(settings_dict.get("waiver_bid", 0) or 0)
                notes_str = str((tx.get("metadata") or {}).get("notes") or "")
                ts_val = int(tx.get("status_updated") or tx.get("created") or 0)

                for a_pid, a_rid in adds_dict.items():
                    rid_int = int(a_rid) if str(a_rid).isdigit() else 0
                    gm_bidder = rid_to_gm_local.get(rid_int, f"Roster_{a_rid}")
                    p_meta = players_db.get(str(a_pid), {})
                    p_full = f"{p_meta.get('first_name', '')} {p_meta.get('last_name', '')}".strip() or str(a_pid)
                    p_pos = p_meta.get("position", "N/A")
                    p_team = norm_nfl_team(p_meta.get("team"))

                    raw_waiver_bids.append({
                        "Week": w_tx,
                        "PID": str(a_pid),
                        "Player": p_full,
                        "Pos": p_pos,
                        "Team": p_team,
                        "Roster_ID": rid_int,
                        "GM": gm_bidder,
                        "Bid": bid_amt,
                        "Status": tx_status,
                        "Notes": notes_str,
                        "Timestamp": ts_val,
                    })

    # 4. Live DynastyProcess Market Values & Positional Live Rankings (QB1, RB1, WR1, TE1)
    val_map, rank_map, pos_rank_map, pos_rank_num_map = {}, {}, {}, {}
    try:
        df_val = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/master/files/values.csv")
        val_col = "value_2qb" if (is_superflex and "value_2qb" in df_val.columns) else "value_1qb"
        ecr_col = "ecr_2qb" if (is_superflex and "ecr_2qb" in df_val.columns) else "ecr_1qb"
        df_val["clean"] = df_val["player"].apply(clean_name)
        df_val = df_val.sort_values(by=[val_col, ecr_col], ascending=[False, True])

        pos_counters = defaultdict(int)
        for _, r in df_val.iterrows():
            cn = r["clean"]
            pos_str = str(r.get("pos") or "").upper().strip()
            v = int(r[val_col]) if pd.notnull(r[val_col]) else 0
            rk = int(r[ecr_col]) if pd.notnull(r[ecr_col]) else 999
            if cn not in val_map:
                val_map[cn] = v
                rank_map[cn] = rk
                if pos_str in ["QB", "RB", "WR", "TE"]:
                    pos_counters[pos_str] += 1
                    p_num = pos_counters[pos_str]
                    pos_rank_map[cn] = f"{pos_str}{p_num}"
                    pos_rank_num_map[cn] = p_num
    except Exception:
        pass

    # 5. Fetch NFL Schedule, Bye Weeks & Vegas Implied Team Totals
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
            h_imp = round((tot_line / 2.0) + (spr_line / 2.0), 1)
            a_imp = round((tot_line / 2.0) - (spr_line / 2.0), 1)
            schedule_map[(wk, ht)] = {"opp": at, "implied": h_imp, "total": tot_line, "is_home": True}
            schedule_map[(wk, at)] = {"opp": ht, "implied": a_imp, "total": tot_line, "is_home": False}
            if wk == curr_week:
                playing_teams_curr_wk.update([ht, at])
    except Exception:
        pass

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

    # 6. Fetch Official nflverse Practice & Injury Reports
    nflverse_injuries = {}
    for yr in [curr_season, curr_season - 1]:
        try:
            url_inj = f"https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{yr}.csv"
            df_inj = pd.read_csv(url_inj)
            df_inj["clean"] = df_inj["full_name"].apply(clean_name)
            max_inj_wk = df_inj["week"].max()
            rec_inj = df_inj[df_inj["week"] == max_inj_wk]
            for _, r in rec_inj.iterrows():
                nflverse_injuries[r["clean"]] = {
                    "report_status": str(r.get("report_status") or "") if pd.notnull(r.get("report_status")) else "",
                    "practice_status": str(r.get("practice_status") or "") if pd.notnull(r.get("practice_status")) else "",
                    "primary_injury": str(r.get("practice_primary_injury") or r.get("report_primary_injury") or "")
                    if pd.notnull(r.get("practice_primary_injury") or r.get("report_primary_injury"))
                    else "",
                }
            break
        except Exception:
            continue

    # 7. Pull Weekly Usage, Granular Fantasy Points, Floor/Bust Metrics, xFP, FPOE & DvP
    completed_weeks = [w for w in range(max(1, curr_week - lookback_weeks), curr_week)] or [1]
    all_completed_season_wks = [w for w in range(1, curr_week)] or [1]
    dvp_pts_allowed = defaultdict(lambda: defaultdict(list))

    weekly_usage = defaultdict(lambda: {
        "wks": 0, "snap_pcts": [], "tgts": [], "carries": [], "rz_touches": [],
        "air_yds": [], "rec_yds": [], "fds": [], "pts": [], "xfp": [], "all_season_pts": []
    })

    for w in all_completed_season_wks:
        w_stats = get_json(f"https://api.sleeper.app/v1/stats/nfl/regular/{curr_season}/{w}") or {}
        if not isinstance(w_stats, dict):
            continue

        team_pos_pts_wk = defaultdict(float)

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

            if off_snp > 0 or tgt > 0 or car > 0 or custom_pts > 0:
                weekly_usage[str(pid)]["all_season_pts"].append(custom_pts)

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

        for (off_tm, pos), tot_p in team_pos_pts_wk.items():
            opp_def = (schedule_map.get((w, off_tm)) or {}).get("opp")
            if opp_def and opp_def in ALL_32_NFL_TEAMS:
                dvp_pts_allowed[opp_def][pos].append(tot_p)

    dvp_ranks = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        avg_allowed = []
        for dt in ALL_32_NFL_TEAMS:
            vals = dvp_pts_allowed[dt][pos]
            avg_allowed.append((dt, float(np.mean(vals)) if vals else 15.0))
        avg_allowed.sort(key=lambda x: x[1])
        for idx, (dt, _) in enumerate(avg_allowed, 1):
            dvp_ranks[(dt, pos)] = idx

    # 8. Enrich with nflverse WOPR, PACR, Target Share & Next Gen Separation
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
        prev_matchups,
        traded_picks,
        recently_dropped_pids,
        raw_waiver_bids,
        trending_map,
        val_map,
        rank_map,
        pos_rank_map,
        pos_rank_num_map,
        schedule_map,
        bye_teams_curr_wk,
        nflverse_injuries,
        dvp_ranks,
        dict(weekly_usage),
        nflverse_metrics,
        ngs_sep_map,
    )


# ==============================================================================
# WAIVER BID AUDIT & GM BIDDING BEHAVIOR ENGINE (100% ENGLISH)
# ==============================================================================
def analyze_waiver_auctions_and_gm_profiles(raw_waiver_bids, all_gm_rosters, total_faab_budget):
    dedup = {}
    for b in raw_waiver_bids:
        key = (b["Week"], b["PID"], b["GM"])
        if key not in dedup:
            dedup[key] = b
        else:
            prev = dedup[key]
            if b["Status"] == "complete" and prev["Status"] != "complete":
                dedup[key] = b
            elif b["Status"] == prev["Status"] and b["Bid"] > prev["Bid"]:
                dedup[key] = b

    clean_bids = list(dedup.values())

    auctions_by_player_wk = defaultdict(list)
    for b in clean_bids:
        auctions_by_player_wk[(b["Week"], b["PID"])].append(b)

    auction_audit_rows = []
    gm_stats = defaultdict(lambda: {
        "total_bids": 0,
        "wins": 0,
        "failed": 0,
        "all_bids": [],
        "winning_bids": [],
        "overpays": [],
        "pos_bids": defaultdict(list),
    })

    for (wk, pid), bid_list in sorted(auctions_by_player_wk.items(), key=lambda x: x[0][0], reverse=True):
        sorted_bids = sorted(
            bid_list,
            key=lambda x: (1 if x["Status"] == "complete" else 0, x["Bid"]),
            reverse=True,
        )
        winner_entry = next((x for x in sorted_bids if x["Status"] == "complete"), None)
        if not winner_entry:
            continue

        w_gm = winner_entry["GM"]
        w_bid = winner_entry["Bid"]
        p_name = winner_entry["Player"]
        p_pos = winner_entry["Pos"]
        p_team = winner_entry["Team"]

        other_bidders = [x for x in sorted_bids if x["GM"] != w_gm]
        other_bidders.sort(key=lambda x: x["Bid"], reverse=True)

        if other_bidders:
            runner_up_gm = other_bidders[0]["GM"]
            runner_up_bid = other_bidders[0]["Bid"]
            overpay = max(0, w_bid - runner_up_bid)
            all_competing_str = " | ".join([f"{x['GM']}: ${x['Bid']}" for x in other_bidders])
        else:
            runner_up_gm = "Uncontested"
            runner_up_bid = 0
            overpay = w_bid
            all_competing_str = "No competing bids ($0)"

        auction_audit_rows.append({
            "Week": wk,
            "Player": f"{p_name} ({p_pos} - {p_team})",
            "Pos": p_pos,
            "Winner_GM": w_gm,
            "Winning_Bid": w_bid,
            "Winning_Bid_Str": f"${w_bid}",
            "Runner_Up_GM": runner_up_gm,
            "Runner_Up_Bid": runner_up_bid,
            "Runner_Up_Bid_Str": f"${runner_up_bid}" if other_bidders else "-",
            "Overpay_Margin": overpay,
            "Overpay_Str": f"${overpay} (+{int(round(overpay / max(1, w_bid) * 100))}%)" if w_bid > 0 else "$0",
            "Total_Bidders": len(sorted_bids),
            "All_Competing_Bids": all_competing_str,
        })

        for b in sorted_bids:
            g = b["GM"]
            gm_stats[g]["total_bids"] += 1
            gm_stats[g]["all_bids"].append(b["Bid"])
            gm_stats[g]["pos_bids"][p_pos].append(b["Bid"])
            if b["Status"] == "complete":
                gm_stats[g]["wins"] += 1
                gm_stats[g]["winning_bids"].append(b["Bid"])
                gm_stats[g]["overpays"].append(overpay)
            else:
                gm_stats[g]["failed"] += 1

    gm_profile_rows = []
    gm_aggression_map = {}

    for gm_u, rdata in all_gm_rosters.items():
        st_g = gm_stats[gm_u]
        tot_b = st_g["total_bids"]
        wins_b = st_g["wins"]
        lost_b = st_g["failed"]
        max_b = max(st_g["all_bids"]) if st_g["all_bids"] else 0
        avg_b = int(round(float(np.mean(st_g["all_bids"])))) if st_g["all_bids"] else 0
        avg_win_b = int(round(float(np.mean(st_g["winning_bids"])))) if st_g["winning_bids"] else 0

        agg_ratio = max_b / max(1.0, float(total_faab_budget))
        if agg_ratio >= 0.35 or avg_b >= (total_faab_budget * 0.18):
            archetype = "🦈 Aggressive Spender"
            est_bid_pct = 0.42
        elif tot_b >= 4 and avg_b >= (total_faab_budget * 0.08):
            archetype = "🎯 Tactical Sniper"
            est_bid_pct = 0.28
        elif tot_b > 0:
            archetype = "🐢 Value Saver"
            est_bid_pct = 0.14
        else:
            archetype = "💤 Inactive / $0 Bidder"
            est_bid_pct = 0.10

        pos_pref_list = []
        for p_k in ["RB", "WR", "QB", "TE"]:
            if st_g["pos_bids"][p_k]:
                pos_pref_list.append(f"{p_k}: max ${max(st_g['pos_bids'][p_k])}")
        pos_pref_str = " | ".join(pos_pref_list) if pos_pref_list else "No bids logged"

        gm_aggression_map[gm_u] = {
            "archetype": archetype,
            "est_bid_pct": est_bid_pct,
            "max_bid": max_b,
            "avg_bid": avg_b,
            "pos_max": {p_k: max(st_g["pos_bids"][p_k]) if st_g["pos_bids"][p_k] else 0 for p_k in ["QB", "RB", "WR", "TE"]},
        }

        gm_profile_rows.append({
            "GM": gm_u,
            "Status": rdata["chop_status"],
            "Remaining_FAAB": f"${rdata['faab_rem']}",
            "FAAB_Raw": rdata["faab_rem"],
            "Bidder_Archetype": archetype,
            "Total_Bids (Won-Lost)": f"{tot_b} ({wins_b}W - {lost_b}L)",
            "Max_Single_Bid": f"${max_b}",
            "Avg_Bid_Amount": f"${avg_b}",
            "Avg_Winning_Bid": f"${avg_win_b}",
            "Max_Bid_By_Pos": pos_pref_str,
        })

    df_auctions = pd.DataFrame(auction_audit_rows)
    if not df_auctions.empty:
        df_auctions = df_auctions.sort_values(by=["Week", "Winning_Bid"], ascending=[False, False])

    df_gm_profiles = pd.DataFrame(gm_profile_rows).sort_values("FAAB_Raw", ascending=False)
    return df_auctions, df_gm_profiles, gm_aggression_map


def build_player_profile(
    pid,
    curr_week,
    players_db,
    val_map,
    rank_map,
    pos_rank_map,
    pos_rank_num_map,
    schedule_map,
    bye_teams_curr_wk,
    nflverse_injuries,
    dvp_ranks,
    weekly_usage,
    nflverse_metrics,
    ngs_sep_map,
    trending_map,
    recently_dropped_pids,
):
    p = players_db.get(str(pid), {})
    full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or str(pid)
    pos = p.get("position", "N/A")
    team = norm_nfl_team(p.get("team"))
    age = float(p.get("age") or 0.0)
    depth_order = int(p.get("depth_chart_order") or 9)
    raw_inj = p.get("injury_status") or ""
    cn = clean_name(full)

    slp_prac = str(p.get("practice_participation") or "").strip()
    slp_desc = str(p.get("practice_description") or "").strip()
    slp_body = str(p.get("injury_body_part") or "").strip()
    slp_notes = str(p.get("injury_notes") or "").strip()

    nv_inj = nflverse_injuries.get(cn, {})
    prac_raw = slp_prac or nv_inj.get("practice_status", "")
    body_part = slp_body or nv_inj.get("primary_injury", "")

    if "dnp" in prac_raw.lower() or "did not" in prac_raw.lower():
        practice_badge = "🔴 DNP (Did Not Practice)"
        practice_risk = "HIGH RISK"
    elif "lim" in prac_raw.lower() or "lp" in prac_raw.lower():
        practice_badge = "🟡 LP (Limited Practice)"
        practice_risk = "MONITOR"
    elif "full" in prac_raw.lower() or "fp" in prac_raw.lower():
        practice_badge = "🟢 FP (Full Practice)"
        practice_risk = "CLEARED"
    elif raw_inj in ["Out", "IR", "PUP", "Doubtful"]:
        practice_badge = f"🔴 {raw_inj} (No Practice)"
        practice_risk = "OUT"
    elif raw_inj == "Questionable":
        practice_badge = "🟡 Questionable (Check Fri Report)"
        practice_risk = "MONITOR"
    else:
        practice_badge = "🟢 Full / No Injury Designation"
        practice_risk = "HEALTHY"

    if slp_desc:
        practice_badge = f"{practice_badge} [{slp_desc}]"

    is_on_bye = team in bye_teams_curr_wk
    if is_on_bye:
        status_str = f"BYE (Wk {curr_week})" if not raw_inj else f"BYE / {raw_inj}"
    else:
        status_str = f"{raw_inj} ({body_part})" if (raw_inj and body_part) else (raw_inj if raw_inj else "Healthy")

    sched_info = schedule_map.get((curr_week, team))
    if is_on_bye:
        opp_str, vegas_imp, dvp_rk, dvp_str = "BYE", 0.0, 0, "BYE"
    elif sched_info:
        opp_tm = sched_info["opp"]
        prefix = "vs" if sched_info["is_home"] else "@"
        opp_str = f"{prefix} {opp_tm}"
        vegas_imp = sched_info["implied"]
        dvp_rk = dvp_ranks.get((opp_tm, pos), 16)
        dvp_tier = "🟢 Easy" if dvp_rk >= 22 else ("🔴 Tough" if dvp_rk <= 10 else "🟡 Neutral")
        dvp_str = f"#{dvp_rk} ({dvp_tier})"
    else:
        opp_str, vegas_imp, dvp_rk, dvp_str = "TBD", 21.5, 16, "#16 (🟡 Neutral)"

    val = val_map.get(cn, 0)
    rnk = rank_map.get(cn, 999)
    pos_rnk_str = pos_rank_map.get(cn, f"{pos}-")
    pos_rnk_num = pos_rank_num_map.get(cn, 999)
    trend_adds = trending_map.get(str(pid), 0)

    u = weekly_usage.get(str(pid), {})
    snap_list = u.get("snap_pcts", [])
    avg_snap = round(float(np.mean(snap_list)), 1) if snap_list else 0.0
    snap_delta = round(snap_list[-1] - snap_list[0], 1) if len(snap_list) >= 2 else 0.0

    tgt_g = round(float(np.mean(u.get("tgts", [0.0]))), 1) if u.get("tgts") else 0.0
    car_g = round(float(np.mean(u.get("carries", [0.0]))), 1) if u.get("carries") else 0.0
    touches_g = round(tgt_g + car_g, 1)
    rz_g = round(float(np.mean(u.get("rz_touches", [0.0]))), 1) if u.get("rz_touches") else 0.0
    fd_g = round(float(np.mean(u.get("fds", [0.0]))), 1) if u.get("fds") else 0.0
    ppg = round(float(np.mean(u.get("pts", [0.0]))), 1) if u.get("pts") else 0.0
    xfp_g = round(float(np.mean(u.get("xfp", [0.0]))), 1) if u.get("xfp") else 0.0
    fpoe = round(ppg - xfp_g, 1)

    all_pts = u.get("all_season_pts", [])
    if all_pts:
        floor_pts = round(float(np.percentile(all_pts, 25)), 1)
        bust_games = sum(1 for pt in all_pts if pt < 7.0)
        bust_rate = round((bust_games / len(all_pts)) * 100.0, 1)
    else:
        floor_pts, bust_rate = 0.0, 100.0

    if is_on_bye or raw_inj in ["Out", "IR", "PUP", "Doubtful"]:
        chop_floor_score = 0.0
    else:
        prac_penalty = -15.0 if practice_risk == "HIGH RISK" else (-5.0 if practice_risk == "MONITOR" else 0.0)
        chop_floor_score = round(
            max(0.0, (avg_snap * 0.35) + (xfp_g * 2.5) + (floor_pts * 2.0) - (bust_rate * 0.25) + prac_penalty),
            1,
        )

    tot_air = sum(u.get("air_yds", [0.0]))
    tot_rec_yd = sum(u.get("rec_yds", [0.0]))
    calc_pacr = round(tot_rec_yd / tot_air, 2) if tot_air >= 10 else 0.0

    nv = nflverse_metrics.get(cn, {})
    tgt_share = nv.get("tgt_share", round(tgt_g * 3.1, 1))
    wopr = nv.get("wopr", round((tgt_g * 0.045) + (avg_snap * 0.003), 2))
    pacr = nv.get("pacr", calc_pacr)
    sep = ngs_sep_map.get(cn, 0.0)

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

    contingency_score = 0.0
    if pos == "RB":
        tm_off_bonus = max(0.0, vegas_imp - 18.0) * 3.5
        youth_bonus = max(0.0, (26.0 - age) * 6.0) if age > 0 else 10.0
        role_factor = 25.0 if depth_order == 2 else (15.0 if depth_order >= 3 else 10.0)
        contingency_score = round(role_factor + tm_off_bonus + youth_bonus + (avg_snap * 0.4) + (val * 0.015), 1)

    cliff_warning = "✅ Prime Window"
    if pos == "RB" and age >= 26.0:
        cliff_warning = f"⚠️ RB Age Cliff ({age:.0f}yo)"
    elif pos == "WR" and age >= 28.5:
        cliff_warning = f"⚠️ WR Age Cliff ({age:.0f}yo)"
    elif pos == "TE" and age >= 29.5:
        cliff_warning = f"⚠️ TE Age Cliff ({age:.0f}yo)"
    elif pos == "QB" and age >= 33.5:
        cliff_warning = f"⚠️ QB Age Cliff ({age:.0f}yo)"

    drop_info = recently_dropped_pids.get(str(pid))
    chopped_tag = f"🥩 Chopped Wk {drop_info['week']}" if drop_info else "Standard FA"

    return {
        "PID": str(pid),
        "Player": full,
        "Pos": pos,
        "Team": team,
        "Age": age if age > 0 else None,
        "Status": status_str,
        "Raw_Injury": raw_inj,
        "Body_Part": body_part,
        "Practice_Participation": practice_badge,
        "Practice_Risk": practice_risk,
        "Injury_Notes": slp_notes,
        "Is_Bye": is_on_bye,
        "Matchup": opp_str,
        "Vegas_Implied": vegas_imp,
        "DvP_Matchup": dvp_str,
        "DvP_Rank": dvp_rk,
        "Pos_Rank": pos_rnk_str,
        "Pos_Rank_Num": pos_rnk_num,
        "Overall_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
        "Dynasty_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
        "Dynasty_Val": val,
        "Volume_Score": vol_score,
        "Chop_Floor_Score": chop_floor_score,
        "Floor_25th_Pts": floor_pts,
        "Bust_Rate_%": bust_rate,
        "Touches_Opps/G": touches_g,
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
        "Chopped_Status": chopped_tag,
        "24h_Adds": trend_adds,
    }


# ==============================================================================
# SIDEBAR: ONE-CLICK LEAGUE SWITCHER & CONTROLS
# ==============================================================================
with st.sidebar:
    st.title("🏈 Fantasy Command Center")
    selected_preset = st.radio(
        "📁 Select Active League",
        list(LEAGUE_PRESETS.keys()),
        index=0,
    )
    preset_cfg = LEAGUE_PRESETS[selected_preset]

    if preset_cfg["id"]:
        league_id_input = preset_cfg["id"]
        active_mode = preset_cfg["mode"]
        st.caption(f"Connected League ID: `{league_id_input}` ({active_mode} Mode)")
    else:
        league_id_input = st.text_input("🏆 Custom Sleeper League ID", value="1313348860042571776")
        active_mode = st.selectbox("League Strategy Mode", ["DYNASTY", "CHOP"])

    gm_input = st.text_input("👤 Your Sleeper Username", value=preset_cfg["default_gm"])
    lookback_wks = st.slider("📊 Usage Lookback Window (Weeks)", min_value=1, max_value=6, value=3)

    if active_mode == "CHOP":
        st.markdown("---")
        endgame_faab_reserve = st.number_input(
            "🛡️ Endgame FAAB Reserve Floor ($)",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            help="Locks this amount for the final weeks so you can scoop elite stars for $1-$2 when rivals hit $0 FAAB.",
        )
    else:
        endgame_faab_reserve = 0

    st.markdown("---")
    api_key = st.text_input(
        "🔑 Gemini API Key",
        value=st.secrets.get("GEMINI_API_KEY", "") if "GEMINI_API_KEY" in st.secrets else "",
        type="password",
    )
    model_choice = st.selectbox(
        "🤖 Gemini Model (Auto-Fallback Enabled)",
        ["gemini-3.8-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
        index=0,
    )

    if st.button("🔄 Refresh Live Sleeper Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch Master Data for Active League
with st.spinner(f"Syncing {selected_preset} settings, waiver history, live rankings & usage..."):
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
        prev_matchups,
        traded_picks,
        recently_dropped_pids,
        raw_waiver_bids,
        trending_map,
        val_map,
        rank_map,
        pos_rank_map,
        pos_rank_num_map,
        schedule_map,
        bye_teams_curr_wk,
        nflverse_injuries,
        dvp_ranks,
        weekly_usage,
        nflverse_metrics,
        ngs_sep_map,
    ) = fetch_all_war_room_data(league_id_input.strip(), lookback_wks)

# ==============================================================================
# PARSE GRANULAR LEAGUE SETTINGS, FAAB, IR RULES & ROSTERS
# ==============================================================================
total_faab_budget = int(league_settings.get("waiver_budget", 1000 if active_mode == "CHOP" else 100) or 100)
max_ir_slots = int(league_settings.get("reserve_slots", 0) or 0)
waiver_day_num = int(league_settings.get("waiver_day_of_week", 2) or 2)
waiver_day_str = WAIVER_DAY_MAP.get(waiver_day_num, "Wednesday")
waiver_clear_days = int(league_settings.get("waiver_clear_days", 2) or 2)
disable_trades_flag = int(league_settings.get("disable_trades", 0) or 0)
trade_deadline_wk = int(league_settings.get("trade_deadline", 99) or 99)
trades_allowed_str = (
    "🚫 Trades Disabled"
    if (disable_trades_flag == 1 or trade_deadline_wk == 0 or active_mode == "CHOP")
    else f"✅ Trades Open (Deadline: Wk {trade_deadline_wk})"
)

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

curr_wk_pts_by_rid = {}
for m in matchups:
    if m and m.get("roster_id") is not None:
        curr_wk_pts_by_rid[int(m["roster_id"])] = float(m.get("points") or 0.0)

prev_wk_pts_by_rid = {}
for m in prev_matchups:
    if m and m.get("roster_id") is not None:
        prev_wk_pts_by_rid[int(m["roster_id"])] = float(m.get("points") or 0.0)

rostered_pids = set()
my_starters_pids, my_bench_pids, my_ir_pids, my_taxi_pids = [], [], [], []
my_roster_id = None
my_faab_rem = total_faab_budget
all_gm_rosters = {}
rid_to_gm = {}

for r in rosters:
    rid = int(r.get("roster_id"))
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

    is_alive = len(pids) > 0

    all_gm_rosters[gm] = {
        "roster_id": rid,
        "players": pids,
        "starters": starters,
        "bench": active_bench,
        "ir": ir_list,
        "taxi": taxi_list,
        "is_alive": is_alive,
        "chop_status": "🟢 SURVIVING" if is_alive else "💀 CHOPPED",
        "curr_wk_pts": round(curr_wk_pts_by_rid.get(rid, 0.0), 2),
        "prev_wk_pts": round(prev_wk_pts_by_rid.get(rid, 0.0), 2),
        "wins": wins,
        "losses": losses,
        "fpts": round(fpts, 2),
        "max_pf": round(ppts, 2) if ppts > 0 else round(fpts * 1.12, 2),
        "coach_eff_%": coach_eff,
        "faab_rem": faab_rem,
        "faab_used": faab_used,
    }

    if gm.lower() == gm_input.strip().lower():
        my_roster_id = rid
        my_starters_pids = starters
        my_bench_pids = active_bench
        my_ir_pids = ir_list
        my_taxi_pids = taxi_list
        my_faab_rem = faab_rem

# Max Spendable FAAB respecting the $50 Endgame Reserve Rule in Chop Mode
my_spendable_faab = max(0, my_faab_rem - endgame_faab_reserve) if active_mode == "CHOP" else my_faab_rem

df_auctions, df_gm_profiles, gm_aggression_map = analyze_waiver_auctions_and_gm_profiles(
    raw_waiver_bids, all_gm_rosters, total_faab_budget
)


def build_df_from_pids(pid_list):
    if not pid_list:
        return pd.DataFrame()
    return pd.DataFrame([
        build_player_profile(
            p, curr_week, players_db, val_map, rank_map, pos_rank_map, pos_rank_num_map,
            schedule_map, bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
            nflverse_metrics, ngs_sep_map, trending_map, recently_dropped_pids
        )
        for p in pid_list
    ])


my_starters_df = build_df_from_pids(my_starters_pids)
my_bench_df = build_df_from_pids(my_bench_pids)
my_taxi_df = build_df_from_pids(my_taxi_pids)
my_ir_df = build_df_from_pids(my_ir_pids)
my_full_roster_df = build_df_from_pids(my_starters_pids + my_bench_pids + my_taxi_pids + my_ir_pids)

# Build Unrostered Waiver Wire Pool
unrostered_list = []
for pid, p in players_db.items():
    if str(pid) not in rostered_pids and p.get("position") in ["QB", "RB", "WR", "TE"] and p.get("team"):
        prof = build_player_profile(
            pid, curr_week, players_db, val_map, rank_map, pos_rank_map, pos_rank_num_map,
            schedule_map, bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
            nflverse_metrics, ngs_sep_map, trending_map, recently_dropped_pids
        )
        if (
            prof["Volume_Score"] >= 10.0
            or prof["Dynasty_Val"] >= 100
            or prof["24h_Adds"] >= 15
            or prof["Contingency_Score"] >= 45.0
            or "Chopped" in prof["Chopped_Status"]
        ):
            unrostered_list.append(prof)

df_waivers = pd.DataFrame(unrostered_list)
if not df_waivers.empty:
    sort_cols = ["Chop_Floor_Score", "xFP/G"] if active_mode == "CHOP" else ["Volume_Score", "xFP/G", "Dynasty_Val"]
    df_waivers = df_waivers.sort_values(by=sort_cols, ascending=False)

# ==============================================================================
# RIVAL POSITIONAL NEED & PREDICTIVE FAAB BIDDING ENGINE (RESPECTING $50 RESERVE)
# ==============================================================================
rival_pos_needs = defaultdict(list)
surv_by_pts = sorted(
    [(g, d) for g, d in all_gm_rosters.items() if d["is_alive"]],
    key=lambda x: x[1]["fpts"],
)
bottom4_danger_gms = {g for g, _ in surv_by_pts[:4]}

for r_gm, r_data in all_gm_rosters.items():
    if not r_data["is_alive"] or r_gm.lower() == gm_input.strip().lower():
        continue
    r_faab = r_data["faab_rem"]
    if r_faab <= 0:
        continue

    agg_info = gm_aggression_map.get(r_gm, {"est_bid_pct": 0.20, "max_bid": 0})
    is_desperate = r_gm in bottom4_danger_gms
    desp_mult = 1.35 if is_desperate else 1.0

    pos_weakness_reasons = defaultdict(list)

    for s_pid in r_data["starters"]:
        p_m = players_db.get(str(s_pid), {})
        pos = p_m.get("position", "N/A")
        if pos not in ["QB", "RB", "WR", "TE"]:
            continue
        tm = norm_nfl_team(p_m.get("team"))
        inj_st = p_m.get("injury_status") or ""
        prac_st = str(p_m.get("practice_participation") or "").lower()
        p_nm = f"{p_m.get('first_name', '')} {p_m.get('last_name', '')}".strip()

        u_st = weekly_usage.get(str(s_pid), {})
        s_xfp = round(float(np.mean(u_st.get("xfp", [0.0]))), 1) if u_st.get("xfp") else 0.0

        if tm in bye_teams_curr_wk:
            pos_weakness_reasons[pos].append(f"{p_nm} on BYE")
        elif inj_st in ["Out", "IR", "Doubtful", "PUP"] or "dnp" in prac_st:
            pos_weakness_reasons[pos].append(f"{p_nm} ({inj_st or 'DNP'})")
        elif s_xfp < 7.5 and pos in ["RB", "WR"]:
            pos_weakness_reasons[pos].append(f"{p_nm} low vol ({s_xfp} xFP)")
        elif s_xfp < 6.0 and pos == "TE":
            pos_weakness_reasons[pos].append(f"{p_nm} low vol ({s_xfp} xFP)")

    for pos in ["QB", "RB", "WR", "TE"]:
        reasons = pos_weakness_reasons[pos]
        if is_desperate and not reasons:
            reasons.append("Bottom-4 Cut Danger")
        if reasons:
            base_pct = agg_info["est_bid_pct"] * desp_mult
            phase_mult = 0.85 if curr_week <= 5 else (1.10 if curr_week <= 10 else 1.35)
            est_pct = min(0.85, max(0.08, base_pct * phase_mult))
            est_bid = int(round(r_faab * est_pct))
            rival_pos_needs[pos].append({
                "gm": r_gm,
                "faab": r_faab,
                "reason": ", ".join(reasons[:2]),
                "est_bid": est_bid,
            })

for pos in ["QB", "RB", "WR", "TE"]:
    rival_pos_needs[pos].sort(key=lambda x: x["est_bid"], reverse=True)


def predict_smart_faab_for_player(wr_row, spendable_faab, max_rival_faab, avg_surv_faab):
    """Calculates predicted rival threats and optimal bids capped strictly at spendable_faab (preserving $50 reserve)."""
    pos = wr_row["Pos"]
    floor_sc = wr_row["Chop_Floor_Score"]
    xfp = wr_row["xFP/G"]

    if floor_sc >= 35.0 or xfp >= 14.5:
        tier_factor = 1.0
    elif floor_sc >= 24.0 or xfp >= 10.5:
        tier_factor = 0.55
    else:
        tier_factor = 0.20

    needy_rivals = rival_pos_needs.get(pos, [])
    if needy_rivals:
        top_threats = needy_rivals[:2]
        threat_strs = []
        max_est_threat_bid = 0
        max_needy_faab = 0
        for t in top_threats:
            scaled_bid = min(t["faab"], int(round(t["est_bid"] * tier_factor)))
            max_est_threat_bid = max(max_est_threat_bid, scaled_bid)
            max_needy_faab = max(max_needy_faab, t["faab"])
            threat_strs.append(f"{t['gm']} (${t['faab']} FAAB | Est: ~${scaled_bid} [{t['reason']}])")
        threats_summary = " | ".join(threat_strs)
    else:
        max_est_threat_bid = int(round(avg_surv_faab * 0.15 * tier_factor))
        max_needy_faab = max_rival_faab
        threats_summary = "No immediate starter hole detected"

    raw_opt = max_est_threat_bid + 3
    if raw_opt % 5 == 0:
        raw_opt += 2
    elif raw_opt % 10 == 0:
        raw_opt += 6

    if spendable_faab <= 0:
        return threats_summary, "$0 (At Reserve Floor)", "$0 (At Reserve Floor)", "$0"

    smart_bid = min(spendable_faab, max(1, raw_opt))
    smart_tag = f"${smart_bid}" if raw_opt <= spendable_faab else f"${smart_bid} (Capped by Reserve)"

    lock_target = max_needy_faab + 1
    if lock_target <= spendable_faab:
        lock_tag = f"${lock_target} (Guaranteed)"
    else:
        lock_tag = f"${spendable_faab} (Max Spendable)"

    val_bid = min(spendable_faab, max(1, int(round(max_est_threat_bid * 0.65))))
    return threats_summary, smart_tag, lock_tag, f"${val_bid}"


# ==============================================================================
# LEAGUE-WIDE POSITIONAL RANKINGS & FULL ROSTER SCOUT ENGINE (BOTH LEAGUES)
# ==============================================================================
def build_league_positional_scout(all_gm_rosters, players_db, val_map, pos_rank_num_map):
    pos_matrix_rows = []
    for gm_u, rdata in all_gm_rosters.items():
        if not rdata["is_alive"] and len(rdata["players"]) == 0:
            continue
        pos_vals = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        top12_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
        top24_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}

        for pid in rdata["players"]:
            p = players_db.get(str(pid), {})
            pos = p.get("position")
            if pos in pos_vals:
                cn = clean_name(f"{p.get('first_name', '')} {p.get('last_name', '')}")
                v = val_map.get(cn, 0)
                pr_num = pos_rank_num_map.get(cn, 999)
                pos_vals[pos] += v
                if pr_num <= 12:
                    top12_counts[pos] += 1
                if pr_num <= 24:
                    top24_counts[pos] += 1

        tot_v = sum(pos_vals.values())
        pos_matrix_rows.append({
            "GM": gm_u,
            "Record_Status": f"{rdata['wins']}-{rdata['losses']}" if active_mode == "DYNASTY" else rdata["chop_status"],
            "FAAB": f"${rdata['faab_rem']}",
            "Total_Val": tot_v,
            "QB_Val": pos_vals["QB"],
            "RB_Val": pos_vals["RB"],
            "WR_Val": pos_vals["WR"],
            "TE_Val": pos_vals["TE"],
            "QB_Studs (Top12/24)": f"{top12_counts['QB']} / {top24_counts['QB']}",
            "RB_Studs (Top12/24)": f"{top12_counts['RB']} / {top24_counts['RB']}",
            "WR_Studs (Top12/24)": f"{top12_counts['WR']} / {top24_counts['WR']}",
            "TE_Studs (Top12/24)": f"{top12_counts['TE']} / {top24_counts['TE']}",
        })

    df_pos_scout = pd.DataFrame(pos_matrix_rows)
    if not df_pos_scout.empty:
        for pos in ["QB", "RB", "WR", "TE"]:
            df_pos_scout[f"{pos}_League_Rank"] = (
                df_pos_scout[f"{pos}_Val"].rank(ascending=False, method="min").astype(int)
            )
            df_pos_scout[f"{pos}_Rank_Unit"] = df_pos_scout.apply(
                lambda r: f"#{r[f'{pos}_League_Rank']} ({r[f'{pos}_Val']:,} val | {r[f'{pos}_Studs (Top12/24)']} T12/T24)",
                axis=1,
            )
        df_pos_scout = df_pos_scout.sort_values("Total_Val", ascending=False)
    return df_pos_scout


df_league_pos_scout = build_league_positional_scout(all_gm_rosters, players_db, val_map, pos_rank_num_map)


def render_rival_roster_scout_tab(tab_key_prefix):
    st.subheader("👁️ League-Wide Positional Power Rankings & Full Roster Scout")
    st.caption(
        "Compare every franchise's positional ranking (#1 to #Last in `QB`, `RB`, `WR`, `TE`), count of Top-12 / Top-24 assets, and inspect any rival GM's complete roster with live positional rankings (`QB1`, `RB8`, `WR14`, `TE3`)."
    )

    if not df_league_pos_scout.empty:
        st.markdown("#### 📊 League Positional Power Rankings (Value & Top-12 / Top-24 Stud Counts)")
        scout_summary_cols = [
            "GM", "Record_Status", "FAAB", "Total_Val",
            "QB_Rank_Unit", "RB_Rank_Unit", "WR_Rank_Unit", "TE_Rank_Unit"
        ]
        st.dataframe(df_league_pos_scout[scout_summary_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🔍 Deep Team Roster Inspector (Starters, Active Bench, Taxi & IR)")

    alive_or_all_gms = sorted(all_gm_rosters.keys())
    default_idx = next(
        (i for i, g in enumerate(alive_or_all_gms) if g.lower() != gm_input.strip().lower()),
        0,
    )
    sc_col1, sc_col2 = st.columns([1, 2])
    inspected_gm = sc_col1.selectbox(
        "Select GM Roster to Inspect",
        alive_or_all_gms,
        index=default_idx,
        key=f"{tab_key_prefix}_gm_sel",
    )
    pos_filter_scout = sc_col2.multiselect(
        "Filter by Position",
        ["QB", "RB", "WR", "TE"],
        default=["QB", "RB", "WR", "TE"],
        key=f"{tab_key_prefix}_pos_sel",
    )

    r_info = all_gm_rosters.get(inspected_gm, {})
    if r_info:
        st_df = build_df_from_pids(r_info.get("starters", []))
        bn_df = build_df_from_pids(r_info.get("bench", []))
        tx_df = build_df_from_pids(r_info.get("taxi", []))
        ir_df_sc = build_df_from_pids(r_info.get("ir", []))

        scout_cols = [
            "Player", "Pos", "Pos_Rank", "Overall_Rank", "Team", "Age", "Status",
            "Practice_Participation", "Matchup", "Dynasty_Val", "xFP/G", "Recent_PPG",
            "Snap_%", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR"
        ]

        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown(f"##### 🏆 {inspected_gm}'s Active Starters")
            if not st_df.empty:
                st_v = st_df[st_df["Pos"].isin(pos_filter_scout)]
                st.dataframe(st_v[scout_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No active starters.")
        with c_b:
            st.markdown(f"##### 🪑 {inspected_gm}'s Active Bench (Sorted by Live Value)")
            if not bn_df.empty:
                bn_v = bn_df[bn_df["Pos"].isin(pos_filter_scout)].sort_values("Dynasty_Val", ascending=False)
                st.dataframe(bn_v[scout_cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No active bench players.")

        if active_mode == "DYNASTY" or not ir_df_sc.empty:
            c_c, c_d = st.columns(2)
            with c_c:
                if active_mode == "DYNASTY":
                    st.markdown(f"##### 🚕 {inspected_gm}'s Taxi Squad")
                    if not tx_df.empty:
                        tx_v = tx_df[tx_df["Pos"].isin(pos_filter_scout)].sort_values("Dynasty_Val", ascending=False)
                        st.dataframe(tx_v[scout_cols], use_container_width=True, hide_index=True)
                    else:
                        st.caption("Empty Taxi Squad.")
            with c_d:
                st.markdown(f"##### 🏥 {inspected_gm}'s Injured Reserve (IR)")
                if not ir_df_sc.empty:
                    ir_v = ir_df_sc[ir_df_sc["Pos"].isin(pos_filter_scout)]
                    st.dataframe(ir_v[scout_cols], use_container_width=True, hide_index=True)
                else:
                    st.caption("Empty IR.")


# ==============================================================================
# AUTOMATED ALERTS & PRACTICE PREVENTION ENGINE (BOTH LEAGUES)
# ==============================================================================
lineup_alerts = []
ir_alerts = []
volume_swap_alerts = []
practice_prevention_rows = []

if not my_starters_df.empty:
    for _, s_row in my_starters_df.iterrows():
        if s_row["Is_Bye"]:
            lineup_alerts.append(
                f"🛑 **BYE WEEK ALERT:** Starter **{s_row['Player']} ({s_row['Pos']} - {s_row['Team']})** is on a **BYE** in Week {curr_week}! Swap them out immediately."
            )
        elif s_row["Raw_Injury"] in ["Out", "Doubtful", "IR", "PUP", "Questionable", "Sus"] or s_row["Practice_Risk"] in ["HIGH RISK", "MONITOR"]:
            lineup_alerts.append(
                f"🚨 **Starter Practice/Injury Alert:** **{s_row['Player']} ({s_row['Pos']})** — Status: **{s_row['Status']}** | Practice: **{s_row['Practice_Participation']}**."
            )

open_ir_spots = max(0, max_ir_slots - len(my_ir_pids))
if open_ir_spots > 0:
    active_non_ir_df = pd.concat([my_starters_df, my_bench_df], ignore_index=True)
    if not active_non_ir_df.empty:
        for _, r_row in active_non_ir_df.iterrows():
            if r_row["Raw_Injury"] in allowed_ir_tags:
                ir_alerts.append(
                    f"🏥 **Verified IR Stash Eligible:** **{r_row['Player']}** carries designation **`{r_row['Raw_Injury']}`** (Allowed by league IR rules). Move them to one of your **{open_ir_spots} open IR slot(s)**!"
                )

if not my_full_roster_df.empty:
    healthy_bench = my_bench_df[
        (~my_bench_df["Is_Bye"]) & (my_bench_df["Practice_Risk"].isin(["HEALTHY", "CLEARED"]))
    ] if not my_bench_df.empty else pd.DataFrame()

    for _, r_row in my_full_roster_df.iterrows():
        if r_row["Raw_Injury"] or r_row["Practice_Risk"] != "HEALTHY":
            pos = r_row["Pos"]
            is_starter = r_row["PID"] in my_starters_pids
            role_tag = "🏆 STARTER" if is_starter else "🪑 Bench / IR"

            pos_bench = healthy_bench[healthy_bench["Pos"] == pos] if not healthy_bench.empty else pd.DataFrame()
            if not pos_bench.empty:
                best_b = pos_bench.sort_values("Chop_Floor_Score" if active_mode == "CHOP" else "Volume_Score", ascending=False).iloc[0]
                bench_pivot = f"{best_b['Player']} ({best_b['Matchup']} | {best_b['xFP/G']} xFP/G)"
            else:
                bench_pivot = "⚠️ No Healthy Bench Pivot at Pos!"

            if not df_waivers.empty:
                w_pos = df_waivers[(df_waivers["Pos"] == pos) & (~df_waivers["Is_Bye"]) & (df_waivers["Practice_Risk"] == "HEALTHY")]
                if not w_pos.empty:
                    best_w = w_pos.iloc[0]
                    waiver_pivot = f"{best_w['Player']} ({best_w['Team']} | {best_w['Snap_%']}% Snaps, {best_w['xFP/G']} xFP/G)"
                else:
                    waiver_pivot = "None"
            else:
                waiver_pivot = "None"

            action_rec = (
                "🚨 PIVOT / CLAIM INSURANCE NOW (Before Sunday)"
                if (is_starter and r_row["Practice_Risk"] in ["HIGH RISK", "OUT"])
                else ("👀 Check Friday Report (Prepare Pivot)" if is_starter else "Monitor on Bench")
            )

            practice_prevention_rows.append({
                "Role": role_tag,
                "Player": r_row["Player"],
                "Pos": pos,
                "Team": r_row["Team"],
                "Injury_Status": r_row["Status"],
                "Practice_Report": r_row["Practice_Participation"],
                "Pre_Sunday_Action": action_rec,
                "Best_Healthy_Bench_Pivot": bench_pivot,
                "Top_Waiver_Insurance": waiver_pivot,
                "Notes": r_row["Injury_Notes"] or r_row["Body_Part"] or "-",
            })

if not my_starters_df.empty and not my_bench_df.empty:
    eligible_bench = my_bench_df[
        (~my_bench_df["Is_Bye"]) & (~my_bench_df["Raw_Injury"].isin(["Out", "Doubtful", "IR", "PUP", "NA", "Sus"]))
    ]
    metric_key = "Chop_Floor_Score" if active_mode == "CHOP" else "Volume_Score"
    for pos in ["QB", "RB", "WR", "TE"]:
        st_pos = my_starters_df[my_starters_df["Pos"] == pos]
        bn_pos = eligible_bench[eligible_bench["Pos"] == pos]
        for _, s_r in st_pos.iterrows():
            for _, b_r in bn_pos.iterrows():
                if s_r["Is_Bye"] or (
                    b_r[metric_key] >= s_r[metric_key] + 7.5 and b_r["xFP/G"] >= s_r["xFP/G"]
                ):
                    volume_swap_alerts.append(
                        f"⚡ **Start/Sit Upgrade ({pos}):** Start Bench **{b_r['Player']}** ({b_r['Matchup']}, {b_r['xFP/G']} xFP/G, {b_r['Bust_Rate_%']}% Bust Rate) over **{s_r['Player']}** ({s_r['xFP/G']} xFP/G, {s_r['Bust_Rate_%']}% Bust Rate)."
                    )

# ==============================================================================
# GRANULAR LEAGUE RULES SUMMARY BANNER
# ==============================================================================
fmt_tag = "Superflex" if is_superflex else "1QB"
tep_val = float(scoring_settings.get("bonus_rec_te", 0.0) or 0.0)
ppr_val = float(scoring_settings.get("rec", 1.0) or 1.0)
pass_td_pts = float(scoring_settings.get("pass_td", 4.0) or 4.0)
rec_fd_pts = float(scoring_settings.get("bonus_rec_fd", 0.0) or 0.0)
rush_fd_pts = float(scoring_settings.get("bonus_rush_fd", 0.0) or 0.0)

st.markdown(f"## {selected_preset} — {curr_season} Week {curr_week}")
st.caption(
    f"**Granular League Rules:** `{len(rosters)} Teams` | `{fmt_tag}` | `{ppr_val} PPR` | `Pass TD: {pass_td_pts}pt` | "
    f"`TE Prem: +{tep_val}` | `1st Down Bonus: +{rec_fd_pts} Rec / +{rush_fd_pts} Rush` | "
    f"**Waivers:** `Clears {waiver_day_str}s ({waiver_clear_days}d wait)` | **FAAB Budget:** `${total_faab_budget}` | "
    f"**Trades:** `{trades_allowed_str}` | **Teams on Bye (Wk {curr_week}):** `{', '.join(bye_teams_curr_wk) if bye_teams_curr_wk else 'None'}`"
)

# ==============================================================================
# MODE A: 🪓 CHOP / GUILLOTINE LEAGUE COMMAND CENTER (18 TEAMS)
# ==============================================================================
if active_mode == "CHOP":
    surviving_teams = [(gm, d) for gm, d in all_gm_rosters.items() if d["is_alive"]]
    chopped_teams = [(gm, d) for gm, d in all_gm_rosters.items() if not d["is_alive"]]

    any_live_pts = any(d["curr_wk_pts"] > 0 for _, d in surviving_teams)
    if any_live_pts:
        surviving_sorted = sorted(surviving_teams, key=lambda x: (x[1]["curr_wk_pts"], x[1]["fpts"]), reverse=True)
        cut_line_gm, cut_line_data = surviving_sorted[-1]
        cut_line_score = cut_line_data["curr_wk_pts"]
        my_curr_score = all_gm_rosters.get(gm_input, {}).get("curr_wk_pts", 0.0)
        safety_buffer = round(my_curr_score - cut_line_score, 2)
        score_label = f"Wk {curr_week} Live Pts"
    else:
        surviving_sorted = sorted(surviving_teams, key=lambda x: x[1]["fpts"], reverse=True)
        cut_line_gm, cut_line_data = surviving_sorted[-1]
        cut_line_score = cut_line_data["fpts"]
        my_curr_score = all_gm_rosters.get(gm_input, {}).get("fpts", 0.0)
        safety_buffer = round(my_curr_score - cut_line_score, 2)
        score_label = "Season Total Pts (Pre-Kickoff)"

    rival_surviving_faabs = [d["faab_rem"] for g, d in surviving_teams if g.lower() != gm_input.lower()]
    max_rival_faab = max(rival_surviving_faabs) if rival_surviving_faabs else 0
    avg_surviving_faab = int(np.mean([d["faab_rem"] for _, d in surviving_teams])) if surviving_teams else 0

    c_k1, c_k2, c_k3, c_k4, c_k5 = st.columns(5)
    c_k1.metric("Teams Alive / Total", f"{len(surviving_teams)} / {len(all_gm_rosters)}", delta=f"{len(chopped_teams)} Chopped")
    c_k2.metric(f"Guillotine Cut Line ({cut_line_gm})", f"{cut_line_score:.2f} pts", help=score_label)
    c_k3.metric("Your Safety Buffer", f"{safety_buffer:+.2f} pts", delta="Safe Above Cut Line" if safety_buffer > 10 else "⚠️ DANGER ZONE")
    c_k4.metric("Total FAAB Remaining", f"${my_faab_rem} / ${total_faab_budget}", delta=f"${endgame_faab_reserve} Locked for Endgame")
    c_k5.metric("Max Spendable FAAB Now", f"${my_spendable_faab}", delta=f"Avg Alive: ${avg_surviving_faab}")

    ctab1, ctab2, ctab3, ctab4, ctab5, ctab6, ctab7 = st.tabs([
        "💬 AI Survival GM (Chat)",
        "🪓 Cut-Line & 18-Team Board",
        "🛡️ Floor & Bust-Rate Start/Sit",
        "🥩 Waiver Radar & Bid Predictor",
        "🕵️‍♂️ League Auction Archive",
        "👁️ Rival Roster & Positional Scout",
        "🏥 Practice & Injury Tracker",
    ])

    # --------------------------------------------------------------------------
    # CHOP TAB 1: AI SURVIVAL GM CHAT (100% ENGLISH + $50 RESERVE ENFORCED)
    # --------------------------------------------------------------------------
    with ctab1:
        st.subheader("💬 AI Guillotine Survival Strategist")
        st.caption(
            f"Synced with live waiver auctions, rival positional needs, pre-Sunday practice reports, and your **${endgame_faab_reserve} Endgame FAAB Reserve Rule** (Max Spendable Now: **${my_spendable_faab}**)."
        )

        if "chop_chat_messages" not in st.session_state:
            st.session_state.chop_chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hey Carlos! Welcome to **The Chop Block** Survival Command Center (Week {curr_week}).\n\n"
                        f"• **Surviving Teams:** `{len(surviving_teams)} / {len(all_gm_rosters)}`\n"
                        f"• **Your Total FAAB:** `${my_faab_rem}` | **Locked Endgame Reserve:** `${endgame_faab_reserve}` | **Max Spendable Now:** `${my_spendable_faab}`\n"
                        f"• **Highest Rival FAAB:** `${max_rival_faab}` | **Surviving Avg FAAB:** `${avg_surviving_faab}`\n\n"
                        "Ask me about **upcoming waiver wire targets and optimal bids**, **high-floor start/sit decisions**, or **pre-Sunday practice report pivots**!"
                    ),
                }
            ]

        cb1, cb2, cb3, cb4 = st.columns(4)
        chop_quick = None
        if cb1.button("🥩 Predict Upcoming Waiver Bids", use_container_width=True):
            chop_quick = (
                f"I have ${my_faab_rem} total FAAB, with a strict ${endgame_faab_reserve} Endgame Reserve (so my Max Spendable FAAB right now is ${my_spendable_faab}). "
                "Analyze the top available players on waivers this week, identify which surviving rivals need those positions, and recommend exact optimal bids."
            )
        if cb2.button("🛡️ Audit My Lineup for Floor & Bust Risk", use_container_width=True):
            chop_quick = (
                f"Audit my Week {curr_week} starting lineup vs my bench for an 18-team Chop league. "
                "Flag any starter with a high Bust Rate (%), low Floor Score, Bye week, or DNP/LP practice report, and tell me who to swap in."
            )
        if cb3.button("🕵️‍♂️ Analyze Rival Bidding Habits", use_container_width=True):
            chop_quick = (
                "Analyze the League Auction Archive and rival bidding profiles. Which surviving managers are aggressive overspenders, who is running out of FAAB, and how can I exploit their habits?"
            )
        if cb4.button("🏥 Pre-Sunday Practice & Injury Check", use_container_width=True):
            chop_quick = (
                "Review the Practice & Injury Prevention table for my roster. Which players are at risk of missing Sunday, and what is my safest high-floor pivot?"
            )

        for msg in st.session_state.chop_chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        chop_query = st.chat_input("Ask about upcoming waiver bids, rival FAAB threats, floor start/sit, or practice reports...") or chop_quick

        if chop_query:
            st.session_state.chop_chat_messages.append({"role": "user", "content": chop_query})
            with st.chat_message("user"):
                st.markdown(chop_query)

            if not api_key:
                with st.chat_message("assistant"):
                    st.warning("Please enter your Gemini API Key in the left sidebar.")
            else:
                chop_sys_instruction = f"""
                You are an elite Guillotine (Chop) Fantasy Football Strategist advising Carlos ('{gm_input}') in an 18-team Chop League (League ID: {league_id_input}).
                Respond strictly in English.

                RULES & ENDGAME FAAB STRATEGY:
                - 18 Teams, 1QB, {ppr_val} PPR. Trades are strictly DISABLED.
                - 1 Team is eliminated every week (lowest weekly points gets chopped and their entire roster drops to waivers).
                - Starting FAAB Budget: $1,000. Carlos currently has ${my_faab_rem} total FAAB remaining.
                - CRITICAL ENDGAME RESERVE RULE: Carlos locks an untouched ${endgame_faab_reserve} FAAB reserve for the final weeks of the season (when rival teams hit $0 FAAB and elite stars can be won for $1-$2 bids). Therefore, Carlos's MAXIMUM SPENDABLE FAAB right now is strictly ${my_spendable_faab}. Never recommend a bid or combination of bids that exceeds ${my_spendable_faab}!
                - Highest Rival FAAB: ${max_rival_faab}. Average Surviving Team FAAB: ${avg_surviving_faab}.
                - Teams on BYE in Week {curr_week}: {bye_teams_curr_wk}.

                GUILLOTINE STRATEGY MANDATES:
                1. NEVER recommend starting any player on a BYE ({bye_teams_curr_wk}) or with an Out/IR status.
                2. Focus strictly on current and upcoming waiver cycles and lineup decisions—do not fixate on past completed claims unless explicitly asked.
                3. Prioritize high Chop_Floor_Score, high xFP/G, high Snap %, high Touches_Opps/G, and LOW Bust_Rate_% (% of games < 7.0 pts).
                4. Flag any starter with a 🔴 DNP or 🟡 LP practice report so Carlos can pivot before Sunday.

                LIVE DATA CONTEXT:
                - My Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
                - My Bench: {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
                - Practice & Injury Alerts: {practice_prevention_rows}
                - Top 25 Available Waiver / Chopped Players: {df_waivers.head(25).to_dict(orient='records') if not df_waivers.empty else []}
                - Surviving Rival Positional Needs & Estimated Bids: {dict(rival_pos_needs)}
                - Rival GM Bidding Behavior Profiles: {df_gm_profiles.to_dict(orient='records') if not df_gm_profiles.empty else []}
                - Historical League Waiver Auctions: {df_auctions.head(30).to_dict(orient='records') if not df_auctions.empty else []}
                """
                try:
                    client = genai.Client(api_key=api_key)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing waiver threats, floor metrics & practice reports..."):
                            ans, used_m = generate_ai_response_with_fallback(
                                client, model_choice, chop_query, chop_sys_instruction
                            )
                            st.markdown(ans)
                            st.session_state.chop_chat_messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Gemini API Error: {e}")

    # --------------------------------------------------------------------------
    # CHOP TAB 2: GUILLOTINE CUT-LINE & 18-TEAM FAAB POWER MATRIX
    # --------------------------------------------------------------------------
    with ctab2:
        st.subheader(f"🪓 Week {curr_week} Guillotine Cut-Line & 18-Team Survival Leaderboard")
        board_rows = []
        for rank_idx, (gm_u, d) in enumerate(surviving_sorted, 1):
            buffer_vs_cut = round((d["curr_wk_pts"] if any_live_pts else d["fpts"]) - cut_line_score, 2)
            if rank_idx == len(surviving_sorted):
                zone_tag = "🩸 ON THE CHOPPING BLOCK (Cut Line)"
            elif rank_idx >= len(surviving_sorted) - 2:
                zone_tag = "⚠️ DANGER ZONE (Bottom 3)"
            else:
                zone_tag = "🟢 Safe Cushion"

            board_rows.append({
                "Survival_Rank": f"#{rank_idx}",
                "GM": f"🦅 {gm_u}" if gm_u.lower() == gm_input.lower() else gm_u,
                "Status": zone_tag,
                f"Wk_{curr_week}_Live_Pts": d["curr_wk_pts"],
                f"Wk_{max(1, curr_week - 1)}_Pts": d["prev_wk_pts"],
                "Season_Total_Pts": d["fpts"],
                "Buffer_vs_Cut_Line": f"{buffer_vs_cut:+.2f}",
                "Remaining_FAAB": f"${d['faab_rem']}",
                "FAAB_Spent": f"${d['faab_used']}",
                "Bidder_Profile": gm_aggression_map.get(gm_u, {}).get("archetype", "-"),
            })

        for gm_u, d in chopped_teams:
            board_rows.append({
                "Survival_Rank": "💀 OUT",
                "GM": gm_u,
                "Status": "💀 CHOPPED (Eliminated)",
                f"Wk_{curr_week}_Live_Pts": 0.0,
                f"Wk_{max(1, curr_week - 1)}_Pts": d["prev_wk_pts"],
                "Season_Total_Pts": d["fpts"],
                "Buffer_vs_Cut_Line": "-",
                "Remaining_FAAB": f"${d['faab_rem']}",
                "FAAB_Spent": f"${d['faab_used']}",
                "Bidder_Profile": gm_aggression_map.get(gm_u, {}).get("archetype", "-"),
            })

        st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # CHOP TAB 3: FLOOR & BUST-RATE START/SIT OPTIMIZER
    # --------------------------------------------------------------------------
    with ctab3:
        st.subheader(f"🛡️ Week {curr_week} High-Floor & Anti-Bust Start/Sit Optimizer")
        for al in lineup_alerts:
            st.error(al)
        for sw in volume_swap_alerts:
            st.warning(sw)
        if not (lineup_alerts or volume_swap_alerts):
            st.success("✅ Maximum Floor Lineup: All starters are active, practicing, and carry your highest Floor Scores!")

        chop_cols = [
            "Player", "Pos", "Pos_Rank", "Team", "Status", "Practice_Participation", "Matchup",
            "Vegas_Implied", "DvP_Matchup", "Chop_Floor_Score", "Floor_25th_Pts",
            "Bust_Rate_%", "xFP/G", "Recent_PPG", "Touches_Opps/G", "Snap_%", "RZ_Touch/G"
        ]

        st.markdown("#### 🏆 Current Starting Lineup")
        if not my_starters_df.empty:
            st.dataframe(my_starters_df[chop_cols], use_container_width=True, hide_index=True)

        st.markdown("#### 🪑 Active Bench (Sorted by Highest Chop Floor Score)")
        if not my_bench_df.empty:
            st.dataframe(
                my_bench_df.sort_values(by=["Is_Bye", "Chop_Floor_Score"], ascending=[True, False])[chop_cols],
                use_container_width=True,
                hide_index=True,
            )

    # --------------------------------------------------------------------------
    # CHOP TAB 4: CHOPPED ROSTER GOLDMINE & PREDICTIVE RIVAL BID ADVISOR
    # --------------------------------------------------------------------------
    with ctab4:
        st.subheader("🥩 Newly Chopped Roster Goldmine & Predictive Rival FAAB Advisor")
        st.info(
            f"🛡️ **Endgame Reserve Active:** Total FAAB: **`${my_faab_rem}`** | Locked Endgame Floor: **`${endgame_faab_reserve}`** (for late-season $1-$2 sweeps) | **Max Spendable Now: `${my_spendable_faab}`** | Highest Rival: `${max_rival_faab}`"
        )

        w_f1, w_f2, w_f3 = st.columns(3)
        chop_pos_filter = w_f1.selectbox("Filter Position", ["ALL", "RB", "WR", "TE", "QB"], key="chop_pos")
        only_chopped_toggle = w_f2.checkbox("🥩 Show Only Recently Chopped / Dropped Players", value=False)
        chop_sort_col = w_f3.selectbox(
            "Sort Waiver Targets By",
            ["Chop_Floor_Score", "xFP/G", "Recent_PPG", "Snap_%", "24h_Adds"],
        )

        if not df_waivers.empty:
            cw = df_waivers.copy()
            if chop_pos_filter != "ALL":
                cw = cw[cw["Pos"] == chop_pos_filter]
            if only_chopped_toggle:
                cw = cw[cw["Chopped_Status"].str.contains("Chopped")]

            threats_col, smart_bids, lock_bids, val_bids = [], [], [], []
            for _, wr in cw.iterrows():
                th_str, sm_b, lk_b, vl_b = predict_smart_faab_for_player(
                    wr, my_spendable_faab, max_rival_faab, avg_surviving_faab
                )
                threats_col.append(th_str)
                smart_bids.append(sm_b)
                lock_bids.append(lk_b)
                val_bids.append(vl_b)

            cw["🎯 Rival Threats & Est. Bid"] = threats_col
            cw["💡 Recommended Optimal Bid"] = smart_bids
            cw["🔒 Lock-In Bid (Capped)"] = lock_bids
            cw["💰 Value Saver Bid"] = val_bids

            cw = cw.sort_values(by=chop_sort_col, ascending=False)
            chop_waiver_cols = [
                "Player", "Pos", "Pos_Rank", "Team", "Chopped_Status", "Status", "Matchup",
                "Chop_Floor_Score", "xFP/G", "Recent_PPG", "Bust_Rate_%",
                "🎯 Rival Threats & Est. Bid",
                "💡 Recommended Optimal Bid", "🔒 Lock-In Bid (Capped)", "💰 Value Saver Bid"
            ]
            st.dataframe(cw[chop_waiver_cols].head(40), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # CHOP TAB 5: LEAGUE AUCTION ARCHIVE & GM BIDDING PROFILES (100% ENGLISH)
    # --------------------------------------------------------------------------
    with ctab5:
        st.subheader("🕵️‍♂️ League-Wide FAAB Auction Archive (Winning & Losing Bids)")
        st.caption(
            "Extracted from Sleeper's `complete` and `failed` waiver transactions across all weeks. Filter by any player, position, or GM to inspect runner-up bids, overpay margins, and competing offers."
        )

        s_c1, s_c2 = st.columns([2, 1])
        search_player = s_c1.text_input("🔍 Search Player or GM in Auction Archive:", value="")
        pos_auc_filter = s_c2.selectbox("Filter Archive by Position", ["ALL", "RB", "WR", "QB", "TE"])

        if not df_auctions.empty:
            view_auc = df_auctions.copy()
            if pos_auc_filter != "ALL":
                view_auc = view_auc[view_auc["Pos"] == pos_auc_filter]
            if search_player.strip():
                q_s = search_player.strip().lower()
                view_auc = view_auc[
                    view_auc["Player"].str.lower().str.contains(q_s)
                    | view_auc["Winner_GM"].str.lower().str.contains(q_s)
                    | view_auc["All_Competing_Bids"].str.lower().str.contains(q_s)
                ]
            auc_cols = [
                "Week", "Player", "Winner_GM", "Winning_Bid_Str",
                "Runner_Up_GM", "Runner_Up_Bid_Str", "Overpay_Str",
                "Total_Bidders", "All_Competing_Bids"
            ]
            st.dataframe(view_auc[auc_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No processed waiver auctions logged yet this season.")

        st.markdown("---")
        st.markdown("#### 🧠 Rival GM Historical Bidding Behavior Profiles")
        if not df_gm_profiles.empty:
            prof_cols = [
                "GM", "Status", "Remaining_FAAB", "Bidder_Archetype",
                "Total_Bids (Won-Lost)", "Max_Single_Bid", "Avg_Bid_Amount",
                "Avg_Winning_Bid", "Max_Bid_By_Pos"
            ]
            st.dataframe(df_gm_profiles[prof_cols], use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # CHOP TAB 6: RIVAL ROSTER & POSITIONAL LIVE RANKINGS SCOUT
    # --------------------------------------------------------------------------
    with ctab6:
        render_rival_roster_scout_tab("chop_scout")

    # --------------------------------------------------------------------------
    # CHOP TAB 7: NFL PRACTICE & INJURY PREVENTION TRACKER
    # --------------------------------------------------------------------------
    with ctab7:
        st.subheader("🏥 Pre-Sunday NFL Practice & Injury Prevention Tracker")
        if practice_prevention_rows:
            st.dataframe(pd.DataFrame(practice_prevention_rows), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Zero Injury or Practice Designations on Your Roster! All players practiced in full.")

# ==============================================================================
# MODE B: 🦅 DYNASTY LEAGUE COMMAND CENTER (12 TEAMS)
# ==============================================================================
else:
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
                opp_rid = int(opp_m.get("roster_id"))
                opp_gm_name = rid_to_gm.get(opp_rid, f"Roster {opp_rid}")
                opp_starters_df = build_df_from_pids(
                    [str(x) for x in (opp_m.get("starters") or []) if x and str(x) != "0"]
                )

    # Pre-compute Max PF Draft Order & Future Pick Ownership Inventory (2027-2029)
    sorted_by_max_pf = sorted(all_gm_rosters.items(), key=lambda x: (x[1]["max_pf"], x[1]["wins"]))
    rid_to_proj_slot = {}
    for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
        rid_to_proj_slot[rdata["roster_id"]] = slot_idx

    # Map (season_str, round_int, orig_rid) -> current_owner_rid
    future_years = [str(curr_season + 1), str(curr_season + 2), str(curr_season + 3)]
    pick_owner_lookup = {}
    for yr_s in future_years:
        for rnd in [1, 2, 3]:
            for r_gm, r_d in all_gm_rosters.items():
                orid = r_d["roster_id"]
                pick_owner_lookup[(yr_s, rnd, orid)] = orid

    for tp in traded_picks:
        yr_s = str(tp.get("season"))
        rnd = int(tp.get("round", 1))
        orid = int(tp.get("roster_id", 0))
        cur_id = int(tp.get("owner_id", 0))
        if (yr_s, rnd, orid) in pick_owner_lookup:
            pick_owner_lookup[(yr_s, rnd, orid)] = cur_id

    # Build pick valuation & inventory per GM
    gm_picks_inventory = defaultdict(list)
    for (yr_s, rnd, orid), owner_rid in pick_owner_lookup.items():
        owner_gm = rid_to_gm.get(owner_rid, f"Roster_{owner_rid}")
        orig_gm = rid_to_gm.get(orid, f"Roster_{orid}")
        slot_num = rid_to_proj_slot.get(orid, 6)

        # Estimate realistic DynastyProcess market value by projected slot & year
        if yr_s == str(curr_season + 1):
            slot_tag = f"Proj {rnd}.{slot_num:02d}"
            if rnd == 1:
                p_val = 7400 if slot_num == 1 else (5800 if slot_num <= 4 else (4300 if slot_num <= 8 else 3200))
            elif rnd == 2:
                p_val = 2100 if slot_num <= 4 else (1450 if slot_num <= 8 else 950)
            else:
                p_val = 550 if slot_num <= 4 else (350 if slot_num <= 8 else 220)
        elif yr_s == str(curr_season + 2):
            slot_tag = "Future"
            p_val = 3850 if rnd == 1 else (1300 if rnd == 2 else 320)
        else:
            slot_tag = "Future"
            p_val = 3350 if rnd == 1 else (1100 if rnd == 2 else 260)

        label_str = f"{yr_s} Rd {rnd} (via {orig_gm} — {slot_tag}) [Val: {p_val:,}]"
        gm_picks_inventory[owner_gm].append({
            "Label": label_str,
            "Short_Label": f"{yr_s} Rd {rnd} (via {orig_gm}, {slot_tag})",
            "Season": yr_s,
            "Round": rnd,
            "Orig_GM": orig_gm,
            "Proj_Slot": slot_tag,
            "Pos": "PICK",
            "Dynasty_Val": p_val,
            "xFP/G": 0.0,
            "Recent_PPG": 0.0,
            "Age": None,
        })

    for g_k in gm_picks_inventory:
        gm_picks_inventory[g_k].sort(key=lambda x: (x["Season"], x["Round"], -x["Dynasty_Val"]))

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Week Matchup", f"vs. {opp_gm_name}")
    k2.metric("Your FAAB Remaining", f"${my_faab_rem} / ${total_faab_budget}")
    k3.metric("Practice / Injury Flags", len(practice_prevention_rows))
    k4.metric("Start/Sit Volume Swaps", len(volume_swap_alerts))
    my_eff = all_gm_rosters.get(gm_input, {}).get("coach_eff_%", 100.0)
    k5.metric("Coaching Efficiency", f"{my_eff}%", help="Actual Points Scored divided by Optimal Max PF")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "💬 AI Dynasty GM",
        "⚡ Start/Sit & Game Env",
        "🏥 Practice & Injury Tracker",
        "📈 xFP Regression & Age Cliffs",
        "📡 Waiver Radar & Auction Archive",
        "🔮 Max PF Pick Forecaster",
        "👁️ Rival Roster & Positional Scout",
        "⚖️ Trade Machine & GM Pitch",
    ])

    # --------------------------------------------------------------------------
    # DYNASTY TAB 1: CONVERSATIONAL AI DYNASTY GM (100% ENGLISH)
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("💬 Conversational AI Dynasty Scientist")
        st.caption(
            f"Powered by `{model_choice}` (with Auto-Fallback) — Synced with Bye Weeks, Practice Reports (`DNP`/`LP`/`FP`), Strict IR rules, Taxi separation, Vegas/DvP, xFP/FPOE, and League Positional Rankings."
        )

        if "dynasty_chat_messages" not in st.session_state:
            st.session_state.dynasty_chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"Hey Carlos! I'm synced with **{gm_input}**'s Week {curr_week} Dynasty matchup against **{opp_gm_name}**.\n\n"
                        f"• **Teams on Bye this week:** `{', '.join(bye_teams_curr_wk) if bye_teams_curr_wk else 'None'}`\n"
                        f"• **Practice / Injury Alerts on Roster:** `{len(practice_prevention_rows)}`\n"
                        f"• **Your Remaining FAAB:** `${my_faab_rem}`\n\n"
                        "Ask me about **Start/Sit decisions**, **Practice Report injury pivots**, **xFP buy-low candidates**, or **Dynasty trade packages**!"
                    ),
                }
            ]

        b1, b2, b3, b4 = st.columns(4)
        quick_prompt = None
        if b1.button("⚡ Optimize Week Lineup", use_container_width=True):
            quick_prompt = (
                f"Audit my Week {curr_week} starting lineup vs my Active Bench (excluding Taxi and players on BYE). "
                "Use xFP/G, Volume Score, Snap %, Practice Reports, Vegas Implied Totals, and DvP Matchup ranks to tell me who to start or sit."
            )
        if b2.button("🏥 Check Practice & Injury Pivots", use_container_width=True):
            quick_prompt = (
                "Review my roster's Practice & Injury Prevention alerts (DNP/LP/FP). Who is at risk of missing Sunday, and who is my best bench or waiver pivot?"
            )
        if b3.button("📈 Find Buy-Low / Sell-Highs", use_container_width=True):
            quick_prompt = (
                "Analyze my roster using Expected Fantasy Points (xFP/G) vs Actual PPG (FPOE) and Age Cliffs. "
                "Who on my roster is due for positive regression, and who should I sell high?"
            )
        if b4.button("🤝 Propose 2 Contender Trades", use_container_width=True):
            quick_prompt = (
                "Review the League Positional Power Rankings and Contender vs Rebuilder segmentation of my leaguemates. Propose 2 mutually beneficial trades that upgrade my starting lineup."
            )

        for msg in st.session_state.dynasty_chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_query = st.chat_input("Ask about start/sit volume, practice reports, xFP regression, waivers, or trades...") or quick_prompt

        if user_query:
            st.session_state.dynasty_chat_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            if not api_key:
                with st.chat_message("assistant"):
                    st.warning("Please enter your Gemini API Key in the sidebar to enable AI analysis.")
            else:
                system_instruction = f"""
                You are an elite NFL Fantasy Football Data Scientist and Dynasty GM Advisor for Carlos (Sleeper username: '{gm_input}', team: 'Birdgang') in League {league_id_input}.
                Respond strictly in English.
                Current NFL Context: {curr_season} Season, Week {curr_week}. Opponent: {opp_gm_name}.
                League Format: {fmt_tag}, {ppr_val} PPR, +{tep_val} TE Premium.
                Strict IR Eligible Designations: {sorted(list(allowed_ir_tags))}. (Players marked 'Out' or 'NA' CANNOT go on IR unless listed here).
                NFL Teams on BYE in Week {curr_week}: {bye_teams_curr_wk}.

                CRITICAL RULES:
                1. NEVER recommend starting any player whose Status contains 'BYE' or whose team is in {bye_teams_curr_wk}.
                2. NEVER recommend starting a Taxi Squad player unless explicitly discussing promoting them from Taxi.
                3. NEVER tell the user to move an 'Out', 'Doubtful', or 'NA' player to IR unless that designation is in {sorted(list(allowed_ir_tags))}.
                4. Always factor in Practice_Participation (🔴 DNP, 🟡 LP, 🟢 FP), Pos_Rank, xFP/G, FPOE, Snap %, Target Share %, WOPR, Vegas Implied Totals, and DvP Matchup Rank.

                LIVE DATA CONTEXT:
                - Active Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
                - Active Bench (Eligible to Start): {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
                - Taxi Squad: {my_taxi_df.to_dict(orient='records') if not my_taxi_df.empty else []}
                - IR Slot Players: {my_ir_df.to_dict(orient='records') if not my_ir_df.empty else []}
                - My Future Draft Picks: {gm_picks_inventory.get(gm_input, [])}
                - Practice & Injury Prevention Alerts: {practice_prevention_rows}
                - Opponent ({opp_gm_name}) Starters: {opp_starters_df.to_dict(orient='records') if not opp_starters_df.empty else []}
                - League Positional Power Rankings: {df_league_pos_scout.to_dict(orient='records') if not df_league_pos_scout.empty else []}
                - Top 20 Unrostered Free Agents: {df_waivers.head(20).to_dict(orient='records') if not df_waivers.empty else []}
                """
                try:
                    client = genai.Client(api_key=api_key)
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing snaps, practice reports & matchup data..."):
                            ans, used_m = generate_ai_response_with_fallback(
                                client, model_choice, user_query, system_instruction
                            )
                            st.markdown(ans)
                            st.session_state.dynasty_chat_messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"Gemini API Error: {e}")

    # --------------------------------------------------------------------------
    # DYNASTY TAB 2: START / SIT VOLUME COMPARATOR & GAME ENVIRONMENT
    # --------------------------------------------------------------------------
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
            "Player", "Pos", "Pos_Rank", "Team", "Status", "Practice_Participation", "Matchup",
            "Vegas_Implied", "DvP_Matchup", "Volume_Score", "xFP/G", "Recent_PPG",
            "FPOE", "Snap_%", "Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "Dynasty_Val"
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

    # --------------------------------------------------------------------------
    # DYNASTY TAB 3: NFL PRACTICE & INJURY PREVENTION TRACKER
    # --------------------------------------------------------------------------
    with tab3:
        st.subheader("🏥 Pre-Sunday NFL Practice & Injury Prevention Tracker")
        if practice_prevention_rows:
            st.dataframe(pd.DataFrame(practice_prevention_rows), use_container_width=True, hide_index=True)
        else:
            st.success("✅ Zero Injury or Practice Designations on Your Roster! All players practiced in full.")

    # --------------------------------------------------------------------------
    # DYNASTY TAB 4: EXPECTED FANTASY POINTS (xFP), FPOE REGRESSION & AGE CLIFFS
    # --------------------------------------------------------------------------
    with tab4:
        st.subheader("📈 Expected Fantasy Points (xFP), FPOE Regression & Dynasty Value Cliffs")
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
                "Player", "Pos", "Pos_Rank", "Team", "Age", "Age_Cliff", "xFP/G", "Recent_PPG",
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

    # --------------------------------------------------------------------------
    # DYNASTY TAB 5: WAIVER RADAR, RB CONTINGENCY & AUCTION ARCHIVE
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("📡 Unrostered Waiver Radar, RB Contingency Index & FAAB Auction Archive")
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
                "Player", "Pos", "Pos_Rank", "Team", "Age", "Status", "Practice_Participation", "Matchup",
                "Volume_Score", "xFP/G", "Recent_PPG", "Contingency_Score", "24h_Adds",
                "Snap_%", "Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "Dynasty_Val"
            ]
            st.dataframe(vw[w_cols].head(35), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🕵️‍♂️ League-Wide Waiver Auction Archive (Winning & Losing Bids)")
        if not df_auctions.empty:
            auc_cols = [
                "Week", "Player", "Winner_GM", "Winning_Bid_Str",
                "Runner_Up_GM", "Runner_Up_Bid_Str", "Overpay_Str",
                "Total_Bidders", "All_Competing_Bids"
            ]
            st.dataframe(df_auctions[auc_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No processed FAAB waiver auctions logged yet this season.")

    # --------------------------------------------------------------------------
    # DYNASTY TAB 6: MAX PF ROOKIE DRAFT PICK FORECASTER (2027-2029)
    # --------------------------------------------------------------------------
    with tab6:
        st.subheader(f"🔮 Live {curr_season + 1} Rookie Draft Pick Forecaster (Sorted by Inverse Max PF)")
        next_draft_yr = str(curr_season + 1)
        pick_forecast_rows = []
        for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
            orig_rid = rdata["roster_id"]
            r1_owner_gm = rid_to_gm.get(pick_owner_lookup.get((next_draft_yr, 1, orig_rid), orig_rid), orig_gm)
            r2_owner_gm = rid_to_gm.get(pick_owner_lookup.get((next_draft_yr, 2, orig_rid), orig_rid), orig_gm)
            r3_owner_gm = rid_to_gm.get(pick_owner_lookup.get((next_draft_yr, 3, orig_rid), orig_rid), orig_gm)

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

    # --------------------------------------------------------------------------
    # DYNASTY TAB 7: RIVAL ROSTER & POSITIONAL LIVE RANKINGS SCOUT
    # --------------------------------------------------------------------------
    with tab7:
        render_rival_roster_scout_tab("dynasty_scout")

    # --------------------------------------------------------------------------
    # DYNASTY TAB 8: OVERHAULED TRADE MACHINE (WITH PICKS, POSITIONAL GROUPS & GM PITCH)
    # --------------------------------------------------------------------------
    with tab8:
        st.subheader("⚖️ Executive Dynasty Trade Machine & GM Negotiation Pitch Generator")
        st.caption(
            "Select players separated by position (`QB`, `RB`, `WR`, `TE`) plus **Future Rookie Draft Picks (2027–2029)** to evaluate mutual roster fit and generate a persuasive pitch for your trade partner."
        )

        all_max_pfs = [d["max_pf"] for d in all_gm_rosters.values()]
        median_max_pf = float(np.median(all_max_pfs)) if all_max_pfs else 500.0

        def get_gm_tier(gm_u):
            rdata = all_gm_rosters.get(gm_u, {})
            if rdata.get("wins", 0) > rdata.get("losses", 0) and rdata.get("max_pf", 0) >= median_max_pf:
                return "🏆 True Contender"
            elif rdata.get("wins", 0) < rdata.get("losses", 0) and rdata.get("max_pf", 0) < median_max_pf:
                return "🔨 Rebuilder"
            return "⚖️ Middle Tier / Pivot"

        rival_gms = [g for g in sorted(all_gm_rosters.keys()) if g.lower() != gm_input.strip().lower()]
        partner_gm = st.selectbox("🤝 Select Trade Partner GM", rival_gms)
        partner_tier = get_gm_tier(partner_gm)
        my_tier = get_gm_tier(gm_input)

        # Build positional lookup dicts for both GMs
        def build_pos_grouped_assets(gm_u):
            r_d = all_gm_rosters.get(gm_u, {})
            pids = r_d.get("players", [])
            df_p = build_df_from_pids(pids)
            by_pos = {"QB": {}, "RB": {}, "WR": {}, "TE": {}, "PICK": {}}
            if not df_p.empty:
                df_p = df_p.sort_values("Dynasty_Val", ascending=False)
                for _, row in df_p.iterrows():
                    pos = row["Pos"]
                    if pos in ["QB", "RB", "WR", "TE"]:
                        lbl = f"{row['Player']} ({row['Pos_Rank']} | Val: {row['Dynasty_Val']:,} | {row['xFP/G']} xFP/G)"
                        by_pos[pos][lbl] = row.to_dict()
            for pk in gm_picks_inventory.get(gm_u, []):
                by_pos["PICK"][pk["Label"]] = pk
            return by_pos

        my_assets_by_pos = build_pos_grouped_assets(gm_input)
        partner_assets_by_pos = build_pos_grouped_assets(partner_gm)

        col_send, col_rec = st.columns(2)

        sent_items = []
        with col_send:
            st.markdown(f"#### 📤 Assets `{gm_input}` ({my_tier}) Sends")
            for pos_k, pos_title in [("QB", "Quarterbacks (QB)"), ("RB", "Running Backs (RB)"), ("WR", "Wide Receivers (WR)"), ("TE", "Tight Ends (TE)"), ("PICK", "Future Draft Picks (2027-2029)")]:
                opts = list(my_assets_by_pos[pos_k].keys())
                sel = st.multiselect(f"Send {pos_title}", opts, key=f"send_{pos_k}")
                for k in sel:
                    sent_items.append(my_assets_by_pos[pos_k][k])

        rec_items = []
        with col_rec:
            st.markdown(f"#### 📥 Assets `{partner_gm}` ({partner_tier}) Sends")
            for pos_k, pos_title in [("QB", "Quarterbacks (QB)"), ("RB", "Running Backs (RB)"), ("WR", "Wide Receivers (WR)"), ("TE", "Tight Ends (TE)"), ("PICK", "Future Draft Picks (2027-2029)")]:
                opts = list(partner_assets_by_pos[pos_k].keys())
                sel = st.multiselect(f"Receive {pos_title}", opts, key=f"rec_{pos_k}")
                for k in sel:
                    rec_items.append(partner_assets_by_pos[pos_k][k])

        if sent_items or rec_items:
            st.markdown("---")
            send_val = sum(x["Dynasty_Val"] for x in sent_items)
            get_val = sum(x["Dynasty_Val"] for x in rec_items)
            val_delta = get_val - send_val

            send_xfp = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in sent_items), 1)
            get_xfp = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in rec_items), 1)
            xfp_delta = round(get_xfp - send_xfp, 1)

            send_ppg = round(sum(x.get("Recent_PPG", 0.0) or 0.0 for x in sent_items), 1)
            get_ppg = round(sum(x.get("Recent_PPG", 0.0) or 0.0 for x in rec_items), 1)
            ppg_delta = round(get_ppg - send_ppg, 1)

            send_ages = [x["Age"] for x in sent_items if x.get("Age")]
            get_ages = [x["Age"] for x in rec_items if x.get("Age")]
            avg_send_age = round(float(np.mean(send_ages)), 1) if send_ages else 0.0
            avg_get_age = round(float(np.mean(get_ages)), 1) if get_ages else 0.0
            age_delta = round(avg_get_age - avg_send_age, 1) if (send_ages and get_ages) else 0.0

            # Fairness Grade
            max_side = max(1, send_val, get_val)
            pct_diff = abs(val_delta) / max_side * 100.0
            if pct_diff <= 8.0:
                fair_badge = "⚖️ FAIR MARKET DEAL (Within 8%)"
            elif val_delta > 0:
                fair_badge = f"🟢 SURPLUS WIN FOR {gm_input.upper()} (+{pct_diff:.0f}%)"
            else:
                fair_badge = f"🟠 SURPLUS WIN FOR {partner_gm.upper()} (+{pct_diff:.0f}%)"

            st.markdown(f"### 📊 Executive Trade Outcome: {fair_badge}")
            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
            m_c1.metric("You Receive (Dynasty Val)", f"{get_val:,}", delta=f"{val_delta:+,} Net Value vs Sent ({send_val:,})")
            m_c2.metric("Weekly Volume Impact (xFP/G)", f"{get_xfp:.1f} xFP/G", delta=f"{xfp_delta:+.1f} xFP/G vs Sent ({send_xfp:.1f})")
            m_c3.metric("Weekly Scoring Impact (PPG)", f"{get_ppg:.1f} PPG", delta=f"{ppg_delta:+.1f} PPG vs Sent ({send_ppg:.1f})")
            m_c4.metric("Avg Player Age Acquired", f"{avg_get_age:.1f} yrs" if avg_get_age > 0 else "Picks Only", delta=f"{age_delta:+.1f} yrs vs Sent" if (send_ages and get_ages) else None, delta_color="inverse")

            # Positional Breakdown Table
            pos_breakdown_rows = []
            for p_cat in ["QB", "RB", "WR", "TE", "PICK"]:
                s_cat = [x for x in sent_items if x["Pos"] == p_cat]
                r_cat = [x for x in rec_items if x["Pos"] == p_cat]
                if not s_cat and not r_cat:
                    continue
                s_names = " + ".join([x.get("Player") or x.get("Short_Label") for x in s_cat]) or "None"
                r_names = " + ".join([x.get("Player") or x.get("Short_Label") for x in r_cat]) or "None"
                s_v = sum(x["Dynasty_Val"] for x in s_cat)
                r_v = sum(x["Dynasty_Val"] for x in r_cat)
                s_x = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in s_cat), 1)
                r_x = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in r_cat), 1)
                pos_breakdown_rows.append({
                    "Position_Group": "Draft Capital" if p_cat == "PICK" else p_cat,
                    f"{gm_input}_Sends": s_names,
                    "Sent_Val": f"{s_v:,}",
                    "Sent_xFP/G": f"{s_x:.1f}",
                    f"{gm_input}_Receives": r_names,
                    "Received_Val": f"{r_v:,}",
                    "Received_xFP/G": f"{r_x:.1f}",
                    "Net_Positional_Delta": f"{r_v - s_v:+,} val ({r_x - s_x:+.1f} xFP/G)",
                })

            st.markdown("#### 🧩 Positional Package Breakdown")
            st.dataframe(pd.DataFrame(pos_breakdown_rows), use_container_width=True, hide_index=True)

            # Automated Negotiation Talking Points & Copy-Paste Pitch for the Other GM
            st.markdown("#### 💬 Ready-to-Send Trade Pitch for Your Trade Partner")
            sent_names_clean = ", ".join([x.get("Player") or x.get("Short_Label") for x in sent_items]) or "None"
            rec_names_clean = ", ".join([x.get("Player") or x.get("Short_Label") for x in rec_items]) or "None"

            # Identify partner's weakest positional rank from df_league_pos_scout
            partner_weak_pos = []
            if not df_league_pos_scout.empty:
                p_sc_row = df_league_pos_scout[df_league_pos_scout["GM"] == partner_gm]
                if not p_sc_row.empty:
                    for pos_c in ["QB", "RB", "WR", "TE"]:
                        rk_c = int(p_sc_row.iloc[0][f"{pos_c}_League_Rank"])
                        if rk_c >= 7:
                            partner_weak_pos.append(f"{pos_c} (Ranked #{rk_c} in league)")

            pitch_points = []
            if send_val >= get_val:
                pitch_points.append(
                    f"• **Positive Dynasty Value for {partner_gm}:** You gain **+{send_val - get_val:,} in consensus Dynasty market value** ({send_val:,} vs {get_val:,})."
                )
            else:
                pitch_points.append(
                    f"• **Balanced Core Consolidation:** Total Dynasty market value is closely aligned ({send_val:,} to {get_val:,}) while addressing starting lineup needs."
                )

            if send_xfp > get_xfp:
                pitch_points.append(
                    f"• **Immediate Weekly Scoring Boost for {partner_gm}:** The players you receive add **+{send_xfp - get_xfp:.1f} Expected Fantasy Points per game** ({send_xfp:.1f} xFP/G vs {get_xfp:.1f} xFP/G)."
                )
            if any(x["Pos"] == "PICK" for x in sent_items):
                picks_sent_str = ", ".join([x["Short_Label"] for x in sent_items if x["Pos"] == "PICK"])
                pitch_points.append(
                    f"• **Insulated Future Draft Capital:** You secure liquid draft capital (**{picks_sent_str}**) that appreciates in value heading into the rookie draft and never suffers injuries."
                )
            if partner_weak_pos:
                pitch_points.append(
                    f"• **Addresses Your Roster Needs:** Directly reinforces your positional depth ({', '.join(partner_weak_pos)})."
                )

            copy_paste_msg = (
                f"Hey {partner_gm}! Put together a trade idea that fits both of our rosters:\n\n"
                f"• You Receive: {sent_names_clean} (Dynasty Value: {send_val:,} | {send_xfp:.1f} xFP/G)\n"
                f"• I Receive: {rec_names_clean} (Dynasty Value: {get_val:,} | {get_xfp:.1f} xFP/G)\n\n"
                f"Why it makes sense for your squad ({partner_tier}):\n" + "\n".join([p.replace("**", "") for p in pitch_points])
            )

            for pt in pitch_points:
                st.markdown(pt)

            st.code(copy_paste_msg, language="text")

            if st.button("✨ Generate Custom AI Persuasion Pitch for This Deal", use_container_width=True):
                if not api_key:
                    st.warning("Please enter your Gemini API Key in the sidebar to generate a custom AI pitch.")
                else:
                    pitch_prompt = f"""
                    Write a persuasive, friendly, data-backed 2-paragraph trade message from '{gm_input}' ({my_tier}) to '{partner_gm}' ({partner_tier}) pitching this exact Dynasty trade:
                    - {partner_gm} Receives: {sent_items} (Total Val: {send_val}, Total xFP/G: {send_xfp})
                    - {gm_input} Receives: {rec_items} (Total Val: {get_val}, Total xFP/G: {get_xfp})
                    - {partner_gm}'s Positional Weaknesses in League: {partner_weak_pos}
                    Highlight exact snap %, xFP/G, positional ranks, or projected draft pick slots to show why this is a win-win deal for {partner_gm}. Keep it natural and ready to send in Sleeper chat.
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        with st.spinner("Crafting custom GM trade pitch..."):
                            ai_pitch, _ = generate_ai_response_with_fallback(
                                client,
                                model_choice,
                                pitch_prompt,
                                "You are an expert Dynasty Fantasy Football negotiator. Write in clear, persuasive English.",
                            )
                            st.info(ai_pitch)
                    except Exception as e:
                        st.error(f"Gemini API Error: {e}")
