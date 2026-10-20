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

    # Crawl ALL weeks (1 to curr_week + 1) to extract:
    # A) Recently dropped/chopped players
    # B) Complete Waiver Bid History (BOTH 'complete' winning bids AND 'failed' losing bids!)
    recently_dropped_pids = {}
    raw_waiver_bids = []  # list of dicts for every waiver bid placed

    for w_tx in range(1, curr_week + 2):
        tx_list = get_json(f"https://api.sleeper.app/v1/league/{league_id}/transactions/{w_tx}") or []
        for tx in tx_list:
            tx_status = tx.get("status")
            tx_type = tx.get("type")

            # Track drops from complete transactions in recent weeks
            if tx_status == "complete" and w_tx >= max(1, curr_week - 1):
                for d_pid, d_rid in (tx.get("drops") or {}).items():
                    recently_dropped_pids[str(d_pid)] = {"week": w_tx, "rid": d_rid}

            # Track ALL waiver bids (complete & failed)
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
                        "Status": tx_status,  # 'complete' or 'failed'
                        "Notes": notes_str,
                        "Timestamp": ts_val,
                    })

    # 4. Live DynastyProcess Market Values
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
        schedule_map,
        bye_teams_curr_wk,
        nflverse_injuries,
        dvp_ranks,
        dict(weekly_usage),
        nflverse_metrics,
        ngs_sep_map,
    )


# ==============================================================================
# WAIVER BID AUDIT & GM BIDDING BEHAVIOR ENGINE
# ==============================================================================
def analyze_waiver_auctions_and_gm_profiles(raw_waiver_bids, all_gm_rosters, total_faab_budget):
    """
    1. Deduplicates conditional claims by the same GM for the same player in the same week.
    2. Builds a complete auction breakdown per (Week, Player): Winner, Winning Bid, Runner-Up, Overpay, All Bids.
    3. Builds a historical bidding profile for every GM in the league.
    """
    # Deduplicate: if GM placed multiple conditional claims on the same player in the same week,
    # keep the 'complete' one if won, else keep highest bid.
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

    # Group by (Week, PID) to reconstruct each player's auction
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
        # Sort bids: winner ('complete') first, then by descending bid amount
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
            runner_up_gm = "Sin Competencia (Uncontested)"
            runner_up_bid = 0
            overpay = w_bid
            all_competing_str = "Ninguna otra oferta ($0)"

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

    # Build GM Bidding Behavior Profiles for all teams in the league
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

        # Calculate typical aggressiveness ratio (Max Bid relative to starting budget)
        agg_ratio = max_b / max(1.0, float(total_faab_budget))
        if agg_ratio >= 0.35 or avg_b >= (total_faab_budget * 0.18):
            archetype = "🦈 Tiburón Agresivo (Heavy Spender)"
            est_bid_pct = 0.42
        elif tot_b >= 4 and avg_b >= (total_faab_budget * 0.08):
            archetype = "🎯 Francotirador Activo (Calculated Bidder)"
            est_bid_pct = 0.28
        elif tot_b > 0:
            archetype = "🐢 Cazador de Gangas (Low-Ball Saver)"
            est_bid_pct = 0.14
        else:
            archetype = "💤 Pasivo / Ahorrador Total ($0 Bids)"
            est_bid_pct = 0.10

        pos_pref_list = []
        for p_k in ["RB", "WR", "QB", "TE"]:
            if st_g["pos_bids"][p_k]:
                pos_pref_list.append(f"{p_k}: max ${max(st_g['pos_bids'][p_k])}")
        pos_pref_str = " | ".join(pos_pref_list) if pos_pref_list else "Sin pujas registradas"

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
        "Dynasty_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
        "Dynasty_Val": val,
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
with st.spinner(f"Syncing {selected_preset} settings, waiver bid history, practice reports & usage..."):
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

# Run Historical Waiver Bid Audit & GM Profiling
df_auctions, df_gm_profiles, gm_aggression_map = analyze_waiver_auctions_and_gm_profiles(
    raw_waiver_bids, all_gm_rosters, total_faab_budget
)


def build_df_from_pids(pid_list):
    if not pid_list:
        return pd.DataFrame()
    return pd.DataFrame([
        build_player_profile(
            p, curr_week, players_db, val_map, rank_map, schedule_map,
            bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
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
            pid, curr_week, players_db, val_map, rank_map, schedule_map,
            bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
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
# RIVAL POSITIONAL NEED & PREDICTIVE FAAB BIDDING ENGINE
# ==============================================================================
# Scan every surviving rival's starting lineup to detect positional weaknesses (BYE, Out/DNP, or weak starter)
rival_pos_needs = defaultdict(list)  # pos -> list of dicts {'gm': gm, 'faab': faab, 'reason': reason, 'est_bid': est_bid}
surviving_gms_set = {g for g, d in all_gm_rosters.items() if d["is_alive"]}

# Identify bottom 4 scoring surviving teams (Desperation Multiplier!)
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

    # Check rival's starters by position
    pos_weakness_reasons = defaultdict(list)
    pos_starter_counts = defaultdict(int)

    for s_pid in r_data["starters"]:
        p_m = players_db.get(str(s_pid), {})
        pos = p_m.get("position", "N/A")
        if pos not in ["QB", "RB", "WR", "TE"]:
            continue
        pos_starter_counts[pos] += 1
        tm = norm_nfl_team(p_m.get("team"))
        inj_st = p_m.get("injury_status") or ""
        prac_st = str(p_m.get("practice_participation") or "").lower()
        p_nm = f"{p_m.get('first_name', '')} {p_m.get('last_name', '')}".strip()

        u_st = weekly_usage.get(str(s_pid), {})
        s_xfp = round(float(np.mean(u_st.get("xfp", [0.0]))), 1) if u_st.get("xfp") else 0.0

        if tm in bye_teams_curr_wk:
            pos_weakness_reasons[pos].append(f"{p_nm} en BYE")
        elif inj_st in ["Out", "IR", "Doubtful", "PUP"] or "dnp" in prac_st:
            pos_weakness_reasons[pos].append(f"{p_nm} ({inj_st or 'DNP'})")
        elif s_xfp < 7.5 and pos in ["RB", "WR"]:
            pos_weakness_reasons[pos].append(f"{p_nm} bajo vol ({s_xfp} xFP)")
        elif s_xfp < 6.0 and pos == "TE":
            pos_weakness_reasons[pos].append(f"{p_nm} bajo vol ({s_xfp} xFP)")

    for pos in ["QB", "RB", "WR", "TE"]:
        reasons = pos_weakness_reasons[pos]
        if is_desperate and not reasons:
            reasons.append("En Zona Peligro Guillotina")
        if reasons:
            # Estimate what this rival will bid on an elite starter at this position
            base_pct = agg_info["est_bid_pct"] * desp_mult
            # Adjust for historical Guillotine phase (Weeks 1-5: 0.85x, Weeks 6-10: 1.1x, Weeks 11+: 1.35x)
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


def predict_smart_faab_for_player(wr_row, my_faab, max_rival_faab, avg_surv_faab):
    """Calculates predicted rival threats, estimated rival bid range, and optimal smart bid."""
    pos = wr_row["Pos"]
    floor_sc = wr_row["Chop_Floor_Score"]
    xfp = wr_row["xFP/G"]

    # Determine player tier multiplier (1.0 = Elite Bellcow/Alpha, 0.55 = Solid Starter, 0.20 = Depth/Streamer)
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
        threats_summary = "Sin urgencia directa detectada"

    # Add psychological tie-breaker (+$2 or +$6 to beat round numbers like $100, $150, $200, $250)
    raw_opt = max_est_threat_bid + 3
    if raw_opt % 5 == 0:
        raw_opt += 2
    elif raw_opt % 10 == 0:
        raw_opt += 6

    smart_bid = min(my_faab, max(1, raw_opt))
    lock_bid = min(my_faab, max_needy_faab + 1) if my_faab > max_needy_faab else my_faab
    val_bid = min(my_faab, max(0, int(round(max_est_threat_bid * 0.65))))

    return threats_summary, f"${smart_bid}", f"${lock_bid}", f"${val_bid}"


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
    c_k4.metric("Your FAAB Remaining", f"${my_faab_rem} / ${total_faab_budget}", delta=f"Avg Alive: ${avg_surviving_faab}")
    c_k5.metric("Waiver Auctions Logged", len(df_auctions))

    ctab1, ctab2, ctab3, ctab4, ctab5, ctab6 = st.tabs([
        "💬 AI Survival GM (Chop Chat)",
        "🪓 Guillotine Cut-Line & 18-Team Board",
        "🛡️ Floor & Bust-Rate Start/Sit",
        "🥩 Chopped Waiver Goldmine & Rival Bid Predictor",
        "🕵️‍♂️ Historial de Pujas FAAB (McCaffrey & Rivales)",
        "🏥 Practice & Injury Prevention Tracker",
    ])

    # --------------------------------------------------------------------------
    # CHOP TAB 1: AI SURVIVAL GM CHAT
    # --------------------------------------------------------------------------
    with ctab1:
        st.subheader("💬 AI Guillotine Survival Strategist")
        st.caption(
            "Conoce todas las pujas ganadoras y fallidas de tu liga (incluyendo quién pujó contra ti por McCaffrey), el perfil de gasto de cada rival, las necesidades posicionales de los equipos vivos y los reportes de práctica."
        )

        if "chop_chat_messages" not in st.session_state:
            st.session_state.chop_chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"¡Hola Carlos! Estoy sincronizado con **The Chop Block** (Semana {curr_week}).\n\n"
                        f"• **Equipos Vivos:** `{len(surviving_teams)} / {len(all_gm_rosters)}`\n"
                        f"• **Tu FAAB Restante:** `${my_faab_rem}` (Máximo Rival: `${max_rival_faab}` | Promedio Vivos: `${avg_surviving_faab}`)\n"
                        f"• **Subastas FAAB Registradas (Ganadoras + Fallidas):** `{len(df_auctions)}`\n\n"
                        "Pregúntame **quién más pujó por Christian McCaffrey**, **cuánto pujarán tus rivales esta semana según sus necesidades**, o **a quién iniciar para asegurar piso**."
                    ),
                }
            ]

        cb1, cb2, cb3, cb4 = st.columns(4)
        chop_quick = None
        if cb1.button("🕵️‍♂️ Analizar Pujas por McCaffrey y Waivers", use_container_width=True):
            chop_quick = (
                "Analiza el historial de subastas de waivers de mi liga (incluyendo las pujas fallidas). "
                "¿Quién más pujó por Christian McCaffrey y cuánto ofrecieron? ¿Qué patrones de gasto ves en mis rivales?"
            )
        if cb2.button("🥩 Predecir Pujas Rivales Esta Semana", use_container_width=True):
            chop_quick = (
                f"Tengo ${my_faab_rem} de FAAB. Revisa los mejores jugadores disponibles en waivers, dime qué rivales vivos tienen necesidad en esas posiciones y cuánto dinero debo pujar exactamente para ganarlos sin sobrepagar."
            )
        if cb3.button("🛡️ Auditar Mi Alineación (Piso y Bust Rate)", use_container_width=True):
            chop_quick = (
                f"Audita mi alineación titular de la Semana {curr_week} contra mi banca para una liga Chop. "
                "Detecta cualquier jugador con alto Bust Rate (%), en BYE o con reporte DNP/LP."
            )
        if cb4.button("🏥 Revisar Reportes de Práctica Pre-Domingo", use_container_width=True):
            chop_quick = (
                "Revisa la tabla de reportes de práctica (DNP/LP/FP) de mi equipo. ¿Qué jugadores corren riesgo para el domingo y qué plan B debo activar?"
            )

        for msg in st.session_state.chop_chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        chop_query = st.chat_input("Pregunta sobre pujas pasadas (McCaffrey), predicción de FAAB rival, o start/sit...") or chop_quick

        if chop_query:
            st.session_state.chop_chat_messages.append({"role": "user", "content": chop_query})
            with st.chat_message("user"):
                st.markdown(chop_query)

            if not api_key:
                with st.chat_message("assistant"):
                    st.warning("Por favor ingresa tu Gemini API Key en la barra lateral izquierda.")
            else:
                chop_sys_instruction = f"""
                You are an elite Guillotine (Chop) Fantasy Football Strategist advising Carlos ('{gm_input}') in an 18-team Chop League (League ID: {league_id_input}).
                RULES OF THIS LEAGUE:
                - 18 Teams, 1QB, {ppr_val} PPR. Trades are strictly DISABLED.
                - 1 Team is eliminated every week (lowest weekly points gets chopped and their entire roster drops to waivers).
                - Starting FAAB Budget: $1,000. Carlos currently has ${my_faab_rem} remaining. Highest rival has ${max_rival_faab}. Average surviving team has ${avg_surviving_faab}.
                - Teams on BYE in Week {curr_week}: {bye_teams_curr_wk}.

                HISTORICAL GUILLOTINE FAAB BENCHMARKS ($1,000 BUDGET):
                - Weeks 1-5 (Early Survival): Elite RBs/WRs typically go for $250-$450 (25%-45%). Overspending >$500 early cripples late-season flexibility.
                - Weeks 6-10 (Mid-Season Crunch): Elite stars go for 35%-55% of remaining FAAB stacks.
                - Weeks 11+ (Endgame): FAAB dominance dictates locked-in winning bids.

                LIVE DATA CONTEXT:
                - My Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
                - My Bench: {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
                - Practice & Injury Alerts: {practice_prevention_rows}
                - Complete Historical Waiver Bid Audit (Winning + Failed Bids, including McCaffrey): {df_auctions.head(35).to_dict(orient='records') if not df_auctions.empty else []}
                - Rival GM Bidding Behavior Profiles: {df_gm_profiles.to_dict(orient='records') if not df_gm_profiles.empty else []}
                - Surviving Rival Positional Needs & Estimated Bids: {dict(rival_pos_needs)}
                - Top 25 Available Waiver / Chopped Players: {df_waivers.head(25).to_dict(orient='records') if not df_waivers.empty else []}
                """
                try:
                    client = genai.Client(api_key=api_key)
                    with st.chat_message("assistant"):
                        with st.spinner("Analizando historial de pujas, necesidades rivales y piso..."):
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
            "Player", "Pos", "Team", "Status", "Practice_Participation", "Matchup",
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
        st.caption(
            f"**Tu FAAB:** `${my_faab_rem}` | **Máximo Rival:** `${max_rival_faab}` | **Promedio Vivos:** `${avg_surviving_faab}` — "
            "Calcula qué rivales vivos tienen necesidad en esa posición (por BYE, lesión o peligro de guillotina) y cuánto pujarán según su historial."
        )

        w_f1, w_f2, w_f3 = st.columns(3)
        chop_pos_filter = w_f1.selectbox("Filter Position", ["ALL", "RB", "WR", "TE", "QB"], key="chop_pos")
        only_chopped_toggle = w_f2.checkbox("🥩 Mostrar Solo Jugadores Cortados Recientemente", value=False)
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
                    wr, my_faab_rem, max_rival_faab, avg_surviving_faab
                )
                threats_col.append(th_str)
                smart_bids.append(sm_b)
                lock_bids.append(lk_b)
                val_bids.append(vl_b)

            cw["🎯 Rivales Amenaza y Puja Estimada"] = threats_col
            cw["💡 Puja Óptima Recomendada"] = smart_bids
            cw["🔒 Lock-In Bid (Garantizada)"] = lock_bids
            cw["💰 Puja Ahorro (Value)"] = val_bids

            cw = cw.sort_values(by=chop_sort_col, ascending=False)
            chop_waiver_cols = [
                "Player", "Pos", "Team", "Chopped_Status", "Status", "Matchup",
                "Chop_Floor_Score", "xFP/G", "Recent_PPG", "Bust_Rate_%",
                "🎯 Rivales Amenaza y Puja Estimada",
                "💡 Puja Óptima Recomendada", "🔒 Lock-In Bid (Garantizada)", "💰 Puja Ahorro (Value)"
            ]
            st.dataframe(cw[chop_waiver_cols].head(40), use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # CHOP TAB 5: HISTORIAL DE SUBASTAS FAAB (PUJAS GANADORAS Y FALLIDAS)
    # --------------------------------------------------------------------------
    with ctab5:
        st.subheader("🕵️‍♂️ Auditoría de Subastas FAAB (Pujas Ganadoras vs. Pujas Perdedoras de Rivales)")
        st.caption(
            "Extraído directamente de las transacciones `complete` y `failed` de Sleeper. Muestra quién ganó cada jugador (ej. **Christian McCaffrey**), quién quedó en 2.º lugar, cuánto dinero se pagó de más (`Overpay_Str`) y todas las ofertas rivales."
        )

        search_player = st.text_input("🔍 Buscar Jugador o Manager (ej. McCaffrey, arturohd):", value="")
        if not df_auctions.empty:
            view_auc = df_auctions.copy()
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
            st.info("Aún no se registran subastas de waivers procesadas en esta temporada.")

        st.markdown("---")
        st.markdown("#### 🧠 Perfil Histórico de Agresividad en Waivers por Manager")
        if not df_gm_profiles.empty:
            prof_cols = [
                "GM", "Status", "Remaining_FAAB", "Bidder_Archetype",
                "Total_Bids (Won-Lost)", "Max_Single_Bid", "Avg_Bid_Amount",
                "Avg_Winning_Bid", "Max_Bid_By_Pos"
            ]
            st.dataframe(df_gm_profiles[prof_cols], use_container_width=True, hide_index=True)

    # --------------------------------------------------------------------------
    # CHOP TAB 6: NFL PRACTICE & INJURY PREVENTION TRACKER
    # --------------------------------------------------------------------------
    with ctab6:
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

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Week Matchup", f"vs. {opp_gm_name}")
    k2.metric("Your FAAB Remaining", f"${my_faab_rem} / ${total_faab_budget}")
    k3.metric("Practice / Injury Flags", len(practice_prevention_rows))
    k4.metric("Start/Sit Volume Swaps", len(volume_swap_alerts))
    my_eff = all_gm_rosters.get(gm_input, {}).get("coach_eff_%", 100.0)
    k5.metric("Coaching Efficiency", f"{my_eff}%", help="Actual Points Scored divided by Optimal Max PF")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💬 AI Dynasty GM",
        "⚡ Start/Sit & Game Env",
        "🏥 Practice & Injury Tracker",
        "📈 xFP Regression & Age Cliffs",
        "📡 Waiver Radar & Bid History",
        "🔮 Max PF Pick Forecaster",
        "🔄 Rival Segments & Trade Machine",
    ])

    # --------------------------------------------------------------------------
    # DYNASTY TAB 1: CONVERSATIONAL AI DYNASTY GM
    # --------------------------------------------------------------------------
    with tab1:
        st.subheader("💬 Conversational AI Dynasty Scientist")
        st.caption(
            f"Powered by `{model_choice}` (con Auto-Fallback anti-503) — Sincronizado con Bye Weeks, Reportes de Práctica (`DNP`/`LP`/`FP`), IR estricto, Taxi, Vegas/DvP, xFP/FPOE e Historial de Pujas."
        )

        if "dynasty_chat_messages" not in st.session_state:
            st.session_state.dynasty_chat_messages = [
                {
                    "role": "assistant",
                    "content": (
                        f"¡Hola Carlos! Estoy sincronizado con el duelo de la Semana {curr_week} de **{gm_input}** contra **{opp_gm_name}**.\n\n"
                        f"• **Equipos en Bye esta semana:** `{', '.join(bye_teams_curr_wk) if bye_teams_curr_wk else 'Ninguno'}`\n"
                        f"• **Alertas de Práctica / Lesión:** `{len(practice_prevention_rows)}`\n"
                        f"• **Tu FAAB Restante:** `${my_faab_rem}`\n\n"
                        "Pregúntame sobre **decisiones Start/Sit**, **reportes de práctica**, **candidatos Buy-Low por xFP**, o **paquetes de trade**."
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
                "Review the Contender vs Rebuilder segmentation of my leaguemates and propose 2 win-now trades that upgrade my starting lineup."
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
                Current NFL Context: {curr_season} Season, Week {curr_week}. Opponent: {opp_gm_name}.
                League Format: {fmt_tag}, {ppr_val} PPR, +{tep_val} TE Premium.
                Strict IR Eligible Designations: {sorted(list(allowed_ir_tags))}. (Players marked 'Out' or 'NA' CANNOT go on IR unless listed here).
                NFL Teams on BYE in Week {curr_week}: {bye_teams_curr_wk}.

                CRITICAL RULES:
                1. NEVER recommend starting any player whose Status contains 'BYE' or whose team is in {bye_teams_curr_wk}.
                2. NEVER recommend starting a Taxi Squad player unless explicitly discussing promoting them from Taxi.
                3. NEVER tell the user to move an 'Out', 'Doubtful', or 'NA' player to IR unless that designation is in {sorted(list(allowed_ir_tags))}.
                4. Always factor in Practice_Participation (🔴 DNP, 🟡 LP, 🟢 FP), xFP/G, FPOE, Snap %, Target Share %, WOPR, Vegas Implied Totals, and DvP Matchup Rank.

                LIVE DATA CONTEXT:
                - Active Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
                - Active Bench (Eligible to Start): {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
                - Taxi Squad: {my_taxi_df.to_dict(orient='records') if not my_taxi_df.empty else []}
                - IR Slot Players: {my_ir_df.to_dict(orient='records') if not my_ir_df.empty else []}
                - Practice & Injury Prevention Alerts: {practice_prevention_rows}
                - Opponent ({opp_gm_name}) Starters: {opp_starters_df.to_dict(orient='records') if not opp_starters_df.empty else []}
                - Top 20 Unrostered Free Agents: {df_waivers.head(20).to_dict(orient='records') if not df_waivers.empty else []}
                - Historical League Waiver Bids: {df_auctions.head(20).to_dict(orient='records') if not df_auctions.empty else []}
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
            "Player", "Pos", "Team", "Status", "Practice_Participation", "Matchup",
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

    # --------------------------------------------------------------------------
    # DYNASTY TAB 5: WAIVER RADAR, RB CONTINGENCY & HISTORICAL BID AUDIT
    # --------------------------------------------------------------------------
    with tab5:
        st.subheader("📡 Unrostered Waiver Radar, RB Contingency Index & FAAB Bid Audit")
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
                "Player", "Pos", "Team", "Age", "Status", "Practice_Participation", "Matchup",
                "Volume_Score", "xFP/G", "Recent_PPG", "Contingency_Score", "24h_Adds",
                "Snap_%", "Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "Dynasty_Val"
            ]
            st.dataframe(vw[w_cols].head(35), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 🕵️‍♂️ Historial de Subastas de Waivers (Pujas Ganadoras y Fallidas)")
        if not df_auctions.empty:
            auc_cols = [
                "Week", "Player", "Winner_GM", "Winning_Bid_Str",
                "Runner_Up_GM", "Runner_Up_Bid_Str", "Overpay_Str",
                "Total_Bidders", "All_Competing_Bids"
            ]
            st.dataframe(df_auctions[auc_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No se registran pujas con FAAB en lo que va de la temporada.")

    # --------------------------------------------------------------------------
    # DYNASTY TAB 6: MAX PF ROOKIE DRAFT PICK FORECASTER (2027-2029)
    # --------------------------------------------------------------------------
    with tab6:
        st.subheader(f"🔮 Live {curr_season + 1} Rookie Draft Pick Forecaster (Sorted by Inverse Max PF)")
        next_draft_yr = str(curr_season + 1)
        pick_owner_map = {}
        for tp in traded_picks:
            if str(tp.get("season")) == next_draft_yr:
                pick_owner_map[(int(tp.get("round", 1)), int(tp.get("roster_id")))] = int(tp.get("owner_id"))

        sorted_by_max_pf = sorted(all_gm_rosters.items(), key=lambda x: (x[1]["max_pf"], x[1]["wins"]))
        pick_forecast_rows = []
        for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
            orig_rid = rdata["roster_id"]
            r1_owner_gm = rid_to_gm.get(pick_owner_map.get((1, orig_rid), orig_rid), orig_gm)
            r2_owner_gm = rid_to_gm.get(pick_owner_map.get((2, orig_rid), orig_rid), orig_gm)
            r3_owner_gm = rid_to_gm.get(pick_owner_map.get((3, orig_rid), orig_rid), orig_gm)

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
    # DYNASTY TAB 7: RIVAL CONTENDER/REBUILDER SEGMENTATION & TRADE MACHINE
    # --------------------------------------------------------------------------
    with tab7:
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
                    bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
                    nflverse_metrics, ngs_sep_map, trending_map, recently_dropped_pids
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
                        bye_teams_curr_wk, nflverse_injuries, dvp_ranks, weekly_usage,
                        nflverse_metrics, ngs_sep_map, trending_map, recently_dropped_pids
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
