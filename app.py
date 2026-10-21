import datetime
import re
import time
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

# Prioritized High-to-Low AI Model Cascade
def generate_ai_response_with_fallback(
    client, preferred_model, prompt_text, sys_instruction
):
  """Dynamically discovers all models available to your API key, prioritizes

  highest versions first, and retries with exponential backoff on 503 high-demand
  spikes.
  """
  # 1. Build candidate list starting with preferred + cascade
  candidate_models = []
  for m in [preferred_model] + AI_MODEL_CASCADE:
    if m and m not in candidate_models:
      candidate_models.append(m)

  # 2. Dynamically append any other active generateContent models from your API key
  try:
    for m_obj in client.models.list():
      m_id = getattr(m_obj, "name", "").replace("models/", "")
      if "gemini" in m_id and "embedding" not in m_id and "imagen" not in m_id:
        if m_id not in candidate_models:
          candidate_models.append(m_id)
  except Exception:
    pass

  last_err = None
  # 3. Two full passes with backoff so temporary 503 spikes resolve automatically
  for attempt in range(2):
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
        if response and response.text:
          return response.text, m_name
      except Exception as e:
        last_err = e
        err_str = str(e).upper()
        if "API_KEY_INVALID" in err_str or "INVALID API KEY" in err_str:
          raise e
        # If 503 high demand, pause briefly and retry the same working model once before moving on
        if "503" in err_str or "UNAVAILABLE" in err_str:
          time.sleep(1.5)
          try:
            retry_resp = client.models.generate_content(
                model=m_name,
                contents=prompt_text,
                config=types.GenerateContentConfig(
                    system_instruction=sys_instruction,
                    temperature=0.25,
                ),
            )
            if retry_resp and retry_resp.text:
              return retry_resp.text, m_name
          except Exception:
            pass
        continue
    time.sleep(1.0)

  raise last_err


# ==============================================================================
# HELPER UTILITIES & SILENT AI CASCADE
# ==============================================================================
def get_json(url, bust_cache=False):
    """Fetches JSON from URL; appends a timestamp query param when bust_cache=True to bypass CDN cache."""
    try:
        target_url = url
        if bust_cache:
            sep = "&" if "?" in url else "?"
            target_url = f"{url}{sep}_cb={int(time.time() * 1000)}"
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        r = requests.get(target_url, headers=headers, timeout=12)
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


def generate_ai_response_with_fallback(client, preferred_model, prompt_text, sys_instruction):
    """
    Prioritizes highest-version AI models first (gemini-3.1-pro-preview -> gemini-3.8-flash -> etc.)
    and silently falls back to the next available model if free-tier quota (429), demand (503), or 404 is hit.
    """
    candidate_models = []
    for m in [preferred_model] + AI_MODEL_CASCADE:
        if m and m not in candidate_models:
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
            if response and response.text:
                return response.text, m_name
        except Exception as e:
            last_err = e
            err_str = str(e).upper()
            # If key itself is invalid, stop immediately; otherwise silently cascade to next model
            if "API_KEY_INVALID" in err_str or "INVALID API KEY" in err_str:
                break
            continue
    raise last_err


# ==============================================================================
# LAYER 1: HEAVY REFERENCE DATA PIPELINE (CACHED 30 MINS FOR SPEED)
# ==============================================================================
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_reference_data(curr_season, target_week, lookback_weeks, is_superflex, scoring_items_tuple):
    scoring_settings = dict(scoring_items_tuple)

    players_db = get_json("https://api.sleeper.app/v1/players/nfl") or {}
    trending_adds_raw = (
        get_json("https://api.sleeper.app/v1/players/nfl/trending/add?lookback_hours=24&limit=100")
        or []
    )
    trending_map = {str(item["player_id"]): int(item["count"]) for item in trending_adds_raw}

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

    schedule_map = {}
    playing_teams_target_wk = set()
    now_utc = datetime.datetime.now(datetime.timezone.utc)

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

            is_final = pd.notnull(r.get("home_score")) and pd.notnull(r.get("away_score"))
            g_state = "FINAL"
            if not is_final:
                kickoff_str = f"{r.get('gameday', '')} {r.get('gametime', '13:00')}"
                try:
                    kickoff_dt = datetime.datetime.strptime(kickoff_str, "%Y-%m-%d %H:%M").replace(
                        tzinfo=datetime.timezone.utc
                    )
                    time_diff = (now_utc - kickoff_dt).total_seconds()
                    if time_diff < 0:
                        g_state = "UPCOMING"
                    elif time_diff <= 14400:
                        g_state = "IN_PLAY"
                    else:
                        g_state = "FINAL"
                except Exception:
                    g_state = "UPCOMING"

            schedule_map[(wk, ht)] = {
                "opp": at, "implied": h_imp, "total": tot_line, "is_home": True, "state": g_state
            }
            schedule_map[(wk, at)] = {
                "opp": ht, "implied": a_imp, "total": tot_line, "is_home": False, "state": g_state
            }
            if wk == target_week:
                playing_teams_target_wk.update([ht, at])
    except Exception:
        pass

    if not playing_teams_target_wk:
        wk_proj = get_json(f"https://api.sleeper.app/v1/projections/nfl/regular/{curr_season}/{target_week}") or {}
        if isinstance(wk_proj, dict):
            for pid, p_proj in wk_proj.items():
                if isinstance(p_proj, dict):
                    pts_p = p_proj.get("pts_ppr") or p_proj.get("pts_std") or p_proj.get("gp") or 0
                    if float(pts_p or 0) > 0:
                        t_abbr = norm_nfl_team((players_db.get(str(pid)) or {}).get("team"))
                        if t_abbr in ALL_32_NFL_TEAMS:
                            playing_teams_target_wk.add(t_abbr)

    bye_teams_target_wk = sorted(list(ALL_32_NFL_TEAMS - playing_teams_target_wk)) if playing_teams_target_wk else []

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

    completed_weeks = [w for w in range(max(1, target_week - lookback_weeks), target_week)] or [1]
    all_completed_season_wks = [w for w in range(1, target_week)] or [1]
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
        players_db,
        trending_map,
        val_map,
        rank_map,
        pos_rank_map,
        pos_rank_num_map,
        schedule_map,
        bye_teams_target_wk,
        nflverse_injuries,
        dvp_ranks,
        dict(weekly_usage),
        nflverse_metrics,
        ngs_sep_map,
    )


# ==============================================================================
# LAYER 2: LIVE SLEEPER LEAGUE STATE (BYPASSES CDN CACHE ON REFRESH)
# ==============================================================================
@st.cache_data(ttl=45, show_spinner=False)
def fetch_live_sleeper_league(league_id, target_week, sync_token=0):
    league_info = get_json(f"https://api.sleeper.app/v1/league/{league_id}", bust_cache=True) or {}
    users = get_json(f"https://api.sleeper.app/v1/league/{league_id}/users", bust_cache=True) or []
    rosters = get_json(f"https://api.sleeper.app/v1/league/{league_id}/rosters", bust_cache=True) or []
    matchups = get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{target_week}", bust_cache=True) or []
    prev_matchups = (
        get_json(f"https://api.sleeper.app/v1/league/{league_id}/matchups/{max(1, target_week - 1)}", bust_cache=True)
        or []
    )
    traded_picks = get_json(f"https://api.sleeper.app/v1/league/{league_id}/traded_picks", bust_cache=True) or []

    transactions_by_wk = {}
    for w_tx in range(1, target_week + 2):
        transactions_by_wk[w_tx] = (
            get_json(
                f"https://api.sleeper.app/v1/league/{league_id}/transactions/{w_tx}",
                bust_cache=(w_tx >= target_week - 1),
            )
            or []
        )

    return league_info, users, rosters, matchups, prev_matchups, traded_picks, transactions_by_wk


# ==============================================================================
# WAIVER BID AUDIT & GM BIDDING BEHAVIOR ENGINE
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
        "total_bids": 0, "wins": 0, "failed": 0, "all_bids": [],
        "winning_bids": [], "overpays": [], "pos_bids": defaultdict(list)
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

    auction_cols = [
        "Week", "Player", "Pos", "Winner_GM", "Winning_Bid", "Winning_Bid_Str",
        "Runner_Up_GM", "Runner_Up_Bid", "Runner_Up_Bid_Str", "Overpay_Margin",
        "Overpay_Str", "Total_Bidders", "All_Competing_Bids",
    ]
    df_auctions = pd.DataFrame(auction_audit_rows, columns=auction_cols)
    if not df_auctions.empty:
        df_auctions = df_auctions.sort_values(by=["Week", "Winning_Bid"], ascending=[False, False])

    profile_cols = [
        "GM", "Status", "Remaining_FAAB", "FAAB_Raw", "Bidder_Archetype",
        "Total_Bids (Won-Lost)", "Max_Single_Bid", "Avg_Bid_Amount",
        "Avg_Winning_Bid", "Max_Bid_By_Pos",
    ]
    df_gm_profiles = pd.DataFrame(gm_profile_rows, columns=profile_cols)
    if not df_gm_profiles.empty:
        df_gm_profiles = df_gm_profiles.sort_values("FAAB_Raw", ascending=False)

    return df_auctions, df_gm_profiles, gm_aggression_map


def build_player_profile(
    pid,
    target_week,
    players_db,
    val_map,
    rank_map,
    pos_rank_map,
    pos_rank_num_map,
    schedule_map,
    bye_teams_target_wk,
    nflverse_injuries,
    dvp_ranks,
    weekly_usage,
    nflverse_metrics,
    ngs_sep_map,
    trending_map,
    recently_dropped_pids,
    live_pts=0.0,
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
        practice_badge = "🔴 DNP"
        practice_risk = "HIGH RISK"
    elif "lim" in prac_raw.lower() or "lp" in prac_raw.lower():
        practice_badge = "🟡 LP (Limited)"
        practice_risk = "MONITOR"
    elif "full" in prac_raw.lower() or "fp" in prac_raw.lower():
        practice_badge = "🟢 FP (Full)"
        practice_risk = "CLEARED"
    elif raw_inj in ["Out", "IR", "PUP", "Doubtful"]:
        practice_badge = f"🔴 {raw_inj}"
        practice_risk = "OUT"
    elif raw_inj == "Questionable":
        practice_badge = "🟡 Questionable"
        practice_risk = "MONITOR"
    else:
        practice_badge = "🟢 Full"
        practice_risk = "HEALTHY"

    if slp_desc:
        practice_badge = f"{practice_badge} [{slp_desc}]"

    is_on_bye = team in bye_teams_target_wk
    if is_on_bye:
        status_str = f"🛑 BYE (Wk {target_week})" if not raw_inj else f"🛑 BYE / {raw_inj}"
        live_game_state = "🛑 BYE"
    else:
        status_str = f"{raw_inj} ({body_part})" if (raw_inj and body_part) else (raw_inj if raw_inj else "Healthy")

    sched_info = schedule_map.get((target_week, team))
    if is_on_bye:
        opp_str, vegas_imp, dvp_rk, dvp_str, live_game_state = "BYE", 0.0, 0, "BYE", "🛑 BYE"
    elif sched_info:
        opp_tm = sched_info["opp"]
        prefix = "vs" if sched_info["is_home"] else "@"
        opp_str = f"{prefix} {opp_tm}"
        vegas_imp = sched_info["implied"]
        dvp_rk = dvp_ranks.get((opp_tm, pos), 16)
        dvp_tier = "🟢 Easy" if dvp_rk >= 22 else ("🔴 Tough" if dvp_rk <= 10 else "🟡 Neutral")
        dvp_str = f"#{dvp_rk} {dvp_tier}"
        st_state = sched_info.get("state", "UPCOMING")
        if st_state == "FINAL":
            live_game_state = "🟢 Final"
        elif st_state == "IN_PLAY":
            live_game_state = "🔴 In-Play"
        else:
            live_game_state = "⏳ Upcoming"
    else:
        opp_str, vegas_imp, dvp_rk, dvp_str, live_game_state = "TBD", 21.5, 16, "#16 🟡 Neutral", "⏳ Upcoming"

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
            min(100.0, max(0.0, (avg_snap * 0.35) + (xfp_g * 2.5) + (floor_pts * 2.0) - (bust_rate * 0.25) + prac_penalty)),
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

    vol_score = 0.0 if is_on_bye else round(min(100.0, max(0.0, base_vol)), 1)

    contingency_score = 0.0
    if pos == "RB":
        tm_off_bonus = max(0.0, vegas_imp - 18.0) * 3.5
        youth_bonus = max(0.0, (26.0 - age) * 6.0) if age > 0 else 10.0
        role_factor = 25.0 if depth_order == 2 else (15.0 if depth_order >= 3 else 10.0)
        contingency_score = round(min(100.0, role_factor + tm_off_bonus + youth_bonus + (avg_snap * 0.4) + (val * 0.015)), 1)

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

    # Combined compact game environment string to prevent table clutter
    env_summary = "🛑 BYE" if is_on_bye else f"{opp_str} (Imp: {vegas_imp} | DvP {dvp_str})"

    return {
        "PID": str(pid),
        "Player": full,
        "Pos": pos,
        "Team": team,
        "Age": age if age > 0 else None,
        "Status": status_str,
        "Raw_Injury": raw_inj,
        "Body_Part": body_part,
        "Practice": practice_badge,
        "Practice_Risk": practice_risk,
        "Injury_Notes": slp_notes,
        "Is_Bye": is_on_bye,
        "Matchup": opp_str,
        "Game_Env": env_summary,
        "Game_State": live_game_state,
        "Wk_Pts": round(float(live_pts), 2),
        "Vegas_Implied": vegas_imp,
        "DvP_Matchup": dvp_str,
        "DvP_Rank": dvp_rk,
        "Pos_Rank": pos_rnk_str,
        "Pos_Rank_Num": pos_rnk_num,
        "Overall_Rank": f"#{rnk}" if rnk < 999 else "Unranked",
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
# SIDEBAR CONTROLS & LIFECYCLE ENGINE
# ==============================================================================
if "sync_token" not in st.session_state:
    st.session_state.sync_token = 0

state_raw = get_json("https://api.sleeper.app/v1/state/nfl", bust_cache=True) or {}
curr_season = int(state_raw.get("season", 2026))
live_nfl_week = max(1, int(state_raw.get("week") or state_raw.get("display_week") or 7))

today_weekday = datetime.datetime.now().weekday()
if today_weekday == 1:
    lifecycle_banner = f"🎯 Tuesday Waiver Strategy Phase (Week {live_nfl_week} Prep)"
elif today_weekday == 2:
    lifecycle_banner = f"📋 Wednesday Post-Waiver Consolidation & Practice Reports"
elif today_weekday in [3, 6, 0]:
    lifecycle_banner = f"🔴 Live Game Scoring Window (Week {live_nfl_week})"
else:
    lifecycle_banner = f"👀 Pre-Game Lineup Optimization (Week {live_nfl_week})"

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
        st.caption(f"Connected ID: `{league_id_input}` (`{active_mode}`)")
    else:
        league_id_input = st.text_input("🏆 Custom Sleeper League ID", value="1313348860042571776")
        active_mode = st.selectbox("League Strategy Mode", ["DYNASTY", "CHOP"])

    selected_week = st.selectbox(
        "📅 Select NFL Week",
        list(range(1, 19)),
        index=min(17, max(0, live_nfl_week - 1)),
        help="Defaults to current operational week. Jump back to review past scores or forward to plan ahead.",
    )

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
        "🤖 Primary AI Model (Auto-Cascades)",
        AI_MODEL_CASCADE,
        index=0,
        help="Starts with the highest model and automatically cascades down if free-tier rate limits are reached.",
    )

    if st.button("⚡ Force Live Sync / Refresh", use_container_width=True):
        st.session_state.sync_token += 1
        fetch_live_sleeper_league.clear()
        st.rerun()

# ==============================================================================
# FETCH LIVE SLEEPER LEAGUE STATE & REFERENCE DATA
# ==============================================================================
with st.spinner(f"Syncing live Sleeper rosters & Week {selected_week} data..."):
    league_info, users, rosters, matchups, prev_matchups, traded_picks, transactions_by_wk = (
        fetch_live_sleeper_league(league_id_input.strip(), selected_week, st.session_state.sync_token)
    )

    roster_positions = league_info.get("roster_positions") or []
    scoring_settings = league_info.get("scoring_settings") or {}
    league_settings = league_info.get("settings") or {}
    is_superflex = ("SUPER_FLEX" in roster_positions) or (roster_positions.count("QB") >= 2)

    (
        players_db,
        trending_map,
        val_map,
        rank_map,
        pos_rank_map,
        pos_rank_num_map,
        schedule_map,
        bye_teams_target_wk,
        nflverse_injuries,
        dvp_ranks,
        weekly_usage,
        nflverse_metrics,
        ngs_sep_map,
    ) = fetch_reference_data(
        curr_season,
        selected_week,
        lookback_wks,
        is_superflex,
        tuple(sorted(scoring_settings.items())),
    )

# ==============================================================================
# PARSE SETTINGS, FAAB, IN-PLAY PMR & ROSTERS
# ==============================================================================
u_map = {
    u["user_id"]: normalize_gm(u.get("display_name", "User"))
    for u in users
    if u.get("user_id")
}

rid_to_gm = {}
for r in rosters:
    if r and r.get("roster_id") is not None:
        rid_int = int(r.get("roster_id"))
        rid_to_gm[rid_int] = u_map.get(r.get("owner_id"), f"Roster_{rid_int}")

recently_dropped_pids = {}
raw_waiver_bids = []

for w_tx, tx_list in transactions_by_wk.items():
    for tx in tx_list:
        tx_status = tx.get("status")
        tx_type = tx.get("type")

        if tx_status == "complete" and w_tx >= max(1, selected_week - 1):
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
                gm_bidder = rid_to_gm.get(rid_int, f"Roster_{a_rid}")
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
player_live_pts_by_rid = defaultdict(dict)
matchup_starters_by_rid = {}

for m in matchups:
    if m and m.get("roster_id") is not None:
        rid_k = int(m["roster_id"])
        curr_wk_pts_by_rid[rid_k] = float(m.get("points") or 0.0)
        m_starters = [str(x) for x in (m.get("starters") or []) if x and str(x) != "0"]
        if m_starters:
            matchup_starters_by_rid[rid_k] = m_starters

        p_pts = m.get("players_points") or {}
        for p_id_k, pt_v in p_pts.items():
            player_live_pts_by_rid[rid_k][str(p_id_k)] = float(pt_v or 0.0)
        for st_id, st_pt in zip(m.get("starters") or [], m.get("starters_points") or []):
            if st_id:
                player_live_pts_by_rid[rid_k][str(st_id)] = float(st_pt or 0.0)

prev_wk_pts_by_rid = {}
for m in prev_matchups:
    if m and m.get("roster_id") is not None:
        prev_wk_pts_by_rid[int(m["roster_id"])] = float(m.get("points") or 0.0)

rostered_pids = set()
my_starters_pids, my_bench_pids, my_ir_pids, my_taxi_pids = [], [], [], []
my_roster_id = None
my_faab_rem = total_faab_budget
all_gm_rosters = {}

for r in rosters:
    if not r or r.get("roster_id") is None:
        continue
    rid = int(r.get("roster_id"))
    gm = u_map.get(r.get("owner_id"), f"Roster_{rid}")
    rid_to_gm[rid] = gm

    pids = [str(x) for x in (r.get("players") or []) if x and str(x) != "0"]
    roster_starters = [str(x) for x in (r.get("starters") or []) if x and str(x) != "0"]

    if selected_week < live_nfl_week and rid in matchup_starters_by_rid:
        starters = matchup_starters_by_rid[rid]
    else:
        starters = roster_starters if roster_starters else matchup_starters_by_rid.get(rid, [])

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

    pmr_in_play = 0
    pmr_upcoming = 0
    for s_pid in starters:
        tm_s = norm_nfl_team((players_db.get(str(s_pid)) or {}).get("team"))
        sc_s = (schedule_map.get((selected_week, tm_s)) or {}).get("state", "UPCOMING")
        if sc_s == "IN_PLAY":
            pmr_in_play += 1
        elif sc_s == "UPCOMING" and tm_s not in bye_teams_target_wk:
            pmr_upcoming += 1

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
        "pmr_in_play": pmr_in_play,
        "pmr_upcoming": pmr_upcoming,
        "pmr_total": pmr_in_play + pmr_upcoming,
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

my_spendable_faab = max(0, my_faab_rem - endgame_faab_reserve) if active_mode == "CHOP" else my_faab_rem

df_auctions, df_gm_profiles, gm_aggression_map = analyze_waiver_auctions_and_gm_profiles(
    raw_waiver_bids, all_gm_rosters, total_faab_budget
)


def build_df_from_pids(pid_list, rid_context=None):
    if not pid_list:
        return pd.DataFrame()
    res = []
    for p in pid_list:
        live_score = player_live_pts_by_rid.get(rid_context or my_roster_id or 0, {}).get(str(p), 0.0)
        res.append(
            build_player_profile(
                p, selected_week, players_db, val_map, rank_map, pos_rank_map, pos_rank_num_map,
                schedule_map, bye_teams_target_wk, nflverse_injuries, dvp_ranks, weekly_usage,
                nflverse_metrics, ngs_sep_map, trending_map, recently_dropped_pids, live_pts=live_score
            )
        )
    return pd.DataFrame(res)


my_starters_df = build_df_from_pids(my_starters_pids, my_roster_id)
my_bench_df = build_df_from_pids(my_bench_pids, my_roster_id)
my_taxi_df = build_df_from_pids(my_taxi_pids, my_roster_id)
my_ir_df = build_df_from_pids(my_ir_pids, my_roster_id)
my_full_roster_df = build_df_from_pids(my_starters_pids + my_bench_pids + my_taxi_pids + my_ir_pids, my_roster_id)

unrostered_list = []
for pid, p in players_db.items():
    if str(pid) not in rostered_pids and p.get("position") in ["QB", "RB", "WR", "TE"] and p.get("team"):
        prof = build_player_profile(
            pid, selected_week, players_db, val_map, rank_map, pos_rank_map, pos_rank_num_map,
            schedule_map, bye_teams_target_wk, nflverse_injuries, dvp_ranks, weekly_usage,
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
# RIVAL POSITIONAL NEEDS, DRAFT PICK INVENTORY & POSITIONAL SCOUT
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

        if tm in bye_teams_target_wk:
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
            phase_mult = 0.85 if selected_week <= 5 else (1.10 if selected_week <= 10 else 1.35)
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
            threat_strs.append(f"{t['gm']} (${t['faab']} | Est ~${scaled_bid} [{t['reason']}])")
        threats_summary = " | ".join(threat_strs)
    else:
        max_est_threat_bid = int(round(avg_surv_faab * 0.15 * tier_factor))
        max_needy_faab = max_rival_faab
        threats_summary = "No immediate starter hole"

    raw_opt = max_est_threat_bid + 3
    if raw_opt % 5 == 0:
        raw_opt += 2
    elif raw_opt % 10 == 0:
        raw_opt += 6

    if spendable_faab <= 0:
        return threats_summary, "$0 (Reserve Floor)", "$0 (Reserve Floor)", "$0"

    smart_bid = min(spendable_faab, max(1, raw_opt))
    smart_tag = f"${smart_bid}" if raw_opt <= spendable_faab else f"${smart_bid} (Capped)"

    lock_target = max_needy_faab + 1
    lock_tag = f"${lock_target} (Lock)" if lock_target <= spendable_faab else f"${spendable_faab} (Max)"
    val_bid = min(spendable_faab, max(1, int(round(max_est_threat_bid * 0.65))))
    return threats_summary, smart_tag, lock_tag, f"${val_bid}"


# Pre-compute Max PF Draft Order & Future Pick Inventory (2027-2029)
sorted_by_max_pf = sorted(all_gm_rosters.items(), key=lambda x: (x[1]["max_pf"], x[1]["wins"]))
rid_to_proj_slot = {}
for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
    rid_to_proj_slot[rdata["roster_id"]] = slot_idx

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

gm_picks_inventory = defaultdict(list)
for (yr_s, rnd, orid), owner_rid in pick_owner_lookup.items():
    owner_gm = rid_to_gm.get(owner_rid, f"Roster_{owner_rid}")
    orig_gm = rid_to_gm.get(orid, f"Roster_{orid}")
    slot_num = rid_to_proj_slot.get(orid, 6)

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


def build_league_positional_scout(all_gm_rosters, players_db, val_map, pos_rank_num_map):
    pos_matrix_rows = []
    all_max_pfs = [d["max_pf"] for d in all_gm_rosters.values()]
    median_max_pf = float(np.median(all_max_pfs)) if all_max_pfs else 500.0

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
        if active_mode == "CHOP":
            tier_str = rdata["chop_status"]
        else:
            if rdata["wins"] > rdata["losses"] and rdata["max_pf"] >= median_max_pf:
                tier_str = "🏆 Contender"
            elif rdata["wins"] < rdata["losses"] and rdata["max_pf"] < median_max_pf:
                tier_str = "🔨 Rebuilder"
            else:
                tier_str = "⚖️ Middle Tier"

        pos_matrix_rows.append({
            "GM": gm_u,
            "Tier_Status": tier_str,
            "Record": f"{rdata['wins']}-{rdata['losses']}",
            "FAAB": f"${rdata['faab_rem']}",
            "Total_Val": tot_v,
            "QB_Val": pos_vals["QB"],
            "RB_Val": pos_vals["RB"],
            "WR_Val": pos_vals["WR"],
            "TE_Val": pos_vals["TE"],
            "QB_Studs": f"{top12_counts['QB']}/{top24_counts['QB']}",
            "RB_Studs": f"{top12_counts['RB']}/{top24_counts['RB']}",
            "WR_Studs": f"{top12_counts['WR']}/{top24_counts['WR']}",
            "TE_Studs": f"{top12_counts['TE']}/{top24_counts['TE']}",
        })

    scout_cols = [
        "GM", "Tier_Status", "Record", "FAAB", "Total_Val", "QB_Val", "RB_Val", "WR_Val", "TE_Val",
        "QB_Studs", "RB_Studs", "WR_Studs", "TE_Studs",
        "QB_League_Rank", "RB_League_Rank", "WR_League_Rank", "TE_League_Rank",
        "QB_Rank_Unit", "RB_Rank_Unit", "WR_Rank_Unit", "TE_Rank_Unit",
    ]
    df_pos_scout = pd.DataFrame(pos_matrix_rows)
    if not df_pos_scout.empty and "Total_Val" in df_pos_scout.columns:
        for pos in ["QB", "RB", "WR", "TE"]:
            df_pos_scout[f"{pos}_League_Rank"] = (
                df_pos_scout[f"{pos}_Val"].rank(ascending=False, method="min").astype(int)
            )
            df_pos_scout[f"{pos}_Rank_Unit"] = df_pos_scout.apply(
                lambda r: f"#{r[f'{pos}_League_Rank']} ({r[f'{pos}_Val']:,} | {r[f'{pos}_Studs']} T12/24)",
                axis=1,
            )
        df_pos_scout = df_pos_scout.sort_values("Total_Val", ascending=False)
    else:
        df_pos_scout = pd.DataFrame(columns=scout_cols)
    return df_pos_scout


df_league_pos_scout = build_league_positional_scout(all_gm_rosters, players_db, val_map, pos_rank_num_map)

# ==============================================================================
# AUTOMATED ALERTS & PRACTICE PREVENTION ENGINE
# ==============================================================================
lineup_alerts = []
ir_alerts = []
volume_swap_alerts = []
practice_prevention_rows = []

if not my_starters_df.empty:
    for _, s_row in my_starters_df.iterrows():
        if s_row["Is_Bye"]:
            lineup_alerts.append(
                f"🛑 **BYE WEEK ALERT:** Starter **{s_row['Player']} ({s_row['Pos']} - {s_row['Team']})** is on a **BYE** in Week {selected_week}! Swap them out immediately."
            )
        elif s_row["Raw_Injury"] in ["Out", "Doubtful", "IR", "PUP", "Questionable", "Sus"] or s_row["Practice_Risk"] in ["HIGH RISK", "MONITOR"]:
            lineup_alerts.append(
                f"🚨 **Starter Practice/Injury Alert:** **{s_row['Player']} ({s_row['Pos']})** — Status: **{s_row['Status']}** | Practice: **{s_row['Practice']}**."
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
                "Practice_Report": r_row["Practice"],
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
# MODE-SPECIFIC METRICS & OPPONENT / CUT-LINE CONTEXT
# ==============================================================================
fmt_tag = "Superflex" if is_superflex else "1QB"
tep_val = float(scoring_settings.get("bonus_rec_te", 0.0) or 0.0)
ppr_val = float(scoring_settings.get("rec", 1.0) or 1.0)

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
                [str(x) for x in (opp_m.get("starters") or []) if x and str(x) != "0"], opp_rid
            )

surviving_teams = [(gm, d) for gm, d in all_gm_rosters.items() if d["is_alive"]]
chopped_teams = [(gm, d) for gm, d in all_gm_rosters.items() if not d["is_alive"]]
any_live_pts = any(d["curr_wk_pts"] > 0 for _, d in surviving_teams)

if surviving_teams:
    if any_live_pts:
        surviving_sorted = sorted(surviving_teams, key=lambda x: (x[1]["curr_wk_pts"], x[1]["fpts"]), reverse=True)
        cut_line_gm, cut_line_data = surviving_sorted[-1]
        cut_line_score = cut_line_data["curr_wk_pts"]
        my_curr_score = all_gm_rosters.get(gm_input, {}).get("curr_wk_pts", 0.0)
        safety_buffer = round(my_curr_score - cut_line_score, 2)
    else:
        surviving_sorted = sorted(surviving_teams, key=lambda x: x[1]["fpts"], reverse=True)
        cut_line_gm, cut_line_data = surviving_sorted[-1]
        cut_line_score = cut_line_data["fpts"]
        my_curr_score = all_gm_rosters.get(gm_input, {}).get("fpts", 0.0)
        safety_buffer = round(my_curr_score - cut_line_score, 2)
else:
    surviving_sorted = []
    cut_line_gm, cut_line_data = "N/A", {}
    cut_line_score, my_curr_score, safety_buffer = 0.0, 0.0, 0.0

my_pmr = all_gm_rosters.get(gm_input, {}).get("pmr_total", 0)
cut_pmr = cut_line_data.get("pmr_total", 0)
rival_surviving_faabs = [d["faab_rem"] for g, d in surviving_teams if g.lower() != gm_input.lower()]
max_rival_faab = max(rival_surviving_faabs) if rival_surviving_faabs else 0
avg_surviving_faab = int(np.mean([d["faab_rem"] for _, d in surviving_teams])) if surviving_teams else 0

# ==============================================================================
# TOP COMMAND BAR & PERSISTENT FLOATING AI COPILOT
# ==============================================================================
top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    st.markdown(f"## {selected_preset} — NFL Week {selected_week}")
    st.caption(
        f"**{lifecycle_banner}** | `{len(rosters)} Teams` | `{fmt_tag}` | `{ppr_val} PPR` | "
        f"`TE Prem: +{tep_val}` | **Waivers:** `Clears {waiver_day_str}s` | "
        f"**Bye Teams (Wk {selected_week}):** `{', '.join(bye_teams_target_wk) if bye_teams_target_wk else 'None'}`"
    )

with top_col2:
    # Persistent Floating AI Copilot Popover (Accessible from ANY Hub!)
    with st.popover("💬 Open AI War Room Copilot", use_container_width=True):
        st.markdown(f"### 🤖 {active_mode.title()} AI Copilot (Wk {selected_week})")
        st.caption("Automatically cascades from `gemini-3.1-pro-preview` down to Flash models if rate-limited.")

        chat_state_key = f"copilot_msgs_{league_id_input}_{selected_week}"
        if chat_state_key not in st.session_state:
            st.session_state[chat_state_key] = [
                {
                    "role": "assistant",
                    "content": (
                        f"Synced with **{gm_input}** for **Week {selected_week}** ({active_mode} Mode).\n"
                        "Use a quick button below or ask any question while browsing the hubs!"
                    ),
                }
            ]

        qa1, qa2 = st.columns(2)
        copilot_quick = None
        if qa1.button("⚡ Audit Wk Lineup", key="cp_q1", use_container_width=True):
            copilot_quick = (
                f"Audit my Week {selected_week} starting lineup vs my Active Bench (excluding Bye teams {bye_teams_target_wk} and Taxi). "
                "Recommend exact start/sit swaps using xFP/G, Snap %, Practice status, and Floor/Volume Score."
            )
        if qa2.button("📡 Top Waiver Claims", key="cp_q2", use_container_width=True):
            copilot_quick = (
                f"I have ${my_spendable_faab} spendable FAAB. Scan the top available waiver players for Week {selected_week}, "
                "check rival positional needs, and give me my top 3 claims with exact FAAB bids."
            )

        qa3, qa4 = st.columns(2)
        if qa3.button("🏥 Practice Pivots", key="cp_q3", use_container_width=True):
            copilot_quick = "Check my roster's Practice & Injury alerts (DNP/LP). Who is at risk for Sunday and who is my best pivot?"
        if qa4.button("🤝 Trade / Strategy", key="cp_q4", use_container_width=True):
            if active_mode == "DYNASTY":
                active_trade_ctx = st.session_state.get("active_trade_context", "No trade currently selected in Hub 3.")
                copilot_quick = f"Evaluate my current trade desk context ({active_trade_ctx}) or propose 2 high-leverage contender trades."
            else:
                copilot_quick = "Analyze the 18-team Guillotine leaderboard and rival FAAB stacks. How should I manage my budget over the next 3 weeks?"

        chat_container = st.container(height=340)
        with chat_container:
            for m_msg in st.session_state[chat_state_key]:
                with st.chat_message(m_msg["role"]):
                    st.markdown(m_msg["content"])

        cp_input = st.chat_input("Ask Copilot anything...", key="cp_chat_in") or copilot_quick
        if cp_input:
            st.session_state[chat_state_key].append({"role": "user", "content": cp_input})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(cp_input)

                if not api_key:
                    with st.chat_message("assistant"):
                        st.warning("Please enter your Gemini API Key in the left sidebar.")
                else:
                    active_trade_ctx = st.session_state.get("active_trade_context", "None")
                    copilot_sys = f"""
                    You are an elite NFL Fantasy Football Data Scientist and GM Advisor for Carlos ('{gm_input}') in League {league_id_input} ({active_mode} Mode).
                    Respond strictly in English. Be concise, sharp, and cite exact numbers (xFP/G, Snap %, Floor/Volume Score, Dynasty Val, FAAB).
                    Current Context: {curr_season} Season, Week {selected_week}. Phase: {lifecycle_banner}.
                    NFL Teams on BYE in Week {selected_week}: {bye_teams_target_wk}.
                    Allowed IR Designations: {sorted(list(allowed_ir_tags))}.
                    Total FAAB: ${my_faab_rem} | Spendable FAAB (after ${endgame_faab_reserve} Endgame Reserve): ${my_spendable_faab}.

                    CRITICAL RULES:
                    1. NEVER recommend starting any player on BYE ({bye_teams_target_wk}) or Out/IR.
                    2. NEVER recommend starting Taxi Squad players without noting they must be promoted.
                    3. In CHOP mode, never recommend bids exceeding ${my_spendable_faab} (protect the ${endgame_faab_reserve} endgame reserve).

                    LIVE DATA:
                    - Starters: {my_starters_df.to_dict(orient='records') if not my_starters_df.empty else []}
                    - Active Bench: {my_bench_df.to_dict(orient='records') if not my_bench_df.empty else []}
                    - Practice & Injury Alerts: {practice_prevention_rows}
                    - Top 20 Waiver Players: {df_waivers.head(20).to_dict(orient='records') if not df_waivers.empty else []}
                    - Rival Positional Needs: {dict(rival_pos_needs)}
                    - League Positional Rankings: {df_league_pos_scout.to_dict(orient='records') if not df_league_pos_scout.empty else []}
                    - Active Trade Loaded in Hub 3: {active_trade_ctx}
                    """
                    try:
                        client = genai.Client(api_key=api_key)
                        with st.chat_message("assistant"):
                            with st.spinner("Analyzing..."):
                                ans, used_m = generate_ai_response_with_fallback(
                                    client, model_choice, cp_input, copilot_sys
                                )
                                full_reply = f"{ans}\n\n*⚡ Answered via `{used_m}`*"
                                st.markdown(full_reply)
                                st.session_state[chat_state_key].append({"role": "assistant", "content": full_reply})
                    except Exception as e:
                        st.error(f"API Error: {e}")

# Top KPI Strip (Mode-Adaptive)
if active_mode == "CHOP":
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Teams Alive", f"{len(surviving_teams)} / {len(all_gm_rosters)}", delta=f"{len(chopped_teams)} Chopped")
    k2.metric(f"Cut Line ({cut_line_gm})", f"{cut_line_score:.2f} pts", delta=f"{cut_pmr} PMR")
    k3.metric("Your Safety Buffer", f"{safety_buffer:+.2f} pts", delta=f"{my_pmr} PMR")
    k4.metric("Spendable FAAB Now", f"${my_spendable_faab}", delta=f"${endgame_faab_reserve} Reserved (Total: ${my_faab_rem})")
    k5.metric("Action Alerts", len(lineup_alerts) + len(volume_swap_alerts))
else:
    my_score_w = all_gm_rosters.get(gm_input, {}).get("curr_wk_pts", 0.0)
    opp_score_w = all_gm_rosters.get(opp_gm_name, {}).get("curr_wk_pts", 0.0)
    opp_pmr = all_gm_rosters.get(opp_gm_name, {}).get("pmr_total", 0)
    my_eff = all_gm_rosters.get(gm_input, {}).get("coach_eff_%", 100.0)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Week Matchup", f"vs. {opp_gm_name}", delta=f"{my_score_w:.1f} - {opp_score_w:.1f}")
    k2.metric("Players Remaining (PMR)", f"{my_pmr} PMR", delta=f"Opp: {opp_pmr} PMR")
    k3.metric("Remaining FAAB", f"${my_faab_rem} / ${total_faab_budget}")
    k4.metric("Practice / Lineup Alerts", len(lineup_alerts) + len(volume_swap_alerts))
    k5.metric("Coaching Efficiency", f"{my_eff}%")

# Shared Column Configuration for Clean Visual Progress Bars
SHARED_COL_CFG = {
    "Snap_%": st.column_config.ProgressColumn("Snap %", format="%.1f%%", min_value=0, max_value=100),
    "Volume_Score": st.column_config.ProgressColumn("Vol Score", format="%.1f", min_value=0, max_value=100),
    "Chop_Floor_Score": st.column_config.ProgressColumn("Floor Score", format="%.1f", min_value=0, max_value=100),
    "Contingency_Score": st.column_config.ProgressColumn("Contingency", format="%.1f", min_value=0, max_value=100),
    "Dynasty_Val": st.column_config.NumberColumn("Dynasty Val", format="%d"),
    "xFP/G": st.column_config.NumberColumn("xFP/G", format="%.1f"),
    "Recent_PPG": st.column_config.NumberColumn("PPG", format="%.1f"),
    "Wk_Pts": st.column_config.NumberColumn("Wk Pts", format="%.2f"),
}

# ==============================================================================
# THE 4 UNIFIED WORKFLOW HUBS (SHARED STRUCTURE FOR BOTH LEAGUES)
# ==============================================================================
hub1, hub2, hub3, hub4 = st.tabs([
    "⚡ 1. Weekly Gameplan & Lineup",
    "📡 2. Waiver & FAAB Intelligence",
    "🤝 3. Rivals, Scouting & Trade Desk" if active_mode == "DYNASTY" else "👁️ 3. Rival Survival & Vulnerability Scout",
    "📈 4. Deep Analytics & Portfolio Lab",
])

# ------------------------------------------------------------------------------
# HUB 1: WEEKLY GAMEPLAN, INTEGRATED PRACTICE PIVOTS & LINEUP CENTER
# ------------------------------------------------------------------------------
with hub1:
    # 1A. Actionable Alerts & Integrated Pre-Sunday Practice Pivots
    if lineup_alerts or ir_alerts or volume_swap_alerts:
        st.markdown("#### 🚨 Immediate Lineup & Pre-Sunday Action Items")
        for al in lineup_alerts:
            st.error(al)
        for sw in volume_swap_alerts:
            st.warning(sw)
        for ir_a in ir_alerts:
            st.info(ir_a)
    else:
        st.success(f"✅ Week {selected_week} Lineup Optimized: All starters are active, practicing, and carry your highest volume/floor scores!")

    if practice_prevention_rows:
        with st.expander(f"🏥 Pre-Sunday Practice Report & Injury Pivot Plan ({len(practice_prevention_rows)} Flagged Players)", expanded=True):
            st.dataframe(pd.DataFrame(practice_prevention_rows), use_container_width=True, hide_index=True)

    # 1B. Live Scoreboard / Guillotine Cut-Line Summary
    if active_mode == "CHOP":
        with st.expander("🪓 Live 18-Team Guillotine Cut-Line Standings & PMR", expanded=False):
            board_rows = []
            for rank_idx, (gm_u, d) in enumerate(surviving_sorted, 1):
                buffer_vs_cut = round((d["curr_wk_pts"] if any_live_pts else d["fpts"]) - cut_line_score, 2)
                zone_tag = (
                    "🩸 CUT LINE (Last Place)"
                    if rank_idx == len(surviving_sorted)
                    else ("⚠️ DANGER ZONE" if rank_idx >= len(surviving_sorted) - 2 else "🟢 Safe")
                )
                board_rows.append({
                    "Rank": f"#{rank_idx}",
                    "GM": f"🦅 {gm_u}" if gm_u.lower() == gm_input.lower() else gm_u,
                    "Zone": zone_tag,
                    f"Wk_{selected_week}_Pts": d["curr_wk_pts"],
                    "PMR (Left)": f"{d['pmr_total']} ({d['pmr_in_play']} 🔴)",
                    "Buffer": f"{buffer_vs_cut:+.2f}",
                    "FAAB": f"${d['faab_rem']}",
                    "Season_Pts": d["fpts"],
                    "Bidder_Style": gm_aggression_map.get(gm_u, {}).get("archetype", "-"),
                })
            st.dataframe(pd.DataFrame(board_rows), use_container_width=True, hide_index=True)

    # 1C. Decluttered Start/Sit Comparator with Optional Deep Metrics Toggle
    col_hdr, col_tog = st.columns([3, 1])
    col_hdr.markdown(f"#### 🏆 Week {selected_week} Starting Lineup vs. Active Bench")
    show_deep_cols = col_tog.checkbox("🔬 Show Deep Metrics (WOPR, PACR, NGS, FPOE)", value=False)

    primary_score_col = "Chop_Floor_Score" if active_mode == "CHOP" else "Volume_Score"
    base_cols = [
        "Player", "Pos", "Pos_Rank", "Team", "Status", "Practice",
        "Game_Env", "Game_State", "Wk_Pts", primary_score_col, "xFP/G", "Recent_PPG", "Snap_%"
    ]
    if active_mode == "CHOP":
        base_cols.insert(10, "Bust_Rate_%")
        base_cols.insert(11, "Floor_25th_Pts")
    else:
        base_cols.append("Dynasty_Val")

    deep_extra_cols = ["Snap_Trend", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "PACR", "NGS_Sep", "FPOE"]
    active_table_cols = base_cols + (deep_extra_cols if show_deep_cols else [])

    if not my_starters_df.empty:
        st.dataframe(
            my_starters_df[active_table_cols],
            column_config=SHARED_COL_CFG,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(f"#### 🪑 Active Bench (Sorted by {primary_score_col.replace('_', ' ')})")
    if not my_bench_df.empty:
        st.dataframe(
            my_bench_df.sort_values(by=["Is_Bye", primary_score_col], ascending=[True, False])[active_table_cols],
            column_config=SHARED_COL_CFG,
            use_container_width=True,
            hide_index=True,
        )

    if active_mode == "DYNASTY" or not my_ir_df.empty:
        c_tx, c_ir = st.columns(2)
        with c_tx:
            if active_mode == "DYNASTY":
                with st.expander(f"🚕 Taxi Squad ({len(my_taxi_pids)} Stashed — Excluded from Start/Sit Alerts)", expanded=False):
                    if not my_taxi_df.empty:
                        st.dataframe(my_taxi_df[base_cols], column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)
                    else:
                        st.caption("Empty Taxi Squad.")
        with c_ir:
            with st.expander(f"🏥 Injured Reserve / IR ({len(my_ir_pids)} / {max_ir_slots} Slots Used)", expanded=False):
                if not my_ir_df.empty:
                    st.dataframe(my_ir_df[base_cols], column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)
                else:
                    st.caption("Empty IR.")

# ------------------------------------------------------------------------------
# HUB 2: WAIVER WIRE, FAAB PREDICTOR & INTEGRATED AUCTION ARCHIVE
# ------------------------------------------------------------------------------
with hub2:
    st.subheader("📡 Unrostered Waiver Wire Radar & FAAB Bid Intelligence")
    if active_mode == "CHOP":
        st.info(
            f"🛡️ **Endgame FAAB Protocol:** Total FAAB: **`${my_faab_rem}`** | Locked Reserve: **`${endgame_faab_reserve}`** (for late-season $1 sweeps) | **Max Spendable Now: `${my_spendable_faab}`** | Max Rival: `${max_rival_faab}`"
        )
    else:
        avg_rem_faab = int(np.mean([d["faab_rem"] for d in all_gm_rosters.values()])) if all_gm_rosters else 0
        st.info(f"💰 **Your Remaining FAAB:** `${my_faab_rem} / ${total_faab_budget}` (League Average: `${avg_rem_faab}`)")

    wf1, wf2, wf3 = st.columns([1.5, 1, 1.5])
    waiver_preset_mode = wf1.radio(
        "🎯 Target Filter Mode",
        ["All Available", "🥩 Newly Chopped / Dropped", "🛡️ High-Floor Starters", "🔒 RB Contingency Handcuffs"],
        horizontal=True,
    )
    w_pos_sel = wf2.selectbox("Position", ["ALL", "RB", "WR", "TE", "QB"])
    w_sort_sel = wf3.selectbox(
        "Sort Targets By",
        ["Chop_Floor_Score" if active_mode == "CHOP" else "Volume_Score", "xFP/G", "Contingency_Score", "24h_Adds", "Snap_%", "Dynasty_Val"],
    )

    if not df_waivers.empty:
        vw = df_waivers.copy()
        if w_pos_sel != "ALL":
            vw = vw[vw["Pos"] == w_pos_sel]
        if waiver_preset_mode == "🥩 Newly Chopped / Dropped":
            vw = vw[vw["Chopped_Status"].str.contains("Chopped")]
        elif waiver_preset_mode == "🛡️ High-Floor Starters":
            vw = vw[vw["xFP/G"] >= 8.5]
        elif waiver_preset_mode == "🔒 RB Contingency Handcuffs":
            vw = vw[vw["Pos"] == "RB"].sort_values("Contingency_Score", ascending=False)

        if active_mode == "CHOP" and not vw.empty:
            threats_col, smart_bids, lock_bids, val_bids = [], [], [], []
            for _, wr in vw.iterrows():
                th_str, sm_b, lk_b, vl_b = predict_smart_faab_for_player(
                    wr, my_spendable_faab, max_rival_faab, avg_surviving_faab
                )
                threats_col.append(th_str)
                smart_bids.append(sm_b)
                lock_bids.append(lk_b)
                val_bids.append(vl_b)

            vw["🎯 Rival Threats & Est. Bid"] = threats_col
            vw["💡 Optimal Bid"] = smart_bids
            vw["🔒 Lock-In Bid"] = lock_bids
            vw["💰 Value Bid"] = val_bids

            if waiver_preset_mode != "🔒 RB Contingency Handcuffs":
                vw = vw.sort_values(by=w_sort_sel, ascending=False)

            w_display_cols = [
                "Player", "Pos", "Pos_Rank", "Team", "Chopped_Status", "Status", "Game_Env",
                "Chop_Floor_Score", "xFP/G", "Recent_PPG", "Bust_Rate_%", "Snap_%",
                "🎯 Rival Threats & Est. Bid", "💡 Optimal Bid", "🔒 Lock-In Bid", "💰 Value Bid"
            ]
        else:
            if not vw.empty and waiver_preset_mode != "🔒 RB Contingency Handcuffs":
                vw = vw.sort_values(by=w_sort_sel, ascending=False)
            w_display_cols = [
                "Player", "Pos", "Pos_Rank", "Team", "Age", "Status", "Practice", "Game_Env",
                "Volume_Score", "xFP/G", "Recent_PPG", "Contingency_Score", "Snap_%", "24h_Adds", "Dynasty_Val"
            ]

        st.dataframe(vw[w_display_cols].head(35), column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)

    st.markdown("---")
    # Integrated Auction Archive & Bidder Profiles on the Exact Same Screen
    col_arch, col_prof = st.columns([1.2, 1])
    with col_arch:
        st.markdown("#### 🕵️‍♂️ League FAAB Auction Archive (Winning & Losing Bids)")
        search_auc = st.text_input("🔍 Filter Archive by Player or GM:", value="", key="hub2_auc_search")
        if not df_auctions.empty:
            v_auc = df_auctions.copy()
            if search_auc.strip():
                q_a = search_auc.strip().lower()
                v_auc = v_auc[
                    v_auc["Player"].str.lower().str.contains(q_a)
                    | v_auc["Winner_GM"].str.lower().str.contains(q_a)
                    | v_auc["All_Competing_Bids"].str.lower().str.contains(q_a)
                ]
            auc_show_cols = [
                "Week", "Player", "Winner_GM", "Winning_Bid_Str",
                "Runner_Up_GM", "Runner_Up_Bid_Str", "Overpay_Str", "All_Competing_Bids"
            ]
            st.dataframe(v_auc[auc_show_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No waiver auctions logged yet.")

    with col_prof:
        st.markdown("#### 🧠 Rival GM Bidding Behavior Profiles")
        if not df_gm_profiles.empty:
            prof_show_cols = [
                "GM", "Status", "Remaining_FAAB", "Bidder_Archetype",
                "Total_Bids (Won-Lost)", "Max_Single_Bid", "Avg_Bid_Amount", "Max_Bid_By_Pos"
            ]
            st.dataframe(df_gm_profiles[prof_show_cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------------------
# HUB 3: UNIFIED RIVAL SCOUTING & EXECUTIVE TRADE DESK
# ------------------------------------------------------------------------------
with hub3:
    st.subheader("🤝 League Positional Power Matrix & Integrated Rival Scout / Trade Desk")

    if not df_league_pos_scout.empty:
        with st.expander("📊 League-Wide Positional Power Matrix (#1 to #Last in QB, RB, WR, TE)", expanded=True):
            scout_summary_cols = [
                "GM", "Tier_Status", "Record", "FAAB", "Total_Val",
                "QB_Rank_Unit", "RB_Rank_Unit", "WR_Rank_Unit", "TE_Rank_Unit"
            ]
            st.dataframe(df_league_pos_scout[scout_summary_cols], use_container_width=True, hide_index=True)

    rival_gms = [g for g in sorted(all_gm_rosters.keys()) if g.lower() != gm_input.strip().lower()]
    if not rival_gms:
        st.info("No rival rosters available.")
    else:
        st.markdown("---")
        partner_gm = st.selectbox("🎯 Select Rival GM to Scout & Negotiate With", rival_gms, key="hub3_partner")
        p_r_info = all_gm_rosters.get(partner_gm, {})
        p_rid = p_r_info.get("roster_id")

        # Show Partner's Positional Ranks & Held Future Picks Immediately!
        p_scout_row = df_league_pos_scout[df_league_pos_scout["GM"] == partner_gm]
        p_tier_str = p_scout_row.iloc[0]["Tier_Status"] if not p_scout_row.empty else "Rival"
        p_picks_held = [pk["Short_Label"] for pk in gm_picks_inventory.get(partner_gm, []) if pk["Round"] in [1, 2]]

        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("Selected Rival", f"{partner_gm}", delta=p_tier_str)
        if not p_scout_row.empty:
            sc2.metric("QB Unit Rank", f"#{p_scout_row.iloc[0]['QB_League_Rank']}", delta=f"{p_scout_row.iloc[0]['QB_Val']:,} val")
            sc3.metric("RB Unit Rank", f"#{p_scout_row.iloc[0]['RB_League_Rank']}", delta=f"{p_scout_row.iloc[0]['RB_Val']:,} val")
            sc4.metric("WR Unit Rank", f"#{p_scout_row.iloc[0]['WR_League_Rank']}", delta=f"{p_scout_row.iloc[0]['WR_Val']:,} val")
            sc5.metric("TE Unit Rank", f"#{p_scout_row.iloc[0]['TE_League_Rank']}", delta=f"{p_scout_row.iloc[0]['TE_Val']:,} val")

        if active_mode == "DYNASTY" and p_picks_held:
            st.caption(f"**🔮 {partner_gm}'s Key Future Picks (Rd 1–2):** `{' | '.join(p_picks_held[:6])}`")

        # Expandable Full Roster Inspector right above the Trade Machine
        with st.expander(f"👁️ Inspect `{partner_gm}`'s Full Roster & Live Positional Ranks (Starters, Bench, Taxi, IR)", expanded=(active_mode == "CHOP")):
            p_st_df = build_df_from_pids(p_r_info.get("starters", []), p_rid)
            p_bn_df = build_df_from_pids(p_r_info.get("bench", []), p_rid)
            p_tx_df = build_df_from_pids(p_r_info.get("taxi", []), p_rid)
            scout_view_cols = [
                "Player", "Pos", "Pos_Rank", "Overall_Rank", "Team", "Age", "Status",
                "Practice", "Game_Env", "Wk_Pts", "xFP/G", "Recent_PPG", "Snap_%", "Dynasty_Val"
            ]
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**🏆 {partner_gm}'s Starters**")
                if not p_st_df.empty:
                    st.dataframe(p_st_df[scout_view_cols], column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)
            with rc2:
                st.markdown(f"**🪑 {partner_gm}'s Bench**")
                if not p_bn_df.empty:
                    st.dataframe(p_bn_df.sort_values("Dynasty_Val", ascending=False)[scout_view_cols], column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)
                if active_mode == "DYNASTY" and not p_tx_df.empty:
                    st.markdown(f"**🚕 {partner_gm}'s Taxi Squad**")
                    st.dataframe(p_tx_df.sort_values("Dynasty_Val", ascending=False)[scout_view_cols], column_config=SHARED_COL_CFG, use_container_width=True, hide_index=True)

        # Dynasty Trade Machine (Only in Dynasty Mode)
        if active_mode == "DYNASTY":
            st.markdown("#### ⚖️ Positional Trade Machine & Persuasion Pitch Generator")

            def build_pos_grouped_assets(gm_u):
                r_d = all_gm_rosters.get(gm_u, {})
                pids = r_d.get("players", [])
                df_p = build_df_from_pids(pids, r_d.get("roster_id"))
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
                st.markdown(f"##### 📤 `{gm_input}` Sends")
                for pos_k, pos_title in [("QB", "QB"), ("RB", "RB"), ("WR", "WR"), ("TE", "TE"), ("PICK", "Draft Picks (2027-2029)")]:
                    opts = list(my_assets_by_pos[pos_k].keys())
                    sel = st.multiselect(f"Send {pos_title}", opts, key=f"send_{pos_k}")
                    for k in sel:
                        sent_items.append(my_assets_by_pos[pos_k][k])

            rec_items = []
            with col_rec:
                st.markdown(f"##### 📥 `{partner_gm}` Sends")
                for pos_k, pos_title in [("QB", "QB"), ("RB", "RB"), ("WR", "WR"), ("TE", "TE"), ("PICK", "Draft Picks (2027-2029)")]:
                    opts = list(partner_assets_by_pos[pos_k].keys())
                    sel = st.multiselect(f"Receive {pos_title}", opts, key=f"rec_{pos_k}")
                    for k in sel:
                        rec_items.append(partner_assets_by_pos[pos_k][k])

            if sent_items or rec_items:
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

                sent_names_clean = ", ".join([x.get("Player") or x.get("Short_Label") for x in sent_items]) or "None"
                rec_names_clean = ", ".join([x.get("Player") or x.get("Short_Label") for x in rec_items]) or "None"

                # Sync active trade to session_state so the Persistent AI Copilot can see it!
                st.session_state["active_trade_context"] = (
                    f"Trade with {partner_gm}: {gm_input} sends [{sent_names_clean}] (Val: {send_val}, {send_xfp} xFP/G) "
                    f"and receives [{rec_names_clean}] (Val: {get_val}, {get_xfp} xFP/G)."
                )

                max_side = max(1, send_val, get_val)
                pct_diff = abs(val_delta) / max_side * 100.0
                fair_badge = (
                    "⚖️ FAIR MARKET DEAL (Within 8%)"
                    if pct_diff <= 8.0
                    else (f"🟢 SURPLUS WIN FOR {gm_input.upper()} (+{pct_diff:.0f}%)" if val_delta > 0 else f"🟠 SURPLUS WIN FOR {partner_gm.upper()} (+{pct_diff:.0f}%)")
                )

                st.markdown(f"##### 📊 Trade Verdict: {fair_badge}")
                m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                m_c1.metric("You Receive (Dynasty Val)", f"{get_val:,}", delta=f"{val_delta:+,} vs Sent ({send_val:,})")
                m_c2.metric("Volume Impact (xFP/G)", f"{get_xfp:.1f} xFP/G", delta=f"{xfp_delta:+.1f} xFP/G")
                m_c3.metric("Scoring Impact (PPG)", f"{get_ppg:.1f} PPG", delta=f"{ppg_delta:+.1f} PPG")
                m_c4.metric(
                    "Avg Age Acquired",
                    f"{avg_get_age:.1f} yrs" if avg_get_age > 0 else "Picks Only",
                    delta=f"{age_delta:+.1f} yrs vs Sent" if (send_ages and get_ages) else None,
                    delta_color="inverse",
                )

                pos_breakdown_rows = []
                for p_cat in ["QB", "RB", "WR", "TE", "PICK"]:
                    s_cat = [x for x in sent_items if x["Pos"] == p_cat]
                    r_cat = [x for x in rec_items if x["Pos"] == p_cat]
                    if not s_cat and not r_cat:
                        continue
                    s_v = sum(x["Dynasty_Val"] for x in s_cat)
                    r_v = sum(x["Dynasty_Val"] for x in r_cat)
                    s_x = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in s_cat), 1)
                    r_x = round(sum(x.get("xFP/G", 0.0) or 0.0 for x in r_cat), 1)
                    pos_breakdown_rows.append({
                        "Group": "Draft Picks" if p_cat == "PICK" else p_cat,
                        f"{gm_input}_Sends": " + ".join([x.get("Player") or x.get("Short_Label") for x in s_cat]) or "-",
                        "Sent_Val": f"{s_v:,}",
                        f"{gm_input}_Gets": " + ".join([x.get("Player") or x.get("Short_Label") for x in r_cat]) or "-",
                        "Got_Val": f"{r_v:,}",
                        "Net_Delta": f"{r_v - s_v:+,} val ({r_x - s_x:+.1f} xFP/G)",
                    })
                st.dataframe(pd.DataFrame(pos_breakdown_rows), use_container_width=True, hide_index=True)

                partner_weak_pos = []
                if not p_scout_row.empty:
                    for pos_c in ["QB", "RB", "WR", "TE"]:
                        rk_c = int(p_scout_row.iloc[0][f"{pos_c}_League_Rank"])
                        if rk_c >= 7:
                            partner_weak_pos.append(f"{pos_c} (#{rk_c})")

                copy_paste_msg = (
                    f"Hey {partner_gm}! Put together a trade idea that fits both of our rosters:\n"
                    f"• You Receive: {sent_names_clean} (Dynasty Value: {send_val:,} | {send_xfp:.1f} xFP/G)\n"
                    f"• I Receive: {rec_names_clean} (Dynasty Value: {get_val:,} | {get_xfp:.1f} xFP/G)\n"
                )
                st.code(copy_paste_msg, language="text")

                if st.button("✨ Generate Custom AI Persuasion Pitch for This Deal", use_container_width=True):
                    if not api_key:
                        st.warning("Please enter your Gemini API Key in the sidebar.")
                    else:
                        pitch_prompt = f"""
                        Write a persuasive, friendly, data-backed 2-paragraph trade message from '{gm_input}' to '{partner_gm}' ({p_tier_str}) pitching this Dynasty trade:
                        - {partner_gm} Receives: {sent_items} (Total Val: {send_val}, Total xFP/G: {send_xfp})
                        - {gm_input} Receives: {rec_items} (Total Val: {get_val}, Total xFP/G: {get_xfp})
                        - {partner_gm}'s Weakest Positional Ranks: {partner_weak_pos}
                        Cite exact snap %, xFP/G, positional ranks, or projected draft pick slots. Keep it natural and ready to paste into Sleeper chat.
                        """
                        try:
                            client = genai.Client(api_key=api_key)
                            with st.spinner("Crafting custom GM trade pitch..."):
                                ai_pitch, used_m = generate_ai_response_with_fallback(
                                    client, model_choice, pitch_prompt, "You are an expert Dynasty Fantasy Football negotiator."
                                )
                                st.info(f"{ai_pitch}\n\n*⚡ Crafted via `{used_m}`*")
                        except Exception as e:
                            st.error(f"API Error: {e}")

# ------------------------------------------------------------------------------
# HUB 4: DEEP ANALYTICS, xFP REGRESSION, AGE CLIFFS & MAX PF PICK BOARD
# ------------------------------------------------------------------------------
with hub4:
    st.subheader("📈 Expected Fantasy Points (xFP) Regression Lab & Dynasty Portfolio")
    st.caption(
        "**FPOE (Fantasy Points Over Expected)** = `Actual PPG - xFP/G`. Negative FPOE with high xFP/G flags an imminent **Buy-Low / Positive Regression Breakout**."
    )

    if not my_full_roster_df.empty:
        reg_df = my_full_roster_df.copy()
        reg_df["Regression_Signal"] = np.where(
            (reg_df["xFP/G"] >= 10.0) & (reg_df["FPOE"] <= -2.0),
            "🟢 BUY LOW / MUST START (Due for +TDs)",
            np.where(
                (reg_df["FPOE"] >= 3.5) & (reg_df["xFP/G"] < 12.0),
                "🟠 SELL HIGH (TD Dependent)",
                "⚪ Performing to Volume",
            ),
        )
        reg_cols = [
            "Player", "Pos", "Pos_Rank", "Team", "Age", "Age_Cliff", "xFP/G", "Recent_PPG",
            "FPOE", "Regression_Signal", "Snap_%", "Tgt/G", "Car/G", "RZ_Touch/G", "WOPR", "PACR", "NGS_Sep", "Dynasty_Val"
        ]
        st.dataframe(
            reg_df.sort_values("xFP/G", ascending=False)[reg_cols],
            column_config=SHARED_COL_CFG,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    c_bot1, c_bot2 = st.columns([1.2, 1])

    with c_bot1:
        if active_mode == "DYNASTY":
            st.markdown(f"#### 🔮 Live {curr_season + 1} Rookie Draft Order Forecaster (Inverse Max PF)")
            next_draft_yr = str(curr_season + 1)
            pick_forecast_rows = []
            for slot_idx, (orig_gm, rdata) in enumerate(sorted_by_max_pf, 1):
                orig_rid = rdata["roster_id"]
                r1_owner_gm = rid_to_gm.get(pick_owner_lookup.get((next_draft_yr, 1, orig_rid), orig_rid), orig_gm)
                r2_owner_gm = rid_to_gm.get(pick_owner_lookup.get((next_draft_yr, 2, orig_rid), orig_rid), orig_gm)
                pick_forecast_rows.append({
                    "Proj_Slot": f"1.{slot_idx:02d}",
                    "Orig_GM": orig_gm,
                    "Record": f"{rdata['wins']}-{rdata['losses']}",
                    "Max_PF": rdata["max_pf"],
                    f"{next_draft_yr}_1st_Owner": f"🦅 {r1_owner_gm}" if r1_owner_gm.lower() == gm_input.lower() else r1_owner_gm,
                    f"{next_draft_yr}_2nd_Owner": f"🦅 {r2_owner_gm}" if r2_owner_gm.lower() == gm_input.lower() else r2_owner_gm,
                })
            st.dataframe(pd.DataFrame(pick_forecast_rows), use_container_width=True, hide_index=True)
        else:
            st.markdown("#### 💵 Surviving Teams FAAB Power Trajectory")
            faab_traj_rows = [
                {"GM": g, "Status": d["chop_status"], "FAAB_Left": f"${d['faab_rem']}", "FAAB_Spent": f"${d['faab_used']}", "Season_Pts": d["fpts"]}
                for g, d in surviving_sorted
            ]
            st.dataframe(pd.DataFrame(faab_traj_rows), use_container_width=True, hide_index=True)

    with c_bot2:
        st.markdown("#### 🧠 League Coaching Lineup Efficiency (`Actual PF / Max PF`)")
        eff_rows = []
        for gm_u, d in all_gm_rosters.items():
            pts_left = round(max(0.0, d["max_pf"] - d["fpts"]), 2)
            eff_rows.append({
                "GM": gm_u,
                "Actual_PF": d["fpts"],
                "Optimal_Max_PF": d["max_pf"],
                "Bench_Pts_Left": pts_left,
                "Efficiency": f"{d['coach_eff_%']}%",
            })
        eff_df = pd.DataFrame(eff_rows)
        if not eff_df.empty:
            eff_df = eff_df.sort_values("Actual_PF", ascending=False)
        st.dataframe(eff_df, use_container_width=True, hide_index=True)
