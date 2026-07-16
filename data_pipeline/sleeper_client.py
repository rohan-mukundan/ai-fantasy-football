"""
sleeper_client.py
-----------------
Fetches NFL player data and weekly stats from the Sleeper API
for multiple seasons (2022, 2023, 2024).

Sleeper API docs: https://docs.sleeper.com/
- No API key required — it's completely free and public.

Usage:
    python data_pipeline/sleeper_client.py
"""

import requests
import pandas as pd
import json
import time
import os
from pathlib import Path
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL    = "https://api.sleeper.app/v1"
SEASON_TYPE = "regular"
TOTAL_WEEKS = 18

# All seasons to fetch — listed oldest to newest
SEASONS = ["2022", "2023", "2024"]

# Fantasy-relevant positions
# DEF = team defense (player_id is the team abbreviation e.g. "KC", "BAL")
# K   = kicker (standard numeric player_id)
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

# Folder paths
RAW_DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Helper ────────────────────────────────────────────────────────────────────

def get(endpoint: str) -> dict | list:
    """Makes a GET request to the Sleeper API with basic error handling."""
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        time.sleep(0.2)
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP error for {url}: {e}")
        return {}
    except requests.exceptions.ConnectionError:
        print(f"  ✗ Connection error — are you connected to the internet?")
        return {}
    except requests.exceptions.Timeout:
        print(f"  ✗ Request timed out for {url}")
        return {}

# ── Step 1: Fetch all NFL players (once, shared across seasons) ───────────────

def fetch_all_players() -> pd.DataFrame:
    """
    Downloads the full Sleeper player database.
    The player list is season-agnostic — player IDs are consistent across years,
    so we only need to fetch this once and reuse it for all seasons.
    """
    print("\n📥 Fetching NFL player list...")
    raw_path = RAW_DATA_DIR / "all_players.json"

    if raw_path.exists():
        print(f"  → Loading from disk (already fetched).")
        with open(raw_path) as f:
            players_raw = json.load(f)
    else:
        players_raw = get("/players/nfl")
        if not players_raw:
            print("  ✗ Failed to fetch player data.")
            return pd.DataFrame()
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w") as f:
            json.dump(players_raw, f)
        print(f"  ✓ Saved {len(players_raw):,} players.")

    rows = []
    for player_id, info in players_raw.items():
        if not isinstance(info, dict):
            continue
        position = info.get("position", "")
        team     = info.get("team", "FA") or "FA"

        # DEF entries have full_name=None in Sleeper — use "{TEAM} Defense" instead
        if position == "DEF":
            full_name = f"{team} Defense"
        else:
            full_name = info.get("full_name") or "Unknown"

        rows.append({
            "player_id":   player_id,
            "full_name":   full_name,
            "position":    position,
            "team":        team,
            "age":         info.get("age"),
            "years_exp":   info.get("years_exp"),
            "status":      info.get("status", ""),
        })

    df = pd.DataFrame(rows)
    df_fantasy = df[df["position"].isin(FANTASY_POSITIONS)].copy()
    df_fantasy = df_fantasy.sort_values("full_name").reset_index(drop=True)

    out_path = PROCESSED_DATA_DIR / "players.csv"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_fantasy.to_csv(out_path, index=False)
    print(f"  ✓ {len(df_fantasy):,} fantasy players saved to players.csv")
    return df_fantasy

# ── Step 2: Fetch weekly stats for one season ─────────────────────────────────

def fetch_weekly_stats_for_season(season: str) -> pd.DataFrame:
    """
    Fetches all 18 weeks of stats for a given season year.
    Raw files are saved per-season so we don't re-fetch on reruns.
    """
    season_raw_dir = RAW_DATA_DIR / season
    season_raw_dir.mkdir(parents=True, exist_ok=True)

    all_weeks = []
    for week in tqdm(range(1, TOTAL_WEEKS + 1), desc=f"  Weeks ({season})", leave=False):
        raw_path = season_raw_dir / f"stats_week_{week:02d}.csv"

        if raw_path.exists():
            df_week = pd.read_csv(raw_path)
        else:
            data = get(f"/stats/nfl/{SEASON_TYPE}/{season}/{week}")
            if not data:
                continue
            rows = []
            for player_id, stats in data.items():
                if isinstance(stats, dict):
                    row = {"player_id": player_id, "week": week}
                    row.update(stats)
                    rows.append(row)
            df_week = pd.DataFrame(rows)
            df_week.to_csv(raw_path, index=False)

        all_weeks.append(df_week)

    if not all_weeks:
        return pd.DataFrame()

    return pd.concat(all_weeks, ignore_index=True)

# ── Step 3: Build per-season summary ─────────────────────────────────────────

def build_season_summary(players_df: pd.DataFrame, stats_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Joins player info with weekly stats and collapses to one row per player
    with full-season totals, per-game averages, and early/late season splits.

    Early/late splits feed into the breakout score computed in data_processor.py:
      - Early season: weeks 1–9  (role/opportunity established)
      - Late season:  weeks 13–18 (finishing trajectory heading into next season)

    Saves to data/processed/season_{season}_summary.csv.
    """
    if players_df.empty or stats_df.empty:
        return pd.DataFrame()

    stat_cols = [
        "player_id", "week",
        "pts_ppr", "pts_half_ppr", "pts_std",
        "pass_yd", "pass_td", "pass_att", "pass_cmp", "pass_int",
        "rush_yd", "rush_td", "rush_att",
        "rec", "rec_yd", "rec_td", "rec_tgt",
        "gp",
    ]
    available_cols = [c for c in stat_cols if c in stats_df.columns]
    stats_filtered = stats_df[available_cols].copy()

    # Merge with player info
    merged = stats_filtered.merge(players_df[["player_id","full_name","position","team"]], on="player_id", how="left")
    merged = merged[merged["position"].isin(FANTASY_POSITIONS)]

    # ── Early / late season splits ────────────────────────────────────────────
    # We only count weeks where the player actually scored (pts_ppr > 0) so that
    # bye weeks and inactive weeks don't pull the average down artificially.
    EARLY_WEEKS = list(range(1, 10))   # weeks 1–9
    LATE_WEEKS  = list(range(14, 19))  # weeks 14–18

    early_active = merged[merged["week"].isin(EARLY_WEEKS) & (merged["pts_ppr"] > 0)]
    late_active  = merged[merged["week"].isin(LATE_WEEKS)  & (merged["pts_ppr"] > 0)]

    early_avg_ppr = early_active.groupby("player_id")["pts_ppr"].mean().round(1).rename("early_season_avg_ppr")
    late_avg_ppr  = late_active.groupby("player_id")["pts_ppr"].mean().round(1).rename("late_season_avg_ppr")

    # Target splits — only meaningful for pass-catchers but stored for all positions
    if "rec_tgt" in merged.columns:
        # Use all weeks (not just active) for targets so zero-target games count
        early_tgt = (
            merged[merged["week"].isin(EARLY_WEEKS)]
            .groupby("player_id")["rec_tgt"].mean().round(1)
            .rename("early_season_avg_targets")
        )
        late_tgt = (
            merged[merged["week"].isin(LATE_WEEKS)]
            .groupby("player_id")["rec_tgt"].mean().round(1)
            .rename("late_season_avg_targets")
        )
    else:
        early_tgt = pd.Series(dtype=float, name="early_season_avg_targets")
        late_tgt  = pd.Series(dtype=float, name="late_season_avg_targets")

    # ── Count weeks played and sum counting stats ─────────────────────────────
    weeks_played = (
        merged[merged["pts_ppr"] > 0]
        .groupby("player_id").size()
        .reset_index(name="weeks_played")
    )

    total_cols = ["pts_ppr","pts_half_ppr","pts_std",
                  "pass_yd","pass_td","pass_att","pass_cmp","pass_int",
                  "rush_yd","rush_td","rush_att","rec","rec_yd","rec_td","rec_tgt"]
    available_total_cols = [c for c in total_cols if c in merged.columns]

    totals = merged.groupby("player_id")[available_total_cols].sum().reset_index()
    totals.columns = ["player_id"] + [f"total_{c}" for c in available_total_cols]

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary = totals.merge(weeks_played, on="player_id", how="left")
    summary["weeks_played"] = summary["weeks_played"].fillna(0).astype(int)

    player_info = merged[["player_id","full_name","position","team"]].drop_duplicates("player_id")
    summary = player_info.merge(summary, on="player_id", how="right")

    # Per-game averages (divide by weeks actually played, not 18)
    played = summary["weeks_played"].replace(0, 1)
    summary["avg_pts_ppr"]      = (summary["total_pts_ppr"]      / played).round(1)
    summary["avg_pts_half_ppr"] = (summary["total_pts_half_ppr"] / played).round(1)
    summary["avg_pts_std"]      = (summary["total_pts_std"]       / played).round(1)

    # Attach early/late splits
    summary = summary.join(early_avg_ppr, on="player_id")
    summary = summary.join(late_avg_ppr,  on="player_id")
    summary = summary.join(early_tgt,     on="player_id")
    summary = summary.join(late_tgt,      on="player_id")

    # Surge score: positive means the player was finishing stronger than they started
    summary["surge_score"] = (summary["late_season_avg_ppr"] - summary["early_season_avg_ppr"]).round(1)

    # Target trend: positive means targets were increasing late in the season
    if "early_season_avg_targets" in summary.columns and "late_season_avg_targets" in summary.columns:
        summary["target_trend"] = (summary["late_season_avg_targets"] - summary["early_season_avg_targets"]).round(1)

    summary = summary.sort_values("total_pts_ppr", ascending=False).reset_index(drop=True)
    summary.insert(0, "season_rank", summary.index + 1)

    out_path = PROCESSED_DATA_DIR / f"season_{season}_summary.csv"
    summary.to_csv(out_path, index=False)
    return summary

# ── Programmatic entry point (called from app.py Season Setup) ────────────────

def run_pipeline(seasons: list[str], log=print) -> bool:
    """
    Fetches player data and builds per-season summaries for the given seasons.

    Parameters:
        seasons: List of season year strings, e.g. ["2022", "2023", "2024"]
        log:     Callable for progress messages — defaults to print, but the
                 Streamlit UI passes its own function to stream output live.

    Returns True on success, False if the player fetch failed.
    """
    log(f"Seasons to fetch: {', '.join(seasons)}")

    players_df = fetch_all_players()
    if players_df.empty:
        log("✗ Could not fetch player data. Check your internet connection.")
        return False
    log(f"✓ Player database loaded ({len(players_df):,} fantasy players)")

    for season in seasons:
        log(f"\nFetching {season} season stats (18 weeks)...")
        stats_df = fetch_weekly_stats_for_season(season)
        if stats_df.empty:
            log(f"  ✗ No stats returned for {season}.")
            continue
        summary = build_season_summary(players_df, stats_df, season)
        if not summary.empty:
            log(f"  ✓ {season}: {len(summary):,} players saved")

    log("\n✅ Season stats complete.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Fantasy Football — Sleeper Data Pipeline")
    print(f"  Seasons: {', '.join(SEASONS)}")
    print("=" * 60)
    run_pipeline(SEASONS)

if __name__ == "__main__":
    main()
