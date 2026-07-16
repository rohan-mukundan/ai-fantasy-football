"""
current_season.py
------------------
Incrementally fetches in-season weekly stats for the *current* fantasy
season from the Sleeper API and builds a running summary of how each
player has performed so far this season.

Unlike sleeper_client.py (which fetches a full prior season once during
Season Setup), this is designed to be called from the League View each
time the user steps forward to a new week. We assume the user moves
through weeks in order, so each call just needs to fetch whatever new
week(s) haven't been saved yet — already-fetched weeks are loaded from
disk (data/raw/{season}/stats_week_NN.csv), exactly like the historical
pipeline does.

Usage (from app.py):
    from data_pipeline.current_season import update_current_season_stats
    summary = update_current_season_stats("2026", through_week=4)
"""

import json
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

BASE_URL    = "https://api.sleeper.app/v1"
SEASON_TYPE = "regular"

RAW_DATA_DIR       = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def _get(endpoint: str) -> dict | list:
    url = f"{BASE_URL}{endpoint}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        time.sleep(0.2)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching {url}: {e}")
        return {}


def refresh_current_rosters(season: str, log=print) -> dict:
    """
    Re-fetches the Sleeper player database and saves a fresh snapshot of
    every player's current NFL team to:
        data/processed/rosters_{season}_current.json

    The saved file is a dict: { player_id: team_abbr, ... }

    Call this from the app whenever the user wants up-to-date team
    assignments — after a trade deadline, when a kicker is cut/signed,
    or at the start of any new week.  It hits the network only once per
    call; the file is then read locally by load_current_rosters().

    Returns the {player_id: team} dict, or {} on failure.
    """
    log("  📡 Fetching latest NFL roster data from Sleeper...")
    data = _get("/players/nfl")
    if not data:
        log("  ✗ Could not reach Sleeper — roster refresh skipped.")
        return {}

    rosters = {}
    for player_id, info in data.items():
        if not isinstance(info, dict):
            continue
        team = info.get("team") or "FA"
        rosters[player_id] = team

    out = {
        "fetched_at": datetime.now().isoformat(),
        "season":     season,
        "rosters":    rosters,
    }
    path = PROCESSED_DATA_DIR / f"rosters_{season}_current.json"
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)

    log(f"  ✓ Roster snapshot saved ({len(rosters):,} players, {sum(1 for t in rosters.values() if t != 'FA'):,} on active rosters)")
    return rosters


def load_current_rosters(season: str) -> dict:
    """
    Loads the most recent roster snapshot saved by refresh_current_rosters().
    Returns a {player_id: team_abbr} dict, or {} if no snapshot exists yet.
    Also returns the fetch timestamp via the 'fetched_at' key in the raw file,
    but callers that just need team lookups can use the returned dict directly.
    """
    path = PROCESSED_DATA_DIR / f"rosters_{season}_current.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
        return raw.get("rosters", {})
    except (json.JSONDecodeError, OSError):
        return {}


def get_roster_refresh_time(season: str) -> str | None:
    """Returns a human-readable string of when rosters were last refreshed, or None."""
    path = PROCESSED_DATA_DIR / f"rosters_{season}_current.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            raw = json.load(f)
        ts = raw.get("fetched_at")
        if ts:
            dt = datetime.fromisoformat(ts)
            return dt.strftime("%b %d at %I:%M %p")
    except Exception:
        pass
    return None


def _fetch_week(season: str, week: int) -> pd.DataFrame:
    """Fetches (or loads from disk cache) stats for one week of the season."""
    season_raw_dir = RAW_DATA_DIR / season
    season_raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = season_raw_dir / f"stats_week_{week:02d}.csv"

    if raw_path.exists():
        return pd.read_csv(raw_path)

    data = _get(f"/stats/nfl/{SEASON_TYPE}/{season}/{week}")
    if not data:
        return pd.DataFrame()

    rows = []
    for player_id, stats in data.items():
        if isinstance(stats, dict):
            row = {"player_id": player_id, "week": week}
            row.update(stats)
            rows.append(row)

    df_week = pd.DataFrame(rows)
    if not df_week.empty:
        df_week.to_csv(raw_path, index=False)
    return df_week


def clear_current_season_data(season: str, log=print) -> int:
    """
    Deletes all saved in-season data for `season` so it can be re-fetched
    from scratch — useful for testing the week-by-week lineup logic from
    Week 1 again.

    Removes:
      - data/raw/{season}/stats_week_*.csv  (per-week raw stat dumps)
      - data/processed/season_{season}_current_summary.csv  (rolled-up summary)

    Returns the number of files deleted.
    """
    deleted = 0

    season_raw_dir = RAW_DATA_DIR / season
    if season_raw_dir.exists():
        for f in season_raw_dir.glob("stats_week_*.csv"):
            f.unlink()
            deleted += 1

    summary_path = PROCESSED_DATA_DIR / f"season_{season}_current_summary.csv"
    if summary_path.exists():
        summary_path.unlink()
        deleted += 1

    log(f"  ✓ Cleared in-season data for {season} ({deleted} file(s) deleted)")
    return deleted


def update_current_season_stats(season: str, through_week: int, log=print) -> pd.DataFrame:
    """
    Ensures we have weekly stats for `season` through `through_week`
    (fetching any not-yet-saved weeks from Sleeper), then rebuilds the
    running current-season summary and saves it to
    data/processed/season_{season}_current_summary.csv.

    Returns a DataFrame with one row per player who has played at least
    one game this season: full_name, position, team, total_pts_ppr_current,
    weeks_played_current, avg_pts_ppr_current.

    If through_week < 1 (i.e. we're looking at Week 1, before any games
    have been played), returns an empty DataFrame.
    """
    if through_week < 1:
        return pd.DataFrame()

    players_path = PROCESSED_DATA_DIR / "players.csv"
    if not players_path.exists():
        log("  ⚠ players.csv not found — run Season Setup first.")
        return pd.DataFrame()
    players_df = pd.read_csv(players_path)

    # If a live roster snapshot exists, overlay the current team assignments.
    # This accounts for mid-season trades, cuts, and signings that happened
    # after Season Setup was run.
    live_rosters = load_current_rosters(season)
    if live_rosters:
        players_df["team"] = players_df["player_id"].astype(str).map(live_rosters).fillna(players_df["team"])
        log(f"  ✓ Applied live roster data ({len(live_rosters):,} team assignments)")

    all_weeks = []
    for week in range(1, through_week + 1):
        df_week = _fetch_week(season, week)
        if not df_week.empty:
            all_weeks.append(df_week)
        else:
            log(f"  ⚠ No stats available yet for {season} week {week}.")

    if not all_weeks:
        return pd.DataFrame()

    stats_df = pd.concat(all_weeks, ignore_index=True)

    stat_cols = ["player_id", "week", "pts_ppr"]
    available_cols = [c for c in stat_cols if c in stats_df.columns]
    stats_filtered = stats_df[available_cols].copy()

    merged = stats_filtered.merge(
        players_df[["player_id", "full_name", "position", "team"]],
        on="player_id", how="left"
    )
    merged = merged[merged["position"].isin(FANTASY_POSITIONS)]

    weeks_played = (
        merged[merged["pts_ppr"] > 0]
        .groupby("player_id").size()
        .reset_index(name="weeks_played_current")
    )

    totals = merged.groupby("player_id")[["pts_ppr"]].sum().reset_index()
    totals = totals.rename(columns={"pts_ppr": "total_pts_ppr_current"})

    player_info = merged[["player_id", "full_name", "position", "team"]].drop_duplicates("player_id")
    summary = player_info.merge(totals, on="player_id", how="right")
    summary = summary.merge(weeks_played, on="player_id", how="left")
    summary["weeks_played_current"] = summary["weeks_played_current"].fillna(0).astype(int)

    played = summary["weeks_played_current"].replace(0, 1)
    summary["avg_pts_ppr_current"] = (summary["total_pts_ppr_current"] / played).round(1)

    out_path = PROCESSED_DATA_DIR / f"season_{season}_current_summary.csv"
    summary.to_csv(out_path, index=False)
    log(f"  ✓ Current-season stats updated through week {through_week} "
        f"({len(summary):,} players with at least one game)")

    return summary
