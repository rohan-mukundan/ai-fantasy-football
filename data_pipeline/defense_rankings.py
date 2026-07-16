"""
defense_rankings.py
-------------------
Computes how many PPR fantasy points each NFL defense has allowed per
position (QB, RB, WR, TE) across the current season, week by week.

Data sources:
  - Sleeper weekly stats (already cached in data/raw/{season}/stats_week_XX.csv
    by current_season.py — defense_rankings.py never re-fetches those)
  - ESPN scoreboard API for weekly team matchups (same URL as schedule_fetcher.py)
  - data/processed/players.csv for player → NFL team mapping

Two main public functions:

  refresh_defense_rankings(season, through_week, log)
      Rebuilds data/processed/defense_rankings_{season}.csv from disk-cached
      weekly stats + ESPN schedule data.  Call this after updating season stats.

  get_def_matchup(player_team, position, upcoming_week, season)
      Returns matchup context for a player's next game:
        {
          "opponent":      "BUF",    # team they're facing
          "rank":          5,        # season-long rank (1 = fewest pts allowed = toughest)
          "recent_rank":   3,        # last-4-weeks rank
          "matchup_label": "tough",  # "easy", "neutral", or "tough"
          "avg_allowed":   20.4,     # season-avg PPR pts this defense gave up to this position
          "recent_avg":    22.1,     # last-4-week average
        }
      Returns None if data isn't available or the team has a bye.
"""

import json
import requests
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.parent
RAW_DATA_DIR   = BASE_DIR / "data" / "raw"
PROCESSED_DIR  = BASE_DIR / "data" / "processed"

# ── Constants ─────────────────────────────────────────────────────────────────

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# ESPN uses slightly different abbreviations than our player data
ESPN_TEAM_MAP = {
    "WSH": "WAS",
}

# Positions we rank defenses against (K and DEF excluded)
RANKED_POSITIONS = {"QB", "RB", "WR", "TE"}

# PPR scoring weights — must match the rest of the app
PPR_WEIGHTS = {
    "rec":      1.0,
    "rec_yd":   0.1,
    "rec_td":   6.0,
    "rush_yd":  0.1,
    "rush_td":  6.0,
    "pass_yd":  0.04,
    "pass_td":  4.0,
    "pass_int": -2.0,
}

# Matchup label thresholds (based on rank percentile within each position group)
# rank 1 = fewest pts allowed = hardest matchup
LABEL_EASY_THRESHOLD    = 0.67   # bottom 33% (easy to score on)
LABEL_NEUTRAL_THRESHOLD = 0.33   # top 33% to middle 33%
# above LABEL_EASY_THRESHOLD → "easy", below LABEL_NEUTRAL_THRESHOLD → "tough"

# ── Prior-season blending constants ───────────────────────────────────────────
# Early in the season (weeks 1-3), current stats are a tiny sample and rankings
# can be heavily skewed by one game.  We blend in the prior year's numbers to
# anchor the rankings until enough current-season data has accumulated.
#
# Weight formula:  w = through_week / (through_week + PRIOR_STRENGTH)
#   PRIOR_STRENGTH = 6  means the prior is treated as equivalent to 6 weeks
#   of data — so at week 3 the prior gets 67% weight, at week 9 it gets 40%.
#
# PRIOR_BLEND_CUTOFF = week after which we stop blending (enough current data).

PRIOR_STRENGTH       = 6    # weeks of current data the prior is worth
PRIOR_BLEND_CUTOFF   = 13   # weeks ≥ this → use current data only


# ── Abbreviation normalizer ───────────────────────────────────────────────────

def _normalize(abbr: str) -> str:
    """Converts ESPN team abbreviations to the ones used in our player data."""
    return ESPN_TEAM_MAP.get(abbr, abbr)


# ── Prior-season helpers ───────────────────────────────────────────────────────

def _load_prior_team_map(prior_season: str) -> dict[str, tuple[str, str]]:
    """
    Returns {player_id: (position, team)} using the prior season's summary CSV
    for team assignments, which is more accurate than current players.csv for
    players who changed teams since then.  Falls back to players.csv if the
    prior summary doesn't exist.
    """
    summary_path = PROCESSED_DIR / f"season_{prior_season}_summary.csv"
    if summary_path.exists():
        try:
            df = pd.read_csv(
                summary_path,
                usecols=["player_id", "position", "team"],
                dtype={"player_id": str},
            )
            df = df[df["position"].isin(RANKED_POSITIONS) & df["team"].notna() & (df["team"] != "FA")]
            return {row["player_id"]: (row["position"], row["team"]) for _, row in df.iterrows()}
        except Exception:
            pass
    # Fall back to current players.csv
    players_df = _load_players()
    return {
        row["player_id"]: (row["position"], row["team"])
        for _, row in players_df.iterrows()
        if row["position"] in RANKED_POSITIONS and row["team"] and row["team"] != "FA"
    }


def _compute_prior_offense_avgs(prior_season: str) -> dict[str, float]:
    """
    Computes full-season offensive output averages from the prior year.
    Returns {team_abbr: avg_ppr_pts_per_game} across all 18 weeks.
    Used to anchor early-season offense rankings before current data is reliable.
    """
    player_map = _load_prior_team_map(prior_season)
    if not player_map:
        return {}

    team_week_pts: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for week in range(1, 19):
        stats_df = _load_week_stats(prior_season, week)
        if stats_df.empty:
            continue
        for _, row in stats_df.iterrows():
            pid  = str(row.get("player_id", ""))
            info = player_map.get(pid)
            if not info:
                continue
            position, team = info
            pts = _compute_ppr(row)
            if pts > 0:
                team_week_pts[team][week] += pts

    return {
        team: round(sum(wk.values()) / len(wk), 2)
        for team, wk in team_week_pts.items()
        if wk
    }


def _compute_prior_defense_avgs(prior_season: str) -> dict[tuple[str, str], float]:
    """
    Computes full-season defensive averages from the prior year.
    Returns {(team_abbr, position): avg_ppr_pts_allowed_per_game} across all 18 weeks.
    Requires ESPN matchup data for the prior year (fetched on demand, cached to disk).
    Used to anchor early-season defense rankings before current data is reliable.
    """
    player_map = _load_prior_team_map(prior_season)
    if not player_map:
        return {}

    # defense_stats[opp_team][position][week] = [pts, ...]
    defense_stats: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for week in range(1, 19):
        stats_df = _load_week_stats(prior_season, week)
        if stats_df.empty:
            continue
        matchups = _fetch_week_matchups(prior_season, week)
        if not matchups:
            continue

        for _, row in stats_df.iterrows():
            pid      = str(row.get("player_id", ""))
            info     = player_map.get(pid)
            if not info:
                continue
            position, team = info
            opponent = matchups.get(team)
            if not opponent:
                continue
            pts = _compute_ppr(row)
            if pts > 0:
                defense_stats[opponent][position][week].append(pts)

    result: dict[tuple[str, str], float] = {}
    for team, pos_data in defense_stats.items():
        for pos, week_data in pos_data.items():
            total = sum(sum(pts) for pts in week_data.values())
            games = sum(len(pts) for pts in week_data.values())
            if games > 0:
                result[(team, pos)] = round(total / games, 2)
    return result


# ── Schedule / matchup fetching ───────────────────────────────────────────────

def _fetch_week_matchups(season: str, week: int) -> dict[str, str]:
    """
    Fetches (and disk-caches) the ESPN schedule for one week.
    Returns a dict of {team_abbr: opponent_abbr} for every team that played.

    For example, if KC played BUF and CIN played BAL:
      {"KC": "BUF", "BUF": "KC", "CIN": "BAL", "BAL": "CIN"}

    Returns {} on API failure or if no games found.
    """
    cache_dir = PROCESSED_DIR / "matchup_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"matchups_{season}_week{week:02d}.json"

    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            # A real NFL regular-season week has 16 games = 32 entries (each team once).
            # If the cached file has fewer than 10 entries it was either a test artifact
            # or an incomplete fetch — discard it and re-fetch.
            if len(cached) >= 10:
                return cached
        except Exception:
            pass  # corrupt cache — re-fetch below

    try:
        url = f"{SCOREBOARD_URL}?week={week}&seasontype=2&year={season}"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    matchups: dict[str, str] = {}
    for event in data.get("events", []):
        for competition in event.get("competitions", []):
            competitors = competition.get("competitors", [])
            if len(competitors) == 2:
                a = _normalize(competitors[0].get("team", {}).get("abbreviation", ""))
                b = _normalize(competitors[1].get("team", {}).get("abbreviation", ""))
                if a and b:
                    matchups[a] = b
                    matchups[b] = a

    with open(cache_path, "w") as f:
        json.dump(matchups, f)

    return matchups


def get_upcoming_opponent(player_team: str, week: int, season: str) -> str | None:
    """
    Returns the opponent abbreviation for `player_team` in the given week,
    or None if the team is on a bye or data is unavailable.
    """
    if not player_team or player_team == "FA":
        return None
    matchups = _fetch_week_matchups(season, week)
    return matchups.get(player_team)


# ── Weekly stats loader ───────────────────────────────────────────────────────

def _load_week_stats(season: str, week: int) -> pd.DataFrame:
    """
    Loads the disk-cached Sleeper weekly stats (written by current_season.py).
    Returns an empty DataFrame if the file isn't there yet.
    """
    path = RAW_DATA_DIR / season / f"stats_week_{week:02d}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"player_id": str})
    except Exception:
        return pd.DataFrame()


def _load_players() -> pd.DataFrame:
    """
    Loads player_id → (position, team) from data/processed/players.csv.
    """
    path = PROCESSED_DIR / "players.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, usecols=["player_id", "position", "team"])
        df["player_id"] = df["player_id"].astype(str)
        return df[df["position"].isin(RANKED_POSITIONS)].copy()
    except Exception:
        return pd.DataFrame()


# ── PPR scorer ────────────────────────────────────────────────────────────────

def _compute_ppr(row: pd.Series) -> float:
    """
    Returns PPR score from a stat row.
    Prefers the pre-computed pts_ppr column if it exists and is positive;
    otherwise reconstructs from raw stat columns.
    """
    val = row.get("pts_ppr")
    if val is not None and pd.notna(val) and float(val) > 0:
        return float(val)
    total = 0.0
    for field, weight in PPR_WEIGHTS.items():
        v = row.get(field, 0)
        if pd.notna(v):
            total += float(v) * weight
    return max(0.0, total)


# ── Core ranking computation ──────────────────────────────────────────────────

def refresh_defense_rankings(
    season: str,
    through_week: int,
    prior_season: str | None = None,
    log=print,
) -> pd.DataFrame:
    """
    Builds (or refreshes) defensive rankings for `season` through `through_week`.

    For each week 1..through_week:
      1. Load cached player stats from disk (from current_season.py's cache)
      2. Fetch matchup data from ESPN (who played whom)
      3. For each skill-position player with pts > 0:
           player team → opponent this week → tally PPR pts allowed by that defense

    When `prior_season` is given and `through_week` < PRIOR_BLEND_CUTOFF, the
    current-season averages are blended with the prior year's full-season averages
    to prevent early-season sample-size skew:
        blended = current × w  +  prior × (1 - w)
        w = through_week / (through_week + PRIOR_STRENGTH)

    The result is saved to data/processed/defense_rankings_{season}.csv and returned.

    Output columns:
      team, position,
      total_pts_allowed, games_counted, avg_pts_allowed,    ← season-long (blended)
      recent_4wk_pts, recent_4wk_games, recent_4wk_avg,     ← last 4 weeks (blended)
      rank, recent_rank,                                     ← 1=toughest, 32=easiest
      matchup_label                                          ← "tough" / "neutral" / "easy"
    """
    if through_week < 1:
        log("  ℹ No completed weeks yet — skipping defense rankings.")
        return pd.DataFrame()

    players_df = _load_players()
    if players_df.empty:
        log("  ⚠ players.csv not found — run Season Setup first.")
        return pd.DataFrame()

    # Build fast lookup: player_id → (position, team)
    player_map: dict[str, tuple[str, str]] = {
        row["player_id"]: (row["position"], row["team"])
        for _, row in players_df.iterrows()
    }

    # defense_stats[defense_team][position][week] = [pts, pts, ...]
    # Each element is one player's total PPR pts scored against that defense that week
    defense_stats: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    weeks_processed = 0
    for week in range(1, through_week + 1):
        stats_df = _load_week_stats(season, week)
        if stats_df.empty:
            continue

        matchups = _fetch_week_matchups(season, week)
        if not matchups:
            log(f"  ⚠ Week {week}: no matchup data found (bye week or API issue).")
            continue

        for _, row in stats_df.iterrows():
            pid = str(row.get("player_id", ""))
            info = player_map.get(pid)
            if not info:
                continue
            position, team = info
            if not team or team == "FA":
                continue
            opponent = matchups.get(team)
            if not opponent:
                continue  # this player's team was on bye or not in schedule

            pts = _compute_ppr(row)
            if pts <= 0:
                continue  # didn't score — exclude so bye/DNP don't skew defense down

            defense_stats[opponent][position][week].append(pts)

        weeks_processed += 1

    if weeks_processed == 0:
        log("  ⚠ No weekly stats cached yet — run the weekly data update first.")
        return pd.DataFrame()

    # ── Build summary rows ────────────────────────────────────────────────────
    recent_weeks = set(range(max(1, through_week - 3), through_week + 1))

    rows = []
    for team in sorted(defense_stats.keys()):
        for position in sorted(RANKED_POSITIONS):
            week_data = defense_stats[team][position]

            total_pts   = sum(sum(pts) for pts in week_data.values())
            games       = sum(len(pts) for pts in week_data.values())
            avg_allowed = round(total_pts / games, 2) if games > 0 else 0.0

            recent_pts   = sum(sum(pts) for wk, pts in week_data.items() if wk in recent_weeks)
            recent_games = sum(len(pts) for wk, pts in week_data.items() if wk in recent_weeks)
            recent_avg   = round(recent_pts / recent_games, 2) if recent_games > 0 else 0.0

            rows.append({
                "team":              team,
                "position":          position,
                "total_pts_allowed": round(total_pts, 1),
                "games_counted":     games,
                "avg_pts_allowed":   avg_allowed,
                "recent_4wk_pts":    round(recent_pts, 1),
                "recent_4wk_games":  recent_games,
                "recent_4wk_avg":    recent_avg,
            })

    if not rows:
        log("  ⚠ No defensive data computed.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Blend with prior season to stabilize early-season rankings ────────────
    if prior_season and through_week < PRIOR_BLEND_CUTOFF:
        w = through_week / (through_week + PRIOR_STRENGTH)
        log(f"  ↳ Blending with {prior_season} prior (current weight {w:.0%}, prior {1-w:.0%})")
        prior_def = _compute_prior_defense_avgs(prior_season)

        def _blend_def_row(row: pd.Series) -> pd.Series:
            prior_avg = prior_def.get((row["team"], row["position"]))
            if prior_avg is not None:
                row["avg_pts_allowed"]  = round(row["avg_pts_allowed"] * w + prior_avg * (1 - w), 2)
                row["recent_4wk_avg"]   = round(row["recent_4wk_avg"]  * w + prior_avg * (1 - w), 2)
            return row

        df = df.apply(_blend_def_row, axis=1)

    # ── Rank within each position group ──────────────────────────────────────
    # Rank 1 = fewest pts allowed per game played (toughest matchup)
    df["rank"] = (
        df.groupby("position")["avg_pts_allowed"]
        .rank(method="min", ascending=True)
        .astype(int)
    )
    df["recent_rank"] = (
        df.groupby("position")["recent_4wk_avg"]
        .rank(method="min", ascending=True)
        .astype(int)
    )

    # ── Matchup label ─────────────────────────────────────────────────────────
    # Number of teams with data per position (usually close to 32)
    n_by_pos = df.groupby("position")["team"].nunique()

    def _label(row: pd.Series) -> str:
        n    = n_by_pos.get(row["position"], 32)
        pct  = row["rank"] / n          # 0 → hardest, 1 → easiest
        if pct > LABEL_EASY_THRESHOLD:
            return "easy"
        elif pct > LABEL_NEUTRAL_THRESHOLD:
            return "neutral"
        else:
            return "tough"

    df["matchup_label"] = df.apply(_label, axis=1)

    # ── Save ──────────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"defense_rankings_{season}.csv"
    df.to_csv(out_path, index=False)

    log(
        f"  ✓ Defense rankings built through Week {through_week} "
        f"({weeks_processed} weeks, {len(df)} team/position rows) "
        f"→ defense_rankings_{season}.csv"
    )
    return df


# ── Public lookup ─────────────────────────────────────────────────────────────

def load_defense_rankings(season: str) -> pd.DataFrame:
    """
    Loads the saved defensive rankings CSV.
    Returns an empty DataFrame if the file doesn't exist yet.
    """
    path = PROCESSED_DIR / f"defense_rankings_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_def_matchup(
    player_team: str,
    position: str,
    upcoming_week: int,
    season: str,
    rankings_df: pd.DataFrame | None = None,
) -> dict | None:
    """
    Returns defensive matchup context for a player in the upcoming week.

    Parameters
    ----------
    player_team   : The player's current NFL team abbreviation (e.g. "KC")
    position      : The player's fantasy position ("QB", "RB", "WR", "TE")
    upcoming_week : The week number to look up (the one they're about to play)
    season        : Season year string (e.g. "2025")
    rankings_df   : Optional pre-loaded DataFrame from load_defense_rankings()
                    to avoid repeated disk reads in loops.

    Returns
    -------
    dict or None
      {
        "opponent":      "BUF",    # team they're facing
        "rank":          5,        # season-long rank (1=fewest pts allowed=toughest)
        "recent_rank":   3,        # last-4-weeks rank
        "matchup_label": "tough",  # "easy", "neutral", or "tough"
        "avg_allowed":   20.4,     # season avg PPR pts this defense gave up at this position
        "recent_avg":    22.1,     # last-4-week average
      }

    None is returned when:
      - position is not QB/RB/WR/TE
      - the player's team is on a bye that week
      - defensive rankings haven't been built yet
      - the opponent hasn't faced enough data to appear in the rankings
    """
    if position not in RANKED_POSITIONS:
        return None

    opponent = get_upcoming_opponent(player_team, upcoming_week, season)
    if not opponent:
        return None

    if rankings_df is None:
        rankings_df = load_defense_rankings(season)
    if rankings_df.empty:
        return None

    row = rankings_df[
        (rankings_df["team"] == opponent) & (rankings_df["position"] == position)
    ]
    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "opponent":      opponent,
        "rank":          int(r["rank"]),
        "recent_rank":   int(r["recent_rank"]),
        "matchup_label": str(r["matchup_label"]),
        "avg_allowed":   float(r["avg_pts_allowed"]),
        "recent_avg":    float(r["recent_4wk_avg"]),
    }


# ── Convenience formatter ─────────────────────────────────────────────────────

MATCHUP_EMOJI = {
    "easy":    "🟢",
    "neutral": "🟡",
    "tough":   "🔴",
}

def format_matchup_label(matchup: dict | None) -> str:
    """
    Returns a short display string like "🟢 vs BUF (rank 28)" or "—" if no data.
    Uses the recent rank (last 4 weeks) since that's most predictive for
    start/sit decisions.
    """
    if not matchup:
        return "—"
    emoji = MATCHUP_EMOJI.get(matchup["matchup_label"], "🟡")
    opp   = matchup["opponent"]
    rank  = matchup["recent_rank"]
    return f"{emoji} vs {opp} (#{rank})"


# ── Offense rankings ─────────────────────────────────────────────────────────
#
# For DEF streaming decisions we need the inverse view: how strong is each
# NFL team's offense?  Weak offenses = easy DEF matchups.
#
# We compute this from the same Sleeper weekly stats already on disk:
# sum all skill-position (QB+RB+WR+TE) PPR points scored BY each team per game.
# High output → dangerous offense → bad week to stream that team's opponent DEF.
# Low output  → weak offense      → great week to stream the defense facing them.
#
# Rank 1 = weakest offense (fewest pts scored) = BEST DEF streaming target.
#
# For the first 2 weeks of the season, not enough game data exists to compute
# reliable rankings.  Instead we load expert preseason rankings from a JSON file:
#   data/processed/preseason_offense_rankings_{season}.json
# Those use The Ringer's ordering (1=strongest offense like BUF) which we invert
# so our internal rank 1 = weakest offense (best DEF streaming target).


def _build_preseason_offense_df(season: str, log=print) -> pd.DataFrame:
    """
    Loads expert preseason offense rankings from
      data/processed/preseason_offense_rankings_{season}.json
    and returns a DataFrame compatible with the offense_rankings_df schema:
      team, total_pts_scored, games_played, avg_pts_scored,
      recent_4wk_pts, recent_4wk_games, recent_4wk_avg,
      rank, recent_rank, offense_label

    The source JSON uses rank 1 = best offense (e.g. Buffalo).
    We invert: internal_rank = 33 - source_rank, so rank 1 = weakest offense
    (fewest expected pts) = best DEF streaming target.

    Returns an empty DataFrame if the file is missing.
    """
    path = PROCESSED_DIR / f"preseason_offense_rankings_{season}.json"
    if not path.exists():
        log(f"  ⚠ No preseason offense rankings file found at {path.name}")
        return pd.DataFrame()

    try:
        with open(path) as f:
            data = json.load(f)
        ringer_ranks: dict[str, int] = data.get("rankings", {})
    except Exception as e:
        log(f"  ⚠ Failed to load preseason offense rankings: {e}")
        return pd.DataFrame()

    if not ringer_ranks:
        return pd.DataFrame()

    n = len(ringer_ranks)
    rows = []
    for team, ringer_rank in ringer_ranks.items():
        internal_rank = n + 1 - ringer_rank  # invert: rank 1 = weakest offense

        pct = internal_rank / n
        if pct <= 0.33:
            label = "weak"      # 🟢 easy DEF matchup
        elif pct <= 0.67:
            label = "average"   # 🟡 neutral
        else:
            label = "strong"    # 🔴 tough DEF matchup

        rows.append({
            "team":             team,
            "total_pts_scored": 0.0,
            "games_played":     0,
            "avg_pts_scored":   0.0,
            "recent_4wk_pts":   0.0,
            "recent_4wk_games": 0,
            "recent_4wk_avg":   0.0,
            "rank":             internal_rank,
            "recent_rank":      internal_rank,
            "offense_label":    label,
        })

    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def refresh_offense_rankings(
    season: str,
    through_week: int,
    prior_season: str | None = None,
    log=print,
) -> pd.DataFrame:
    """
    Builds (or refreshes) offensive output rankings for `season` through
    `through_week`.  No matchup data needed — we just sum the PPR points
    scored by each team's skill-position players each week.

    When `prior_season` is given and `through_week` < PRIOR_BLEND_CUTOFF, blends
    current-season averages with the prior year's full-season averages to prevent
    early-season sample-size skew:
        blended = current × w  +  prior × (1 - w)
        w = through_week / (through_week + PRIOR_STRENGTH)

    Output columns:
      team, total_pts_scored, games_played, avg_pts_scored,
      recent_4wk_pts, recent_4wk_games, recent_4wk_avg,
      rank, recent_rank, offense_label

    rank 1 = fewest pts scored (weakest offense = easiest DEF matchup)
    rank 32 = most pts scored (strongest offense = toughest DEF matchup)

    Saves to data/processed/offense_rankings_{season}.csv and returns the df.
    """
    if through_week < 1:
        return pd.DataFrame()

    # ── Weeks 1-3: not enough game data — use expert preseason rankings ────────
    PRESEASON_WEEKS = 3
    if through_week <= PRESEASON_WEEKS:
        log(
            f"  ↳ Week {through_week} ≤ {PRESEASON_WEEKS}: using expert preseason "
            f"offense rankings instead of computed stats"
        )
        df = _build_preseason_offense_df(season, log=log)
        if not df.empty:
            PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            out_path = PROCESSED_DIR / f"offense_rankings_{season}.csv"
            df.to_csv(out_path, index=False)
            log(f"  ✓ Preseason offense rankings saved → offense_rankings_{season}.csv")
            return df
        # Fall through to computed rankings if preseason file is missing

    players_df = _load_players()
    if players_df.empty:
        log("  ⚠ players.csv not found — run Season Setup first.")
        return pd.DataFrame()

    # player_id → (position, team)
    player_map: dict[str, tuple[str, str]] = {
        row["player_id"]: (row["position"], row["team"])
        for _, row in players_df.iterrows()
    }

    # team_week_pts[team][week] = total PPR pts scored by that team's offense
    team_week_pts: dict[str, dict[int, float]] = defaultdict(lambda: defaultdict(float))

    weeks_processed = 0
    for week in range(1, through_week + 1):
        stats_df = _load_week_stats(season, week)
        if stats_df.empty:
            continue

        for _, row in stats_df.iterrows():
            pid = str(row.get("player_id", ""))
            info = player_map.get(pid)
            if not info:
                continue
            position, team = info
            if position not in RANKED_POSITIONS or not team or team == "FA":
                continue
            pts = _compute_ppr(row)
            if pts > 0:
                team_week_pts[team][week] += pts

        weeks_processed += 1

    if weeks_processed == 0:
        log("  ⚠ No weekly stats cached yet.")
        return pd.DataFrame()

    recent_weeks = set(range(max(1, through_week - 3), through_week + 1))

    rows = []
    for team in sorted(team_week_pts.keys()):
        week_data = team_week_pts[team]

        total   = sum(week_data.values())
        games   = len(week_data)
        avg     = round(total / games, 2) if games > 0 else 0.0

        recent_total = sum(v for wk, v in week_data.items() if wk in recent_weeks)
        recent_games = sum(1 for wk in week_data if wk in recent_weeks)
        recent_avg   = round(recent_total / recent_games, 2) if recent_games > 0 else 0.0

        rows.append({
            "team":            team,
            "total_pts_scored": round(total, 1),
            "games_played":    games,
            "avg_pts_scored":  avg,
            "recent_4wk_pts":  round(recent_total, 1),
            "recent_4wk_games": recent_games,
            "recent_4wk_avg":  recent_avg,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Blend with prior season to stabilize early-season rankings ────────────
    if prior_season and through_week < PRIOR_BLEND_CUTOFF:
        w = through_week / (through_week + PRIOR_STRENGTH)
        log(f"  ↳ Blending offense with {prior_season} prior (current {w:.0%}, prior {1-w:.0%})")
        prior_off = _compute_prior_offense_avgs(prior_season)

        prior_series = df["team"].map(prior_off)
        has_prior    = prior_series.notna()

        df.loc[has_prior, "avg_pts_scored"] = (
            df.loc[has_prior, "avg_pts_scored"] * w
            + prior_series[has_prior] * (1 - w)
        ).round(2)
        df.loc[has_prior, "recent_4wk_avg"] = (
            df.loc[has_prior, "recent_4wk_avg"] * w
            + prior_series[has_prior] * (1 - w)
        ).round(2)

    # Rank 1 = weakest offense (lowest blended avg pts scored)
    df["rank"]        = df["avg_pts_scored"].rank(method="min", ascending=True).astype(int)
    df["recent_rank"] = df["recent_4wk_avg"].rank(method="min", ascending=True).astype(int)

    n = len(df)
    def _label(row: pd.Series) -> str:
        pct = row["recent_rank"] / n
        if pct <= 0.33:
            return "weak"      # 🟢 easy DEF matchup
        elif pct <= 0.67:
            return "average"   # 🟡 neutral
        else:
            return "strong"    # 🔴 tough DEF matchup

    df["offense_label"] = df.apply(_label, axis=1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"offense_rankings_{season}.csv"
    df.to_csv(out_path, index=False)

    prior_note = f" (blended with {prior_season})" if prior_season and through_week < PRIOR_BLEND_CUTOFF else ""
    log(
        f"  ✓ Offense rankings built through Week {through_week}{prior_note} "
        f"({weeks_processed} weeks, {len(df)} teams) → offense_rankings_{season}.csv"
    )
    return df


def load_offense_rankings(season: str) -> pd.DataFrame:
    """Loads the saved offensive rankings CSV. Returns empty df if not built yet."""
    path = PROCESSED_DIR / f"offense_rankings_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_opponent_offense(
    def_team: str,
    upcoming_week: int,
    season: str,
    offense_rankings_df: pd.DataFrame | None = None,
) -> dict | None:
    """
    Returns the offensive strength of the upcoming opponent for `def_team`.

    Parameters
    ----------
    def_team          : NFL team abbreviation (e.g. "DEN")
    upcoming_week     : The week the DEF is about to play
    season            : Season year string
    offense_rankings_df : Pre-loaded offense rankings (pass to avoid repeated reads)

    Returns
    -------
    dict or None:
      {
        "opponent":        "TEN",
        "rank":            3,        # 1=weakest offense (best DEF matchup)
        "recent_rank":     2,
        "offense_label":   "weak",   # "weak" / "average" / "strong"
        "avg_pts_scored":  28.5,
        "recent_avg":      26.1,
      }
    """
    opponent = get_upcoming_opponent(def_team, upcoming_week, season)
    if not opponent:
        return None

    if offense_rankings_df is None:
        offense_rankings_df = load_offense_rankings(season)
    if offense_rankings_df.empty:
        return None

    row = offense_rankings_df[offense_rankings_df["team"] == opponent]
    if row.empty:
        return None

    r = row.iloc[0]
    return {
        "opponent":       opponent,
        "rank":           int(r["rank"]),
        "recent_rank":    int(r["recent_rank"]),
        "offense_label":  str(r["offense_label"]),
        "avg_pts_scored": float(r["avg_pts_scored"]),
        "recent_avg":     float(r["recent_4wk_avg"]),
    }


# ── DEF quality (own fantasy output) ─────────────────────────────────────────
#
# Separate from matchup: how good is each DEF unit itself?
# A great matchup on paper still loses value if the DEF is terrible.
# We measure this by the DEF player's actual fantasy pts_ppr each week.
#
# def_quality_rank 1 = worst DEF (fewest avg pts scored)
# def_quality_rank 32 = best DEF (most avg pts scored)

def refresh_def_quality_rankings(
    season: str,
    through_week: int,
    log=print,
) -> pd.DataFrame:
    """
    Ranks each team's DEF unit by average fantasy PPR points scored per game
    through `through_week`, using pts_ppr from the DEF player rows in the
    weekly stats CSVs.

    Output columns:
      team, total_def_pts, games_played, avg_def_pts, def_quality_rank

    def_quality_rank 1 = worst DEF (lowest avg pts/gm)
    def_quality_rank 32 = best DEF (highest avg pts/gm)

    Saves to data/processed/def_quality_rankings_{season}.csv and returns df.
    """
    if through_week < 1:
        return pd.DataFrame()

    # Load players.csv directly (not via _load_players() which filters to skill positions only)
    players_path = PROCESSED_DIR / "players.csv"
    if not players_path.exists():
        log("  ⚠ players.csv not found — run Season Setup first.")
        return pd.DataFrame()
    try:
        all_players_df = pd.read_csv(players_path, dtype={"player_id": str})
    except Exception:
        return pd.DataFrame()

    # DEF player_ids match team abbreviations (ARI, BAL, etc.)
    def_player_ids = set(
        all_players_df[all_players_df["position"] == "DEF"]["player_id"].astype(str)
    )

    team_week_pts: dict[str, dict[int, float]] = defaultdict(dict)

    for week in range(1, through_week + 1):
        stats_df = _load_week_stats(season, week)
        if stats_df.empty:
            continue
        def_rows = stats_df[stats_df["player_id"].astype(str).isin(def_player_ids)]
        for _, row in def_rows.iterrows():
            team = str(row["player_id"])
            raw  = row.get("pts_ppr")
            if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                continue  # skip weeks where DEF had no score recorded (bye/missing)
            pts = float(raw)
            team_week_pts[team][week] = pts

    if not team_week_pts:
        log("  ⚠ No DEF stats found for quality rankings.")
        return pd.DataFrame()

    rows = []
    for team, week_data in sorted(team_week_pts.items()):
        total = sum(week_data.values())
        games = len(week_data)
        avg   = round(total / games, 2) if games > 0 else 0.0
        rows.append({
            "team":          team,
            "total_def_pts": round(total, 1),
            "games_played":  games,
            "avg_def_pts":   avg,
        })

    df = pd.DataFrame(rows)
    # Rank ascending: 1 = worst, 32 = best
    df["def_quality_rank"] = df["avg_def_pts"].rank(method="min", ascending=True).astype(int)

    n = len(df)
    def _quality_label(row: pd.Series) -> str:
        pct = row["def_quality_rank"] / n
        if pct > 0.67:   return "strong"   # top third: great fantasy DEF
        elif pct > 0.33: return "average"
        else:            return "weak"      # bottom third: poor fantasy DEF

    df["def_quality_label"] = df.apply(_quality_label, axis=1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"def_quality_rankings_{season}.csv"
    df.to_csv(out_path, index=False)
    log(f"  ✓ DEF quality rankings built through Week {through_week} ({len(df)} teams)")
    return df


def load_def_quality_rankings(season: str) -> pd.DataFrame:
    """Loads saved DEF quality rankings CSV. Returns empty df if not built yet."""
    path = PROCESSED_DIR / f"def_quality_rankings_{season}.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def get_def_streaming_recs(
    week: int,
    season: str,
    available_def_teams: list[str],
    offense_rankings_df: pd.DataFrame | None = None,
    def_quality_df: pd.DataFrame | None = None,
    top_n: int = 3,
) -> list[dict]:
    """
    Returns the top `top_n` available defenses to stream in `week`, scored by a
    blended formula: 75% matchup quality (how weak the opponent's offense is) +
    25% DEF quality (how many fantasy points the DEF unit itself scores per game).

    Parameters
    ----------
    week                : The upcoming week number
    season              : Season year string
    available_def_teams : List of NFL team abbreviations whose DEF is undrafted
    offense_rankings_df : Pre-loaded offense rankings df (optional, avoids re-read)
    def_quality_df      : Pre-loaded DEF quality rankings df (optional).
                          If None or empty, falls back to 100% matchup weighting.
    top_n               : How many recommendations to return (default 3)

    Scoring (lower combined_score = better pick):
        matchup_component = 0.75 × (opp_offense_rank / n_teams)
            rank 1 (weakest offense) → 0.03  [best matchup]
            rank 32 (strongest)      → 0.75  [worst matchup]
        quality_component = 0.25 × ((n_teams + 1 - def_quality_rank) / n_teams)
            rank 32 (best DEF)       → 0.008 [adds little penalty]
            rank 1  (worst DEF)      → 0.25  [adds large penalty]

    Returns a list of dicts, sorted best pick first:
      [{
        "team":              "DEN",
        "full_name":         "DEN Defense",
        "opponent":          "TEN",
        "rank":              2,       # opponent's offense rank (1=weakest=best)
        "recent_rank":       2,
        "offense_label":     "weak",
        "avg_pts_scored":    28.5,
        "recent_avg":        26.1,
        "def_quality_rank":  24,     # 1=worst DEF, 32=best DEF (None if no data)
        "avg_def_pts":       11.3,   # DEF's own avg fantasy pts/gm (None if no data)
        "def_quality_label": "average",  # "strong" / "average" / "weak"
        "combined_score":    0.19,
      }, ...]
    """
    if offense_rankings_df is None:
        offense_rankings_df = load_offense_rankings(season)
    if def_quality_df is None:
        def_quality_df = load_def_quality_rankings(season)

    n_off = len(offense_rankings_df) if not offense_rankings_df.empty else 32
    have_quality = def_quality_df is not None and not def_quality_df.empty
    n_def = len(def_quality_df) if have_quality else 32

    recs = []
    for def_team in available_def_teams:
        opp = get_opponent_offense(def_team, week, season, offense_rankings_df)
        if not opp:
            continue  # bye or no matchup data

        # ── Matchup component (75%) ───────────────────────────────────────────
        matchup_component = 0.75 * (opp["rank"] / n_off)

        # ── DEF quality component (25%) ───────────────────────────────────────
        def_quality_rank  = None
        avg_def_pts       = None
        def_quality_label = None

        if have_quality:
            q_row = def_quality_df[def_quality_df["team"] == def_team]
            if not q_row.empty:
                def_quality_rank  = int(q_row.iloc[0]["def_quality_rank"])
                avg_def_pts       = float(q_row.iloc[0]["avg_def_pts"])
                def_quality_label = str(q_row.iloc[0]["def_quality_label"])
                quality_component = 0.25 * ((n_def + 1 - def_quality_rank) / n_def)
            else:
                quality_component = 0.25 * 0.5  # unknown: treat as average
        else:
            quality_component = 0.0  # no quality data → pure matchup

        combined_score = matchup_component + quality_component

        recs.append({
            "team":              def_team,
            "full_name":         f"{def_team} Defense",
            "nfl_team":          def_team,
            "position":          "DEF",
            **opp,               # opponent, rank, recent_rank, offense_label, avg_pts_scored, recent_avg
            "def_quality_rank":  def_quality_rank,
            "avg_def_pts":       avg_def_pts,
            "def_quality_label": def_quality_label,
            "combined_score":    round(combined_score, 4),
        })

    # Sort: lowest combined_score = best pick
    recs.sort(key=lambda x: x["combined_score"])
    return recs[:top_n]


# ── Offense matchup emoji / formatter ─────────────────────────────────────────

OFFENSE_EMOJI = {
    "weak":    "🟢",   # easy DEF matchup
    "average": "🟡",
    "strong":  "🔴",   # tough DEF matchup
}

def format_opponent_offense(opp: dict | None) -> str:
    """
    Returns a display string like "🟢 TEN (weak, #3 offense)" or "—".
    Used to show how tough a defense's upcoming opponent is.
    """
    if not opp:
        return "—"
    emoji = OFFENSE_EMOJI.get(opp["offense_label"], "🟡")
    return f"{emoji} vs {opp['opponent']} ({opp['offense_label']} offense, #{opp['recent_rank']})"


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    season     = sys.argv[1] if len(sys.argv) > 1 else "2025"
    week       = int(sys.argv[2]) if len(sys.argv) > 2 else 9
    print(f"Building defense rankings for {season} through Week {week}...")
    df = refresh_defense_rankings(season, week)
    if not df.empty:
        print(df.sort_values(["position", "rank"]).to_string(index=False))
    print()
    print(f"Building offense rankings for {season} through Week {week}...")
    odf = refresh_offense_rankings(season, week)
    if not odf.empty:
        print(odf.sort_values("rank").to_string(index=False))
