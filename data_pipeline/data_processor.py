"""
data_processor.py
-----------------
Builds a multi-year player summary by combining 2022, 2023, and 2024
season data with weighted averages, performance trend labels, and a
breakout score that identifies young players on a rising trajectory.

Weighting scheme (normalized if a player didn't play all three years):
    2024 → 50%
    2023 → 30%
    2022 → 20%

Breakout score (0–100, skill positions only):
    Combines four signals to flag players likely to outperform their
    weighted average heading into the next season:

    1. Late-season surge (30 pts): Finishing 2024 stronger than they started.
       A player trending up in the final 6 weeks is likely entering the next
       season in an expanded role.

    2. Year-over-year improvement (30 pts): Raw magnitude of scoring jump
       from 2023 to 2024. The trend label already flags direction; this
       captures how big the leap was.

    3. Target trend (25 pts, WR/TE only): Targets increasing late in 2024
       signals growing opportunity — a leading indicator for future production
       even if the points haven't fully materialized yet.

    4. Youth bonus (15 pts): Second- and third-year players tend to make their
       biggest jumps as they gain experience. A young player with the above
       signals is more likely to continue improving than a veteran.

Output: data/processed/multi_year_summary.csv

Usage:
    python data_pipeline/data_processor.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

SEASONS = ["2022", "2023", "2024", "2025"]  # oldest to newest — default, overridable

# Base weights assigned by recency (most recent year first).
# Normalized at runtime so they always sum to 1.0 regardless of how many
# seasons are provided. Example for 3 seasons: 0.50/0.30/0.20 → sums to 1.0.
# Example for 2 seasons: 0.50/0.30 → normalized to 0.625/0.375.
RECENCY_WEIGHTS = [0.50, 0.30, 0.20, 0.12, 0.07]  # position 0 = most recent


def compute_season_weights(seasons: list) -> dict:
    """
    Returns a dict mapping each season string to its raw (un-normalized) weight.
    Seasons should be ordered oldest → newest.

    Example:
        compute_season_weights(["2022","2023","2024"])
        → {"2022": 0.20, "2023": 0.30, "2024": 0.50}
    """
    seasons_newest_first = list(reversed(seasons))
    return {
        season: RECENCY_WEIGHTS[i]
        for i, season in enumerate(seasons_newest_first)
        if i < len(RECENCY_WEIGHTS)
    }

# How many PPR points per game difference counts as a meaningful change.
TREND_THRESHOLD = 2.5

# Breakout score thresholds (used for normalization)
# These represent the value at which a player earns the *full* component score.
# Players above the threshold are capped at the max; below zero is capped at 0.
SURGE_MAX_PTS        = 5.0   # pts/gm late-season surge that earns full 30 pts
YOY_MAX_DELTA        = 85.0  # total pts YoY improvement that earns full 30 pts (+5/gm × 17 games)
DEPTH_IMPROVEMENT_MAX = 1    # positions moved up (e.g. WR2→WR1) that earns full 30 pts

# ── Injury penalty thresholds ──────────────────────────────────────────────────
# Career games played % (total career games / career seasons × 17)
#   ≥ 88% → no penalty (avg ≥15 games/season)
#   75–88% → -8 pts
#   60–75% → -15 pts
#   < 60%  → -20 pts
CAREER_DURABILITY_PENALTY = [
    (0.88, 0),    # ≥88% → 0
    (0.75, -8),   # 75–88% → -8
    (0.60, -15),  # 60–75% → -15
    (0.00, -20),  # <60%  → -20
]

# Games played in most recent season (serious injury flag)
#   ≥ 14 games → no penalty
#   9–13 games → -10 pts (missed 4-8 games)
#   < 9 games  → -20 pts (missed more than half the season)
SERIOUS_INJURY_THRESHOLD_HIGH = 14   # no penalty at or above this
SERIOUS_INJURY_THRESHOLD_MID  = 9    # moderate penalty between mid and high
SERIOUS_INJURY_PENALTY_MID    = -10
SERIOUS_INJURY_PENALTY_HIGH   = -20

# ── Current season injury / suspension flags ──────────────────────────────────
# Update this dict whenever a key player has a multi-week injury or suspension.
# status options: "IR" (out 4+ weeks), "OUT" (season-ending), "SUSP" (suspension), "Q" (week-to-week, 2+ weeks)
# The note is passed to the draft agent as context.
CURRENT_INJURY_FLAGS: dict[str, dict] = {
    "Ricky Pearsall":  {"status": "OUT",  "note": "Season-ending injury"},
    "Jordyn Tyson":    {"status": "IR",   "note": "Hamstring — expected to miss ~2 months"},
    "Josh Jacobs":     {"status": "Q",    "note": "Injury/suspension concerns — return timeline unclear"},
}

# ── 2026 rookies (no historical data — injected after each pipeline run) ───────
# Add/remove players here. weighted_avg_ppr set from ADP + depth chart + projections.
ROOKIES_2026: list[dict] = [
    {"full_name": "Jeremiyah Love",    "position": "RB", "team": "ARI", "weighted_avg_ppr": 14.5, "injury_status": "Q",  "injury_note": "Ankle — monitor heading into season"},
    {"full_name": "Jadarian Price",    "position": "RB", "team": "SEA", "weighted_avg_ppr": 12.0, "injury_status": "",   "injury_note": ""},
    {"full_name": "Carnell Tate",      "position": "WR", "team": "TEN", "weighted_avg_ppr": 12.0, "injury_status": "",   "injury_note": ""},
    {"full_name": "De'Zhaun Stribling","position": "WR", "team": "SF",  "weighted_avg_ppr":  9.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Chris Bell",        "position": "WR", "team": "MIA", "weighted_avg_ppr":  9.0, "injury_status": "Q",  "injury_note": "ACL recovery — monitor"},
    {"full_name": "KC Concepcion",     "position": "WR", "team": "CLE", "weighted_avg_ppr":  8.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Makai Lemon",       "position": "WR", "team": "PHI", "weighted_avg_ppr":  8.0, "injury_status": "",   "injury_note": ""},
    {"full_name": "Kenyon Sadiq",      "position": "TE", "team": "NYJ", "weighted_avg_ppr":  6.0, "injury_status": "",   "injury_note": ""},
    {"full_name": "Omar Cooper Jr.",   "position": "WR", "team": "NYJ", "weighted_avg_ppr":  7.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Eli Stowers",       "position": "TE", "team": "PHI", "weighted_avg_ppr":  4.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Fernando Mendoza",  "position": "QB", "team": "LV",  "weighted_avg_ppr": 12.0, "injury_status": "",   "injury_note": ""},
    {"full_name": "Germie Bernard",    "position": "WR", "team": "PIT", "weighted_avg_ppr":  5.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Jonah Coleman",     "position": "RB", "team": "DEN", "weighted_avg_ppr":  4.5, "injury_status": "",   "injury_note": ""},
    {"full_name": "Denzel Boston",     "position": "WR", "team": "CLE", "weighted_avg_ppr":  5.0, "injury_status": "",   "injury_note": ""},
]

# ── Load per-season summaries ──────────────────────────────────────────────────

def load_season_summary(season: str) -> pd.DataFrame:
    """Loads the per-season summary CSV produced by sleeper_client.py."""
    path = PROCESSED_DIR / f"season_{season}_summary.csv"
    if not path.exists():
        print(f"  ⚠  season_{season}_summary.csv not found — run sleeper_client.py first.")
        return pd.DataFrame()
    df = pd.read_csv(path)

    # Columns to carry into the multi-year merge — base stats plus breakout inputs
    keep = [
        "player_id", "full_name", "position", "team",
        "avg_pts_ppr", "total_pts_ppr", "weeks_played",
        "surge_score", "target_trend",
        "early_season_avg_ppr", "late_season_avg_ppr",
        "early_season_avg_targets", "late_season_avg_targets",
        "total_rec_tgt",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Suffix columns with the season year so they don't collide after merging
    rename = {
        "avg_pts_ppr":                f"avg_pts_ppr_{season}",
        "total_pts_ppr":              f"total_pts_ppr_{season}",
        "weeks_played":               f"weeks_played_{season}",
        "surge_score":                f"surge_score_{season}",
        "target_trend":               f"target_trend_{season}",
        "early_season_avg_ppr":       f"early_season_avg_ppr_{season}",
        "late_season_avg_ppr":        f"late_season_avg_ppr_{season}",
        "early_season_avg_targets":   f"early_season_avg_targets_{season}",
        "late_season_avg_targets":    f"late_season_avg_targets_{season}",
        "total_rec_tgt":              f"total_rec_tgt_{season}",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

# ── Weighted average ───────────────────────────────────────────────────────────

def compute_weighted_avg(row: pd.Series, seasons: list, raw_weights: dict) -> float:
    """
    Computes the normalized weighted average of avg_pts_ppr across
    whichever seasons the player has data for.

    Breakout override (non-QB only): if the most recent season avg is 18+
    AND it is more than 3 pts above the standard weighted avg, prior seasons
    are irrelevant — use the most recent season avg directly.
    """
    contributions = []
    for season in seasons:
        col = f"avg_pts_ppr_{season}"
        val = row.get(col)
        if pd.notna(val) and val > 0:
            contributions.append((val, raw_weights[season]))

    if not contributions:
        return 0.0

    total_raw_weight = sum(w for _, w in contributions)
    weighted_sum = sum(avg * (w / total_raw_weight) for avg, w in contributions)
    standard = round(weighted_sum, 1)

    position = row.get("position", "")
    most_recent_col = f"avg_pts_ppr_{seasons[-1]}"
    most_recent = row.get(most_recent_col)

    if position != "QB" and pd.notna(most_recent):
        # Rule 1 — single breakout year: most recent >= 18 and old data
        # drags the blended avg down by 3+ pts.
        if most_recent >= 18 and most_recent - standard > 3:
            return round(float(most_recent), 1)

        # Rule 2 — back-to-back elite: last 2 seasons both >= 18.
        # Older seasons are irrelevant; re-weight using only the last 2.
        if len(seasons) >= 2:
            prev_col = f"avg_pts_ppr_{seasons[-2]}"
            prev = row.get(prev_col)
            if pd.notna(prev) and prev >= 18 and most_recent >= 18:
                w1 = raw_weights[seasons[-1]]
                w2 = raw_weights[seasons[-2]]
                two_season_avg = (most_recent * w1 + prev * w2) / (w1 + w2)
                return round(two_season_avg, 1)

    return standard

# ── Trend label ────────────────────────────────────────────────────────────────

def compute_trend(row: pd.Series, seasons: list) -> str:
    """
    Determines whether a player's per-game scoring improved, declined,
    or stayed stable across the available years.
    """
    year_avgs = []
    for season in seasons:
        col = f"avg_pts_ppr_{season}"
        val = row.get(col)
        if pd.notna(val) and val > 0:
            year_avgs.append((season, val))

    if len(year_avgs) < 2:
        return "Insufficient Data"

    oldest_avg = year_avgs[0][1]
    newest_avg = year_avgs[-1][1]
    delta = newest_avg - oldest_avg

    if delta >= TREND_THRESHOLD:
        return "Improving ↑"
    elif delta <= -TREND_THRESHOLD:
        return "Declining ↓"
    else:
        return "Stable →"

# ── Count years of data ────────────────────────────────────────────────────────

def count_years_played(row: pd.Series, seasons: list) -> int:
    """Returns how many seasons the player has non-zero avg PPR data for."""
    return sum(
        1 for season in seasons
        if pd.notna(row.get(f"avg_pts_ppr_{season}")) and row.get(f"avg_pts_ppr_{season}", 0) > 0
    )

# ── Breakout score ─────────────────────────────────────────────────────────────

def compute_breakout_score(row: pd.Series) -> float:
    """
    Computes a 0–100 breakout score for skill position players (QB, RB, WR, TE).
    Returns None for K, DEF, and players who already averaged 18+ pts in the
    most recent season (already broken out — score doesn't apply to them).

    Four components (max score 100). Year references are dynamic — always uses
    SEASONS[-1] as the most recent season and SEASONS[-2] as the prior season.

    1. Late-season surge (max 30 pts)
       surge_score = late_season_avg_ppr (wks 14–18) - early_season_avg_ppr (wks 1–9)
       from the most recent season. Full marks at +5 pts/gm improvement.

    2. Year-over-year improvement (max 30 pts)
       total pts in most recent season minus total pts in prior season.
       Full marks at +85 total pts (≈ +5 pts/gm over 17 games).
       Both seasons must have ≥12 games and differ by ≤3 games.
       Rookies with no prior season data get full 30 pts automatically.

    3. Depth chart improvement (max 30 pts)
       depth_improvement = prior depth position − current depth position.
       Positive means moved up (e.g. WR3→WR1 = +2). Full marks at +1 position.
       Uses the two most recent depth chart files (e.g. 2025 vs 2026).

    4. Youth bonus (max 10 pts)
       Based on adjusted_years_exp (experience as of the draft year):
         1–2 years: 10 pts
         3 years:    7 pts (2/3 of max)
         4 years:    3 pts (1/3 of max)
         5+ years:   0 pts

    """
    position = row.get("position", "")
    if position not in {"QB", "RB", "WR", "TE"}:
        return None

    # Use the two most recent seasons dynamically so this works for any draft year.
    recent_yr  = SEASONS[-1]   # e.g. "2025"
    prior_yr   = SEASONS[-2] if len(SEASONS) >= 2 else None  # e.g. "2024"

    # Players who already averaged 18+ pts/game in their most recent season have
    # already broken out — the breakout score doesn't apply to them.
    avg_recent = row.get(f"avg_pts_ppr_{recent_yr}")
    if pd.notna(avg_recent) and avg_recent >= 18:
        return None

    score = 0.0

    # ── Component 1: Late-season surge (max 30 pts) ───────────────────────────
    surge = row.get(f"surge_score_{recent_yr}")
    if pd.notna(surge):
        score += min(30.0, max(0.0, (surge / SURGE_MAX_PTS) * 30))

    # ── Component 2: Year-over-year improvement (max 30 pts) ──────────────────
    total_recent = row.get(f"total_pts_ppr_{recent_yr}")
    total_prior  = row.get(f"total_pts_ppr_{prior_yr}") if prior_yr else None
    weeks_recent = row.get(f"weeks_played_{recent_yr}", 0) or 0
    weeks_prior  = row.get(f"weeks_played_{prior_yr}", 0) or 0 if prior_yr else 0

    if pd.notna(total_recent) and pd.notna(total_prior) and total_prior > 0:
        games_ok = (weeks_recent >= 12 and weeks_prior >= 12
                    and abs(weeks_recent - weeks_prior) <= 3)
        if games_ok:
            yoy_delta = total_recent - total_prior
            score += min(30.0, max(0.0, (yoy_delta / YOY_MAX_DELTA) * 30))
    elif pd.notna(total_recent):
        score += 30.0   # rookie — give full marks, can't penalise for no prior season

    # ── Component 3: Depth chart improvement (max 30 pts) ─────────────────────
    # depth_improvement > 0 means the player moved up (lower number = higher rank).
    depth_improvement = row.get("depth_improvement")
    if pd.notna(depth_improvement) and depth_improvement > 0:
        score += min(30.0, (depth_improvement / DEPTH_IMPROVEMENT_MAX) * 30)

    # ── Component 4: Youth bonus (max 10 pts) ─────────────────────────────────
    years_exp = row.get("adjusted_years_exp", row.get("years_exp"))
    if pd.notna(years_exp):
        years_exp = max(0, int(years_exp))
        if years_exp <= 2:
            score += 10.0
        elif years_exp == 3:
            score += round(10 * 2/3)   # 7 pts
        elif years_exp == 4:
            score += round(10 * 1/3)   # 3 pts

    return round(score, 1)

# ── Durability score (separate from breakout score) ───────────────────────────

def compute_durability_score(row: pd.Series) -> float | None:
    """
    Computes a 0–100 durability score reflecting a player's injury history.
    Higher = more durable / lower injury risk.

    Two components:
      1. Career availability (70 pts) — Bayesian-adjusted
         Blends the player's actual career_games_pct toward the league-average
         (87%) based on seasons of data. A 3-year player with one missed year
         is pulled toward average more than a 7-year veteran with a pattern.

         adjusted_pct = (career_pct × seasons + LEAGUE_AVG × PRIOR_SEASONS)
                        / (seasons + PRIOR_SEASONS)

         PRIOR_SEASONS = 2 → a player needs ~3+ seasons before their own
         history dominates over the league average.

      2. Recent season health (30 pts)
         Based on games played in most recent season:
           ≥14 games → 30 pts (healthy full season)
           9–13 games → 15 pts (missed 4–8 games)
           <9 games  →  0 pts  (missed more than half the season)
    """
    LEAGUE_AVG_AVAILABILITY      = 0.87  # approx NFL skill-position average
    PRIOR_SEASONS                = 2     # how much to weight the prior (seasons equivalent)
    REPEATED_INJURY_PENALTY_PER  = 8     # points deducted per season with <12 games
    REPEATED_INJURY_CAP          = 25    # max total penalty from this flag
    REPEATED_INJURY_MIN_SEASONS  = 2     # flag only triggers if 2+ seasons under threshold

    position = row.get("position", "")
    if position not in {"QB", "RB", "WR", "TE"}:
        return None

    # Component 1: Bayesian-adjusted career availability (70 pts)
    career_pct     = row.get("career_games_pct")
    career_seasons = row.get("career_seasons")

    if pd.notna(career_pct) and pd.notna(career_seasons) and career_seasons > 0:
        n = float(career_seasons)
        adjusted_pct = (float(career_pct) * n + LEAGUE_AVG_AVAILABILITY * PRIOR_SEASONS) \
                       / (n + PRIOR_SEASONS)
    else:
        adjusted_pct = LEAGUE_AVG_AVAILABILITY   # no data → assume average

    career_pts = round(adjusted_pct * 70, 1)

    # Component 2: Recent season health (30 pts)
    recent_games = row.get(f"games_played_{SEASONS[-1]}")
    if pd.isna(recent_games):
        recent_games = row.get(f"weeks_played_{SEASONS[-1]}")
    if pd.notna(recent_games):
        recent_games = int(recent_games)
        if recent_games >= SERIOUS_INJURY_THRESHOLD_HIGH:
            recent_pts = 30
        elif recent_games >= SERIOUS_INJURY_THRESHOLD_MID:
            recent_pts = 15
        else:
            recent_pts = 0
    else:
        recent_pts = 15   # no data → assume moderate

    # Component 3: Repeated injury flag penalty
    # Triggers only if a player had 2+ seasons with fewer than 12 games played.
    # -8 pts per such season, capped at -25. This catches CMC/Kittle-style
    # patterns that the career average alone doesn't penalize enough.
    seasons_under_12 = row.get("seasons_under_12")
    if pd.notna(seasons_under_12) and int(seasons_under_12) >= REPEATED_INJURY_MIN_SEASONS:
        flag_penalty = min(int(seasons_under_12) * REPEATED_INJURY_PENALTY_PER, REPEATED_INJURY_CAP)
    else:
        flag_penalty = 0

    return round(max(0.0, min(100.0, career_pts + recent_pts - flag_penalty)), 1)

# ── Build multi-year summary ───────────────────────────────────────────────────

def build_multi_year_summary(seasons: list = None, draft_year: int = None, log=print) -> pd.DataFrame:
    """
    Merges per-season summaries, computes weighted averages, trend labels,
    and breakout scores, then saves to multi_year_summary.csv.

    Parameters:
        seasons:    List of season year strings oldest→newest, e.g. ["2022","2023","2024"].
                    Defaults to the module-level SEASONS constant.
        draft_year: The season being drafted for (e.g. 2025). Used to compute
                    adjusted_years_exp so the youth bonus reflects how experienced
                    a player was at draft time, not today. Defaults to current year.
        log:        Callable for progress messages.
    """
    if seasons is None:
        seasons = SEASONS

    raw_weights = compute_season_weights(seasons)
    most_recent = seasons[-1]

    log("\nLoading per-season summaries...")

    season_dfs = {}
    for season in seasons:
        df = load_season_summary(season)
        if not df.empty:
            season_dfs[season] = df
            log(f"  ✓ {season}: {len(df):,} players loaded")

    if not season_dfs:
        log("  ✗ No season data found. Run the season stats fetch first.")
        return pd.DataFrame()

    # ── Merge all seasons on player_id ────────────────────────────────────────
    # Start with the most recent season as the base (current team/name).
    base = season_dfs.get(most_recent, list(season_dfs.values())[-1])
    identity_cols = ["player_id", "full_name", "position", "team"]

    merged = base.copy()
    for season in reversed(seasons[:-1]):   # older seasons, newest-first
        if season not in season_dfs:
            continue
        older = season_dfs[season].drop(
            columns=[c for c in identity_cols if c != "player_id"],
            errors="ignore"
        )
        merged = merged.merge(older, on="player_id", how="outer")

    # Fill missing identity info from older seasons
    for season in reversed(seasons[:-1]):
        if season not in season_dfs:
            continue
        src = season_dfs[season][["player_id"] + [c for c in identity_cols if c != "player_id"]]
        for col in ["full_name", "position", "team"]:
            if col in src.columns:
                merged[col] = merged[col].fillna(
                    merged["player_id"].map(src.set_index("player_id")[col])
                )

    # Keep only fantasy positions
    merged = merged[merged["position"].isin({"QB","RB","WR","TE","K","DEF"})].copy()

    # ── Carry years_exp and age from players.csv ──────────────────────────────
    players_path = PROCESSED_DIR / "players.csv"
    if players_path.exists():
        players_df = pd.read_csv(players_path)[["player_id", "years_exp", "age"]]
        merged = merged.merge(players_df, on="player_id", how="left")
        log(f"  ✓ Merged years_exp and age from players.csv")

        # Adjust years_exp to reflect the draft year, not today.
        # years_exp from Sleeper is as of the current date, so a player who
        # had 2 years of experience going into the 2025 season might show 3
        # if we're fetching data in 2026.
        current_year = datetime.now().year
        effective_draft_year = draft_year if draft_year else current_year
        year_offset = current_year - effective_draft_year
        merged["adjusted_years_exp"] = (merged["years_exp"] - year_offset).clip(lower=0)
        log(f"  ✓ Adjusted years_exp for {effective_draft_year} draft year (offset: {year_offset})")
    else:
        log(f"  ⚠  players.csv not found — years_exp/age will be missing from breakout score")

    # ── Merge depth chart positions (prior year vs draft year) ───────────────
    # Compare each player's depth chart position in the prior season vs the
    # upcoming season. A player who moved from WR2 → WR1 gets depth_improvement = +1.
    # Uses the two most recent depth chart files available.
    all_dc_files = sorted(PROCESSED_DIR.glob("depth_charts_*.csv"))
    if len(all_dc_files) >= 2:
        dc_prev_path = all_dc_files[-2]   # e.g. depth_charts_2025.csv
        dc_curr_path = all_dc_files[-1]   # e.g. depth_charts_2026.csv
        prev_yr = dc_prev_path.stem.replace("depth_charts_", "")
        curr_yr = dc_curr_path.stem.replace("depth_charts_", "")

        dc_prev = pd.read_csv(dc_prev_path)[["full_name", "fantasy_position", "depth_position"]]
        dc_curr = pd.read_csv(dc_curr_path)[["full_name", "fantasy_position", "depth_position"]]

        dc_prev = dc_prev.rename(columns={"depth_position": f"depth_position_{prev_yr}",
                                           "fantasy_position": "dc_position"})
        dc_curr = dc_curr.rename(columns={"depth_position": f"depth_position_{curr_yr}",
                                           "fantasy_position": f"dc_position_{curr_yr}"})

        # Normalise names: strip suffixes, then use lowercase keys for
        # case-insensitive matching (handles "Ceedee Lamb", "Dk Metcalf", etc.)
        suffix_pattern = r'(?i)\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
        for df_ in [dc_prev, dc_curr]:
            df_["full_name"] = df_["full_name"].str.replace(
                suffix_pattern, "", regex=True
            ).str.strip().str.lower()
        merged["_name_lower"] = merged["full_name"].str.replace(
            suffix_pattern, "", regex=True
        ).str.strip().str.lower()

        dc_prev = dc_prev.sort_values(f"depth_position_{prev_yr}").drop_duplicates("full_name")
        dc_curr = dc_curr.sort_values(f"depth_position_{curr_yr}").drop_duplicates("full_name")

        dc_combined = dc_prev.merge(dc_curr[["full_name", f"depth_position_{curr_yr}"]],
                                    on="full_name", how="outer")
        dc_combined["depth_improvement"] = (dc_combined[f"depth_position_{prev_yr}"]
                                             - dc_combined[f"depth_position_{curr_yr}"])

        merged = merged.merge(
            dc_combined[["full_name", f"depth_position_{prev_yr}",
                          f"depth_position_{curr_yr}", "depth_improvement"]],
            left_on="_name_lower", right_on="full_name", how="left", suffixes=("", "_dc")
        )
        # Drop helper columns
        for c in ["_name_lower", "full_name_dc"]:
            if c in merged.columns:
                merged.drop(columns=[c], inplace=True)
        log(f"  ✓ Depth chart data merged ({len(dc_combined):,} players) [{prev_yr} vs {curr_yr}]")
    else:
        log("  ⚠  Need at least 2 depth chart CSVs for depth_improvement — run depth chart scraper")

    # ── Add per-season team columns from depth charts ─────────────────────────
    # The generic `team` column always reflects the player's CURRENT team (as
    # of when the Sleeper player database was last fetched).  For historical
    # season simulations we need the team the player was actually on during
    # that season.  Depth charts are scraped with the correct season context,
    # so they're the authoritative source for season-specific team assignments.
    #
    # We add a `team_{year}` column for every depth chart file we can find.
    # app.py's season-aware player loader will swap `team` for `team_{year}`
    # when the user is running a simulation for that season.
    # Scan all available depth chart files (not just seasons+1) so that
    # e.g. depth_charts_2026.csv is picked up even when SEASONS stops at 2024.
    all_dc_years = sorted(
        p.stem.replace("depth_charts_", "")
        for p in PROCESSED_DIR.glob("depth_charts_*.csv")
    )
    for yr_str in all_dc_years:
        dc_path = PROCESSED_DIR / f"depth_charts_{yr_str}.csv"
        col_name = f"team_{yr_str}"
        if not dc_path.exists():
            continue
        try:
            dc = pd.read_csv(dc_path)[["full_name", "nfl_team"]].drop_duplicates("full_name")
            # Normalize suffixes; then match case-insensitively to handle
            # title-cased names like "Ceedee Lamb", "Dk Metcalf", "Dj Moore"
            suffix_pattern = r'(?i)\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
            dc["full_name"] = dc["full_name"].str.replace(suffix_pattern, "", regex=True).str.strip()
            # Build a lowercase → team lookup, map via lowercase player names
            team_map = dc.set_index(dc["full_name"].str.lower())["nfl_team"]
            merged[col_name] = merged["full_name"].str.lower().map(team_map)
            n_matched = merged[col_name].notna().sum()
            log(f"  ✓ Added {col_name} for {n_matched:,} players from depth_charts_{yr_str}.csv")
        except Exception as e:
            log(f"  ⚠  Could not add {col_name}: {e}")

    # ── Compute avg_targets_pg per season (total_rec_tgt / weeks_played) ────────
    # This gives a clean full-season targets-per-game average, injury-adjusted,
    # which is more accurate than averaging the early/late season splits.
    for season in seasons:
        tgt_col = f"total_rec_tgt_{season}"
        wks_col = f"weeks_played_{season}"
        out_col = f"avg_targets_pg_{season}"
        if tgt_col in merged.columns and wks_col in merged.columns:
            played = merged[wks_col].replace(0, 1)
            merged[out_col] = (merged[tgt_col] / played).round(1)

    # ── Merge career durability data ──────────────────────────────────────────
    durability_path = PROCESSED_DIR / "career_durability.csv"
    if durability_path.exists():
        dur_cols = ["full_name", "career_seasons", "career_games_played",
                    "career_games_pct", "games_played_2025"]
        if "seasons_under_12" in pd.read_csv(durability_path, nrows=0).columns:
            dur_cols.append("seasons_under_12")
        dur = pd.read_csv(durability_path)[dur_cols]
        # Normalize suffixes to match
        suffix_pattern = r'(?i)\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
        dur["full_name"] = dur["full_name"].str.replace(suffix_pattern, "", regex=True).str.strip()
        dur = dur.drop_duplicates("full_name")
        merged = merged.merge(dur, on="full_name", how="left")
        n_matched = merged["career_games_pct"].notna().sum()
        log(f"  ✓ Career durability merged for {n_matched:,} players")
    else:
        log("  ⚠  career_durability.csv not found — run build_career_durability.py for injury penalties")

    # ── Compute weighted average, trend, and breakout score ───────────────────
    log("\nComputing weighted averages, trend labels, and breakout scores...")

    merged["years_of_data"]      = merged.apply(lambda r: count_years_played(r, seasons), axis=1)
    merged["weighted_avg_ppr"]   = merged.apply(lambda r: compute_weighted_avg(r, seasons, raw_weights), axis=1)
    merged["trend"]              = merged.apply(lambda r: compute_trend(r, seasons), axis=1)
    merged["breakout_score"]     = merged.apply(compute_breakout_score, axis=1)
    merged["durability_score"]   = merged.apply(compute_durability_score, axis=1)

    # ── Compute Value Over Replacement (VOR) ─────────────────────────────────
    # Raw PPR averages put QBs at the top (they score 20-25 pts/game vs 15-20
    # for skill positions), but in real drafts QBs go rounds 3-5+ because
    # every team needs one and the dropoff between QB12 and QB24 is small.
    #
    # VOR = player_avg - replacement_level_avg, where replacement level is the
    # last positional starter that a typical team would draft.  Players with
    # higher VOR are genuinely scarce at their position and should be ranked
    # (and drafted) earlier.
    #
    # Replacement ranks for a 12-team PPR league (scale linearly with league size):
    #   QB : 14  (12 starters + 2 handcuff/backup buffer)
    #   RB : 36  (12 × 2 starters + ~12 FLEX slots)
    #   WR : 36  (12 × 2 starters + ~12 FLEX slots)
    #   TE : 14  (12 starters + 2 buffer)
    #   K  : 14  (12 starters + 2 buffer)
    #   DEF: 14

    league_size = 12    # default; transactions pipeline can override if needed
    repl_ranks = {
        "QB":  max(1, round(league_size * 1.15)),
        "RB":  max(1, round(league_size * 3.0)),
        "WR":  max(1, round(league_size * 3.0)),
        "TE":  max(1, round(league_size * 1.15)),
        "K":   max(1, round(league_size * 1.15)),
        "DEF": max(1, round(league_size * 1.15)),
    }

    replacement_avgs: dict[str, float] = {}
    for pos, repl_n in repl_ranks.items():
        pos_players = (
            merged[merged["position"] == pos]
            .sort_values("weighted_avg_ppr", ascending=False)
        )
        if len(pos_players) >= repl_n:
            replacement_avgs[pos] = float(pos_players.iloc[repl_n - 1]["weighted_avg_ppr"])
        elif not pos_players.empty:
            replacement_avgs[pos] = float(pos_players.iloc[-1]["weighted_avg_ppr"])
        else:
            replacement_avgs[pos] = 0.0

    merged["replacement_avg"] = merged["position"].map(replacement_avgs).fillna(0)
    merged["value_over_replacement"] = (
        merged["weighted_avg_ppr"] - merged["replacement_avg"]
    ).round(2)

    # K and DEF are always drafted last (rounds 14-15 in a 15-round draft).
    # Penalize their VOR so they sort below all skill positions regardless of
    # how many points they score — the ranking should reflect actual draft order.
    merged.loc[merged["position"].isin(["K", "DEF"]), "value_over_replacement"] -= 4

    log(f"  ✓ VOR replacement baselines: " +
        ", ".join(f"{p}={v:.1f}" for p, v in sorted(replacement_avgs.items())))

    # ── Freshen team from live Sleeper /v1/players/nfl endpoint ─────────────
    # Single API call, no bot-detection issues. Sleeper updates within ~24hrs
    # of most roster moves. Use SLEEPER_CORRECTIONS below for known stale data.
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen(
            "https://api.sleeper.app/v1/players/nfl", timeout=15
        ) as resp:
            sleeper_players = _json.loads(resp.read())

        skill_pos = {"QB", "RB", "WR", "TE", "K"}
        sleeper_team_map: dict[str, str] = {}
        for pid, p in sleeper_players.items():
            pos  = (p.get("position") or "").upper()
            name = (p.get("full_name") or "").strip()
            team = (p.get("team") or "").strip() or "FA"
            if pos in skill_pos and name:
                sleeper_team_map[name.lower()] = team

        merged["_name_lower"] = merged["full_name"].str.lower()
        merged["team"] = merged["_name_lower"].map(sleeper_team_map).fillna(merged["team"])
        merged.drop(columns=["_name_lower"], inplace=True)
        log(f"  ✓ Team column freshened from Sleeper API ({len(sleeper_team_map):,} players)")

    except Exception as e:
        log(f"  ⚠  Sleeper API failed ({e}) — using depth chart team columns")
        priority_years = sorted(
            c.replace("team_", "") for c in merged.columns
            if c.startswith("team_") and c[5:].isdigit()
        )
        for yr in priority_years:
            col = f"team_{yr}"
            if col not in merged.columns:
                continue
            valid = merged[col].notna() & (merged[col].astype(str).str.strip() != "")
            merged.loc[valid, "team"] = merged.loc[valid, col]

    # ── Sleeper corrections: update when Sleeper lags behind a signing/cut ───
    # Remove an entry once Sleeper's DB catches up (usually within a few days).
    SLEEPER_CORRECTIONS = {
        "Joe Mixon":      "FA",    # released by HOU
        "Tyreek Hill":    "FA",    # cut by MIA
        "Keenan Allen":   "IND",   # signed with Colts
        "Jonnu Smith":    "GB",    # signed with Packers
        "Daniel Carlson": "NO",    # signed with Saints
        "Darren Waller":  "CAR",   # signed with Panthers
    }
    for player, team in SLEEPER_CORRECTIONS.items():
        mask = merged["full_name"] == player
        if mask.any():
            merged.loc[mask, "team"] = team
    log(f"  ✓ Applied {len(SLEEPER_CORRECTIONS)} Sleeper corrections")

    # ── Auto FA: skill-position players with no 2026 depth chart entry AND
    # Sleeper shows them as FA are genuinely not on a roster.
    if "team_2026" in merged.columns:
        no_2026 = (
            merged["team_2026"].isna() &
            merged["position"].isin({"QB", "RB", "WR", "TE"}) &
            (merged["team"] == "FA")
        )
        n_fa = no_2026.sum()
        log(f"  ✓ Confirmed {n_fa} skill-position players as FA (no 2026 depth chart + Sleeper FA)")


    # ── Apply current injury / suspension flags ───────────────────────────────
    merged["injury_status"] = ""
    merged["injury_note"]   = ""
    for player, info in CURRENT_INJURY_FLAGS.items():
        mask = merged["full_name"].str.lower() == player.lower()
        merged.loc[mask, "injury_status"] = info.get("status", "")
        merged.loc[mask, "injury_note"]   = info.get("note", "")
    n_flagged = (merged["injury_status"] != "").sum()
    log(f"  ✓ Applied {n_flagged} injury/suspension flags")

    # ── Inject 2026 rookies (no historical data) ──────────────────────────────
    REPL_AVG = {"RB": 11.0, "WR": 11.9, "TE": 9.9, "QB": 18.2}
    rookie_names = [r["full_name"] for r in ROOKIES_2026]
    merged = merged[~merged["full_name"].isin(rookie_names)].copy()  # drop any stale rows
    import numpy as np
    rookie_rows = []
    for r in ROOKIES_2026:
        repl = REPL_AVG.get(r["position"], 11.0)
        row  = {c: np.nan for c in merged.columns}
        row["full_name"]              = r["full_name"]
        row["position"]               = r["position"]
        row["team"]                   = r["team"]
        row["weighted_avg_ppr"]       = r["weighted_avg_ppr"]
        row["replacement_avg"]        = repl
        row["value_over_replacement"] = round(r["weighted_avg_ppr"] - repl, 2)
        row["years_exp"]              = 0
        row["adjusted_years_exp"]     = 0
        row["trend"]                  = 0
        row["breakout_score"]         = 0
        row["durability_score"]       = 0
        row["injury_status"]          = r.get("injury_status", "")
        row["injury_note"]            = r.get("injury_note", "")
        rookie_rows.append(row)
    merged = pd.concat([merged, pd.DataFrame(rookie_rows)], ignore_index=True)
    log(f"  ✓ Injected {len(rookie_rows)} 2026 rookies")

    # Sort by VOR descending — this gives a realistic draft-order ranking
    merged = merged.sort_values("value_over_replacement", ascending=False).reset_index(drop=True)
    merged.insert(0, "season_rank", merged.index + 1)

    # ── Merge 2026 ADP (FantasyPros consensus) if available ──────────────────
    adp_path = PROCESSED_DIR / "adp_2026.csv"
    if adp_path.exists():
        import re as _re
        adp_df = pd.read_csv(adp_path)
        def _strip_suffix(n):
            return _re.sub(r'\s+(jr\.?|sr\.?|ii+|iv|v)$', '', str(n).lower().strip(), flags=_re.I).strip()
        adp_map_exact   = {r["name"].lower(): r["adp_rank"] for _, r in adp_df.iterrows()}
        adp_map_nosufx  = {_strip_suffix(k): v for k, v in adp_map_exact.items()}
        def _lookup_adp(name):
            lo = name.lower()
            return adp_map_exact.get(lo) or adp_map_nosufx.get(_strip_suffix(lo))
        merged["adp_2026"] = merged["full_name"].apply(_lookup_adp)
        n_adp = merged["adp_2026"].notna().sum()
        log(f"  ✓ Merged ADP rankings for {n_adp} players")
    else:
        log("  ℹ  adp_2026.csv not found — skipping ADP merge")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "multi_year_summary.csv"
    merged.to_csv(out_path, index=False)
    log(f"  ✓ Saved {len(merged):,} players to multi_year_summary.csv")

    return merged

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Fantasy Football — Multi-Year Data Processor")
    weights = compute_season_weights(SEASONS)
    wt_str = ", ".join(f"{s}={int(w*100)}%" for s, w in sorted(weights.items(), reverse=True))
    print(f"  Seasons: {', '.join(SEASONS)}  |  Weights (pre-norm): {wt_str}")
    print("=" * 60)

    summary = build_multi_year_summary(SEASONS)
    if summary.empty:
        return

    # ── Spot-check: top 5 per position ────────────────────────────────────────
    print("\n📊 Top 5 players per position (weighted PPR avg):\n")
    display_cols = ["season_rank","full_name","team",
                    "avg_pts_ppr_2022","avg_pts_ppr_2023","avg_pts_ppr_2024",
                    "weighted_avg_ppr","trend","years_of_data"]
    display_cols = [c for c in display_cols if c in summary.columns]

    for pos in ["QB","RB","WR","TE"]:
        top5 = summary[summary["position"] == pos].head(5)
        if top5.empty:
            continue
        print(f"  ── {pos} ──────────────────────────────────────────────")
        print(top5[display_cols].to_string(index=False))
        print()

    # ── Spot-check: top breakout candidates ───────────────────────────────────
    print("\n🚀 Top 10 breakout candidates (highest breakout score):\n")
    _recent = SEASONS[-1]
    _prior  = SEASONS[-2] if len(SEASONS) >= 2 else SEASONS[-1]
    breakout_cols = ["full_name","position","team","breakout_score",
                     f"surge_score_{_recent}",f"avg_pts_ppr_{_prior}",f"avg_pts_ppr_{_recent}",
                     f"target_trend_{_recent}","years_exp","trend"]
    breakout_cols = [c for c in breakout_cols if c in summary.columns]
    top_breakout = summary[summary["breakout_score"] > 0].nlargest(10, "breakout_score")
    if not top_breakout.empty:
        print(top_breakout[breakout_cols].to_string(index=False))

    # ── Sanity check: verify weights sum to 1.0 for a sample player ───────────
    raw_weights = compute_season_weights(SEASONS)
    sample = summary[summary["years_of_data"] == 3].head(1)
    if not sample.empty:
        row = sample.iloc[0]
        contributions = [(row[f"avg_pts_ppr_{s}"], raw_weights[s]) for s in SEASONS
                         if pd.notna(row.get(f"avg_pts_ppr_{s}")) and row[f"avg_pts_ppr_{s}"] > 0]
        total_w = sum(w for _, w in contributions)
        print(f"\n  ✓ Weight check ({row['full_name']}): raw weights sum = {total_w:.2f} → normalized to 1.0")

    print("\n✅ Multi-year summary complete!")
    print(f"   Output: data/processed/multi_year_summary.csv")
    print(f"   New columns: surge_score_{SEASONS[-1]}, target_trend_{SEASONS[-1]}, breakout_score")

if __name__ == "__main__":
    main()
