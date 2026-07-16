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

SEASONS = ["2022", "2023", "2024"]  # oldest to newest — default, overridable

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
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df

# ── Weighted average ───────────────────────────────────────────────────────────

def compute_weighted_avg(row: pd.Series, seasons: list, raw_weights: dict) -> float:
    """
    Computes the normalized weighted average of avg_pts_ppr across
    whichever seasons the player has data for.
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
    return round(weighted_sum, 1)

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
    Returns 0 for K and DEF — the concept doesn't apply to those positions.

    Four components (max score 100):

    1. Late-season surge (max 30 pts)
       surge_score = late_season_avg_ppr (wks 14–18) - early_season_avg_ppr (wks 1–9)
       Full marks at +5 pts/gm improvement.

    2. Year-over-year improvement (max 30 pts)
       yoy_delta = total_pts_ppr_2024 - total_pts_ppr_2023
       Full marks at +85 total pts (≈ +5 pts/gm over 17 games).
       Both seasons must have ≥12 games and differ by ≤3 games.
       Rookies (only 2024 data) get a neutral 15 pts.

    3. Depth chart improvement (max 30 pts)
       depth_improvement = depth_position_2024 - depth_position_2025
       Positive means moved up (e.g. WR2→WR1 = +1). Full marks at +1 position.

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

    # RBs and WRs who already averaged 18+ pts/game in 2024 have already broken
    # out — the score doesn't apply to them. Return None (displayed as N/A in UI).
    if position in {"RB", "WR"}:
        avg_2024 = row.get("avg_pts_ppr_2024")
        if pd.notna(avg_2024) and avg_2024 >= 18:
            return None

    score = 0.0

    # ── Component 1: Late-season surge (max 30 pts) ───────────────────────────
    surge = row.get("surge_score_2024")
    if pd.notna(surge):
        score += min(30.0, max(0.0, (surge / SURGE_MAX_PTS) * 30))

    # ── Component 2: Year-over-year improvement (max 30 pts) ──────────────────
    total_2024 = row.get("total_pts_ppr_2024")
    total_2023 = row.get("total_pts_ppr_2023")
    weeks_2024 = row.get("weeks_played_2024", 0) or 0
    weeks_2023 = row.get("weeks_played_2023", 0) or 0

    if pd.notna(total_2024) and pd.notna(total_2023) and total_2023 > 0:
        games_ok = (weeks_2024 >= 12 and weeks_2023 >= 12
                    and abs(weeks_2024 - weeks_2023) <= 3)
        if games_ok:
            yoy_delta = total_2024 - total_2023
            score += min(30.0, max(0.0, (yoy_delta / YOY_MAX_DELTA) * 30))
    elif pd.notna(total_2024):
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

    # ── Merge depth chart positions (2024 vs 2025) ────────────────────────────
    # Compare each player's depth chart position going into 2024 vs 2025.
    # A player who moved from WR2 → WR1 gets depth_improvement = +1.
    dc_2024_path = PROCESSED_DIR / "depth_charts_2024.csv"
    dc_2025_path = PROCESSED_DIR / "depth_charts_2025.csv"

    if dc_2024_path.exists() and dc_2025_path.exists():
        dc_2024 = pd.read_csv(dc_2024_path)[["full_name", "fantasy_position", "depth_position"]]
        dc_2025 = pd.read_csv(dc_2025_path)[["full_name", "fantasy_position", "depth_position"]]

        dc_2024 = dc_2024.rename(columns={"depth_position": "depth_position_2024",
                                           "fantasy_position": "dc_position"})
        dc_2025 = dc_2025.rename(columns={"depth_position": "depth_position_2025",
                                           "fantasy_position": "dc_position_2025"})

        # Normalise names by stripping common suffixes (Jr., Sr., II, III, IV)
        # so "Brian Thomas Jr." matches "Brian Thomas" in the Sleeper data.
        suffix_pattern = r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
        for df_ in [dc_2024, dc_2025, merged]:
            df_["full_name"] = df_["full_name"].str.replace(
                suffix_pattern, "", regex=True
            ).str.strip()

        # Keep only the best (lowest) depth number per player per position
        dc_2024 = dc_2024.sort_values("depth_position_2024").drop_duplicates("full_name")
        dc_2025 = dc_2025.sort_values("depth_position_2025").drop_duplicates("full_name")

        dc_combined = dc_2024.merge(dc_2025[["full_name", "depth_position_2025"]],
                                    on="full_name", how="outer")
        dc_combined["depth_improvement"] = (dc_combined["depth_position_2024"]
                                             - dc_combined["depth_position_2025"])

        merged = merged.merge(
            dc_combined[["full_name", "depth_position_2024",
                          "depth_position_2025", "depth_improvement"]],
            on="full_name", how="left"
        )
        log(f"  ✓ Depth chart data merged ({len(dc_combined):,} players)")
    else:
        log("  ⚠  Depth chart CSVs not found — run depth_chart_scraper.py for 2024 and 2025")

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
    for yr_str in seasons + [str(int(seasons[-1]) + 1)]:
        dc_path = PROCESSED_DIR / f"depth_charts_{yr_str}.csv"
        col_name = f"team_{yr_str}"
        if not dc_path.exists():
            continue
        try:
            dc = pd.read_csv(dc_path)[["full_name", "nfl_team"]].drop_duplicates("full_name")
            # Normalize suffixes to match how names were already normalized above
            suffix_pattern = r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
            dc["full_name"] = dc["full_name"].str.replace(suffix_pattern, "", regex=True).str.strip()
            team_map = dc.set_index("full_name")["nfl_team"]
            merged[col_name] = merged["full_name"].map(team_map)
            n_matched = merged[col_name].notna().sum()
            log(f"  ✓ Added {col_name} for {n_matched:,} players from depth_charts_{yr_str}.csv")
        except Exception as e:
            log(f"  ⚠  Could not add {col_name}: {e}")

    # ── Compute weighted average, trend, and breakout score ───────────────────
    log("\nComputing weighted averages, trend labels, and breakout scores...")

    merged["years_of_data"]    = merged.apply(lambda r: count_years_played(r, seasons), axis=1)
    merged["weighted_avg_ppr"] = merged.apply(lambda r: compute_weighted_avg(r, seasons, raw_weights), axis=1)
    merged["trend"]            = merged.apply(lambda r: compute_trend(r, seasons), axis=1)
    merged["breakout_score"]   = merged.apply(compute_breakout_score, axis=1)

    # Sort by weighted average (best first) and assign a new overall rank
    merged = merged.sort_values("weighted_avg_ppr", ascending=False).reset_index(drop=True)
    merged.insert(0, "season_rank", merged.index + 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = PROCESSED_DIR / "multi_year_summary.csv"
    merged.to_csv(out_path, index=False)
    log(f"  ✓ Saved {len(merged):,} players to multi_year_summary.csv")

    return merged

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  AI Fantasy Football — Multi-Year Data Processor")
    print(f"  Seasons: {', '.join(SEASONS)}  |  Weights: 2024=50%, 2023=30%, 2022=20%")
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
    breakout_cols = ["full_name","position","team","breakout_score",
                     "surge_score_2024","avg_pts_ppr_2023","avg_pts_ppr_2024",
                     "target_trend_2024","years_exp","trend"]
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
    print(f"   New columns: surge_score_2024, target_trend_2024, breakout_score")

if __name__ == "__main__":
    main()
