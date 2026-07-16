"""
apply_depth_corrections.py
--------------------------
Applies manual WR depth chart corrections from depth_chart_corrections.json
to the depth_charts_{year}.csv files.

For teams listed in the corrections file, the specified players are assigned
depth positions 1, 2, 3... in the order given. All other WRs on that team
are re-numbered starting after the corrected players.

For teams NOT in the corrections file, the original ordering is kept.

Usage:
    python data_pipeline/apply_depth_corrections.py
"""

import json
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
CORRECTIONS_FILE = Path(__file__).parent / "depth_chart_corrections.json"

YEARS = [2024, 2025]


def apply_corrections(year: int, corrections: dict) -> None:
    path = PROCESSED_DIR / f"depth_charts_{year}.csv"
    if not path.exists():
        print(f"  ⚠  depth_charts_{year}.csv not found — skipping")
        return

    df = pd.read_csv(path)
    year_corrections = corrections.get(str(year), {})

    rows_updated = 0

    for team, ordered_names in year_corrections.items():
        # Work only on WRs for this team
        mask = (df["nfl_team"] == team) & (df["fantasy_position"] == "WR")
        team_wrs = df[mask].copy()

        if team_wrs.empty:
            print(f"  ⚠  {year} {team}: no WR rows found in data")
            continue

        # Normalise names to lowercase for matching
        team_wrs["name_lower"] = team_wrs["full_name"].str.lower().str.strip()

        # Assign corrected depth positions to the listed players
        used_indices = []
        for depth_pos, name in enumerate(ordered_names, start=1):
            name_lower = name.lower().strip()
            match = team_wrs[team_wrs["name_lower"] == name_lower]

            if match.empty:
                # Try partial match on last name
                last = name_lower.split()[-1]
                match = team_wrs[team_wrs["name_lower"].str.contains(last, na=False)]

            if not match.empty:
                idx = match.index[0]
                df.at[idx, "depth_position"] = depth_pos
                used_indices.append(idx)
                rows_updated += 1
            else:
                # Player not in data — insert a new row
                new_row = {
                    "full_name":       name,
                    "nfl_team":        team,
                    "fantasy_position": "WR",
                    "depth_position":  depth_pos,
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                rows_updated += 1

        # Re-number the remaining WRs on this team (those not in the corrections list)
        remaining_mask = (df["nfl_team"] == team) & (df["fantasy_position"] == "WR") \
                         & (~df.index.isin(used_indices))
        remaining = df[remaining_mask].sort_values("depth_position")
        next_pos = len(ordered_names) + 1
        for idx in remaining.index:
            df.at[idx, "depth_position"] = next_pos
            next_pos += 1

    # Sort and save
    df = df.sort_values(["nfl_team", "fantasy_position", "depth_position"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"  ✓ {year}: corrections applied ({rows_updated} rows updated) → depth_charts_{year}.csv")


def main():
    print("=" * 60)
    print("  Applying depth chart corrections")
    print("=" * 60)

    with open(CORRECTIONS_FILE) as f:
        corrections = json.load(f)

    for year in YEARS:
        apply_corrections(year, corrections)

    print("\n✅ Done. Re-run data_processor.py to rebuild multi_year_summary.csv.")


if __name__ == "__main__":
    main()
