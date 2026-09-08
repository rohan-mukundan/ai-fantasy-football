"""
import_depth_edits_2025.py
--------------------------
After editing depth_charts_2025_EDIT.xlsx, run this script to
save your changes back to data/processed/depth_charts_2025.csv.

Run:  python import_depth_edits_2025.py
"""
import pandas as pd
from pathlib import Path

BASE  = Path(__file__).parent
XLSX  = BASE / "depth_charts_2025_EDIT.xlsx"
OUT   = BASE / "data" / "processed" / "depth_charts_2025.csv"

print("Reading edited Excel...")
wr    = pd.read_excel(XLSX, sheet_name=0, header=0)
other = pd.read_excel(XLSX, sheet_name=1, header=0)

# Normalise column names to match the CSV schema
wr.columns    = ["full_name", "nfl_team", "fantasy_position", "depth_position"]
other.columns = ["full_name", "nfl_team", "fantasy_position", "depth_position"]

combined = pd.concat([wr, other], ignore_index=True)
combined["depth_position"] = combined["depth_position"].astype(int)

combined.to_csv(OUT, index=False)
print(f"✅ Saved {len(combined)} rows → {OUT}")

# Quick sanity check: show WR1s (depth=1) per team
wr1 = combined[(combined["fantasy_position"] == "WR") & (combined["depth_position"] == 1)]
print(f"\nWR1s ({len(wr1)} teams):")
for _, r in wr1.sort_values("nfl_team").iterrows():
    print(f"  {r['nfl_team']:4s}  {r['full_name']}")
