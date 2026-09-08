"""
build_career_durability.py
--------------------------
Builds career_durability.csv using:
  1. Pre-fetched PFR fantasy pages cached locally (2018–2021)
  2. Existing season_YYYY_summary.csv files (2022–2025)

Outputs:
  data/processed/career_durability.csv
  Columns: full_name, fantasy_position, career_seasons, career_games_played,
           career_games_pct, games_played_2025

Run from the project root:
    python3 build_career_durability.py
"""

import re
import csv
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────

MAX_GAMES = 17   # NFL regular season max (2021+)
# Season-specific max games: 16 games before 2021, 17 from 2021 onward
MAX_GAMES_BY_YEAR: dict[int, int] = {yr: (16 if yr <= 2020 else 17) for yr in range(2016, 2030)}
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE"}

BASE_DIR  = Path(__file__).parent
PROCESSED = BASE_DIR / "data" / "processed"
OUT_CSV   = PROCESSED / "career_durability.csv"

# Cached PFR web-fetch files written during the last Claude session.
# These are plain-text markdown representations of the fantasy.htm pages.
# The parser handles the markdown table format: | Rk | [Player](url) | Tm | FantPos | Age | G | ...
CACHED_PFR_FILES = {
    2021: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788729969351.txt"),
    2020: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788730451389.txt"),
    2019: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788730456584.txt"),
    2018: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788730461676.txt"),
    2017: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788799714934.txt"),
    2016: Path("/var/folders/qf/16_769kn7fld8yb6db8dsnsr0000gn/T/claude-hostloop-plugins/ce7119fe0a59e1d0/projects/-Users-rohanmukundan-Library-Application-Support-Claude-local-agent-mode-sessions-f389ffe9-8edb-46b3-98f6-b4466a485950-ea0115fb-8109-4207-b5ce-69795edf9a4a-local-f5d82b23-465c-41da-b45b-b16f7038b27f-o-24rgbl/63186a6f-ffca-4152-b55f-bcc5183ab97f/tool-results/mcp-workspace-web_fetch-1788799710992.txt"),
}

# ── Parse a single PFR markdown cache file ───────────────────────────────────

def extract_player_name(cell: str) -> str:
    """Extract clean name from '[Name](url)' or plain 'Name' cell."""
    # Match markdown link: [Name](url)
    m = re.search(r'\[([^\]]+)\]', cell)
    name = m.group(1) if m else cell
    # Strip PFR decorators (* = Pro Bowl, + = All-Pro)
    name = name.rstrip("*+").strip()
    return name

def parse_pfr_markdown(path: Path, year: int) -> dict[str, tuple[str, int]]:
    """
    Parse a cached PFR fantasy page (markdown format).
    Returns {full_name: (fantasy_pos, games_played)}.
    Table row format (pipe-separated):
      | Rk | Player | Tm | FantPos | Age | G | GS | ...
    """
    results = {}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue

        rk_cell   = cells[0]
        name_cell = cells[1]
        pos_cell  = cells[3].upper()
        g_cell    = cells[5]

        # Skip header rows (Rk cell is literally "Rk" or "---")
        if rk_cell in ("Rk", "---", "") or "---" in rk_cell:
            continue

        if pos_cell not in FANTASY_POSITIONS:
            continue

        if not g_cell.isdigit():
            continue

        name = extract_player_name(name_cell)
        if not name:
            continue

        games = min(int(g_cell), MAX_GAMES_BY_YEAR.get(year, MAX_GAMES))
        results[name] = (pos_cell, games)

    print(f"  Parsed {year}: {len(results)} players")
    return results

# ── Load existing season summaries (2022–2025) ────────────────────────────────

def load_season_summary(year: int) -> dict[str, tuple[str, int]]:
    """Load {full_name: (position, weeks_played)} from season_YYYY_summary.csv."""
    for fname in [f"season_{year}_summary.csv", f"season_{year}.csv"]:
        p = PROCESSED / fname
        if p.exists():
            df = pd.read_csv(p)
            if "weeks_played" in df.columns and "full_name" in df.columns:
                pos_col = "position" if "position" in df.columns else None
                out = {}
                for _, row in df.iterrows():
                    name = row["full_name"]
                    g    = min(int(row["weeks_played"] or 0), MAX_GAMES_BY_YEAR.get(year, MAX_GAMES))
                    pos  = row[pos_col] if pos_col else ""
                    out[name] = (pos, g)
                print(f"  Loaded season {year} summary: {len(out)} players")
                return out
    print(f"  ⚠  No summary found for {year}")
    return {}

# ── Manual overrides for seasons missing from PFR cache ──────────────────────
# These are known seasons that didn't parse from the cached PFR files.
# Format: {full_name: {year: games_played}}
MANUAL_SEASON_OVERRIDES: dict[str, dict[int, int]] = {
    "Christian McCaffrey": {2020: 3,  2021: 7},   # torn shoulder/ribs; missing from PFR cache
    "George Kittle":       {2020: 8,  2021: 10},  # torn ankle ligaments / calf; missing from PFR cache
    "Saquon Barkley":      {2020: 2},              # torn ACL week 2; missing from PFR cache
}

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # {full_name: {year: games}}
    games_by_player: dict[str, dict[int, int]] = defaultdict(dict)
    pos_by_player:   dict[str, str]             = {}

    # Load 2022–2025 from local season summaries (Sleeper data)
    for yr in [2022, 2023, 2024, 2025]:
        data = load_season_summary(yr)
        for name, (pos, g) in data.items():
            if g > 0:
                games_by_player[name][yr] = g
            if name not in pos_by_player and pos:
                pos_by_player[name] = pos

    # Parse 2018–2021 from cached PFR files
    for yr, path in sorted(CACHED_PFR_FILES.items()):
        if not path.exists():
            print(f"  ⚠  Cached file for {yr} not found — skipping")
            continue
        data = parse_pfr_markdown(path, yr)
        for name, (pos, g) in data.items():
            if g > 0:
                games_by_player[name][yr] = g
            if name not in pos_by_player and pos:
                pos_by_player[name] = pos

    # ── Apply manual overrides for seasons missing from PFR cache ────────────
    for name, season_map in MANUAL_SEASON_OVERRIDES.items():
        for yr, g in season_map.items():
            games_by_player[name][yr] = g
            print(f"  ✎ Manual override: {name} {yr} → {g} games")

    # ── Compute career metrics ────────────────────────────────────────────────
    rows = []
    for name, year_games in games_by_player.items():
        active = {yr: g for yr, g in year_games.items() if g > 0}
        if not active:
            continue

        total_games      = sum(active.values())
        num_seasons      = len(active)
        possible_games   = sum(MAX_GAMES_BY_YEAR.get(yr, MAX_GAMES) for yr in active)
        career_pct       = round(total_games / possible_games, 4)
        games_2025       = year_games.get(2025)
        seasons_under_12 = sum(1 for g in active.values() if g < 12)

        rows.append({
            "full_name":           name,
            "fantasy_position":    pos_by_player.get(name, ""),
            "career_seasons":      num_seasons,
            "career_games_played": total_games,
            "career_games_pct":    career_pct,
            "games_played_2025":   games_2025,
            "seasons_under_12":    seasons_under_12,
        })

    rows.sort(key=lambda r: r["full_name"])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    fieldnames = ["full_name", "fantasy_position", "career_seasons",
                  "career_games_played", "career_games_pct", "games_played_2025",
                  "seasons_under_12"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Wrote {len(rows)} players → {OUT_CSV}")

    # Spot-check
    df = pd.read_csv(OUT_CSV)
    checks = ["Christian McCaffrey", "George Kittle", "Ricky Pearsall",
              "Justin Jefferson", "Davante Adams", "Ja'Marr Chase"]
    print("\nSpot-check:")
    for name in checks:
        row = df[df["full_name"].str.contains(name.split()[-1], case=False)]
        if len(row):
            r = row.iloc[0]
            yrs = r['career_seasons']
            pct = f"{r['career_games_pct']*100:.0f}%"
            g25 = int(r['games_played_2025']) if pd.notna(r['games_played_2025']) else "N/A"
            print(f"  {r['full_name']:25s}  {yrs} seasons  {pct} avail  2025: {g25} games")


if __name__ == "__main__":
    print("Building career_durability.csv...\n")
    main()
