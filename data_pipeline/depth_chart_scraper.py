"""
depth_chart_scraper.py
----------------------
Fetches NFL depth charts for all 32 teams from ourlads.com for a given
archive date, extracts fantasy-relevant positions (QB, RB, WR, TE),
and saves a clean CSV to data/processed/depth_charts_{year}.csv.

The archive URL pattern is:
    https://www.ourlads.com/nfldepthcharts/archive/{archive_id}/{TEAM}

Archive ID 300 = 08/01/2025 (preseason depth charts heading into 2025).

Usage:
    python data_pipeline/depth_chart_scraper.py
"""

import requests
import pandas as pd
import re
import time
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

# Known archive IDs — add more here as needed
ARCHIVES = {
    2024: (287, "08/01/2024"),
    2025: (300, "08/01/2025"),
}

BASE_URL = "https://www.ourlads.com/nfldepthcharts/archive"

# All 32 NFL team abbreviations as used by ourlads
TEAMS = [
    "BUF", "MIA", "NE",  "NYJ",   # AFC East
    "BAL", "CIN", "CLE", "PIT",   # AFC North
    "HOU", "IND", "JAX", "TEN",   # AFC South
    "DEN", "KC",  "LV",  "SD",    # AFC West
    "DAL", "NYG", "PHI", "WAS",   # NFC East
    "CHI", "DET", "GB",  "MIN",   # NFC North
    "ATL", "CAR", "NO",  "TB",    # NFC South
    "ARZ", "RAM", "SF",  "SEA",   # NFC West
]

# Ourlads position codes → fantasy position
# We only keep skill positions relevant to fantasy drafting
POSITION_MAP = {
    "QB":  "QB",
    "RB":  "RB",
    "FB":  "RB",   # Fullbacks count as RB depth
    "LWR": "WR",   # Left wide receiver
    "RWR": "WR",   # Right wide receiver
    "SWR": "WR",   # Slot wide receiver
    "TE":  "TE",
}

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# ── Name parser ────────────────────────────────────────────────────────────────

def parse_player_name(raw: str) -> str:
    """
    Converts ourlads player strings to "First Last" format.

    Input examples:
        "Aiyuk, Brandon 20/1"       → "Brandon Aiyuk"
        "ROBINSON, DEMARCUS U/LAR"  → "Demarcus Robinson"
        "McCaffrey, Christian T/Car"→ "Christian McCaffrey"
        "Taylor Jr., Patrick U/GB"  → "Patrick Taylor Jr."
        "Guerendo, Isaac 24/4"      → "Isaac Guerendo"
    """
    raw = raw.strip().rstrip("*").strip()

    # Remove the designation suffix — anything after the name that contains
    # digits or slashes (e.g. "20/1", "U/LAR", "CF25", "T/Car", "CC/NYJ")
    name_part = re.sub(r"\s+\S*[0-9/]\S*$", "", raw).strip()
    # Clean up any remaining asterisks
    name_part = name_part.strip("*").strip()

    if not name_part:
        return ""

    # Convert "Last, First" → "First Last"
    if "," in name_part:
        parts = name_part.split(",", 1)
        last  = parts[0].strip().title()
        first = parts[1].strip().title()
        return f"{first} {last}"

    return name_part.title()


# ── HTML parser ────────────────────────────────────────────────────────────────

def parse_depth_chart(html: str, team: str) -> list[dict]:
    """
    Parses a raw ourlads depth chart HTML page and returns a list of dicts,
    one per player, with keys: full_name, team, fantasy_position, depth_position.

    depth_position is 1-based within each fantasy position group on the team.
    """
    rows = []

    # Find the depth chart table — it's a standard HTML <table>
    # We use simple regex since BeautifulSoup isn't guaranteed to be installed.
    # Extract all <tr> rows from the main data table.
    table_match = re.search(r"<table[^>]*>(.*?)</table>", html, re.DOTALL | re.IGNORECASE)
    if not table_match:
        return rows

    table_html = table_match.group(1)
    tr_blocks  = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL | re.IGNORECASE)

    # Track depth position per fantasy position on this team
    depth_counters: dict[str, int] = {}

    for tr in tr_blocks:
        # Extract all <td> cell texts, stripping tags and whitespace
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]

        if len(cells) < 3:
            continue

        # First cell is the ourlads position code (e.g. "QB", "LWR", "RB")
        ourlads_pos = cells[0].strip().upper()
        if ourlads_pos not in POSITION_MAP:
            continue

        fantasy_pos = POSITION_MAP[ourlads_pos]

        # Remaining cells alternate: jersey_number, player_name, jersey, player, ...
        # Cells: [pos, no1, player1, no2, player2, no3, player3, ...]
        player_cells = cells[2::2]   # every other cell starting at index 2

        for raw_name in player_cells:
            raw_name = raw_name.strip()
            if not raw_name:
                continue

            name = parse_player_name(raw_name)
            if not name:
                continue

            # Assign depth position (1 = starter, 2 = backup, etc.)
            depth_counters[fantasy_pos] = depth_counters.get(fantasy_pos, 0) + 1
            depth_pos = depth_counters[fantasy_pos]

            rows.append({
                "full_name":       name,
                "nfl_team":        team,
                "fantasy_position": fantasy_pos,
                "depth_position":  depth_pos,
            })

    return rows


# ── Main ───────────────────────────────────────────────────────────────────────

def fetch_all_depth_charts(year: int) -> pd.DataFrame:
    """
    Fetches depth charts for all 32 teams for the given year and returns
    a combined DataFrame. Saves to data/processed/depth_charts_{year}.csv.
    """
    if year not in ARCHIVES:
        print(f"✗ No archive ID configured for {year}. Add it to ARCHIVES in the script.")
        return pd.DataFrame()

    archive_id, archive_date = ARCHIVES[year]

    print("=" * 60)
    print(f"  Ourlads Depth Chart Scraper — {archive_date} ({year})")
    print(f"  Archive ID: {archive_id}  |  Fetching {len(TEAMS)} teams...")
    print("=" * 60)

    all_rows = []
    for team in TEAMS:
        url = f"{BASE_URL}/{archive_id}/{team}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            rows = parse_depth_chart(response.text, team)
            if rows:
                print(f"  ✓ {team}: {len(rows)} players")
                all_rows.extend(rows)
            else:
                print(f"  ✗ {team}: no data parsed")
        except Exception as e:
            print(f"  ✗ {team}: {e}")
        time.sleep(0.5)

    if not all_rows:
        print("\n✗ No data fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"depth_charts_{year}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n✅ Done! {len(df):,} rows saved to depth_charts_{year}.csv")
    return df


if __name__ == "__main__":
    import sys
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    fetch_all_depth_charts(year)
