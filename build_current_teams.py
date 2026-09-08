"""
build_current_teams.py
----------------------
Fetches CURRENT ourlads depth charts for all 32 teams and writes
data/processed/current_teams.csv with name → team mappings.

Run whenever you want fresh team data (updated daily by ourlads):
    python3 build_current_teams.py

Then re-run data_pipeline/data_processor.py to apply the changes.
Does NOT touch depth_charts_2026.csv — only updates team assignments.

Requires: pip3 install requests beautifulsoup4 lxml
"""

import re
import time
import csv
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("Install dependencies:\n  pip3 install requests beautifulsoup4 lxml")

# ── Config ────────────────────────────────────────────────────────────────────

OUT_CSV = Path(__file__).parent / "data" / "processed" / "current_teams.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.ourlads.com/",
}

# ourlads URL code → standard team code used in our CSVs
TEAMS = {
    "ARZ": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB":  "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC":  "KC",
    "LAC": "LAC", "RAM": "LAR", "LV":  "LV",  "MIA": "MIA",
    "MIN": "MIN", "NE":  "NE",  "NO":  "NO",  "NYG": "NYG",
    "NYJ": "NYJ", "PHI": "PHI", "PIT": "PIT", "SEA": "SEA",
    "SF":  "SF",  "TB":  "TB",  "TEN": "TEN", "WAS": "WAS",
}

POSITION_MAP = {
    "QB": "QB",
    "RB": "RB", "FB": "RB",
    "LWR": "WR", "RWR": "WR", "SWR": "WR", "WR": "WR",
    "TE": "TE", "LTE": "TE", "RTE": "TE",
    "PK": "K",
}

# ── Name parser (same as build_depth_charts_2025.py) ─────────────────────────

def parse_player_name(raw: str) -> str:
    """'Murray, Kyler 19/1' → 'Kyler Murray'"""
    raw = raw.strip().rstrip("*").strip()
    name_part = re.sub(r"\s+\S*[0-9/]\S*$", "", raw).strip().strip("*").strip()
    if not name_part or name_part == "-":
        return ""
    if "," in name_part:
        last, first = name_part.split(",", 1)
        return f"{first.strip().title()} {last.strip().title()}"
    return name_part.title()

# ── Fetch one team ────────────────────────────────────────────────────────────

def fetch_team(ourlads_code: str, csv_code: str, session: requests.Session) -> list[dict]:
    url = f"https://www.ourlads.com/nfldepthcharts/depthchart/{ourlads_code}"
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ⚠  {ourlads_code}: fetch failed ({e})")
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    players = []
    seen = set()

    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue
            pos_raw = tds[0].get_text(strip=True).upper()
            if pos_raw not in POSITION_MAP:
                continue
            fantasy_pos = POSITION_MAP[pos_raw]
            player_cells = [tds[i].get_text(strip=True) for i in range(2, len(tds), 2)]
            for raw_name in player_cells:
                name = parse_player_name(raw_name)
                if name and name not in seen:
                    seen.add(name)
                    players.append({"full_name": name, "team": csv_code, "position": fantasy_pos})

    return players

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Fetching current ourlads depth charts for all 32 teams...\n")
    all_players: dict[str, dict] = {}
    session = requests.Session()
    session.headers.update(HEADERS)

    for ourlads_code, csv_code in TEAMS.items():
        players = fetch_team(ourlads_code, csv_code, session)
        print(f"  {csv_code:4s}  {len(players)} skill-position players")
        for p in players:
            # Keep first occurrence (depth 1 player wins if name appears twice)
            if p["full_name"] not in all_players:
                all_players[p["full_name"]] = p
        time.sleep(0.6)

    if not all_players:
        raise SystemExit("No data fetched — check your internet connection.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(all_players.values(), key=lambda r: r["full_name"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["full_name", "team", "position"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Wrote {len(rows)} players → {OUT_CSV}")

    # Spot-check (case-insensitive so "DJ"/"Dj", "McCaffrey"/"Mccaffrey" etc. match)
    checks = ["Patrick Mahomes", "CeeDee Lamb", "Joe Mixon", "Tyreek Hill",
              "Daniel Carlson", "Keenan Allen", "Darren Waller", "Kenneth Gainwell",
              "Christian McCaffrey", "DJ Moore"]
    print("\nSpot-check:")
    name_map_lower = {r["full_name"].lower(): r for r in rows}
    for name in checks:
        r = name_map_lower.get(name.lower())
        if r:
            print(f"  {name:25s}  {r['team']}  {r['position']}")
        else:
            print(f"  {name:25s}  NOT FOUND (likely FA/retired)")

if __name__ == "__main__":
    main()
