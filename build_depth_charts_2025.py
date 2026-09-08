"""
build_depth_charts_2025.py
--------------------------
Fetches all 32 NFL teams from ourlads.com archive 304 (12/01/2025)
and writes data/processed/depth_charts_2025.csv

Run from the project root:
    python3 build_depth_charts_2025.py

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
    raise SystemExit(
        "Install dependencies first:\n"
        "  pip3 install requests beautifulsoup4 lxml"
    )

# ── Config ───────────────────────────────────────────────────────────────────
ARCHIVE_ID = 304   # 12/01/2025 snapshot
BASE_URL   = "https://www.ourlads.com/nfldepthcharts/archive/{archive}/{team}"
OUT_CSV    = Path(__file__).parent / "data" / "processed" / "depth_charts_2025.csv"

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

# ourlads code → standard CSV code
TEAMS = {
    "ARZ": "ARI", "ATL": "ATL", "BAL": "BAL", "BUF": "BUF",
    "CAR": "CAR", "CHI": "CHI", "CIN": "CIN", "CLE": "CLE",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GB":  "GB",
    "HOU": "HOU", "IND": "IND", "JAX": "JAX", "KC":  "KC",
    "LAC": "LAC", "LV":  "LV",  "MIA": "MIA", "MIN": "MIN",
    "NE":  "NE",  "NO":  "NO",  "NYG": "NYG", "NYJ": "NYJ",
    "PHI": "PHI", "PIT": "PIT", "LAR": "LAR", "SEA": "SEA",
    "SF":  "SF",  "TB":  "TB",  "TEN": "TEN", "WAS": "WAS",
}

# Positions we care about → fantasy position
POSITION_MAP = {
    "QB":  "QB",
    "RB":  "RB", "FB": "RB",
    "LWR": "WR", "RWR": "WR", "SWR": "WR", "WR": "WR",
    "TE":  "TE", "LTE": "TE", "RTE": "TE",
}

# ── Name parser ──────────────────────────────────────────────────────────────

def parse_player_name(raw: str) -> str:
    """'Murray, Kyler 19/1' → 'Kyler Murray'"""
    raw = raw.strip().rstrip("*").strip()
    # strip suffix tokens like "19/1", "T/Mia", "SF25", "CC/NE", "U/KC"
    name_part = re.sub(r"\s+\S*[0-9/]\S*$", "", raw).strip().strip("*").strip()
    if not name_part or name_part == "-":
        return ""
    if "," in name_part:
        last, first = name_part.split(",", 1)
        return f"{first.strip().title()} {last.strip().title()}"
    return name_part.title()

# ── HTML table parser ────────────────────────────────────────────────────────

def parse_page(html: str, csv_team: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    depth_counters: dict[str, int] = {}

    # Find the main depth chart table (there's usually one big table)
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if not tds:
                continue

            pos_raw = tds[0].get_text(strip=True).upper()
            if pos_raw not in POSITION_MAP:
                continue

            fantasy_pos = POSITION_MAP[pos_raw]

            # Player cells: columns 2, 4, 6, 8, 10 (0-indexed)
            # Table format: Pos | No. | Player1 | No | Player2 | ...
            player_cells = [tds[i].get_text(strip=True) for i in range(2, len(tds), 2)]

            for raw_name in player_cells:
                name = parse_player_name(raw_name)
                if not name:
                    continue
                depth_counters[pos_raw] = depth_counters.get(pos_raw, 0) + 1
                rows.append({
                    "full_name":        name,
                    "nfl_team":         csv_team,
                    "fantasy_position": fantasy_pos,
                    "depth_position":   depth_counters[pos_raw],
                })

    return rows

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_rows = []
    session  = requests.Session()
    session.headers.update(HEADERS)

    for ourlads_code, csv_code in TEAMS.items():
        url = BASE_URL.format(archive=ARCHIVE_ID, team=ourlads_code)
        print(f"  Fetching {ourlads_code} ({csv_code}) ... ", end="", flush=True)

        try:
            resp = session.get(url, timeout=20)
            resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: {e}")
            continue

        rows = parse_page(resp.text, csv_code)
        if not rows:
            print("WARNING: no rows parsed (may be empty table for this team)")
        else:
            print(f"{len(rows)} rows")

        all_rows.extend(rows)
        time.sleep(0.6)   # be polite to the server

    if not all_rows:
        raise SystemExit(
            "\nNo data was parsed.\n"
            "Try: pip3 install lxml  (or replace 'lxml' with 'html.parser' in the script)\n"
            "Also check: curl -I https://www.ourlads.com/ to verify network access."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["full_name", "nfl_team", "fantasy_position", "depth_position"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n✅  Wrote {len(all_rows)} rows → {OUT_CSV}")

    # Spot-check: WR1s
    from collections import Counter
    team_counts = Counter(r["nfl_team"] for r in all_rows)
    print(f"Teams with data: {len(team_counts)}/32")
    wr1s = [r for r in all_rows if r["fantasy_position"] == "WR" and r["depth_position"] == 1]
    print("\nWR1s (depth 1 by sub-position order):")
    for r in sorted(wr1s, key=lambda x: x["nfl_team"]):
        print(f"  {r['nfl_team']:4s}  {r['full_name']}")


if __name__ == "__main__":
    print(f"Building depth_charts_2025.csv from ourlads archive {ARCHIVE_ID} (12/01/2025)...\n")
    main()
