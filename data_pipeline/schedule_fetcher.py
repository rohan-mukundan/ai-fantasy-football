"""
schedule_fetcher.py
--------------------
Fetches the NFL schedule for a given season from ESPN's public scoreboard
API and figures out each team's bye week (the one week 1-18 where that
team doesn't appear in any game).

The result is saved to data/processed/bye_weeks_{season}.json, e.g.:
    {
        "ATL": 5,
        "CHI": 5,
        ...
    }

This only needs to run once per season (during Season Setup) — the saved
file is then read whenever the app needs to check byes for a weekly lineup.

Usage:
    python data_pipeline/schedule_fetcher.py
"""

import requests
import json
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

# All 32 NFL team abbreviations, as used in our player data (multi_year_summary.csv)
TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
]

# ESPN uses a few different abbreviations than our player data does.
# Map ESPN's abbreviation -> our abbreviation.
ESPN_TEAM_MAP = {
    "WSH": "WAS",
}

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"


def _normalize(abbr: str) -> str:
    return ESPN_TEAM_MAP.get(abbr, abbr)


def fetch_bye_weeks(season: str, log=print) -> dict:
    """
    Determines each team's bye week for the given season by checking the
    NFL schedule for weeks 1-18 (regular season) and seeing which teams
    don't have a game that week.

    Saves the result to data/processed/bye_weeks_{season}.json and also
    returns it as a dict of {team_abbr: bye_week}.
    """
    bye_weeks: dict[str, int] = {}

    for week in range(1, 19):
        url = f"{SCOREBOARD_URL}?week={week}&seasontype=2&year={season}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log(f"  ⚠ Week {week}: failed to fetch schedule ({e})")
            continue

        playing_teams = set()
        for event in data.get("events", []):
            for competition in event.get("competitions", []):
                for competitor in competition.get("competitors", []):
                    abbr = competitor.get("team", {}).get("abbreviation")
                    if abbr:
                        playing_teams.add(_normalize(abbr))

        byes_this_week = [t for t in TEAMS if t not in playing_teams]
        if byes_this_week:
            log(f"  Week {week}: bye teams — {', '.join(byes_this_week)}")
        for team in byes_this_week:
            bye_weeks[team] = week

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"bye_weeks_{season}.json"
    with open(out_path, "w") as f:
        json.dump(bye_weeks, f, indent=2)

    log(f"  ✓ Saved bye weeks for {len(bye_weeks)} teams → bye_weeks_{season}.json")
    return bye_weeks


if __name__ == "__main__":
    import datetime
    season = str(datetime.datetime.now().year)
    fetch_bye_weeks(season)
