"""
transactions_scraper.py
-----------------------
Fetches NFL transaction data from ESPN's public JSON API and identifies
offseason moves that affect fantasy football player values.

Think of this like a news wire service: the NFL makes hundreds of moves
every offseason (signings, trades, releases), and this scraper filters
that fire hose down to only the moves that matter for drafting.

Outputs:
  data/processed/transaction_impacts.csv — one row per relevant move

Columns:
  player_name      — matched name from our database (or raw ESPN name)
  espn_name        — name exactly as ESPN wrote it
  position         — QB / RB / WR / TE / K
  transaction_type — RELEASED, TRADED, SIGNED, RE_SIGNED, RETIRED, IR,
                     PRACTICE_SQUAD, ROOKIE_SIGNED, CLAIMED, OTHER
  team             — the team handling the transaction (per ESPN)
  from_team        — previous team (for trades / signings from elsewhere)
  to_team          — new team (for trades / incoming signings)
  date             — transaction date (YYYY-MM-DD)
  fantasy_impact   — HIGH / MEDIUM / LOW / NONE
  impact_note      — short human-readable explanation
  boost            — integer 0–20: how much to add to breakout_score
  raw_description  — the full raw text from ESPN

Usage (standalone):
  python data_pipeline/transactions_scraper.py

Usage (from app.py):
  from data_pipeline.transactions_scraper import run_transaction_pipeline
  summary = run_transaction_pipeline()  # returns dict of stats
"""

import re
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR     = Path(__file__).resolve().parent.parent
PROCESSED    = ROOT_DIR / "data" / "processed"
SUMMARY_PATH = PROCESSED / "multi_year_summary.csv"
OUTPUT_PATH  = PROCESSED / "transaction_impacts.csv"

# ── ESPN API config ───────────────────────────────────────────────────────────

ESPN_BASE_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/nfl/transactions"
)
# How many transactions per API page (ESPN max is 500).
# We use 200 as a safe middle ground.
PAGE_LIMIT = 200

# Positions we care about for fantasy (the ones we draft)
FANTASY_POSITIONS = {"QB", "RB", "WR", "TE", "K"}

# ── Transaction type keywords ─────────────────────────────────────────────────
# Each entry: (transaction_type_string, list_of_keywords_that_imply_it)
# Checked in order — first match wins.

TX_TYPE_RULES = [
    ("RETIRED",        ["reserve/retired", "retirement", "retired", "announced retirement"]),
    ("IR",             ["reserve/injured", "reserve/non-football", "reserve/pup",
                        "injured reserve", "/ir ", "placed on ir"]),
    ("SUSPENDED",      ["reserve/suspended", "suspended"]),
    ("TRADED",         ["traded", "trade"]),
    ("RELEASED",       ["waived", "released", "cut "]),
    ("RE_SIGNED",      ["re-signed", "extended his contract", "extension"]),
    ("ROOKIE_SIGNED",  ["rookie contract", "rookie deal"]),
    ("SIGNED",         ["signed", "agreed to", "agreement"]),
    ("CLAIMED",        ["claimed"]),
    ("ACTIVATED",      ["activated"]),
    ("PRACTICE_SQUAD", ["practice squad", "practice-squad"]),
]

# ── Name normalization helpers ────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """
    Strip periods, trim whitespace, and lowercase a player name for comparison.
    'D.K. Metcalf' → 'dk metcalf'
    'T.J. Hockenson' → 'tj hockenson'
    'Josh Allen Jr.' → 'josh allen'   (suffixes also stripped)
    """
    # Remove name suffixes (Jr., Sr., II, III, IV, V)
    name = re.sub(r"\s+(Jr\.?|Sr\.?|II|III|IV|V)$", "", name.strip(), flags=re.I)
    # Remove all periods and extra whitespace, then lowercase
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def normalize_team(team_str: str) -> str:
    """
    Normalize ESPN team names to short 2-3 letter codes where possible.
    ESPN often uses city names like 'New York Jets'; this maps them to 'NYJ'.
    """
    if not team_str:
        return ""

    # Common ESPN full names → short codes
    team_map = {
        "arizona cardinals": "ARI", "atlanta falcons": "ATL",
        "baltimore ravens": "BAL", "buffalo bills": "BUF",
        "carolina panthers": "CAR", "chicago bears": "CHI",
        "cincinnati bengals": "CIN", "cleveland browns": "CLE",
        "dallas cowboys": "DAL", "denver broncos": "DEN",
        "detroit lions": "DET", "green bay packers": "GB",
        "houston texans": "HOU", "indianapolis colts": "IND",
        "jacksonville jaguars": "JAX", "kansas city chiefs": "KC",
        "las vegas raiders": "LV", "los angeles chargers": "LAC",
        "los angeles rams": "LAR", "miami dolphins": "MIA",
        "minnesota vikings": "MIN", "new england patriots": "NE",
        "new orleans saints": "NO", "new york giants": "NYG",
        "new york jets": "NYJ", "n.y. giants": "NYG",
        "n.y. jets": "NYJ", "philadelphia eagles": "PHI",
        "pittsburgh steelers": "PIT", "san francisco 49ers": "SF",
        "seattle seahawks": "SEA", "tampa bay buccaneers": "TB",
        "tennessee titans": "TEN", "washington commanders": "WAS",
    }

    key = team_str.strip().lower()
    return team_map.get(key, team_str.strip())

# ── Core parsing functions ────────────────────────────────────────────────────

def classify_transaction_type(sentence: str) -> str:
    """
    Return the transaction type for a single sentence from a description.
    E.g. 'Released WR Deven Thompkins.' → 'RELEASED'
    """
    s = sentence.lower()
    for tx_type, keywords in TX_TYPE_RULES:
        if any(kw in s for kw in keywords):
            return tx_type
    return "OTHER"


def extract_player_moves_from_description(description: str, team_name: str) -> list[dict]:
    """
    Parse one ESPN transaction description into zero or more player-move dicts.

    ESPN descriptions often bundle multiple moves in one text block, separated
    by '. ' — e.g. 'Claimed WR X from NYJ. Released WR Y.'

    For each FANTASY_POSITIONS player found, we return a dict with:
      espn_name, position, transaction_type, team, from_team, to_team, raw_sentence
    """
    # Pattern: captures position abbreviation immediately followed by a player name.
    # The name is 2–4 capitalized words (handles 'D.K. Metcalf', 'Amon-Ra St. Brown',
    # "D'Andre Swift", 'Josh Allen III', etc.)
    name_pattern = re.compile(
        r"\b(QB|RB|WR|TE|K)\b\s+"            # position tag
        r"((?:[A-Z][a-zA-Z'.-]*\.?\s*){2,4})"  # 2–4 name words starting with uppercase
    )

    # Split on '. ' only when followed by a capital letter (= new sentence/move).
    # This avoids splitting on 'N.Y. Jets' or 'Jr.'
    sentences = re.split(r"\.\s+(?=[A-Z])", description)

    moves = []
    for sentence in sentences:
        sentence = sentence.strip().rstrip(".")
        if not sentence:
            continue

        tx_type = classify_transaction_type(sentence)

        for m in name_pattern.finditer(sentence):
            position = m.group(1)
            if position not in FANTASY_POSITIONS:
                continue

            raw_name = m.group(2).strip()
            # Remove trailing single-word suffixes (Jr, Sr) if they slipped through
            raw_name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)$", "", raw_name, flags=re.I).strip()

            # Determine from/to team based on transaction type + sentence content
            from_team = ""
            to_team   = ""
            s_lower   = sentence.lower()
            team_norm = normalize_team(team_name)

            if tx_type in ("RELEASED", "RETIRED", "SUSPENDED", "IR"):
                # The hosting team released/IR'd this player
                from_team = team_norm
                to_team   = "FA" if tx_type == "RELEASED" else ""
            elif tx_type == "TRADED":
                # Try to detect direction: 'traded X to [team]' vs 'acquired X from [team]'
                if "acquired" in s_lower or "received" in s_lower:
                    from_team = _extract_other_team(sentence, team_norm)
                    to_team   = team_norm
                else:
                    from_team = team_norm
                    to_team   = _extract_other_team(sentence, team_norm)
            elif tx_type == "CLAIMED":
                # 'Claimed X off waivers from [team]'
                from_team = _extract_other_team(sentence, team_norm)
                to_team   = team_norm
            elif tx_type in ("SIGNED", "RE_SIGNED", "ROOKIE_SIGNED"):
                # Player joining this team
                from_team = _extract_other_team(sentence, team_norm)
                to_team   = team_norm
            elif tx_type == "PRACTICE_SQUAD":
                from_team = ""
                to_team   = team_norm
            else:
                to_team = team_norm

            moves.append({
                "espn_name":        raw_name,
                "position":         position,
                "transaction_type": tx_type,
                "team":             team_norm,
                "from_team":        from_team,
                "to_team":          to_team,
                "raw_sentence":     sentence,
            })

    return moves


def _extract_other_team(sentence: str, current_team_norm: str) -> str:
    """
    Try to extract a second team name mentioned in a sentence.
    E.g. 'Claimed X off waivers from the N.Y. Jets.' → 'NYJ'
    Falls back to '' if we can't parse it.
    """
    # Look for common patterns like 'from [Team]', 'to the [Team]'
    m = re.search(
        r"(?:from|to) (?:the )?([A-Z][a-zA-Z.\s]{2,30}?)(?:[,.]|$)",
        sentence,
        re.IGNORECASE
    )
    if m:
        candidate = m.group(1).strip()
        norm = normalize_team(candidate)
        if norm != current_team_norm:
            return norm
    return ""

# ── Fantasy impact assessment ─────────────────────────────────────────────────

def assess_impact(tx_type: str, position: str, in_db: bool) -> tuple[str, str, int]:
    """
    Returns (fantasy_impact, impact_note, boost).

    fantasy_impact : 'HIGH' / 'MEDIUM' / 'LOW' / 'NONE'
    impact_note    : short text shown in the draft UI (max ~60 chars)
    boost          : integer added to breakout_score (0–20)

    Logic (business analogy: think of each player as a department):
    - RETIRED/IR = the employee quit or went on long-term leave → big hole for teammates
    - RELEASED  = employee let go, now freelancing → unstable, avoid until re-hired
    - TRADED    = transferred to a new office → new opportunity, uncertain fit
    - SIGNED    = hired away from another team → may step into a starting role
    - RE_SIGNED = renewed contract → role stability confirmed
    - ROOKIE    = new hire fresh out of college → upside unknown, low immediate fantasy value
    """
    if tx_type == "RETIRED":
        impact = "HIGH" if in_db else "MEDIUM"
        note   = "Retired — targets/carries shift to teammates"
        boost  = 0   # The retiree gets 0; teammates benefit (handled in data_processor)
    elif tx_type == "IR":
        impact = "HIGH" if in_db else "LOW"
        note   = "Injured reserve — unavailable; targets shift to teammates"
        boost  = 0
    elif tx_type == "SUSPENDED":
        impact = "MEDIUM"
        note   = "Suspended — uncertain availability"
        boost  = 0
    elif tx_type == "RELEASED":
        impact = "MEDIUM" if in_db else "LOW"
        note   = "Released — now a free agent; avoid until signed"
        boost  = 0
    elif tx_type == "TRADED":
        impact = "MEDIUM" if in_db else "LOW"
        note   = "Traded to new team — evaluate new scheme fit"
        boost  = 5   # New team often = new role/opportunity
    elif tx_type == "CLAIMED":
        impact = "LOW"
        note   = "Claimed off waivers — backup/depth role"
        boost  = 0
    elif tx_type == "SIGNED":
        impact = "MEDIUM" if in_db else "LOW"
        note   = "Signed as free agent — may step into starting role"
        boost  = 8   # FA signing often = team needed someone at that position
    elif tx_type == "RE_SIGNED":
        impact = "LOW"
        note   = "Re-signed — confirms current role/depth"
        boost  = 3   # Stability confirmation
    elif tx_type == "ROOKIE_SIGNED":
        impact = "LOW"
        note   = "Rookie contract signed — ceiling play, limited near-term value"
        boost  = 0
    elif tx_type == "PRACTICE_SQUAD":
        impact = "NONE"
        note   = "Practice squad — no fantasy value"
        boost  = 0
    elif tx_type == "ACTIVATED":
        impact = "LOW"
        note   = "Activated from reserve"
        boost  = 2
    else:
        impact = "NONE"
        note   = ""
        boost  = 0

    # Cap boost to 20 (the total breakout_score is 0–100, so this is meaningful but bounded)
    boost = min(boost, 20)
    return impact, note, boost

# ── Database cross-reference ──────────────────────────────────────────────────

def build_name_index(players_df: pd.DataFrame) -> dict[str, str]:
    """
    Build a dict of {normalized_name → canonical_full_name} from our player DB.

    Also pre-computes a 'first_initial + last_name' secondary key so we can
    handle cases where ESPN omits a middle name we track (or vice-versa).
    """
    index = {}
    for _, row in players_df.iterrows():
        canon = row["full_name"]
        norm  = normalize_name(canon)
        index[norm] = canon

        # Secondary key: first initial + last word (e.g. "j barkley" for "Saquon Barkley")
        parts = norm.split()
        if len(parts) >= 2:
            secondary = parts[0][0] + " " + parts[-1]
            # Only store secondary if not already used by another player
            if secondary not in index:
                index[secondary] = canon

    return index


def find_in_database(espn_name: str, name_index: dict) -> str | None:
    """
    Look up an ESPN player name in our database.
    Returns the canonical full_name string, or None if not found.

    Tries 3 strategies:
    1. Exact normalized match   ('dk metcalf' == 'dk metcalf')
    2. Abbreviated-first-name   ('d metcalf' → check all 'd ___talf' entries)
    3. None if no match found
    """
    norm = normalize_name(espn_name)

    # Strategy 1: direct normalized match
    if norm in name_index:
        return name_index[norm]

    # Strategy 2: first-initial + last-name
    parts = norm.split()
    if len(parts) >= 2:
        short_key = parts[0][0] + " " + parts[-1]
        if short_key in name_index:
            return name_index[short_key]

    # Strategy 3: partial — last name exact + first initial match
    if len(parts) >= 2:
        first_init = parts[0][0]
        last       = parts[-1]
        for key, canon in name_index.items():
            key_parts = key.split()
            if len(key_parts) >= 2 and key_parts[-1] == last and key_parts[0][0] == first_init:
                return canon

    return None

# ── Depth-chart-based transaction detector ────────────────────────────────────
#
# Instead of scraping ESPN or PFR (both have blocked access), we derive
# transaction data directly from our existing depth chart CSVs.
#
# The logic is simple:
#   Player in 2024 depth chart but NOT in 2025  → left the NFL (released/retired)
#   Player in both but on DIFFERENT team        → changed teams (traded/signed)
#   Player NEW in 2025                          → rookie or new signing
#
# This is actually more accurate than parsing ESPN text, because the depth
# charts represent the CURRENT roster state rather than raw announcement text.

DC_PATH_TEMPLATE = PROCESSED / "depth_charts_{year}.csv"

DEPTH_POSITIONS_RELEVANT = {"WR", "TE", "RB", "QB", "K"}


def load_depth_chart(year: str) -> pd.DataFrame:
    path = DC_PATH_TEMPLATE.with_name(f"depth_charts_{year}.csv")
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalize names the same way the summary does
    suffix_pattern = r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$'
    df["full_name"] = df["full_name"].str.replace(suffix_pattern, "", regex=True).str.strip()
    return df


def generate_impacts_from_depth_charts(
    summary_df: pd.DataFrame,
    from_year:  str = "2024",
    to_year:    str = "2025",
) -> pd.DataFrame:
    """
    Compare two seasons of depth charts and generate transaction impacts.

    Returns a DataFrame with the same schema as parse_transactions() so all
    downstream logic (target rank shifts, breakout boosts, UI display) works
    identically regardless of data source.
    """
    dc_old = load_depth_chart(from_year)
    dc_new = load_depth_chart(to_year)

    if dc_old.empty or dc_new.empty:
        print(f"  ⚠  Could not load depth charts for {from_year} or {to_year}")
        return pd.DataFrame()

    print(f"\nComparing depth charts {from_year} → {to_year}...")

    # Position column may be 'fantasy_position' or 'position'
    pos_col_old = "fantasy_position" if "fantasy_position" in dc_old.columns else "position"
    pos_col_new = "fantasy_position" if "fantasy_position" in dc_new.columns else "position"

    # Filter to fantasy-relevant positions
    dc_old = dc_old[dc_old[pos_col_old].isin(DEPTH_POSITIONS_RELEVANT)].copy()
    dc_new = dc_new[dc_new[pos_col_new].isin(DEPTH_POSITIONS_RELEVANT)].copy()

    # Build lookup dicts: name → {team, position, depth}
    def to_lookup(df, pos_col):
        out = {}
        for _, row in df.iterrows():
            out[row["full_name"]] = {
                "team":  row["nfl_team"],
                "pos":   row[pos_col],
                "depth": int(row["depth_position"]),
            }
        return out

    old_lookup = to_lookup(dc_old, pos_col_old)
    new_lookup = to_lookup(dc_new, pos_col_new)

    # Avg targets lookup from summary (use most recent season available)
    tgt_season = _find_best_target_season(summary_df)
    tgt_col    = f"avg_targets_pg_{tgt_season}" if tgt_season else None

    tgt_lookup: dict[str, float] = {}
    if tgt_col and tgt_col in summary_df.columns:
        for _, row in summary_df.iterrows():
            if pd.notna(row.get(tgt_col, None)):
                tgt_lookup[row["full_name"]] = float(row[tgt_col])

    # Also build a name index for cross-referencing
    name_index = build_name_index(summary_df)

    rows = []
    today = datetime.now().strftime("%Y-%m-%d")

    # ── 1. Players who LEFT between seasons (in old but not in new) ───────────
    for name, info in old_lookup.items():
        if name in new_lookup:
            continue
        pos   = info["pos"]
        depth = info["depth"]
        team  = info["team"]

        # Only care about positions we draft
        if pos not in DEPTH_POSITIONS_RELEVANT:
            continue

        # Impact depends on how prominent they were
        # Depth 1-2 on their team = starting role lost (big void)
        # Depth 3-4 = rotational player
        # Depth 5+  = depth/practice squad, low impact
        if depth <= 2:
            impact, note, boost = "HIGH", "Left NFL (retired/cut/IR) — starting targets available", 0
        elif depth <= 4:
            impact, note, boost = "MEDIUM", "Left NFL — rotational targets available", 0
        else:
            impact, note, boost = "LOW", "Depth player departed", 0

        # Only include if they actually played enough to matter
        hist_tgt = tgt_lookup.get(name, 0)
        if hist_tgt < 2.0 and depth > 4:
            continue   # truly fringe, skip

        canon = find_in_database(name, name_index) or name
        rows.append({
            "player_name":      canon,
            "espn_name":        name,
            "position":         pos,
            "transaction_type": "RELEASED",  # generic: left the team
            "team":             team,
            "from_team":        team,
            "to_team":          "",
            "date":             today,
            "fantasy_impact":   impact,
            "impact_note":      note,
            "boost":            boost,
            "in_database":      canon != name or name in {r["full_name"] for _, r in summary_df.iterrows() if True},
            "raw_description":  f"Not in {to_year} depth chart (was {from_year} depth #{depth} on {team})",
        })

    # ── 2. Players who CHANGED TEAMS ─────────────────────────────────────────
    for name, new_info in new_lookup.items():
        if name not in old_lookup:
            continue
        old_info = old_lookup[name]
        if old_info["team"] == new_info["team"]:
            continue   # same team, not a move

        pos       = new_info["pos"]
        old_team  = old_info["team"]
        new_team  = new_info["team"]
        new_depth = new_info["depth"]

        if pos not in DEPTH_POSITIONS_RELEVANT:
            continue

        # A player joining a new team at depth 1-2 = clear role → bigger boost
        if new_depth <= 2:
            impact, note, boost = "MEDIUM", f"Joined {new_team} — stepping into starting role", 8
        else:
            impact, note, boost = "LOW", f"Joined {new_team} — depth role", 2

        canon = find_in_database(name, name_index) or name
        rows.append({
            "player_name":      canon,
            "espn_name":        name,
            "position":         pos,
            "transaction_type": "TRADED",
            "team":             new_team,
            "from_team":        old_team,
            "to_team":          new_team,
            "date":             today,
            "fantasy_impact":   impact,
            "impact_note":      note,
            "boost":            boost,
            "in_database":      True,
            "raw_description":  f"Team changed: {old_team} ({from_year} depth #{old_info['depth']}) → {new_team} ({to_year} depth #{new_depth})",
        })

    if not rows:
        print("  ⚠  No transaction impacts detected from depth chart comparison")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Mark in_database properly using the name index
    df["in_database"] = df["player_name"].isin(set(summary_df["full_name"].values))

    # Deduplicate by player (keep highest-impact row)
    impact_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
    df["_ord"] = df["fantasy_impact"].map(impact_order)
    df = df.sort_values("_ord").drop_duplicates(subset=["player_name"], keep="first")
    df = df.drop(columns=["_ord"]).reset_index(drop=True)

    departed  = (df["transaction_type"] == "RELEASED").sum()
    moved     = (df["transaction_type"] == "TRADED").sum()
    in_db_ct  = df["in_database"].sum()
    print(f"  ✓ {departed} players departed, {moved} changed teams "
          f"({in_db_ct} matched to our player database)")

    return df


# ── ESPN API fetcher (kept for reference but replaced by depth-chart approach)

def fetch_all_transactions(season_year: int | None = None) -> list[dict]:
    """
    Fetch all NFL transactions from ESPN's JSON API, with pagination.

    ESPN's endpoint returns a 'total' count; we page through until we've
    collected all items. With 851 2025-offseason transactions at 200/page,
    that's 5 requests.

    Returns a flat list of raw ESPN transaction dicts.
    """
    params = {"limit": PAGE_LIMIT}
    if season_year:
        params["season"] = season_year

    all_items = []
    page = 1
    total_pages = None

    # ESPN's site.api.espn.com only serves requests that look like they
    # originate from ESPN's own pages — it checks User-Agent, Referer,
    # and Origin headers. We use a Session so cookies from the ESPN
    # homepage carry over to the API call.
    BROWSER_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":           "application/json, text/plain, */*",
        "Accept-Language":  "en-US,en;q=0.9",
        "Accept-Encoding":  "gzip, deflate, br",
        "Referer":          "https://www.espn.com/nfl/transactions",
        "Origin":           "https://www.espn.com",
        "Sec-Fetch-Dest":   "empty",
        "Sec-Fetch-Mode":   "cors",
        "Sec-Fetch-Site":   "same-site",
        "Connection":       "keep-alive",
    }

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    # Warm up the session by visiting ESPN's homepage first — this sets
    # cookies that the API endpoint may require to verify the request
    # is coming from a real browser session on ESPN.
    print(f"Fetching NFL transactions from ESPN API...")
    try:
        session.get("https://www.espn.com/nfl/transactions", timeout=10)
    except Exception:
        pass  # best-effort cookie grab; proceed either way

    while True:
        params["page"] = page

        try:
            resp = session.get(ESPN_BASE_URL, params=params, timeout=15)
            print(f"  HTTP {resp.status_code} on page {page}")
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            print(f"  ⚠  Page {page} timed out — stopping pagination")
            break
        except requests.exceptions.RequestException as e:
            print(f"  ✗  Request error on page {page}: {e}")
            break

        try:
            data = resp.json()
        except json.JSONDecodeError:
            print(f"  ✗  Could not parse JSON from page {page} (response: {resp.text[:200]})")
            break

        items = data.get("items", [])
        if not items:
            break

        all_items.extend(items)

        # On first page, figure out how many pages there are
        if total_pages is None:
            total = data.get("count", 0)
            total_pages = (total + PAGE_LIMIT - 1) // PAGE_LIMIT
            season_info = data.get("requestedYear", {}).get("displayName", "")
            print(f"  Found {total:,} total transactions for {season_info} "
                  f"({total_pages} pages)")

        print(f"  ✓ Page {page}/{total_pages} — {len(items)} items "
              f"(running total: {len(all_items)})")

        if page >= total_pages:
            break

        page += 1
        time.sleep(0.3)  # polite pause between API calls

    return all_items

# ── Main parsing pass ─────────────────────────────────────────────────────────

def parse_transactions(
    items:      list[dict],
    players_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process raw ESPN items into a clean DataFrame of fantasy-relevant moves.

    Steps:
    1. Extract team + description + date from each ESPN item
    2. Parse description → individual player moves (may be many per item)
    3. Filter to FANTASY_POSITIONS only
    4. Cross-reference each name against our player database
    5. Assess fantasy impact and compute breakout boost
    """
    print(f"\nParsing {len(items):,} ESPN transactions...")

    # Build lookup index from our player database
    name_index = build_name_index(players_df)

    rows = []
    n_in_db = 0
    n_not_in_db = 0

    for item in items:
        team_name   = item.get("team", {}).get("displayName", "")
        description = item.get("description", "").strip()
        date_raw    = item.get("date", "")

        # Parse date to YYYY-MM-DD
        if date_raw:
            try:
                # ESPN returns ISO 8601 like '2025-03-10T00:00Z'
                date_str = datetime.strptime(date_raw[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
            except ValueError:
                date_str = date_raw[:10]
        else:
            date_str = ""

        # Skip items with no useful description
        if not description:
            continue

        # Parse description → player moves
        moves = extract_player_moves_from_description(description, team_name)

        for mv in moves:
            pos      = mv["position"]
            espn_nm  = mv["espn_name"]
            tx_type  = mv["transaction_type"]

            # Skip practice squad moves and generic 'OTHER' with no real impact
            if tx_type in ("PRACTICE_SQUAD", "OTHER", "ACTIVATED"):
                continue

            # Cross-reference with our player database
            canon_name = find_in_database(espn_nm, name_index)
            in_db = canon_name is not None

            if in_db:
                n_in_db += 1
            else:
                n_not_in_db += 1

            # Assess the fantasy impact of this move
            impact, note, boost = assess_impact(tx_type, pos, in_db)

            if impact == "NONE":
                continue   # Skip truly irrelevant moves (e.g. 7th-round rookies)

            rows.append({
                "player_name":      canon_name or espn_nm,
                "espn_name":        espn_nm,
                "position":         pos,
                "transaction_type": tx_type,
                "team":             mv["team"],
                "from_team":        mv["from_team"],
                "to_team":          mv["to_team"],
                "date":             date_str,
                "fantasy_impact":   impact,
                "impact_note":      note,
                "boost":            boost,
                "in_database":      in_db,
                "raw_description":  description,
            })

    # Deduplicate: keep only the most recent/impactful transaction per player
    if rows:
        df = pd.DataFrame(rows)

        # Sort by date descending and impact (HIGH > MEDIUM > LOW)
        impact_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "NONE": 3}
        df["_impact_ord"] = df["fantasy_impact"].map(impact_order)
        df = df.sort_values(["date", "_impact_ord"], ascending=[False, True])

        # For each player, keep only the single most significant recent move
        df = df.drop_duplicates(subset=["player_name"], keep="first")
        df = df.drop(columns=["_impact_ord"]).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=[
            "player_name", "espn_name", "position", "transaction_type",
            "team", "from_team", "to_team", "date", "fantasy_impact",
            "impact_note", "boost", "in_database", "raw_description"
        ])

    in_db_count = df["in_database"].sum() if not df.empty else 0
    print(f"  ✓ Found {len(df):,} unique fantasy-relevant moves "
          f"({in_db_count} matched to our player database)")
    print(f"  ℹ  {n_in_db} total matches in DB | {n_not_in_db} unknown players (ignored)")

    return df

# ── Apply boosts to multi_year_summary.csv ────────────────────────────────────

def apply_transaction_impacts(
    summary_df: pd.DataFrame,
    impacts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge transaction data into the multi-year summary and apply boosts.

    New columns added to summary_df:
      transaction_type — the most recent significant move (e.g. 'TRADED')
      transaction_note — short text the DraftAgent reads (e.g. 'Traded to KC')
      breakout_score   — existing column, boosted by impacts_df['boost']

    Think of this like updating an employee performance score (breakout_score)
    after a big personnel event (transaction). A trade to a better team is like
    a promotion — it boosts the upside signal the draft agent uses.
    """
    if impacts_df.empty:
        print("  ⚠  No transaction impacts to apply")
        # Still add the columns so downstream code doesn't KeyError
        summary_df["transaction_type"] = ""
        summary_df["transaction_note"] = ""
        return summary_df

    # Keep only in-database players and meaningful moves
    relevant = impacts_df[impacts_df["in_database"]].copy()

    # Build a mapping: canonical player name → (tx_type, note, boost)
    tx_map = {}
    for _, row in relevant.iterrows():
        tx_map[row["player_name"]] = {
            "transaction_type": row["transaction_type"],
            "transaction_note": row["impact_note"],
            "boost":            row["boost"],
        }

    # Apply to summary
    def _get_tx_type(name):
        return tx_map.get(name, {}).get("transaction_type", "")

    def _get_tx_note(name):
        return tx_map.get(name, {}).get("transaction_note", "")

    def _get_boost(name):
        return tx_map.get(name, {}).get("boost", 0)

    summary_df["transaction_type"] = summary_df["full_name"].apply(_get_tx_type)
    summary_df["transaction_note"] = summary_df["full_name"].apply(_get_tx_note)

    # Add boost to existing breakout_score (capped at 100)
    boost_series = summary_df["full_name"].apply(_get_boost)
    if "breakout_score" in summary_df.columns:
        summary_df["breakout_score"] = (
            summary_df["breakout_score"].fillna(0) + boost_series
        ).clip(upper=100)

    # How many players got boosts?
    boosted = (boost_series > 0).sum()
    flagged = (summary_df["transaction_type"] != "").sum()
    print(f"  ✓ Transaction impacts applied: {flagged} players flagged, "
          f"{boosted} received breakout_score boost")

    return summary_df

# ── Target rank shift logic ───────────────────────────────────────────────────
#
# The big idea: within each NFL team, WR/TE are ranked by targets/game.
# When the #1 target-getter leaves, the #2 becomes the new #1 — a major
# fantasy upgrade. When a new WR joins and slots in above you, that's a
# downgrade.
#
# Think of a team's WR/TE group like a corporate org chart: if the VP leaves,
# the next in line gets promoted, with all the responsibilities (and salary).
#
# Boost table — driven by the NEW rank after transactions settle:
#   New rank 1: +20  (you're now the go-to target — elite value)
#   New rank 2: +10  (you're now a clear WR2/TE2 — startable)
#   New rank 3: +3   (you're now a WR3 — late-round flier at best)
#   New rank 4+:  0  (depth, mostly undraftable)
#
# Penalties for rank drops (a star WR joining your team pushes you down):
#   1→2: -12  (lost your WR1 role — significant value drop)
#   2→3: -7   (no longer a consistent start)
#   3→4: -2   (already marginal, now buried)

RANK_BOOSTS    = {1: 20, 2: 10, 3: 3}   # boost when reaching this rank
RANK_PENALTIES = {2: -12, 3: -7, 4: -2} # penalty when dropped to this rank


def _find_best_target_season(summary_df: pd.DataFrame) -> str | None:
    """Return the most recent season that has avg_targets_pg data."""
    for yr in ["2025", "2024", "2023"]:
        if f"avg_targets_pg_{yr}" in summary_df.columns:
            return yr
    return None


def _best_team_col(summary_df: pd.DataFrame, season: str) -> str:
    """
    Return the team column to use for a given season.
    Priority: team_{season} → team_{prev_season} → team
    """
    for col in [f"team_{season}", f"team_{int(season)-1}", "team"]:
        if col in summary_df.columns:
            return col
    return "team"


def compute_target_rank_shifts(
    summary_df:  pd.DataFrame,
    impacts_df:  pd.DataFrame,
    from_year:   str = "2024",
    to_year:     str = "2025",
) -> pd.DataFrame:
    """
    Compute genuine WR/TE target rank shifts by comparing actual depth charts.

    Correct algorithm:
    1. Load depth_charts_{from_year}.csv → baseline 2024 team rosters
    2. Load depth_charts_{to_year}.csv   → who is still on each 2024 team in 2025
    3. For each team's 2024 WR/TE group, rank players by avg_targets_pg (pre-tx rank)
    4. Remove players not on the same team in 2025 → they left
    5. Add players who joined from another team in 2025 → they arrived
    6. Re-rank remaining+arrivals → post-tx rank
    7. Delta = pre_rank − post_rank; apply boost/penalty table

    This uses depth_charts as the source of truth for WHO is WHERE, and
    avg_targets_pg as the measure of TARGET VOLUME — the combination gives
    accurate role-change signals grounded in real data.
    """
    # Always rank the 2024 roster by 2024 targets — that's the pre-transaction
    # pecking order. Using 2025 targets would mix pre/post states.
    tgt_col = f"avg_targets_pg_{from_year}"
    if tgt_col not in summary_df.columns:
        # Fall back to any available season
        tgt_season = _find_best_target_season(summary_df)
        if tgt_season is None:
            print("  ⚠  No avg_targets_pg_* column found — skipping rank shift analysis")
            summary_df["target_rank_boost"] = 0
            summary_df["target_rank_note"]  = ""
            return summary_df
        tgt_col = f"avg_targets_pg_{tgt_season}"

    print(f"\nComputing target rank shifts using depth charts "
          f"{from_year}→{to_year} + {tgt_col}...")

    # ── Load depth charts ─────────────────────────────────────────────────────
    dc_old = load_depth_chart(from_year)
    dc_new = load_depth_chart(to_year)

    if dc_old.empty or dc_new.empty:
        print("  ⚠  Missing depth chart data — skipping rank shifts")
        summary_df["target_rank_boost"] = 0
        summary_df["target_rank_note"]  = ""
        return summary_df

    pos_col_o = "fantasy_position" if "fantasy_position" in dc_old.columns else "position"
    pos_col_n = "fantasy_position" if "fantasy_position" in dc_new.columns else "position"

    # Filter to WR + TE only
    dc_old = dc_old[dc_old[pos_col_o].isin(["WR", "TE"])].copy()
    dc_new = dc_new[dc_new[pos_col_n].isin(["WR", "TE"])].copy()

    # ── Target lookup keyed on NORMALIZED names ───────────────────────────────
    # Depth chart names often differ in capitalisation, suffixes, or punctuation
    # from Sleeper names (e.g. "Dk Metcalf" vs "DK Metcalf", "Deebo Samuel Sr."
    # vs "Deebo Samuel", "Devonta Smith" vs "DeVonta Smith"). Normalizing both
    # sides before lookup resolves these silently.
    tgt_lookup:    dict[str, float] = {}   # norm → targets/game
    canon_lookup:  dict[str, str]   = {}   # norm → display name from summary
    for _, row in summary_df.iterrows():
        v = row.get(tgt_col)
        if pd.notna(v) and float(v) > 0:
            norm = normalize_name(str(row["full_name"]))
            tgt_lookup[norm]   = float(v)
            canon_lookup[norm] = str(row["full_name"])

    # ── Build 2024 team rosters ───────────────────────────────────────────────
    # {team → [{name (canonical), norm, tgt_pg, position}]}
    pre_rosters: dict[str, list[dict]] = {}
    for _, row in dc_old.iterrows():
        norm    = normalize_name(str(row["full_name"]))
        team    = row["nfl_team"]
        pos     = row[pos_col_o]
        tgt     = tgt_lookup.get(norm, 0.0)
        if tgt == 0:
            continue    # no target data → can't rank meaningfully
        canon = canon_lookup.get(norm, str(row["full_name"]))
        pre_rosters.setdefault(team, [])
        if any(p["norm"] == norm for p in pre_rosters[team]):
            continue    # deduplicate (same player listed at multiple depth slots)
        pre_rosters[team].append({"name": canon, "norm": norm,
                                   "tgt_pg": tgt, "pos": pos})

    # Rank pre-transaction: within each 2024 team, rank by avg_targets_pg desc.
    # Key is (canonical_name, team) — team-scoped so a player who switches teams
    # does NOT carry their old-team rank into their new-team post-roster calc.
    pre_rank_lookup: dict[tuple, int] = {}   # {(canonical_name, team) → rank}
    for team, roster in pre_rosters.items():
        sorted_r = sorted(roster, key=lambda p: p["tgt_pg"], reverse=True)
        for rank_0, player in enumerate(sorted_r):
            pre_rank_lookup[(player["name"], team)] = rank_0 + 1

    # ── Build 2025 team lookup: normalized_name → new_team ───────────────────
    # Normalize DC25 names the same way so lookups against DC24 norms match.
    new_team_lookup: dict[str, str] = {}   # norm → 2025 team
    for _, row in dc_new.iterrows():
        norm = normalize_name(str(row["full_name"]))
        if norm not in new_team_lookup:   # keep first entry if on two teams
            new_team_lookup[norm] = row["nfl_team"]

    # ── Simulate post-transaction rosters ────────────────────────────────────
    post_rosters: dict[str, list[dict]] = {
        team: list(players) for team, players in pre_rosters.items()
    }

    # Remove players who left their 2024 team (changed teams or left the NFL)
    for team in post_rosters:
        post_rosters[team] = [
            p for p in post_rosters[team]
            if new_team_lookup.get(p["norm"]) == team
        ]

    # Add players who joined a team in 2025 from a different 2024 team
    for _, row in dc_new.iterrows():
        norm     = normalize_name(str(row["full_name"]))
        new_team = row["nfl_team"]
        pos      = row[pos_col_n]
        tgt      = tgt_lookup.get(norm, 0.0)

        if tgt == 0:
            continue
        # Find which 2024 team this player was on
        old_team = None
        for t, roster in pre_rosters.items():
            if any(p["norm"] == norm for p in roster):
                old_team = t
                break

        if old_team == new_team:
            continue   # stayed on same team, already in post_roster
        if old_team is None:
            continue   # wasn't in dc_old at all (true rookie / no target data)

        # Changed teams → slot into their new team's post-roster
        canon = canon_lookup.get(norm, str(row["full_name"]))
        post_rosters.setdefault(new_team, [])
        if not any(p["norm"] == norm for p in post_rosters[new_team]):
            post_rosters[new_team].append({"name": canon, "norm": norm,
                                            "tgt_pg": tgt, "pos": pos})

    # ── Compute rank deltas ───────────────────────────────────────────────────
    rank_boost_map: dict[str, int] = {}
    rank_note_map:  dict[str, str] = {}

    for team, post_roster in post_rosters.items():
        if not post_roster:
            continue
        sorted_post = sorted(post_roster, key=lambda p: p["tgt_pg"], reverse=True)

        for post_rank_0, player in enumerate(sorted_post):
            pname     = player["name"]
            post_rank = post_rank_0 + 1
            # Only score players who were on THIS team in 2024 — arrivals
            # (who have no pre-rank on this team) are skipped here; their
            # effect is captured as penalties for teammates they push down.
            pre_rank = pre_rank_lookup.get((pname, team))

            if pre_rank is None:
                continue   # new arrival to this team — no pre-rank here

            if pre_rank == post_rank:
                continue   # no change

            if post_rank < pre_rank:   # moved up
                boost = RANK_BOOSTS.get(post_rank, 0)
                note  = f"Team target rank: #{pre_rank}→#{post_rank} (teammate departed)"
            else:                       # moved down
                boost = RANK_PENALTIES.get(post_rank, 0)
                note  = f"Team target rank: #{pre_rank}→#{post_rank} (higher-volume player joined)"

            if boost != 0:
                rank_boost_map[pname] = boost
                rank_note_map[pname]  = note

    # ── Apply to summary_df ───────────────────────────────────────────────────
    summary_df["target_rank_boost"] = (
        summary_df["full_name"].map(rank_boost_map).fillna(0).astype(int)
    )
    summary_df["target_rank_note"] = (
        summary_df["full_name"].map(rank_note_map).fillna("")
    )

    if "breakout_score" in summary_df.columns:
        summary_df["breakout_score"] = (
            summary_df["breakout_score"].fillna(0) + summary_df["target_rank_boost"]
        ).clip(lower=0, upper=100)

    upgraded   = (summary_df["target_rank_boost"] > 0).sum()
    downgraded = (summary_df["target_rank_boost"] < 0).sum()
    print(f"  ✓ Target rank shifts: {upgraded} players upgraded, "
          f"{downgraded} downgraded")

    movers = summary_df[summary_df["target_rank_boost"] != 0].copy()
    if not movers.empty:
        movers = movers.sort_values("target_rank_boost", ascending=False)
        print(f"\n  Top rank-shift beneficiaries:")
        for _, row in movers.head(8).iterrows():
            print(f"    {row['full_name']:<22} {row['position']}  "
                  f"boost={row['target_rank_boost']:+d}  {row['target_rank_note']}")
        negatives = movers[movers["target_rank_boost"] < 0]
        if not negatives.empty:
            print(f"  Top rank-shift losers:")
            for _, row in negatives.head(5).iterrows():
                print(f"    {row['full_name']:<22} {row['position']}  "
                      f"boost={row['target_rank_boost']:+d}  {row['target_rank_note']}")

    return summary_df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_transaction_pipeline(season_year: int | None = None) -> dict:
    """
    End-to-end pipeline:
      1. Fetch all ESPN transactions
      2. Filter to fantasy-relevant moves
      3. Cross-reference against our player DB
      4. Save transaction_impacts.csv
      5. Apply boosts to multi_year_summary.csv (if it exists)
      6. Re-save multi_year_summary.csv with new columns

    Returns a stats dict:
      {
        "total_fetched": int,     # raw ESPN items
        "total_filtered": int,    # fantasy-relevant moves in output
        "in_database": int,       # moves matched to our DB
        "high_impact": int,       # moves rated HIGH
        "medium_impact": int,     # moves rated MEDIUM
        "applied_to_summary": bool
      }
    """
    print("=" * 60)
    print("  AI Fantasy Football — Transaction Impact Scraper")
    print("  (source: depth chart comparison 2024 → 2025)")
    print("=" * 60)

    # Step 1: Load player database
    if SUMMARY_PATH.exists():
        players_df = pd.read_csv(SUMMARY_PATH, low_memory=False)
        print(f"\nLoaded {len(players_df):,} players from multi_year_summary.csv")
    else:
        print("\n⚠  multi_year_summary.csv not found — run data_processor.py first")
        return {"error": "multi_year_summary.csv not found"}

    # Step 2: Generate impacts from depth chart comparison (no external API needed)
    impacts_df = generate_impacts_from_depth_charts(players_df)
    if impacts_df.empty:
        print("⚠  No impacts detected — check that depth_charts_2024.csv and "
              "depth_charts_2025.csv exist in data/processed/")
        return {"error": "No depth chart data found"}

    # Step 4: Save transaction_impacts.csv
    PROCESSED.mkdir(parents=True, exist_ok=True)
    impacts_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved {len(impacts_df):,} transactions → "
          f"data/processed/transaction_impacts.csv")

    # Step 5 & 6: Apply to multi_year_summary if it exists
    applied = False
    if SUMMARY_PATH.exists() and not impacts_df.empty:
        print("\nApplying transaction boosts to multi_year_summary.csv...")
        summary_df = pd.read_csv(SUMMARY_PATH, low_memory=False)

        # Drop any existing transaction columns before re-applying
        for col in ("transaction_type", "transaction_note"):
            if col in summary_df.columns:
                summary_df = summary_df.drop(columns=[col])

        summary_df = apply_transaction_impacts(summary_df, impacts_df)

        # Target rank shifts: boost players who move up in team target hierarchy
        # after departures/arrivals, and penalize those who get pushed down.
        summary_df = compute_target_rank_shifts(summary_df, impacts_df)

        summary_df.to_csv(SUMMARY_PATH, index=False)
        print(f"✓ Updated multi_year_summary.csv")
        applied = True

    # Build and return stats summary for the Streamlit UI
    high_count   = (impacts_df["fantasy_impact"] == "HIGH").sum()   if not impacts_df.empty else 0
    med_count    = (impacts_df["fantasy_impact"] == "MEDIUM").sum() if not impacts_df.empty else 0
    in_db_count  = impacts_df["in_database"].sum()                  if not impacts_df.empty else 0

    stats = {
        "total_filtered":     len(impacts_df),
        "in_database":        int(in_db_count),
        "high_impact":        int(high_count),
        "medium_impact":      int(med_count),
        "applied_to_summary": applied,
    }

    print("\n📊 Summary:")
    print(f"   Fantasy-relevant moves:      {stats['total_filtered']:,}")
    print(f"   Matched to our DB:           {stats['in_database']:,}")
    print(f"   High-impact moves:           {stats['high_impact']:,}")
    print(f"   Medium-impact moves:         {stats['medium_impact']:,}")
    print(f"   Applied to summary:          {'Yes' if applied else 'No'}")
    print("\n✅ Transaction pipeline complete!")

    return stats


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_transaction_pipeline()
