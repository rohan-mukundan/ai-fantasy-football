"""
app.py
------
The Streamlit web app for the AI Fantasy Football Draft Agent.

Run with:
    streamlit run app.py

The app lets you:
  1. Set up a new draft (number of teams, your AI's draft slot, strategy)
  2. Enter which player each other team picks
  3. Ask the AI agent to recommend your next pick
  4. See each team's growing roster in the sidebar
"""

import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
from pathlib import Path
from agents.draft_agent import DraftAgent
from agents.strategy_agent import StrategyAgent
from strategy import DEFAULT_STRATEGY, STRATEGIES, get_league_strategy
from data_pipeline.sleeper_client import run_pipeline
from data_pipeline.data_processor import build_multi_year_summary
from data_pipeline.schedule_fetcher import fetch_bye_weeks
from data_pipeline.transactions_scraper import run_transaction_pipeline
from data_pipeline.current_season import (
    update_current_season_stats,
    clear_current_season_data,
    refresh_current_rosters,
    load_current_rosters,
    get_roster_refresh_time,
)
from agents.lineup_agent import LineupAgent
from agents.trade_agent  import TradeAgent
from data_pipeline.defense_rankings import (
    refresh_defense_rankings,
    load_defense_rankings,
    get_def_matchup,
    format_matchup_label,
    refresh_offense_rankings,
    load_offense_rankings,
    get_opponent_offense,
    get_def_streaming_recs,
    format_opponent_offense,
    refresh_def_quality_rankings,
    load_def_quality_rankings,
    get_remaining_schedule_strength,
    OFFENSE_EMOJI,
)

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fantasy Football Draft",
    page_icon="🏈",
    layout="wide"
)

# ── Load player data ───────────────────────────────────────────────────────────

DATA_PATH = Path(__file__).parent / "data" / "processed" / "multi_year_summary.csv"
LEAGUES_PATH = Path(__file__).parent / "data" / "leagues.json"

@st.cache_data  # Cache so we don't reload the CSV on every click
def load_players() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Fill missing team values
    df["team"] = df["team"].fillna("FA")
    return df


@st.cache_data
def load_players_for_season(season: str | None) -> pd.DataFrame:
    """
    Returns the player DataFrame with `team` set to the most accurate team
    assignment available for the given season year.

    Team data is applied in priority order (most accurate wins):

      1. Live roster snapshot  — rosters_{season}_current.json
         Fetched on demand via the "Refresh Rosters" button.  Accounts for
         mid-season trades, cut/signed kickers, and any other roster moves
         that happened after Season Setup was run.

      2. Depth-chart column   — team_{season} in multi_year_summary.csv
         Written by data_processor.py from depth_charts_{year}.csv scraped
         at the start of the season.  Correct for opening-day rosters but
         doesn't know about in-season moves.

      3. players.csv team     — fetched at Season Setup from Sleeper's live
         player database.  Always reflects today's team, so it's wrong for
         historical season simulations (e.g. Evans shows SF instead of TB).

    Falls back gracefully at each level when data is missing.
    """
    df = load_players().copy()
    if not season:
        return df

    # ── Layer 2: depth-chart season snapshot ─────────────────────────────────
    col = f"team_{season}"
    if col in df.columns:
        mask = df[col].notna() & (df[col].astype(str).str.strip() != "nan") & (df[col].astype(str).str.strip() != "")
        df.loc[mask, "team"] = df.loc[mask, col]

    # ── Layer 1: live roster snapshot (most current) ──────────────────────────
    # player_id is numeric in players.csv but stored as string keys in JSON
    live_rosters = load_current_rosters(season)
    if live_rosters:
        live_series = df["player_id"].astype(str).map(live_rosters)
        live_mask = live_series.notna() & (live_series != "nan")
        df.loc[live_mask, "team"] = live_series[live_mask]

    # Ensure any team that ended up empty/NaN is marked FA
    df["team"] = df["team"].fillna("FA").replace("", "FA").replace("nan", "FA")
    return df


# ── League persistence ─────────────────────────────────────────────────────────
# Leagues are saved to a small JSON file so they persist between app restarts.

def load_leagues() -> list[dict]:
    if not LEAGUES_PATH.exists():
        return []
    try:
        with open(LEAGUES_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_leagues(leagues: list[dict]) -> None:
    LEAGUES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEAGUES_PATH, "w") as f:
        json.dump(leagues, f, indent=2)

# ── Snake draft logic ──────────────────────────────────────────────────────────

def get_team_for_pick(pick_number: int, num_teams: int) -> int:
    """
    Returns which team (1 to num_teams) picks at a given overall pick number.
    Uses standard snake draft order:
      Round 1: Team 1, 2, 3, 4
      Round 2: Team 4, 3, 2, 1
      Round 3: Team 1, 2, 3, 4  ... and so on
    """
    pick_index = pick_number - 1          # convert to 0-based
    round_index = pick_index // num_teams  # which round (0-based)
    slot = pick_index % num_teams          # position within the round (0-based)

    if round_index % 2 == 0:
        return slot + 1           # forward: teams 1 → num_teams
    else:
        return num_teams - slot   # reverse: teams num_teams → 1


def get_round_and_slot(pick_number: int, num_teams: int) -> tuple[int, int]:
    """Returns (round_number, pick_within_round), both 1-based."""
    pick_index = pick_number - 1
    round_num = pick_index // num_teams + 1
    slot = pick_index % num_teams + 1
    return round_num, slot


# ── Session state initialisation ──────────────────────────────────────────────
# Streamlit reruns the entire script on every user interaction.
# session_state persists variables across reruns — like a memory for the app.

def _load_saved_draft_year() -> int:
    """
    Returns the draft year from the last Season Setup run, or the current
    calendar year as a fallback.  Reads data/processed/season_config.json
    which is written whenever Generate Data is clicked — so it survives app
    restarts and means the user never has to re-enter the year just to draft
    or view their lineup.
    """
    config_path = Path(__file__).parent / "data" / "processed" / "season_config.json"
    try:
        if config_path.exists():
            with open(config_path) as f:
                return int(json.load(f).get("draft_year", datetime.now().year))
    except Exception:
        pass
    return datetime.now().year


def init_state():
    data_exists = DATA_PATH.exists()
    defaults = {
        "app_page":            "Season Setup" if not data_exists else "Draft",
        "draft_year":          _load_saved_draft_year(),
        "draft_started":       False,
        "num_teams":           4,
        "num_rounds":          15,
        "ai_team":             1,          # which team slot the AI occupies
        "current_pick":        1,          # overall pick counter (1 to total picks)
        "picks":               [],         # list of all picks made so far
        "strategy_name":       "AI Agent Strategy", # display name of selected strategy
        "strategy":            DEFAULT_STRATEGY,    # active strategy text used by DraftAgent
        "pending_draft":       None,       # holds config after AI strategy generation, before draft starts
        "recommendation":      None,       # set only when there's an API error to display
        "selected_player":     None,       # player clicked in the available players table
        "last_ai_pick":        None,       # persists AI reasoning until the next human pick
        "team_names":          None,       # draft order team names from the active league, if any
        "active_league":       None,       # the league dict the current draft was started from
        "creating_league":     False,      # whether the "create new league" form is open
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()


# ── Helper: get available players ─────────────────────────────────────────────

def get_available_players(all_players: pd.DataFrame) -> pd.DataFrame:
    """Returns players who haven't been drafted yet AND are on an active NFL roster.

    Players with team == "FA" (unsigned, retired, or cut before the season
    started) are excluded — there's no point drafting someone who has no team.
    """
    drafted_names = {p["full_name"] for p in st.session_state.picks}
    on_team = all_players["team"].notna() & (all_players["team"] != "FA")
    available = all_players[
        ~all_players["full_name"].isin(drafted_names) & on_team
    ].copy()
    return available.reset_index(drop=True)


def get_team_roster(team_num: int) -> list[dict]:
    """Returns all picks made by a specific team."""
    return [p for p in st.session_state.picks if p["team_num"] == team_num]


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — Draft status and team rosters
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.title("🏈 AI Fantasy Football")

        # ── Page navigation ───────────────────────────────────────────────────
        # "League View" is reached via the My Leagues tab, not shown as its own
        # nav option — but it must be a valid value for the radio's index lookup.
        nav_options = ["Season Setup", "My Leagues", "Draft"]
        current_page = st.session_state.app_page
        radio_page = "My Leagues" if current_page == "League View" else current_page

        chosen = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(radio_page) if radio_page in nav_options else 0,
            label_visibility="collapsed",
        )

        if current_page == "League View" and chosen == "My Leagues":
            # Radio unchanged — stay in League View
            pass
        else:
            st.session_state.app_page = chosen
            if current_page == "League View":
                # User navigated away via the radio — drop out of League View
                st.session_state.active_league = None

        st.divider()

        if st.session_state.app_page in ("My Leagues",) or current_page == "League View":
            return

        if st.session_state.app_page == "Season Setup" or not st.session_state.draft_started:
            if DATA_PATH.exists():
                mtime = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
                st.caption(f"📊 Data last generated: {mtime.strftime('%b %d, %Y %H:%M')}")
            else:
                st.warning("No player data found. Run Season Setup first.")
            return

        num_teams  = st.session_state.num_teams
        num_rounds = st.session_state.num_rounds
        total_picks = num_teams * num_rounds
        current_pick = st.session_state.current_pick

        # Draft progress
        st.markdown("### Draft Progress")
        if current_pick <= total_picks:
            round_num, slot = get_round_and_slot(current_pick, num_teams)
            st.metric("Round", f"{round_num} of {num_rounds}")
            st.metric("Overall Pick", f"{current_pick} of {total_picks}")
            st.progress(current_pick / total_picks)
        else:
            st.success("✅ Draft Complete!")

        st.divider()

        # Team rosters
        st.markdown("### Team Rosters")
        team_names = st.session_state.get("team_names")
        for team_num in range(1, num_teams + 1):
            roster = get_team_roster(team_num)
            team_label = team_names[team_num - 1] if team_names and team_num <= len(team_names) \
                          else f"Team {team_num}"
            label = f"🤖 **{team_label} (AI)**" if team_num == st.session_state.ai_team \
                    else f"👤 {team_label}"
            with st.expander(f"{label} — {len(roster)} picks"):
                if not roster:
                    st.caption("No picks yet")
                else:
                    for p in roster:
                        st.caption(f"{p['position']} · {p['full_name']} ({p['nfl_team']})")

        # Injury / suspension notes for the AI agent
        st.divider()
        st.markdown("### 🚨 Injury / Suspension Notes")
        st.caption("Type any current injury or suspension info. The AI will factor this into every pick recommendation.")
        st.session_state.injury_notes = st.text_area(
            label="injury_notes",
            label_visibility="collapsed",
            value=st.session_state.get("injury_notes", ""),
            placeholder="e.g. Josh Jacobs injury/suspension risk, return timeline unclear. Ricky Pearsall season-ending injury. Jordyn Tyson out ~2 months (hamstring).",
            height=120,
            key="injury_notes_input",
        )

        # Reset button
        st.divider()
        if st.button("🔄 Reset Draft", use_container_width=True):
            for key in ["draft_started", "current_pick", "picks",
                        "recommendation", "selected_player", "last_ai_pick"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MY LEAGUES — Create and manage leagues
# ═══════════════════════════════════════════════════════════════════════════════

SCORING_OPTIONS = ["PPR", "Half PPR", "Standard"]
TEAM_OPTIONS = [2, 4, 6, 8, 10, 12]

# Standard starting lineup slots for a league roster view.
STARTER_SLOTS = [
    ("QB",   1, {"QB"}),
    ("RB",   2, {"RB"}),
    ("WR",   2, {"WR"}),
    ("TE",   1, {"TE"}),
    ("FLEX", 1, {"RB", "WR"}),
    ("K",    1, {"K"}),
    ("DEF",  1, {"DEF"}),
]


def build_lineup(roster: list[dict], sort_key: str = "avg_pts_ppr") -> tuple[list[dict], list[dict]]:
    """
    Splits a team's roster into starters and bench based on the standard
    lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB/WR/TE), 1 K, 1 DEF.

    Within each slot, the highest-scoring eligible players (sorted by
    `sort_key`, defaulting to overall avg PPR pts/game) are chosen as
    starters. Returns (starters, bench), where starters is a list of dicts
    with an added "slot" key.
    """
    remaining = sorted(roster, key=lambda p: p.get(sort_key, 0) or 0, reverse=True)
    starters = []

    for slot_label, count, eligible_positions in STARTER_SLOTS:
        for _ in range(count):
            # Find the best remaining player eligible for this slot
            pick = next((p for p in remaining if p["position"] in eligible_positions), None)
            if pick is None:
                continue
            remaining.remove(pick)
            starters.append({**pick, "slot": slot_label})

    bench = remaining
    return starters, bench


# ── In-season performance blending ──────────────────────────────────────────────
# Once games have been played in the current season, we blend each player's
# preseason projection (avg_pts_ppr, based on prior years) with how they've
# actually performed so far this season. This lets the lineup react to
# breakout players (outperforming their projection) and regressing veterans
# (underperforming it) — without throwing away the prior-years signal when
# only a game or two has been played.

# How many "games" of prior-year evidence the preseason average is worth.
# A higher number means it takes longer for in-season results to take over.
# Set deliberately high so a single bad (or great) game doesn't swing the
# lineup — e.g. one quiet Week 1 shouldn't be enough to bench a proven
# top performer.
CURRENT_SEASON_PRIOR_STRENGTH = 8

# Don't let the current season affect lineup decisions at all until the
# player has at least this many games in the books — one data point is
# noise, not signal.
MIN_WEEKS_FOR_BLEND = 2

# Minimum blended-avg-pts/gm improvement a free agent must offer over the
# weakest same-position player on your roster before we recommend the pickup.
MIN_UPGRADE_IMPROVEMENT = 2.0

# A rostered player is a "confirmed non-performer" if they've played at least
# this many games and are averaging below this pts/gm threshold.  Non-performers
# get much lower prior protection so current reality dominates their score —
# six straight bad games is not the same as one bad game.
LOW_PRODUCTION_THRESHOLD = 3.0   # pts/gm — below this consistently = not contributing
LOW_PRODUCTION_MIN_WEEKS = 2     # games needed before we call it confirmed
LOW_PRODUCTION_PRIOR_STRENGTH = 1  # vs CURRENT_SEASON_PRIOR_STRENGTH=8 for normal players
# Threshold for recommending a swap when the rostered player is a confirmed non-performer.
# Lower than MIN_UPGRADE_IMPROVEMENT because nearly any active player beats zero production.
LOW_PRODUCTION_UPGRADE_THRESHOLD = 1.0

# ── Trade recommendations ──────────────────────────────────────────────────────
# Minimum blended-score gain at the target position to bother proposing a trade.
MIN_TRADE_IMPROVEMENT = 1.0
# How many of each position I must keep AFTER giving one away.
# Using 2 for RB/WR means your two clear starters are protected; flex/bench is
# available to package.
TRADE_KEEP_MIN = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
# Positions where trading is pointless or distorting.
# QB is excluded because: everyone has a serviceable QB, the value differential
# between QB1 and QB10 is small, and QB-focused trades rarely reflect real value.
# RB and WR are where real trade market value lives.
TRADE_NEVER_GIVE = {"K", "DEF", "QB"}


@st.cache_data
def load_current_season_stats(season: str, through_week: int) -> pd.DataFrame:
    """
    Returns this season's running player stats through `through_week`,
    fetching/updating data/processed/season_{season}_current_summary.csv
    as needed (already-fetched weeks are loaded from disk, so this is cheap
    to call again for the same week).
    """
    return update_current_season_stats(season, through_week)


@st.cache_data(show_spinner=False)
def get_defense_rankings(season: str, through_week: int, prior_season: str | None = None) -> pd.DataFrame:
    """
    Builds (or loads from disk cache) defensive rankings for `season` through
    `through_week`. Uses Sleeper weekly stats already on disk + ESPN schedule.
    When prior_season is provided and through_week < PRIOR_BLEND_CUTOFF, blends
    in the prior year's full-season averages to prevent early-season skew.
    Returns an empty DataFrame if stats haven't been fetched yet.
    """
    if through_week < 1:
        return pd.DataFrame()
    return refresh_defense_rankings(season, through_week, prior_season=prior_season)


@st.cache_data(show_spinner=False)
def get_offense_rankings(season: str, through_week: int, prior_season: str | None = None) -> pd.DataFrame:
    """
    Builds (or loads from disk cache) offensive output rankings for `season`
    through `through_week`. Used to identify weak offenses for DEF streaming.
    Rank 1 = weakest offense (best DEF streaming target).
    When prior_season is provided and through_week < PRIOR_BLEND_CUTOFF, blends
    in the prior year's full-season averages to stabilize early-season rankings.
    """
    if through_week < 1:
        return pd.DataFrame()
    return refresh_offense_rankings(season, through_week, prior_season=prior_season)


@st.cache_data(show_spinner=False)
def get_def_quality_rankings_cached(season: str, through_week: int) -> pd.DataFrame:
    """
    Builds (or loads from disk cache) DEF quality rankings for `season` through
    `through_week`. Ranks each team's DEF by average fantasy pts_ppr scored per game.
    def_quality_rank 1 = worst DEF, 32 = best DEF.
    Returns empty DataFrame if no weekly stats are cached yet.
    """
    if through_week < 1:
        return pd.DataFrame()
    return refresh_def_quality_rankings(season, through_week)


def apply_current_season_blend(roster: list[dict], season: str | None, week: int) -> list[dict]:
    """
    Returns a copy of `roster` where each player has a "blended_avg_ppr"
    key — a mix of their preseason weighted_avg_ppr (prior years) and their
    actual scoring so far this season. Also adds "current_avg_ppr" and
    "weeks_played_current" for display.

    PRESEASON PRIOR
    ---------------
    The prior is resolved in this order:
      1. weighted_avg_ppr from the full player dataset (3-year weighted avg,
         the same ranking used by the draft agent)
      2. avg_pts_ppr stored in the player dict (recorded at draft time)
      3. 0 as a last resort

    This handles the case where avg_pts_ppr was stored as 0 at draft time
    because the column didn't exist in multi_year_summary.csv — by going
    back to the source DataFrame we always get the meaningful baseline.

    BLENDING
    --------
    Intentionally conservative: MIN_WEEKS_FOR_BLEND games must be played
    before current-season results affect blended_avg_ppr at all, and
    CURRENT_SEASON_PRIOR_STRENGTH keeps prior years dominant early on.

    If there's no current-season data yet (Week 1, or season unknown),
    blended_avg_ppr falls back to the preseason prior.
    """
    enriched = [dict(p) for p in roster]

    # Build a name→player lookup using the season-aware loader so that the
    # nfl_team refresh below reflects mid-season trades and signings (not just
    # the Season Setup snapshot).
    try:
        df = load_players_for_season(season)
        players_by_name = df.set_index("full_name") if not df.empty else pd.DataFrame()
    except Exception:
        players_by_name = pd.DataFrame()

    def _prior(p: dict) -> float:
        name = p["full_name"]
        if not players_by_name.empty and name in players_by_name.index:
            r = players_by_name.loc[name]
            v = r.get("weighted_avg_ppr") or r.get("avg_pts_ppr")
            if v and float(v) > 0:
                return float(v)
        return p.get("avg_pts_ppr", 0) or 0

    if not season:
        for p in enriched:
            p["blended_avg_ppr"] = _prior(p)
        return enriched

    through_week = week - 1  # only weeks already played feed the blend
    current_stats = load_current_season_stats(season, through_week)

    if current_stats.empty:
        for p in enriched:
            p["blended_avg_ppr"] = _prior(p)
        return enriched

    stats_by_name = current_stats.set_index("full_name")

    for p in enriched:
        prior_avg = _prior(p)

        # Always refresh nfl_team from the live DataFrame so that players
        # who were cut after the draft (team → "FA") are flagged correctly
        # by get_lineup_for_week's cut-player check below.
        name = p["full_name"]
        if not players_by_name.empty and name in players_by_name.index:
            r = players_by_name.loc[name]
            current_team = r.get("team")
            if current_team and str(current_team) not in ("nan", ""):
                p["nfl_team"] = str(current_team)
            else:
                p["nfl_team"] = "FA"

        if name not in stats_by_name.index:
            p["blended_avg_ppr"] = prior_avg
            continue

        row = stats_by_name.loc[name]
        weeks_played = int(row["weeks_played_current"])
        current_avg  = float(row["avg_pts_ppr_current"])

        if weeks_played == 0:
            p["blended_avg_ppr"] = prior_avg
            continue

        # Always record the current-season number for display...
        p["current_avg_ppr"]      = current_avg
        p["weeks_played_current"] = weeks_played

        # ...but only let it influence the lineup once we have enough games
        # to call it a trend rather than noise (see MIN_WEEKS_FOR_BLEND).
        if weeks_played < MIN_WEEKS_FOR_BLEND:
            p["blended_avg_ppr"] = prior_avg
            continue

        # Shrinkage: the more games played this season, the more the lineup
        # trusts the current-season number over the preseason projection.
        current_weight = weeks_played / (weeks_played + CURRENT_SEASON_PRIOR_STRENGTH)
        p["blended_avg_ppr"] = round(
            current_avg * current_weight + prior_avg * (1 - current_weight), 1
        )

    return enriched


# ── Bye week lookup ─────────────────────────────────────────────────────────────
# Bye week data is fetched ONCE per season during Season Setup (see
# render_season_setup) and saved to data/processed/bye_weeks_{season}.json.
# We just read that saved file here — no live fetching during lineup building.

@st.cache_data
def load_bye_weeks(season: str) -> dict:
    """Loads the saved bye-week map for a season: {team_abbr: bye_week}."""
    path = Path(__file__).parent / "data" / "processed" / f"bye_weeks_{season}.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def get_free_agent_pool(all_players: pd.DataFrame, drafted_names: set) -> pd.DataFrame:
    """Returns undrafted players who are on an active NFL roster.

    Excludes players with team == "FA" (unsigned, cut, or retired) — claiming
    a player with no team on the waiver wire would accomplish nothing.
    """
    on_team = all_players["team"].notna() & (all_players["team"] != "FA")
    return all_players[
        ~all_players["full_name"].isin(drafted_names) & on_team
    ].copy()


def get_waiver_candidates(
    eligible_positions: set,
    week: int,
    season: str | None,
    bye_weeks: dict,
    all_players: pd.DataFrame | None,
    drafted_names: set | None,
    n: int = 8,
) -> list[dict]:
    """
    Returns up to `n` free-agent candidates eligible for `eligible_positions`
    whose NFL team isn't on a bye this week, ranked by blended avg pts.

    multi_year_summary.csv uses "weighted_avg_ppr" as its composite ranking
    column; falls back to that when a plain "avg_pts_ppr" column isn't present.
    """
    if all_players is None or drafted_names is None:
        return []

    pool = get_free_agent_pool(all_players, drafted_names)
    pool = pool[pool["position"].isin(eligible_positions)]
    if not pool.empty:
        pool = pool[pool["team"].apply(lambda t: bye_weeks.get(t) != week)]
    if pool.empty:
        return []

    sort_col = "avg_pts_ppr" if "avg_pts_ppr" in pool.columns else "weighted_avg_ppr"
    # Fetch extra rows so we still have `n` after blending + re-sorting
    top_raw = pool.sort_values(sort_col, ascending=False).head(n * 2)

    candidates = []
    for _, row in top_raw.iterrows():
        fa = {
            "full_name":   row["full_name"],
            "position":    row["position"],
            "nfl_team":    row.get("team", "FA"),
            "avg_pts_ppr": float(row.get("avg_pts_ppr") or row.get("weighted_avg_ppr") or 0),
        }
        fa = apply_current_season_blend([fa], season, week)[0]
        candidates.append(fa)

    candidates.sort(key=lambda p: p.get("blended_avg_ppr", 0) or 0, reverse=True)
    return candidates[:n]


def get_upgrade_recommendations(
    enriched_roster: list[dict],
    season: str | None,
    week: int,
    all_players: pd.DataFrame,
    drafted_names: set,
    dismissed_names: set | None = None,
    max_recs: int = 5,
) -> list[dict]:
    """
    Scans the free-agent pool for players who are better than the weakest
    same-position player on the roster by at least MIN_UPGRADE_IMPROVEMENT
    pts/gm, using the same ranking criteria as the draft agent.

    KEY RULES
    ---------
    1. No suggestions for Week 1. The draft already considered all available
       preseason data to build the best possible roster — there's nothing new
       to act on until actual games have been played.

    2. The comparison baseline is `weighted_avg_ppr` (the 3-year weighted
       average used by the draft agent: 2024=50%, 2023=30%, 2022=20%), NOT
       a single-season avg. This prevents one or two outlier games from making
       a mediocre player look like a waiver gem.

    3. Both FAs and rostered players are blended the same way:
         blended = weighted_avg_ppr * (1 - w)  +  current_season_avg * w
       where w = weeks_played / (weeks_played + CURRENT_SEASON_PRIOR_STRENGTH)
       and w=0 when weeks_played < MIN_WEEKS_FOR_BLEND.
       The same conservative parameters as the lineup blending are used,
       so short hot streaks don't trigger premature waiver recommendations.

    Each returned entry:
        {
          "pickup":      FA player dict (with blended_avg_ppr),
          "drop":        roster player dict being replaced,
          "improvement": float  (pts/gm gained),
          "position":    str,
        }
    """
    from collections import defaultdict

    # Rule 1: no suggestions before any games have been played.
    if week <= 1:
        return []

    if dismissed_names is None:
        dismissed_names = set()

    pool = get_free_agent_pool(all_players, drafted_names)
    if pool.empty:
        return []

    # Index all_players by name so we can look up weighted_avg_ppr for anyone.
    players_by_name = all_players.set_index("full_name") if not all_players.empty else pd.DataFrame()

    # Load current-season stats once for efficient per-player lookup.
    through_week = week - 1
    if season and through_week >= 1:
        current_stats_df = load_current_season_stats(season, through_week)
        current_by_name  = (
            current_stats_df.set_index("full_name")
            if not current_stats_df.empty else pd.DataFrame()
        )
    else:
        current_by_name = pd.DataFrame()

    def _weighted_blended(name: str, fallback_prior: float) -> tuple[float, dict]:
        """
        Returns (blended_score, info_dict) for any player, using weighted_avg_ppr
        as the preseason prior. Info dict carries current_avg_ppr / weeks_played
        for display, mirroring apply_current_season_blend's output keys.
        """
        # Preseason prior: weighted 3-year avg (same as draft ranking)
        if name in players_by_name.index:
            p_row = players_by_name.loc[name]
            prior = float(p_row.get("weighted_avg_ppr") or p_row.get("avg_pts_ppr") or fallback_prior)
        else:
            prior = fallback_prior

        info: dict = {"avg_pts_ppr": prior, "blended_avg_ppr": prior}

        if current_by_name.empty or name not in current_by_name.index:
            return prior, info

        cs_row  = current_by_name.loc[name]
        weeks   = int(cs_row["weeks_played_current"])
        cur_avg = float(cs_row["avg_pts_ppr_current"])

        info["current_avg_ppr"]      = cur_avg
        info["weeks_played_current"] = weeks

        # ── Two-path non-performer detection ──────────────────────────────────
        #
        # Path A — poor per-game average across multiple active games.
        #   Catches players who played regularly but scored next to nothing.
        #   Example: 3 games played, avg 0.5 pts/gm.
        #
        # Path B — poor total output spread across actual season weeks.
        #   Catches players who barely show up at all: a 3rd-string RB who
        #   scores 0 in 10 weeks and 4 pts in 2 garbage-time games has a
        #   fine per-game avg (2.0) but only 8 pts over 12 season weeks.
        #   This is the case that was letting players like Guerendo survive
        #   all season — one decent game kept their per-game avg above the
        #   threshold even though they're a dead roster spot.
        total_pts    = float(cs_row.get("total_pts_ppr_current", 0) or 0)
        pts_per_week = total_pts / through_week if through_week > 0 else 0.0

        low_per_game   = (weeks >= LOW_PRODUCTION_MIN_WEEKS
                          and cur_avg < LOW_PRODUCTION_THRESHOLD)
        low_per_season = (through_week >= 3
                          and pts_per_week < LOW_PRODUCTION_THRESHOLD)

        # Early-exit for too little data — unless Path B already flags them
        if weeks < MIN_WEEKS_FOR_BLEND and not low_per_season:
            # Even one game of near-zero production is a signal when the
            # preseason projection was also low (never a real contributor).
            if weeks >= 1 and cur_avg < LOW_PRODUCTION_THRESHOLD and prior < LOW_PRODUCTION_THRESHOLD:
                info["confirmed_low_production"] = True
            return prior, info

        if low_per_game or low_per_season:
            # Use whichever signal gives the more damning blended score.
            # Path B uses the per-season-week average so that weeks of 0
            # output are properly counted against the player.
            if low_per_season:
                eff_weeks      = through_week
                effective_avg  = pts_per_week
            else:
                eff_weeks      = weeks
                effective_avg  = cur_avg
            w       = eff_weeks / (eff_weeks + LOW_PRODUCTION_PRIOR_STRENGTH)
            blended = round(effective_avg * w + prior * (1 - w), 1)
            info["blended_avg_ppr"]          = blended
            info["confirmed_low_production"] = True
            return blended, info

        # Normal blending for players with adequate production
        w = weeks / (weeks + CURRENT_SEASON_PRIOR_STRENGTH)
        blended = round(cur_avg * w + prior * (1 - w), 1)
        info["blended_avg_ppr"] = blended
        return blended, info

    # Build position groups for rostered players, scored with weighted baseline.
    by_pos: defaultdict[str, list] = defaultdict(list)
    for p in enriched_roster:
        score, info = _weighted_blended(p["full_name"], p.get("avg_pts_ppr", 0) or 0)
        by_pos[p["position"]].append({
            **p,
            "_cmp_score":        score,
            "_is_low_prod":      info.get("confirmed_low_production", False),
        })
    for pos in by_pos:
        by_pos[pos].sort(key=lambda p: p["_cmp_score"])

    # ── Excess single-slot cleanup ────────────────────────────────────────────
    # DEF and K only have 1 starter slot; a second one on the bench is dead
    # weight. If somehow 2 ended up on the roster (usually a draft agent error),
    # surface it as a priority drop recommendation with no required pickup.
    cleanup_recs: list[dict] = []
    for single_pos in ("DEF", "K"):
        singles_on_roster = by_pos.get(single_pos, [])
        if len(singles_on_roster) > 1:
            # Drop the lowest-scoring one; keep the better one
            excess = singles_on_roster[0]   # already sorted ascending by _cmp_score
            cleanup_recs.append({
                "pickup":      None,          # no pickup needed — just drop the extra
                "drop":        excess,
                "improvement": 0.0,
                "position":    single_pos,
                "cleanup":     True,          # flag so the UI can render a different message
            })

    recs: list[dict] = []
    seen_pickup_names: set[str] = set()
    seen_drop_names:   set[str] = set()

    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        pos_roster = by_pos.get(pos, [])
        if not pos_roster:
            continue

        worst            = pos_roster[0]
        worst_cmp_score  = worst["_cmp_score"]
        worst_is_low_prod = worst["_is_low_prod"]

        # Confirmed non-performers need a much smaller improvement to be worth
        # replacing — any active, healthy FA who's even slightly better is a
        # better use of that roster spot.
        effective_threshold = (
            LOW_PRODUCTION_UPGRADE_THRESHOLD if worst_is_low_prod
            else MIN_UPGRADE_IMPROVEMENT
        )

        pos_pool = pool[pool["position"] == pos]
        if pos_pool.empty:
            continue

        # Rule 2: sort FA pool by weighted_avg_ppr (draft-consistent ranking)
        sort_col = (
            "weighted_avg_ppr" if "weighted_avg_ppr" in pos_pool.columns
            else "avg_pts_ppr"  if "avg_pts_ppr"      in pos_pool.columns
            else pos_pool.columns[0]
        )
        top_raw = pos_pool.sort_values(sort_col, ascending=False).head(20)

        for _, row in top_raw.iterrows():
            fa_name = row["full_name"]
            if fa_name in dismissed_names or fa_name in seen_pickup_names:
                continue
            if worst["full_name"] in seen_drop_names:
                continue

            fa_prior = float(row.get("weighted_avg_ppr") or row.get("avg_pts_ppr") or 0)
            fa_score, fa_info = _weighted_blended(fa_name, fa_prior)

            improvement = fa_score - worst_cmp_score
            if improvement >= effective_threshold:
                seen_pickup_names.add(fa_name)
                seen_drop_names.add(worst["full_name"])
                # Build a display-ready FA dict using the blended info
                fa_dict = {
                    "full_name":   fa_name,
                    "position":    row["position"],
                    "nfl_team":    row.get("team", "FA"),
                    **fa_info,
                }
                recs.append({
                    "pickup":      fa_dict,
                    "drop":        worst,
                    "improvement": round(improvement, 1),
                    "position":    pos,
                })
                break

    recs.sort(key=lambda r: r["improvement"], reverse=True)
    # Cleanup recs go first — fixing an illegal roster is more urgent than upgrades
    return (cleanup_recs + recs)[:max_recs]


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADE RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_trade_recommendations(
    my_roster:            list[dict],
    all_picks:            list[dict],
    all_players:          pd.DataFrame,
    season:               str | None,
    week:                 int,
    num_teams:            int,
    ai_team:              int,
    team_names:           list[str],
    dismissed:            set | None = None,
    max_trades:           int = 3,
    defense_rankings_df:  pd.DataFrame | None = None,
) -> list[dict]:   # noqa: C901
    """
    Finds trade opportunities for the AI team and returns up to max_trades
    proposals.  Supports any package structure: 1-for-1, 2-for-1, 3-for-1,
    1-for-2, 2-for-2, etc.  Each proposal contains:

        give         – list of player dicts from my roster being given away
        get          – list of player dicts from the other team being received
        target_team  – team number of the trade partner
        target_name  – display name of the trade partner
        improvement  – pts/gm gain across the positions I'm improving
        give_value   – combined blended score of the give package
        get_value    – combined blended score of the get package
        trade_type   – string like "1for1", "2for1", "1for2", "3for1" etc.
        sell_high    – True if any give player is trading above real production
        buy_low      – True if any get player has a strong prior but rough start
        pitch        – Claude-generated explanation (str or None)

    Proposals are only generated when:
      - We're past week 3 (need enough season data)
      - The trade meaningfully improves the team (MIN_TRADE_IMPROVEMENT)
      - The package exchange is roughly fair (TRADE_FAIRNESS_RATIO)
      - I still have minimum required depth at each position after the trade
    """
    from collections import defaultdict
    from itertools import combinations

    if dismissed is None:
        dismissed = set()

    if week <= 3:
        return []

    through_week = week - 1
    players_by_name = (
        all_players.set_index("full_name") if not all_players.empty else pd.DataFrame()
    )

    if season and through_week >= 1:
        current_stats_df = load_current_season_stats(season, through_week)
        current_by_name  = (
            current_stats_df.set_index("full_name")
            if not current_stats_df.empty else pd.DataFrame()
        )
    else:
        current_by_name = pd.DataFrame()

    # For trade evaluation, use a lower prior strength than lineup decisions.
    # Lineup logic is conservative (don't bench a proven vet after 1 bad week).
    # Trade logic should reflect what a player is actually doing THIS season —
    # that's how real managers evaluate trade offers.
    TRADE_PRIOR_STRENGTH = 4  # vs CURRENT_SEASON_PRIOR_STRENGTH=8 for lineups

    # If a player's current-season avg is below this fraction of their prior,
    # they're clearly not what they used to be (committee back, cut risk, etc.).
    # Switch to aggressive discounting so their inflated prior doesn't make
    # them look tradeable when they're not.
    TRADE_DECLINE_RATIO    = 0.60  # below 60% of prior = significant decline
    TRADE_DECLINE_MIN_WEEKS = 5    # need at least 5 games — protects injured players
    # with small samples (e.g. scored 2 pts in week 1 before getting hurt;
    # season avg = 2 would falsely trigger decline detection)

    def _blended(player_dict: dict) -> tuple[float, dict]:
        """
        Blended value for trade evaluation.

        Uses a lower prior strength than lineup decisions (4 vs 8) so
        current-season performance matters more when assessing trade value.
        Also applies aggressive discounting when a player is scoring
        significantly below their historical baseline — this catches
        declining veterans whose prior-year average would otherwise
        make them look far more valuable than they actually are.
        """
        name = player_dict["full_name"]
        if not players_by_name.empty and name in players_by_name.index:
            pr    = players_by_name.loc[name]
            prior = float(pr.get("weighted_avg_ppr") or pr.get("avg_pts_ppr") or
                          player_dict.get("avg_pts_ppr", 0) or 0)
        else:
            prior = float(player_dict.get("avg_pts_ppr", 0) or 0)

        info: dict = {"avg_pts_ppr": prior, "blended_avg_ppr": prior}

        if current_by_name.empty or name not in current_by_name.index:
            return prior, info

        cs         = current_by_name.loc[name]
        wks        = int(cs["weeks_played_current"])
        cur        = float(cs["avg_pts_ppr_current"])
        total_pts  = float(cs.get("total_pts_ppr_current", 0) or 0)
        pts_per_wk = total_pts / through_week if through_week > 0 else 0.0

        info["current_avg_ppr"]      = cur
        info["weeks_played_current"] = wks

        low_per_game   = wks >= LOW_PRODUCTION_MIN_WEEKS and cur < LOW_PRODUCTION_THRESHOLD
        low_per_season = through_week >= 3 and pts_per_wk < LOW_PRODUCTION_THRESHOLD

        if low_per_game or low_per_season:
            eff_wks = through_week if low_per_season else wks
            eff_avg = pts_per_wk   if low_per_season else cur
            w       = eff_wks / (eff_wks + LOW_PRODUCTION_PRIOR_STRENGTH)
            blended = round(eff_avg * w + prior * (1 - w), 1)
            info["blended_avg_ppr"] = blended
            info["confirmed_low_production"] = True
            return blended, info

        if wks < MIN_WEEKS_FOR_BLEND:
            return prior, info

        # Significant underperformance check: if the player is producing less
        # than 60% of their historical baseline over 3+ games, they are clearly
        # not what they used to be (aging back in a committee, injury-limited,
        # changed role, etc.).  Use aggressive discounting so their inflated
        # prior year average doesn't make them appear trade-worthy.
        if (wks >= TRADE_DECLINE_MIN_WEEKS
                and prior > 5.0
                and cur < prior * TRADE_DECLINE_RATIO):
            w       = wks / (wks + LOW_PRODUCTION_PRIOR_STRENGTH)
            blended = round(cur * w + prior * (1 - w), 1)
            info["blended_avg_ppr"]    = blended
            info["confirmed_declining"] = True
            return blended, info

        # Normal case: use a lower prior strength than lineup decisions
        # so current-season reality carries more weight in trade value.
        w       = wks / (wks + TRADE_PRIOR_STRENGTH)
        blended = round(cur * w + prior * (1 - w), 1)
        info["blended_avg_ppr"] = blended
        return blended, info

    # ── Build per-team, per-position sorted player lists ────────────────────
    team_pos: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for p in all_picks:
        score, info = _blended(p)
        enriched = {
            **p,
            "_blended_score":  score,
            "blended_avg_ppr": score,
            **{k: v for k, v in info.items() if k not in p},
        }
        team_pos[p["team_num"]][p["position"]].append(enriched)

    for tn in team_pos:
        for pos in team_pos[tn]:
            team_pos[tn][pos].sort(key=lambda x: x["_blended_score"], reverse=True)

    # ── Value Over Replacement (VOR) for position-fair trade valuation ───────
    # QBs score 18-20 raw pts/gm but the gap between QB1 and QB12 in any league
    # is small — everyone has a serviceable QB.  Raw blended treats Jared Goff
    # (18.8 pts) as equivalent to a top WR, which is wrong.  VOR subtracts the
    # "replacement level" (best waiver-wire player at the position) so that
    # positional scarcity is baked into trade value, not just raw scoring.
    #
    # Example (10-team league):
    #   QB replacement = blended score of the 11th-best QB (last starter's backup)
    #   Goff 18.8 - QB_repl 15.0 = VOR 3.8
    #   McLaurin 12.8 - WR_repl 7.5 = VOR 5.3  → McLaurin is actually more scarce
    STARTERS_PER_TEAM_VOR = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1}
    _all_pos: dict[str, list] = defaultdict(list)
    for _tn_players in team_pos.values():
        for _pos, _players in _tn_players.items():
            _all_pos[_pos].extend(_players)
    for _pos in _all_pos:
        _all_pos[_pos].sort(key=lambda x: x["_blended_score"], reverse=True)

    _repl_level: dict[str, float] = {}
    for _pos, _players in _all_pos.items():
        _n_start   = STARTERS_PER_TEAM_VOR.get(_pos, 1)
        _repl_idx  = num_teams * _n_start          # first player off the waiver wire
        if _repl_idx < len(_players):
            _repl_level[_pos] = max(0.0, _players[_repl_idx]["_blended_score"])
        elif _players:
            _repl_level[_pos] = max(0.0, _players[-1]["_blended_score"] * 0.8)
        else:
            _repl_level[_pos] = 0.0

    def _vor(player: dict) -> float:
        """Value Over Replacement: strips positional scoring baseline so QB/RB/WR compare fairly."""
        return max(0.0, player["_blended_score"] - _repl_level.get(player["position"], 0.0))

    def _cur_score(player: dict) -> float:
        """Current-season avg if ≥2 games played; otherwise fall back to blended.
        Used for the counterparty sanity check — real managers judge trade offers
        by what players are actually doing this season, not preseason projections."""
        wks = player.get("weeks_played_current", 0) or 0
        cur = player.get("current_avg_ppr")
        if wks >= 2 and cur is not None:
            return float(cur)
        return player["_blended_score"]

    my_by_pos = team_pos[ai_team]

    # ── My tradeable players (give candidates) ───────────────────────────────
    # Any player beyond the keep-minimum at their position, with real value.
    my_give_candidates: list[dict] = []
    for pos, players in my_by_pos.items():
        if pos in TRADE_NEVER_GIVE:
            continue
        keep_n = TRADE_KEEP_MIN.get(pos, 1)
        for p in players[keep_n:]:
            if p["_blended_score"] >= 4.0 and p["full_name"] not in dismissed:
                my_give_candidates.append(p)

    if not my_give_candidates:
        return []

    # ── Positions I want to improve ──────────────────────────────────────────
    # RB and WR are always trade targets — every team can use better backs and
    # receivers, and those are where the real trade market value lives.
    # QB is excluded because we never give QB either; TE stays if genuinely weak.
    STARTER_NEED_THRESHOLD = 8.0
    my_needs: set[str] = set()

    # Always seek RB/WR upgrades (the improvement check gates whether a specific
    # deal is actually worth it — no need for a blanket "weak enough" filter here)
    for pos in ("RB", "WR"):
        if my_by_pos.get(pos):   # only if we have players at the position at all
            my_needs.add(pos)

    # TE only if genuinely thin or weak
    for pos, n_starters in {"TE": 1}.items():
        my_at_pos = my_by_pos.get(pos, [])
        if len(my_at_pos) < n_starters:
            my_needs.add(pos)
        elif my_at_pos[n_starters - 1]["_blended_score"] < STARTER_NEED_THRESHOLD:
            my_needs.add(pos)

    if not my_needs:
        return []

    # ── Generate give-packages (sizes 1, 2, 3) ───────────────────────────────
    give_packages: list[list[dict]] = []
    for size in range(1, 4):
        if len(my_give_candidates) >= size:
            for combo in combinations(my_give_candidates, size):
                give_packages.append(list(combo))

    # ── For each other team, collect their spendable players at my need pos ──
    all_proposals: list[dict] = []

    for target_team in range(1, num_teams + 1):
        if target_team == ai_team:
            continue

        target_name = (
            team_names[target_team - 1]
            if team_names and target_team <= len(team_names)
            else f"Team {target_team}"
        )
        their_by_pos = team_pos[target_team]

        # Their spendable players at my need positions
        their_spendable: list[dict] = []
        for need_pos in my_needs:
            their_at_pos = their_by_pos.get(need_pos, [])
            keep_n = TRADE_KEEP_MIN.get(need_pos, 1)
            for p in their_at_pos[keep_n:]:
                if p["_blended_score"] >= 4.0 and p["full_name"] not in dismissed:
                    their_spendable.append(p)

        if not their_spendable:
            continue

        # Generate get-packages (sizes 1, 2, 3)
        get_packages: list[list[dict]] = []
        for size in range(1, 4):
            if len(their_spendable) >= size:
                for combo in combinations(their_spendable, size):
                    get_packages.append(list(combo))

        for give_combo in give_packages:
            give_names = {p["full_name"] for p in give_combo}
            give_value = sum(p["_blended_score"] for p in give_combo)

            # I must keep minimum depth at each give position after the trade
            give_count_by_pos: dict[str, int] = defaultdict(int)
            for p in give_combo:
                give_count_by_pos[p["position"]] += 1

            depth_ok = all(
                len(my_by_pos.get(pos, [])) - n >= TRADE_KEEP_MIN.get(pos, 1)
                for pos, n in give_count_by_pos.items()
            )
            if not depth_ok:
                continue

            # Counterparty value is evaluated inside the get_combo loop via
            # roster simulation — the real question is whether their lineup
            # improves position by position, not whether combined values match.

            for get_combo in get_packages:
                get_names = {p["full_name"] for p in get_combo}

                # No overlap between give and get
                if give_names & get_names:
                    continue

                get_value = sum(p["_blended_score"] for p in get_combo)

                # ── Counterparty roster simulation ───────────────────────────
                # The real question for a trade is not "are the combined values
                # equal?" — it's "does each team's lineup improve at the
                # positions that matter to them?"
                #
                # Classic example (user's case):
                #   We give [12 RB + 12 TE], we get [16 RB]
                #   Target had: RB=16, TE=7  → starting pts = 23
                #   Target gets: RB=12, TE=12 → starting pts = 24  → NET +1 ✓
                #
                # We simulate the trade on the target team's roster and sum the
                # change in blended pts across their starting slots at every
                # affected position.  If their lineup gets materially worse
                # they won't accept; a small dip at one position is fine if
                # another improves.
                _STARTERS_AT = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
                _affected_pos = (
                    {p["position"] for p in give_combo}
                    | {p["position"] for p in get_combo}
                )
                _cpty_net = 0.0
                for _apos in _affected_pos:
                    _n_s = _STARTERS_AT.get(_apos, 1)
                    _their_sc = sorted(
                        [p["_blended_score"] for p in their_by_pos.get(_apos, [])],
                        reverse=True,
                    )
                    _losing  = [p["_blended_score"] for p in get_combo  if p["position"] == _apos]
                    _gaining = [p["_blended_score"] for p in give_combo if p["position"] == _apos]
                    _sim = list(_their_sc)
                    for _s in _losing:
                        try:
                            _sim.remove(_s)
                        except ValueError:
                            if _sim:
                                _sim.remove(min(_sim, key=lambda x: abs(x - _s)))
                    _sim.extend(_gaining)
                    _sim.sort(reverse=True)
                    _cpty_net += sum(_sim[:_n_s]) - sum(_their_sc[:_n_s])

                # Allow a minor dip at one position if another improves enough
                if _cpty_net < -1.5:
                    continue

                # ── Current-season sanity check ──────────────────────────────
                # Blended scores include the preseason prior, which props up
                # underperforming players and can make a bad package look fair.
                # Real managers judge trades by what players are DOING THIS
                # SEASON.  If our give players are averaging 5 pts/gm while
                # their give players average 16, nobody accepts — even if the
                # blended numbers are balanced.
                give_cur = sum(_cur_score(p) for p in give_combo)
                get_cur  = sum(_cur_score(p) for p in get_combo)
                if give_cur < get_cur:
                    if (get_cur - give_cur) / max(get_cur, 0.1) > 0.45:
                        continue

                # Improvement: how much does get_combo improve my lineup?
                # Measure each position separately, require at least one gain.
                improvement = 0.0
                for need_pos in my_needs:
                    pos_gets = [p for p in get_combo if p["position"] == need_pos]
                    if not pos_gets:
                        continue
                    best_get = max(pos_gets, key=lambda x: x["_blended_score"])
                    my_worst = (
                        my_by_pos[need_pos][-1]["_blended_score"]
                        if my_by_pos.get(need_pos) else 0.0
                    )
                    improvement += max(0.0, best_get["_blended_score"] - my_worst)

                if improvement < MIN_TRADE_IMPROVEMENT:
                    continue

                # Trade structure label
                give_n, get_n = len(give_combo), len(get_combo)
                trade_type = f"{give_n}for{get_n}"

                # Sell-high: any give player's blended is propped up by preseason prior
                sell_high = any(
                    p.get("weeks_played_current", 0) >= 2
                    and p["_blended_score"] > 0
                    and (p["_blended_score"] - (p.get("current_avg_ppr") or p["_blended_score"]))
                        / p["_blended_score"] > 0.25
                    for p in give_combo
                )

                # Buy-low: any get player has strong prior but poor recent production
                buy_low = any(
                    p.get("avg_pts_ppr", 0) >= 8.0
                    and p.get("current_avg_ppr") is not None
                    and p.get("weeks_played_current", 0) >= 2
                    and p.get("current_avg_ppr") < p.get("avg_pts_ppr", 0) * 0.7
                    for p in get_combo
                )

                all_proposals.append({
                    "give":        list(give_combo),
                    "get":         list(get_combo),
                    "target_team": target_team,
                    "target_name": target_name,
                    "improvement": round(improvement, 1),
                    "give_value":  round(give_value, 1),
                    "get_value":   round(get_value, 1),
                    "trade_type":  trade_type,
                    "sell_high":   sell_high,
                    "buy_low":     buy_low,
                    "pitch":       None,
                })

    if not all_proposals:
        return []

    # ── Rank: best improvement first, prefer fairer deals on ties ────────────
    all_proposals.sort(
        key=lambda p: (p["improvement"], -abs(p["give_value"] - p["get_value"])),
        reverse=True,
    )

    # ── Greedy dedup: select top proposals with no player overlap ────────────
    selected: list[dict] = []
    used_names: set[str] = set()

    for prop in all_proposals:
        all_names = {p["full_name"] for p in prop["give"]} | {p["full_name"] for p in prop["get"]}
        if not (all_names & used_names):
            selected.append(prop)
            used_names.update(all_names)
        if len(selected) >= max_trades:
            break

    # ── Generate Claude pitches ───────────────────────────────────────────────
    agent = TradeAgent()
    my_team_label = (
        team_names[ai_team - 1]
        if team_names and ai_team <= len(team_names)
        else f"Team {ai_team}"
    )
    surplus_positions = sorted({p["position"] for p in my_give_candidates})
    needs_list        = sorted(my_needs)

    for prop in selected:
        # Attach remaining schedule strength to each player (light tiebreaker only)
        _sched_df = defense_rankings_df if defense_rankings_df is not None else pd.DataFrame()
        for p in prop["give"] + prop["get"]:
            try:
                p["schedule_strength"] = get_remaining_schedule_strength(
                    player_team        = p.get("nfl_team", ""),
                    position           = p.get("position", ""),
                    current_week       = week,
                    season             = season or "",
                    defense_rankings_df= _sched_df if not _sched_df.empty else None,
                )
            except Exception:
                p["schedule_strength"] = None

        prop["pitch"] = agent.generate_pitch(
            give_players    = prop["give"],
            get_players     = prop["get"],
            my_team_name    = my_team_label,
            other_team_name = prop["target_name"],
            my_needs        = needs_list,
            my_surplus      = surplus_positions,
            week            = week,
            season          = season,
            sell_high       = prop["sell_high"],
            buy_low         = prop["buy_low"],
        )

    return selected


def get_lineup_for_week(
    roster: list[dict],
    week: int,
    season: str | None = None,
    all_players: pd.DataFrame | None = None,
    drafted_names: set | None = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (starters, bench, swapped) for a given week.

    Step 1 — blend in this season's results so far (apply_current_season_blend):
    each player gets a "blended_avg_ppr" combining their preseason projection
    with their actual performance through last week. Starters/bench are then
    picked by that blended number, so breakout players can earn a starting
    spot and regressing veterans can lose one.

    Step 2 — bye-week check: if `season` is given and we have saved bye-week
    data for it, any starter whose NFL team is on a bye that week is swapped
    out for the best eligible bench player (by blended avg) whose team is NOT
    on a bye that week.

    Step 3 — waiver flag: if a starter is on a bye and there's no eligible
    bench player to cover the slot, a "waiver_needed" entry is added to
    `swapped` with the ranked list of free-agent candidates. The starter stays
    in the lineup — the UI will present the recommendations one at a time and
    let the user confirm (or reject) each waiver claim before persisting the
    roster change.

    `swapped` entries:
      {"out": player, "in": player, "reason": "bye"}         — bench-covered bye swap
      {"out": player, "in": player, "reason": "cut"}         — bench-covered cut player
      {"out": player, "reason": "waiver_needed",
       "candidates": [...], "slot": str}                     — needs waiver UI action
    """
    enriched = apply_current_season_blend(roster, season, week)
    starters, bench = build_lineup(enriched, sort_key="blended_avg_ppr")
    swapped: list[dict] = []

    bye_weeks = load_bye_weeks(season) if season else {}

    # Map slot label -> eligible positions, for finding bench replacements
    slot_eligibility = {label: positions for label, _, positions in STARTER_SLOTS}

    for i, starter in enumerate(starters):
        team = starter.get("nfl_team")

        on_bye    = bye_weeks.get(team) == week if bye_weeks else False
        was_cut   = not team or team == "FA"   # no NFL team → cut/unsigned

        if not on_bye and not was_cut:
            continue  # starter is active and available this week

        reason_label = "bye" if on_bye else "cut"

        eligible_positions = slot_eligibility[starter["slot"]]

        # Find the best bench player eligible for this slot whose team
        # isn't on bye and hasn't been cut
        def _available_this_week(p):
            t = p.get("nfl_team")
            if not t or t == "FA":
                return False   # bench player also cut — skip
            if bye_weeks.get(t) == week:
                return False   # bench player on bye — skip
            return p["position"] in eligible_positions

        replacement = next(
            (
                p for p in sorted(bench, key=lambda p: p.get("blended_avg_ppr", 0) or 0, reverse=True)
                if _available_this_week(p)
            ),
            None,
        )

        if replacement is not None:
            bench.remove(replacement)
            bench.append(starter)  # benched/cut starter moves to bench
            starters[i] = {**replacement, "slot": starter["slot"]}
            swapped.append({"out": starter, "in": starters[i], "reason": reason_label})
            continue

        # No healthy bench player can cover this slot — surface waiver candidates
        # for the user to act on.  Don't change the lineup here; the UI will
        # present the options and persist the roster change only after the user
        # confirms a successful waiver claim.
        candidates = get_waiver_candidates(
            eligible_positions, week, season, bye_weeks, all_players, drafted_names
        )
        swapped.append({
            "out":        starter,
            "slot":       starter["slot"],
            "reason":     "waiver_needed",
            "out_reason": "cut" if was_cut else "bye",   # WHY this slot needs a waiver pickup
            "candidates": candidates,
        })

    return starters, bench, swapped


# ── LLM lineup rationale ──────────────────────────────────────────────────────
# Mirrors the bye-week explanation, but covers the whole lineup: bye swaps,
# breakout players earning a starting spot, and proven players kept despite
# a slow start. Cached so we don't re-call Claude every time the page reruns
# for the same week's lineup.

@st.cache_data(show_spinner="🧠 Thinking through this week's lineup...")
def get_lineup_rationale(
    league_id: str,
    week: int,
    season: str | None,
    starters: list[dict],
    bench: list[dict],
    swapped: list[dict],
) -> str | None:
    agent = LineupAgent()
    return agent.summarize_lineup(week, season, starters, bench, swapped)


def save_league_roster(league_id: str, picks: list[dict], team_names, ai_team: int, num_teams: int, season: str | None = None):
    """
    Saves the draft/roster results to the league's saved record.

    `draft_picks` is written only on the first save (when no roster_data exists
    yet) and then preserved unchanged on all subsequent saves — even when waiver
    pickups mutate `picks`.  This lets "Reset to Week 1" always restore the
    exact drafted rosters regardless of how many waiver moves have been made.
    """
    leagues = load_leagues()
    for league in leagues:
        if league["id"] == league_id:
            # Carry forward the original draft snapshot if one already exists.
            existing = league.get("roster_data", {})
            draft_picks = existing.get("draft_picks", picks)
            league["roster_data"] = {
                "picks":       picks,
                "draft_picks": draft_picks,   # immutable original — never overwritten
                "team_names":  team_names,
                "ai_team":     ai_team,
                "num_teams":   num_teams,
                "season":      season,
                "saved_at":    datetime.now().isoformat(),
            }
            break
    save_leagues(leagues)


def render_my_leagues():
    st.title("🏆 My Leagues")
    st.markdown(
        "Create a league to save its settings — number of teams, draft rounds, "
        "scoring format, and draft order. When you're ready, click **Begin Draft** "
        "to jump straight into the draft with that league's settings."
    )

    st.divider()

    # ── Create new league button ──────────────────────────────────────────────
    if not st.session_state.creating_league:
        if st.button("➕ Create New League", type="primary"):
            st.session_state.creating_league = True
            st.rerun()
    else:
        render_create_league_form()

    st.divider()

    # ── List existing leagues ─────────────────────────────────────────────────
    leagues = load_leagues()

    if not leagues:
        st.info("No leagues yet. Click **Create New League** to set one up.")
        return

    st.markdown("### Your Leagues")

    for league in leagues:
        with st.expander(f"🏈 {league['name']}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Teams:** {league['num_teams']}")
                st.markdown(f"**Rounds:** {league['num_rounds']}")
                st.markdown(f"**Scoring:** {league['scoring']}")
                st.markdown("**Draft Order:**")
                for i, team_name in enumerate(league["draft_order"], start=1):
                    you = "  *(you — AI managed)*" if i == league["ai_team"] else ""
                    st.markdown(f"  {i}. {team_name}{you}")

            with col2:
                if league.get("roster_data"):
                    if st.button("➡️ Enter", key=f"enter_{league['id']}", use_container_width=True, type="primary"):
                        st.session_state.active_league = league
                        st.session_state.app_page = "League View"
                        st.rerun()

                    if st.button("🔄 Re-draft", key=f"redraft_{league['id']}", use_container_width=True):
                        _begin_draft_for_league(league)

                    if st.button("⏮️ Reset to Week 1", key=f"reset_week1_{league['id']}", use_container_width=True):
                        st.session_state[f"confirm_reset_{league['id']}"] = True

                    if st.session_state.get(f"confirm_reset_{league['id']}"):
                        st.warning(
                            "This will delete all in-season performance data and waiver moves "
                            "for this league's season. The drafted rosters stay intact."
                        )
                        col_confirm, col_cancel = st.columns(2)
                        with col_confirm:
                            if st.button("✅ Confirm", key=f"confirm_reset_yes_{league['id']}", use_container_width=True, type="primary"):
                                rd = league["roster_data"]
                                season = rd.get("season")

                                # Restore picks to the original drafted roster,
                                # undoing all waiver pickups that were saved since.
                                original_picks = rd.get("draft_picks", rd["picks"])
                                save_league_roster(
                                    league_id=league["id"],
                                    picks=original_picks,
                                    team_names=rd.get("team_names"),
                                    ai_team=rd["ai_team"],
                                    num_teams=rd["num_teams"],
                                    season=season,
                                )

                                # Clear all in-season stat files for this season
                                if season:
                                    clear_current_season_data(season)
                                    load_current_season_stats.clear()
                                    get_lineup_rationale.clear()

                                # Clear all waiver session-state keys for this league
                                waiver_keys = [
                                    k for k in list(st.session_state.keys())
                                    if k.startswith(f"waiver_{league['id']}_")
                                ]
                                for k in waiver_keys:
                                    del st.session_state[k]

                                # Reset the week picker to 1
                                st.session_state.selected_week = 1
                                st.session_state[f"confirm_reset_{league['id']}"] = False

                                # Refresh the active league reference if it's the open one
                                refreshed = load_leagues()
                                updated = next((l for l in refreshed if l["id"] == league["id"]), None)
                                if updated and st.session_state.get("active_league", {}).get("id") == league["id"]:
                                    st.session_state.active_league = updated

                                st.success("✅ Reset to Week 1. Roster restored to original draft — all waiver moves and in-season data cleared.")
                                st.rerun()
                        with col_cancel:
                            if st.button("✗ Cancel", key=f"confirm_reset_no_{league['id']}", use_container_width=True):
                                st.session_state[f"confirm_reset_{league['id']}"] = False
                                st.rerun()
                else:
                    if st.button("▶️ Begin Draft", key=f"begin_{league['id']}", use_container_width=True, type="primary"):
                        _begin_draft_for_league(league)

                if st.button("🗑️ Delete League", key=f"delete_{league['id']}", use_container_width=True):
                    leagues = [l for l in leagues if l["id"] != league["id"]]
                    save_leagues(leagues)
                    st.rerun()


def render_create_league_form():
    st.markdown("### Create New League")

    # num_teams must live OUTSIDE the form so changing it rerenders the team-name inputs
    col1, col2 = st.columns(2)
    with col1:
        num_teams = st.selectbox(
            "Number of teams", options=TEAM_OPTIONS,
            index=TEAM_OPTIONS.index(st.session_state.get("create_num_teams", TEAM_OPTIONS[1])),
            key="create_num_teams",
        )

    with st.form("create_league_form"):
        league_name = st.text_input("League name", placeholder="e.g. Office League 2026")

        col_r, col_s = st.columns(2)
        with col_r:
            num_rounds = st.number_input("Number of rounds in the draft", min_value=5, max_value=20, value=15)
        with col_s:
            scoring = st.selectbox("Scoring format", options=SCORING_OPTIONS, index=0)

        st.markdown("**Draft order** — enter each team's name in draft order (pick 1 first), "
                     "and select which slot is your AI-managed team.")

        # Team name inputs — driven by num_teams which is outside the form
        team_name_inputs = []
        for i in range(num_teams):
            default_name = f"Team {i + 1}"
            team_name_inputs.append(
                st.text_input(f"Pick {i + 1}", value=default_name, key=f"new_league_team_{i}")
            )

        ai_team = st.selectbox(
            "Which draft slot is your AI-managed team?",
            options=list(range(1, num_teams + 1)),
            index=0,
            help="Select your draft position (1 = first pick).",
        )

        col_save, col_cancel = st.columns(2)
        with col_save:
            submitted = st.form_submit_button("💾 Save League", type="primary", use_container_width=True)
        with col_cancel:
            cancelled = st.form_submit_button("Cancel", use_container_width=True)

    if cancelled:
        st.session_state.creating_league = False
        st.rerun()

    if submitted:
        if not league_name.strip():
            st.error("⚠️ Please enter a league name.")
            return

        new_league = {
            "id":          str(uuid.uuid4()),
            "name":        league_name.strip(),
            "num_teams":   int(num_teams),
            "num_rounds":  int(num_rounds),
            "scoring":     scoring,
            "draft_order": [name.strip() or f"Team {i+1}" for i, name in enumerate(team_name_inputs)],
            "ai_team":     int(ai_team),
            "created_at":  datetime.now().isoformat(),
        }

        leagues = load_leagues()
        leagues.append(new_league)
        save_leagues(leagues)

        st.session_state.creating_league = False
        st.success(f"✅ League '{new_league['name']}' saved!")
        st.rerun()


def _begin_draft_for_league(league: dict):
    """Pre-fills draft settings from a saved league and sends the user to the Draft page."""
    st.session_state.num_teams       = league["num_teams"]
    st.session_state.num_rounds      = league["num_rounds"]
    st.session_state.ai_team         = league["ai_team"]
    st.session_state.team_names      = league["draft_order"]
    st.session_state.active_league   = league
    st.session_state.app_page        = "Draft"
    st.session_state.draft_started   = False
    st.session_state.pending_draft   = None
    st.session_state.current_pick    = 1
    st.session_state.picks           = []
    st.session_state.recommendation  = None
    st.session_state.selected_player = None
    st.session_state.last_ai_pick    = None
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  WAIVER WIRE UI — Interactive pick-up/drop recommendation flow
# ═══════════════════════════════════════════════════════════════════════════════

def render_waiver_ui(
    league: dict,
    week: int,
    season: str | None,
    bench: list[dict],
    waiver_needs: list[dict],
    picks: list[dict],
    ai_team: int,
    roster_data: dict,
) -> None:
    """
    For each starting slot that can't be covered by a bench player this week,
    shows one waiver-wire candidate at a time and asks the user to confirm
    whether they were actually able to claim that player.

    - "Yes, I claimed them" → persists the drop/add to the roster and reruns.
    - "Not available, show next" → advances to the next candidate and reruns.
    - Once all candidates are exhausted for a slot, a fallback message is shown.

    State is stored in st.session_state keyed by (league_id, week) so it
    survives Streamlit reruns for the same week, but resets when the week changes.

    Waiver-wire order in real fantasy leagues is determined by the inverse of
    current standings (worst record → first priority). We display a reminder
    of this but don't enforce the order — the user confirms what actually
    happened in their league platform.
    """
    waiver_key = f"waiver_{league['id']}_{week}"

    # Build a stable list of out-player names for this set of needs
    current_out_names = [s["out"]["full_name"] for s in waiver_needs]

    # Initialize (or reinitialize if the needs list has changed, e.g. after a
    # prior confirmed pickup removed one slot from waiver_needs on rerun)
    if (
        waiver_key not in st.session_state
        or st.session_state[waiver_key].get("out_names") != current_out_names
    ):
        st.session_state[waiver_key] = {
            "out_names": current_out_names,
            "indices":   {name: 0 for name in current_out_names},
        }

    state = st.session_state[waiver_key]

    st.divider()
    st.markdown("### 📋 Waiver Wire Action Needed")
    st.caption(
        "💡 **Waiver wire order** is based on league standings — the team with the "
        "worst record gets first priority. Check your league platform to see if you "
        "can claim each recommended player before confirming here."
    )

    for s in waiver_needs:
        out_player = s["out"]
        out_name   = out_player["full_name"]
        candidates = s["candidates"]
        idx        = state["indices"].get(out_name, 0)
        out_reason = s.get("out_reason", "bye")   # "bye" or "cut"

        # Explain WHY this slot needs a waiver pickup
        if out_reason == "cut":
            st.markdown(
                f"**{out_name}** is no longer on an NFL roster and needs to be replaced. "
                f"Claim a free agent for the **{s['slot']}** slot below."
            )
        else:
            st.markdown(
                f"**{out_name}** ({out_player.get('nfl_team', '')}) is on bye in Week {week} "
                f"and you have no eligible bench player for the **{s['slot']}** slot."
            )

        if not candidates or idx >= len(candidates):
            st.error(
                f"⚠️ No more waiver candidates available for the {s['slot']} slot. "
                f"You may need to manage this slot manually in your league platform."
            )
            st.divider()
            continue

        candidate    = candidates[idx]
        total        = len(candidates)
        blended      = candidate.get("blended_avg_ppr", candidate.get("avg_pts_ppr", 0)) or 0
        preseason    = candidate.get("avg_pts_ppr", 0) or 0
        weeks_played = candidate.get("weeks_played_current", 0)
        current_avg  = candidate.get("current_avg_ppr")

        # Drop logic:
        #  - Cut player (no team): drop the cut player themselves — they're useless
        #  - K or DEF on bye: drop the K/DEF themselves. They can't score this week
        #    anyway, and kickers/defenses outside the elite tier are easy to re-claim
        #    off waivers the following week. Never burn a skill-position player for a
        #    throwaway kicker pickup.
        #  - Skill position on bye: drop the weakest bench player (the bye player
        #    stays on the roster so you can start them next week).
        out_pos = out_player.get("position", "")
        if out_reason == "cut":
            drop_candidate  = out_player   # no team → drop them
            drop_is_out_player = True
        elif out_pos in ("K", "DEF"):
            drop_candidate  = out_player   # on bye → drop themselves, re-add next week
            drop_is_out_player = True
        else:
            drop_candidate = (
                min(bench, key=lambda p: p.get("blended_avg_ppr", 0) or 0)
                if bench else None
            )
            drop_is_out_player = False

        # ── Hold vs. flip signal ──────────────────────────────────────────────
        # Tell the user upfront whether this is a one-week rental or a keeper.
        # Decision is based on position, preseason value, and current-season
        # production (if any games have been played).
        ELITE_K_DEF_THRESHOLD   = 8.0   # pts/gm — Aubrey/Boswell tier
        KEEP_SKILL_THRESHOLD    = 8.0   # pts/gm — roughly a reliable starter floor
        breakout = candidate.get("breakout_score")
        strong_breakout = isinstance(breakout, (int, float)) and not (isinstance(breakout, float) and pd.isna(breakout)) and breakout >= 50

        if out_reason == "bye":   # only applies to bye fills, not cut replacements
            if out_pos in ("K", "DEF"):
                if preseason >= ELITE_K_DEF_THRESHOLD:
                    hold_msg = (
                        f"💡 **Elite {out_pos} — may be worth keeping.** "
                        f"{candidate['full_name']} is a top-tier {out_pos}. "
                        f"Re-evaluate after the week whether dropping them makes sense."
                    )
                    hold_color = "success"
                else:
                    hold_msg = (
                        f"🔄 **One-week rental.** Drop {candidate['full_name']} after Week {week} "
                        f"and re-add {out_name} when they're off bye — "
                        f"most {out_pos}s are easy to re-claim off waivers."
                    )
                    hold_color = "warning"
            else:
                # Skill position fill-in
                effective_val = blended if weeks_played else preseason
                if effective_val >= KEEP_SKILL_THRESHOLD or strong_breakout:
                    hold_msg = (
                        f"⭐ **Worth keeping an eye on.** {candidate['full_name']} has real upside "
                        f"({'breakout candidate · ' if strong_breakout else ''}preseason {preseason:.1f} pts/gm). "
                        f"Keep them and re-evaluate your depth next week."
                    )
                    hold_color = "success"
                else:
                    hold_msg = (
                        f"🔄 **Short-term fill-in.** {candidate['full_name']} doesn't have a lot of upside. "
                        f"Drop them once {out_name} is off bye if a better option is available."
                    )
                    hold_color = "warning"
        else:
            hold_msg = None
            hold_color = None

        col_pickup, col_drop = st.columns(2)

        with col_pickup:
            st.markdown("**🟢 Recommended pickup:**")
            detail_parts = [f"Preseason avg: **{preseason:.1f}** pts/gm"]
            if weeks_played and current_avg is not None:
                detail_parts.append(
                    f"{season} avg: **{current_avg:.1f}** over {weeks_played} game(s)"
                )
                detail_parts.append(f"Blended: **{blended:.1f}**")
            st.info(
                f"**{candidate['full_name']}** "
                f"({candidate['position']}, {candidate['nfl_team']})\n\n"
                + " · ".join(detail_parts)
                + f"\n\n*Candidate {idx + 1} of {total}*"
            )
            if hold_msg:
                if hold_color == "success":
                    st.success(hold_msg)
                else:
                    st.warning(hold_msg)

        with col_drop:
            if drop_candidate:
                drop_blended = drop_candidate.get("blended_avg_ppr", 0) or 0
                st.markdown("**🔴 Recommended drop:**")
                if drop_is_out_player:
                    if out_reason == "bye" and out_pos in ("K", "DEF"):
                        drop_note = (
                            f"*(on bye this week — drop them and re-add next week. "
                            f"Most {out_pos}s are easy to re-claim off waivers.)*"
                        )
                    else:
                        drop_note = "*(no longer on an NFL roster — release them)*"
                    st.info(
                        f"**{drop_candidate['full_name']}** "
                        f"({drop_candidate['position']}, {drop_candidate.get('nfl_team', 'FA')})\n\n"
                        + drop_note
                    )
                else:
                    st.info(
                        f"**{drop_candidate['full_name']}** "
                        f"({drop_candidate['position']}, {drop_candidate.get('nfl_team', '')})\n\n"
                        f"Blended avg: **{drop_blended:.1f}** pts/gm *(lowest on bench)*"
                    )
            else:
                st.warning("Bench is empty — add the pickup without dropping anyone.")

        col_yes, col_no = st.columns(2)

        with col_yes:
            yes_label = f"✅ Yes — I claimed {candidate['full_name']}"
            if st.button(yes_label, key=f"waiver_yes_{league['id']}_{week}_{out_name}_{idx}",
                         use_container_width=True, type="primary"):
                # Persist the pickup (and drop if bench non-empty) to the roster
                updated_picks = list(picks)

                if drop_candidate:
                    updated_picks = [
                        p for p in updated_picks
                        if not (
                            p["team_num"] == ai_team
                            and p["full_name"] == drop_candidate["full_name"]
                        )
                    ]

                updated_picks.append({
                    "full_name":   candidate["full_name"],
                    "position":    candidate["position"],
                    "nfl_team":    candidate["nfl_team"],
                    "team_num":    ai_team,
                    "pick_number": max((p["pick_number"] for p in updated_picks), default=0) + 1,
                    "avg_pts_ppr": candidate.get("avg_pts_ppr", 0) or 0,
                })

                save_league_roster(
                    league_id=league["id"],
                    picks=updated_picks,
                    team_names=roster_data.get("team_names"),
                    ai_team=ai_team,
                    num_teams=roster_data["num_teams"],
                    season=season,
                )
                get_lineup_rationale.clear()

                # Refresh the active league in session state so the next render
                # sees the updated roster (and no longer flags this slot as
                # waiver_needed, since the new player is now on the bench)
                refreshed = load_leagues()
                st.session_state.active_league = next(
                    (l for l in refreshed if l["id"] == league["id"]), league
                )
                # Clear waiver state so it re-evaluates from the new roster
                if waiver_key in st.session_state:
                    del st.session_state[waiver_key]
                st.rerun()

        with col_no:
            no_label = f"❌ Not available — show next option"
            if st.button(no_label, key=f"waiver_no_{league['id']}_{week}_{out_name}_{idx}",
                         use_container_width=True):
                state["indices"][out_name] = idx + 1
                st.rerun()

        st.divider()


def render_def_streaming_ui(
    league: dict,
    week: int,
    season: str | None,
    starters: list[dict],
    all_players: pd.DataFrame,
    drafted_names: set,
    picks: list[dict],
    ai_team: int,
    roster_data: dict,
    offense_rankings_df: pd.DataFrame,
    bye_weeks: dict,
    def_quality_df: pd.DataFrame | None = None,
) -> None:
    """
    DEF streaming recommendations: shows the current DEF's upcoming opponent
    strength, and surfaces better free-agent defenses when the matchup is tough.

    Strategy: defenses should be streamed week-to-week against the weakest
    available offense. A DEF facing a strong offense (top-third of scoring) is
    a candidate to drop. Available defenses with weak opponents are surfaced as
    pickups, scored 75% matchup quality + 25% DEF unit quality.
    """
    if week <= 1 or offense_rankings_df.empty or not season:
        return

    # ── Find the current started DEF ─────────────────────────────────────────
    current_def = next(
        (p for p in starters if p["position"] == "DEF"),
        None,
    )

    # ── Find available (undrafted) DEFs not on bye ────────────────────────────
    all_def_teams = (
        all_players[all_players["position"] == "DEF"]["full_name"].tolist()
        if "position" in all_players.columns else []
    )
    available_def_names = [n for n in all_def_teams if n not in drafted_names]
    # Extract team abbr from "{TEAM} Defense" naming convention
    available_def_teams = []
    for name in available_def_names:
        team_abbr = name.replace(" Defense", "").strip()
        if bye_weeks.get(team_abbr) != week:   # skip those on bye
            available_def_teams.append(team_abbr)

    # Get current DEF's opponent strength
    current_opp = None
    current_def_team = None
    if current_def:
        current_def_team = current_def.get("nfl_team", "")
        current_opp = get_opponent_offense(
            current_def_team, week, season, offense_rankings_df
        )

    # Get streaming recommendations
    recs = get_def_streaming_recs(
        week        = week,
        season      = season,
        available_def_teams = available_def_teams,
        offense_rankings_df = offense_rankings_df,
        def_quality_df      = def_quality_df,
        top_n       = 3,
    )

    # ── Decide whether to show the section ────────────────────────────────────
    # Always show if there are recs and the current DEF has a tough matchup.
    # Show with a softer header if neutral/no-data matchup but good options exist.
    current_is_tough = (
        current_opp is not None and current_opp["offense_label"] == "strong"
    )
    current_is_neutral = (
        current_opp is not None and current_opp["offense_label"] == "average"
    )
    has_easy_option = any(r["offense_label"] == "weak" for r in recs)

    # Don't surface the section at all if current matchup is easy and nothing better exists
    if not current_is_tough and not (current_is_neutral and has_easy_option) and not (current_opp is None and recs):
        # Current DEF has an easy matchup — just note it quietly and skip the expander
        if current_def and current_opp and current_opp["offense_label"] == "weak":
            return  # easy matchup, no need to say anything
        if not recs:
            return

    # Build the expander label
    if current_is_tough:
        expander_label = f"🔄 DEF Streaming — consider a Week {week} upgrade"
        expanded = True
    elif recs and has_easy_option:
        expander_label = f"🔄 DEF Streaming — Week {week} options"
        expanded = False
    else:
        return

    dismissed_key = f"def_stream_dismissed_{league['id']}_{week}"
    if dismissed_key not in st.session_state:
        st.session_state[dismissed_key] = set()
    dismissed: set[str] = st.session_state[dismissed_key]

    # Filter out dismissed recs
    recs = [r for r in recs if r["team"] not in dismissed]
    if not recs and not current_is_tough:
        return

    with st.expander(expander_label, expanded=expanded):
        st.caption(
            "The best fantasy strategy for DEF is to stream week-to-week — drop your current "
            "defense and pick up whichever free-agent DEF has the easiest matchup. A weak "
            "opponent offense means more fantasy points: more sacks, turnovers, and points allowed."
        )

        # ── Current DEF situation ─────────────────────────────────────────────
        _using_preseason = (week <= 3)  # weeks 1-3 use expert preseason rankings
        if current_def and current_opp:
            opp_emoji = OFFENSE_EMOJI.get(current_opp["offense_label"], "🟡")
            if _using_preseason:
                _pts_note  = "preseason projection"
                _rank_note = f"#{current_opp['rank']}"
            else:
                _pts_note  = f"{current_opp['avg_pts_scored']:.1f} pts/gm season avg"
                _rank_note = f"#{current_opp['rank']}"
            st.markdown(f"**Your DEF: {current_def['full_name']}**")
            st.markdown(
                f"Week {week} matchup: **{opp_emoji} vs {current_opp['opponent']}** "
                f"({current_opp['offense_label']} offense · {_rank_note} "
                f"in scoring · {_pts_note})"
            )
            if current_is_tough:
                st.warning(
                    f"⚠️ {current_opp['opponent']} has one of the strongest offenses in the league "
                    f"— a tough week to ride {current_def['full_name']}. Check the options below."
                )
            elif current_is_neutral:
                st.info(
                    f"🟡 {current_opp['opponent']} is an average offense. You can stick with "
                    f"{current_def['full_name']}, but a better matchup may be available below."
                )
        elif current_def and not current_opp:
            st.markdown(f"**Your DEF:** {current_def['full_name']} · matchup data not available yet")
        elif not current_def:
            st.markdown("**No DEF currently starting** — pick one up from the options below.")

        if recs:
            st.markdown("---")
            st.markdown(f"**Available DEFs with easier matchups this week:**")

        for i, rec in enumerate(recs):
            opp_emoji  = OFFENSE_EMOJI.get(rec["offense_label"], "🟡")
            n_teams    = len(offense_rankings_df["team"].unique()) if not offense_rankings_df.empty else 32
            rank_label = f"#{rec['rank']} of {n_teams}"

            # DEF quality badge (only shown when real data is available)
            _def_quality_label = rec.get("def_quality_label")
            _avg_def_pts       = rec.get("avg_def_pts")
            if _def_quality_label and _avg_def_pts is not None:
                _def_emoji = {"strong": "💪", "average": "🤝", "weak": "⚠️"}.get(_def_quality_label, "🤝")
                _def_badge = f" · {_def_emoji} {_def_quality_label} DEF ({_avg_def_pts:.1f} pts/gm)"
            else:
                _def_badge = ""

            col_info, col_btn = st.columns([3, 1])
            with col_info:
                _rec_pts_note = "preseason projection" if _using_preseason else f"{rec['avg_pts_scored']:.1f} pts/gm season avg"
                st.success(
                    f"**{rec['full_name']}** — "
                    f"{opp_emoji} vs {rec['opponent']} "
                    f"({rec['offense_label']} offense · {rank_label} · "
                    f"{_rec_pts_note}{_def_badge})"
                )
            with col_btn:
                if current_def:
                    btn_label = f"✅ Pick up {rec['team']}"
                    btn_key   = f"def_stream_{league['id']}_{week}_{rec['team']}"
                    if st.button(btn_label, key=btn_key, use_container_width=True, type="primary"):
                        # Drop current DEF, add streamed DEF
                        updated_picks = [
                            p for p in picks
                            if not (
                                p["team_num"] == ai_team
                                and p["position"] == "DEF"
                                and p["full_name"] == current_def["full_name"]
                            )
                        ]
                        updated_picks.append({
                            "full_name":   rec["full_name"],
                            "position":    "DEF",
                            "nfl_team":    rec["nfl_team"],
                            "team_num":    ai_team,
                            "pick_number": max(
                                (p["pick_number"] for p in updated_picks), default=0
                            ) + 1,
                            "avg_pts_ppr": 0,  # DEF pts are matchup-driven, not history-driven
                        })
                        save_league_roster(
                            league_id  = league["id"],
                            picks      = updated_picks,
                            team_names = roster_data.get("team_names"),
                            ai_team    = ai_team,
                            num_teams  = roster_data["num_teams"],
                            season     = season,
                        )
                        get_defense_rankings.clear()
                        get_offense_rankings.clear()
                        get_lineup_rationale.clear()
                        refreshed = load_leagues()
                        st.session_state.active_league = next(
                            (l for l in refreshed if l["id"] == league["id"]), league
                        )
                        st.rerun()
                else:
                    st.button(
                        f"✅ Add {rec['team']}",
                        key=f"def_stream_add_{league['id']}_{week}_{rec['team']}",
                        use_container_width=True,
                        type="primary",
                        disabled=True,   # no DEF to drop — handled by waiver UI
                    )

            # Dismiss button
            if st.button(
                "Skip",
                key=f"def_stream_skip_{league['id']}_{week}_{rec['team']}",
                use_container_width=False,
            ):
                dismissed.add(rec["team"])
                st.rerun()

            if i < len(recs) - 1:
                st.divider()


def render_upgrade_ui(
    league: dict,
    week: int,
    season: str | None,
    enriched_roster: list[dict],
    all_players: pd.DataFrame,
    drafted_names: set,
    picks: list[dict],
    ai_team: int,
    roster_data: dict,
) -> None:
    """
    Surfaces free-agent upgrades: FAs who are better than the weakest same-
    position player on the roster by at least MIN_UPGRADE_IMPROVEMENT pts/gm.

    Shown in a collapsible expander so it doesn't clutter the lineup view.
    Each suggestion has:
      ✅ "Add [player]"  — persists the drop/add and reruns.
      ❌ "Not available" — dismisses that specific FA for this week (doesn't
                           affect other suggestions or future weeks).
    """
    dismissed_key = f"upgrade_dismissed_{league['id']}_{week}"
    if dismissed_key not in st.session_state:
        st.session_state[dismissed_key] = set()

    dismissed: set[str] = st.session_state[dismissed_key]

    recs = get_upgrade_recommendations(
        enriched_roster, season, week, all_players, drafted_names, dismissed
    )

    if not recs:
        return

    label = f"📈 {len(recs)} Waiver Wire Upgrade{'s' if len(recs) != 1 else ''} Available"
    with st.expander(label, expanded=False):
        st.caption(
            "These free agents project better than the weakest player at their "
            "position on your roster. Players marked ⚠️ have had multiple weeks of "
            "near-zero production — those roster spots are being wasted and should be "
            "prioritized for pickup regardless of their preseason projection. "
            "Check waiver priority in your league platform before confirming."
        )

        for i, rec in enumerate(recs):
            pickup      = rec["pickup"]
            drop        = rec["drop"]
            improvement = rec["improvement"]
            pos         = rec["position"]
            is_cleanup  = rec.get("cleanup", False)

            # ── Cleanup rec: extra DEF/K that shouldn't be on the roster ─────
            if is_cleanup:
                st.warning(
                    f"⚠️ **Roster issue — extra {pos} detected.**  \n"
                    f"You have more than one {pos} on your roster. You can only ever start one and "
                    f"a backup {pos} wastes a valuable roster spot. Drop "
                    f"**{drop['full_name']}** (the weaker one) immediately."
                )
                drop_blended = drop.get("blended_avg_ppr", 0) or 0
                st.error(
                    f"➖ **{drop['full_name']}** "
                    f"({drop['position']}, {drop.get('nfl_team', '')})\n\n"
                    f"Blended: **{drop_blended:.1f}** pts/gm · Drop this — you don't need two {pos}s"
                )
                if st.button(
                    f"✅ Drop {drop['full_name']} (free up roster spot)",
                    key=f"cleanup_drop_{league['id']}_{week}_{drop['full_name']}",
                    use_container_width=True,
                    type="primary",
                ):
                    updated_picks = [
                        p for p in picks
                        if not (p["team_num"] == ai_team and p["full_name"] == drop["full_name"])
                    ]
                    save_league_roster(
                        league_id=league["id"],
                        picks=updated_picks,
                        team_names=roster_data.get("team_names"),
                        ai_team=ai_team,
                        num_teams=roster_data["num_teams"],
                        season=season,
                    )
                    get_lineup_rationale.clear()
                    refreshed = load_leagues()
                    st.session_state.active_league = next(
                        (l for l in refreshed if l["id"] == league["id"]), league
                    )
                    st.rerun()
                if i < len(recs) - 1:
                    st.divider()
                continue

            # ── Normal upgrade rec ────────────────────────────────────────────
            st.markdown(f"**{pos} · +{improvement:.1f} pts/gm projected improvement**")

            fa_blended   = pickup.get("blended_avg_ppr", pickup.get("avg_pts_ppr", 0)) or 0
            fa_preseason = pickup.get("avg_pts_ppr", 0) or 0
            weeks_played = pickup.get("weeks_played_current", 0)
            drop_blended = drop.get("blended_avg_ppr", 0) or 0

            col_add, col_drop = st.columns(2)

            with col_add:
                detail_parts = [f"Preseason: **{fa_preseason:.1f}**"]
                if weeks_played and pickup.get("current_avg_ppr") is not None:
                    detail_parts.append(
                        f"{season} avg: **{pickup['current_avg_ppr']:.1f}** ({weeks_played} gm)"
                    )
                detail_parts.append(f"Blended: **{fa_blended:.1f}**")
                st.success(
                    f"➕ **{pickup['full_name']}** "
                    f"({pickup['position']}, {pickup['nfl_team']})\n\n"
                    + " · ".join(detail_parts)
                )

            with col_drop:
                drop_cur_avg   = drop.get("current_avg_ppr")
                drop_wks       = drop.get("weeks_played_current", 0)
                drop_is_low    = drop.get("_is_low_prod", False)
                drop_lines     = [f"Blended: **{drop_blended:.1f}** pts/gm"]
                if drop_cur_avg is not None and drop_wks:
                    drop_lines.append(f"{season} avg: **{drop_cur_avg:.1f}** ({drop_wks} gm)")
                if drop_is_low:
                    drop_lines.append("⚠️ *Near-zero production — roster spot being wasted*")
                st.error(
                    f"➖ **{drop['full_name']}** "
                    f"({drop['position']}, {drop.get('nfl_team', '')})\n\n"
                    + "\n\n".join(drop_lines)
                )

            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button(
                    f"✅ Add {pickup['full_name']}",
                    key=f"upg_yes_{league['id']}_{week}_{pickup['full_name']}",
                    use_container_width=True,
                    type="primary",
                ):
                    updated_picks = [
                        p for p in picks
                        if not (p["team_num"] == ai_team and p["full_name"] == drop["full_name"])
                    ]
                    updated_picks.append({
                        "full_name":   pickup["full_name"],
                        "position":    pickup["position"],
                        "nfl_team":    pickup["nfl_team"],
                        "team_num":    ai_team,
                        "pick_number": max((p["pick_number"] for p in updated_picks), default=0) + 1,
                        "avg_pts_ppr": pickup.get("avg_pts_ppr", 0) or 0,
                    })
                    save_league_roster(
                        league_id=league["id"],
                        picks=updated_picks,
                        team_names=roster_data.get("team_names"),
                        ai_team=ai_team,
                        num_teams=roster_data["num_teams"],
                        season=season,
                    )
                    get_lineup_rationale.clear()
                    refreshed = load_leagues()
                    st.session_state.active_league = next(
                        (l for l in refreshed if l["id"] == league["id"]), league
                    )
                    st.rerun()

            with col_no:
                if st.button(
                    "❌ Not available",
                    key=f"upg_no_{league['id']}_{week}_{pickup['full_name']}",
                    use_container_width=True,
                ):
                    dismissed.add(pickup["full_name"])
                    st.rerun()

            if i < len(recs) - 1:
                st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  TRADE RECOMMENDATIONS UI
# ═══════════════════════════════════════════════════════════════════════════════

def render_trade_ui(
    league:               dict,
    week:                 int,
    season:               str | None,
    my_roster:            list[dict],
    all_picks:            list[dict],
    all_players:          pd.DataFrame,
    num_teams:            int,
    ai_team:              int,
    team_names:           list[str],
    roster_data:          dict,
    defense_rankings_df:  pd.DataFrame | None = None,
):
    """
    Renders trade proposals for the AI team.  Supports any package size
    (1-for-1, 2-for-1, 3-for-1, 1-for-2, etc.).  Only shown when there
    are meaningful opportunities.  Proposals can be dismissed per session.
    """
    dismiss_key = f"dismissed_trades_{league['id']}_{week}"
    if dismiss_key not in st.session_state:
        st.session_state[dismiss_key] = set()
    dismissed = st.session_state[dismiss_key]

    proposals = get_trade_recommendations(
        my_roster            = my_roster,
        all_picks            = all_picks,
        all_players          = all_players,
        season               = season,
        week                 = week,
        num_teams            = num_teams,
        ai_team              = ai_team,
        team_names           = team_names,
        dismissed            = dismissed,
        max_trades           = 3,
        defense_rankings_df  = defense_rankings_df,
    )

    if not proposals:
        return

    st.divider()
    st.markdown("### 💱 Trade Opportunities")
    st.caption(
        "Trade proposals are only generated when there's a fair-value exchange that "
        "meaningfully improves your roster. You'll need to submit the actual trade "
        "through your league platform."
    )

    def _player_card(p: dict, is_sell_high: bool = False, is_buy_low: bool = False) -> str:
        """Returns a markdown string summarising one player for display."""
        prior = p.get("avg_pts_ppr", 0) or 0
        val   = p.get("blended_avg_ppr", prior) or 0
        cur   = p.get("current_avg_ppr")
        wks   = p.get("weeks_played_current", 0)
        parts = [f"Preseason **{prior:.1f}**"]
        if wks and cur is not None:
            parts.append(f"{season} avg **{cur:.1f}** ({wks} gm)")
        parts.append(f"Blended **{val:.1f}**")
        detail = " · ".join(parts)
        flag = " 📈" if is_sell_high else (" 📉" if is_buy_low else "")
        return (
            f"**{p['full_name']}{flag}** ({p['position']}, {p.get('nfl_team', '')})\n\n"
            + detail
        )

    for i, prop in enumerate(proposals):
        give_list   = prop["give"]   # list of player dicts
        get_list    = prop["get"]    # list of player dicts
        target      = prop["target_name"]
        improvement = prop["improvement"]
        give_value  = prop["give_value"]
        get_value   = prop["get_value"]
        trade_type  = prop["trade_type"]   # "1for1", "2for1", etc.
        sell_high   = prop["sell_high"]
        buy_low     = prop["buy_low"]

        give_n, get_n = len(give_list), len(get_list)

        # ── Header ────────────────────────────────────────────────────────────
        if give_n == get_n == 1:
            structure_label = "1-for-1 swap"
        elif give_n > get_n:
            structure_label = f"{give_n}-for-{get_n} — package {give_n} for a star"
        else:
            structure_label = f"{give_n}-for-{get_n} — trade a star for depth"

        angle_tags = ""
        if sell_high: angle_tags += " 📈 *Sell High*"
        if buy_low:   angle_tags += " 📉 *Buy Low*"

        st.markdown(
            f"**Trade with {target} · {structure_label} · +{improvement:.1f} pts/gm**"
            + angle_tags
        )

        # Build sell-high set (players whose blended >> their actual production)
        sell_high_names = {
            p["full_name"] for p in give_list
            if p.get("weeks_played_current", 0) >= 2
            and p.get("blended_avg_ppr", 0) > 0
            and (p.get("blended_avg_ppr", 0) - (p.get("current_avg_ppr") or p.get("blended_avg_ppr", 0)))
                / p.get("blended_avg_ppr", 1) > 0.20
        }
        buy_low_names = {
            p["full_name"] for p in get_list
            if p.get("avg_pts_ppr", 0) >= 8.0
            and p.get("current_avg_ppr") is not None
            and p.get("current_avg_ppr") < p.get("avg_pts_ppr", 0) * 0.7
        }

        col_give, col_get = st.columns(2)

        with col_give:
            give_header = f"📤 **You give ({give_n} player{'s' if give_n > 1 else ''}, combined {give_value:.1f} pts):**"
            st.markdown(give_header)
            for p in give_list:
                is_sh = p["full_name"] in sell_high_names
                st.warning(_player_card(p, is_sell_high=is_sh))

        with col_get:
            get_header = f"📥 **You get ({get_n} player{'s' if get_n > 1 else ''}, combined {get_value:.1f} pts):**"
            st.markdown(get_header)
            for p in get_list:
                is_bl = p["full_name"] in buy_low_names
                st.success(_player_card(p, is_buy_low=is_bl))

        # Value balance note
        gap_pct = abs(give_value - get_value) / max(give_value, get_value, 1) * 100
        if gap_pct <= 10:
            st.caption(f"⚖️ Near-equal value exchange ({give_value:.1f} vs {get_value:.1f} pts combined)")
        elif give_value > get_value:
            st.caption(f"⚖️ You're giving slightly more value ({give_value:.1f} vs {get_value:.1f}) — justified if it fills a real need")
        else:
            st.caption(f"⚖️ You're getting slightly more value ({get_value:.1f} vs {give_value:.1f}) — a good deal if they accept")

        if prop.get("pitch"):
            st.info(f"🧠 {prop['pitch']}")

        # Dismiss key uses all player names to be unique
        give_key = "_".join(p["full_name"].replace(" ", "") for p in give_list)
        get_key  = "_".join(p["full_name"].replace(" ", "") for p in get_list)
        if st.button(
            "❌ Dismiss this trade",
            key=f"trade_dismiss_{league['id']}_{week}_{give_key}_{get_key}",
            use_container_width=False,
        ):
            for p in give_list + get_list:
                dismissed.add(p["full_name"])
            st.rerun()

        if i < len(proposals) - 1:
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
#  LEAGUE VIEW — My team & other teams' rosters after the draft
# ═══════════════════════════════════════════════════════════════════════════════

def render_league_view():
    league = st.session_state.get("active_league")
    if not league or not league.get("roster_data"):
        st.warning("⚠️ This league hasn't been drafted yet.")
        if st.button("⬅️ Back to My Leagues"):
            st.session_state.app_page = "My Leagues"
            st.rerun()
        return

    roster_data = league["roster_data"]
    picks       = roster_data["picks"]
    team_names  = roster_data["team_names"] or [f"Team {i+1}" for i in range(roster_data["num_teams"])]
    ai_team     = roster_data["ai_team"]
    num_teams   = roster_data["num_teams"]

    if st.button("⬅️ Back to My Leagues"):
        st.session_state.app_page = "My Leagues"
        st.rerun()

    st.title(f"🏈 {league['name']}")

    def roster_for(team_num: int) -> list[dict]:
        return [p for p in picks if p["team_num"] == team_num]

    tab_my_team, tab_league = st.tabs(["My Team", "League"])

    # ── My Team tab ────────────────────────────────────────────────────────────
    with tab_my_team:
        my_team_label = team_names[ai_team - 1] if ai_team <= len(team_names) else f"Team {ai_team}"
        st.markdown(f"### {my_team_label}")

        # Week picker — for now the user selects the week manually.
        # TODO: later, auto-select based on today's date.
        if "selected_week" not in st.session_state:
            st.session_state.selected_week = 1

        _week_col, _refresh_col = st.columns([3, 2])
        with _week_col:
            st.selectbox(
                "Week",
                options=list(range(1, 18)),
                key="selected_week",
            )
        with _refresh_col:
            _roster_season = roster_data.get("season")
            _last_refresh  = get_roster_refresh_time(_roster_season) if _roster_season else None
            st.markdown("**Roster data**")
            if _last_refresh:
                st.caption(f"Last refreshed: {_last_refresh}")
            else:
                st.caption("Using Season Setup snapshot")
            if st.button("🔄 Refresh Rosters", help="Pull the latest trades/signings/cuts from Sleeper", use_container_width=True):
                if _roster_season:
                    with st.spinner("Fetching latest NFL rosters..."):
                        refresh_current_rosters(_roster_season)
                    load_players_for_season.clear()
                    load_current_season_stats.clear()
                    st.success("✅ Rosters updated!")
                    st.rerun()
                else:
                    st.warning("No season set for this league.")

        my_roster = roster_for(ai_team)
        if not my_roster:
            st.info("No players drafted yet.")
        else:
            week = st.session_state.selected_week
            season = roster_data.get("season")
            # Use season-aware loader so in-season team assignments are correct
            # for the simulated season (e.g. TB for Mike Evans in a 2025 sim)
            all_players = load_players_for_season(season)
            drafted_names = {p["full_name"] for p in picks}
            starters, bench, swapped = get_lineup_for_week(
                my_roster, week, season,
                all_players=all_players, drafted_names=drafted_names,
            )

            if week > 1:
                if week - 1 >= MIN_WEEKS_FOR_BLEND:
                    st.caption(
                        f"📈 Lineup is ranked using a blend of preseason projections and each "
                        f"player's actual scoring through Week {week - 1} of {season}. "
                        f"A proven performer won't lose their spot over one bad game — it takes "
                        f"a few weeks of data for this season's results to meaningfully shift the rankings."
                    )
                else:
                    st.caption(
                        f"📊 This season's stats through Week {week - 1} are shown below, but "
                        f"lineup decisions still use preseason projections — we wait for at least "
                        f"{MIN_WEEKS_FOR_BLEND} games before letting in-season results affect the lineup."
                    )

            # ── Confirmed bye-swap / cut-player notices ───────────────────────
            for s in swapped:
                if s.get("reason") == "bye":
                    st.info(
                        f"🔁 **{s['out']['full_name']}** ({s['out']['nfl_team']}) is on a bye in Week {week} — "
                        f"starting **{s['in']['full_name']}** ({s['in']['nfl_team']}) in their place."
                    )
                elif s.get("reason") == "cut":
                    st.warning(
                        f"✂️ **{s['out']['full_name']}** is no longer on an NFL roster — "
                        f"starting **{s['in']['full_name']}** ({s['in']['nfl_team']}) in their place. "
                        f"Consider dropping {s['out']['full_name']} via the waiver wire."
                    )

            # ── Waiver wire — bye emergencies (interactive) ───────────────────
            waiver_needs = [s for s in swapped if s.get("reason") == "waiver_needed"]
            if waiver_needs:
                render_waiver_ui(
                    league=league,
                    week=week,
                    season=season,
                    bench=bench,
                    waiver_needs=waiver_needs,
                    picks=picks,
                    ai_team=ai_team,
                    roster_data=roster_data,
                )

            # ── Ranking data (defensive rankings + offense rankings) ──────────
            # Both are built from disk-cached Sleeper weekly stats — fast to call.
            # through_week = completed weeks only (week - 1).
            # prior_season = previous year, used to blend out early-season skew
            # (first 3 weeks can be wild; prior anchors rankings until ~week 8).
            def_rankings_df     = pd.DataFrame()
            offense_rankings_df = pd.DataFrame()
            def_quality_df      = pd.DataFrame()
            _bye_weeks_for_streaming = load_bye_weeks(season) if season else {}
            _prior_season = str(int(season) - 1) if season else None
            if season and week > 1:
                def_rankings_df     = get_defense_rankings(season, week - 1, prior_season=_prior_season)
                offense_rankings_df = get_offense_rankings(season, week - 1, prior_season=_prior_season)
                def_quality_df      = get_def_quality_rankings_cached(season, week - 1)

            # ── DEF streaming recommendations ─────────────────────────────────
            # Must run before the starters list gets matchup annotations so the
            # un-annotated starters list is available for the DEF slot lookup.
            render_def_streaming_ui(
                league              = league,
                week                = week,
                season              = season,
                starters            = starters,   # not yet matchup-annotated, just position data needed
                all_players         = all_players,
                drafted_names       = drafted_names,
                picks               = picks,
                ai_team             = ai_team,
                roster_data         = roster_data,
                offense_rankings_df = offense_rankings_df,
                def_quality_df      = def_quality_df,
                bye_weeks           = _bye_weeks_for_streaming,
            )

            # ── Waiver wire — proactive upgrade suggestions ────────────────────
            # Blend the full roster (starters + bench) for position comparisons.
            enriched_roster = apply_current_season_blend(my_roster, season, week)
            render_upgrade_ui(
                league=league,
                week=week,
                season=season,
                enriched_roster=enriched_roster,
                all_players=all_players,
                drafted_names=drafted_names,
                picks=picks,
                ai_team=ai_team,
                roster_data=roster_data,
            )

            # ── Trade recommendations ─────────────────────────────────────────
            render_trade_ui(
                league               = league,
                week                 = week,
                season               = season,
                my_roster            = my_roster,
                all_picks            = picks,
                all_players          = all_players,
                num_teams            = num_teams,
                ai_team              = ai_team,
                team_names           = team_names,
                roster_data          = roster_data,
                defense_rankings_df  = def_rankings_df,
            )

            def _attach_matchup(player: dict) -> dict:
                """Returns player dict with a 'matchup' key added (or None)."""
                if def_rankings_df.empty or not season:
                    return {**player, "matchup": None}
                matchup = get_def_matchup(
                    player_team   = player.get("nfl_team", ""),
                    position      = player["position"],
                    upcoming_week = week,
                    season        = season,
                    rankings_df   = def_rankings_df,
                )
                return {**player, "matchup": matchup}

            starters = [_attach_matchup(p) for p in starters]
            bench    = [_attach_matchup(p) for p in bench]

            # ── LLM rationale (only for confirmed swaps — waiver_needed entries
            #    are still pending so we exclude them from the rationale prompt) ─
            confirmed_swapped = [s for s in swapped if s.get("reason") == "bye"]
            rationale = get_lineup_rationale(league["id"], week, season, starters, bench, confirmed_swapped)
            if rationale:
                st.markdown("##### 🧠 Why this lineup")
                st.info(rationale)

            def _row(p: dict) -> dict:
                row = {
                    "Slot":     p.get("slot", ""),
                    "Player":   p["full_name"],
                    "Pos":      p["position"],
                    "NFL Team": p["nfl_team"],
                    "Preseason Avg": round(p.get("avg_pts_ppr", 0) or 0, 1),
                }
                if week > 1:
                    weeks_played = p.get("weeks_played_current", 0)
                    row[f"{season} Avg"] = round(p["current_avg_ppr"], 1) if weeks_played else "—"
                    row["Blended Avg"]   = round(p.get("blended_avg_ppr", 0) or 0, 1)
                # Matchup column — shows opponent + difficulty rating for the upcoming week
                row["Matchup"] = format_matchup_label(p.get("matchup"))
                return row

            st.markdown(f"#### Week {week} Starters")
            starter_rows = [_row(p) for p in starters]
            st.dataframe(pd.DataFrame(starter_rows), hide_index=True, use_container_width=True)

            st.markdown(f"#### Week {week} Bench")
            if bench:
                bench_rows = [_row(p) for p in bench]
                for r in bench_rows:
                    r.pop("Slot", None)  # bench players have no slot
                st.dataframe(pd.DataFrame(bench_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No bench players.")

    # ── League tab ─────────────────────────────────────────────────────────────
    with tab_league:
        st.markdown("### Teams")

        if "league_view_selected_team" not in st.session_state:
            st.session_state.league_view_selected_team = ai_team

        cols = st.columns(min(num_teams, 6))
        for i, team_num in enumerate(range(1, num_teams + 1)):
            label = team_names[team_num - 1] if team_num <= len(team_names) else f"Team {team_num}"
            if team_num == ai_team:
                label += " (You)"
            with cols[i % len(cols)]:
                if st.button(label, key=f"team_btn_{team_num}", use_container_width=True):
                    st.session_state.league_view_selected_team = team_num

        st.divider()

        selected_team = st.session_state.league_view_selected_team
        selected_label = team_names[selected_team - 1] if selected_team <= len(team_names) else f"Team {selected_team}"
        st.markdown(f"#### {selected_label}'s Roster")

        roster = roster_for(selected_team)
        if not roster:
            st.info("No players drafted yet.")
        else:
            roster_rows = [
                {
                    "Player":   p["full_name"],
                    "Pos":      p["position"],
                    "NFL Team": p["nfl_team"],
                    "Avg PPR":  round(p.get("avg_pts_ppr", 0) or 0, 1),
                }
                for p in roster
            ]
            st.dataframe(pd.DataFrame(roster_rows), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SEASON SETUP — Fetch data and build player summary
# ═══════════════════════════════════════════════════════════════════════════════

def render_season_setup():
    st.title("⚙️ Season Setup")
    st.markdown(
        "Configure which season you're drafting for and how many years of "
        "historical data to include. Then click **Generate Data** to fetch "
        "player stats and build the player rankings. You only need to do this "
        "once at the start of each season."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        current_year = datetime.now().year
        draft_year = st.number_input(
            "Season you are drafting for",
            min_value=2020,
            max_value=current_year + 1,
            value=st.session_state.get("draft_year", current_year),
            help="The upcoming NFL season year. Historical data will be fetched for the years before this."
        )
        # Write to session state immediately so switching to Draft in the same
        # session picks up the correct year — even if Generate Data isn't clicked.
        st.session_state.draft_year = int(draft_year)

    with col2:
        num_prior_years = st.number_input(
            "Years of historical data",
            min_value=1,
            max_value=5,
            value=3,
            help="How many prior seasons to pull stats from. More years gives better trend data."
        )

    # Compute the seasons that will be fetched
    seasons = [str(draft_year - i) for i in range(num_prior_years, 0, -1)]

    st.info(
        f"📅 **Seasons to fetch:** {', '.join(seasons)}  \n"
        f"Weights: most recent ({seasons[-1]}) → {int(50)}% · "
        f"{seasons[-2] if len(seasons) > 1 else 'n/a'} → {int(30)}% · "
        f"{seasons[0] if len(seasons) > 2 else 'n/a'} → {int(20)}%"
        if len(seasons) >= 3 else
        f"📅 **Seasons to fetch:** {', '.join(seasons)}"
    )

    if DATA_PATH.exists():
        mtime = datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
        st.success(f"✅ Player data already exists (last generated {mtime.strftime('%b %d, %Y at %H:%M')}). You can regenerate below or go to Draft.")

    st.divider()

    if st.button("🔄 Generate Data", use_container_width=True, type="primary"):
        log_lines = []
        log_area  = st.empty()

        def log(msg):
            log_lines.append(msg)
            log_area.text("\n".join(log_lines))

        log("═" * 50)
        log("  Step 1 of 3 — Fetching player stats from Sleeper API")
        log("═" * 50)

        success = run_pipeline(seasons, log=log)

        if not success:
            st.error("⚠️ Data fetch failed. Check your internet connection and try again.")
            return

        log("\n" + "═" * 50)
        log("  Step 2 of 3 — Building multi-year player summary")
        log("═" * 50)

        st.session_state.draft_year = int(draft_year)

        # Persist draft_year to disk so it survives app restarts
        config_path = Path(__file__).parent / "data" / "processed" / "season_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as _f:
            json.dump({"draft_year": int(draft_year), "seasons": seasons}, _f)

        summary = build_multi_year_summary(seasons, draft_year=int(draft_year), log=log)

        if summary.empty:
            st.error("⚠️ Could not build player summary. Check the logs above.")
            return

        # Reload the cached player data so the draft picks up the new file
        load_players.clear()
        load_players_for_season.clear()

        log("\n" + "═" * 50)
        log("  Step 3 of 3 — Fetching NFL schedule / bye weeks")
        log("═" * 50)

        fetch_bye_weeks(str(int(draft_year)), log=log)

        # Reload the cached bye-week data so the lineup picks up the new file
        load_bye_weeks.clear()

        log("\n✅ All done! Switch to Draft to begin.")
        log_area.text("\n".join(log_lines))

        st.success("✅ Data generated successfully! Go to **Draft** in the sidebar to begin.")

    # ── Refresh NFL Transactions ────────────────────────────────────────────────
    # This fetches the latest offseason transactions from ESPN and applies them
    # to the player database. Run this after 'Generate Data' to capture trades,
    # signings, retirements, and IR placements that affect draft values.
    #
    # Think of it like getting the morning news: historical stats are the resume,
    # but the transactions tell you who got promoted, transferred, or fired.
    if DATA_PATH.exists():
        st.divider()
        st.subheader("📰 NFL Transactions")
        st.caption(
            "Fetch the latest offseason moves from ESPN (trades, signings, releases, IR). "
            "These update each player's breakout score and add transaction notes the AI "
            "draft agent reads during picks. Run this after 'Generate Data'."
        )

        # Show a summary if transaction_impacts.csv already exists
        tx_path = Path(__file__).parent / "data" / "processed" / "transaction_impacts.csv"
        if tx_path.exists():
            try:
                tx_df = pd.read_csv(tx_path)
                high   = (tx_df["fantasy_impact"] == "HIGH").sum()
                medium = (tx_df["fantasy_impact"] == "MEDIUM").sum()
                in_db  = tx_df["in_database"].sum() if "in_database" in tx_df.columns else len(tx_df)
                st.info(
                    f"Last refresh: **{len(tx_df):,} transactions** loaded "
                    f"({in_db} matched to our database · {high} high-impact · {medium} medium-impact)"
                )
            except Exception:
                pass

        if st.button(
            "🔄 Refresh Transactions",
            help="Fetches the latest NFL transactions from ESPN and updates breakout scores",
            use_container_width=False,
        ):
            tx_log_lines = []
            tx_log_area  = st.empty()

            def tx_log(msg):
                tx_log_lines.append(msg)
                tx_log_area.text("\n".join(tx_log_lines))

            with st.spinner("Fetching NFL transactions from ESPN..."):
                try:
                    # Redirect stdout-style prints to our log area
                    import io, sys
                    old_stdout = sys.stdout
                    sys.stdout = buffer = io.StringIO()

                    stats = run_transaction_pipeline()

                    sys.stdout = old_stdout
                    captured = buffer.getvalue()
                    if captured:
                        tx_log(captured)

                    # Show a clean summary card
                    if "error" in stats:
                        st.error(f"⚠️ Transaction fetch failed: {stats['error']}")
                    else:
                        st.success(
                            f"✅ Transactions refreshed! "
                            f"**{stats['total_filtered']:,}** fantasy-relevant moves detected "
                            f"({stats['in_database']} matched to our database · "
                            f"{stats['high_impact']} high-impact · "
                            f"{stats['medium_impact']} medium-impact)"
                        )

                        # Show high-impact moves in an expander so the user can review them
                        if tx_path.exists():
                            try:
                                tx_df = pd.read_csv(tx_path)
                                hi_df = tx_df[tx_df["fantasy_impact"] == "HIGH"].copy()
                                if not hi_df.empty:
                                    display_cols = [c for c in
                                        ["player_name", "position", "transaction_type",
                                         "team", "impact_note", "date"]
                                        if c in hi_df.columns]
                                    with st.expander(
                                        f"🔴 {len(hi_df)} High-Impact Moves", expanded=True
                                    ):
                                        st.dataframe(
                                            hi_df[display_cols].rename(columns={
                                                "player_name":      "Player",
                                                "position":         "Pos",
                                                "transaction_type": "Move",
                                                "team":             "Team",
                                                "impact_note":      "Fantasy Note",
                                                "date":             "Date",
                                            }),
                                            use_container_width=True,
                                            hide_index=True,
                                        )

                                med_df = tx_df[tx_df["fantasy_impact"] == "MEDIUM"].copy()
                                if not med_df.empty:
                                    display_cols = [c for c in
                                        ["player_name", "position", "transaction_type",
                                         "team", "impact_note", "date"]
                                        if c in med_df.columns]
                                    with st.expander(
                                        f"🟡 {len(med_df)} Medium-Impact Moves", expanded=False
                                    ):
                                        st.dataframe(
                                            med_df[display_cols].rename(columns={
                                                "player_name":      "Player",
                                                "position":         "Pos",
                                                "transaction_type": "Move",
                                                "team":             "Team",
                                                "impact_note":      "Fantasy Note",
                                                "date":             "Date",
                                            }),
                                            use_container_width=True,
                                            hide_index=True,
                                        )
                            except Exception as e:
                                st.warning(f"Could not display transaction table: {e}")

                        # Reload cached player data so the draft agent sees updated scores
                        load_players.clear()
                        load_players_for_season.clear()

                except Exception as e:
                    sys.stdout = old_stdout
                    st.error(f"⚠️ Unexpected error: {e}")

    # ── Player list preview ─────────────────────────────────────────────────────
    if DATA_PATH.exists():
        st.divider()
        preview_season = str(int(draft_year)) if draft_year else None
        all_players = load_players_for_season(preview_season)
        # Filter out FA/cut players for the preview (same as draft table)
        preview_players = all_players[
            all_players["team"].notna() & (all_players["team"] != "FA")
        ].reset_index(drop=True)
        render_available_players(preview_players, selectable=False)

    # ── Testing tools ────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("🧪 Testing tools"):
        st.markdown(
            "In-season stats (each week's fetched game results, used to update weekly "
            "lineups) are stored per season and shared across all your leagues for that "
            "season. Use this to wipe that data and start a league's week-by-week "
            "testing back at Week 1."
        )
        reset_season = st.number_input(
            "Season to reset",
            min_value=2020,
            max_value=current_year + 1,
            value=int(draft_year),
            key="reset_season_input",
            help="The season whose in-season (current-season) stats should be cleared."
        )
        if st.button(f"🗑️ Clear in-season data for {int(reset_season)}"):
            log_lines = []
            log_area = st.empty()

            def log(msg):
                log_lines.append(msg)
                log_area.text("\n".join(log_lines))

            clear_current_season_data(str(int(reset_season)), log=log)

            # Clear cached readers so the app doesn't keep showing stale data
            load_current_season_stats.clear()
            get_lineup_rationale.clear()

            st.success(
                f"✅ Cleared in-season data for {int(reset_season)}. "
                f"Select Week 1 in a league's lineup to start fresh."
            )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Setup form
# ═══════════════════════════════════════════════════════════════════════════════

def render_setup():
    st.title("🏈 AI Fantasy Football Draft")

    active_league = st.session_state.get("active_league")
    if active_league:
        st.markdown(f"League: **{active_league['name']}**")
        st.caption("Settings below are pre-filled from this league. Pick a strategy, then click **Start Draft** to begin.")
    else:
        st.markdown("Set up your draft below, then click **Start Draft** to begin.")

    col1, col2 = st.columns(2)

    with col1:
        team_options = [2, 4, 6, 8, 10, 12]
        saved_num_teams = st.session_state.get("num_teams", 4)
        team_idx = team_options.index(saved_num_teams) if saved_num_teams in team_options else 1

        num_teams = st.selectbox(
            "Number of teams",
            options=team_options,
            index=team_idx,
            help="For testing, 4 teams is recommended.",
            disabled=bool(active_league),
        )

        saved_ai_team = st.session_state.get("ai_team", 1)
        ai_idx = min(saved_ai_team - 1, num_teams - 1)

        ai_position = st.selectbox(
            "AI team's draft position",
            options=list(range(1, num_teams + 1)),
            index=ai_idx,
            help="Which pick slot does the AI have in round 1?",
            disabled=bool(active_league),
        )

    with col2:
        num_rounds = st.number_input(
            "Number of rounds",
            min_value=5, max_value=20,
            value=st.session_state.get("num_rounds", 15),
            disabled=bool(active_league),
        )
        st.selectbox(
            "Scoring format",
            options=["PPR", "Half PPR", "Standard"],
            index=SCORING_OPTIONS.index(active_league["scoring"]) if active_league else 0,
            disabled=bool(active_league),
        )

    # ── Strategy selector ─────────────────────────────────────────────────────
    strategy_names = list(STRATEGIES.keys())
    saved_strategy_name = st.session_state.get("strategy_name", strategy_names[0])
    strategy_idx = strategy_names.index(saved_strategy_name) \
                   if saved_strategy_name in strategy_names else 0

    strategy_name = st.selectbox(
        "Draft Strategy",
        options=strategy_names,
        index=strategy_idx,
        help=(
            "League-Size Optimized (PPR): automatically applies the data-backed strategy "
            "for your exact league size (derived from PFR 2020-2025 actual PPR finishes). "
            "AI Agent Strategy: Claude analyzes the player pool and writes its own plan. "
            "My Draft Rules: follows your hand-written strategy."
        )
    )

    # Show a preview of the selected strategy (read-only)
    strategy_val = STRATEGIES[strategy_name]
    if strategy_val == "__LEAGUE_SIZE__":
        # Data-backed league-size strategy — show the version matching the current num_teams
        league_strategy_text = get_league_strategy(num_teams)
        st.success(
            f"📊 **League-Size Optimized selected.** The strategy below is automatically "
            f"calibrated for a **{num_teams}-team PPR league** based on positional scarcity "
            f"data from PFR 2020–2025."
        )
        with st.expander("📋 View strategy for this league size", expanded=False):
            st.text(league_strategy_text)
    elif strategy_val is not None:
        # Human-written strategy — show the text so the user can review it
        with st.expander("📋 View strategy", expanded=False):
            st.text(strategy_val)
    else:
        # AI-generated — nothing to preview yet; explain what will happen
        st.info(
            "🤖 **AI Agent Strategy selected.** When you click Start Draft, Claude will "
            "analyze the current player pool and write its own strategy before the draft begins. "
            "This takes about 10–15 seconds."
        )

    if st.button("🚀 Start Draft", use_container_width=True, type="primary"):

        # Save the draft config regardless of strategy type
        draft_config = {
            "num_teams":     num_teams,
            "num_rounds":    int(num_rounds),
            "ai_team":       ai_position,
            "strategy_name": strategy_name,
        }

        if strategy_val == "__LEAGUE_SIZE__":
            # League-Size Optimized — resolve the strategy text for this team count now
            draft_config["strategy"] = get_league_strategy(num_teams)
            _start_draft(draft_config)
        elif strategy_val is None:
            # AI Agent Strategy — generate it now, then pause so the user can read it
            with st.spinner("🧠 Claude is analyzing the player pool and writing its strategy..."):
                all_players = load_players()
                agent = StrategyAgent(
                    scoring="ppr",
                    num_rounds=int(num_rounds),
                    num_teams=num_teams,
                )
                result = agent.generate(all_players)

            if result["error"]:
                st.error(f"⚠️ Could not generate strategy: {result['error']}")
                st.stop()

            # Store the generated strategy and config, but don't start the draft yet.
            # render_strategy_preview() will display it and provide a "Begin Draft" button.
            draft_config["strategy"] = result["strategy"]
            st.session_state.pending_draft = draft_config
            st.rerun()
        else:
            # Human-written strategy — start immediately
            draft_config["strategy"] = STRATEGIES[strategy_name]
            _start_draft(draft_config)


def render_strategy_preview():
    """
    Shown after the AI Agent Strategy is generated.
    Displays the full strategy text and waits for the user to click Begin Draft.
    This gives the user time to read Claude's plan before picks start.
    """
    config = st.session_state.pending_draft

    st.title("🏈 AI Draft Strategy")
    st.success("✅ Claude has analyzed the player pool and written its strategy. Review it below, then begin the draft when you're ready.")

    st.markdown("### Claude's Draft Plan")
    st.text(config["strategy"])

    st.divider()

    if st.button("▶️ Begin Draft", use_container_width=True, type="primary"):
        st.session_state.pending_draft = None
        _start_draft(config)


def _start_draft(config: dict):
    """Commits all draft config to session state and kicks off the draft."""
    st.session_state.draft_started   = True
    st.session_state.num_teams       = config["num_teams"]
    st.session_state.num_rounds      = config["num_rounds"]
    st.session_state.ai_team         = config["ai_team"]
    st.session_state.strategy_name   = config["strategy_name"]
    st.session_state.strategy        = config["strategy"]
    st.session_state.current_pick    = 1
    st.session_state.picks           = []
    st.session_state.recommendation  = None
    st.session_state.selected_player = None
    st.session_state.last_ai_pick    = None
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — Active draft
# ═══════════════════════════════════════════════════════════════════════════════

def render_draft(all_players: pd.DataFrame):
    num_teams  = st.session_state.num_teams
    num_rounds = st.session_state.num_rounds
    total_picks = num_teams * num_rounds
    current_pick = st.session_state.current_pick
    ai_team = st.session_state.ai_team

    # ── Draft complete ─────────────────────────────────────────────────────────
    if current_pick > total_picks:
        st.title("🎉 Draft Complete!")
        st.markdown("Here's a summary of every team's final roster:")

        team_names = st.session_state.get("team_names")
        cols = st.columns(num_teams)
        for i, team_num in enumerate(range(1, num_teams + 1)):
            with cols[i]:
                team_label = team_names[team_num - 1] if team_names and team_num <= len(team_names) \
                              else f"Team {team_num}"
                label = f"🤖 {team_label} (AI)" if team_num == ai_team \
                        else f"👤 {team_label}"
                st.markdown(f"**{label}**")
                roster = get_team_roster(team_num)
                for p in roster:
                    st.write(f"{p['position']} — {p['full_name']}")

        # Save the final rosters to the active league (if the draft was started from one)
        active_league = st.session_state.get("active_league")
        if active_league:
            save_league_roster(
                league_id=active_league["id"],
                picks=st.session_state.picks,
                team_names=team_names,
                ai_team=ai_team,
                num_teams=num_teams,
                season=str(st.session_state.get("draft_year", datetime.now().year)),
            )

            st.divider()
            if st.button("🏆 Go to My League", type="primary"):
                # Reload the league so roster_data is picked up
                leagues = load_leagues()
                refreshed = next((l for l in leagues if l["id"] == active_league["id"]), active_league)
                st.session_state.active_league = refreshed
                st.session_state.app_page = "League View"
                st.rerun()
        return

    # ── Determine whose turn it is ─────────────────────────────────────────────
    picking_team = get_team_for_pick(current_pick, num_teams)
    round_num, slot = get_round_and_slot(current_pick, num_teams)
    available = get_available_players(all_players)
    is_ai_turn = (picking_team == ai_team)

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🏈 AI Fantasy Football Draft")
    st.markdown(f"**Round {round_num} of {num_rounds} · Pick {current_pick} of {total_picks}**")

    # ── Auto-record pick for human teams ──────────────────────────────────────
    # When the user clicks a row in the table below, selected_player is set in
    # session state and a rerun is triggered. We catch it here (before rendering
    # anything) and immediately record the pick, then rerun to advance the draft.
    if not is_ai_turn and st.session_state.get("selected_player"):
        selected_name = st.session_state.selected_player
        match = available[available["full_name"] == selected_name]
        if not match.empty:
            row = match.iloc[0]
            _record_pick(
                full_name=selected_name,
                position=row["position"],
                nfl_team=row["team"],
                team_num=picking_team,
            )
        st.session_state.selected_player = None
        st.session_state.last_ai_pick = None   # clear AI reasoning once a human picks
        st.rerun()

    # ── Last AI pick reasoning (persists until next human pick) ───────────────
    last = st.session_state.get("last_ai_pick")
    if last and not last.get("error"):
        st.info(
            f"🤖 **AI drafted {last['player_name']}** "
            f"({last['position']}, {last['nfl_team']})  \n"
            f"**Reasoning:** {last['reasoning']}"
        )

    # ── Pick area ─────────────────────────────────────────────────────────────
    if is_ai_turn:
        render_ai_turn(available, round_num, slot, num_teams,
                       injury_notes=st.session_state.get("injury_notes", ""))
    else:
        render_human_turn(picking_team)

    st.divider()

    # ── Available players table ────────────────────────────────────────────────
    # Pass selectable=True on human turns so clicking a row selects that pick.
    render_available_players(available, selectable=not is_ai_turn)


def render_ai_turn(
    available: pd.DataFrame,
    round_num: int,
    slot: int,
    num_teams: int,
    injury_notes: str = "",
):
    """Renders the UI when it's the AI's turn to pick."""
    st.success(f"🤖 **It's your AI team's turn to pick!**")

    agent = DraftAgent(
        strategy=st.session_state.strategy,
        scoring="ppr",
        num_rounds=st.session_state.num_rounds
    )

    my_roster = get_team_roster(st.session_state.ai_team)

    # Hard filter: remove single-slot positions the AI already has.
    # DEF and K only ever have one starter slot and no bench purpose —
    # drafting a second one wastes a roster spot and should never happen.
    SINGLE_SLOT_POSITIONS = {"DEF", "K"}
    ai_positions = {p["position"] for p in my_roster}
    filled_singles = SINGLE_SLOT_POSITIONS & ai_positions   # positions already covered
    if filled_singles:
        ai_available = available[~available["position"].isin(filled_singles)].copy()
    else:
        ai_available = available

    col1, col2 = st.columns([1, 2])

    with col1:
        if st.button("🧠 Get AI Recommendation", use_container_width=True, type="primary"):
            with st.spinner("Claude is thinking..."):
                rec = agent.get_recommendation(
                    available_players=ai_available,
                    my_roster=my_roster,
                    current_round=round_num,
                    current_pick_in_round=slot,
                    num_teams=num_teams,
                    injury_notes=injury_notes,
                )

            if rec.get("error"):
                # On error, store it so we can display it without auto-picking
                st.session_state.recommendation = rec
            else:
                # Auto-record the pick immediately — no confirm button needed
                _record_pick(
                    full_name=rec["player_name"],
                    position=rec["position"],
                    nfl_team=rec["nfl_team"],
                    team_num=st.session_state.ai_team
                )
                # Persist the reasoning so it stays visible on the next team's turn
                st.session_state.last_ai_pick = rec
                st.session_state.recommendation = None
                st.rerun()

    # Only shown if there was an API error
    rec = st.session_state.recommendation
    if rec and rec.get("error"):
        st.error(f"⚠️ {rec['error']}")


def render_human_turn(picking_team: int):
    """Renders the instruction banner when it's another team's turn."""
    team_names = st.session_state.get("team_names")
    team_label = team_names[picking_team - 1] if team_names and picking_team <= len(team_names) \
                  else f"Team {picking_team}"
    st.info(f"👤 **{team_label}'s turn to pick.** Click a player row in the table below to record their pick.")


def render_available_players(available: pd.DataFrame, selectable: bool = False):
    """
    Renders the available players table with position filters.

    When selectable=True (human turn), clicking a row stores that player
    in session state so the confirm button above can use it.
    """
    st.markdown("### Available Players")
    st.caption("Breakout column: blank = already elite (18+ avg pts/game in 2025), not applicable.")
    if selectable:
        st.caption("👆 Click any row to select that player as the pick.")

    # Initialise filter in session state on first load
    if "pos_filter" not in st.session_state:
        st.session_state.pos_filter = "All"

    # Filter buttons — each writes directly to session state when clicked.
    # Also clear any pending player selection so a stale pick from the
    # previous filter view doesn't carry over.
    col_all, col_qb, col_rb, col_wr, col_te, col_k, col_def = st.columns(7)
    for col, label in zip(
        [col_all, col_qb, col_rb, col_wr, col_te, col_k, col_def],
        ["All", "QB", "RB", "WR", "TE", "K", "DEF"]
    ):
        if col.button(label, use_container_width=True):
            st.session_state.pos_filter = label
            st.session_state.selected_player = None

    filtered = available if st.session_state.pos_filter == "All" \
               else available[available["position"] == st.session_state.pos_filter]

    # Sort by consensus ADP (FantasyPros 2026) when available; fall back to VOR rank
    if "adp_2026" in filtered.columns:
        filtered = filtered.sort_values("adp_2026", ascending=True, na_position="last")
    else:
        filtered = filtered.sort_values("season_rank", ascending=True)

    # Columns to display (keep full_name for selection lookup, hide it visually via rename)
    display_cols = {
        "adp_2026":           "ADP",
        "full_name":          "Player",
        "position":           "Pos",
        "team":               "NFL Team",
        "weighted_avg_ppr":   "Wtd Avg PPR",
        "avg_pts_ppr_2025":   "2025 Avg/Gm",
        "total_pts_ppr_2025": "2025 Total",
        "weeks_played_2025":  "2025 Wks",
        "avg_pts_ppr_2024":   "2024 Avg/Gm",
        "total_pts_ppr_2024": "2024 Total",
        "weeks_played_2024":  "2024 Wks",
        "avg_pts_ppr_2023":   "2023 Avg/Gm",
        "total_pts_ppr_2023": "2023 Total",
        "weeks_played_2023":  "2023 Wks",
        "trend":              "Trend",
        "years_of_data":      "Yrs Data",
        "breakout_score":     "Breakout",
        "durability_score":   "Durability",
        "depth_improvement":  "Depth ↑",
        "surge_score_2025":   "Surge",
        "injury_status":      "Status",
    }
    available_display_cols = {k: v for k, v in display_cols.items() if k in filtered.columns}
    display_df = filtered[list(available_display_cols.keys())].rename(
        columns=available_display_cols
    ).head(50).reset_index(drop=True)

    # Breakout column stays numeric so sorting works correctly.
    # NaN (blank) = already elite (18+ avg pts/game), not applicable.

    if selectable:
        # on_select="rerun" means Streamlit reruns the script when a row is clicked,
        # and event.selection.rows contains the 0-based index of the clicked row.
        event = st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
        )
        if event.selection.rows:
            clicked_idx = event.selection.rows[0]
            st.session_state.selected_player = filtered.iloc[clicked_idx]["full_name"]
            # Trigger a second rerun so the auto-pick logic at the top of
            # render_draft can catch the selection and record it immediately.
            st.rerun()
    else:
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400,
        )


# ── Helper: record a pick ──────────────────────────────────────────────────────

def _record_pick(full_name: str, position: str, nfl_team: str, team_num: int):
    """Adds a pick to the picks list and advances the pick counter."""
    st.session_state.picks.append({
        "full_name":  full_name,
        "position":   position,
        "nfl_team":   nfl_team,
        "team_num":   team_num,
        "pick_number": st.session_state.current_pick,
        "avg_pts_ppr": _get_avg_pts(full_name),
    })
    st.session_state.current_pick += 1


def _get_avg_pts(full_name: str) -> float:
    """
    Looks up a player's preseason baseline for storage in the picks list.
    Uses weighted_avg_ppr (the 3-year weighted average used for drafting)
    with avg_pts_ppr as a fallback, so blending has a meaningful prior even
    if the plain avg_pts_ppr column doesn't exist in multi_year_summary.csv.
    """
    try:
        all_players = load_players()
        row = all_players[all_players["full_name"] == full_name]
        if not row.empty:
            r = row.iloc[0]
            return float(r.get("weighted_avg_ppr") or r.get("avg_pts_ppr") or 0)
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

render_sidebar()

if st.session_state.app_page == "League View":
    render_league_view()
elif st.session_state.app_page == "My Leagues":
    render_my_leagues()
elif st.session_state.app_page == "Season Setup":
    render_season_setup()
else:
    # Draft page — only load player data here, after Season Setup has run
    if not DATA_PATH.exists():
        st.warning("⚠️ No player data found. Please run **Season Setup** first.")
        st.stop()

    # Use the season-aware loader so players show their team for the draft year
    # (e.g. Mike Evans shows TB for a 2025 simulation, not SF from 2026 signings)
    draft_season = str(st.session_state.get("draft_year", "")) or None
    all_players = load_players_for_season(draft_season)

    if st.session_state.pending_draft is not None:
        render_strategy_preview()
    elif not st.session_state.draft_started:
        render_setup()
    else:
        render_draft(all_players)
