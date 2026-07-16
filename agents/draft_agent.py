"""
draft_agent.py
--------------
The Draft Agent uses Claude to recommend which player to pick
during a fantasy football draft.

It takes the current draft situation — what round it is, what players
are still available, and what the AI's roster looks like so far —
and asks Claude to reason through the best pick given a strategy.

Think of it like hiring an analyst: you give them the data and the
strategy, and they come back with a recommendation and an explanation.
"""

import os
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # Loads ANTHROPIC_API_KEY from your .env file


class DraftAgent:
    """
    Uses Claude to recommend draft picks based on a strategy and draft context.

    Parameters:
        strategy (str): Plain-English draft strategy, e.g.
            "Prioritize running backs in rounds 1-3, then target elite wide
             receivers. Wait on quarterback and tight end."
        scoring (str): "ppr", "half_ppr", or "std"
        num_rounds (int): Total rounds in the draft (default 15)
    """

    # Standard fantasy lineup requirements for a 15-round PPR league.
    # The agent uses these to understand what positions it still needs.
    ROSTER_TARGETS = {
        "QB":  2,
        "RB":  5,
        "WR":  5,
        "TE":  2,
        "K":   1,
        "DEF": 1,
    }

    def __init__(self, strategy: str, scoring: str = "ppr", num_rounds: int = 15):
        self.strategy = strategy
        self.scoring = scoring
        self.num_rounds = num_rounds
        self.pts_col = f"total_pts_{scoring}"  # column name in the summary CSV
        self.avg_col = f"avg_pts_{scoring}"

        # Initialize the Anthropic client using the API key from .env
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Main public method ─────────────────────────────────────────────────────

    def get_recommendation(
        self,
        available_players: pd.DataFrame,
        my_roster: list[dict],
        current_round: int,
        current_pick_in_round: int,
        num_teams: int,
    ) -> dict:
        """
        Asks Claude to recommend the best available player to draft.

        Parameters:
            available_players: DataFrame of players not yet drafted,
                               sorted by season_rank (best first)
            my_roster: List of dicts already on the AI's team, each with
                       keys: full_name, position, total_pts_ppr, avg_pts_ppr
            current_round: Which round we're in (1–15)
            current_pick_in_round: Which pick within the round (1–num_teams)
            num_teams: Total teams in the league

        Returns:
            dict with keys:
                "player_name"  — the name of the recommended player
                "position"     — their position (QB/RB/WR/TE)
                "nfl_team"     — their NFL team
                "reasoning"    — Claude's explanation (2–3 sentences)
                "raw_response" — the full text Claude returned
                "error"        — set if something went wrong, None otherwise
        """
        # Build the prompt with all the relevant context
        prompt = self._build_prompt(
            available_players, my_roster, current_round,
            current_pick_in_round, num_teams
        )

        try:
            # Call Claude
            message = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            raw = message.content[0].text

            # Parse Claude's structured response
            return self._parse_response(raw, available_players)

        except anthropic.AuthenticationError:
            return {
                "player_name": None, "position": None, "nfl_team": None,
                "reasoning": None, "raw_response": None,
                "error": "Invalid API key. Check your .env file."
            }
        except Exception as e:
            return {
                "player_name": None, "position": None, "nfl_team": None,
                "reasoning": None, "raw_response": None,
                "error": f"API error: {str(e)}"
            }

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        available_players: pd.DataFrame,
        my_roster: list[dict],
        current_round: int,
        current_pick_in_round: int,
        num_teams: int,
    ) -> str:
        """
        Builds the text prompt sent to Claude.

        Good prompts give Claude exactly what a knowledgeable human analyst
        would need: the strategy, the roster situation, and the available options.
        """

        # ── Section 1: Roster summary ──────────────────────────────────────────
        if my_roster:
            roster_lines = []
            for p in my_roster:
                roster_lines.append(
                    f"  - {p['full_name']} ({p['position']}, {p['nfl_team']}) "
                    f"— {p['avg_pts_ppr']:.1f} avg PPR pts/game in 2024"
                )
            roster_text = "\n".join(roster_lines)
        else:
            roster_text = "  (no players drafted yet)"

        # ── Section 2: Positional needs ───────────────────────────────────────
        # Count how many of each position I already have
        position_counts = {}
        for p in my_roster:
            pos = p["position"]
            position_counts[pos] = position_counts.get(pos, 0) + 1

        needs_lines = []
        for pos, target in self.ROSTER_TARGETS.items():
            have = position_counts.get(pos, 0)
            still_need = max(0, target - have)
            needs_lines.append(f"  {pos}: have {have}, ideally want {target} (need {still_need} more)")
        needs_text = "\n".join(needs_lines)

        # ── Section 3: Available players, grouped by position ─────────────────
        # Show up to 40 players per position (or all available if fewer than 40).
        # Grouping by position gives Claude clear visibility into every positional
        # pool — the strategy already instructs it on which positions to target
        # and when, so no overall ranking list is needed.

        def season_detail(row, season: str) -> str:
            """Returns 'YY: avg(wks/total)' for one season, or blank if no data."""
            avg  = row.get(f"avg_pts_ppr_{season}")
            tot  = row.get(f"total_pts_ppr_{season}")
            wks  = row.get(f"weeks_played_{season}")
            if avg is None or (isinstance(avg, float) and pd.isna(avg)) or avg == 0:
                return ""
            return f"{season[-2:]}:{avg:.1f}avg/{int(wks)}wk/{int(tot)}tot"

        def player_line(row) -> str:
            wtd_avg = row.get("weighted_avg_ppr", row.get(self.avg_col, 0))
            trend   = row.get("trend", "")
            breakout = row.get("breakout_score")
            # Build a compact per-season breakdown so Claude can see total points
            # and games played alongside the average — critical for spotting players
            # who scored well in only a handful of games (inflated avg, low total).
            seasons = [s for s in ["2024", "2023", "2022"]
                       if season_detail(row, s)]
            season_str = "  |  " + "  ".join(season_detail(row, s) for s in seasons) if seasons else ""
            if breakout is None or (isinstance(breakout, float) and pd.isna(breakout)):
                breakout_str = "  breakout: N/A (already elite)"
            elif breakout > 0:
                breakout_str = f"  breakout: {breakout:.0f}"
            else:
                breakout_str = ""
            return (
                f"  #{int(row['season_rank']):>3}  {row['full_name']:<22} "
                f"{str(row.get('team', 'FA')):<4}  "
                f"{wtd_avg:.1f} wtd avg/gm{season_str}  trend: {trend}{breakout_str}"
            )

        position_sections = []
        for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
            pos_players = available_players[available_players["position"] == pos].head(40)
            count = len(pos_players)
            if count == 0:
                section = f"  {pos} — none available"
            else:
                lines = [player_line(row) for _, row in pos_players.iterrows()]
                section = f"  {pos} ({count} available):\n" + "\n".join(lines)
            position_sections.append(section)

        players_text = "\n\n".join(position_sections)

        # ── Assemble the full prompt ───────────────────────────────────────────
        prompt = f"""You are an expert fantasy football draft assistant. Your job is to recommend the single best player to draft right now.

DRAFT SITUATION:
- Round {current_round} of {self.num_rounds}, Pick {current_pick_in_round} of {num_teams}
- Scoring: PPR (1 point per reception)
- Total picks remaining for my team: {self.num_rounds - len(my_roster)}
- Player rankings use a weighted 3-year average (2024=50%, 2023=30%, 2022=20%) with a trend label showing whether each player is Improving, Stable, or Declining.

MY DRAFT STRATEGY:
{self.strategy}

MY CURRENT ROSTER ({len(my_roster)} players drafted so far):
{roster_text}

POSITIONAL NEEDS (based on a standard PPR roster build):
{needs_text}

HARD RULES — these override everything else, including strategy:
- NEVER draft a second DEF. You only start 1 defense and there is zero benefit to a backup DEF.
- NEVER draft a second K. You only start 1 kicker and there is zero benefit to a backup K.
- If you already have 1 DEF and 1 K, ignore both positions entirely regardless of round or availability.

AVAILABLE PLAYERS (up to 40 per position, ranked by weighted PPR average):
  Format: Rank  Name  Team  WeightedAvg/gm  |  YY:avg/gm  /  weeks_played  /  season_total  (per season)  trend  breakout
  Use weeks_played and season_total alongside the per-game average to assess reliability.
  A high avg/gm over very few weeks is less trustworthy than a similar avg/gm over a full season.
  Breakout score (0–100, skill positions only): flags players likely to outperform their weighted average.
  Combines late-season surge (30 pts), year-over-year total pts improvement (30 pts), depth chart position improvement from 2024→2025 (30 pts), and youth bonus (10 pts).
  A breakout score above 40 is notable; above 60 is a strong upside signal worth considering even if the weighted average looks modest.

{players_text}

Based on my strategy, my current roster needs, and the players available, who should I pick?

Respond in EXACTLY this format — do not add anything else:
PICK: [Full player name exactly as shown above]
REASONING: [2-3 sentences explaining why this is the best pick given my strategy and roster needs]"""

        return prompt

    # ── Response parser ────────────────────────────────────────────────────────

    def _parse_response(self, raw: str, available_players: pd.DataFrame) -> dict:
        """
        Parses Claude's structured text response into a clean dictionary.

        Claude is asked to respond in a specific format, so we can reliably
        extract the player name and reasoning with simple string parsing.
        """
        lines = raw.strip().split("\n")

        pick_line = ""
        reasoning_line = ""

        for line in lines:
            if line.startswith("PICK:"):
                pick_line = line.replace("PICK:", "").strip()
            elif line.startswith("REASONING:"):
                reasoning_line = line.replace("REASONING:", "").strip()

        if not pick_line:
            # If parsing failed, return the raw response so the user can see it
            return {
                "player_name": None,
                "position": None,
                "nfl_team": None,
                "reasoning": raw,
                "raw_response": raw,
                "error": "Could not parse player name from Claude's response."
            }

        # Try to match the recommended name to a player in our dataset
        match = available_players[
            available_players["full_name"].str.lower() == pick_line.lower()
        ]

        if not match.empty:
            row = match.iloc[0]
            position = row.get("position", "")
            nfl_team = row.get("team", "")
        else:
            # Claude may have slightly altered the name — do a partial match
            match = available_players[
                available_players["full_name"].str.lower().str.contains(
                    pick_line.lower().split()[0], na=False  # match on first name
                )
            ]
            row = match.iloc[0] if not match.empty else None
            position = row.get("position", "") if row is not None else ""
            nfl_team = row.get("team", "") if row is not None else ""
            # Use the exact name from our data if we found a partial match
            if row is not None:
                pick_line = row["full_name"]

        return {
            "player_name": pick_line,
            "position":    position,
            "nfl_team":    nfl_team,
            "reasoning":   reasoning_line,
            "raw_response": raw,
            "error":       None
        }
