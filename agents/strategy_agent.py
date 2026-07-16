"""
strategy_agent.py
-----------------
The Strategy Agent uses Claude to generate its own draft strategy by
analyzing the current player pool before the draft begins.

Rather than following a human-written set of rules, this agent looks at
the top available players at each position — their weighted averages,
scoring totals, and trends — and decides for itself how to approach
the draft. Think of it like asking an expert analyst to study the player
rankings and then write their own game plan.

The generated strategy is stored in session state and handed off to the
DraftAgent, which uses it to make individual pick decisions.
"""

import os
import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class StrategyAgent:
    """
    Generates a draft strategy by having Claude analyze the player pool.

    Parameters:
        scoring (str): "ppr", "half_ppr", or "std"
        num_rounds (int): Total rounds in the draft
        num_teams (int): Number of teams in the league
    """

    # Standard PPR starting lineup — same as DraftAgent so the strategies align.
    LINEUP = {
        "QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX (RB/WR)": 1, "K": 1, "DEF": 1
    }

    def __init__(self, scoring: str = "ppr", num_rounds: int = 15, num_teams: int = 10):
        self.scoring = scoring
        self.num_rounds = num_rounds
        self.num_teams = num_teams
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, available_players: pd.DataFrame) -> dict:
        """
        Analyzes the player pool and asks Claude to write a draft strategy.

        Parameters:
            available_players: Full DataFrame of available players, sorted by
                               weighted PPR average (best first).

        Returns:
            dict with keys:
                "strategy"     — the generated strategy text (used by DraftAgent)
                "raw_response" — full Claude response
                "error"        — error message if something went wrong, else None
        """
        prompt = self._build_prompt(available_players)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            strategy_text = message.content[0].text.strip()
            return {
                "strategy":     strategy_text,
                "raw_response": strategy_text,
                "error":        None,
            }

        except anthropic.AuthenticationError:
            return {
                "strategy":     None,
                "raw_response": None,
                "error":        "Invalid API key. Check your .env file.",
            }
        except Exception as e:
            return {
                "strategy":     None,
                "raw_response": None,
                "error":        f"API error: {str(e)}",
            }

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(self, available_players: pd.DataFrame) -> str:
        """
        Builds the prompt that asks Claude to study the player pool and write
        its own draft strategy.

        We give Claude:
          - The league format and roster requirements
          - A compact snapshot of the top players at each position
            (weighted 3-year average, trend, years of data)
          - Instructions to write a strategy in the same structured format
            that the DraftAgent will follow when making picks
        """

        # ── Positional snapshots (top 50 per position) ────────────────────────
        # 50 gives Claude enough depth to reason about scarcity in any league
        # size. Even at 50 × 6 positions the token count is well under 1% of
        # Claude's 200K context window, so there's no meaningful limit here.
        def position_snapshot(pos: str) -> str:
            subset = available_players[available_players["position"] == pos].head(50)
            if subset.empty:
                return f"  {pos}: no players available"
            lines = []
            for _, row in subset.iterrows():
                wtd  = row.get("weighted_avg_ppr", 0)
                trend = row.get("trend", "")
                yrs  = int(row.get("years_of_data", 1))
                lines.append(
                    f"  #{int(row['season_rank']):>3}  {row['full_name']:<22} "
                    f"{str(row.get('team','FA')):<4}  "
                    f"{wtd:.1f} wtd avg/gm  trend: {trend}  ({yrs}yr)"
                )
            return f"  {pos} (top {len(subset)}):\n" + "\n".join(lines)

        snapshots = "\n\n".join(
            position_snapshot(pos) for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]
        )

        # ── Scoring gap analysis — helps Claude spot positional cliffs ────────
        # Show the drop-off from #1 to #12 so Claude can see where depth falls off.
        def scoring_gap(pos: str) -> str:
            subset = available_players[available_players["position"] == pos].head(12)
            if len(subset) < 2:
                return f"  {pos}: insufficient data"
            top  = subset.iloc[0].get("weighted_avg_ppr", 0)
            last = subset.iloc[-1].get("weighted_avg_ppr", 0)
            return f"  {pos}: #{1} = {top:.1f}  →  #{len(subset)} = {last:.1f}  (drop-off: {top-last:.1f} pts/gm)"

        gaps = "\n".join(scoring_gap(pos) for pos in ["QB", "RB", "WR", "TE"])

        prompt = f"""You are an expert fantasy football analyst preparing for a {self.num_teams}-team PPR snake draft with {self.num_rounds} rounds.

Your job is to study the player pool below and write a complete draft strategy that an AI agent will follow when making picks.

LEAGUE FORMAT:
- Scoring: PPR (1 point per reception)
- Starting lineup: 1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX (RB or WR only), 1 K, 1 DEF
- Roster size: {self.num_rounds} players per team
- Teams: {self.num_teams}

SCORING GAP ANALYSIS (weighted 3-year PPR avg, top 6 per position):
{gaps}

AVAILABLE PLAYERS (top 15 per position, ranked by weighted 3-year PPR average):
  Format: Rank  Name  Team  WeightedAvg/gm  trend  (years of data)

{snapshots}

Based on this player pool, write a draft strategy the AI agent should follow.
Your strategy should:
1. Identify which positions have the biggest drop-off between elite and average players — those deserve early picks.
2. Identify which positions have depth — those can be left for later rounds.
3. Give clear round-by-round guidance (e.g., "Rounds 1-2: target RB/WR", "Wait on QB until round X").
4. Address kicker and defense timing.
5. Note any standout players or positions worth targeting based on what you see in the data.

Write the strategy in this exact format (the DraftAgent reads it directly):

TEAM STRUCTURE:

Starting lineup composition:
[Describe the lineup and what it means for draft priorities]

Starting slot priority:
[Which positions to fill first and why]

Backup philosophy:
[Which positions to draft backups for and which to skip]

---

ROUND-BY-ROUND STRATEGY:

Rounds 1-2 (Foundation):
[What to target and why, based on the player pool above]

Rounds 3-4 (Core):
[Continued priorities and first flexibility decisions]

Quarterback timing:
[When to draft QB and why, based on the depth you see]

Tight end timing:
[When to draft TE and why]

Kicker and Defense:
[When to draft these and why]

Do not add anything outside this format. Write the strategy now."""

        return prompt
