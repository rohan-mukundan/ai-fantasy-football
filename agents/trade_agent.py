"""
trade_agent.py
--------------
The Trade Agent uses Claude to generate a short, friendly pitch for a
proposed trade package — explaining why the deal makes sense for the
AI team and framing it in a way that the other manager might actually
accept.

Supports any package structure: 1-for-1, 2-for-1, 1-for-2, 3-for-1,
2-for-2, etc.  The algorithmic work (finding fair-value packages, scoring
value changes, checking roster needs) is done in app.py.  This agent is
purely responsible for the narrative.

Usage:
    from agents.trade_agent import TradeAgent

    agent = TradeAgent()
    pitch = agent.generate_pitch(
        give_players=[{"full_name": "Josh Gibson", "position": "RB", ...},
                      {"full_name": "Kadarius Toney", "position": "WR", ...}],
        get_players=[{"full_name": "Ja'Marr Chase", "position": "WR", ...}],
        my_team_name="Rohan's Team",
        other_team_name="Team 3",
        my_needs=["WR"],
        my_surplus=["RB", "WR"],
        week=6,
        season="2025",
    )
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()


class TradeAgent:
    """Uses Claude to write the pitch for a proposed trade package."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_pitch(
        self,
        give_players:    list[dict],
        get_players:     list[dict],
        my_team_name:    str,
        other_team_name: str,
        my_needs:        list[str],
        my_surplus:      list[str],
        week:            int,
        season:          str | None,
        sell_high:       bool = False,
        buy_low:         bool = False,
    ) -> str | None:
        """
        Generates a 2-3 sentence trade pitch for any package structure.

        Parameters
        ----------
        give_players     list of player dicts my team is giving away
        get_players      list of player dicts my team is receiving
        my_team_name     display name for the AI team
        other_team_name  display name for the trade partner
        my_needs         positions where AI team wants improvement
        my_surplus       positions where AI team has depth to spare
        week             current week number
        season           season year string e.g. "2025"
        sell_high        True if any give player is over-valued by the market
        buy_low          True if any get player has strong prior but rough start

        Returns the pitch text, or None if the API call fails.
        """
        prompt = self._build_prompt(
            give_players, get_players, my_team_name, other_team_name,
            my_needs, my_surplus, week, season, sell_high, buy_low,
        )
        try:
            message = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=220,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception:
            return None

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        give_players:    list[dict],
        get_players:     list[dict],
        my_team_name:    str,
        other_team_name: str,
        my_needs:        list[str],
        my_surplus:      list[str],
        week:            int,
        season:          str | None,
        sell_high:       bool,
        buy_low:         bool,
    ) -> str:
        season_note = f"the {season} season" if season else "this season"

        def fmt_player(p: dict) -> str:
            name  = p["full_name"]
            pos   = p["position"]
            team  = p.get("nfl_team", "")
            prior = p.get("avg_pts_ppr", 0) or 0
            val   = p.get("blended_avg_ppr", prior)
            cur   = p.get("current_avg_ppr")
            wks   = p.get("weeks_played_current", 0)
            line  = f"{name} ({pos}, {team}) — preseason {prior:.1f} pts/gm"
            if wks and cur is not None:
                line += f", {season_note} avg {cur:.1f} over {int(wks)} games"
            line += f", blended {val:.1f}"
            return line

        give_lines = "\n".join(f"  • {fmt_player(p)}" for p in give_players)
        get_lines  = "\n".join(f"  • {fmt_player(p)}" for p in get_players)

        give_n = len(give_players)
        get_n  = len(get_players)
        if give_n == get_n == 1:
            structure = "1-for-1 swap"
        elif give_n > get_n:
            structure = f"{give_n}-for-{get_n} package (consolidating depth for a star)"
        else:
            structure = f"{give_n}-for-{get_n} package (trading a star for depth)"

        angle_notes = []
        if sell_high:
            sell_names = [
                p["full_name"] for p in give_players
                if p.get("weeks_played_current", 0) >= 2
                and p.get("blended_avg_ppr", 0) > 0
                and (p.get("blended_avg_ppr", 0) - (p.get("current_avg_ppr") or p.get("blended_avg_ppr", 0)))
                    / p.get("blended_avg_ppr", 1) > 0.20
            ]
            if sell_names:
                angle_notes.append(
                    f"- SELL HIGH opportunity: {', '.join(sell_names)} "
                    f"still have high market value from preseason projections "
                    f"despite underperforming this season. Good time to move them."
                )
        if buy_low:
            buy_names = [
                p["full_name"] for p in get_players
                if p.get("avg_pts_ppr", 0) >= 8.0
                and p.get("current_avg_ppr") is not None
                and p.get("current_avg_ppr") < p.get("avg_pts_ppr", 0) * 0.7
            ]
            if buy_names:
                angle_notes.append(
                    f"- BUY LOW opportunity: {', '.join(buy_names)} "
                    f"have a strong preseason pedigree but have had a rough start. "
                    f"Their owner may be willing to deal them at a discount."
                )

        angle_text = "\n".join(angle_notes) if angle_notes else "(no special angle)"
        needs_text  = ", ".join(my_needs)   if my_needs   else "none identified"
        surplus_text = ", ".join(my_surplus) if my_surplus else "none identified"

        return f"""You are a fantasy football trade advisor for Week {week} of {season_note}.

THE PROPOSED TRADE — {structure}:
  {my_team_name} gives:
{give_lines}

  {other_team_name} gives:
{get_lines}

MY TEAM'S SITUATION:
  Positions where I need improvement: {needs_text}
  Positions where I have surplus depth: {surplus_text}

SPECIAL ANGLES:
{angle_text}

Write a short 2-3 sentence trade pitch that:
1. Explains clearly why {my_team_name} benefits from this deal
2. Frames WHY {other_team_name} would actually want to accept (what they gain)
3. If there's a sell-high or buy-low angle, weave it in naturally without using those exact terms

Keep it conversational and direct — the kind of message you'd actually send in the trade chat.
No bullet points, no headers, just 2-3 natural sentences."""
