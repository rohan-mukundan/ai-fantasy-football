"""
lineup_agent.py
----------------
The Lineup Agent uses Claude to explain, in plain English, why this
week's starting lineup looks the way it does.

It's given the computed starters/bench (including each player's
preseason projection, this season's actuals so far, and the blended
number actually used to rank them), plus any bye-week swaps, and asked
for a short, friendly summary a manager could read before kickoff —
the same idea as the bye-week explanation message, but covering the
whole lineup and the in-season performance trends behind it.

Usage:
    from agents.lineup_agent import LineupAgent

    agent = LineupAgent()
    summary = agent.summarize_lineup(week, season, starters, bench, swapped)
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()  # Loads ANTHROPIC_API_KEY from your .env file


class LineupAgent:
    """Uses Claude to summarize the rationale behind a week's lineup."""

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Main public method ─────────────────────────────────────────────────────

    def summarize_lineup(
        self,
        week: int,
        season: str | None,
        starters: list[dict],
        bench: list[dict],
        swapped: list[dict],
    ) -> str | None:
        """
        Asks Claude for a short (3-5 sentence) explanation of this week's
        lineup — notable bye-week swaps, breakout players who earned a
        starting spot based on this season's results, and proven players
        being kept despite a slow start.

        Returns the summary text, or None if the API call fails (the UI
        should just skip showing a rationale in that case).
        """
        prompt = self._build_prompt(week, season, starters, bench, swapped)

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception:
            # If the API key is missing/invalid or the call fails for any
            # reason, just don't show a rationale rather than erroring the page.
            return None

    # ── Prompt builder ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        week: int,
        season: str | None,
        starters: list[dict],
        bench: list[dict],
        swapped: list[dict],
    ) -> str:

        def fmt(p: dict) -> str:
            parts = [f"{p['full_name']} ({p['slot']}, {p['position']}, {p.get('nfl_team', '')})"]
            parts.append(f"preseason avg {p.get('avg_pts_ppr', 0) or 0:.1f} pts/gm")
            weeks_played = p.get("weeks_played_current", 0)
            if weeks_played:
                parts.append(
                    f"this season avg {p.get('current_avg_ppr', 0):.1f} pts/gm over {weeks_played} game(s)"
                )
                parts.append(f"blended {p.get('blended_avg_ppr', 0):.1f}")
            matchup = p.get("matchup")
            if matchup:
                parts.append(
                    f"Week {week} matchup: vs {matchup['opponent']} "
                    f"({matchup['matchup_label']} defense for {p['position']}, "
                    f"#{matchup['recent_rank']} last 4 weeks, "
                    f"avg {matchup['recent_avg']:.1f} pts/gm allowed)"
                )
            return "  - " + ", ".join(parts)

        starters_text = "\n".join(fmt(p) for p in starters) or "  (none)"

        # Bench: just show the top few by blended/preseason average for context
        bench_sorted = sorted(
            bench, key=lambda p: p.get("blended_avg_ppr", p.get("avg_pts_ppr", 0) or 0), reverse=True
        )[:6]
        bench_text = "\n".join(
            f"  - {p['full_name']} ({p['position']}, {p.get('nfl_team', '')}), "
            f"preseason avg {p.get('avg_pts_ppr', 0) or 0:.1f} pts/gm"
            + (f", this season avg {p.get('current_avg_ppr', 0):.1f} over {p.get('weeks_played_current', 0)} game(s)"
               if p.get("weeks_played_current") else "")
            for p in bench_sorted
        ) or "  (none)"

        if swapped:
            swap_lines = []
            for s in swapped:
                reason = s.get("reason", "")
                if reason == "bye":
                    swap_lines.append(
                        f"  - {s['out']['full_name']} ({s['out'].get('nfl_team', '')}) is on a bye, "
                        f"so {s['in']['full_name']} ({s['in'].get('nfl_team', '')}) starts in their place."
                    )
                elif reason == "cut":
                    swap_lines.append(
                        f"  - {s['out']['full_name']} is no longer on an NFL roster (released/unsigned), "
                        f"so {s['in']['full_name']} ({s['in'].get('nfl_team', '')}) starts in their place."
                    )
                # "waiver_needed" entries are pre-confirmation — omit from rationale
                # since the manager hasn't actually claimed anyone yet.
            swap_text = "\n".join(swap_lines) if swap_lines else "  (none)"
        else:
            swap_text = "  (none)"

        season_note = f"the {season} season" if season else "this season"

        prompt = f"""You are a friendly fantasy football assistant. Briefly explain the rationale behind a fantasy manager's Week {week} starting lineup for {season_note}.

This week's STARTERS (with preseason projection, this-season actuals if any games have been played, the blended number used to rank them, and — when available — this week's defensive matchup):
{starters_text}

Notable BENCH players for context:
{bench_text}

BYE-WEEK SWAPS AND WAIVER MOVES made this week:
{swap_text}

Write a short, friendly 3-5 sentence summary for the manager covering:
- Any bye-week swaps and why they were necessary.
- Any players who are starting (or moved up) because they're outperforming their preseason projection this season — call out breakout candidates.
- Any proven/highly-projected players who are still starting despite a slow start this season, and reassure the manager this is intentional (small sample size, not overreacting to one or two bad games).
- If any starters have notably easy or tough matchups this week (labeled as "easy" or "tough" in the matchup data), briefly mention it — especially if two players had similar scores and matchup was a deciding factor.
- If nothing notable changed from a straightforward "best players start" lineup, just say so briefly.

Keep it conversational and concise — no headers, no bullet points, just a short paragraph."""

        return prompt
