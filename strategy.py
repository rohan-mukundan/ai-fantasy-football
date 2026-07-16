"""
strategy.py
-----------
Defines named draft strategies available in the app.

STRATEGIES is a dict mapping strategy display names to their strategy text.
Add new strategies here and they will automatically appear in the UI dropdown.

Special value: None means the strategy will be generated at draft time by
the StrategyAgent (Claude analyzes the player pool and writes its own plan).
"""

# ── Individual strategy definitions ───────────────────────────────────────────

DEFAULT_STRATEGY = """
TEAM STRUCTURE:

Starting lineup composition:
The starting lineup consists of 1 quarterback, 2 running backs, 2 wide receivers, 1 tight end, 1 flex (RB or WR only), 1 kicker, and 1 defense. Every draft decision should be made with this structure in mind.

Starting slot priority:
Before drafting any backups, fully fill the starting slots for wide receiver, running back, and flex. These are the highest-scoring and most impactful positions in the lineup.

Backup philosophy:
Do NOT draft backup quarterbacks, kickers, or defenses. These positions can be addressed via the waiver wire during bye weeks. Using a roster spot on a backup at these positions wastes a pick that should instead go toward high-upside running backs and wide receivers who can contribute to the starting lineup.

---

ROUND-BY-ROUND STRATEGY:

Rounds 1-2 (Foundation):
Select the best available running back or wide receiver. These two positions have the widest gap between elite and average players, so securing top talent here is the highest priority.

Rounds 3-4 (Core + First Flexibility):
Continue prioritizing running backs and wide receivers. However, if a top-tier tight end (one of the top 2 at the position overall) is available, they may be considered and compared directly against the best available running backs and wide receivers — only draft them if their value is clearly competitive.

Quarterback timing:
Quarterbacks score more total points than other positions, but the gap between the QB1 and QB12 is much narrower than the gap between top and average running backs or wide receivers. Do not select a quarterback before round 4. Begin considering a quarterback in round 4 only if top options remain available. If the top quarterbacks are gone, deprioritize the position entirely and target one in rounds 8-12, where the remaining quarterbacks will deliver similar output anyway.

Tight end timing:
Top tight ends may be considered in rounds 3-4 if they are genuine elite options. If consistent scoring tight ends are no longer available, drop tight end on the priority list — similar to the quarterback approach. Tight ends do not match the scoring ceiling of elite running backs and wide receivers, so taking upside chances at those positions is preferable to reaching for a mid-tier tight end.

Kicker and Defense:
Leave kicker and defense for the final two picks of the draft. They have the lowest and least predictable scoring of any position and should never take a roster spot that could go to a high-upside skill player.
""".strip()

# ── Strategy registry ──────────────────────────────────────────────────────────
# Maps the display name (shown in the UI dropdown) to the strategy text.
# None means the strategy is generated dynamically by the StrategyAgent.

STRATEGIES = {
    "AI Agent Strategy": None,   # generated at draft time by StrategyAgent
    "My Draft Rules":    DEFAULT_STRATEGY,
}
