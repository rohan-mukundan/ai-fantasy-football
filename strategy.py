"""
strategy.py
-----------
Defines named draft strategies available in the app.

STRATEGIES is a dict mapping strategy display names to their strategy text.
Add new strategies here and they will automatically appear in the UI dropdown.

Special values:
  None              → strategy is generated at draft time by the StrategyAgent
  "__LEAGUE_SIZE__" → strategy is selected at draft time based on num_teams
                      using LEAGUE_SIZE_STRATEGIES below (data-backed PPR strategies)
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


# ── League-size-specific data-backed PPR strategies ───────────────────────────
# Derived from PFR 2020-2025 actual PPR finishes and positional scarcity analysis.
# Keyed by exact team count; get_league_strategy() handles rounding for odd sizes.

LEAGUE_SIZE_STRATEGIES = {

    6: """
DATA-BACKED PPR STRATEGY — 6-TEAM LEAGUE
Source: PFR 2020-2025 actual PPR finish data + positional scarcity analysis

CORE PHILOSOPHY: Pure BPA (Best Player Available). With only 90 total roster spots
and ~910 players on waivers, no positional scarcity exists. The waiver wire covers
everything. Never reach for a position out of urgency.

POSITIONAL SCARCITY (6 teams):
- QB: 30% of NFL starters claimed — massive waiver pool. Never draft before round 6.
- RB: 42% claimed — 21 RBs remain on waivers. No urgency.
- WR: 25% claimed — 45 WRs on waivers. Pure BPA.
- TE: 43% claimed — 8 TEs on waivers. Stream freely or grab elite TE if available round 3.

ROUND-BY-ROUND GUIDE:
Rounds 1-2: Pure BPA — best available RB or WR. No positional lean needed.
Round 3: Continue BPA. If a top-3 TE (Kelce/Andrews tier) slips here, take them —
  PFR data shows top-2 TEs finish top-8 overall every year.
Rounds 4-5: BPA. First QB consideration window, but not required.
Rounds 6-7: Draft your QB by round 7. With 14 QBs on waivers, take highest upside option.
Rounds 8-9: Draft TE if not taken. Otherwise, RB/WR depth and FLEX candidates.
Rounds 10-12: One RB handcuff ONLY if your RB1 is a true workhorse (CMC/Henry tier).
  Otherwise pure upside shots — high-ceiling rookies, situation RBs, PPR slot receivers.
Rounds 13-14: High-upside lottery picks. Undrafted/late-round players finish top-30
  every year (PFR data: ~1.2 RBs, ~1.0 WRs per year). These rounds find them.
Round 15 (last): Kicker. Always last.
Round 14: DEF — stream matchup-by-matchup from 26 available on waivers.

BACKUP COUNTS (data-justified):
- QB: 0 backups (14 QBs on waivers; 0% injury rate in top-20 QBs per PFR 2020-25)
- RB: 0-1 backups (only handcuff a true workhorse RB1; 21 RBs on waivers otherwise)
- WR: 0 backups (45 WRs on waivers; stream freely)
- TE: 0 backups (8 TEs on waivers)
- K: 0 — always stream
- DEF: 0 — always stream

KEY DATA INSIGHTS:
- Every year, a QB finishes top-5 who was available in ADP rounds 5-8 (Love 2023, Daniels 2024)
- ~3 top-15 ADP RBs bust every year (Henry 2021, Taylor 2022, CMC 2020) — don't reach
- 1 WR and 1 RB from waiver wire finish top-30 every year — the wire is the strategy
- NEVER draft K before round 15, NEVER draft 2 DEF or 2 K
""".strip(),

    8: """
DATA-BACKED PPR STRATEGY — 8-TEAM LEAGUE
Source: PFR 2020-2025 actual PPR finish data + positional scarcity analysis

CORE PHILOSOPHY: BPA with a slight RB lean. RB scarcity is starting (56% demanded),
but the waiver wire is still large (~880 players). No panic-drafting needed, but
when two players are within 1 tier of each other and one is RB, take the RB.

POSITIONAL SCARCITY (8 teams):
- QB: 40% claimed — 12 QBs on waivers. Still very streamable. Draft rounds 5-7.
- RB: 56% claimed — 16 RBs on waivers. Getting thin. One handcuff is now justified.
- WR: 33% claimed — 40 WRs on waivers. Abundant. No urgency.
- TE: 57% claimed — 6 TEs on waivers. Thin but streamable. Elite TE or stream.

ROUND-BY-ROUND GUIDE:
Round 1: Pure BPA. Elite RBs and elite WRs both belong in round 1.
Round 2: BPA with RB lean — if you have WR in R1, weight toward RB here.
Round 3: Elite TE window. If a top-3 TE is available, take them — top-2 TEs averaged
  top-8 overall finishes 2020-2024. Otherwise take BPA RB/WR.
Round 4: Balance your starters. Aim for 2 RBs and 1-2 WRs by end of round 4.
Rounds 5-6: QB window. Draft your QB here. Per PFR data, QB5-6 still finishes QB3-5 most years.
Rounds 7-8: Draft your RB1 handcuff — mandatory. With 16 RBs on waivers, the wire
  empties when an elite RB gets hurt. Lock in the handcuff before it's gone.
Rounds 9-10: WR/RB depth — high upside shots. PPR slot receivers, situation RBs.
Rounds 11-12: Lottery picks and DEF (strong Week 1 matchup).
Rounds 13-14: Pure upside: rookies with roles, injury-return players.
Round 15: Kicker. Always last.

BACKUP COUNTS (data-justified):
- QB: 0 backups (12 QBs on waivers; QB is always streamable in 8-team)
- RB: 1 backup — ONLY handcuff your RB1 if they are a workhorse (bell-cow role).
  Committee backs' handcuffs are not worth a roster spot (8-10 carries, not 20+).
- WR: 0 backups (40 WRs on waivers)
- TE: 0 backups (stream from 6 available TEs)
- K: 0 — always stream
- DEF: 0 — always stream

KEY DATA INSIGHTS:
- RB injury rate: 14% of top-24 RBs miss 3+ games per year (PFR 2020-25)
- In 8-team league, 16 RBs remain on wire — when elite RB goes down, wire empties fast
- WR bust rate at top-10 ADP: only 1-2 per year — safer to hold elite WRs than elite RBs
- NEVER draft K before round 15, NEVER draft 2 DEF or 2 K
""".strip(),

    10: """
DATA-BACKED PPR STRATEGY — 10-TEAM LEAGUE
Source: PFR 2020-2025 actual PPR finish data + positional scarcity analysis

CORE PHILOSOPHY: RB-heavy draft. RB scarcity hits 69% and TE hits 71% — real scarcity
that affects your season. Draft RBs in rounds 1-4 and 7, let the waiver wire handle WR
emergencies (35 WRs remain on waivers). QB still rounds 5-7 — never before.

POSITIONAL SCARCITY (10 teams):
- QB: 50% claimed — 10 QBs on waivers. Draft by round 7, no earlier.
- RB: 69% claimed (HIGH) — only 11 RBs on waivers. MANDATORY handcuff.
- WR: 42% claimed — 35 WRs on waivers. Still abundant. No panic-drafting.
- TE: 71% claimed (HIGH) — only 4 TEs on waivers. Elite TE or pure stream from thin pool.

ROUND-BY-ROUND GUIDE:
Round 1: BPA. Elite RBs and elite WRs both belong here. If picking early, lean RB.
Round 2: RB lean. If you have WR in R1, take an RB here. 69% scarcity means RB2 tier
  is the most important position to lock down early.
Round 3: Elite TE window — last realistic shot at a top-4 TE. In 10-team drafts,
  top-3 TEs go by pick 25. If one is available, take them even over a solid RB.
  Otherwise take BPA RB/WR and plan to stream TE from 4 thin waiver options.
Round 4: Lock in your starting lineup. Aim for 2 RBs + 2 WRs by end of round 4.
Rounds 5-6: QB draft window. Always draft QB by round 7. Per PFR data, top-5 QB is
  available rounds 5-8 every year. Also grab a TE here if streaming is not acceptable.
Round 7: MANDATORY RB handcuff. With only 11 RBs on waivers, when your starter goes
  down the wire empties within hours. The handcuff must be held before injury happens.
Rounds 8-9: WR depth (FLEX insurance) — PPR slot receivers, WR2s with high target share.
  35 WRs on waivers means 1 WR backup is sufficient; don't hoard WRs.
Rounds 10-11: DEF (strong matchup), upside shots on RBs/WRs.
Rounds 12-14: Pure lottery — high ceiling rookies, breakout candidates, injury-return.
Round 15: Kicker. Always last.

BACKUP COUNTS (data-justified):
- QB: 0 backups (10 QBs on waivers; 0% QB injury rate in PFR 2020-25 top-20)
- RB: 1 handcuff — MANDATORY for your RB1. PFR 2020-25: 3-4 top-20 RBs miss 4+ games
  every year. With 11 RBs on waivers, the replacement is gone before you can claim him.
- WR: 0-1 backup (35 WRs on waivers; optional backup for injury insurance)
- TE: 0 backups (stream from 4 thin waiver TEs if you chose not to draft one)
- K: 0 — always stream
- DEF: 0 — always stream (22 DEFs on waivers)

KEY DATA INSIGHTS:
- Top-12 PPR finishers composition per year: avg 4 RBs, 3.5 WRs, 3.5 QBs, 1 TE
- RBs at 69% scarcity: the elite tier is the scarcest commodity in a 10-team league
- WR value: 1 undrafted WR finishes top-30 every year — waiver wire still produces
- TE scarcity means being TE12+ all year is a real competitive disadvantage
- NEVER draft K before round 15, NEVER draft 2 DEF or 2 K
""".strip(),

    12: """
DATA-BACKED PPR STRATEGY — 12-TEAM LEAGUE
Source: PFR 2020-2025 actual PPR finish data + positional scarcity analysis

CORE PHILOSOPHY: RB-first. RB scarcity hits 83% and TE hits 86% — only 6 RBs and
2 TEs remain on waivers after drafts. Your draft must account for this: 3-4 RBs in
first 8 picks, elite TE in rounds 1-3 or accept a season-long streaming penalty,
QB in rounds 7-9. Treat the waiver wire as nearly empty for RB and TE.

POSITIONAL SCARCITY (12 teams):
- QB: 60% claimed — 8 QBs on waivers. Draft by round 9. Streamable but inconsistent.
- RB: 83% claimed (CRITICAL) — only 6 RBs on waivers. 2 MANDATORY handcuffs.
- WR: 50% claimed — 30 WRs on waivers. Moderate. 1 WR backup now justified.
- TE: 86% claimed (CRITICAL) — only 2 TEs on waivers. Elite TE now changes your season.

ROUND-BY-ROUND GUIDE:
Round 1: BPA with strong RB lean. In a 12-team draft, round 1 often goes 8-10 RBs.
  If two players within a tier and one is RB, always take the RB.
Round 2: RB priority. If you have WR in R1, take an RB. If you have 2 RBs, take BPA WR.
  RB2-tier (RB7-14 range) players are your season-long starters — lock them down early.
Round 3: TE NOW OR NEVER. With only 2 TEs on waivers, streaming TE is a real
  competitive disadvantage all season. PFR data: gap between TE1 and TE15 is 8-12
  PPR pts/game (140-200 pts over 17 games). If a top-4 TE is available, take them
  even over a solid RB. This is the highest-leverage round 3 decision.
Round 4: Balance. Aim for 2 RBs + 2 WRs. If missing a position's second starter, get it now.
Rounds 5-6: Add RB3 (a real depth pick, not just a handcuff — will start multiple weeks).
  Also consider WR3 for FLEX. QB not urgent yet.
Rounds 7-8: RB1 handcuff (mandatory) AND QB window. Per PFR data, top-8 QB still
  available rounds 7-8 every year. Draft BOTH in these rounds.
Round 9: RB2 handcuff — mandatory at 12 teams. With only 6 RBs on waivers, when your
  RB2 gets hurt there is nothing on the wire. Two handcuffs is the correct call.
Rounds 10-11: WR depth (1 backup) and DEF. With 30 WRs on waivers, 1 WR backup is enough.
Rounds 12-14: Pure upside shots. Injury-return players, breakout candidates.
Round 15: Kicker. Always last.

BACKUP COUNTS (data-justified):
- QB: 0 backups (8 QBs on waivers; QB injury rare — 0% rate in PFR 2020-25 top-20)
- RB: 2 handcuffs — MANDATORY for RB1 AND RB2. Only 6 RBs on waivers total.
  When elite RB goes down, the wire is picked clean within 12 hours. Pre-emptive
  handcuffing is the only protection in a 12-team league.
- WR: 1 backup (30 WRs on waivers — enough to find one, but getting thin)
- TE: 0 backups (only 2 TEs on waivers; no real backup exists to draft)
- K: 0 — always stream
- DEF: 0 — stream matchup weekly (20 DEFs on waivers)

KEY DATA INSIGHTS:
- Managers finishing with 2 top-12 RBs win 70%+ of matchups (primary win driver)
- TE1 vs TE15 gap: 8-12 PPR pts/game = 140-200 pts over the season
- WR3 upside beats WR3 safety — 30 WRs on waivers means you can replace a bust
- NEVER draft K before round 15, NEVER draft 2 DEF or 2 K
""".strip(),

    14: """
DATA-BACKED PPR STRATEGY — 14-TEAM LEAGUE
Source: PFR 2020-2025 actual PPR finish data + positional scarcity analysis

CORE PHILOSOPHY: RB hoarding. RB scarcity hits 97% — the NFL has ~36 startable RBs
and a 14-team league claims 35 of them. TE hits 100% — all TEs are drafted, zero on
waivers. QB hits 70% — even QB timing matters for the first time. Target 5-6 RBs total.
The waiver wire cannot bail you out at RB or TE. Draft with this as your foundation.

POSITIONAL SCARCITY (14 teams):
- QB: 70% claimed — 6 QBs on waivers. DRAFT BY ROUND 7-8. Late QBs = QB14 territory.
- RB: 97% claimed (EXTREME CRISIS) — only 1 RB on waivers after drafts. It's RB36.
  Every injury to any RB without a handcuff is season-ending.
- WR: 58% claimed — 25 WRs on waivers. Still the most findable position. WR value here.
- TE: 100% claimed (DEPLETED) — ZERO TEs on waivers. Starting TE15+ all year = penalty.

ROUND-BY-ROUND GUIDE:
Round 1: RB-heavy lean. Take an elite RB unless a top-5 overall WR is on the board.
  In a 14-team draft, round 1 often goes 10 RBs and 4 WRs.
Round 2: TE NOW — last realistic window for elite TE. With ZERO TEs on waivers, elite TE
  is a must-draft. PFR: top-2 TEs finish top-8 overall every year. A top-3 TE available
  in round 2 is taken even over a solid RB — the positional advantage lasts 17 weeks.
  If no elite TE is available, take RB and accept streaming TE15+ all season.
Round 3: Lock in 2 RBs + at least 1 WR. TE2 tier (rank 4-7) may still be available here.
Round 4: STARTER LOCK — must have 2 RBs + 2 WRs. If missing either, correct now.
  The wire cannot rescue you. What you draft is what you have.
Rounds 5-6: QB window + RB3. Draft your QB in rounds 5-7. Also add your RB3 here —
  in a 14-team league, your RB3 starts multiple weeks due to injuries.
Rounds 7-8: RB1 handcuff — MANDATORY. With only 1 RB on waivers, your handcuff is
  the ONLY replacement available when your starter goes down. No exceptions.
Round 9: RB2 handcuff — MANDATORY. Two handcuffs, two protected RB starters.
Rounds 10-11: WR depth (2 WR backups justified — wire gets thin by week 4) and DEF.
  DEF must be drafted by round 11 in a 14-team league — matchup streaming unreliable.
Rounds 12-13: Consider a 3rd RB handcuff if roster allows. Otherwise pure upside shots.
Round 14: Lottery picks — pure ceiling.
Round 15: Kicker. Always last.

BACKUP COUNTS (data-justified):
- QB: 0-1 backup (consider 1 only if back-to-back bye weeks create gaps; 6 QBs on wire)
- RB: 2-3 handcuffs — MANDATORY. 1 RB on waivers total. This is not optional.
  PFR 2020-25: 3-4 top-20 RBs miss 4+ games every year. No handcuff = dead roster spot.
- WR: 2 backups (25 WRs on waivers — findable, but wire thin by midseason)
- TE: 0 backups (no TEs on waivers; only option is what's already drafted)
- K: 0 — always stream
- DEF: 0 — stream weekly but draft 1 good matchup DEF by round 11

KEY DATA INSIGHTS:
- RB is the entire game at 14 teams: 97% scarcity, 1 wire RB, 3-4 busts per year
- TE is completely exhausted: no waivers, streaming TE15+ = 8-12 pts/game penalty
- WR is where you find value: 25 on waivers, 1 undrafted WR finishes top-30 every year
- Trade mid-season: WR depth is currency; RBs are always at a premium at 14 teams
- NEVER draft K before round 15, NEVER draft 2 DEF or 2 K
""".strip(),

}


def get_league_strategy(num_teams: int) -> str:
    """
    Returns the data-backed PPR draft strategy for the given league size.
    For league sizes not explicitly defined (e.g. 7, 9, 11, 13), rounds to
    the nearest defined size: 6, 8, 10, 12, or 14.
    """
    defined_sizes = sorted(LEAGUE_SIZE_STRATEGIES.keys())  # [6, 8, 10, 12, 14]
    # Find the closest defined size
    closest = min(defined_sizes, key=lambda s: abs(s - num_teams))
    return LEAGUE_SIZE_STRATEGIES[closest]


# ── Strategy registry ──────────────────────────────────────────────────────────
# Maps the display name (shown in the UI dropdown) to the strategy text.
#   None              → generated at draft time by StrategyAgent
#   "__LEAGUE_SIZE__" → selected at draft time using get_league_strategy(num_teams)

STRATEGIES = {
    "League-Size Optimized (PPR)": "__LEAGUE_SIZE__",  # data-backed, auto-selects by team count
}
