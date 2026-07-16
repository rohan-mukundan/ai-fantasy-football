# AI Fantasy Football — Project Architecture

## The Big Picture

You're running two teams in the same fantasy league:
- **Team A (Human):** You make every decision manually
- **Team B (AI Agent):** Claude makes draft picks and lineup decisions automatically

At season's end, you compare their records head-to-head to measure whether AI decision-making outperforms human judgment.

---

## System Overview

```
                          ┌─────────────────────────────┐
                          │       Data Layer             │
                          │                              │
                          │  Sleeper API (free, no key)  │
                          │  - Player roster/info        │
                          │  - Weekly stats              │
                          │  - Injury reports            │
                          └────────────┬────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────────┐
                          │     data_pipeline/           │
                          │                              │
                          │  sleeper_client.py           │
                          │  - Fetches raw data via API  │
                          │                              │
                          │  data_processor.py           │
                          │  - Cleans & shapes data      │
                          │  - Saves to /data/ folder    │
                          └────────────┬────────────────┘
                                       │
                          ┌────────────┴────────────────┐
                          │                             │
                          ▼                             ▼
           ┌──────────────────────┐     ┌──────────────────────┐
           │   agents/            │     │   agents/            │
           │   draft_agent.py     │     │   lineup_agent.py    │
           │                      │     │                      │
           │  Input:              │     │  Input:              │
           │  - Strategy prompt   │     │  - My current roster │
           │  - Players taken     │     │  - Weekly stats      │
           │  - Available players │     │  - Matchup data      │
           │                      │     │  - Injury news       │
           │  Output:             │     │                      │
           │  - Pick recommendation│    │  Output:             │
           │  - Reasoning         │     │  - Starting lineup   │
           └──────────┬───────────┘     │  - Reasoning         │
                      │                 └──────────┬───────────┘
                      └────────────┬───────────────┘
                                   │
                                   ▼
                      ┌────────────────────────┐
                      │   Anthropic Claude API  │
                      │                        │
                      │  Model: claude-sonnet  │
                      │  - Reasons over data   │
                      │  - Returns decisions   │
                      │    with explanations   │
                      └────────────────────────┘
```

---

## Project Folder Structure

```
ai-fantasy-football/
│
├── ARCHITECTURE.md         ← You are here
├── SETUP.md                ← Start here if setting up for the first time
├── requirements.txt        ← Python packages to install
├── .env.example            ← Template for your API key (copy to .env)
├── .env                    ← YOUR real API key (never share this)
│
├── data/
│   ├── raw/                ← Data pulled directly from APIs (don't edit)
│   └── processed/          ← Cleaned, analysis-ready data
│
├── data_pipeline/
│   ├── sleeper_client.py   ← Fetches data from Sleeper API
│   └── data_processor.py   ← Cleans and prepares data for agents
│
├── agents/
│   ├── draft_agent.py      ← Makes draft picks using Claude
│   └── lineup_agent.py     ← Sets weekly lineup using Claude
│
├── notebooks/
│   └── exploration.ipynb   ← Jupyter notebook for exploring data
│
└── main.py                 ← Entry point — run agents from here
```

---

## The Two Agents Explained

### 1. Draft Agent (`agents/draft_agent.py`)

**When it runs:** Once, at the start of the season during your fantasy draft.

**What it does:**
1. Receives a strategy prompt (e.g., "Prioritize running backs in rounds 1-3, then target wide receivers")
2. Receives a list of players already drafted by other teams
3. Looks at the remaining available players and their projected stats
4. Sends all of this to Claude with a prompt asking: "Given this strategy and available players, who should I pick next and why?"
5. Returns Claude's pick recommendation and the reasoning behind it

**Key insight:** Claude doesn't just return a name — it explains *why*, which helps you learn about fantasy football strategy.

### 2. Weekly Lineup Agent (`agents/lineup_agent.py`)

**When it runs:** Once per week, before the game-day roster lock deadline.

**What it does:**
1. Looks at your full roster of players
2. Checks each player's upcoming matchup difficulty (e.g., a running back facing a weak run defense)
3. Checks for injury news (who is questionable, out, etc.)
4. Sends all of this to Claude with the prompt: "Given this roster, matchup data, and injury news, what's my optimal starting lineup and why?"
5. Returns the recommended starters with reasoning

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Language | Python 3.11+ | Industry standard for data/AI work |
| Data fetching | Sleeper API | Free, no auth needed, great NFL coverage |
| Data manipulation | pandas | The standard for working with tabular data |
| AI reasoning | Anthropic Claude API | Best-in-class reasoning for structured decisions |
| Config management | python-dotenv | Keeps API keys out of your code |
| Local storage | CSV files + SQLite | Simple, no server needed |

---

## Data Flow: Summer Testing vs. Live Season

### Summer (Historical Testing)
```
Sleeper API (2024 historical stats)
    → data_pipeline/sleeper_client.py (fetch week-by-week)
    → data/processed/ (save as CSV files)
    → agents/ (simulate draft + weekly decisions using real 2024 data)
    → Compare: what would the AI have done vs. actual outcomes?
```

### Fall (Live Season)
```
Sleeper API (live 2025 stats, updated weekly)
    → agents/ (make real draft picks and lineup decisions)
    → Your fantasy league (apply AI's decisions to Team B)
    → Track: AI team record vs. your team record week by week
```

The data pipeline code is identical for both phases — you just change the season year and the data goes from historical to live.

---

## Development Phases

### Phase 1 — Foundation (Now through June)
- [x] Set up development environment
- [x] Understand the architecture
- [ ] Pull 2024 historical data from Sleeper API
- [ ] Explore and understand the data

### Phase 2 — Draft Agent (June–July)
- [ ] Build the Draft Agent
- [ ] Test it against the 2024 draft using historical ADP data
- [ ] Tune the strategy prompts

### Phase 3 — Lineup Agent (July–August)
- [ ] Build the Weekly Lineup Agent
- [ ] Back-test it against every week of the 2024 season
- [ ] Measure: how often would the AI have started the optimal lineup?

### Phase 4 — Live Season (September onward)
- [ ] Switch data source to live 2025 season
- [ ] Run both teams in your fantasy league
- [ ] Track and compare performance weekly

---

## Key Concepts for a Business Student

**Why Claude instead of simple rules?**
A rules-based system ("always start the highest-projected player") can't reason about context: a star player on a bye week, a great matchup for a backup, or injury risk. Claude reads the full situation and explains its reasoning — much like a knowledgeable analyst would.

**How to measure success?**
- Draft Agent: Did the AI team draft better value players than average draft position (ADP) suggested?
- Lineup Agent: What percentage of weeks did it set the theoretically optimal lineup (determined in hindsight)?
- Overall: Does the AI team finish with a better record than your human-managed team?
