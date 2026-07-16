# AI Fantasy Football Assistant

A multi-agent AI fantasy football manager built with Python, Streamlit, and Claude (Anthropic). Connect your Sleeper fantasy league and deploy a team of specialized AI agents that handle every aspect of your season — from draft day through the championship — each powered by real NFL data and explained in plain English.

---

## The Agent Team

| Agent | Role |
|---|---|
| **AI Draft Assistant** | Ranks players using 3 years of weighted PPR stats, blended with in-season performance. Explains every pick in the context of your team's needs. |
| **AI Weekly Lineup Agent** | Sets your optimal starting lineup each week with matchup-aware start/sit reasoning. Flags tough and favorable defensive matchups for every player. |
| **AI Waiver Wire Agent** | Surfaces the best available free-agent upgrades at each position and tells you who to drop to make room. |
| **AI Trade Analyzer Agent** | Evaluates multi-player trade packages considering positional value and team needs, then writes a persuasive pitch you can send straight to your opponent. |
| **AI Defensive Streaming Agent** | Each week, identifies the best available free-agent defense using a blended score: 75% opponent offense weakness + 25% DEF unit quality. |

Each agent is stateless and task-specific — it receives structured data from the pipeline, reasons over it with Claude, and returns a plain-English recommendation. No agent shares memory with another; all shared intelligence lives in the data layer.

---

## How It Works

```
                        ┌─────────────────────────────────┐
                        │         Sleeper API              │
                        │    (rosters, stats, leagues)     │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │         data_pipeline/           │
                        │  • Multi-year PPR stat history   │
                        │  • Live in-season stat blending  │
                        │  • Defense rankings (pts allowed)│
                        │  • Offense rankings (pts scored) │
                        │  • Weekly ESPN matchup cache     │
                        └────────────────┬────────────────┘
                                         │
                        ┌────────────────▼────────────────┐
                        │        app.py (Streamlit UI)     │
                        │                                  │
                        │   ┌──────────────────────────┐  │
                        │   │       agents/            │  │
                        │   │  ┌─────────────────────┐ │  │
                        │   │  │  Draft Agent        │ │  │
                        │   │  │  Lineup Agent       │ │  │
                        │   │  │  Waiver Wire Agent  │◄────── Claude API
                        │   │  │  Trade Agent        │ │  │   (Anthropic)
                        │   │  │  Strategy Agent     │ │  │
                        │   │  └─────────────────────┘ │  │
                        │   └──────────────────────────┘  │
                        └─────────────────────────────────┘
```

- **Data** is pulled from the free [Sleeper API](https://docs.sleeper.com/) and [ESPN's public scoreboard API](https://site.api.espn.com). No paid subscriptions required.
- **Rankings** use 3 seasons of weighted PPR averages (2022–2024) as a preseason baseline, then blend in current-season stats as the year progresses.
- **Defense rankings** compute PPR points allowed per position (QB/RB/WR/TE) per team each week, with prior-season blending to stabilize early-season noise.
- **Offense rankings** use expert preseason projections (The Ringer) for weeks 1–3, then switch to cumulative season averages for weeks 4–18.
- **AI reasoning** is handled by Claude (claude-sonnet-4-5) via the Anthropic API.

### The Agents

**AI Draft Assistant**
Receives the full player pool ranked by weighted 3-year PPR averages, your current roster, and the picks already made around the league. Recommends the best available player for your team's needs at each pick and explains the reasoning — why this player over the next best alternative, what hole it fills on your roster, and what to watch out for.

**AI Weekly Lineup Agent**
Each week, receives your full roster with each player's blended PPR average (preseason history + current season stats), their upcoming opponent, and a 🟢/🟡/🔴 defensive matchup grade based on how many fantasy points that defense has allowed to the player's position this season. Sets the optimal starting lineup and writes a plain-English summary explaining close calls and any players worth keeping an eye on.

**AI Waiver Wire Agent**
Scans all available free agents, compares them to your current roster by position, and surfaces the best upgrades ranked by projected output. Tells you who to drop to make room and why the swap is worth making.

**AI Trade Analyzer Agent**
Evaluates multi-player trade packages from both sides — factoring in positional scarcity, roster construction, and remaining schedule — then writes a persuasive pitch tailored to what the other manager needs. The goal isn't just to assess fairness; it's to help you close the deal.

**AI Defensive Streaming Agent**
Recommends which free-agent defense to pick up and start each week. Unlike skill-position players who you hold all season, defenses are best streamed weekly against the softest available opponent. The agent scores every available defense using a blended formula:

**Score = 75% matchup quality + 25% DEF unit quality**

- **Matchup quality**: how weak the upcoming opponent's offense is (rank 1 = weakest = best matchup)
- **DEF unit quality**: how many fantasy points the defense itself scores per game on average this season

**Weeks 1–3**: opponent offense strength is sourced from expert preseason rankings (The Ringer) rather than small-sample game data. Switches to cumulative season averages from week 4 onward.

---

## Setup

### 1. Prerequisites

- Python 3.10 or later
- A [Sleeper](https://sleeper.com) account in a fantasy football league
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/ai-fantasy-football.git
cd ai-fantasy-football
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and paste your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## First-Time Setup (inside the app)

1. **Season Setup tab** → enter your Sleeper username → the app fetches your leagues and downloads player stats for 2022–2024.
2. **My Leagues tab** → select your league → enter the current NFL week.
3. Done. All agents are ready.

The app caches all data locally in `data/` so subsequent loads are fast.

---

## Project Structure

```
├── app.py                          # Main Streamlit application
├── agents/
│   ├── draft_agent.py              # AI Draft Assistant
│   ├── lineup_agent.py             # AI Weekly Lineup Agent
│   ├── trade_agent.py              # AI Trade Analyzer Agent
│   └── strategy_agent.py           # AI Waiver Wire + Strategy Agent
├── data_pipeline/
│   ├── sleeper_client.py           # Sleeper API wrapper
│   ├── data_processor.py           # Multi-year stats aggregation
│   ├── current_season.py           # Live in-season stat fetching
│   ├── defense_rankings.py         # PPR pts allowed/scored + DEF streaming logic
│   ├── depth_chart_scraper.py      # Depth chart data
│   └── schedule_fetcher.py         # NFL schedule fetching
├── data/
│   ├── processed/
│   │   ├── preseason_offense_rankings_2025.json   # The Ringer expert rankings
│   │   ├── matchup_cache/          # Weekly ESPN matchup data (cached)
│   │   └── depth_chart_corrections.json
│   └── raw/                        # Auto-fetched weekly stats (git-ignored)
├── requirements.txt
└── .env.example
```

---

## Scoring System

All player value calculations use **PPR scoring**:

| Stat | Points |
|---|---|
| Reception | 1.0 |
| Receiving yard | 0.1 |
| Rushing yard | 0.1 |
| Receiving TD | 6.0 |
| Rushing TD | 6.0 |
| Passing yard | 0.04 |
| Passing TD | 4.0 |
| Interception | −2.0 |

---

## Key Design Decisions

- **Multi-agent architecture** — each agent is single-purpose and stateless. Swapping or upgrading one agent doesn't affect the others.
- **No paid data sources** — everything uses free public APIs (Sleeper, ESPN scoreboard).
- **Bayesian blending** — early-season rankings are anchored to the prior year to prevent week-1 flukes from distorting recommendations.
- **Local-first** — all data is cached on disk. The app works offline once data is fetched.

---

## License

MIT License. Feel free to fork and adapt for your own league.
