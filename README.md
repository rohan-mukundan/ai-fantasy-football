# AI Fantasy Football Assistant

An AI-powered fantasy football manager built with Python, Streamlit, and Claude (Anthropic). Connect your Sleeper fantasy league and get data-driven recommendations for drafting, weekly lineup decisions, waiver wire pickups, trade analysis, and defensive streaming — all explained in plain English by an AI agent.

---

## What It Does

| Feature | Description |
|---|---|
| **AI Draft Assistant** | Ranks players using 3 years of weighted PPR stats, blended with in-season performance. Claude explains each pick. |
| **Weekly Lineup Optimizer** | Sets your optimal starting lineup each week with matchup-aware start/sit advice. |
| **Waiver Wire Recommendations** | Surfaces the best available upgrades at each position based on projected output. |
| **Trade Analyzer** | Evaluates multi-player trade packages considering team needs and player value. Claude pitches the trade to your opponent. |
| **DEF Streaming** | Each week, recommends the best available free-agent defense based on opponent offense strength (75%) and DEF unit quality (25%). |
| **Defensive Matchup Grades** | Every player in your lineup shows a 🟢/🟡/🔴 matchup grade based on how many fantasy points their upcoming opponent's defense has allowed this season. |

---

## How It Works

```
Sleeper API  ──►  data_pipeline/  ──►  app.py (Streamlit UI)
                  (stats, rosters,           │
                   schedules,                │
                   defense rankings)    Claude API
                                            │
                                       agents/
                                       (draft, lineup,
                                        trade, strategy)
```

- **Data** is pulled from the free [Sleeper API](https://docs.sleeper.com/) and [ESPN's public scoreboard API](https://site.api.espn.com). No paid subscriptions required.
- **Rankings** use 3 seasons of weighted PPR averages (2022–2024) as a preseason baseline, then blend in current-season stats as the year progresses.
- **Defense rankings** compute PPR points allowed per position (QB/RB/WR/TE) per team each week, with prior-season blending to stabilize early-season noise.
- **Offense rankings** use expert preseason projections (The Ringer) for weeks 1–3, then switch to cumulative season averages for weeks 4–18.
- **AI reasoning** is handled by Claude (claude-sonnet-4-5) via the Anthropic API.

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
3. Done. You're ready to draft or manage your team.

The app caches all data locally in `data/` so subsequent loads are fast.

---

## Project Structure

```
├── app.py                          # Main Streamlit application
├── agents/
│   ├── draft_agent.py              # AI draft pick explanations
│   ├── lineup_agent.py             # AI weekly lineup rationale
│   ├── trade_agent.py              # AI trade pitch generator
│   └── strategy_agent.py           # General strategy advice
├── data_pipeline/
│   ├── sleeper_client.py           # Sleeper API wrapper
│   ├── data_processor.py           # Multi-year stats aggregation
│   ├── current_season.py           # Live in-season stat fetching
│   ├── defense_rankings.py         # PPR pts allowed + offense rankings
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

## Defense Streaming Logic

The DEF streaming system scores available free-agent defenses each week using a blended formula:

**Score = 75% matchup quality + 25% DEF unit quality**

- **Matchup quality**: how weak the upcoming opponent's offense is (rank 1 = weakest = best matchup)
- **DEF unit quality**: how many fantasy points the defense itself scores per game on average this season

**Preseason schedule (weeks 1–3)**: opponent offense strength uses expert preseason rankings from The Ringer rather than small-sample game data. Switches to cumulative season averages from week 4 onward.

---

## Key Design Decisions

- **No paid data sources** — everything uses free public APIs (Sleeper, ESPN scoreboard).
- **Bayesian blending** — early-season rankings are anchored to the prior year to prevent week-1 flukes from distorting recommendations.
- **Stateless AI agents** — Claude is called fresh each week with structured context. No memory between calls; all intelligence comes from the data pipeline.
- **Local-first** — all data is cached on disk. The app works offline once data is fetched.

---

## License

MIT License. Feel free to fork and adapt for your own league.
