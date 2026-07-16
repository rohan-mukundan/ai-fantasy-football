# Setup Guide — AI Fantasy Football

Follow these steps in order. This should take about 15–20 minutes.

---

## Step 1: Install Python

1. Go to **https://www.python.org/downloads/**
2. Click the big yellow "Download Python 3.12.x" button
3. Run the installer
   - **Windows:** Check the box that says **"Add Python to PATH"** before clicking Install (this is easy to miss!)
   - **Mac:** Follow the prompts, defaults are fine
4. Verify it worked — open Terminal (Mac) or Command Prompt (Windows) and type:
   ```
   python --version
   ```
   You should see something like `Python 3.12.3`. If you see an error, restart your terminal and try again.

---

## Step 2: Open a Terminal in the Project Folder

Your project files are already on your computer at:
```
/Users/rohanmukundan/Documents/Claude/Projects/AI Fantasy Football
```

You just need to open a terminal window pointed at that folder:

- **Mac:** Open Finder, navigate to that folder, then right-click it → "New Terminal at Folder"
- **Windows:** Open File Explorer, navigate to that folder, hold Shift and right-click it → "Open PowerShell window here"

All commands in the steps below should be run inside that terminal.

---

## Step 3: Create a Virtual Environment

A virtual environment keeps this project's packages separate from everything else on your computer. Think of it as a clean room just for this project.

```bash
# Create the virtual environment (only do this once)
python -m venv venv
```

Now activate it (you need to do this every time you open a new terminal):

**Mac/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You'll know it's activated because your terminal prompt will show `(venv)` at the start.

---

## Step 4: Install Dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs all the Python packages the project needs (requests, pandas, anthropic, etc.). It may take a minute.

---

## Step 5: Set Up Your Anthropic API Key

1. Go to **https://console.anthropic.com/** and sign up (or log in)
2. Navigate to **API Keys** and create a new key
3. Copy the key (it starts with `sk-ant-...`)
4. In the project folder, create a file called `.env` (copy from `.env.example`):

**Mac/Linux:**
```bash
cp .env.example .env
```

**Windows:**
```bash
copy .env.example .env
```

5. Open `.env` in a text editor and replace `your_anthropic_api_key_here` with your actual key:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

> ⚠️ Never share your `.env` file or upload it to GitHub. It contains your private API key.

---

## Step 6: Pull 2024 NFL Data

Run the data pipeline to download historical 2024 season data:

```bash
python data_pipeline/sleeper_client.py
```

You should see output like:
```
============================================================
  AI Fantasy Football — Sleeper Data Pipeline
  Season: 2024 | Type: regular
============================================================

📥 Fetching all NFL players from Sleeper...
  ✓ Saved 2,847 players to data/raw/all_players.json
  ✓ Saved 1,203 fantasy-relevant players to data/processed/players.csv

  Players fetched: 1,203
  Positions: {'WR': 512, 'RB': 387, 'QB': 168, 'TE': 136}

📥 Fetching weekly stats for all 18 weeks of 2024 season...
Fetching weeks: 100%|████████████| 18/18 [00:04<00:00,  4.2it/s]
  ✓ Combined 42,318 player-week rows

🔗 Merging player info with weekly stats...
  ✓ Saved 18,245 rows to data/processed/season_2024.csv

📊 Top 10 scoring players in Week 1 (PPR):
         full_name position team  pts_ppr
    Lamar Jackson       QB  BAL    42.12
         ...

✅ Data pipeline complete!
```

The data is now saved locally — you won't need to re-fetch it unless you want to refresh it.

---

## Step 7: Explore the Data (Optional but Recommended)

Open a Jupyter notebook to explore what you just downloaded:

```bash
jupyter notebook
```

This opens a browser window. Navigate to `notebooks/` and open `exploration.ipynb`.

---

## Troubleshooting

**"python is not recognized" (Windows):**
Python wasn't added to PATH during installation. Uninstall and reinstall, making sure to check "Add Python to PATH".

**"pip: command not found":**
Try `pip3` instead of `pip`.

**"ModuleNotFoundError: No module named 'requests'":**
Your virtual environment isn't activated. Run `source venv/bin/activate` (Mac) or `venv\Scripts\activate` (Windows).

**API key error when running agents:**
Make sure your `.env` file exists (not just `.env.example`) and has the real key in it.

**No data returned from Sleeper API:**
Check your internet connection. The Sleeper API is free and requires no key.

---

## You're Ready!

Once Step 6 completes successfully, you have:
- ✅ Python environment set up
- ✅ All packages installed
- ✅ 2024 NFL season data saved locally
- ✅ Project structure in place

**Next step:** Read `ARCHITECTURE.md` to understand how the agents work, then we'll build the Draft Agent.
