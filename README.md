# FIFA Bracket Intelligence Bot

A production-ready Python application that sends a **daily email at 9:00 AM America/Los_Angeles** containing FIFA World Cup match summaries, bracket scoring updates, and AI-powered insights based on your bracket predictions.

---

## Features

- 📊 **Daily bracket scoring** with configurable point weights
- ⚽ **Yesterday's results** vs. your predictions
- 🎯 **"Who Should I Root For?" guide** with Elo-based win probabilities
- 🤖 **OpenAI-powered analysis** (executive summary, threats, outlook)
- 📧 **Email delivery** via Gmail SMTP or Gmail API (OAuth2)
- 🔄 **Multiple data providers** (API-Football, FotMob, ESPN) with fallback
- 🚀 **GitHub Actions** — runs automatically every day

---

## Project Structure

```
fifa-bracket-bot/
├── app/
│   ├── main.py              # Entry point orchestrator
│   ├── config.py            # Configuration & env vars
│   ├── fifa_api.py          # Match data provider abstraction
│   ├── bracket.py           # Bracket prediction management
│   ├── scoring.py           # Configurable scoring engine
│   ├── probabilities.py     # Probability & bracket health engine
│   ├── report_generator.py  # Daily analysis + LLM integration
│   ├── email_sender.py      # Gmail SMTP & API delivery
│   └── models.py            # Data models
├── data/
│   └── bracket.json         # Your bracket predictions
├── templates/
│   └── email_template.html  # Responsive HTML email template
├── tests/                   # Test suite (>90% coverage target)
├── requirements.txt
└── .github/workflows/
    └── daily-report.yml     # GitHub Actions workflow
```

---

## Quick Start

### 1. Clone and Install

```bash
cd fifa-bracket-bot
pip install -r requirements.txt
```

### 2. Edit Your Bracket

Open `data/bracket.json` and fill in your predictions:

```json
{
  "winner": "Argentina",
  "runner_up": "France",
  "semifinals": ["Argentina", "Brazil", "France", "Spain"],
  "quarterfinals": ["Argentina", "Netherlands", "Brazil", "England",
                    "France", "Portugal", "Spain", "Germany"],
  "match_predictions": {
    "R16_1": "Argentina",
    "R16_2": "Brazil",
    "QF_1": "Argentina",
    "SF_1": "Argentina",
    "F_1": "Argentina"
  }
}
```

### 3. Configure Environment Variables

Copy the template and fill in your values:

| Variable | Required | Description |
|---|---|---|
| `EMAIL_ADDRESS` | ✅ | Your Gmail address |
| `EMAIL_RECIPIENT` | ✅ | Where to send the report |
| `EMAIL_PROVIDER` | ✅ | `gmail_smtp` or `gmail_api` |
| `EMAIL_PASSWORD` | Gmail SMTP only | Gmail App Password |
| `GMAIL_CLIENT_ID` | Gmail API only | OAuth2 client ID |
| `GMAIL_CLIENT_SECRET` | Gmail API only | OAuth2 client secret |
| `GMAIL_REFRESH_TOKEN` | Gmail API only | OAuth2 refresh token |
| `OPENAI_API_KEY` | Optional | Enables AI analysis |
| `OPENAI_MODEL` | Optional | Default: `gpt-4o-mini` |
| `API_FOOTBALL_KEY` | Optional | api-football.com API key |
| `DATA_PROVIDERS` | Optional | Comma-separated list: `api_football,fotmob,espn` |
| `DRY_RUN` | Optional | `true` = log but don't send email |
| `LOG_LEVEL` | Optional | `DEBUG`, `INFO`, `WARNING` |

### 4. Run Locally

```bash
cd fifa-bracket-bot
export EMAIL_ADDRESS="you@gmail.com"
export EMAIL_RECIPIENT="you@gmail.com"
export DRY_RUN=true
python -m app.main
```

### 5. Run Tests

```bash
cd fifa-bracket-bot
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## GitHub Actions Setup

### Add Repository Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|---|---|
| `EMAIL_ADDRESS` | Your Gmail address |
| `EMAIL_RECIPIENT` | Recipient email |
| `EMAIL_PROVIDER` | `gmail_smtp` or `gmail_api` |
| `EMAIL_PASSWORD` | Gmail App Password (for SMTP) |
| `GMAIL_CLIENT_ID` | OAuth2 Client ID (for API) |
| `GMAIL_CLIENT_SECRET` | OAuth2 Client Secret (for API) |
| `GMAIL_REFRESH_TOKEN` | OAuth2 Refresh Token (for API) |
| `OPENAI_API_KEY` | OpenAI API key |
| `API_FOOTBALL_KEY` | API-Football key (optional) |

### Schedule

The workflow runs daily at **17:00 UTC = 9:00 AM PST**. During PDT (summer),
update the cron to `0 16 * * *` for 9:00 AM PDT.

### Manual Trigger

Go to **Actions → FIFA Bracket Daily Report → Run workflow** to trigger manually.
You can set `dry_run=true` to test without sending an email.

---

## Scoring Weights (Configurable)

| Round | Points |
|---|---|
| Group Stage | 1 |
| Round of 16 | 2 |
| Quarterfinal | 4 |
| Semifinal | 8 |
| Final | 16 |
| Champion | 32 |

---

## Data Providers

Providers are tried in priority order and fall back automatically:

1. **API-Football** (`api-football.com`) — requires `API_FOOTBALL_KEY`
2. **FotMob** — unofficial public API, no key required
3. **ESPN** — public API, no key required (default fallback)

---

## Gmail App Password Setup (SMTP)

1. Enable 2-Factor Authentication on your Google account
2. Go to **Security → App passwords**
3. Generate a password for "Mail"
4. Use this as `EMAIL_PASSWORD`

---

## Gmail API (OAuth2) Setup

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable the Gmail API
3. Create OAuth2 credentials (Desktop app)
4. Use the OAuth Playground to get a refresh token with `gmail.send` scope
5. Set `GMAIL_CLIENT_ID`, `GMAIL_CLIENT_SECRET`, `GMAIL_REFRESH_TOKEN`
