# API and Environment Setup Guide

This project no longer uses the older Streamlit dashboard flow. The current runtime is:

- FastAPI backend on `http://localhost:8000`
- React/Vite frontend on `http://localhost:5173`
- Optional CrewAI agents layered on top of the backend

## Quick summary

| Service | Required | Current use | Notes |
| --- | --- | --- | --- |
| yfinance | No key | Core market data source | Used across price, indicator, prediction, backtest, and portfolio flows |
| Wikipedia | No key | S&P 500 constituent lookup | Used by market-data helpers |
| Alpha Vantage | Optional key | Selected live/history fallback endpoints | Only enabled when `ALPHA_VANTAGE_API_KEY` is set |
| Anthropic | Optional key | CrewAI agent workflows | Needed for `/api/agent` and local agent helpers |
| PostgreSQL / TimescaleDB | Optional | Storage and migration utilities | Only needed if you use the database utilities |

## Environment variables that matter

### Core local development

```env
APP_NAME=QuantVision
APP_VERSION=2.0.0
ENVIRONMENT=development
DEBUG=True
API_HOST=127.0.0.1
API_PORT=8000
FRONTEND_PORT=5173
```

### Data sources

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
DATA_START_DATE=2019-01-01
DATA_END_DATE=2024-12-31
ENABLE_SP500=True
ENABLE_RUSSELL2000=False
```

### Optional agent support

```env
ANTHROPIC_API_KEY=your_anthropic_api_key
QUANTVISION_API_URL=http://localhost:8000
```

### Optional database support

```env
DB_TYPE=sqlite
SQLITE_DB_PATH=database/stock_data.db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=stock_data
```

## Recommended local setup

### 1. Install backend dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd quantvision
npm install
cd ..
```

### 3. Create `.env`

```bash
copy .env.example .env
```

### 4. Start the backend

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

### 5. Start the frontend

```bash
cd quantvision
npm run dev
```

## Verification

### Backend health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

### Source configuration check

```bash
python tests/test_api_keys.py
```

That script verifies:

- yfinance access
- Wikipedia constituent scraping
- optional Alpha Vantage key presence and validity
- optional Anthropic key presence

## What is no longer part of the setup story

- No Streamlit app startup command
- No Dash frontend
- No `requirements-dev.txt`
- No `update_data.py`, `train_model.py`, or `train_all_models.py` workflow
- No Polygon or IEX setup path in the current codebase

## Notes

- Most of the project works without any API key.
- Alpha Vantage is only needed if you want the optional fallback/live endpoints that reference it.
- CrewAI agents are optional; the main UI and API do not require Anthropic credentials.
- The repository still includes SQLite-to-PostgreSQL migration utilities and Docker support for TimescaleDB, but they are optional.
