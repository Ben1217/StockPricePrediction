# QuantVision

QuantVision is a full-stack stock analysis workspace built around a FastAPI backend and a React/Vite frontend. The project combines market data retrieval, technical analysis, forecasting, portfolio optimization, backtesting, exports, and optional agent workflows in a single repository.

## What the project does today

- Analysis dashboard with OHLCV charts, indicator overlays, market-session data, and rule-based sentiment
- Forecasting endpoints and UI for `xgboost`, `random_forest`, and `lstm` models, with statistical fallback forecasts when trained artifacts are unavailable
- Multi-timeframe pattern detection, support/resistance analysis, and confluence ranking
- Portfolio optimization with `max_sharpe`, `min_volatility`, `max_return`, and `risk_parity`, plus efficient frontier, drift, alerts, sector, correlation, and Monte Carlo endpoints
- Backtesting with model-driven or technical signals, equity curves, trade logs, and CSV/PDF exports
- Watchlist, quote, and S&P 500 heatmap workflows in the frontend
- Optional CrewAI-based natural-language and multi-step analysis flows when an Anthropic API key is configured

## Stack

- Frontend: React 19, Vite 7, lightweight-charts 5, Recharts
- Backend: FastAPI, Pydantic, Uvicorn
- Market data: yfinance primary, Wikipedia for S&P 500 constituents, Alpha Vantage optional fallback for selected endpoints
- ML and quant: scikit-learn, XGBoost, PyTorch, hmmlearn, Optuna, SHAP, cvxpy
- Storage: SQLite by default, PostgreSQL/TimescaleDB optional

## Repository layout

- `quantvision/`: React frontend
- `src/api/`: FastAPI app, routers, and schemas
- `src/data/`: market-data acquisition, storage, caching, and live quote helpers
- `src/features/`: indicators, feature engineering, support/resistance, and pattern detection
- `src/models/`: model implementations, trainer, ensemble, regime detection, explainability
- `src/portfolio/`: optimization, metrics, risk controls, sector allocation, and drift tracking
- `src/backtesting/`: backtest engine
- `src/agents/`: CrewAI agents and tool wrappers
- `config/`: YAML configuration
- `scripts/`: utility scripts for downloads, training, and migration
- `models/`: trained artifacts and metadata
- `tests/`: unit and integration tests
- `docker/`: API/frontend container definitions

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+ and npm

### 1. Create a Python environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you also want notebook and development tooling, use:

```bash
pip install -r requirements_working.txt
```

### 2. Install the frontend

```bash
cd quantvision
npm install
cd ..
```

### 3. Configure environment variables

```bash
copy .env.example .env
```

Most features work without extra keys. Optional values are used for:

- `ALPHA_VANTAGE_API_KEY`: Alpha Vantage live/history fallback
- `ANTHROPIC_API_KEY`: CrewAI agent workflows
- `POSTGRES_*`: PostgreSQL and TimescaleDB utilities
- `QUANTVISION_API_URL`: base URL used by agent tool wrappers

### 4. Start the backend

```bash
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at:

- `http://localhost:8000/`
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

### 5. Start the frontend

```bash
cd quantvision
npm run dev
```

The UI runs at `http://localhost:5173`.

## Common workflows

### Run tests

```bash
pytest
```

### Download historical data

```bash
python scripts/download_daily_data.py
```

### Train local model artifacts

```bash
python scripts/train_models.py
```

Trained artifacts are stored under `models/saved_models/`, and metadata is stored under `models/model_metadata/`.

### Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
```

This starts:

- FastAPI on port `8000`
- Frontend on port `80`
- TimescaleDB on port `5432`

## API surface

| Route prefix | Purpose |
| --- | --- |
| `/health` | Basic health check |
| `/api/data` | Price history, indicators, sources, quotes, uploads, S&P 500 list |
| `/api/predict` | Forecasts and historical model signals |
| `/api/training` | Background training jobs and saved model metadata |
| `/api/patterns` | Multi-timeframe patterns, support/resistance, confluence |
| `/api/sentiment` | Rule-based indicator sentiment |
| `/api/portfolio` | Optimization, efficient frontier, metrics, drift, alerts, simulation |
| `/api/backtest` | Backtest runs and stored results |
| `/api/export` | CSV and PDF exports |
| `/api/agent` | Optional CrewAI-backed analysis and Q&A |

## Frontend tabs

| Tab | Current focus |
| --- | --- |
| Analysis | Price action, indicators, session quotes, rule-based sentiment |
| Predictions | Forecast paths and confidence bands |
| Portfolio | Holdings view and allocation snapshot |
| Backtest | Configurable historical strategy runs |
| Optimization | Portfolio optimizer and efficient frontier |
| Heatmap | S&P 500-style market map with live refresh attempts |

## Data and model notes

- yfinance is the default data source across the app
- Alpha Vantage is optional and only used when a key is present and the selected endpoint supports it
- The heatmap uses a seeded sector/company dataset in the frontend and can refresh quote fields from the backend
- The LSTM implementation in this repository uses PyTorch, not TensorFlow
- Agent workflows are optional and rely on the backend being reachable plus a valid Anthropic key

## Status

This repository is currently structured as an academic/research project for experimentation and learning. It is not investment advice and it does not ship a separate open-source license file.
