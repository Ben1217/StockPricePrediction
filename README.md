# QuantVision

QuantVision is a full-stack stock analysis workspace built around a FastAPI backend and a React/Vite frontend. The project combines market data retrieval, technical analysis, forecasting, portfolio optimization, backtesting, exports, and optional agent workflows in a single repository.

## What the project does today

- Analysis dashboard with OHLCV charts, indicator overlays, market-session data, and rule-based sentiment
- Forecasting endpoints and UI for `xgboost`, `random_forest`, and `lstm` models, with statistical fallback forecasts when trained artifacts are unavailable
- A next-day **direction classifier** (`P(up tomorrow)` plus a price *range*) reading candlestick shape through 46 engineered chart-pattern features or through two foundation models (TabPFN, Kronos), with walk-forward evaluation, naive-baseline comparison, a leakage test, and a costed long/flat backtest — the one forecasting path in the repo that is gated on a measured out-of-sample result
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
pip install -e .
```

Dependencies are declared once, in `pyproject.toml`. Optional feature sets:

```bash
pip install -e ".[agents]"     # CrewAI/Claude agent routes (/api/agent/*)
pip install -e ".[postgres]"   # PostgreSQL/TimescaleDB storage backend
pip install -e ".[dev]"        # tests, linters, notebooks
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

### Evaluate the next-day direction classifier

```bash
python scripts/direction_backtest.py --ticker AAPL --start 2015-01-01 --folds 8 --cost-bps 10
make direction TICKER=MSFT MODEL=gradient_boosting COST_BPS=5   # same thing
```

This is a different question from the multi-day price forecast, and a more
answerable one. The 7/15/30/60-day models regress a forward *return*: each bundle
emits one scalar, and the path drawn through it is `P(t) = P(0) * (1 + r)^(t/H)`
— a monotone curve that cannot show a direction change, because there is only
one predicted number in it. The direction classifier predicts the **sign of
tomorrow's move** and is scored on whether it was right.

It ships only if it earns it. Every run reports:

- expanding-window walk-forward folds, fixed 63-row (one quarter) test windows,
  an embargo between train and test, and the threshold chosen on an inner
  validation block that never overlaps the scored rows
- accuracy **with a Wilson confidence interval**, balanced accuracy, per-class
  precision/recall, AUC, Brier, log loss, MCC, and a calibration curve
- four naive baselines scored on the identical folds — majority class,
  momentum-1, reversal-1, and base-rate random — plus a significance test on the
  gap, using the conservative `sqrt(0.25/n)` standard error
- a **shuffled-label leakage test**: labels are permuted inside each training
  window and the model refitted; accuracy must collapse to chance. If it does
  not, the run says every other number is void
- a long/flat backtest with signal at the close of *t*, entry at the open of
  *t+1*, exit at that close, costs charged as a round trip per active day, and
  the **breakeven cost** — the bps above which the edge is gone

Exit status is a gate: `0` when every ship criterion passed, `2` when the run was
clean but the model did not clear the bar, `1` on error. Artifacts land in
`data/direction_backtests/` (report JSON, equity-curve CSV, predictions CSV) and
are gitignored, since the report embeds a hash of the exact bars it saw.

A "do not ship" verdict is the expected outcome on most single names, and it is a
result rather than a bug: next-day direction at this signal-to-noise ratio is
mostly unlearnable from price history alone. The pipeline is built to say so
clearly instead of dressing a coin flip in a dashboard.

### Reading the chart: features and model slots

The direction pipeline asks what a technical analyst asks — *what shape is this
chart, and what usually followed shapes like it* — and answers it one day at a
time. Every prediction is a fresh model call on that day's bars. Nothing is
interpolated between two endpoints.

**Features (46 columns).** The 13 scale-free columns and 6 short-horizon
directional ones from `src/features/direction_features.py`, plus 27
chart-pattern columns from `src/features/chart_patterns.py`:

| Family | Examples | What it captures |
| --- | --- | --- |
| Candle geometry | `Body_Ratio`, `Upper_Shadow_Ratio`, `Lower_Shadow_Ratio`, `Body_To_ATR` | Where the close sits in the bar and how long the shadows are — a long lower shadow closing near the high is a rejection of lower prices |
| Multi-bar shape | `Body_Ratio_Lag1`, `Close_Position_Mean_5`, `Consecutive_Direction_Run`, `Range_Expansion_5_20` | Two-candle patterns, streaks, and whether the range is expanding or coiling |
| Breakout / channel | `High_20d_Break`, `Low_20d_Break`, `Donchian_Position_20/60`, `Bars_Since_High_20` | Did the close take out the prior 20-day high, and where does it sit in the channel |
| Trend cleanliness | `Trend_Slope_20/60`, `Trend_R2_20/60`, `Efficiency_Ratio_10/20` | How straight the trend line is — a 20-day advance at R² 0.95 is a different chart from the same advance through chop |
| Volume confirmation | `Up_Volume_Ratio_20`, `Volume_Price_Confirm`, `OBV_Slope_20` | Whether the move was participated in or drifted |
| Volatility texture | `ATR_Ratio`, `Parkinson_Vol_Ratio` | Intraday range versus close-to-close: a market that moves and closes flat is not the same as one that gaps |

`High_20d_Break` compares the close against the **prior** 20 bars (`.shift(1)`).
Including today's own high would both leak and invert the signal, and a unit
test pins it.

**Model slots.** All share one interface and one evaluation harness, so swapping
the model changes nothing about how it is judged:

| Slot | Eats | Emits |
| --- | --- | --- |
| `logistic` | 46 features | P(up). The reference every other slot must beat |
| `gradient_boosting` | 46 features | P(up). Kept as the "does a foundation model beat GBM here" ablation |
| `tabpfn` | 46 features | P(up), from a tabular foundation model whose forward pass *is* the fit |
| `kronos` | raw OHLCV | P(up) **and** a price band, from sampled future candles |
| `foundation_ensemble` | both | Mean probability, with Kronos' band |

The LSTM is deliberately absent: it was gated on the simpler models clearing the
baselines first, and they did not.

**Price bands.** Every slot reports a 5th/50th/95th-percentile range for
tomorrow's close, never a bare point. Kronos gets it from the percentiles of its
own sampled candles. The tabular slots get it from
`src/models/direction_bands.py`, which fits the empirical quantiles of the
volatility-standardised return (`z = r / sigma_20d`) on the training fold,
bucketed by the model's own probability. The width comes from today's
volatility, so it is genuinely per-day; the skew comes from the model's measured
conviction. Coverage is reported out-of-sample — a band claiming 90% that covers
60% is decoration, and the report says so.

### Foundation models: Kronos and TabPFN

```bash
pip install -e ".[foundation]"      # tabpfn, einops, huggingface_hub
python scripts/setup_kronos.py      # clones Kronos (MIT) into vendor/kronos
python scripts/direction_backtest.py --ticker AAPL --model kronos --kronos-samples 128
```

**Kronos** is a decoder-only transformer pre-trained on ~12 billion K-lines from
45 exchanges; its tokeniser quantises candles into discrete tokens, which is a
learned chart-pattern vocabulary rather than a hand-specified one. Sampling it
repeatedly gives both outputs at once: `P(up)` is the share of sampled closes
above today's, and the band is their percentiles.

Two things worth knowing before running it:

- **Per-sample paths.** `KronosPredictor.predict()` averages over `sample_count`
  internally, which destroys the spread. This wrapper instead calls the public
  `generate()` with the batch replicated N times and `sample_count=1`, so each
  batch row is an independent draw. Verified against the library, and a unit
  test drives it with a fake predictor to prove each date gets its own block of
  samples.
- **Cost.** It is a transformer forward pass per test row. Measured on 6 CPU
  threads: lookback 128 / 128 samples is ~12.8 s per row, so a 504-row
  walk-forward runs close to two hours. On CUDA it is minutes. `--kronos-samples`
  and `--kronos-lookback` trade accuracy for time; attention is quadratic in
  lookback, and the Monte-Carlo error on `P(up)` is `sqrt(0.25/N)` — about 4.4pp
  at N=128, which the report records so it is not mistaken for signal.

**TabPFN** is a prior-data-fitted network: pre-trained on millions of synthetic
tasks so its forward pass *is* Bayesian inference over the training set handed to
it as context. A ~2 500 x 46 dataset sits inside the regime where its authors
showed it matching tuned gradient-boosted trees, so it is a fair contender here
rather than a novelty.

Novelty is not accuracy. Both still have to clear momentum-1 and reversal-1 on
walk-forward folds after costs, on the same harness, or the report says do not
ship.

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
| `/api/direction` | Next-day direction: walk-forward evaluation, gated `P(up)` gauge, rolling hit rate, equity curve |
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
| Predictions | Next-day direction panel (gauge, rolling hit rate, equity vs buy & hold); multi-day forecast paths and confidence bands |
| Portfolio | Holdings view and allocation snapshot |
| Backtest | Configurable historical strategy runs |
| Optimization | Portfolio optimizer and efficient frontier |
| Heatmap | S&P 500-style market map with live refresh attempts |

## Data and model notes

- yfinance is the default data source across the app
- Alpha Vantage is optional and only used when a key is present and the selected endpoint supports it
- The heatmap uses a seeded sector/company dataset in the frontend and can refresh quote fields from the backend
- Direction datasets pull unadjusted bars with `auto_adjust=False, actions=True` and then put every price column on the `Adj Close` basis, so `High`/`Low`/`Open` and the close can never be mixed across bases on a dividend date
- The LSTM implementation in this repository uses PyTorch, not TensorFlow
- Agent workflows are optional and rely on the backend being reachable plus a valid Anthropic key

## Status

This repository is currently structured as an academic/research project for experimentation and learning. It is not investment advice and it does not ship a separate open-source license file.
