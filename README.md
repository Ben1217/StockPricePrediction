# QuantVision

QuantVision is a full-stack stock analysis workspace built around a FastAPI backend and a React/Vite frontend. The project combines market data retrieval, technical analysis, forecasting, portfolio optimization, backtesting, exports, and optional agent workflows in a single repository.

## What the project does today

- Analysis dashboard with indicator overlays, market-session data, and rule-based sentiment; its chart-detail view and the Predictions tab both render the **official TradingView Advanced Chart widget** rather than redrawing candles. The app ships no candlestick engine of its own: TradingView owns price visualisation, the backend owns the analysis
- Forecasting endpoints and UI for `xgboost`, `random_forest`, and `lstm` models, with statistical fallback forecasts when trained artifacts are unavailable
- A **unified next-bar comparison** (`scripts/unified_benchmark.py`) that scores LSTM, XGBoost, Random Forest, the dynamic ensemble and the Kronos foundation model on identical walk-forward folds, reporting price accuracy and directional accuracy separately against the majority-class base rate — the experiment behind the question of whether a foundation model beats the existing stack
- A **combined-evidence direction call** (`/api/direction/{symbol}/analysis`) that reduces the bars to seven named categories — trend, momentum, volume, price action, support/resistance, volatility regime, and what followed the most similar setups in the symbol's own past — fits a logistic stack on them, and blends it with the classifier below by each one's *measured* out-of-sample Brier skill. Every category reports the percentage points it contributed, and a source with no measured skill gets weight zero, so the answer is `NEUTRAL` with a stated reason whenever nothing has demonstrated an edge
- A next-day **direction classifier** (`P(up tomorrow)` plus a price *range*) reading candlestick shape through 46 engineered chart-pattern features or through two foundation models (TabPFN, Kronos), with walk-forward evaluation, naive-baseline comparison, a leakage test, and a costed long/flat backtest — the one forecasting path in the repo that is gated on a measured out-of-sample result
- Multi-timeframe pattern detection, support/resistance analysis, and confluence ranking
- Portfolio optimization with `max_sharpe`, `min_volatility`, `max_return`, and `risk_parity`, plus efficient frontier, drift, alerts, sector, correlation, and Monte Carlo endpoints
- Backtesting with model-driven or technical signals, equity curves, trade logs, and CSV/PDF exports
- Watchlist, quote, and S&P 500 heatmap workflows in the frontend
- Optional CrewAI-based natural-language and multi-step analysis flows when an Anthropic API key is configured

## Stack

- Frontend: React 19, Vite 7, TradingView Advanced Chart widget (price), Recharts (analytics panels)
- Backend: FastAPI, Pydantic, Uvicorn
- Market data: yfinance primary, Wikipedia for S&P 500 constituents, Alpha Vantage optional fallback for selected endpoints
- ML and quant: scikit-learn, XGBoost, PyTorch, hmmlearn, Optuna, SHAP, cvxpy
- Storage: SQLite by default, PostgreSQL/TimescaleDB optional

## Repository layout

- `quantvision/`: React frontend
- `src/api/`: FastAPI app, routers, and schemas
- `src/data/`: market-data acquisition, storage, caching, and live quote helpers
- `src/features/`: indicators, feature engineering, support/resistance, pattern detection, price-action structure (`price_action.py`), and historical analog matching (`historical_analogs.py`)
- `src/models/`: model implementations, trainer, ensemble, regime detection, explainability, and the combined-evidence direction engine (`direction_evidence.py`)
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

### Automatic model preparation

Selecting a ticker in the UI is the whole action. When a serving route finds
nothing to serve for a symbol, it starts the training in the background and the
response carries the job to poll; the tab shows the stages and refreshes itself
when they finish. Nobody clicks "train", and no user is asked to run a Python
command to fill a gap the server can fill itself.

```
GET  /api/models/NVDA              what NVDA can serve, and why not
POST /api/models/NVDA/prepare      train what is missing (idempotent)
GET  /api/models/prepare/{job_id}  poll one job
```

Four things keep an auto-triggered pipeline from overwhelming its own server:
one job per symbol, a bounded worker pool (`QUANTVISION_PREPARE_WORKERS`), a
post-attempt cooldown (`QUANTVISION_PREPARE_COOLDOWN_SECONDS`), and a kill switch
(`QUANTVISION_AUTO_PREPARE=false`). Artifacts older than
`QUANTVISION_MODEL_MAX_AGE_DAYS` are refreshed in the background and keep serving
while that happens.

**Preparation does not retrain a bundle that failed its skill gate.** A model
that trained cleanly and then lost to a constant forecast out-of-sample has been
measured, not skipped, and refitting the same history reproduces the same
verdict. Those symbols report the measurement instead of offering a training
button — which is also why the API can safely start training on its own: there is
no state it can get stuck retrying.

### Evaluate the next-day direction classifier

The API runs this for you when a symbol has no evaluation yet. The CLI is for
running one deliberately — a different model, a longer history, more folds:

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

### Who draws what: TradingView vs the backend

The split is deliberate and holds everywhere in the app.

```
                        NVDA
                          │
          ┌───────────────┴────────────────┐
          ▼                                ▼
    TradingView                       Our backend
    Advanced Chart                    /api/direction/NVDA/analysis
          │                                │
    candles, volume,             historical data → indicators →
    timeframes, studies,         price action → pattern detection →
    drawing tools, replay        ML models → probability → direction
                                            │
                                            ▼
                                   AI Direction Analysis panel
```

The frontend renders the official TradingView embed (`quantvision/src/components/TradingViewChart.jsx`)
and points it at whichever ticker is selected — on the Predictions tab above the analysis
panel, and in the Analysis tab's "Chart Details" view; `quantvision/src/utils/tradingview.js`
translates our Yahoo-style symbols (`^GSPC`, `BRK-B`, `BTC-USD`) into TradingView's
namespace (`SP:SPX`, `BRK.B`, `CRYPTO:BTCUSD`). Nothing in the frontend computes a
number about the market, and nothing in the backend draws a chart.

### The combined-evidence direction call

`GET /api/direction/{symbol}/analysis` answers *up, down, or neither* — and says
why. It reduces the bars to seven named categories, each scored in `[-1, 1]` and
oriented so positive means conventionally bullish:

| Category | Reads |
| --- | --- |
| `trend` | Price against its 20/50/200-day means, plus rolling trend slope |
| `momentum` | RSI, MACD, 5- and 20-day rate of change |
| `volume` | Up-volume share, OBV slope, volume/price confirmation |
| `price_action` | Swing structure over 10/20/60 bars, and closing strength |
| `support_resistance` | What happened *at* the levels: breaks, retests, rejections |
| `volatility` | The regime the move is happening in |
| `historical_analogs` | What followed the 40 most similar setups in this symbol's own past |

The scores are descriptive and hand-oriented; **the weight each one carries is
fitted**, by a logistic stack trained on this symbol's own realised next-day
moves (`src/models/direction_evidence.py`). So a name whose breakouts have
historically failed gets a negative coefficient rather than the positive one a
hand-tuned weight would have assumed. Each category reports its contribution in
percentage points, computed as the probability with that term minus the
probability without it — the actual decomposition of the answer, not an
attribution assembled afterwards.

Price action is read as a chartist would read it (`src/features/price_action.py`):
a swing high over `w` bars against the swing high `w` bars before it gives the
four cases directly — higher high with higher low is `+1`, lower high with lower
low is `-1`, and the two mixed cases are `0`, because an expanding or contracting
range is not a direction. Breakouts, breakdowns, retests that held, failed breaks
and consolidation are detected on top of that and reported by name.

Historical analogs (`src/features/historical_analogs.py`) are a k-nearest-neighbour
search over a ten-column setup descriptor, z-scored against a *trailing* window so
a 2019 row is never scaled by a mean that includes 2024, and restricted to
neighbours whose outcome had already printed. The up-rate ships with its Wilson
interval and the unconditional rate over the same window, because 58% is skill
against a 50% base rate and noise against a 57% one.

#### Two probabilities, blended by measured skill

The evidence stack runs its own five-fold expanding walk-forward and the
classifier has its stored one. The two live probabilities are combined in
log-odds, weighted by **each one's measured out-of-sample Brier skill** — not by
a chosen ratio:

```
logit(P) = (w_evidence · logit(P_evidence) + w_classifier · logit(P_classifier))
           / (w_evidence + w_classifier)          w = max(brier_skill_score, 0)
```

A source that scored no better than always predicting the base rate gets weight
zero. When both do, the blend does not pick whichever number looks more
interesting — it returns the base rate and reports `NEUTRAL`.

#### The right to be uncertain

`direction` is `NEUTRAL`, with `neutral_reason` filled in, whenever either:

- no source has a proven edge — the stack's accuracy interval still covers its
  base rate *and* the classifier failed its ship criteria; or
- today's conviction sits in the bottom third of the stack's own out-of-sample
  `|p − 0.5|` distribution.

Confidence (`Low` / `Moderate` / `High`) is read off those same terciles, so a
model whose probabilities never leave 0.48–0.52 can never report `High`, and it
is downgraded a step when the short, medium and long-horizon reads conflict.

On most single names the honest answer is `NEUTRAL`, for the same reason the
walk-forward verdict is usually "do not ship". That is a result, not an empty
panel.

#### What is deliberately not drawn

No forecast path. The multi-day regression models emit one number for the whole
horizon, so the Predictions tab draws the endpoint and its calibrated 68%/95%
intervals on a price axis rather than a line through days nobody predicted, and
Monte Carlo scenarios appear as a strip of *endpoints* rather than 200 fictional
price histories. The direction analysis likewise reports a range — the
10th-to-90th percentile of what followed the matched historical setups — with the
sample size beside it, and makes no claim about the route.

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

### The unified model comparison: can Kronos beat what is already here?

```bash
python scripts/unified_benchmark.py --symbols AAPL --n-splits 3        # the experiment
python scripts/unified_benchmark.py --quick                            # fast smoke test
python scripts/unified_benchmark.py --models unified_xgboost,unified_kronos
python scripts/unified_benchmark.py --interval 15m --test-size 200     # intraday
```

The timeframe is a flag, not a constant. `--interval` sets the bar size and the
forecast is always the next bar, so daily bars predict the next trading day and
15-minute bars predict the next 15-minute candle. Nothing downstream is
daily-specific: the label is `Close.shift(-horizon)` and the indicators are
window counts, both bar-agnostic. (yfinance caps intraday history, so the
lookback is clamped per interval.)

The research question this repo now answers with evidence:

> Can a modern financial time-series foundation model, particularly Kronos,
> improve next-timeframe price and directional prediction over the existing
> LSTM, XGBoost, Random Forest and ensemble models?

Every model behind `src/models/unified_models.py` answers the same two questions
about the next bar -- a price in dollars, and `P(up)` / `P(down)` -- so they can
be scored against each other rather than described separately:

| Model | What it is | Sees |
|---|---|---|
| `unified_xgboost` | Gradient-boosted trees, the tuned tabular baseline | 46 engineered features |
| `unified_random_forest` | Bagged trees | 46 engineered features |
| `unified_lstm` | Recurrent net over a 60-bar lookback window | 60 x 46 sequence |
| `unified_ensemble` | The existing dynamic ensemble of the three above | as its members |
| `unified_kronos` | Candlestick foundation model, zero-shot | raw OHLCV candles |
| `unified_timesfm` | TimesFM 2.5, optional, zero-shot | close series |
| `unified_chronos` | Chronos-2, optional, zero-shot | close series |

TimesFM and Chronos are optional (`pip install -e ".[comparison]"`). When their
packages are absent the benchmark reports the comparison without them rather
than failing, and the API answers "not installed" rather than 500.

**How the comparison is kept fair.** Every model gets the same rows, the same
expanding-window folds, the same one-bar embargo between train and test, and the
same metric code. Feature scaling is fitted on each fold's training rows alone.
The LSTM is handed real 60-bar sequences rather than a flattened row, so the
"deep learning" entry in the table is actually a sequence model. The ensemble's
members are built from the same `DEFAULT_MODEL_PARAMS` as their standalone runs,
so its row is a baseline rather than a fourth differently-tuned model. Its
weights come from a chronological hold-out inside the training window, softmaxed
the way the existing `EnsemblePredictor` softmaxes rolling Sharpe.

**Two objectives, reported apart.** Price gets MAE / RMSE / MAPE / R-squared;
direction gets accuracy, precision, recall, F1, ROC-AUC and Brier. Sharpe ratio
is deliberately not used to rank forecasters -- it scores a *strategy*, and
belongs to the backtesting layer.

**Two columns worth understanding before reading the table:**

- `price_r2` is computed against the realised price, whose variance is dominated
  by the price level. A model that predicts "roughly today's close" scores near
  1.0 on it. `price_r2_return` recomputes it on the forward return, where that
  same do-nothing forecast scores 0.0. The second column is the one that
  separates models.
- `edge_pp` is directional accuracy minus the majority-class rate on the same
  test windows. At or below zero, the model adds nothing over always guessing
  the more common direction, whatever its raw accuracy looks like. It comes with
  a one-sided `p_value` and `n_required` -- the number of test days an edge that
  size would need before it could be called significant at 5%. On a few hundred
  test days, a two-point edge is noise, and the table says so rather than
  leaving the reader to assume otherwise.

Results land in `artifacts/benchmark_results.csv`, `benchmark_per_fold.csv`,
`benchmark_comparison.csv` and `benchmark_results.json`.

**Serving.** The winner is servable directly: `POST /api/predict` with
`model_type=unified_kronos` (or any `unified_*`) returns the next-bar price,
`probability_up`, `probability_down` and the direction. The trainable ones are
trained via `POST /api/training/train`; asking to train a zero-shot foundation
model returns 400 rather than a job that quietly does nothing. In the dashboard
they appear in the Predictions tab model selector, where the horizon buttons
lock to one bar because that is what these models forecast.

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
| `/api/direction/{symbol}/analysis` | The reasoned direction call: `UP`/`DOWN`/`NEUTRAL`, probability split, confidence, per-category evidence contributions, three horizon reads, price-action structure, historical analogs, and the expected range |
| `/api/models` | Model readiness per symbol, and the automatic preparation jobs that close the gaps |
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
| Analysis | Price action, indicators, session quotes, rule-based sentiment, and the model output from the same pipeline Predictions serves. "Chart Details" opens the real TradingView chart beside our pattern and indicator panels |
| Predictions | The live TradingView chart, then the AI direction analysis (direction, probability, confidence, per-category evidence) and the classifier's own track record beneath it; the multi-day forecast drawn as a *range* at the horizon rather than a path to it |
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
