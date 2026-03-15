# 📈 QuantVision — Stock Price Prediction & Portfolio Optimization

A full-stack machine learning platform for stock market analysis, built with a **React 19** dashboard and a **FastAPI** backend. The system combines state-of-the-art deep learning models with classical quantitative finance techniques to deliver:

- **Real-time price predictions** using LSTM, XGBoost, Random Forest, and ensemble models
- **Candlestick pattern detection** with user-selectable overlays (max 3 concurrent)
- **Automated portfolio optimization** with mean-variance, risk parity, and Top-K strategies
- **Comprehensive backtesting** with realistic transaction costs and Monte Carlo simulations
- **Interactive TradingView-style charts** (lightweight-charts v5) with ML signal overlays
- **Extended market hours** support (pre-market, regular, post-market)
- **Dynamic watchlist & S&P 500 screener** with free-text stock search

---

## 🎓 Academic Context

Developed as part of a **Bachelor of Computer Science (Honours)** thesis at **Universiti Tunku Abdul Rahman (UTAR)**, focusing on the intersection of machine learning and computational finance.

### Research Focus Areas
- Stock price forecasting using deep learning (LSTM networks)
- Technical indicator feature engineering and dimensionality reduction
- ML-driven portfolio construction and risk-adjusted return optimization
- Real-time decision support systems for trading

---

## ✨ Key Features

### 🤖 Machine Learning & Prediction

#### Multiple Model Support
- **Long Short-Term Memory (LSTM)** networks for time-series forecasting
- **XGBoost** gradient-boosted trees for tabular prediction
- **Random Forest** for ensemble diversity
- **Ensemble** model stacking with weighted aggregation
- **Hidden Markov Models (HMM)** for market regime detection

#### Advanced Feature Engineering
- **50+ technical indicators** (momentum, trend, volatility, volume)
- Candlestick pattern recognition (hammer, doji, engulfing, morning/evening star, etc.)
- Pattern confluence scoring
- PCA-based dimensionality reduction
- SHAP-based feature importance & model explainability

#### Training & Validation
- Walk-forward validation (no look-ahead bias)
- Hyperparameter tuning via **Optuna**
- Model registry with version tracking
- Performance tracking (MAE, RMSE, directional accuracy)

---

### 📊 Portfolio Optimization

#### Multiple Optimization Strategies
- Maximum Sharpe Ratio
- Minimum Volatility
- Risk Parity
- Top-K selection based on ML predictions

#### Risk Management
- Position sizing based on risk tolerance and volatility
- Stop-loss automation (2:1 reward-to-risk ratio)
- Portfolio constraints (max/min position size, turnover limits)
- Maximum drawdown monitoring

#### Rebalancing
- Automated rebalancing (daily/weekly/monthly)
- Transaction cost modeling
- Slippage and bid-ask spread consideration

---

### � Signal Generation
- ML-driven **buy / sell / hold** signals
- Multi-factor confluence scoring (technical + ML + pattern)
- Position sizing integration
- Configurable signal thresholds

---

### �📈 Interactive Dashboard

The frontend is a **React 19 SPA** (built with Vite) that communicates with the FastAPI backend.

| Tab | Features |
|-----|----------|
| **Analysis** | Real-time quotes, candlestick charts with lightweight-charts v5, pattern overlays, extended hours data, dynamic 8-slot watchlist, S&P 500 screener |
| **Predictions** | Model predictions vs. actual, confidence intervals (95%), SHAP feature importance, multi-model comparison |
| **Portfolio** | Holdings breakdown, efficient frontier, risk metrics (Sharpe, drawdown), performance attribution |
| **Backtesting** | Historical strategy performance, equity curve vs. S&P 500, trade analysis, Monte Carlo simulations |
| **Heatmap** | Sector/correlation heatmaps for portfolio diversification analysis |
| **Optimization** | Interactive portfolio optimizer with strategy selection and constraint tuning |

#### TradingView-Style Detail View
- Inline **lightweight-charts v5** candlestick charts
- ML buy/sell marker overlays
- Predicted price line & confidence band overlays
- User-selectable candlestick pattern filters (up to 3 concurrent)
- Session-persistent preferences via `localStorage`

---

### 📡 Data Coverage

#### Supported Indices
- **S&P 500** (~503 stocks)
- Custom stock lists via free-text search

#### Data Frequencies
- Daily (5+ years historical)
- 15-minute intraday
- 5-minute intraday
- 1-minute intraday (optional)

#### Data Sources
| Source | Type | Cost |
|--------|------|------|
| Yahoo Finance (`yfinance`) | Primary — OHLCV, extended hours | Free |
| Alpha Vantage | Supplementary — intraday, extended hours | Free tier available |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       FRONTEND  (React 19 + Vite)                   │
│  ┌──────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌────────┐ │
│  │ Analysis │ │ Predictions│ │Portfolio │ │Backtesting│ │Heatmap │ │
│  └────┬─────┘ └─────┬──────┘ └────┬─────┘ └────┬─────┘ └───┬────┘ │
│       │   lightweight-charts v5 / Recharts      │           │      │
└───────┼─────────────┼──────────────┼────────────┼───────────┼──────┘
        │             │              │            │           │
        ▼             ▼              ▼            ▼           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BACKEND  (FastAPI v2.0.0)                         │
│  /api/data  /api/predict  /api/backtest  /api/portfolio             │
│  /api/training  /api/patterns  /api/export                          │
└───────┬─────────────┬──────────────┬────────────┬───────────────────┘
        │             │              │            │
        ▼             ▼              ▼            ▼
┌───────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────────┐
│ Data Layer    │ │ ML Models    │ │  Portfolio   │ │   Signals     │
│ • yfinance    │ │ • LSTM       │ │ • MV Optim.  │ │ • Generator   │
│ • Alpha Vant. │ │ • XGBoost    │ │ • Risk Parity│ │ • Position    │
│ • Live Data   │ │ • RF / SVR   │ │ • Top-K      │ │   Sizing      │
│ • Data Cache  │ │ • Ensemble   │ │ • Rebalance  │ │ • Confluence  │
│ • Validator   │ │ • HMM Regime │ │              │ │               │
└───────────────┘ └──────────────┘ └──────────────┘ └───────────────┘
```

---

## 📁 Project Structure

```
StockPricePrediction/
├── quantvision/                # React 19 frontend (Vite)
│   ├── src/
│   │   ├── App.jsx             # Main application shell & tab router
│   │   ├── components/
│   │   │   ├── TradingViewDetail.jsx   # lightweight-charts v5 detail view
│   │   │   └── UIComponents.jsx        # Shared UI primitives
│   │   ├── tabs/
│   │   │   ├── AnalysisTab.jsx         # Watchlist, quotes, patterns
│   │   │   ├── PredictionsTab.jsx      # ML predictions & SHAP
│   │   │   ├── PortfolioTab.jsx        # Holdings & allocation
│   │   │   ├── BacktestTab.jsx         # Strategy backtesting
│   │   │   ├── HeatmapTab.jsx          # Sector heatmaps
│   │   │   └── OptimizationTab.jsx     # Portfolio optimizer
│   │   └── utils/
│   │       └── api.js                  # API client helpers
│   ├── package.json
│   └── vite.config.js
│
├── src/                        # Python backend
│   ├── api/                    # FastAPI app
│   │   ├── main.py             # App entry point, CORS, routers
│   │   ├── routes/
│   │   │   ├── data.py         # Market data & extended hours
│   │   │   ├── predict.py      # ML predictions
│   │   │   ├── training.py     # Model training endpoints
│   │   │   ├── backtest.py     # Backtesting engine
│   │   │   ├── portfolio.py    # Portfolio optimization
│   │   │   ├── patterns.py     # Candlestick patterns
│   │   │   └── export.py       # Data export (CSV / JSON)
│   │   └── schemas/            # Pydantic request/response models
│   ├── data/                   # Data acquisition & storage
│   │   ├── live_data.py        # Real-time & extended hours data
│   │   ├── market_data.py      # Historical data management
│   │   ├── data_cache.py       # In-memory caching layer
│   │   ├── alpha_vantage_provider.py
│   │   └── data_validator.py
│   ├── features/               # Feature engineering
│   │   ├── technical_indicators.py
│   │   ├── candlestick_patterns.py
│   │   ├── pattern_detector.py
│   │   ├── confluence.py
│   │   └── feature_engineering.py
│   ├── models/                 # ML models
│   │   ├── lstm_model.py
│   │   ├── xgboost_model.py
│   │   ├── random_forest_model.py
│   │   ├── ensemble.py
│   │   ├── regime_detection.py     # HMM regime detection
│   │   ├── explainability.py       # SHAP analysis
│   │   ├── model_trainer.py
│   │   └── model_registry.py
│   ├── portfolio/              # Portfolio optimization
│   ├── backtesting/            # Backtesting engine
│   ├── signals/                # Signal generation & position sizing
│   │   ├── signal_generator.py
│   │   └── position_sizing.py
│   └── utils/                  # Shared utilities
│
├── config/
│   ├── config.yaml             # Main application settings
│   └── logging_config.yaml
├── models/                     # Saved / trained model artifacts
├── data/                       # Raw & processed data
├── notebooks/                  # Jupyter notebooks (exploration)
├── tests/                      # Unit & integration tests
├── docker/                     # Docker configuration
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project metadata
├── Makefile                    # Common commands
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **Node.js 18+** and **npm**

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/StockPricePrediction.git
cd StockPricePrediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac / Linux

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your API keys (optional — Alpha Vantage, etc.)
```

### 3. Start the Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 4. Start the Frontend
```bash
cd quantvision
npm install
npm run dev
# Dashboard opens at http://localhost:5173
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Frontend** | React 19, Vite 7, lightweight-charts v5, Recharts |
| **Backend** | Python 3.11, FastAPI |
| **Data** | pandas 2.x, NumPy, yfinance, Alpha Vantage |
| **ML / DL** | TensorFlow / Keras (LSTM), XGBoost, scikit-learn, HMMlearn |
| **Optimization** | cvxpy, SciPy, Optuna |
| **Explainability** | SHAP |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Dev Tools** | Jupyter, Docker |

---

## 📊 API Endpoints

| Prefix | Description |
|--------|-------------|
| `GET /health` | Health check |
| `/api/data` | Market data, quotes, extended hours |
| `/api/predict` | ML price predictions |
| `/api/training` | Model training & hyperparameter tuning |
| `/api/backtest` | Strategy backtesting |
| `/api/portfolio` | Portfolio optimization & allocation |
| `/api/patterns` | Candlestick pattern detection |
| `/api/export` | CSV / JSON data export |

Full interactive API docs available at **`/docs`** (Swagger UI) when the backend is running.

---

## 📧 Contact

**Tan Yee Hern**  
Bachelor of Computer Science (Honours)  
Universiti Tunku Abdul Rahman  
Faculty of Information and Communication Technology

---

## 📜 License

Academic Project — UTAR 2025  
For educational purposes only.

---

## 🙏 Acknowledgments

- UTAR Faculty of ICT
- TensorFlow, scikit-learn, and FastAPI communities
- Yahoo Finance & Alpha Vantage for data access
- TradingView for the lightweight-charts library
