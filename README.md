# 📈 Stock Price Prediction and Portfolio Optimization Dashboard

This project implements an end-to-end machine learning pipeline for stock market analysis, combining state-of-the-art deep learning models with classical quantitative finance techniques. The system provides:

- **Real-time price predictions** using LSTM, XGBoost, and ensemble models
- **Automated portfolio optimization** with mean-variance optimization and risk management
- **Interactive web dashboard** for visualization and decision support
- **Comprehensive backtesting framework** with realistic transaction costs
- **Multi-timeframe analysis** (daily, 15-min, 5-min, 1-min intervals)

---

## 🎓 Academic Context

This system was developed as part of a **Bachelor of Computer Science (Honours)** thesis at **Universiti Tunku Abdul Rahman (UTAR)**, focusing on the intersection of machine learning and computational finance.

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
- **XGBoost** and **Random Forest** for ensemble predictions
- **Support Vector Regression (SVR)** for non-linear patterns
- Model stacking and weighted ensemble methods

#### Advanced Feature Engineering
- **50+ technical indicators** (momentum, trend, volatility, volume)
- Candlestick pattern recognition (hammer, doji, engulfing, etc.)
- PCA-based dimensionality reduction
- SHAP-based feature importance analysis

#### Robust Validation
- Walk-forward validation (no look-ahead bias)
- Time-series cross-validation
- Out-of-sample testing
- Performance tracking (MAE, RMSE, directional accuracy)

---

### 📊 Portfolio Optimization

#### Multiple Optimization Strategies
- Maximum Sharpe Ratio
- Minimum Volatility
- Risk Parity
- Top-K selection based on predictions

#### Risk Management
- Position sizing based on risk tolerance
- Stop-loss automation (2:1 reward-to-risk ratio)
- Portfolio constraints (max/min position size, turnover limits)
- Maximum drawdown monitoring

#### Rebalancing
- Automated rebalancing (daily/weekly/monthly)
- Transaction cost modeling
- Slippage and bid-ask spread consideration

---

### 📈 Interactive Dashboard

| Tab | Features |
|-----|----------|
| **Market Overview** | Real-time price charts, candlestick patterns with 20/200 MA, volume analysis, support/resistance detection |
| **Predictions** | Model predictions vs. actual, confidence intervals, SHAP feature importance, multi-model comparison |
| **Portfolio** | Holdings breakdown, efficient frontier, risk metrics (Sharpe, drawdown), performance attribution |
| **Backtesting** | Historical strategy performance, equity curve vs. S&P 500, trade analysis, Monte Carlo simulations |

---

### 📡 Data Coverage

#### Supported Indices
- **S&P 500** (~503 stocks)
- **Russell 2000** (~2000 stocks)
- Custom stock lists

#### Data Frequencies
- Daily (5+ years historical)
- 15-minute intraday
- 5-minute intraday
- 1-minute intraday (optional)

#### Data Sources
| Source | Type | Cost |
|--------|------|------|
| Yahoo Finance | Primary | Free |
| Alpha Vantage | Supplement | Free tier available |
| Polygon.io | Premium intraday | Paid |
| IEX Cloud | Real-time | Paid |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                      │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐        │
│  │ Yahoo Finance├───►│ Data Acquisition ├───►│ PostgreSQL/SQLite │        │
│  │     API     │    │     Module       │    │    Database        │        │
│  └─────────────┘    └──────────────────┘    └─────────┬───────────┘        │
└───────────────────────────────────────────────────────┼─────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FEATURE ENGINEERING                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌───────┐   ┌──────────────┐        │
│  │  Technical   ├──►│ Candlestick  ├──►│  PCA  ├──►│    SHAP      │        │
│  │  Indicators  │   │  Patterns    │   │       │   │  Selection   │        │
│  └──────────────┘   └──────────────┘   └───────┘   └──────┬───────┘        │
└──────────────────────────────────────────────────────────┼──────────────────┘
                                                           │
                                                           ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL LAYER                                       │
│  ┌──────┐   ┌─────────┐   ┌───────────────┐   ┌─────┐                       │
│  │ LSTM ├──►│ XGBoost ├──►│ Random Forest ├──►│ SVR │                       │
│  └──┬───┘   └────┬────┘   └───────┬───────┘   └──┬──┘                       │
│     └────────────┴────────────────┴──────────────┘                          │
│                           │                                                  │
│                           ▼                                                  │
│                  ┌─────────────────┐                                         │
│                  │    Ensemble     │                                         │
│                  │   Predictions   │                                         │
│                  └────────┬────────┘                                         │
└───────────────────────────┼─────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PORTFOLIO LAYER                                      │
│  ┌────────────┐   ┌───────────────┐   ┌──────────────┐   ┌─────────────┐   │
│  │ Prediction ├──►│ Mean-Variance ├──►│     Risk     ├──►│  Position   │   │
│  │   Filter   │   │   Optimizer   │   │  Management  │   │   Sizing    │   │
│  └────────────┘   └───────────────┘   └──────────────┘   └──────┬──────┘   │
└─────────────────────────────────────────────────────────────────┼───────────┘
                                                                  │
                                                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER                                     │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────┐   │
│  │   Backtesting   ├──►│   Performance   ├──►│  Streamlit Dashboard    │   │
│  │     Engine      │   │    Metrics      │   │    (User Interface)     │   │
│  └─────────────────┘   └─────────────────┘   └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Market Data │ -> │  Technical   │ -> │   Machine   │ -> │  Portfolio   │
│ (OHLCV)     │    │  Indicators  │    │   Learning  │    │ Optimization │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                                              │
                                              v
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Dashboard  │ <- │  Backtesting │ <- │   Trading   │
│    (UI)     │    │    Engine    │    │   Signals   │
└─────────────┘    └──────────────┘    └─────────────┘
```

---

## 📁 Project Structure

```
stock-prediction-dashboard/
├── .github/workflows/          # CI/CD pipelines
├── config/                     # Configuration files
│   ├── config.yaml             # Main settings
│   └── logging_config.yaml     # Logging settings
├── data/                       # Data storage
│   ├── raw/                    # Raw downloaded data
│   ├── processed/              # Cleaned data
│   └── indicators/             # Calculated indicators
├── database/                   # Database files
│   └── schema.sql              # Database schema
├── docs/                       # Documentation
├── logs/                       # Application logs
├── models/                     # Saved ML models
│   ├── saved_models/           # Trained model files
│   └── scalers/                # Data scalers
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_development.ipynb
├── scripts/                    # Utility scripts
│   └── download_daily_data.py
├── src/                        # Source code
│   ├── data/                   # Data handling
│   ├── features/               # Feature engineering
│   ├── models/                 # ML models
│   ├── portfolio/              # Portfolio optimization
│   ├── backtesting/            # Backtesting engine
│   ├── dashboard/              # Streamlit app
│   └── utils/                  # Utilities
├── tests/                      # Unit tests
├── docker/                     # Docker configuration
├── .env.example                # Environment template
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project configuration
├── Makefile                    # Common commands
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/stock-prediction-dashboard.git
cd stock-prediction-dashboard

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env with your API keys (optional)
```

### 3. Download Data
```bash
python scripts/download_daily_data.py
```

### 4. Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11 |
| **Data** | pandas, numpy, yfinance |
| **ML** | scikit-learn, XGBoost, TensorFlow/Keras |
| **Optimization** | cvxpy, scipy |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Dashboard** | Streamlit |
| **Database** | SQLite (dev), PostgreSQL (prod) |
| **Deployment** | Docker, Streamlit Cloud |

---

## 📊 Model Performance

| Model | MAE | RMSE | Directional Accuracy |
|-------|-----|------|---------------------|
| LSTM | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| Ensemble | TBD | TBD | TBD |

*Performance metrics will be updated after model training*

---

## 📧 Contact

**Tan Yee Hern**  
Bachelor of Computer Science (Honours)  
Universiti Tunku Abdul Rahman  
Faculty of Information and Communication Technology

---

## 📜 License

Academic Project - UTAR 2025  
For educational purposes only.

---

## 🙏 Acknowledgments

- UTAR Faculty of ICT
- TensorFlow and scikit-learn communities
- Yahoo Finance for data access
