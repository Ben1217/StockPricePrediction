# Stock Price Prediction and Portfolio Optimization Dashboard

## Project Description
This project uses Machine Learning (LSTM, XGBoost, Random Forest) and Technical Analysis to predict stock prices and optimize investment portfolios.

## Author
Tan Yee Hern  
Bachelor of Computer Science (Honours)  
Universiti Tunku Abdul Rahman

## Setup Instructions

### 1. Clone Repository
```bash
git clone [your-repo-url]
cd Stock_Prediction_Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Test Setup
```bash
python test_setup.py
```

## Project Structure
```
Stock_Prediction_Project/
├── venv/                    # Virtual environment
├── data/                    # Data storage
│   ├── raw/                # Raw downloaded data
│   └── processed/          # Cleaned and processed data
├── notebooks/              # Jupyter notebooks for exploration
├── src/                    # Source code modules
│   ├── data_loader.py     # Data downloading functions
│   ├── feature_engineering.py  # Technical indicators
│   ├── models.py          # ML model implementations
│   ├── portfolio.py       # Portfolio optimization
│   └── dashboard.py       # Dashboard application
├── models/                 # Saved ML models
│   └── saved_models/      # Model files
├── results/                # Analysis results
│   ├── figures/           # Charts and plots
│   └── reports/           # Generated reports
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── test_setup.py          # Environment test script
└── README.md              # This file
```

## Current Status

✅ **Phase 1: Environment Setup - COMPLETE**
- Python 3.11 installed
- All libraries installed (8/8 tests passed)
- Virtual environment configured

⏳ **Phase 2: Data Collection - IN PROGRESS**
- Downloading 10 years of historical data
- S&P 500, Russell 2000, NASDAQ indices

⏳ **Phase 3: Feature Engineering - UPCOMING**
⏳ **Phase 4: Model Development - UPCOMING**
⏳ **Phase 5: Portfolio Optimization - UPCOMING**
⏳ **Phase 6: Dashboard Development - UPCOMING**

## Technologies Used

### Core Technologies
- **Python 3.11** - Programming language
- **Jupyter Notebook** - Interactive development

### Data Processing
- **pandas 2.3.3** - Data manipulation
- **numpy 2.3.5** - Numerical computing
- **yfinance** - Stock data acquisition

### Visualization
- **matplotlib** - Static plots
- **seaborn** - Statistical visualizations
- **plotly** - Interactive charts

### Technical Analysis
- **ta** - Technical indicators library

### Machine Learning
- **scikit-learn** - Traditional ML algorithms
- **XGBoost** - Gradient boosting
- **TensorFlow 2.20.0** - Deep learning (LSTM)

### Optimization
- **cvxpy** - Portfolio optimization
- **shap** - Model interpretability

### Dashboard
- **Dash** - Interactive web applications

## Usage

### Running Tests
```bash
# Activate virtual environment
venv\Scripts\activate

# Run setup test
python test_setup.py
```

### Starting Jupyter Notebook
```bash
# From project root
jupyter notebook
```

### Data Collection
```bash
# Run data collection notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

## Project Timeline

- **Week 1-2**: Environment setup ✅
- **Week 3-4**: Data collection and exploration
- **Week 5-6**: Feature engineering (technical indicators)
- **Week 7-10**: Model development (LSTM, XGBoost, RF)
- **Week 11-12**: Portfolio optimization
- **Week 13-15**: Dashboard development
- **Week 16-17**: Integration and testing
- **Week 18**: Validation
- **Week 19**: Deployment
- **Week 20**: Final report and presentation

## Key Features (Planned)

1. **Stock Price Prediction**
   - LSTM neural networks
   - XGBoost ensemble models
   - Random Forest regression
   - Technical indicator-based features

2. **Portfolio Optimization**
   - Mean-variance optimization
   - Maximum Sharpe ratio
   - Minimum volatility strategies
   - Risk-adjusted returns

3. **Interactive Dashboard**
   - Real-time price charts
   - Prediction visualizations
   - Portfolio allocation pie charts
   - Backtesting results
   - Feature importance plots

## Data Sources

- **Yahoo Finance** - Historical OHLCV data
- **Indices**: SPY (S&P 500), IWM (Russell 2000), QQQ (NASDAQ)
- **Period**: 10 years of daily data

## Contact

**Tan Yee Hern**  
Universiti Tunku Abdul Rahman  
Faculty of Information and Communication Technology  
Email: [your-email@student.utar.edu.my]

## License

Academic Project - UTAR 2025  
For educational purposes only.

## Acknowledgments

- UTAR Faculty of ICT
- Project Supervisor: [Supervisor Name]
- TensorFlow and scikit-learn communities

