# API Configuration Guide

## Quick Summary

| API/Service | Key Required? | Status | Used In Code? | Priority |
|-------------|---------------|--------|---------------|----------|
| yfinance | ❌ No | ✅ Working | ✅ Yes (4 files) | **HIGH** |
| Wikipedia | ❌ No | ✅ Working | ✅ Yes (2 files) | **HIGH** |
| Alpha Vantage | ✅ Yes | ⚪ Not Used | ❌ No | LOW |
| Polygon.io | ✅ Yes | ⚪ Not Used | ❌ No | LOW |
| IEX Cloud | ✅ Yes | ⚪ Not Used | ❌ No | LOW |

> **TL;DR:** Only yfinance and Wikipedia are actually used. No API keys needed!

---

## No API Key Needed (Primary Sources)

### 1. yfinance
- **Purpose:** Primary stock data source for all features
- **Status:** ✅ WORKING
- **Used in:**
  - `src/data/market_data.py` - Market heatmap data
  - `src/data/data_loader.py` - Stock downloads
  - `src/data/data_acquisition.py` - Bulk data downloads
  - `src/dashboard/heatmap.py` - Market status detection
- **Limits:** Rate limiting by Yahoo Finance (unofficial)
- **Notes:** No API key required. May have ~15 min delay during market hours.

### 2. Wikipedia
- **Purpose:** S&P 500 constituent list with sectors
- **Status:** ✅ WORKING
- **Used in:**
  - `src/data/market_data.py` - S&P 500 list for heatmap
  - `src/data/data_acquisition.py` - S&P 500 ticker list
- **Limits:** None
- **Notes:** Uses web scraping with User-Agent header.

---

## Optional APIs (Not Currently Used)

The following APIs are listed in `.env.example` and `config.yaml` as potential fallbacks, but **NO CODE** currently uses them.

### Alpha Vantage
- **Purpose:** Backup data source (mentioned in config.yaml)
- **Status:** ⚪ NOT IMPLEMENTED
- **Free Tier:** 25 calls/day, 5 calls/min
- **Get Key:** https://www.alphavantage.co/support/#api-key
- **Recommendation:** Keep in config for future use, but not required.

### Polygon.io
- **Purpose:** Intraday data (mentioned in config.yaml)
- **Status:** ⚪ NOT IMPLEMENTED
- **Free Tier:** Limited, paid plans available
- **Get Key:** https://polygon.io/dashboard/signup
- **Recommendation:** Only needed if adding intraday charts.

### IEX Cloud
- **Purpose:** Real-time data (mentioned in README only)
- **Status:** ⚪ NOT IMPLEMENTED
- **Free Tier:** 50,000 messages/month
- **Get Key:** https://iexcloud.io/console/
- **Recommendation:** Can be removed from documentation.

---

## Setup Instructions

### Quick Setup (Works Out of Box)
```bash
# No configuration needed!
# yfinance and Wikipedia work without API keys

# Just run the dashboard
streamlit run src/dashboard/app.py
```

### Full Setup (If Adding Optional APIs)
```bash
# Copy environment template
cp .env.example .env

# Edit .env and fill in (OPTIONAL):
ALPHA_VANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here
```

---

## Testing Configuration

```bash
# Run API test script
python tests/test_api_keys.py
```

**Expected Output:**
```
yfinance             [OK] Working
  Purpose: PRIMARY - Stock data (no key)
  Status:  AAPL: $259.37 - No API key required

Wikipedia            [OK] Working
  Purpose: PRIMARY - S&P 500 list (no key)
  Status:  Found 503 S&P 500 companies - No API key required

Alpha Vantage        [SKIP] Not Configured
  Purpose: OPTIONAL - Backup data source
  Status:  Optional - Add key to .env if needed

Polygon.io           [SKIP] Not Configured
  Purpose: OPTIONAL - Intraday data
  Status:  Optional - Add key to .env if needed

SUMMARY
  [OK]   Working:        2
  [SKIP] Not Configured: 2

[SUCCESS] Primary APIs working! Dashboard is fully functional.
```

---

## Cleanup Recommendations

### Can Be Removed from `.env.example`:
None - Keep Alpha Vantage and Polygon.io as future options.

### Can Be Removed from Documentation:
- IEX Cloud (not in .env.example, only in README)

### Already Correct:
- All APIs are marked as optional
- No hardcoded API keys in source code
- API keys loaded from environment variables

---

## Feature → API Mapping

```
Feature: Market Heatmap
├── S&P 500 List: Wikipedia (no key) ✅
├── Stock Prices: yfinance (no key) ✅
└── Market Status: yfinance (no key) ✅

Feature: Stock Predictions
├── Historical Data: yfinance (no key) ✅
└── Technical Indicators: Calculated locally ✅

Feature: Portfolio Analysis
├── Price Data: yfinance (no key) ✅
└── Metrics: Calculated locally ✅

Feature: Backtesting
└── Price Data: yfinance (no key) ✅
```

---

## Data Freshness Notes

- **Market Closed:** End-of-day prices (expected)
- **Market Open:** ~15 minute delay (yfinance free tier)
- **Real-time Needed?** Consider Polygon.io paid tier

For more details, see `data_freshness_report.md`.
