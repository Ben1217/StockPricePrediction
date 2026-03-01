"""
Alpha Vantage data provider with rate limiting and caching.
"""

import os
import time
import requests
import pandas as pd
from typing import Optional, Dict
from threading import Lock
from cachetools import TTLCache

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiter: Alpha Vantage free tier = 5 calls/min, 500 calls/day
_rate_lock = Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 12.5  # seconds between calls (5/min)

# Cache: 200 entries, 1 hour TTL
_av_cache = TTLCache(maxsize=200, ttl=3600)


class AlphaVantageProvider:
    """Alpha Vantage API data provider with rate limiting."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"
        if not self.api_key or self.api_key == "your_alpha_vantage_key":
            logger.warning("Alpha Vantage API key not configured")

    def _rate_limited_get(self, params: Dict) -> Dict:
        """Make a rate-limited API call."""
        global _last_call_time
        with _rate_lock:
            elapsed = time.time() - _last_call_time
            if elapsed < _MIN_INTERVAL:
                time.sleep(_MIN_INTERVAL - elapsed)
            _last_call_time = time.time()

        params["apikey"] = self.api_key
        try:
            resp = requests.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if "Error Message" in data:
                raise ValueError(data["Error Message"])
            if "Note" in data:
                logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
            return data
        except requests.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            raise

    def get_daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """
        Fetch daily OHLCV data.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        outputsize : str
            'compact' (100 days) or 'full' (20+ years)
        """
        cache_key = f"av_daily:{symbol}:{outputsize}"
        if cache_key in _av_cache:
            return _av_cache[cache_key]

        data = self._rate_limited_get({
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": outputsize,
        })

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            logger.warning(f"No daily data for {symbol}")
            return pd.DataFrame()

        rows = []
        for dt, vals in ts.items():
            rows.append({
                "Date": dt,
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["4. close"]),
                "Adj Close": float(vals["5. adjusted close"]),
                "Volume": int(vals["6. volume"]),
            })

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        _av_cache[cache_key] = df
        logger.info(f"Fetched {len(df)} daily bars for {symbol} from Alpha Vantage")
        return df

    def get_intraday(self, symbol: str, interval: str = "15min") -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        interval : str
            '1min', '5min', '15min', '30min', '60min'
        """
        cache_key = f"av_intraday:{symbol}:{interval}"
        if cache_key in _av_cache:
            return _av_cache[cache_key]

        data = self._rate_limited_get({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": "compact",
        })

        ts_key = f"Time Series ({interval})"
        ts = data.get(ts_key, {})
        if not ts:
            return pd.DataFrame()

        rows = []
        for dt, vals in ts.items():
            rows.append({
                "Datetime": dt,
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": float(vals["4. close"]),
                "Volume": int(vals["5. volume"]),
            })

        df = pd.DataFrame(rows)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df = df.sort_values("Datetime").set_index("Datetime")
        _av_cache[cache_key] = df
        return df

    def get_quote(self, symbol: str) -> Dict:
        """Get real-time quote for a symbol."""
        cache_key = f"av_quote:{symbol}"
        if cache_key in _av_cache:
            return _av_cache[cache_key]

        data = self._rate_limited_get({
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
        })

        quote = data.get("Global Quote", {})
        result = {
            "symbol": quote.get("01. symbol", symbol),
            "price": float(quote.get("05. price", 0)),
            "change": float(quote.get("09. change", 0)),
            "change_pct": quote.get("10. change percent", "0%"),
            "volume": int(quote.get("06. volume", 0)),
        }
        _av_cache[cache_key] = result
        return result
