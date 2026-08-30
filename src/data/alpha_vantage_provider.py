"""
Alpha Vantage data provider with rate limiting and caching.
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from typing import Optional, Dict
from threading import Lock
from cachetools import TTLCache

from ..utils.logger import get_logger
from .provider_errors import (
    PremiumEndpointError,
    QuotaExceededError,
    classify_alpha_vantage_message,
)

logger = get_logger(__name__)

# Rate limiter for the Alpha Vantage free tier, which enforces two separate
# limits: a hard 25 requests/day quota that resets at UTC midnight, and a burst
# limit of roughly one request per second. Exceeding either returns HTTP 200 with
# an "Information" block instead of data, so both are handled before the wire.
#
# The daily cap is a counter (sleeping through it is pointless — it lasts hours);
# the burst limit is a short spacing sleep, which is the only way to comply.
_DAILY_LIMIT = 25
_MIN_INTERVAL = 1.1  # seconds between calls; the free tier documents 1 req/second
_rate_lock = Lock()
_calls_today = 0
_quota_date = None  # UTC date that _calls_today refers to
_last_call_time = 0.0  # time.monotonic() of the last reserved slot


def consume_request_slot() -> None:
    """
    Reserve one Alpha Vantage request: space it against the burst limit, then
    count it against today's quota.

    Every Alpha Vantage call in the app routes through here — this module, the
    /prices route fetcher, and the GLOBAL_QUOTE live-quote helper — so the
    counter reflects real usage rather than one class's share of it.

    Raises
    ------
    QuotaExceededError
        When the daily free-tier quota is already spent.
    """
    global _calls_today, _quota_date, _last_call_time
    with _rate_lock:
        today = datetime.now(timezone.utc).date()
        if _quota_date != today:
            _quota_date = today
            _calls_today = 0
        if _calls_today >= _DAILY_LIMIT:
            raise QuotaExceededError(
                f"Alpha Vantage free-tier daily quota exhausted "
                f"({_DAILY_LIMIT} requests/day, resets at UTC midnight). "
                f"Use source=yfinance or wait for the reset."
            )
        # Held across the sleep on purpose: concurrent callers must queue behind
        # each other, or they all wake up and burst together.
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.monotonic()
        _calls_today += 1


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
        """Make an API call, spaced and counted against the free-tier limits."""
        consume_request_slot()

        params["apikey"] = self.api_key
        try:
            resp = requests.get(self.base_url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            raise

        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        # Alpha Vantage reports BOTH premium-endpoint blocks and daily-quota
        # exhaustion under "Information". Without this check either one falls
        # through and surfaces as a misleading "no data / symbol not found".
        if "Information" in data:
            raise classify_alpha_vantage_message(
                data["Information"], endpoint=params.get("function")
            )
        if "Note" in data:
            logger.warning(f"Alpha Vantage rate limit note: {data['Note']}")
        return data

    def get_daily(self, symbol: str, outputsize: str = "compact") -> pd.DataFrame:
        """
        Fetch daily OHLCV data via the free-tier TIME_SERIES_DAILY endpoint.

        This endpoint returns raw (unadjusted) prices and has no adjusted-close
        column, so "Adj Close" mirrors "Close". Split/dividend-adjusted history
        requires the premium TIME_SERIES_DAILY_ADJUSTED endpoint.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        outputsize : str
            'compact' (latest 100 bars). 'full' is premium-only and is rejected.
        """
        if outputsize != "compact":
            raise ValueError(
                f"Alpha Vantage outputsize={outputsize!r} is not available on the "
                f"free tier; only 'compact' (latest 100 bars) is supported."
            )

        cache_key = f"av_daily:{symbol}:{outputsize}"
        if cache_key in _av_cache:
            return _av_cache[cache_key]

        data = self._rate_limited_get({
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": outputsize,
        })

        ts = data.get("Time Series (Daily)", {})
        if not ts:
            logger.warning(f"No daily data for {symbol}")
            return pd.DataFrame()

        rows = []
        for dt, vals in ts.items():
            close = float(vals["4. close"])
            rows.append({
                "Date": dt,
                "Open": float(vals["1. open"]),
                "High": float(vals["2. high"]),
                "Low": float(vals["3. low"]),
                "Close": close,
                # TIME_SERIES_DAILY has no adjusted close - mirror the raw close.
                "Adj Close": close,
                "Volume": int(vals["5. volume"]),
            })

        df = pd.DataFrame(rows)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        _av_cache[cache_key] = df
        logger.info(f"Fetched {len(df)} daily bars for {symbol} from Alpha Vantage")
        return df

    def get_intraday(self, symbol: str, interval: str = "15min") -> pd.DataFrame:
        """
        Not available on a free Alpha Vantage key.

        TIME_SERIES_INTRADAY is a premium-only endpoint; on a free key Alpha
        Vantage returns an "Information" block instead of data. This raises
        rather than returning an empty DataFrame so callers see the real reason.

        Parameters
        ----------
        symbol : str
            Ticker symbol
        interval : str
            '1min', '5min', '15min', '30min', '60min'
        """
        raise PremiumEndpointError(
            f"Alpha Vantage TIME_SERIES_INTRADAY is a premium-only endpoint and is "
            f"not available on a free API key (requested {symbol} @ {interval}). "
            f"Use source=yfinance for intraday bars, or upgrade the Alpha Vantage plan.",
            endpoint="TIME_SERIES_INTRADAY",
        )

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
