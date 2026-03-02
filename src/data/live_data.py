"""
live_data.py — Live market data helpers.

Provides:
- is_market_open()              : NYSE market-hours detection
- get_market_session()          : PRE_MARKET / REGULAR / POST_MARKET / MARKET_CLOSED
- get_cache_ttl()               : Dynamic TTL (60 s open, 3600 s closed)
- validate_freshness()          : Rejects data older than threshold (open/closed)
- validate_extended_freshness() : Per-session freshness validation
- fetch_live_quote()            : Latest price via fast_info / GLOBAL_QUOTE
- fetch_extended_quote()        : Pre + regular + post prices for all 3 sessions
"""

import os
import logging
from datetime import datetime, timedelta

import pytz
import yfinance as yf

logger = logging.getLogger(__name__)

# US/Eastern timezone — NYSE reference
_ET = pytz.timezone("US/Eastern")
_UTC = pytz.UTC

# Freshness thresholds (simple open/closed)
_LIVE_MAX_AGE  = timedelta(minutes=15)
_CLOSE_MAX_AGE = timedelta(hours=18)

# Cache TTL in seconds
_TTL_OPEN   = 60
_TTL_CLOSED = 3600

# ── Session constants ─────────────────────────────────────────────────────────
SESSION_PRE_MARKET  = "PRE_MARKET"
SESSION_REGULAR     = "REGULAR"
SESSION_POST_MARKET = "POST_MARKET"
SESSION_CLOSED      = "MARKET_CLOSED"

# Per-session freshness thresholds
_SESSION_MAX_AGE = {
    SESSION_PRE_MARKET:  timedelta(minutes=5),
    SESSION_REGULAR:     timedelta(minutes=1),
    SESSION_POST_MARKET: timedelta(minutes=5),
    SESSION_CLOSED:      timedelta(hours=12),
}


# ── Market-hours detection ────────────────────────────────────────────────────

def is_market_open() -> bool:
    """Return True if NYSE is currently open for regular trading."""
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        now_et = datetime.now(_ET)
        today = now_et.strftime("%Y-%m-%d")
        schedule = nyse.schedule(start_date=today, end_date=today)
        if schedule.empty:
            return False
        market_open = schedule.iloc[0]["market_open"].to_pydatetime()
        market_close = schedule.iloc[0]["market_close"].to_pydatetime()
        now_utc = datetime.now(_UTC)
        return market_open <= now_utc <= market_close
    except Exception as e:
        logger.warning(f"Market calendar check failed, defaulting to closed: {e}")
        now_et = datetime.now(_ET)
        if now_et.weekday() >= 5:
            return False
        market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
        return market_open <= now_et <= market_close


def get_market_session() -> str:
    """
    Return the current US market session:
    PRE_MARKET | REGULAR | POST_MARKET | MARKET_CLOSED
    """
    now_et = datetime.now(_ET)

    if now_et.weekday() >= 5:
        return SESSION_CLOSED

    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        today = now_et.strftime("%Y-%m-%d")
        if nyse.schedule(start_date=today, end_date=today).empty:
            return SESSION_CLOSED  # Holiday
    except Exception:
        pass

    t = now_et.strftime("%H:%M")
    if "04:00" <= t < "09:30":
        return SESSION_PRE_MARKET
    elif "09:30" <= t < "16:00":
        return SESSION_REGULAR
    elif "16:00" <= t < "20:00":
        return SESSION_POST_MARKET
    return SESSION_CLOSED


def get_cache_ttl() -> int:
    """Return appropriate cache duration in seconds based on market status."""
    return _TTL_OPEN if is_market_open() else _TTL_CLOSED


# ── Freshness validation ──────────────────────────────────────────────────────

def validate_freshness(data_timestamp: datetime) -> None:
    """Raise ValueError if data_timestamp is older than the open/closed threshold."""
    if data_timestamp.tzinfo is None:
        data_timestamp = _UTC.localize(data_timestamp)
    age = datetime.now(_UTC) - data_timestamp
    threshold = _LIVE_MAX_AGE if is_market_open() else _CLOSE_MAX_AGE
    if age > threshold:
        raise ValueError(
            f"Stale data detected: timestamp={data_timestamp.isoformat()}, "
            f"age={age}, threshold={threshold}."
        )


def validate_extended_freshness(data_timestamp: datetime, session: str | None = None) -> None:
    """
    Validate data freshness using per-session thresholds.

    Parameters
    ----------
    data_timestamp : datetime   Timezone-aware timestamp.
    session        : str | None Active session; auto-detected when None.
    """
    if data_timestamp.tzinfo is None:
        data_timestamp = _UTC.localize(data_timestamp)
    if session is None:
        session = get_market_session()
    age     = datetime.now(_UTC) - data_timestamp
    allowed = _SESSION_MAX_AGE.get(session, timedelta(hours=12))
    if age > allowed:
        raise ValueError(
            f"Stale {session} data: timestamp={data_timestamp.isoformat()}, "
            f"age={int(age.total_seconds() // 60)} min, "
            f"limit={int(allowed.total_seconds() // 60)} min."
        )


# ── Live quote fetching ───────────────────────────────────────────────────────

def fetch_live_quote_yfinance(symbol: str) -> dict:
    """Fetch latest price via fast_info → 1m history → info dict."""
    ticker = yf.Ticker(symbol)

    try:
        fi = ticker.fast_info
        price = fi.last_price
        volume = getattr(fi, "last_volume", None)
        if price and price > 0:
            return {
                "symbol": symbol, "price": round(float(price), 4),
                "volume": int(volume) if volume else None,
                "timestamp": datetime.now(_UTC).isoformat(),
                "source": "yfinance_fast_info",
            }
    except Exception as e:
        logger.debug(f"[{symbol}] fast_info failed: {e}")

    try:
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            last = hist.iloc[-1]
            ts = hist.index[-1]
            if hasattr(ts, "to_pydatetime"):
                ts = ts.to_pydatetime()
            if ts.tzinfo is None:
                ts = _UTC.localize(ts)
            return {
                "symbol": symbol, "price": round(float(last["Close"]), 4),
                "volume": int(last["Volume"]) if "Volume" in last else None,
                "timestamp": ts.isoformat(), "source": "yfinance_1m",
            }
    except Exception as e:
        logger.debug(f"[{symbol}] 1m history failed: {e}")

    try:
        info = ticker.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        if price and price > 0:
            return {
                "symbol": symbol, "price": round(float(price), 4),
                "volume": info.get("regularMarketVolume"),
                "timestamp": datetime.now(_UTC).isoformat(),
                "source": "yfinance_info",
            }
    except Exception as e:
        logger.debug(f"[{symbol}] info dict failed: {e}")

    raise RuntimeError(f"Could not fetch live quote for {symbol} via yfinance")


def fetch_live_quote_alpha_vantage(symbol: str) -> dict:
    """Fetch latest price via Alpha Vantage GLOBAL_QUOTE."""
    import requests
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "your_alpha_vantage_key":
        raise ValueError("Alpha Vantage API key not configured")

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json().get("Global Quote", {})
    if not data or "05. price" not in data:
        raise RuntimeError(f"Empty GLOBAL_QUOTE response for {symbol}")

    price = float(data["05. price"])
    volume = int(data.get("06. volume", 0))
    latest_day = data.get("07. latest trading day", "")
    try:
        ts = _ET.localize(datetime.strptime(latest_day, "%Y-%m-%d").replace(hour=16))
        ts = ts.astimezone(_UTC)
    except Exception:
        ts = datetime.now(_UTC)

    return {
        "symbol": symbol, "price": round(price, 4),
        "volume": volume, "timestamp": ts.isoformat(),
        "source": "alpha_vantage_global_quote",
    }


def fetch_live_quote(symbol: str, source: str = "yfinance") -> dict:
    """Fetch the most recent available price. Returns dict with market_open field."""
    if source == "alpha_vantage":
        result = fetch_live_quote_alpha_vantage(symbol)
    else:
        result = fetch_live_quote_yfinance(symbol)
    result["market_open"] = is_market_open()
    return result


# ── Extended (pre/post) quote fetching ────────────────────────────────────────

def _unix_to_iso(unix_ts) -> str | None:
    if unix_ts is None:
        return None
    try:
        return datetime.fromtimestamp(unix_ts, tz=_UTC).isoformat()
    except Exception:
        return None


def _safe_info(info: dict, key: str, default=None):
    v = info.get(key)
    return v if v not in (None, 0, "N/A") else default


def fetch_extended_quote(symbol: str, source: str = "yfinance") -> dict:
    """
    Fetch pre-market, regular, and post-market prices for *symbol*.

    Returns dict keys:
        symbol, session, market_open,
        regular  : {price, open, prev_close}
        pre      : {price, change, change_pct, time, available}
        post     : {price, change, change_pct, time, available}
        low_volume_warning : bool
    """
    session = get_market_session()
    if source == "alpha_vantage":
        return _extended_quote_alpha_vantage(symbol, session)
    return _extended_quote_yfinance(symbol, session)


def _extended_quote_yfinance(symbol: str, session: str) -> dict:
    ticker = yf.Ticker(symbol)

    try:
        info = ticker.info
    except Exception:
        info = {}

    regular_price  = _safe_info(info, "regularMarketPrice") or _safe_info(info, "currentPrice")
    regular_open   = _safe_info(info, "regularMarketOpen")
    regular_close  = _safe_info(info, "regularMarketPreviousClose")

    pre_price       = _safe_info(info, "preMarketPrice")
    pre_change      = _safe_info(info, "preMarketChange")
    pre_change_pct  = _safe_info(info, "preMarketChangePercent")
    pre_time_unix   = _safe_info(info, "preMarketTime")

    post_price      = _safe_info(info, "postMarketPrice")
    post_change     = _safe_info(info, "postMarketChange")
    post_change_pct = _safe_info(info, "postMarketChangePercent")
    post_time_unix  = _safe_info(info, "postMarketTime")

    # Extended candles with prePost=True to detect low volume
    low_volume = False
    try:
        hist = ticker.history(period="1d", interval="1m", prePost=True, auto_adjust=True)
        if not hist.empty:
            last_vol = int(hist["Volume"].iloc[-1])
            avg_vol  = int(hist["Volume"].mean())
            low_volume = avg_vol > 0 and last_vol < avg_vol * 0.2
    except Exception:
        pass

    def _pct(v):
        # yfinance returns pre/post change_pct as a ratio (e.g. 0.0040 = 0.40%)
        if v is None:
            return None
        # If absolute value < 1 assume ratio, convert to percent
        return round(v * 100, 4) if abs(v) < 1 else round(v, 4)

    return {
        "symbol":  symbol,
        "session": session,
        "market_open": session == SESSION_REGULAR,
        "regular": {
            "price":      round(regular_price, 4) if regular_price else None,
            "open":       round(regular_open,  4) if regular_open  else None,
            "prev_close": round(regular_close, 4) if regular_close else None,
        },
        "pre": {
            "price":      round(pre_price,  4) if pre_price  else None,
            "change":     round(pre_change, 4) if pre_change else None,
            "change_pct": _pct(pre_change_pct),
            "time":       _unix_to_iso(pre_time_unix),
            "available":  pre_price is not None,
        },
        "post": {
            "price":      round(post_price,  4) if post_price  else None,
            "change":     round(post_change, 4) if post_change else None,
            "change_pct": _pct(post_change_pct),
            "time":       _unix_to_iso(post_time_unix),
            "available":  post_price is not None,
        },
        "low_volume_warning": low_volume,
    }


def _extended_quote_alpha_vantage(symbol: str, session: str) -> dict:
    """Alpha Vantage extended hours via TIME_SERIES_INTRADAY with extended_hours=true."""
    import requests
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "your_alpha_vantage_key":
        raise ValueError("Alpha Vantage API key not configured")

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
        f"&interval=1min&outputsize=compact&extended_hours=true&apikey={api_key}"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    ts = resp.json().get("Time Series (1min)", {})
    if not ts:
        raise RuntimeError(f"No Alpha Vantage intraday data for {symbol}")

    latest_time = list(ts.keys())[0]  # first key = most recent
    latest = ts[latest_time]
    extended_price = float(latest["4. close"])

    gq_url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    )
    gq = requests.get(gq_url, timeout=15).json().get("Global Quote", {})
    regular_price = float(gq.get("05. price", 0)) or None
    prev_close    = float(gq.get("08. previous close", 0)) or None

    return {
        "symbol":  symbol,
        "session": session,
        "market_open": session == SESSION_REGULAR,
        "regular": {
            "price":      round(regular_price, 4) if regular_price else None,
            "open":       None,
            "prev_close": round(prev_close, 4)    if prev_close   else None,
        },
        "pre":  {"price": None, "change": None, "change_pct": None, "time": None, "available": False},
        "post": {"price": None, "change": None, "change_pct": None, "time": None, "available": False},
        "low_volume_warning": False,
        "extended_price":     round(extended_price, 4),
        "extended_timestamp": latest_time,
    }
