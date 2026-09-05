"""
Shared OHLCV fetching and caching.

Every router previously carried its own near-identical yfinance fetcher with its own
interval→period table, MultiIndex flattening, and empty-frame fallback. They drifted.
This module is the single implementation, backed by one bounded, TTL'd cache so a
symbol fetched by the prices route is not re-fetched by the sentiment or patterns route.

The cache is bounded (`maxsize`) rather than an open dict: the previous hand-rolled
dict only evicted an entry when it was read after expiry, so any key never requested
again leaked for the lifetime of the process.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

import pandas as pd
import yfinance as yf
from cachetools import TTLCache

from src.data.live_data import get_cache_ttl, is_market_open
from src.data.ohlcv_cache import YF_DOWNLOAD_LOCK, normalize_ohlcv_frame, ticker_variants

logger = logging.getLogger(__name__)

# One entry per (symbol, interval, window). ~600 frames is a few hundred MB worst case
# for daily data and comfortably covers the S&P 500 across the timeframes in use.
MAX_CACHED_FRAMES = 600

# TTL is short while the market is open and long while it is closed; cachetools fixes
# the TTL at construction, so two caches are kept and selected per call.
_OPEN_TTL = 60
_CLOSED_TTL = 3600

_lock = threading.Lock()
_cache_open: TTLCache = TTLCache(maxsize=MAX_CACHED_FRAMES, ttl=_OPEN_TTL)
_cache_closed: TTLCache = TTLCache(maxsize=MAX_CACHED_FRAMES, ttl=_CLOSED_TTL)

# yfinance's `download` is NOT thread-safe: concurrent calls share internal state and
# occasionally return a frame containing two symbols' columns merged together (duplicate
# 'Close', 'Open', ... columns), which then blows up downstream as
# "cannot convert the series to <class 'float'>".
#
# This was latent while every route handler was `async def` — the event loop serialised
# these calls. Now that handlers run in a threadpool they genuinely overlap, so the
# network call itself is serialised here. The cache absorbs most requests, and the point
# of the threadpool is that the event loop stays free, not that Yahoo is hit in parallel.
# The lock lives in ohlcv_cache so that this module, the on-disk cache and every route
# that reaches yfinance share ONE lock object. Two separate locks would each serialise
# their own callers and still let the two groups race against each other.
_download_lock = YF_DOWNLOAD_LOCK

# Exported so routes that call yf.download directly (the batch quotes endpoint) share it.
download_lock = _download_lock

# Interval → yfinance `period` when a date range is not supplied. Yahoo caps intraday
# history hard (7d for 1m, 60d for 5m/15m), so these are ceilings, not preferences.
INTERVAL_PERIODS = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "1h": "730d",
    "4h": "730d",
    "1d": "1y",
    "1wk": "max",
    "1mo": "max",
}

# Used when the primary request comes back empty for an otherwise valid symbol.
FALLBACK_PERIODS = {
    "1d": "1y",
    "1wk": "max",
    "1mo": "max",
}


def _active_cache() -> TTLCache:
    """Pick the short or long TTL cache based on current market status."""
    return _cache_open if is_market_open() else _cache_closed


def cache_get(key: str) -> Optional[pd.DataFrame]:
    with _lock:
        return _active_cache().get(key)


def cache_set(key: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    with _lock:
        _active_cache()[key] = df


def cache_clear() -> None:
    """Drop everything. Used by tests and the /data/cache admin path."""
    with _lock:
        _cache_open.clear()
        _cache_closed.clear()


def cache_stats() -> dict:
    with _lock:
        return {
            "market_open": is_market_open(),
            "active_ttl_seconds": get_cache_ttl(),
            "entries_open_cache": len(_cache_open),
            "entries_closed_cache": len(_cache_closed),
            "maxsize": MAX_CACHED_FRAMES,
        }


def _normalise(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Flatten yfinance's MultiIndex columns and drop rows with no close price.

    Delegates to the shared normalizer, which picks the field level by name and — when
    a concurrent download has merged a second symbol in — keeps `symbol`'s columns
    rather than whichever happened to sort first.
    """
    cleaned = normalize_ohlcv_frame(df, symbol)
    if cleaned is None:
        return pd.DataFrame()
    return cleaned.dropna(subset=["Close"])


def fetch_ohlcv(
    symbol: str,
    interval: str = "1d",
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV bars for one symbol.

    Pass `start`/`end` for an explicit date range (daily and above only — Yahoo rejects
    ranges for most intraday intervals), or `period` to override the interval default.
    Returns an empty DataFrame rather than raising when the symbol has no data, so
    callers decide whether that is a 404 or an empty panel.
    """
    symbol = symbol.upper().strip()
    resolved_period = period or INTERVAL_PERIODS.get(interval, "1y")
    use_range = bool(start and end) and interval in ("1d", "1wk", "1mo")

    key = (
        f"ohlcv:{symbol}:{interval}:{start}:{end}"
        if use_range
        else f"ohlcv:{symbol}:{interval}:{resolved_period}"
    )

    if use_cache:
        cached = cache_get(key)
        if cached is not None:
            return cached

    def _download(spelling: str) -> pd.DataFrame:
        try:
            with _download_lock:
                if use_range:
                    return yf.download(spelling, start=start, end=end, interval=interval, progress=False)
                return yf.download(spelling, period=resolved_period, interval=interval, progress=False)
        except Exception as exc:
            logger.warning("yfinance fetch failed for %s at %s: %s", spelling, interval, exc)
            return pd.DataFrame()

    # `ticker_variants` is the same class-share fallback the prediction path
    # uses: the index prints BRK.B and BF.B, Yahoo indexes them as BRK-B and
    # BF-B, and this is the route the "Search Any Ticker" box validates against
    # — so without it a ticker whose forecast serves fine is rejected as
    # nonexistent before it can ever be added to a watchlist.
    df = pd.DataFrame()
    resolved = symbol
    for spelling in ticker_variants(symbol):
        df = _download(spelling)
        # Yahoo intermittently returns an empty frame for valid symbols; retry
        # once wider before moving on to the next spelling, so a transient
        # empty response is not misread as "wrong spelling".
        if df.empty:
            fallback = FALLBACK_PERIODS.get(interval)
            if fallback:
                try:
                    with _download_lock:
                        df = yf.download(spelling, period=fallback, interval=interval, progress=False)
                except Exception as exc:
                    logger.error("Fallback fetch failed for %s at %s: %s", spelling, interval, exc)
                    df = pd.DataFrame()
        if not df.empty:
            resolved = spelling
            if spelling != symbol:
                logger.info("Resolved %s as %s", symbol, spelling)
            break

    # Normalised against the spelling that answered — the frame's MultiIndex
    # carries that name, not the one the caller asked for.
    df = _normalise(df, resolved)
    if use_cache:
        cache_set(key, df)
    return df
