"""
On-disk OHLCV cache for Yahoo Finance downloads.

Every training run and every prediction request used to issue its own
``yf.download``. Retraining one symbol across 4 horizons x 3 models is a dozen
identical requests for the same five years of daily bars, which is what earns
the 429s. This module keeps the bars on disk in Parquet and serves repeat asks
from there.

Freshness rule: daily bars for a closed session never change, so a cache entry
is stale only when the calendar has moved past the last session it covers.
Entries are therefore keyed by ticker+range+interval and carry the timestamp
they were written; an entry younger than ``ttl_seconds`` is served as-is.

Public API:
    cached_download(ticker, start, end, interval, downloader) -> DataFrame | None
    OHLCVCache.load / store / clear
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache/ohlcv")
# Daily bars settle once a session closes; a few hours keeps intraday reruns off
# the wire without ever serving a stale close for a session that has since ended.
DEFAULT_TTL_SECONDS = 6 * 3600
# Yahoo sometimes answers a multi-year request with only its most recent months.
# Such a frame is not corrupt, just short, so it is cached under a much tighter
# TTL: refusing to cache it at all would hammer the API for tickers whose
# history really is short (a recent IPO), while giving it the full TTL is what
# wedged NVDA predictions behind a 422 for hours -- 153 rows were served from
# disk for a window the caller needed 260 rows from.
PARTIAL_TTL_SECONDS = 15 * 60
# Sessions returned vs. sessions the requested window should hold. The bar sits
# low on purpose: a ticker that listed midway through the window is legitimately
# short (PLTR answers a 10-year ask with ~59% coverage), while the truncations
# seen in practice came back at 12-35%.
MIN_COVERAGE_RATIO = 0.5
_SESSIONS_PER_CALENDAR_DAY = 252 / 365.25


def expected_sessions(start: str, end: str) -> Optional[float]:
    """Approximate trading sessions in [start, end], or None if unparseable."""
    try:
        span_days = (pd.Timestamp(end) - pd.Timestamp(start)).days
    except Exception:
        return None
    if span_days <= 0:
        return None
    return span_days * _SESSIONS_PER_CALENDAR_DAY


def is_partial_frame(df: Optional[pd.DataFrame], start: str, end: str) -> bool:
    """True when `df` covers materially less of [start, end] than it should."""
    if df is None or df.empty:
        return False
    expected = expected_sessions(start, end)
    if not expected:
        return False
    return len(df) < MIN_COVERAGE_RATIO * expected


class OHLCVCache:
    """Parquet-backed cache for OHLCV frames."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.cache_dir = Path(cache_dir)
        self.ttl_seconds = int(ttl_seconds)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(ticker: str, start: str, end: str, interval: str) -> str:
        raw = f"{str(ticker).upper()}|{start}|{end}|{interval}"
        digest = hashlib.md5(raw.encode()).hexdigest()[:16]
        # Ticker in the filename keeps the cache dir readable when debugging.
        safe_ticker = "".join(c for c in str(ticker).upper() if c.isalnum() or c in "-_") or "TICKER"
        return f"{safe_ticker}_{interval}_{digest}"

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.parquet", self.cache_dir / f"{key}.meta.json"

    def load(self, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
        """Return the cached frame when present and fresh, else None."""
        data_path, meta_path = self._paths(self._key(ticker, start, end, interval))
        if not data_path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text())
            age = time.time() - float(meta["written_at_epoch"])
            ttl = self.ttl_seconds
            if meta.get("partial"):
                # A short frame is re-checked soon, so a truncated response
                # cannot hold the slot for the whole normal TTL.
                ttl = min(ttl, PARTIAL_TTL_SECONDS)
            # A ttl of zero means "never serve from cache", so compare
            # inclusively rather than letting a zero-age entry through.
            if ttl <= 0 or age > ttl:
                logger.debug("OHLCV cache expired for %s (age %.0fs)", ticker, age)
                return None
            df = pd.read_parquet(data_path)
            if df.empty:
                return None
            logger.info(
                "OHLCV cache hit for %s [%s..%s] %s — %s rows, age %.0fs (no network call)",
                ticker, start, end, interval, len(df), age,
            )
            return df
        except Exception as exc:
            logger.warning("OHLCV cache read failed for %s: %s", ticker, exc)
            return None

    def store(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        df: pd.DataFrame,
        *,
        partial: bool = False,
    ) -> None:
        if df is None or df.empty:
            return
        data_path, meta_path = self._paths(self._key(ticker, start, end, interval))
        try:
            df.to_parquet(data_path)
            meta_path.write_text(json.dumps({
                "ticker": str(ticker).upper(),
                "start": str(start),
                "end": str(end),
                "interval": interval,
                "rows": int(len(df)),
                "partial": bool(partial),
                "written_at": datetime.now().isoformat(),
                "written_at_epoch": time.time(),
            }, indent=2))
        except Exception as exc:
            logger.warning("OHLCV cache write failed for %s: %s", ticker, exc)

    def clear(self) -> int:
        removed = 0
        for path in self.cache_dir.glob("*"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed


_cache_instance: Optional[OHLCVCache] = None


def get_ohlcv_cache() -> OHLCVCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = OHLCVCache()
    return _cache_instance


def cached_download(
    ticker: str,
    start: str,
    end: str,
    interval: str,
    downloader: Callable[[], Optional[pd.DataFrame]],
    *,
    use_cache: bool = True,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Serve `ticker` from the disk cache, else call `downloader` and cache the result.

    On a rate-limit error the call is retried with exponential backoff. If every
    attempt fails, an expired cache entry is preferred over no data at all: a
    slightly old frame beats failing a training run outright.
    """
    cache = get_ohlcv_cache()
    if use_cache:
        hit = cache.load(ticker, start, end, interval)
        if hit is not None:
            return hit

    delay = 2.0
    last_exc: Optional[Exception] = None
    # Longest frame any attempt produced, kept in case they are all short.
    best_short: Optional[pd.DataFrame] = None
    for attempt in range(1, max_retries + 1):
        try:
            df = downloader()
            if df is not None and not df.empty:
                if not is_partial_frame(df, start, end):
                    cache.store(ticker, start, end, interval, df)
                    return df
                # Short frame: either the response was truncated or the ticker
                # genuinely has little history. Another attempt tells us which,
                # and caching it now would serve the truncation for hours.
                if best_short is None or len(df) > len(best_short):
                    best_short = df
                last_exc = ValueError(f"partial frame: {len(df)} rows")
                logger.warning(
                    "Download for %s [%s..%s] returned %d rows, well under the "
                    "~%.0f sessions the window should hold; not caching",
                    ticker, start, end, len(df), expected_sessions(start, end) or 0.0,
                )
            else:
                last_exc = ValueError("empty frame returned")
        except Exception as exc:  # noqa: BLE001 - provider raises many shapes
            last_exc = exc
            message = str(exc).lower()
            rate_limited = "429" in message or "too many requests" in message or "rate limit" in message
            if not rate_limited and attempt >= max_retries:
                break
            logger.warning(
                "Download for %s failed (attempt %d/%d): %s. Retrying in %.0fs",
                ticker, attempt, max_retries, exc, delay,
            )
        if attempt < max_retries:
            time.sleep(delay)
            delay *= 2

    stale = _load_ignoring_ttl(cache, ticker, start, end, interval)

    if best_short is not None:
        # Every attempt agreed the history is short, so treat it as real rather
        # than as a failure -- but a fuller frame already on disk still wins, so
        # one truncated response cannot evict five years of bars.
        if stale is not None and len(stale) > len(best_short):
            logger.warning(
                "Preferring STALE cached data for %s (%d rows) over a short fresh frame (%d rows)",
                ticker, len(stale), len(best_short),
            )
            return stale
        logger.warning(
            "Serving short frame for %s [%s..%s]: %d rows after %d attempts; caching for %ds only",
            ticker, start, end, len(best_short), max_retries, PARTIAL_TTL_SECONDS,
        )
        cache.store(ticker, start, end, interval, best_short, partial=True)
        return best_short

    if stale is not None:
        logger.warning(
            "Serving STALE cached data for %s after %d failed download attempts (last error: %s)",
            ticker, max_retries, last_exc,
        )
        return stale

    logger.error("Download for %s failed and no cache entry exists: %s", ticker, last_exc)
    return None


def _load_ignoring_ttl(cache: OHLCVCache, ticker: str, start: str, end: str, interval: str) -> Optional[pd.DataFrame]:
    data_path, _ = cache._paths(cache._key(ticker, start, end, interval))
    if not data_path.exists():
        return None
    try:
        df = pd.read_parquet(data_path)
        return df if not df.empty else None
    except Exception:
        return None
