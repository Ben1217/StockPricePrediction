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
import re
import threading
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


# ══════════════════════════════════════════════════════════════════════════════
# THREAD-SAFE DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════
#
# ``yfinance.download`` is not thread-safe. It stores every result in module-level
# globals and resets them on entry::
#
#     shared._DFS = {}                       # cleared by EVERY caller
#     while len(shared._DFS) < len(tickers): ...
#     data = pd.concat(shared._DFS.values(), axis=1, keys=shared._DFS.keys(), ...)
#
# Two overlapping calls therefore share one dict, and each ends up concatenating the
# other's ticker into its own result. The caller for XOM gets a frame whose columns are
# ``[('Close','NVDA'), ('Close','XOM'), ...]``; flattening that to the first level leaves
# two 'Close' columns, so ``df["Close"]`` is a DataFrame rather than a Series and the
# next ``pd.DataFrame(...)`` raises "Data must be 1-dimensional" — an uncaught 500 for
# every request in flight. The same crossover silently files one symbol's bars under
# another's cache key, which is how a correlation matrix reported XOM and NVDA at 1.00.
#
# This was harmless while route handlers were ``async def``, because the event loop
# serialised them. FastAPI runs plain ``def`` handlers in a threadpool, so they now
# genuinely overlap. One process-wide lock is the fix; the disk cache absorbs the
# repeat asks, so serialising the network call costs little.
YF_DOWNLOAD_LOCK = threading.RLock()

_OHLCV_FIELDS = {"open", "high", "low", "close", "adj close", "volume", "dividends", "stock splits"}


def _pick_field_level(columns: pd.MultiIndex) -> int:
    """Return the level of `columns` holding OHLCV field names, preferring level names."""
    names = [str(n).lower() if n is not None else "" for n in columns.names]
    for level, name in enumerate(names):
        if name == "price":
            return level
    for level, name in enumerate(names):
        if name == "ticker":
            return 1 - level if columns.nlevels == 2 else level
    # Unnamed levels: the field level is the one whose values look like OHLCV fields.
    for level in range(columns.nlevels):
        values = {str(v).lower() for v in columns.get_level_values(level)}
        if values & _OHLCV_FIELDS:
            return level
    return 0


def normalize_ohlcv_frame(df: Optional[pd.DataFrame], ticker: str) -> Optional[pd.DataFrame]:
    """
    Reduce a yfinance frame to plain single-level OHLCV columns for one ticker.

    Handles the shapes that reach us in practice: a flat frame, a
    ``(Price, Ticker)`` MultiIndex, the transposed ``(Ticker, Price)`` from
    ``group_by='ticker'``, and — the one that used to crash — a frame carrying a
    second symbol's columns because a concurrent download bled into it. Returns
    None when nothing usable is left, so callers can treat it like an empty frame.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None

    df = df.copy()
    want = str(ticker).upper()

    if isinstance(df.columns, pd.MultiIndex):
        field_level = _pick_field_level(df.columns)
        other_levels = [lvl for lvl in range(df.columns.nlevels) if lvl != field_level]
        # Keep only this ticker's columns when the frame carries several symbols.
        for lvl in other_levels:
            values = [str(v).upper() for v in df.columns.get_level_values(lvl)]
            if want in values:
                df = df.loc[:, [v == want for v in values]]
                break
        df.columns = df.columns.get_level_values(field_level)

    df.columns = [str(c) for c in df.columns]

    # A merged frame can still leave duplicate field names; the first is this
    # ticker's after the selection above, and dropping the rest keeps every
    # column one-dimensional.
    if len(set(df.columns)) != len(df.columns):
        logger.warning(
            "Duplicate columns %s in frame for %s; keeping the first of each",
            [c for c in df.columns if list(df.columns).count(c) > 1][:4], want,
        )
        df = df.loc[:, ~pd.Index(df.columns).duplicated()]

    df = _normalize_index(df, want)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df


def _normalize_index(df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
    """Sort the index, drop tz, and collapse repeated timestamps to one row each."""
    try:
        idx = pd.DatetimeIndex(df.index)
    except (TypeError, ValueError):
        logger.warning("Non-datetime index for %s; leaving it as-is", ticker)
        return df[~df.index.duplicated(keep="last")].sort_index()

    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = idx
    df = df.sort_index()
    if df.index.has_duplicates:
        # Duplicate labels make every later pd.DataFrame({...}) alignment raise
        # "cannot reindex on an axis with duplicate labels".
        logger.warning("Duplicate timestamps in frame for %s; keeping the last of each", ticker)
        df = df[~df.index.duplicated(keep="last")]
    return df


#: A US class share is written with a dot on the exchange and with a dash on
#: Yahoo: the S&P 500 constituent table lists BRK.B and BF.B, and yfinance
#: answers both with "possibly delisted; no timezone found". The dot form
#: reaches us from the index list, from a saved watchlist and from anyone who
#: types the ticker the way it is printed, so it has to resolve.
_CLASS_SHARE_SUFFIX = re.compile(r"^([A-Z0-9]{1,6})\.([A-Z])$")


def ticker_variants(ticker: str) -> list[str]:
    """
    ``ticker`` first, then the spellings Yahoo may know it by instead.

    The dash form is a FALLBACK and never a rewrite. A trailing dot is also how
    Yahoo names a foreign listing's exchange -- SHEL.L, RY.TO, 0700.HK -- and
    those single-letter suffixes are indistinguishable from a share class by
    shape alone. Rewriting up front would break symbols that resolve today;
    trying the dot form first and the dash form only once it comes back empty
    costs one extra request on a symbol that already had none.
    """
    want = str(ticker).upper().strip()
    variants = [want]
    match = _CLASS_SHARE_SUFFIX.match(want)
    if match:
        variants.append(f"{match.group(1)}-{match.group(2)}")
    return variants


def safe_yf_download(ticker: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Download one ticker's bars under `YF_DOWNLOAD_LOCK` and normalize the result.

    Every ``yf.download`` in this process must go through here (or hold the same
    lock); a single unguarded caller is enough to corrupt everybody else's frame.

    Falls back through :func:`ticker_variants` when a spelling returns nothing,
    so a class share reaches the provider in the form it indexes.
    """
    import yfinance as yf

    kwargs.setdefault("progress", False)
    for symbol in ticker_variants(ticker):
        with YF_DOWNLOAD_LOCK:
            raw = yf.download(symbol, **kwargs)
        # Normalized against the spelling actually requested: the frame's
        # MultiIndex carries that name, not the one the caller asked for.
        frame = normalize_ohlcv_frame(raw, symbol)
        if frame is not None and not frame.empty:
            if symbol != str(ticker).upper().strip():
                logger.info("Resolved %s as %s", ticker, symbol)
            return frame
    return None
