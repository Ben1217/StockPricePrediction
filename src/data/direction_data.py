"""
Daily-bar loading for the next-day direction classifier.

Why this module exists rather than reusing ``download_stock_data``:

**Price-basis consistency.** Yahoo returns split-adjusted OHLCV plus a separate
``Adj Close`` that also carries the dividend adjustment. Using ``Adj Close`` for
returns while ``High``/``Low``/``Open`` stay unadjusted is silently wrong on
every ex-dividend date: ``High_Low_Range`` is then a raw range divided by an
adjusted close, and the overnight gap ``Open / Prev_Close - 1`` picks up the
whole dividend as if it were a price move. The error is small per event and
systematically signed, which is exactly the kind of thing a classifier will
happily learn and a backtest will never earn.

This module removes the choice. It downloads unadjusted bars with
``auto_adjust=False, actions=True``, then multiplies ``Open/High/Low/Close`` by
the same ``Adj Close / Close`` ratio, so every column downstream sits on one
fully-adjusted basis and ``Adj Close`` no longer exists to be mixed in. Volume
is left alone: Yahoo already returns it split-adjusted, and a dividend does not
change share count. (This is the same arithmetic ``yfinance`` applies for
``auto_adjust=True``; it is done here so the raw frame, the adjustment factor,
and the corporate-action counts stay inspectable.)

**Reproducibility.** The raw pull is cached to Parquet (via
:mod:`src.data.ohlcv_cache`, which also supplies retry-with-backoff) and the
returned metadata carries a SHA-256 content hash of the adjusted frame. Two runs
that report the same hash saw the same bars; two that do not are not comparable,
whatever their accuracy numbers say.

Public API:
    load_daily_bars(ticker, start, end) -> BarLoad
    apply_dividend_adjustment(df) -> DataFrame
    clean_daily_bars(df) -> DataFrame
    frame_content_hash(df) -> str
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .ohlcv_cache import cached_download

logger = get_logger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]

# The 60-day momentum window is the longest look-back in the direction feature
# set, so it burns ~60 leading rows. The floor is not set by that warm-up but by
# statistical power: a 3pp accuracy edge has a standard error of ~1.6pp at 1000
# observations and ~3.1pp at 252, so below ~1000 rows a real edge and a lucky
# one are indistinguishable (see src.models.direction_metrics).
MIN_USABLE_ROWS = 1000

DateLike = Union[str, date, datetime, pd.Timestamp]


@dataclass
class BarLoad:
    """Adjusted daily bars plus the provenance needed to reproduce them."""

    frame: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frame)


def _as_date_string(value: Optional[DateLike]) -> str:
    if value is None:
        return ""
    return str(pd.Timestamp(value).date())


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the MultiIndex columns yfinance returns and drop duplicate labels."""
    if isinstance(df.columns, pd.MultiIndex):
        # Single-ticker frames put the field name on level 0 and the ticker on
        # level 1; taking level 0 leaves "Open"/"High"/... regardless.
        df.columns = df.columns.get_level_values(0)
    if df.columns.duplicated().any():
        logger.warning("Dropping duplicate columns from downloaded frame: %s", list(df.columns))
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def _download_raw(ticker: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """One unadjusted, action-carrying pull from Yahoo. Retries live in the cache layer."""
    import yfinance as yf  # lazy so the module imports without the network stack present

    frame = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,   # keep Close raw so the adjustment factor is recoverable
        actions=True,        # Dividends / Stock Splits, used for the audit counts
        progress=False,
    )
    if frame is None or frame.empty:
        logger.warning("No rows returned for %s over [%s, %s]", ticker, start, end)
        return None
    return _flatten_columns(frame)


def apply_dividend_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put every price column on the ``Adj Close`` basis and drop ``Adj Close``.

    ``ratio = Adj Close / Close`` is the cumulative dividend factor (splits are
    already reflected in the raw prices Yahoo returns). Multiplying all four
    price columns by it leaves the intraday geometry untouched — ``High/Low``,
    ``Close/Open`` and ``(Close-Low)/(High-Low)`` are ratios, so the factor
    cancels — while making close-to-close returns total-return returns.

    A frame without ``Adj Close`` is returned unchanged: it is already on a
    single basis, whichever that is.
    """
    data = df.copy()
    if "Adj Close" not in data.columns:
        return data

    close = pd.to_numeric(data["Close"], errors="coerce")
    adj_close = pd.to_numeric(data["Adj Close"], errors="coerce")
    ratio = adj_close / close.replace(0, np.nan)

    # A missing or non-positive factor means the pair cannot define an
    # adjustment; 1.0 leaves that bar on the raw basis, and the row is dropped
    # downstream if the prices themselves are unusable. Yahoo occasionally
    # returns a null Adj Close on the most recent bar.
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0), 1.0)

    for col in PRICE_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce") * ratio

    return data.drop(columns=["Adj Close"])


def clean_daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hygiene pass: tz-naive normalised index, deduped, sorted, no degenerate bars.

    Dropped rows:
      - any non-finite OHLCV value
      - non-positive prices (a zero close makes every ratio infinite)
      - zero-volume sessions, which on Yahoo are stale placeholder bars whose
        OHLC repeats the previous close and whose "return" is a fabricated zero
      - bars where ``High < Low`` or the close sits outside ``[Low, High]``

    Dropping mid-series rows leaves a calendar gap. That is the correct trade:
    the alternative is keeping a bar the exchange never printed and letting the
    classifier learn from it. Features that reach back one row (``Prev_Close``)
    simply span the gap, exactly as they span a weekend.
    """
    data = df.copy()
    if data.empty:
        return data

    data.index = pd.to_datetime(data.index, errors="coerce")
    data = data[data.index.notna()]
    if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    data.index = data.index.normalize()
    data.index.name = "Date"

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()

    present = [col for col in OHLCV_COLUMNS if col in data.columns]
    for col in present:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    if present:
        data = data.dropna(subset=present)

    for col in [c for c in PRICE_COLUMNS if c in data.columns]:
        data = data[data[col] > 0]

    if "Volume" in data.columns:
        data = data[data["Volume"] > 0]

    if {"High", "Low"}.issubset(data.columns):
        data = data[data["High"] >= data["Low"]]
    if {"High", "Low", "Close"}.issubset(data.columns):
        tol = 1e-9
        data = data[(data["Close"] <= data["High"] + tol) & (data["Close"] >= data["Low"] - tol)]

    return data


def frame_content_hash(df: pd.DataFrame) -> str:
    """
    Deterministic SHA-256 over a frame's index, column names, and values.

    Values are rounded to 10 decimals before hashing so a frame that survives a
    Parquet round-trip hashes to the same digest; prices are O(1e3) at most, so
    float64 retains far more precision than that.
    """
    if df is None or df.empty:
        return hashlib.sha256(b"empty").hexdigest()

    digest = hashlib.sha256()
    digest.update("|".join(str(c) for c in df.columns).encode("utf-8"))
    index_values = pd.DatetimeIndex(df.index).asi8
    digest.update(np.ascontiguousarray(index_values, dtype=np.int64).tobytes())
    values = df.to_numpy(dtype=np.float64, copy=True, na_value=np.nan)
    digest.update(np.ascontiguousarray(np.round(values, 10), dtype=np.float64).tobytes())
    return digest.hexdigest()


def load_daily_bars(
    ticker: str,
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    *,
    use_cache: bool = True,
    min_rows: int = MIN_USABLE_ROWS,
    require_min_rows: bool = True,
    downloader=None,
) -> BarLoad:
    """
    Fetch, adjust, and clean daily bars for one ticker.

    Parameters
    ----------
    ticker : str
    start, end : date-like, optional
        ``end`` defaults to today; ``start`` defaults to 8 calendar years back,
        which clears ``min_rows`` usable sessions after the indicator warm-up
        with room for holidays.
    use_cache : bool
        Serve the raw pull from the on-disk Parquet cache when a fresh entry
        exists.
    min_rows : int
        Minimum usable rows required after cleaning.
    require_min_rows : bool
        Raise when the history is shorter than ``min_rows``. Set False only for
        exploratory runs; the resulting accuracy numbers will not be separable
        from noise.
    downloader : callable, optional
        Injected by tests. Called as ``downloader()``; must return a raw
        yfinance-shaped frame or None.

    Returns
    -------
    BarLoad
        ``frame`` holds fully-adjusted OHLCV; ``meta`` holds provenance (content
        hash, row counts, corporate-action counts, yfinance version).

    Raises
    ------
    ValueError
        When the download is empty, is missing required columns, or is shorter
        than ``min_rows`` while ``require_min_rows`` is set.
    """
    symbol = str(ticker).upper().strip()
    end_ts = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()
    start_ts = pd.Timestamp(start) if start is not None else end_ts - timedelta(days=365 * 10)
    start_str, end_str = _as_date_string(start_ts), _as_date_string(end_ts)

    fetch = downloader if downloader is not None else (lambda: _download_raw(symbol, start_str, end_str))
    raw = cached_download(symbol, start_str, end_str, "1d", fetch, use_cache=use_cache)
    if raw is None or raw.empty:
        raise ValueError(f"No daily bars available for {symbol} over [{start_str}, {end_str}]")

    raw = _flatten_columns(raw.copy())
    raw_rows = len(raw)

    dividends = int((pd.to_numeric(raw.get("Dividends", 0), errors="coerce").fillna(0) > 0).sum())
    splits = int((pd.to_numeric(raw.get("Stock Splits", 0), errors="coerce").fillna(0) > 0).sum())
    had_adj_close = "Adj Close" in raw.columns

    adjusted = apply_dividend_adjustment(raw)
    adjusted = adjusted[[col for col in OHLCV_COLUMNS if col in adjusted.columns]]
    frame = clean_daily_bars(adjusted)

    missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Downloaded bars for {symbol} are missing required columns: {missing}")

    if require_min_rows and len(frame) < min_rows:
        raise ValueError(
            f"{symbol} returned {len(frame)} usable daily rows over [{start_str}, {end_str}], "
            f"below the {min_rows}-row floor. Widen the date range: after the indicator "
            f"warm-up a shorter history cannot support a statistically separable result."
        )

    try:
        import yfinance as yf
        yf_version = getattr(yf, "__version__", "unknown")
    except Exception:  # noqa: BLE001 - version reporting must never break a load
        yf_version = "unavailable"

    meta: Dict[str, Any] = {
        "ticker": symbol,
        "requested_start": start_str,
        "requested_end": end_str,
        "first_bar": _as_date_string(frame.index[0]) if len(frame) else None,
        "last_bar": _as_date_string(frame.index[-1]) if len(frame) else None,
        "raw_rows": raw_rows,
        "clean_rows": int(len(frame)),
        "dropped_rows": int(raw_rows - len(frame)),
        "price_basis": "dividend_and_split_adjusted" if had_adj_close else "as_returned",
        "adjustment_applied": bool(had_adj_close),
        "dividend_events": dividends,
        "split_events": splits,
        "content_sha256": frame_content_hash(frame),
        "yfinance_version": yf_version,
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }

    logger.info(
        "Loaded %s: %s clean rows [%s..%s], basis=%s, %s dividends / %s splits, sha256=%s",
        symbol, meta["clean_rows"], meta["first_bar"], meta["last_bar"],
        meta["price_basis"], dividends, splits, meta["content_sha256"][:12],
    )
    return BarLoad(frame=frame, meta=meta)
