"""
Data API routes — fetch prices, indicators, S&P 500, upload CSV.

Changes from original:
- Dynamic cache TTL: 60 s when market open, 3600 s when closed.
- _fetch_yfinance: uses period="1d" interval="1m" for intraday, always .iloc[-1].
- _fetch_alpha_vantage: uses GLOBAL_QUOTE (live) instead of a daily time series.
- All responses include market_open + data_timestamp metadata.
- Freshness validation via src.data.live_data.validate_freshness.
- New GET /quote/{symbol} lightweight live-price endpoint.
"""

import os
import io
import logging
import re
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import pytz
from cachetools import LRUCache
from fastapi import APIRouter, Query, UploadFile, File, HTTPException

from src.api.schemas.schemas import (
    PriceResponse, PriceBar, IndicatorResponse, SP500Response, UploadResponse
)
from src.features.technical_indicators import add_all_technical_indicators
from src.data.data_acquisition import get_sp500_tickers
from src.data.live_data import (
    is_market_open, get_market_session,
    validate_freshness, fetch_live_quote, fetch_extended_quote
)
from src.data.ohlcv import cache_get, cache_set, cache_stats, download_lock, fetch_ohlcv
from src.data.timescale_store import (
    load_daily_prices,
    save_daily_prices,
    timescale_enabled,
)
from src.data.alpha_vantage_provider import consume_request_slot
from src.data.provider_errors import (
    PremiumEndpointError,
    ProviderError,
    QuotaExceededError,
    classify_alpha_vantage_message,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_UTC = pytz.UTC

# ── Cache ─────────────────────────────────────────────────────────────────────
# Frame caching lives in src.data.ohlcv so every router shares one bounded store.
# Uploaded datasets are held in a bounded LRU: they are whole DataFrames, and an
# unbounded dict meant a long-lived server retained every CSV ever uploaded.
MAX_UPLOADED_DATASETS = 20
_uploaded_datasets: LRUCache = LRUCache(maxsize=MAX_UPLOADED_DATASETS)

# Upload guards. The filename is client-controlled and is used as a store key and
# echoed back, so it is reduced to a basename over an allow-listed character set.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]")


def _safe_upload_name(filename: Optional[str]) -> str:
    """Reduce a client-supplied filename to a safe basename."""
    raw = os.path.basename((filename or "").replace("\\", "/")).strip()
    cleaned = _SAFE_NAME_RE.sub("_", raw).lstrip(".")
    if not cleaned:
        raise HTTPException(400, "Invalid filename")
    return cleaned[:120]

_INTERVAL_DAY_LIMITS = {
    "1m": {"prices": (7, 7), "indicators": (60, 120)},
    "5m": {"prices": (30, 60), "indicators": (60, 120)},
    "15m": {"prices": (30, 60), "indicators": (60, 120)},
    "1h": {"prices": (180, 730), "indicators": (120, 240)},
    "4h": {"prices": (180, 730), "indicators": (120, 240)},
    "1d": {"prices": (30, 420), "indicators": (120, 320)},
    "1wk": {"prices": (730, 3650), "indicators": (120, 300)},
    "1mo": {"prices": (1825, 3650), "indicators": (120, 180)},
}


def _cache_get(key: str) -> Optional[pd.DataFrame]:
    """Thin alias kept so existing call sites read naturally."""
    return cache_get(key)


def _cache_set(key: str, df: pd.DataFrame) -> None:
    cache_set(key, df)


def _clamp_interval_days(interval: str, value: int, bucket: str) -> int:
    lower, upper = _INTERVAL_DAY_LIMITS.get(interval, _INTERVAL_DAY_LIMITS["1d"])[bucket]
    return min(max(value, lower), upper)


# ── Internal fetchers ─────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch OHLCV via the shared cached fetcher (src.data.ohlcv)."""
    return fetch_ohlcv(symbol, interval, start=start, end=end)


def _fetch_alpha_vantage_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch historical daily data from Alpha Vantage TIME_SERIES_DAILY.
    First row returned by the API is the LATEST — we sort ascending.
    """
    import requests
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "your_alpha_vantage_key":
        raise HTTPException(400, "Alpha Vantage API key not configured")

    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY"
        f"&symbol={symbol}&outputsize=compact&apikey={api_key}"
    )
    # Shares the app-wide Alpha Vantage limiter: spaces this call against the
    # free tier's per-second burst limit and counts it against the daily quota.
    consume_request_slot()
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # A throttled or quota-blocked call still returns HTTP 200 with the reason
    # under "Information"/"Note" and no time series. Without this check it lands
    # on the "no data" branch below and is reported as an unknown symbol.
    for key in ("Error Message", "Information", "Note"):
        if key in data:
            raise classify_alpha_vantage_message(data[key], endpoint="TIME_SERIES_DAILY")
    ts = data.get("Time Series (Daily)", {})
    if not ts:
        raise HTTPException(404, f"No Alpha Vantage data for {symbol}")

    rows = []
    for dt, vals in ts.items():
        rows.append({
            "Date": dt,
            "Open": float(vals["1. open"]),
            "High": float(vals["2. high"]),
            "Low": float(vals["3. low"]),
            "Close": float(vals["4. close"]),
            "Volume": int(vals["5. volume"]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort ascending — index[0] = oldest, index[-1] = LATEST (correct)
    df = df.sort_values("Date").set_index("Date")
    return df.loc[start:end]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _provider_http_error(exc: Exception, context: str) -> HTTPException:
    """
    Map a provider failure onto an honest status code.

    A premium-only endpoint and an exhausted quota are deterministic refusals,
    not upstream faults: 502 would both misdescribe them and mark them as
    retryable to clients that treat 5xx as transient.
    """
    if isinstance(exc, PremiumEndpointError):
        return HTTPException(501, str(exc))
    if isinstance(exc, QuotaExceededError):
        return HTTPException(429, str(exc))
    return HTTPException(502, f"{context}: {exc}")


def _df_to_bars(df: pd.DataFrame) -> list[PriceBar]:
    bars = []
    for dt, row in df.iterrows():
        # Handle datetime index natively for timezone-aware intraday
        format_date = str(dt.date()) if hasattr(dt, "date") and dt.time() == dt.time().replace(hour=0, minute=0, second=0) else str(dt)
        bars.append(PriceBar(
            date=format_date,
            open=round(float(row.get("Open", 0)), 4),
            high=round(float(row.get("High", 0)), 4),
            low=round(float(row.get("Low", 0)), 4),
            close=round(float(row.get("Close", 0)), 4),
            volume=int(row.get("Volume", 0)),
        ))
    return bars


def _get_data_timestamp(df: pd.DataFrame) -> datetime:
    """Return the timestamp of the last row, timezone-aware (UTC)."""
    last_idx = df.index[-1]
    if hasattr(last_idx, "to_pydatetime"):
        ts = last_idx.to_pydatetime()
    else:
        ts = datetime.strptime(str(last_idx), "%Y-%m-%d")
        # Daily bars — treat as market close (4pm ET = 20:00 UTC)
        ts = ts.replace(hour=20)
    if ts.tzinfo is None:
        ts = _UTC.localize(ts)
    return ts


def _daily_store_covers(df: pd.DataFrame, start: str, end: str) -> bool:
    """
    Whether the stored bars can answer a request outright.

    Matching the window exactly would never hold: the endpoints are calendar dates
    while the bars are trading days, so a Friday-to-Monday request has no bar on
    either edge. Both ends get enough slack to span a weekend plus a holiday;
    anything short of that falls through to the provider, which refreshes the
    hypertable on the way past.
    """
    if df is None or df.empty:
        return False
    slack = pd.Timedelta(days=5)
    return (
        df.index[0] <= pd.Timestamp(start) + slack
        and df.index[-1] >= pd.Timestamp(end) - slack
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/cache")
def get_cache_stats():
    """Inspect the shared OHLCV cache — entry counts, bound, and active TTL."""
    stats = cache_stats()
    stats["uploaded_datasets"] = len(_uploaded_datasets)
    return stats


@router.get("/sources")
def list_sources():
    """List available data sources."""
    sources = ["yfinance"]
    ak = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if ak and ak != "your_alpha_vantage_key":
        sources.append("alpha_vantage")
    uploads = list(_uploaded_datasets.keys())
    return {
        "sources": sources,
        "uploaded_datasets": uploads,
        "market_open": is_market_open(),
    }


MAX_BATCH_SYMBOLS = 100


@router.get("/quotes")
def get_batch_quotes(
    symbols: str = Query(..., description="Comma-separated ticker symbols, e.g. AAPL,MSFT,NVDA"),
):
    """
    Batch last-close and daily-change lookup for many symbols in one upstream call.

    Replaces N individual /prices requests from the ticker bar and sector heatmap.
    Unknown or delisted symbols are omitted from `quotes` rather than failing the
    whole request, so one bad ticker cannot blank the caller's grid.
    """
    requested = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    # dict.fromkeys de-duplicates while preserving request order.
    requested = list(dict.fromkeys(requested))
    if not requested:
        raise HTTPException(400, "At least one symbol is required")
    if len(requested) > MAX_BATCH_SYMBOLS:
        raise HTTPException(
            400, f"Too many symbols: {len(requested)} (max {MAX_BATCH_SYMBOLS})"
        )

    cache_key = f"batch-quotes:{','.join(sorted(requested))}"
    cached = _cache_get(cache_key)
    quotes: dict[str, dict] = {}

    if cached is not None:
        frame = cached
    else:
        try:
            # Shares the download lock with src.data.ohlcv — yfinance is not thread-safe
            # and concurrent downloads can return frames with merged columns.
            with download_lock:
                frame = yf.download(
                    tickers=requested,
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    progress=False,
                    threads=True,
                    auto_adjust=False,
                )
        except Exception as exc:
            logger.warning("Batch quote fetch failed for %s symbols: %s", len(requested), exc)
            raise HTTPException(502, f"Quote fetch failed: {exc}")
        if not frame.empty:
            _cache_set(cache_key, frame)

    for symbol in requested:
        try:
            # yfinance returns flat columns for a single ticker and a MultiIndex for many.
            if isinstance(frame.columns, pd.MultiIndex):
                if symbol not in frame.columns.get_level_values(0):
                    continue
                closes = frame[symbol]["Close"].dropna()
                volumes = frame[symbol].get("Volume")
            else:
                closes = frame["Close"].dropna()
                volumes = frame.get("Volume")
            if len(closes) < 1:
                continue
            last = float(closes.iloc[-1])
            prev = float(closes.iloc[-2]) if len(closes) >= 2 else last
            change_pct = ((last - prev) / prev * 100.0) if prev else 0.0
            volume = 0
            if volumes is not None and len(volumes.dropna()) > 0:
                volume = int(volumes.dropna().iloc[-1])
            quotes[symbol] = {
                "symbol": symbol,
                "price": round(last, 4),
                "previous_close": round(prev, 4),
                "change_pct": round(change_pct, 4),
                "volume": volume,
            }
        except Exception as exc:  # one malformed symbol must not fail the batch
            logger.debug("Skipping %s in batch quotes: %s", symbol, exc)

    return {
        "quotes": quotes,
        "requested": len(requested),
        "returned": len(quotes),
        "missing": [s for s in requested if s not in quotes],
        "market_open": is_market_open(),
    }


@router.get("/quote/{symbol}")
def get_live_quote(
    symbol: str,
    source: str = Query("yfinance", enum=["yfinance", "alpha_vantage"]),
):
    """
    Lightweight endpoint — returns only the latest live price + metadata.
    Uses fast_info (yfinance) or GLOBAL_QUOTE (Alpha Vantage).
    """
    symbol = symbol.upper()
    try:
        result = fetch_live_quote(symbol, source=source)
        return result
    except Exception as e:
        raise _provider_http_error(e, f"Live quote fetch failed for {symbol}")


@router.get("/extended-quote/{symbol}")
def get_extended_quote(
    symbol: str,
    source: str = Query("yfinance", enum=["yfinance", "alpha_vantage"]),
):
    """
    Returns pre-market, regular, and post-market prices for a symbol.
    Detects which session is currently active and adds a low_volume_warning
    when extended-hours volume is thin.
    """
    symbol = symbol.upper()
    try:
        result = fetch_extended_quote(symbol, source=source)
        return result
    except Exception as e:
        raise _provider_http_error(e, f"Extended quote fetch failed for {symbol}")


@router.get("/prices/{symbol}", response_model=PriceResponse)
def get_prices(
    symbol: str,
    source: str = Query("yfinance", enum=["yfinance", "alpha_vantage"]),
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"]),
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = Query(120, ge=1, le=20000),
):
    """Fetch OHLCV price data for a symbol."""
    symbol = symbol.upper()
    days = _clamp_interval_days(interval, days, "prices")
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    # Read-through TimescaleDB for daily bars, when DB_TYPE selects it. The
    # hypertable is the store of record: a covered window skips the provider
    # entirely, and whatever a miss fetches is written back. Limited to daily
    # yfinance requests because `daily_prices` is a daily table — intraday belongs
    # in `intraday_prices`, which nothing populates yet.
    use_store = timescale_enabled() and source == "yfinance" and interval == "1d"
    df = None
    served_from = source

    if use_store:
        try:
            stored = load_daily_prices(symbol, start, end)
            if _daily_store_covers(stored, start, end):
                df = stored
                served_from = "timescaledb"
                logger.info("Served %s (%s..%s) from TimescaleDB: %d bars", symbol, start, end, len(df))
        except Exception as e:
            # A database problem must not take the endpoint down — the provider
            # path below is still a complete answer.
            logger.warning("TimescaleDB read failed for %s, falling back to provider: %s", symbol, e)

    if df is None:
        try:
            if source == "alpha_vantage":
                df = _fetch_alpha_vantage_history(symbol, start, end)
            else:
                df = _fetch_yfinance(symbol, start, end, interval=interval)
        except HTTPException:
            raise
        except ProviderError as e:
            raise _provider_http_error(e, "Data fetch failed")
        except Exception as e:
            raise HTTPException(500, f"Data fetch failed: {e}")

        if use_store and not df.empty:
            try:
                save_daily_prices(symbol, df)
            except Exception as e:
                # Persistence is a side effect of answering; failing it must not
                # discard bars the caller already has.
                logger.warning("TimescaleDB write failed for %s: %s", symbol, e)

    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    # Freshness check — warn but don't hard-fail (market may be closed)
    data_ts = _get_data_timestamp(df)
    stale_warning = None
    try:
        validate_freshness(data_ts)
    except ValueError as e:
        stale_warning = str(e)
        logger.warning(stale_warning)

    # Deduplicate index — yfinance can return duplicate timestamps
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    bars = _df_to_bars(df)
    return PriceResponse(
        symbol=symbol,
        source=served_from,
        bars=bars,
        count=len(bars),
    )


@router.get("/indicators/{symbol}", response_model=IndicatorResponse)
def get_indicators(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"]),
    days: int = Query(120, ge=1, le=20000),
):
    """Compute technical indicators for a symbol."""
    symbol = symbol.upper()
    days = _clamp_interval_days(interval, days, "indicators")
    end = datetime.now().strftime("%Y-%m-%d")
    padding_days = {
        "1d": 420,
        "1wk": 2400,
        "1mo": 5600,
    }.get(interval, 200)
    start = (datetime.now() - timedelta(days=days + padding_days)).strftime("%Y-%m-%d")

    df = _fetch_yfinance(symbol, start, end, interval=interval)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    if len(df) < 20:
        raise HTTPException(
            422,
            f"Insufficient candles for {interval} indicators. Need at least 20 bars, got {len(df)}."
        )

    try:
        df = add_all_technical_indicators(df)
    except Exception as e:
        logger.error(f"Indicator calculation failed for {symbol} at {interval}: {e}")
        raise HTTPException(502, f"Indicator calculation failed for {symbol} ({interval}).")
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    df = df.tail(days)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

    indicator_cols = [
        c for c in df.columns
        if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    ]
    data = []
    for dt, row in df.iterrows():
        format_date = str(dt.date()) if hasattr(dt, "date") and dt.time() == dt.time().replace(hour=0, minute=0, second=0) else str(dt)
        entry = {"date": format_date}
        for c in indicator_cols:
            v = row[c]
            if v is None or pd.isna(v) or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
                entry[c] = None
            else:
                entry[c] = round(float(v), 4)
        data.append(entry)

    return IndicatorResponse(
        symbol=symbol,
        indicators=indicator_cols,
        data=data,
        count=len(data),
    )


@router.get("/sp500", response_model=SP500Response)
def get_sp500():
    """
    The S&P 500 constituents, with the company name and GICS sector for each.

    This is what the client builds its stock picker from, so the name and the
    sector are part of the answer rather than a nicety: a list of 503 bare
    tickers cannot be filtered by sector or searched by company name, which is
    how anyone actually finds a stock in it.

    Both of those used to be dropped. ``get_sp500_constituents`` renames the
    scraped columns to Symbol / Company / Sector / Industry, and this handler
    selected the pre-rename ``Security`` and ``GICS Sector`` -- so every call
    raised KeyError into a bare ``except: pass``, fell through to the
    ticker-only path, and scraped Wikipedia a second time to return symbols
    alone. The endpoint answered 200 with a plausible count throughout, which
    is why it read as working.
    """
    try:
        from src.data.market_data import get_sp500_constituents
        df = get_sp500_constituents()
        if df is not None and not df.empty:
            frame = df.rename(columns={"Symbol": "symbol", "Company": "company", "Sector": "sector"})
            columns = [c for c in ("symbol", "company", "sector") if c in frame.columns]
            symbols = frame[columns].astype(str).to_dict("records")
            return SP500Response(symbols=symbols, count=len(symbols))
        logger.warning("S&P 500 constituent scrape returned nothing; falling back to tickers only")
    except Exception as exc:
        # Still a fallback rather than a 500 -- a ticker-only list is a usable
        # picker -- but it is no longer silent about why it degraded.
        logger.warning("Could not build the full S&P 500 constituent table: %s", exc, exc_info=True)
    tickers = get_sp500_tickers()
    return SP500Response(symbols=[{"symbol": t} for t in tickers], count=len(tickers))


@router.post("/upload", response_model=UploadResponse)
def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset for training / backtesting."""
    safe_name = _safe_upload_name(file.filename)
    if not safe_name.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    # Read in chunks with a hard ceiling — an unbounded read() let a single large
    # upload exhaust memory before pandas ever saw it.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = file.file.read(_UPLOAD_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                413, f"File too large: limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"
            )
        chunks.append(chunk)
    contents = b"".join(chunks)
    if not contents:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        df = pd.read_csv(io.BytesIO(contents), parse_dates=True, index_col=0)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse CSV: {e}")

    required = {"Open", "High", "Low", "Close", "Volume"}
    col_map = {c.lower(): c for c in df.columns}
    missing = required - {c.capitalize() for c in col_map}
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}. Required: {required}")

    df.columns = [c.capitalize() for c in df.columns]
    df.index.name = "Date"
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    _uploaded_datasets[safe_name] = df
    return UploadResponse(
        filename=safe_name,
        rows=len(df),
        columns=list(df.columns),
        date_range={"start": str(df.index.min().date()), "end": str(df.index.max().date())},
        message=f"Uploaded {safe_name}: {len(df)} rows",
    )


@router.get("/uploaded/{filename}")
def get_uploaded_data(filename: str, tail: int = Query(120, ge=1)):
    """Retrieve previously uploaded dataset."""
    if filename not in _uploaded_datasets:
        raise HTTPException(404, f"Dataset '{filename}' not found")
    df = _uploaded_datasets[filename].tail(tail)
    bars = []
    for dt, row in df.iterrows():
        bars.append({
            "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
            "open":   round(float(row["Open"]),   4),
            "high":   round(float(row["High"]),   4),
            "low":    round(float(row["Low"]),    4),
            "close":  round(float(row["Close"]),  4),
            "volume": int(row["Volume"]),
        })
    return {"filename": filename, "bars": bars, "count": len(bars)}
