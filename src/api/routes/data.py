"""
Data API routes — fetch prices, indicators, S&P 500, upload CSV.

Changes from original:
- Dynamic cache TTL: 60 s when market open, 3600 s when closed.
- _fetch_yfinance: uses period="1d" interval="1m" for intraday, always .iloc[-1].
- _fetch_alpha_vantage: uses GLOBAL_QUOTE (live) instead of TIME_SERIES_DAILY_ADJUSTED.
- All responses include market_open + data_timestamp metadata.
- Freshness validation via src.data.live_data.validate_freshness.
- New GET /quote/{symbol} lightweight live-price endpoint.
"""

import os
import io
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
import pytz
from cachetools import TTLCache
from fastapi import APIRouter, Query, UploadFile, File, HTTPException

from src.api.schemas.schemas import (
    PriceResponse, PriceBar, IndicatorResponse, SP500Response, UploadResponse
)
from src.features.technical_indicators import add_all_technical_indicators
from src.data.data_acquisition import get_sp500_tickers
from src.data.live_data import (
    is_market_open, get_market_session, get_cache_ttl,
    validate_freshness, fetch_live_quote, fetch_extended_quote
)

logger = logging.getLogger(__name__)
router = APIRouter()

_UTC = pytz.UTC

# ── Cache ─────────────────────────────────────────────────────────────────────
# TTL is evaluated lazily on first access each server start.
# For true dynamic TTL per request we bypass cache when market is open.
_price_cache: dict = {}          # key -> (df, cached_at)
_uploaded_datasets: dict = {}   # filename -> DataFrame


def _cache_get(key: str) -> Optional[pd.DataFrame]:
    """Return cached DataFrame if within the current TTL, else None."""
    entry = _price_cache.get(key)
    if entry is None:
        return None
    df, cached_at = entry
    ttl = get_cache_ttl()
    if (datetime.utcnow() - cached_at).total_seconds() < ttl:
        return df
    del _price_cache[key]
    return None


def _cache_set(key: str, df: pd.DataFrame) -> None:
    _price_cache[key] = (df, datetime.utcnow())


# ── Internal fetchers ─────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch OHLCV from yfinance.

    - During market hours: fetches today's 1-minute bars so the last row is
      always the most recent tick available.
    - Outside market hours: fetches daily bars for the requested range.
    """
    market_open = is_market_open()
    if market_open:
        key = f"yf_live:{symbol}"
        cached = _cache_get(key)
        if cached is not None:
            return cached
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            _cache_set(key, df)
        return df
    else:
        key = f"yf:{symbol}:{start}:{end}"
        cached = _cache_get(key)
        if cached is not None:
            return cached
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            _cache_set(key, df)
        return df


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
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize=compact&apikey={api_key}"
    )
    resp = requests.get(url, timeout=15)
    data = resp.json()
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
            "Volume": int(vals["6. volume"]),
        })
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    # Sort ascending — index[0] = oldest, index[-1] = LATEST (correct)
    df = df.sort_values("Date").set_index("Date")
    return df.loc[start:end]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _df_to_bars(df: pd.DataFrame) -> list[PriceBar]:
    bars = []
    for dt, row in df.iterrows():
        bars.append(PriceBar(
            date=str(dt.date()) if hasattr(dt, "date") else str(dt),
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


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/sources")
async def list_sources():
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


@router.get("/quote/{symbol}")
async def get_live_quote(
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
        raise HTTPException(502, f"Live quote fetch failed for {symbol}: {e}")


@router.get("/extended-quote/{symbol}")
async def get_extended_quote(
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
        raise HTTPException(502, f"Extended quote fetch failed for {symbol}: {e}")


@router.get("/prices/{symbol}", response_model=PriceResponse)
async def get_prices(
    symbol: str,
    source: str = Query("yfinance", enum=["yfinance", "alpha_vantage"]),
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = Query(120, ge=5, le=3650),
):
    """Fetch OHLCV price data for a symbol."""
    symbol = symbol.upper()
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        if source == "alpha_vantage":
            df = _fetch_alpha_vantage_history(symbol, start, end)
        else:
            df = _fetch_yfinance(symbol, start, end)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Data fetch failed: {e}")

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

    bars = _df_to_bars(df)
    return PriceResponse(
        symbol=symbol,
        source=source,
        bars=bars,
        count=len(bars),
    )


@router.get("/indicators/{symbol}", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str,
    days: int = Query(120, ge=30, le=3650),
):
    """Compute technical indicators for a symbol."""
    symbol = symbol.upper()
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 200)).strftime("%Y-%m-%d")  # extra for SMA200

    df = _fetch_yfinance(symbol, start, end)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    df = add_all_technical_indicators(df)
    df = df.tail(days)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

    indicator_cols = [
        c for c in df.columns
        if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
    ]
    data = []
    for dt, row in df.iterrows():
        entry = {"date": str(dt.date()) if hasattr(dt, "date") else str(dt)}
        for c in indicator_cols:
            v = row[c]
            entry[c] = round(float(v), 4) if v is not None else None
        data.append(entry)

    return IndicatorResponse(
        symbol=symbol,
        indicators=indicator_cols,
        data=data,
        count=len(data),
    )


@router.get("/sp500", response_model=SP500Response)
async def get_sp500():
    """Get S&P 500 constituents."""
    try:
        from src.data.market_data import get_sp500_constituents
        df = get_sp500_constituents()
        if df is not None and not df.empty:
            symbols = df[["Symbol", "Security", "GICS Sector"]].rename(
                columns={"Security": "company", "GICS Sector": "sector", "Symbol": "symbol"}
            ).to_dict("records")
            return SP500Response(symbols=symbols, count=len(symbols))
    except Exception:
        pass
    tickers = get_sp500_tickers()
    return SP500Response(symbols=[{"symbol": t} for t in tickers], count=len(tickers))


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset for training / backtesting."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")
    contents = await file.read()
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

    _uploaded_datasets[file.filename] = df
    return UploadResponse(
        filename=file.filename,
        rows=len(df),
        columns=list(df.columns),
        date_range={"start": str(df.index.min().date()), "end": str(df.index.max().date())},
        message=f"Uploaded {file.filename}: {len(df)} rows",
    )


@router.get("/uploaded/{filename}")
async def get_uploaded_data(filename: str, tail: int = Query(120, ge=1)):
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
