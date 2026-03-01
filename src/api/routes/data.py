"""
Data API routes — fetch prices, indicators, S&P 500, upload CSV.
"""

import os
import io
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query, UploadFile, File, HTTPException
from cachetools import TTLCache

from src.api.schemas.schemas import (
    PriceResponse, PriceBar, IndicatorResponse, SP500Response, UploadResponse
)
from src.features.technical_indicators import add_all_technical_indicators
from src.data.data_acquisition import get_sp500_tickers

router = APIRouter()

# In-memory cache: max 200 entries, 5 min TTL
_price_cache = TTLCache(maxsize=200, ttl=300)
_uploaded_datasets = {}  # filename -> DataFrame


def _fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV from yfinance."""
    key = f"yf:{symbol}:{start}:{end}"
    if key in _price_cache:
        return _price_cache[key]
    df = yf.download(symbol, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    _price_cache[key] = df
    return df


def _fetch_alpha_vantage(symbol: str) -> pd.DataFrame:
    """Fetch daily data from Alpha Vantage."""
    import requests
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not api_key or api_key == "your_alpha_vantage_key":
        raise HTTPException(400, "Alpha Vantage API key not configured")
    url = (
        f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
        f"&symbol={symbol}&outputsize=full&apikey={api_key}"
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
    df = df.sort_values("Date").set_index("Date")
    return df


# ── Endpoints ──────────────────────────────────────────────────

@router.get("/sources")
async def list_sources():
    """List available data sources."""
    sources = ["yfinance"]
    ak = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    if ak and ak != "your_alpha_vantage_key":
        sources.append("alpha_vantage")
    uploads = list(_uploaded_datasets.keys())
    return {"sources": sources, "uploaded_datasets": uploads}


@router.get("/prices/{symbol}", response_model=PriceResponse)
async def get_prices(
    symbol: str,
    source: str = Query("yfinance", enum=["yfinance", "alpha_vantage"]),
    start: Optional[str] = None,
    end: Optional[str] = None,
    days: int = Query(120, ge=5, le=3650),
):
    """Fetch OHLCV price data for a symbol."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        if source == "alpha_vantage":
            df = _fetch_alpha_vantage(symbol)
            df = df.loc[start:end]
        else:
            df = _fetch_yfinance(symbol, start, end)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Data fetch failed: {str(e)}")

    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

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
    return PriceResponse(symbol=symbol, source=source, bars=bars, count=len(bars))


@router.get("/indicators/{symbol}", response_model=IndicatorResponse)
async def get_indicators(
    symbol: str,
    days: int = Query(120, ge=30, le=3650),
):
    """Compute technical indicators for a symbol."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 200)).strftime("%Y-%m-%d")  # extra for SMA200
    df = _fetch_yfinance(symbol, start, end)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    df = add_all_technical_indicators(df)
    df = df.tail(days)
    df = df.replace({float("nan"): None, float("inf"): None, float("-inf"): None})

    indicator_cols = [c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close"]]
    data = []
    for dt, row in df.iterrows():
        entry = {"date": str(dt.date()) if hasattr(dt, "date") else str(dt)}
        for c in indicator_cols:
            v = row[c]
            entry[c] = round(float(v), 4) if v is not None else None
        data.append(entry)

    return IndicatorResponse(symbol=symbol, indicators=indicator_cols, data=data, count=len(data))


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

    # Basic validation
    required = {"Open", "High", "Low", "Close", "Volume"}
    # Case-insensitive check
    col_map = {c.lower(): c for c in df.columns}
    missing = required - {c.capitalize() for c in col_map}
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}. Required: {required}")

    # Normalize column names
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
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]),
        })
    return {"filename": filename, "bars": bars, "count": len(bars)}
