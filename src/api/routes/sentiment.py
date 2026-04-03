"""
Sentiment API routes — 3-Indicator Sentiment System.

Computes a rule-based sentiment classification for a stock using:
  1. 200-day Moving Average (Trend)
  2. RSI-14 crossover (Momentum)
  3. Volume vs 20-day average (Volume Confirmation)

Returns a score from -3 to +3 and a Bullish / Bearish / Neutral verdict.
"""

import logging
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, HTTPException

from src.signals.sentiment.indicator_sentiment import compute_indicator_sentiment

logger = logging.getLogger(__name__)
router = APIRouter()


def _fetch_sentiment_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    try:
        if interval in ("1wk", "1mo"):
            df = yf.download(symbol, period="max", interval=interval, progress=False)
        elif interval in ("1d",):
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        else:
            period_map = {
                "1m": "7d",
                "5m": "60d",
                "15m": "60d",
                "1h": "730d",
                "4h": "730d",
            }
            df = yf.download(symbol, period=period_map.get(interval, "1mo"), interval=interval, progress=False)
    except Exception as e:
        raise HTTPException(502, f"Data fetch failed for {symbol}: {e}")

    if df.empty and interval in ("1d", "1wk", "1mo"):
        fallback_period = {"1d": "1y", "1wk": "max", "1mo": "max"}[interval]
        try:
            df = yf.download(symbol, period=fallback_period, interval=interval, progress=False)
        except Exception as e:
            raise HTTPException(502, f"Fallback data fetch failed for {symbol}: {e}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if not df.empty:
        df = df.dropna(subset=["Close"])

    return df


@router.get("/{symbol}")
async def get_sentiment(
    symbol: str,
    days: int = Query(400, ge=30, le=7000),
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"]),
):
    """
    Compute 3-indicator sentiment for a symbol.

    Returns individual indicator signals, aggregate score (-3 to +3),
    sentiment label, and entry signal (BULLISH / BEARISH / WAIT).
    """
    symbol = symbol.upper()

    df = _fetch_sentiment_data(symbol, interval, days)

    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    if len(df) < 201:
        raise HTTPException(
            400,
            f"Insufficient data for {symbol}: need at least 201 bars for "
            f"200-day MA, got {len(df)}. Try increasing the 'days' parameter.",
        )

    try:
        result = compute_indicator_sentiment(df)
    except Exception as e:
        logger.exception("Sentiment computation failed for %s", symbol)
        raise HTTPException(500, f"Computation error: {e}")

    details = result.get("details", {})
    entry_signal = result.get("entry_signal", "WAIT")
    action = "BUY" if entry_signal == "BULLISH" else "SELL" if entry_signal == "BEARISH" else "WAIT"

    return {
        "mode": "INDICATOR",
        "symbol": symbol,
        "timeframe": interval,
        "action": action,
        "rsi": details.get("rsi"),
        "volume": details.get("volume"),
        "atr": details.get("atr"),
        "support": details.get("support"),
        "resistance": details.get("resistance"),
        **result,
    }
