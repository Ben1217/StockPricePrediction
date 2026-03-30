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


@router.get("/{symbol}")
async def get_sentiment(
    symbol: str,
    days: int = Query(400, ge=30, le=800),
):
    """
    Compute 3-indicator sentiment for a symbol.

    Returns individual indicator signals, aggregate score (-3 to +3),
    sentiment label, and entry signal (BULLISH / BEARISH / WAIT).
    """
    symbol = symbol.upper()

    # Fetch enough data for the 200 MA
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        raise HTTPException(502, f"Data fetch failed for {symbol}: {e}")

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

    return {
        "symbol": symbol,
        **result,
    }
