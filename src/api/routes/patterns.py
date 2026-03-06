"""
Pattern Detection API routes — candlestick patterns, chart structure
patterns, and ML signal confluence.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, HTTPException

from src.api.schemas.schemas import (
    PatternResponse, CandlestickPatternItem, ChartPatternItem,
    KeyLevel, ConfluenceSignal,
)
from src.features.candlestick_patterns import detect_candlestick_patterns
from src.features.pattern_detector import detect_chart_patterns
from src.features.technical_indicators import add_all_technical_indicators
from src.features.confluence import compute_confluence

logger = logging.getLogger(__name__)
router = APIRouter()


def _fetch_ohlcv(symbol: str, interval: str, lookback: int) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance."""
    try:
        if interval in ("1d", "1wk"):
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=lookback + 200)).strftime("%Y-%m-%d")
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        else:
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d"}
            period = period_map.get(interval, "60d")
            df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        logger.error(f"YF fetch failed: {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df = df.dropna(subset=["Close"])
    return df


@router.get("/{symbol}", response_model=PatternResponse)
async def get_patterns(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk"]),
    lookback: int = Query(120, ge=30, le=500),
):
    """Detect candlestick patterns, chart structure patterns, and compute confluence."""
    symbol = symbol.upper()

    df = _fetch_ohlcv(symbol, interval, lookback)
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    # ── Candlestick Patterns ──────────────────────────────
    cdl_result = detect_candlestick_patterns(df)
    cdl_patterns = []

    # Map pattern column names to display names and directions
    cdl_map = {
        "cdl_doji": ("Doji", "neutral"),
        "cdl_hammer": ("Hammer", "bullish"),
        "cdl_shooting_star": ("Shooting Star", "bearish"),
        "cdl_bullish_engulfing": ("Bullish Engulfing", "bullish"),
        "cdl_bearish_engulfing": ("Bearish Engulfing", "bearish"),
        "cdl_morning_star": ("Morning Star", "bullish"),
        "cdl_evening_star": ("Evening Star", "bearish"),
        "cdl_bullish_harami": ("Bullish Harami", "bullish"),
        "cdl_bearish_harami": ("Bearish Harami", "bearish"),
    }

    for col, (name, direction) in cdl_map.items():
        if col not in cdl_result.columns:
            continue
        hits = cdl_result[cdl_result[col] == 1].index
        for dt in hits:
            date_str = str(dt.date()) if hasattr(dt, "date") else str(dt)
            # Confidence based on pattern type (multi-bar patterns are more reliable)
            base_conf = 0.7 if "engulfing" in col or "star" in col else 0.55
            cdl_patterns.append(CandlestickPatternItem(
                date=date_str,
                pattern_name=name,
                direction=direction if direction != "neutral" else "bullish",
                confidence=round(base_conf, 2),
            ))

    # Only keep patterns from the lookback window
    cdl_patterns = cdl_patterns[-50:]  # Cap to avoid huge payloads

    # ── Chart Structure Patterns ──────────────────────────
    chart_pats_raw = detect_chart_patterns(df, lookback=lookback)
    chart_patterns = []
    for p in chart_pats_raw:
        chart_patterns.append(ChartPatternItem(
            pattern_name=p["pattern_name"],
            start_date=p["start_date"],
            end_date=p["end_date"],
            key_levels=[KeyLevel(date=kl["date"], price=kl["price"]) for kl in p["key_levels"]],
            neckline=p.get("neckline"),
            breakout_price=p.get("breakout_price"),
            target_price=p.get("target_price"),
            confidence=p["confidence"],
            status=p["status"],
        ))

    # ── Confluence ────────────────────────────────────────
    df_ind = add_all_technical_indicators(df)
    latest = df_ind.iloc[-1] if not df_ind.empty else {}

    indicators = {}
    for key in ["RSI", "MACD", "MACD_Signal", "MACD_Histogram", "ATR"]:
        val = latest.get(key)
        if val is not None and pd.notna(val):
            indicators[key] = float(val)

    # Determine dominant pattern direction
    bullish_count = sum(1 for p in cdl_patterns if p.direction == "bullish")
    bearish_count = sum(1 for p in cdl_patterns if p.direction == "bearish")
    for cp in chart_patterns:
        if "Bottom" in cp.pattern_name or "Inverse" in cp.pattern_name or "Bull" in cp.pattern_name or "Ascending" in cp.pattern_name:
            bullish_count += 2
        elif "Top" in cp.pattern_name or "Head & Shoulders" == cp.pattern_name or "Bear" in cp.pattern_name or "Descending" in cp.pattern_name:
            bearish_count += 2

    if bullish_count > bearish_count:
        pattern_dir = "bullish"
    elif bearish_count > bullish_count:
        pattern_dir = "bearish"
    else:
        pattern_dir = None

    # Simple ML direction from recent returns
    ml_dir = None
    ml_conf = 0.0
    if len(df) > 5:
        recent_return = float((df["Close"].iloc[-1] - df["Close"].iloc[-5]) / df["Close"].iloc[-5])
        ml_dir = "up" if recent_return > 0 else "down"
        ml_conf = min(95, abs(recent_return) * 500)

    confluence = compute_confluence(indicators, pattern_dir, ml_dir, ml_conf)

    return PatternResponse(
        symbol=symbol,
        candlestick_patterns=cdl_patterns,
        chart_patterns=chart_patterns,
        confluence=ConfluenceSignal(**confluence),
    )
