"""
Pattern Detection API routes — multi-timeframe chart patterns.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks

from src.api.schemas.schemas import (
    PatternResponse, MultiTFPatternItem, ConfluenceResponse, ConfluenceSignal
)
from src.features.pattern_detector import detect_chart_patterns

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple in-memory cache for confluence signals
_confluence_store: Dict[str, List[Dict]] = {}
# Simple cache to prevent spamming background tasks
_confluence_last_update: Dict[str, datetime] = {}

TF_CONFIG = {
    "1m": {"yf_interval": "1m", "weight": 1, "lookback_mult": 1, "period": "7d"},
    "1h": {"yf_interval": "1h", "weight": 2, "lookback_mult": 1, "period": "730d"},
    "1d": {"yf_interval": "1d", "weight": 3, "lookback_mult": 1, "period": "max"},
    "1wk": {"yf_interval": "1wk", "weight": 4, "lookback_mult": 5, "period": "max"},
    "1mo": {"yf_interval": "1mo", "weight": 5, "lookback_mult": 20, "period": "max"},
}


def _fetch_yf_data(symbol: str, interval: str, period: str, days_lookback: int) -> pd.DataFrame:
    try:
        if interval in ("1d", "1wk", "1mo") and period == "max":
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=days_lookback)).strftime("%Y-%m-%d")
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        else:
            df = yf.download(symbol, period=period, interval=interval, progress=False)
    except Exception as e:
        logger.error(f"YF fetch failed for {symbol} at {interval}: {e}")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df = df.dropna(subset=["Close"])
    return df


def _compute_confluence_bg(symbol: str):
    """Background task to compute multi-timeframe confluence for a symbol."""
    logger.info(f"Starting background confluence calculation for {symbol}")
    all_patterns = []
    
    # Fetch all 5 timeframes
    for tf, cfg in TF_CONFIG.items():
        df = _fetch_yf_data(symbol, cfg["yf_interval"], cfg["period"], 365*2)
        if df.empty: continue
        
        # We need about 120 bars minimum for good detection
        pats = detect_chart_patterns(df, lookback=200, timeframe=tf, weight=cfg["weight"])
        all_patterns.extend(pats)
        
    # Group patterns by name and direction to find overlapping confidences
    confluence_map = {}
    for p in all_patterns:
        if p["status"] == "broken": continue
        
        key = f"{p['pattern_name']}_{p['direction']}"
        if key not in confluence_map:
            confluence_map[key] = {
                "pattern_name": p["pattern_name"],
                "direction": p["direction"],
                "timeframes": set(),
                "total_weight": 0
            }
        
        # Only add weight if the timeframe hasn't contributed yet for this pattern
        if p["timeframe"] not in confluence_map[key]["timeframes"]:
            confluence_map[key]["timeframes"].add(p["timeframe"])
            confluence_map[key]["total_weight"] += p["weight"]
            
    # Filter for signals that exist on multiple timeframes
    conf_signals = []
    for k, v in confluence_map.items():
        if len(v["timeframes"]) >= 2:
            conf_signals.append({
                "pattern_name": v["pattern_name"],
                "direction": v["direction"],
                "timeframes": list(v["timeframes"]),
                "total_weight": v["total_weight"]
            })
            
    _confluence_store[symbol] = conf_signals
    _confluence_last_update[symbol] = datetime.now()
    logger.info(f"Background confluence finished for {symbol}: found {len(conf_signals)} signals")


@router.get("/confluence/{symbol}", response_model=ConfluenceResponse)
async def get_confluence(symbol: str):
    """Retrieve pre-computed multi-timeframe confluence signals. This relies on background task."""
    symbol = symbol.upper()
    signals = _confluence_store.get(symbol, [])
    
    # Format to response model
    out = []
    for s in signals:
        out.append(ConfluenceSignal(**s))
        
    return ConfluenceResponse(symbol=symbol, confluence_signals=out)


@router.get("/support-resistance/{symbol}", response_model=None)
async def get_support_resistance(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk"]),
    lookback: int = Query(180, ge=60, le=500),
):
    """Detect dynamic Support and Resistance levels based on patterns, MAs, and pivots."""
    from src.features.support_resistance import detect_support_resistance
    
    symbol = symbol.upper()
    
    try:
        if interval in ("1d", "1wk"):
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=lookback + 200)).strftime("%Y-%m-%d")
            df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
        else:
            period_map = {"1m": "7d", "5m": "60d", "15m": "60d", "1h": "730d", "4h": "730d"}
            period = period_map.get(interval, "60d")
            df = yf.download(symbol, period=period, interval=interval, progress=False)
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
    except Exception as e:
        logger.error(f"YF fetch failed: {e}")
        raise HTTPException(500, "Fetch failed")
    
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    current_price = float(df["Close"].iloc[-1])
    
    try:
        sr_data = detect_support_resistance(df, current_price)
    except Exception as e:
        logger.error(f"S&R detection failed for {symbol}: {e}")
        raise HTTPException(500, f"Algorithm error: {e}")

    return {
        "symbol": symbol,
        "current_price": current_price,
        "levels": sr_data["levels"],
        "trendlines": sr_data["trendlines"],
        "dynamic_levels": sr_data["dynamic_levels"],
    }


@router.get("/{symbol}", response_model=PatternResponse)
async def get_patterns(
    symbol: str, 
    background_tasks: BackgroundTasks,
    tf: str = Query("1d", enum=["1m", "1h", "1d", "1wk", "1mo"])
):
    """Detect chart patterns for a specific timeframe."""
    symbol = symbol.upper()

    if tf not in TF_CONFIG:
         raise HTTPException(400, "Invalid timeframe selection")
         
    # Trigger background confluence refresh if stale (> 30 min)
    last_update = _confluence_last_update.get(symbol)
    if not last_update or (datetime.now() - last_update).total_seconds() > 1800:
         background_tasks.add_task(_compute_confluence_bg, symbol)

    cfg = TF_CONFIG[tf]
    df = _fetch_yf_data(symbol, cfg["yf_interval"], cfg["period"], 500)
    
    if df.empty:
        raise HTTPException(404, f"No data for {symbol} at {tf}")

    patterns_raw = detect_chart_patterns(
        df, 
        lookback=300, 
        timeframe=tf, 
        weight=cfg["weight"]
    )
    
    patterns = []
    for p in patterns_raw:
        patterns.append(MultiTFPatternItem(**p))

    return PatternResponse(
        symbol=symbol,
        timeframe=tf,
        patterns=patterns
    )
