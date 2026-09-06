"""
Pattern Detection API routes — multi-timeframe chart patterns.
"""

import logging
import asyncio
import math
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pandas as pd
from cachetools import LRUCache
from fastapi import APIRouter, Query, HTTPException, BackgroundTasks

from src.api.limits import clamp as clamp_interval
from src.data.ohlcv import fetch_ohlcv

from src.api.schemas.schemas import (
    PatternResponse, MultiTFPatternItem, ConfluenceResponse, ConfluenceSignal, BestSetupStatus,
    BestTradeSetup
)
from src.features.pattern_detector import (
    detect_chart_patterns,
    evaluate_best_setup,
    rank_patterns,
    build_market_context,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Confluence signals per symbol, bounded so a long-lived server cannot accumulate
# an entry for every symbol ever requested.
MAX_CONFLUENCE_SYMBOLS = 250
_confluence_store: LRUCache = LRUCache(maxsize=MAX_CONFLUENCE_SYMBOLS)
_confluence_last_update: LRUCache = LRUCache(maxsize=MAX_CONFLUENCE_SYMBOLS)

TF_CONFIG = {
    "1m": {"yf_interval": "1m", "weight": 1, "period": "7d", "pattern_lookback": 180, "min_candles": 120, "analysis_days": 7},
    "1h": {"yf_interval": "1h", "weight": 2, "period": "730d", "pattern_lookback": 240, "min_candles": 160, "analysis_days": 730},
    "1d": {"yf_interval": "1d", "weight": 3, "period": "max", "pattern_lookback": 320, "min_candles": 260, "analysis_days": 900},
    "1wk": {"yf_interval": "1wk", "weight": 4, "period": "max", "pattern_lookback": 300, "min_candles": 280, "analysis_days": 2600},
    "1mo": {"yf_interval": "1mo", "weight": 5, "period": "max", "pattern_lookback": 180, "min_candles": 150, "analysis_days": 5600},
}

#: Candles :func:`detect_support_resistance` actually reads. It slices
#: ``df.iloc[-100:]`` and ignores everything before that, so this is the real
#: analysis window on every interval and the number the response reports.
#: ``lookback`` only decides how much is *fetched*; anything past this is
#: downloaded and discarded, and anything short of it is a thinner read than
#: the algorithm was designed for.
SR_ANALYSIS_BARS = 100

#: ``lookback`` is a count of BARS OF THE REQUESTED INTERVAL, not calendar days.
#:
#: It was read as calendar days here while the callers meant bars, and on the
#: long intervals the two are not close: `lookback=240` on `1mo` asked for 240
#: months and fetched 440 days, which is 15 candles. The detector then had 15
#: bars where it wants 100 and returned no levels at all -- a monthly panel that
#: was empty for every symbol, for a reason nothing in the response mentioned.
#:
#: Bars is the reading that makes the number mean the same thing on all three
#: intervals, and the one the UI copy ("the last N weeks") already assumed.
#: The numbers themselves live in :mod:`src.api.limits` under the ``lookback``
#: bucket, alongside the calendar-day windows the other routes clamp with, so the
#: two units are declared side by side and served together. The floors there sit
#: above :data:`SR_ANALYSIS_BARS` so a clamped request still fills the detector's
#: window; the ceilings are what the data reliably supports -- 300 weeks is six
#: years, 240 months is twenty.

#: Calendar days one bar of each interval spans, for turning a bar count into a
#: download window. The daily figure is 365/252: weekends and holidays mean 100
#: sessions need about 145 days of calendar to fit in.
_SR_CALENDAR_DAYS_PER_BAR = {
    "1d": 365.0 / 252.0,
    "1wk": 7.0,
    "1mo": 30.5,
}

#: Slack on that conversion. The per-bar figures are averages, and a window
#: sized to the average comes up short about half the time -- which costs the
#: detector bars off the far end of the frame it was promised. Cheap insurance:
#: the extra candles are trimmed by the 100-bar slice anyway.
_SR_WINDOW_SAFETY = 1.15
_SR_WINDOW_MARGIN_DAYS = 30


def _clamp_sr_lookback(interval: str, lookback: int) -> int:
    """Bars of ``interval``, clamped to this interval's window."""
    return clamp_interval(interval, lookback, "lookback")


def _sr_window_days(interval: str, bars: int) -> int:
    """Calendar days to download to come away with ``bars`` bars of ``interval``."""
    per_bar = _SR_CALENDAR_DAYS_PER_BAR.get(interval, _SR_CALENDAR_DAYS_PER_BAR["1d"])
    return int(math.ceil(bars * per_bar * _SR_WINDOW_SAFETY)) + _SR_WINDOW_MARGIN_DAYS


def _fetch_yf_data(symbol: str, interval: str, period: str, days_lookback: int) -> pd.DataFrame:
    """Fetch bars via the shared cached fetcher (src.data.ohlcv)."""
    return fetch_ohlcv(symbol, interval, period=period)


def _compute_confluence_bg(symbol: str):
    """Background task to compute multi-timeframe confluence for a symbol."""
    logger.info(f"Starting background confluence calculation for {symbol}")
    all_patterns = []
    
    # Fetch all 5 timeframes
    for tf, cfg in TF_CONFIG.items():
        df = _fetch_yf_data(symbol, cfg["yf_interval"], cfg["period"], cfg["analysis_days"])
        if df.empty: continue
        
        # We need about 120 bars minimum for good detection
        pats = detect_chart_patterns(df, lookback=cfg["pattern_lookback"], timeframe=tf, weight=cfg["weight"])
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
def get_confluence(symbol: str):
    """Retrieve pre-computed multi-timeframe confluence signals. This relies on background task."""
    symbol = symbol.upper()
    signals = _confluence_store.get(symbol, [])
    
    # Format to response model
    out = []
    for s in signals:
        out.append(ConfluenceSignal(**s))
        
    return ConfluenceResponse(symbol=symbol, confluence_signals=out)


@router.get("/support-resistance/{symbol}", response_model=None)
def get_support_resistance(
    symbol: str,
    interval: str = Query("1d", enum=["1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo"]),
    lookback: int = Query(
        180,
        ge=20,
        le=20000,
        description="History to read, in BARS of `interval` — 240 on 1mo is twenty years.",
    ),
):
    """
    Confirmed pivot support and resistance, on the bars of one interval.

    ``lookback`` counts **bars of** ``interval``, not calendar days. That
    distinction is the whole of this route's history: it was read as days while
    its callers passed bars, so a Predictions tab asking for 240 monthly candles
    was served a 440-day window holding 15 of them, and
    :func:`detect_support_resistance` -- which reads the last
    :data:`SR_ANALYSIS_BARS` candles and nothing else -- found no level that
    cleared two touches on any symbol. An empty monthly panel, from a request
    that looked satisfied.

    The response says what was actually read rather than echoing what was asked
    for. ``bars_analysed`` is the detector's own window, which is capped at 100
    however long a lookback is sent; a caller that captions its panel from
    ``lookback`` is describing a download, not an analysis.
    """
    from src.features.support_resistance import detect_support_resistance

    symbol = symbol.upper()
    lookback = _clamp_sr_lookback(interval, lookback)

    try:
        if interval in ("1d", "1wk", "1mo"):
            end = datetime.now().strftime("%Y-%m-%d")
            start = (
                datetime.now() - timedelta(days=_sr_window_days(interval, lookback))
            ).strftime("%Y-%m-%d")
            df = fetch_ohlcv(symbol, interval, start=start, end=end)
        else:
            df = fetch_ohlcv(symbol, interval)
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol}: {e}")
        raise HTTPException(502, f"Data fetch failed for {symbol}")
    
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
        # What was read, in the bars it was read on, so a caller can caption the
        # panel truthfully without knowing this route's internals. `lookback`
        # sizes the download; `bars_analysed` is what reached the detector, and
        # on a short history it is the smaller of the two.
        "interval": interval,
        "lookback_bars": int(lookback),
        "bars_available": int(len(df)),
        "bars_analysed": int(min(len(df), SR_ANALYSIS_BARS)),
    }


@router.get("/{symbol}", response_model=PatternResponse)
def get_patterns(
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
    df = _fetch_yf_data(symbol, cfg["yf_interval"], cfg["period"], cfg["analysis_days"])
    
    if df.empty:
        raise HTTPException(404, f"No data for {symbol} at {tf}")

    market_context = build_market_context(df)
    detected_patterns = detect_chart_patterns(
        df, 
        lookback=cfg["pattern_lookback"], 
        timeframe=tf, 
        weight=cfg["weight"]
    )
    patterns_raw = rank_patterns(detected_patterns, market_context=market_context)
    
    patterns = []
    for p in patterns_raw:
        patterns.append(MultiTFPatternItem(**p))

    setup_status_raw = evaluate_best_setup(
        patterns_raw,
        candle_count=len(df),
        timeframe=tf,
        market_context=market_context,
        min_candles=cfg["min_candles"],
    )
    best_setup_raw = setup_status_raw.pop("best_setup")
    best_pattern_raw = setup_status_raw.pop("best_pattern")
    setup_status = BestSetupStatus(**setup_status_raw)
    best_setup = BestTradeSetup(**best_setup_raw) if best_setup_raw else None
    best_pattern = MultiTFPatternItem(**best_pattern_raw) if best_pattern_raw else None

    return PatternResponse(
        symbol=symbol,
        timeframe=tf,
        status=setup_status.status,
        best_setup_status=setup_status,
        best_setup=best_setup,
        best_pattern=best_pattern,
        patterns=patterns
    )
