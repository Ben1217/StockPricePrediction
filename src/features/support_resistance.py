"""
Support and Resistance Detection Module.

Algorithm incorporates 4 methods from the developer specs:
A. Peaks & Troughs (Local Extrema)
B. Previous Price Levels (Zones of multiple touches)
C. Moving Averages (Dynamic S&R based on MA200 and MA50)
D. Trendlines (Connecting 3+ highs/lows)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from ..utils.logger import get_logger
from .technical_indicators import add_all_technical_indicators
from .candlestick_patterns import detect_candlestick_patterns

logger = get_logger(__name__)


def _find_pivots(df: pd.DataFrame, window: int = 5) -> Tuple[List[int], List[int]]:
    """Local pivot detection."""
    highs = df["High"].values
    lows = df["Low"].values
    n = len(df)
    
    pivot_highs = []
    pivot_lows = []

    for i in range(window, n - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            pivot_highs.append(i)
        if lows[i] == min(lows[i - window : i + window + 1]):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def _merge_levels(levels: List[Dict[str, Any]], tolerance_pct: float = 0.005) -> List[Dict[str, Any]]:
    """Merge price levels that are within tolerance percentage into a single zone."""
    if not levels:
        return []

    # Sort levels strictly by price
    levels = sorted(levels, key=lambda x: x["price"])
    
    merged = []
    current_zone = [levels[0]]
    
    for lvl in levels[1:]:
        avg_price = sum(l["price"] for l in current_zone) / len(current_zone)
        # If within tolerance of the current cluster's average price
        if abs(lvl["price"] - avg_price) / avg_price <= tolerance_pct:
            current_zone.append(lvl)
        else:
            merged.append(_consolidate_zone(current_zone))
            current_zone = [lvl]
            
    if current_zone:
        merged.append(_consolidate_zone(current_zone))
        
    return merged


def _consolidate_zone(zone: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine multiple levels into a single robust zone."""
    avg_price = sum(l["price"] for l in zone) / len(zone)
    method_count = len(zone)
    
    # Check if this zone has confirmations from different methods/sources
    sources = set(l.get("source", "unknown") for l in zone)
    is_strong = len(sources) >= 2 or method_count >= 3
    
    # Determine type based on majority vote of the sub-levels
    # (If mixed, we label it based on where price is currently, but we do that later)
    types = [l["type"] for l in zone]
    support_count = types.count("support")
    resistance_count = types.count("resistance")
    primary_type = "support" if support_count >= resistance_count else "resistance"
    
    return {
        "price": round(float(avg_price), 2),
        "type": primary_type,
        "strength": "strong" if is_strong else "normal",
        "confirmations": method_count,
        "sources": list(sources),
        "zone_low": round(float(min(l["price"] for l in zone)), 2),
        "zone_high": round(float(max(l["price"] for l in zone)), 2)
    }


def _fit_trendline(indices: List[int], prices: np.ndarray, n: int, is_support: bool) -> Optional[Dict[str, Any]]:
    """Fit a trendline to a set of pivot points."""
    if len(indices) < 3:
        return None
        
    x = np.array(indices)
    y = prices[x]
    
    # Simple linear regression
    slope, intercept = np.polyfit(x, y, 1)
    
    # Check quality of fit (R-squared)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # High threshold for R_squared ensures it's actually a line
    if r_squared > 0.85:
        # Calculate start and end points of the line
        start_idx = indices[0]
        end_idx = n - 1 # Project to the end of the chart
        
        start_price = slope * start_idx + intercept
        end_price = slope * end_idx + intercept
        
        return {
            "type": "support" if is_support else "resistance",
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "start_price": round(float(start_price), 2),
            "end_price": round(float(end_price), 2),
            "slope": float(slope),
            "r_squared": float(r_squared)
        }
    return None


def detect_support_resistance(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Detect clean Support and Resistance features based ONLY on key swing highs/lows 
    with multiple touches (>= 2) over the recent price action (approx 100 candles), 
    filtering out noise and returning at most 1 key support and 1 key resistance.
    """
    # Use last 100 candles
    df_recent = df.iloc[-100:] if len(df) > 100 else df
    
    # Pivot window of 5 implies 11-candle formation (5 before, 5 after)
    # Good for major swings, ignores minor noise
    pivot_highs, pivot_lows = _find_pivots(df_recent, window=5)
    
    raw_highs = [{"price": float(df_recent["High"].iloc[idx]), "type": "resistance", "source": "peak"} for idx in pivot_highs]
    raw_lows = [{"price": float(df_recent["Low"].iloc[idx]), "type": "support", "source": "trough"} for idx in pivot_lows]
    
    # Group levels that are within 1.5% of each other into zones
    merged_highs = _merge_levels(raw_highs, tolerance_pct=0.015)
    merged_lows = _merge_levels(raw_lows, tolerance_pct=0.015)
    
    # Filter for valid resistances: >= 2 touches AND strictly above current price
    valid_resistances = [
        l for l in merged_highs 
        if l["price"] > current_price and l["confirmations"] >= 2
    ]
    # Sort to find the CLOSEST resistance above price
    valid_resistances = sorted(valid_resistances, key=lambda x: x["price"])
    
    # Filter for valid supports: >= 2 touches AND strictly below current price
    valid_supports = [
        l for l in merged_lows 
        if l["price"] < current_price and l["confirmations"] >= 2
    ]
    # Sort to find the CLOSEST support below price (descending order)
    valid_supports = sorted(valid_supports, key=lambda x: x["price"], reverse=True)
    
    return {
        "levels": valid_supports[:1] + valid_resistances[:1],  # 1 key support + 1 key resistance
        "trendlines": [],
        "dynamic_levels": []
    }
