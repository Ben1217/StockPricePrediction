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
    Detect all support and resistance features.
    """
    raw_levels = []
    trendlines = []
    n = len(df)
    
    # Method 1: Candlestick Patterns
    cdl_df = detect_candlestick_patterns(df)
    
    bullish_cols = ["cdl_hammer", "cdl_bullish_engulfing", "cdl_morning_star", "cdl_bullish_harami"]
    bearish_cols = ["cdl_shooting_star", "cdl_bearish_engulfing", "cdl_evening_star", "cdl_bearish_harami"]
    
    for i in range(n):
        # Bullish patterns -> Support at the low
        if any(cdl_df.iloc[i].get(col, 0) == 1 for col in bullish_cols):
            raw_levels.append({
                "price": df["Low"].iloc[i],
                "type": "support",
                "source": "pattern_bullish"
            })
            
        # Bearish patterns -> Resistance at the high
        if any(cdl_df.iloc[i].get(col, 0) == 1 for col in bearish_cols):
            raw_levels.append({
                "price": df["High"].iloc[i],
                "type": "resistance",
                "source": "pattern_bearish"
            })

    # Method 2: Peaks & Troughs
    pivot_highs, pivot_lows = _find_pivots(df, window=7)
    
    for idx in pivot_highs:
        raw_levels.append({
            "price": df["High"].iloc[idx],
            "type": "resistance",
            "source": "peak"
        })
        
    for idx in pivot_lows:
        raw_levels.append({
            "price": df["Low"].iloc[idx],
            "type": "support",
            "source": "trough"
        })
        
    # Method 3: Trendlines
    # Try finding support trendline from lows
    support_tl = _fit_trendline(pivot_lows[-10:], df["Low"].values, n, is_support=True)
    if support_tl:
        trendlines.append(support_tl)
        
    # Try finding resistance trendline from highs
    resist_tl = _fit_trendline(pivot_highs[-10:], df["High"].values, n, is_support=False)
    if resist_tl:
        trendlines.append(resist_tl)

    # Process and Merge Levels (Method 2B: Multiple touches = zones)
    merged_levels = _merge_levels(raw_levels, tolerance_pct=0.015) # 1.5% zone tolerance (aggressive merge)
    
    # Dynamic re-labeling: if a "resistance" level is now BELOW current price, it flips to support.
    # If a "support" level is now ABOVE current price, it flips to resistance.
    for level in merged_levels:
        if level["price"] < current_price:
            level["type"] = "support"
        else:
            level["type"] = "resistance"

    # Method 4: Dynamic Moving Averages
    dynamic_levels = []
    
    df_ind = add_all_technical_indicators(df)
    latest_ma200 = df_ind["SMA_200"].iloc[-1] if "SMA_200" in df_ind.columns else None
    latest_ma50  = df_ind["SMA_50"].iloc[-1] if "SMA_50" in df_ind.columns else None
    
    if pd.notna(latest_ma200):
        # Check bounces in last 10 days
        recent_lows = df["Low"].iloc[-10:]
        recent_highs = df["High"].iloc[-10:]
        bounces = []
        
        # If price is above MA200, it's support
        if current_price > latest_ma200:
            for i in range(len(recent_lows)):
                # If low touched or dipped slightly below MA but closed above
                ma_val = df_ind["SMA_200"].iloc[-10+i]
                if recent_lows.iloc[i] <= ma_val * 1.005 and df["Close"].iloc[-10+i] > ma_val:
                    # Record the local index for drawing coordinates
                    bounces.append({"index": n - 10 + i, "price": recent_lows.iloc[i], "type": "support"})
                    
            dynamic_levels.append({
                "type": "support",
                "name": "MA(200)",
                "price": round(float(latest_ma200), 2),
                "bounces": bounces
            })
            
        # If price is below MA200, it's resistance
        else:
            for i in range(len(recent_highs)):
                ma_val = df_ind["SMA_200"].iloc[-10+i]
                if recent_highs.iloc[i] >= ma_val * 0.995 and df["Close"].iloc[-10+i] < ma_val:
                    bounces.append({"index": n - 10 + i, "price": recent_highs.iloc[i], "type": "resistance"})

            dynamic_levels.append({
                "type": "resistance",
                "name": "MA(200)",
                "price": round(float(latest_ma200), 2),
                "bounces": bounces # Used to draw arrows in the UI
            })

    # Return only the relevant merged levels (e.g. 3 closest above, 3 closest below)
    supports = sorted([l for l in merged_levels if l["type"] == "support"], key=lambda x: x["price"], reverse=True)
    resistances = sorted([l for l in merged_levels if l["type"] == "resistance"], key=lambda x: x["price"])
    
    return {
        "levels": supports[:1] + resistances[:1],  # 1 key support + 1 key resistance
        "trendlines": trendlines,
        "dynamic_levels": dynamic_levels
    }
