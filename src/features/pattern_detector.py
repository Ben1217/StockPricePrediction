"""
Chart Pattern Detection Module (Multi-Timeframe Spec)

Detects 4 specific patterns with strict coordinate and targets logic:
  - Head & Shoulders (Bearish)
  - Double Bottom (Bullish)
  - Bull Flag (Bullish)
  - Symmetrical Triangle (Neutral / Breakout)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ── Pivot Detection ─────────────────────────────────────────────

def _find_pivots(df: pd.DataFrame, window: int = 5) -> tuple:
    """Find local high/low pivot points using rolling window."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    pivot_highs = []
    pivot_lows = []

    for i in range(window, n - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            pivot_highs.append(i)
        if lows[i] == min(lows[i - window : i + window + 1]):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def _date_str(df: pd.DataFrame, idx: int) -> str:
    dt = df.index[idx]
    if hasattr(dt, "date") and dt.time() == dt.time().replace(hour=0, minute=0, second=0):
        return str(dt.date())
    return str(dt)


def _price_at(df: pd.DataFrame, idx: int, col: str = "Close") -> float:
    return float(df.iloc[idx][col])


def _make_key_level(df, idx, price=None):
    return {"date": _date_str(df, idx), "price": price or _price_at(df, idx)}


# ── Pattern Matchers ────────────────────────────────────────────

def _detect_head_shoulders(df, pivot_highs, tolerance=0.03):
    """Detect Head & Shoulders pattern (Bearish)."""
    patterns = []
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    for i in range(len(pivot_highs) - 2):
        ls, hd, rs = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
        h_ls, h_hd, h_rs = highs[ls], highs[hd], highs[rs]

        # Head must be highest
        if h_hd <= h_ls or h_hd <= h_rs:
            continue
        # Shoulders roughly equal
        if abs(h_ls - h_rs) / max(h_ls, h_rs) > tolerance:
            continue

        # Neckline from troughs between peaks
        t1_idx = ls + np.argmin(lows[ls:hd])
        t2_idx = hd + np.argmin(lows[hd:rs])
        neckline = (float(lows[t1_idx]) + float(lows[t2_idx])) / 2
        
        target = neckline - (h_hd - neckline)
        stop_loss = h_rs

        last_close = closes[min(rs + 5, len(df) - 1)]
        status = "forming"
        breakout_price = None
        
        if last_close < neckline:
            status = "confirmed"
            breakout_price = neckline

        patterns.append({
            "pattern_name": "Head & Shoulders",
            "direction": "bearish",
            "start_date": _date_str(df, ls),
            "end_date": _date_str(df, rs),
            "key_levels": [
                _make_key_level(df, ls, h_ls),
                _make_key_level(df, t1_idx, float(lows[t1_idx])),
                _make_key_level(df, hd, h_hd),
                _make_key_level(df, t2_idx, float(lows[t2_idx])),
                _make_key_level(df, rs, h_rs),
            ],
            "neckline": round(neckline, 2),
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })

    return patterns


def _detect_double_bottom(df, pivot_lows, tolerance=0.02):
    """Detect Double Bottom pattern (Bullish)."""
    patterns = []
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)

    for i in range(len(pivot_lows) - 1):
        p1, p2 = pivot_lows[i], pivot_lows[i + 1]
        if p2 - p1 < 5:
            continue

        l1, l2 = lows[p1], lows[p2]
        if abs(l1 - l2) / max(l1, l2) > tolerance:
            continue

        peak_idx = p1 + np.argmax(highs[p1:p2])
        resistance = float(highs[peak_idx])
        trough_avg = (l1 + l2) / 2
        height = resistance - trough_avg
        
        target = resistance + height
        stop_loss = trough_avg * 0.995 # lightly below

        last_close = closes[min(p2 + 5, len(df) - 1)]
        status = "forming"
        breakout_price = None

        if last_close > resistance:
            status = "confirmed"
            breakout_price = resistance

        patterns.append({
            "pattern_name": "Double Bottom",
            "direction": "bullish",
            "start_date": _date_str(df, p1),
            "end_date": _date_str(df, p2),
            "key_levels": [
                _make_key_level(df, p1, l1),
                _make_key_level(df, peak_idx, resistance),
                _make_key_level(df, p2, l2),
            ],
            "neckline": round(resistance, 2),
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })

    return patterns


def _detect_bull_flag(df, pivot_highs, pivot_lows):
    """Detect Bull Flag pattern (Bullish)."""
    patterns = []
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    for i in range(20, n - 5):
        # Look for a sharp bullish move (flagpole)
        lookback = 15
        start = max(0, i - lookback)
        move = (closes[i] - closes[start]) / closes[start]

        if move < 0.05: # Need > 5% move for pole
            continue

        pole_length = highs[start:i].max() - lows[start:i].min()

        # Check for consolidation channel
        flag_end = min(i + 15, n)
        
        # Fit trendlines for the flag
        flag_highs = highs[i:flag_end]
        flag_lows = lows[i:flag_end]
        
        if len(flag_highs) < 5: continue
        
        upper_channel = np.polyfit(np.arange(len(flag_highs)), flag_highs, 1)[0]
        lower_channel = np.polyfit(np.arange(len(flag_lows)), flag_lows, 1)[0]
        
        # Upper and lower should be sloping down or flat
        if upper_channel > 0.01 or lower_channel > 0.01:
             continue
             
        breakout = float(flag_highs.max())
        flag_bottom = float(flag_lows.min())

        target = breakout + pole_length
        stop_loss = flag_bottom

        status = "forming"
        breakout_p = None
        if closes[flag_end - 1] > breakout:
            status = "confirmed"
            breakout_p = breakout

        patterns.append({
            "pattern_name": "Bull Flag",
            "direction": "bullish",
            "start_date": _date_str(df, start),
            "end_date": _date_str(df, min(flag_end - 1, n - 1)),
            "key_levels": [
                _make_key_level(df, start, closes[start]),
                _make_key_level(df, i, highs[i]),
                _make_key_level(df, min(flag_end - 1, n - 1), closes[flag_end - 1]),
            ],
            "trendlines": [
                [_make_key_level(df, i, highs[i]), _make_key_level(df, flag_end - 1, flag_highs[-1])],
                [_make_key_level(df, i, lows[i]), _make_key_level(df, flag_end - 1, flag_lows[-1])]
            ],
            "neckline": None,
            "breakout_price": round(breakout_p, 2) if breakout_p else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })
        break # Only report the most recent

    return patterns


def _detect_symmetrical_triangle(df, pivot_highs, pivot_lows):
    """Detect Symmetrical Triangle (Neutral until breakout)."""
    patterns = []
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(df)

    for start_idx in range(0, n - 20, 10):
        end_idx = min(start_idx + 60, n)
        
        # Get pivots strictly within timeline window
        wh = [p for p in pivot_highs if start_idx <= p < end_idx]
        wl = [p for p in pivot_lows if start_idx <= p < end_idx]

        if len(wh) < 2 or len(wl) < 2:
            continue

        h_vals = highs[wh]
        l_vals = lows[wl]

        h_slope = np.polyfit(wh, h_vals, 1)[0]
        l_slope = np.polyfit(wl, l_vals, 1)[0]

        # Both must converge: Upper trendline falls, lower rises
        if not (h_slope < -0.01 and l_slope > 0.01):
            continue

        height = h_vals[0] - l_vals[0]
        
        last_idx = max(max(wh), max(wl))
        last_close = closes[min(n - 1, last_idx + 5)]
        
        # Check breakout
        # Project lines
        proj_h = np.polyval(np.polyfit(wh, h_vals, 1), min(n-1, last_idx + 5))
        proj_l = np.polyval(np.polyfit(wl, l_vals, 1), min(n-1, last_idx + 5))
        
        status = "forming"
        direction = "neutral"
        target = None
        stop_loss = None
        breakout_price = None
        
        if last_close > proj_h:
            status = "confirmed"
            direction = "bullish"
            breakout_price = proj_h
            target = breakout_price + height
            stop_loss = proj_l # inside triangle
        elif last_close < proj_l:
            status = "confirmed"
            direction = "bearish"
            breakout_price = proj_l
            target = breakout_price - height
            stop_loss = proj_h # inside triangle

        key_levels = []
        for idx in sorted(list(wh) + list(wl)):
            key_levels.append(_make_key_level(df, idx, float(closes[idx])))

        trendlines = [
            [_make_key_level(df, wh[0], h_vals[0]), _make_key_level(df, wh[-1], h_vals[-1])],
            [_make_key_level(df, wl[0], l_vals[0]), _make_key_level(df, wl[-1], l_vals[-1])]
        ]

        patterns.append({
            "pattern_name": "Symmetrical Triangle",
            "direction": direction,
            "start_date": _date_str(df, min(min(wh), min(wl))),
            "end_date": _date_str(df, last_idx),
            "key_levels": key_levels[:8],
            "trendlines": trendlines,
            "neckline": None,
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2) if target else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "status": status,
        })

    return patterns


# ── Main Entry Point ────────────────────────────────────────────

def detect_chart_patterns(df: pd.DataFrame, lookback: int = 120, timeframe: str = "1d", weight: int = 1) -> List[Dict[str, Any]]:
    """
    Detect exactly 4 chart patterns.
    """
    data = df.tail(lookback).copy()
    if len(data) < 30:
        return []

    pivot_highs, pivot_lows = _find_pivots(data, window=5)
    all_patterns = []

    try:
        all_patterns.extend(_detect_head_shoulders(data, pivot_highs))
    except Exception as e:
        logger.warning(f"H&S detection failed: {e}")

    try:
        all_patterns.extend(_detect_double_bottom(data, pivot_lows))
    except Exception as e:
        logger.warning(f"Double bottom detection failed: {e}")

    try:
        all_patterns.extend(_detect_bull_flag(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Flag detection failed: {e}")
        
    try:
        all_patterns.extend(_detect_symmetrical_triangle(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Symmetrical triangle failed: {e}")

    # Deduplicate overlapping patterns
    seen = set()
    unique = []
    
    # Enrich with weight and timeframe
    for p in all_patterns:
        key = (p["pattern_name"], p["start_date"])
        if key not in seen:
            seen.add(key)
            p['timeframe'] = timeframe
            p['weight'] = weight
            unique.append(p)

    return unique
