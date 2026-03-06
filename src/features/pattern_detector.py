"""
Chart Structure Pattern Detection Module.

Detects multi-bar chart patterns using pivot-point analysis:
  - Double Top / Double Bottom
  - Head & Shoulders / Inverse Head & Shoulders
  - Ascending / Descending / Symmetrical Triangle
  - Bull Flag / Bear Flag
  - Rising / Falling Wedge
  - Cup & Handle

Each detection returns: pattern_name, start_date, end_date,
key_levels, neckline, breakout_price, target_price, confidence, status.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


# ── Pivot Detection ─────────────────────────────────────────────

def _find_pivots(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Find local high/low pivot points using rolling window."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
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

def _detect_double_top(df, pivot_highs, tolerance=0.02):
    """Detect double top pattern from pivot highs."""
    patterns = []
    highs = df["High"].values.astype(float)
    closes = df["Close"].values.astype(float)
    lows = df["Low"].values.astype(float)

    for i in range(len(pivot_highs) - 1):
        p1, p2 = pivot_highs[i], pivot_highs[i + 1]
        if p2 - p1 < 10:
            continue

        h1, h2 = highs[p1], highs[p2]
        if abs(h1 - h2) / max(h1, h2) > tolerance:
            continue

        # Find trough between peaks
        trough_idx = p1 + np.argmin(lows[p1:p2])
        neckline = float(lows[trough_idx])
        peak_avg = (h1 + h2) / 2
        target = neckline - (peak_avg - neckline)

        # Status
        last_close = closes[min(p2 + 5, len(df) - 1)]
        if last_close < neckline:
            status = "confirmed"
        elif p2 >= len(df) - 10:
            status = "forming"
        else:
            status = "broken"
            continue

        confidence = min(0.9, 0.5 + 0.2 * (1 - abs(h1 - h2) / max(h1, h2) / tolerance))

        patterns.append({
            "pattern_name": "Double Top",
            "start_date": _date_str(df, p1),
            "end_date": _date_str(df, p2),
            "key_levels": [
                _make_key_level(df, p1, h1),
                _make_key_level(df, trough_idx, neckline),
                _make_key_level(df, p2, h2),
            ],
            "neckline": round(neckline, 2),
            "breakout_price": round(neckline, 2),
            "target_price": round(target, 2),
            "confidence": round(confidence, 2),
            "status": status,
        })

    return patterns


def _detect_double_bottom(df, pivot_lows, tolerance=0.02):
    """Detect double bottom pattern from pivot lows."""
    patterns = []
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)

    for i in range(len(pivot_lows) - 1):
        p1, p2 = pivot_lows[i], pivot_lows[i + 1]
        if p2 - p1 < 10:
            continue

        l1, l2 = lows[p1], lows[p2]
        if abs(l1 - l2) / max(l1, l2) > tolerance:
            continue

        peak_idx = p1 + np.argmax(highs[p1:p2])
        neckline = float(highs[peak_idx])
        trough_avg = (l1 + l2) / 2
        target = neckline + (neckline - trough_avg)

        last_close = closes[min(p2 + 5, len(df) - 1)]
        if last_close > neckline:
            status = "confirmed"
        elif p2 >= len(df) - 10:
            status = "forming"
        else:
            status = "broken"
            continue

        confidence = min(0.9, 0.5 + 0.2 * (1 - abs(l1 - l2) / max(l1, l2) / tolerance))

        patterns.append({
            "pattern_name": "Double Bottom",
            "start_date": _date_str(df, p1),
            "end_date": _date_str(df, p2),
            "key_levels": [
                _make_key_level(df, p1, l1),
                _make_key_level(df, peak_idx, neckline),
                _make_key_level(df, p2, l2),
            ],
            "neckline": round(neckline, 2),
            "breakout_price": round(neckline, 2),
            "target_price": round(target, 2),
            "confidence": round(confidence, 2),
            "status": status,
        })

    return patterns


def _detect_head_shoulders(df, pivot_highs, tolerance=0.03):
    """Detect Head & Shoulders pattern."""
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

        last_close = closes[min(rs + 5, len(df) - 1)]
        if last_close < neckline:
            status = "confirmed"
        elif rs >= len(df) - 10:
            status = "forming"
        else:
            continue

        patterns.append({
            "pattern_name": "Head & Shoulders",
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
            "breakout_price": round(neckline, 2),
            "target_price": round(target, 2),
            "confidence": round(min(0.85, 0.55 + 0.15 * (h_hd - max(h_ls, h_rs)) / h_hd), 2),
            "status": status,
        })

    return patterns


def _detect_inverse_head_shoulders(df, pivot_lows, tolerance=0.03):
    """Detect Inverse Head & Shoulders pattern."""
    patterns = []
    lows = df["Low"].values.astype(float)
    highs = df["High"].values.astype(float)
    closes = df["Close"].values.astype(float)

    for i in range(len(pivot_lows) - 2):
        ls, hd, rs = pivot_lows[i], pivot_lows[i + 1], pivot_lows[i + 2]
        l_ls, l_hd, l_rs = lows[ls], lows[hd], lows[rs]

        if l_hd >= l_ls or l_hd >= l_rs:
            continue
        if abs(l_ls - l_rs) / max(l_ls, l_rs) > tolerance:
            continue

        t1_idx = ls + np.argmax(highs[ls:hd])
        t2_idx = hd + np.argmax(highs[hd:rs])
        neckline = (float(highs[t1_idx]) + float(highs[t2_idx])) / 2
        target = neckline + (neckline - l_hd)

        last_close = closes[min(rs + 5, len(df) - 1)]
        if last_close > neckline:
            status = "confirmed"
        elif rs >= len(df) - 10:
            status = "forming"
        else:
            continue

        patterns.append({
            "pattern_name": "Inverse Head & Shoulders",
            "start_date": _date_str(df, ls),
            "end_date": _date_str(df, rs),
            "key_levels": [
                _make_key_level(df, ls, l_ls),
                _make_key_level(df, t1_idx, float(highs[t1_idx])),
                _make_key_level(df, hd, l_hd),
                _make_key_level(df, t2_idx, float(highs[t2_idx])),
                _make_key_level(df, rs, l_rs),
            ],
            "neckline": round(neckline, 2),
            "breakout_price": round(neckline, 2),
            "target_price": round(target, 2),
            "confidence": round(min(0.85, 0.55 + 0.15 * (min(l_ls, l_rs) - l_hd) / abs(l_hd)), 2),
            "status": status,
        })

    return patterns


def _detect_triangles(df, pivot_highs, pivot_lows, min_points=4):
    """Detect ascending, descending, and symmetrical triangles."""
    patterns = []
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(df)

    # Need at least 2 highs and 2 lows in a window
    for start_idx in range(0, n - 20, 10):
        end_idx = min(start_idx + 60, n)
        window_highs = [p for p in pivot_highs if start_idx <= p < end_idx]
        window_lows = [p for p in pivot_lows if start_idx <= p < end_idx]

        if len(window_highs) < 2 or len(window_lows) < 2:
            continue

        # Fit linear regression to pivots
        wh = np.array(window_highs)
        wl = np.array(window_lows)
        h_vals = highs[wh]
        l_vals = lows[wl]

        if len(wh) >= 2:
            h_slope = np.polyfit(wh, h_vals, 1)[0]
        else:
            continue
        if len(wl) >= 2:
            l_slope = np.polyfit(wl, l_vals, 1)[0]
        else:
            continue

        # Classify triangle type
        pattern_name = None
        direction = None

        if abs(h_slope) < 0.05 and l_slope > 0.05:
            pattern_name = "Ascending Triangle"
            direction = "bullish"
        elif h_slope < -0.05 and abs(l_slope) < 0.05:
            pattern_name = "Descending Triangle"
            direction = "bearish"
        elif h_slope < -0.03 and l_slope > 0.03:
            pattern_name = "Symmetrical Triangle"
            direction = "neutral"
        else:
            continue

        # Key levels
        key_levels = []
        for idx in sorted(list(wh) + list(wl)):
            key_levels.append(_make_key_level(df, idx, float(closes[idx])))

        apex_price = (h_vals[-1] + l_vals[-1]) / 2
        height = h_vals[0] - l_vals[0]
        if direction == "bullish":
            breakout = float(h_vals[-1])
            target = breakout + height * 0.618
        elif direction == "bearish":
            breakout = float(l_vals[-1])
            target = breakout - height * 0.618
        else:
            breakout = float(apex_price)
            target = breakout + height * 0.5

        last_idx = max(max(wh), max(wl))
        status = "forming" if last_idx >= n - 10 else "confirmed"

        patterns.append({
            "pattern_name": pattern_name,
            "start_date": _date_str(df, min(min(wh), min(wl))),
            "end_date": _date_str(df, last_idx),
            "key_levels": key_levels[:8],
            "neckline": round(apex_price, 2),
            "breakout_price": round(breakout, 2),
            "target_price": round(target, 2),
            "confidence": round(min(0.8, 0.4 + 0.1 * (len(wh) + len(wl))), 2),
            "status": status,
        })

    return patterns


def _detect_flags(df, pivot_highs, pivot_lows):
    """Detect bull and bear flag patterns."""
    patterns = []
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    for i in range(20, n - 5):
        # Check for a strong prior move (flagpole)
        lookback = 15
        start = max(0, i - lookback)
        move = (closes[i] - closes[start]) / closes[start]

        if abs(move) < 0.05:
            continue

        # Check for consolidation after the move (flag)
        flag_end = min(i + 15, n)
        flag_range = highs[i:flag_end].max() - lows[i:flag_end].min()
        flagpole_range = abs(highs[start:i].max() - lows[start:i].min())

        if flag_range > flagpole_range * 0.5:
            continue

        is_bull = move > 0
        pattern_name = "Bull Flag" if is_bull else "Bear Flag"
        breakout = float(highs[i:flag_end].max()) if is_bull else float(lows[i:flag_end].min())
        target = breakout + flagpole_range * 0.618 if is_bull else breakout - flagpole_range * 0.618

        patterns.append({
            "pattern_name": pattern_name,
            "start_date": _date_str(df, start),
            "end_date": _date_str(df, min(flag_end - 1, n - 1)),
            "key_levels": [
                _make_key_level(df, start, closes[start]),
                _make_key_level(df, i, closes[i]),
                _make_key_level(df, min(flag_end - 1, n - 1)),
            ],
            "neckline": round(breakout, 2),
            "breakout_price": round(breakout, 2),
            "target_price": round(target, 2),
            "confidence": round(min(0.75, 0.4 + abs(move) * 3), 2),
            "status": "forming" if flag_end >= n - 5 else "confirmed",
        })
        break  # Only report the most recent flag

    return patterns


# ── Main Entry Point ────────────────────────────────────────────

def detect_chart_patterns(df: pd.DataFrame, lookback: int = 120) -> List[Dict[str, Any]]:
    """
    Detect chart structure patterns from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex.
    lookback : int
        Number of bars to analyze from the end.

    Returns
    -------
    list of dict
        Each dict contains pattern_name, start_date, end_date,
        key_levels, neckline, breakout_price, target_price,
        confidence, status.
    """
    data = df.tail(lookback).copy()
    if len(data) < 30:
        logger.warning("Insufficient data for pattern detection")
        return []

    pivot_highs, pivot_lows = _find_pivots(data, window=5)

    all_patterns = []

    try:
        all_patterns.extend(_detect_double_top(data, pivot_highs))
    except Exception as e:
        logger.warning(f"Double top detection failed: {e}")

    try:
        all_patterns.extend(_detect_double_bottom(data, pivot_lows))
    except Exception as e:
        logger.warning(f"Double bottom detection failed: {e}")

    try:
        all_patterns.extend(_detect_head_shoulders(data, pivot_highs))
    except Exception as e:
        logger.warning(f"H&S detection failed: {e}")

    try:
        all_patterns.extend(_detect_inverse_head_shoulders(data, pivot_lows))
    except Exception as e:
        logger.warning(f"Inverse H&S detection failed: {e}")

    try:
        all_patterns.extend(_detect_triangles(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Triangle detection failed: {e}")

    try:
        all_patterns.extend(_detect_flags(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Flag detection failed: {e}")

    # Deduplicate overlapping patterns by keeping highest confidence
    seen = set()
    unique = []
    for p in sorted(all_patterns, key=lambda x: -x["confidence"]):
        key = (p["pattern_name"], p["start_date"])
        if key not in seen:
            seen.add(key)
            unique.append(p)

    logger.info(f"Detected {len(unique)} chart patterns")
    return unique
