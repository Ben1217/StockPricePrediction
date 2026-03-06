"""
Signal Confluence Analyzer.

Combines pattern detection, technical indicators, and ML prediction signals
into an overall signal strength assessment.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from ..utils.logger import get_logger

logger = get_logger(__name__)


def compute_confluence(
    indicators: Dict[str, float],
    pattern_direction: Optional[str] = None,
    ml_direction: Optional[str] = None,
    ml_confidence: float = 0.0,
) -> Dict[str, Any]:
    """
    Derive overall signal strength from multiple signal sources.

    Parameters
    ----------
    indicators : dict
        Latest indicator values (RSI, MACD, MACD_Signal, etc.)
    pattern_direction : str or None
        "bullish", "bearish", or None
    ml_direction : str or None
        "up" or "down"
    ml_confidence : float
        ML model confidence (0-100)

    Returns
    -------
    dict with rsi_signal, rsi_value, macd_signal, pattern_signal,
    ml_direction, ml_confidence, overall, strength
    """
    signals = []

    # ── RSI Signal ──────────────────────────────────────────
    rsi = indicators.get("RSI")
    if rsi is not None:
        if rsi > 70:
            rsi_signal = "overbought"
            signals.append(-1)
        elif rsi < 30:
            rsi_signal = "oversold"
            signals.append(1)
        else:
            rsi_signal = "neutral"
            signals.append(0)
    else:
        rsi_signal = "unknown"
        rsi = 0.0

    # ── MACD Signal ─────────────────────────────────────────
    macd = indicators.get("MACD")
    macd_sig = indicators.get("MACD_Signal")
    macd_hist = indicators.get("MACD_Histogram")

    if macd is not None and macd_sig is not None:
        if macd > macd_sig and (macd_hist or 0) > 0:
            macd_signal = "bullish_cross"
            signals.append(1)
        elif macd < macd_sig and (macd_hist or 0) < 0:
            macd_signal = "bearish_cross"
            signals.append(-1)
        else:
            macd_signal = "neutral"
            signals.append(0)
    else:
        macd_signal = "unknown"

    # ── Pattern Signal ──────────────────────────────────────
    if pattern_direction == "bullish":
        signals.append(1)
        pattern_signal = "bullish"
    elif pattern_direction == "bearish":
        signals.append(-1)
        pattern_signal = "bearish"
    else:
        pattern_signal = "none"
        signals.append(0)

    # ── ML Signal ───────────────────────────────────────────
    if ml_direction == "up":
        ml_weight = min(1.0, ml_confidence / 100.0)
        signals.append(ml_weight)
    elif ml_direction == "down":
        ml_weight = min(1.0, ml_confidence / 100.0)
        signals.append(-ml_weight)
    else:
        ml_direction = ml_direction or "neutral"

    # ── Overall ─────────────────────────────────────────────
    if signals:
        avg = sum(signals) / len(signals)
        if avg > 0.5:
            overall = "Strong Buy"
        elif avg > 0.15:
            overall = "Buy"
        elif avg < -0.5:
            overall = "Strong Sell"
        elif avg < -0.15:
            overall = "Sell"
        else:
            overall = "Neutral"
        strength = round(abs(avg), 2)
    else:
        overall = "Neutral"
        strength = 0.0

    return {
        "rsi_signal": rsi_signal,
        "rsi_value": round(float(rsi), 1),
        "macd_signal": macd_signal,
        "pattern_signal": pattern_signal,
        "ml_direction": ml_direction or "neutral",
        "ml_confidence": round(float(ml_confidence), 1),
        "overall": overall,
        "strength": strength,
    }
