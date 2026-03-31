"""
3-Indicator Sentiment Engine

Rule-based sentiment classifier using three technical indicators:
  1. 200-day Moving Average (Trend)
  2. RSI 14 crossover (Momentum)
  3. Volume vs 20-day average (Volume Confirmation)

Produces a score from -3 to +3 and a human-readable classification.
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


# ── Score → label mapping ──────────────────────────────────────────────────
_SENTIMENT_MAP = {
    4: "Extreme Bullish",
    3: "Strong Bullish",
    2: "Bullish",
    1: "Slightly Bullish",
    0: "Neutral",
    -1: "Slightly Bearish",
    -2: "Bearish",
    -3: "Strong Bearish",
    -4: "Extreme Bearish",
}


def _sentiment_label(score: int) -> str:
    return _SENTIMENT_MAP.get(score, "Neutral")


# ── Score → confidence % (signal strength, NOT prediction probability) ─────
_CONFIDENCE_MAP = {4: 95, 3: 85, 2: 70, 1: 50, 0: 25, -1: 50, -2: 70, -3: 85, -4: 95}

# ── Score → strength label ─────────────────────────────────────────────────
_CONFIDENCE_LABEL_MAP = {
    4: "Extreme Bullish", 3: "Strong Bullish", 2: "Bullish", 1: "Weak Bullish",
    0: "Neutral",
    -1: "Weak Bearish", -2: "Bearish", -3: "Strong Bearish", -4: "Extreme Bearish",
}


# ── Individual indicator functions ─────────────────────────────────────────

def compute_ma_signal(close: float, sma_200: float) -> int:
    """
    Trend signal based on 200-day Moving Average.

    Returns +1 if close > SMA 200 (bullish), -1 otherwise (bearish).
    """
    if pd.isna(sma_200):
        return 0
    return 1 if close > sma_200 else -1


def compute_rsi_signal(rsi_current: float, rsi_previous: float) -> int:
    """
    Momentum signal based on RSI-14 crossover.

    - RSI crosses above 30 from below  → +1 (bullish)
    - RSI crosses below 70 from above  → -1 (bearish)
    - RSI between 40–60                → 0  (neutral)
    - Otherwise                        → 0  (neutral)
    """
    if pd.isna(rsi_current) or pd.isna(rsi_previous):
        return 0

    # Bullish: RSI crossed above 30 from below
    if rsi_previous < 30 and rsi_current >= 30:
        return 1

    # Bearish: RSI crossed below 70 from above
    if rsi_previous > 70 and rsi_current <= 70:
        return -1

    # Neutral zone
    if 40 <= rsi_current <= 60:
        return 0

    return 0


def compute_volume_signal(
    volume: float,
    avg_volume_20: float,
    is_bullish_candle: bool,
) -> int:
    """
    Volume confirmation signal.

    - Volume > 20-day avg AND bullish candle → +1
    - Volume > 20-day avg AND bearish candle → -1
    - Otherwise                              → 0
    """
    if pd.isna(volume) or pd.isna(avg_volume_20) or avg_volume_20 == 0:
        return 0

    if volume > avg_volume_20:
        return 1 if is_bullish_candle else -1

    return 0


def compute_sr_signal(close: float, df: pd.DataFrame) -> tuple[int, dict]:
    """
    Support & Resistance proximity signal.

    - Near support (<= 5% distance) → +1 (bullish)
    - Near resistance (<= 5% distance) → -1 (bearish)
    """
    from src.features.support_resistance import detect_support_resistance
    
    sr_data = detect_support_resistance(df, close)
    levels = sr_data.get("levels", [])
    
    support = next((l["price"] for l in levels if l["type"] == "support"), None)
    resistance = next((l["price"] for l in levels if l["type"] == "resistance"), None)
    
    dist_support = None
    dist_resistance = None
    sr_sig = 0
    
    if support is not None:
        dist_support = round(((close - support) / support) * 100, 2)
        if 0 <= dist_support <= 5.0:
            sr_sig = 1
            
    if resistance is not None:
        dist_resistance = round(((resistance - close) / close) * 100, 2)
        if 0 <= dist_resistance <= 5.0:
            sr_sig = -1
            
    return sr_sig, {
        "support": support,
        "resistance": resistance,
        "dist_support": dist_support,
        "dist_resistance": dist_resistance
    }


# ── Main computation ───────────────────────────────────────────────────────

def compute_indicator_sentiment(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute the 3-indicator sentiment for the most recent bar.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: Open, High, Low, Close, Volume.
        Must have at least 201 rows for the 200 MA.

    Returns
    -------
    dict
        Full sentiment output including individual signals, score,
        sentiment label, entry signal, and supporting detail values.
    """
    data = df.copy()

    # Ensure column names are title-case
    col_map = {c: c.title() for c in data.columns if c.lower() in
               ("open", "high", "low", "close", "volume")}
    data.rename(columns=col_map, inplace=True)

    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # ── Compute indicators ───────────────────────────────────────
    # 200 SMA
    data["SMA_200"] = ta.trend.sma_indicator(data["Close"], window=200)

    # RSI 14
    data["RSI"] = ta.momentum.rsi(data["Close"], window=14)

    # 20-day average volume
    data["Avg_Vol_20"] = data["Volume"].rolling(window=20, min_periods=20).mean()

    # ── Extract latest values ────────────────────────────────────
    if len(data) < 2:
        raise ValueError("Need at least 2 rows for RSI crossover detection")

    current = data.iloc[-1]
    previous = data.iloc[-2]

    close = float(current["Close"])
    open_price = float(current["Open"])
    sma_200 = float(current["SMA_200"]) if not pd.isna(current["SMA_200"]) else None
    rsi_current = float(current["RSI"]) if not pd.isna(current["RSI"]) else None
    rsi_previous = float(previous["RSI"]) if not pd.isna(previous["RSI"]) else None
    volume = float(current["Volume"])
    avg_vol_20 = float(current["Avg_Vol_20"]) if not pd.isna(current["Avg_Vol_20"]) else None
    is_bullish_candle = close > open_price

    # ── Compute signals ──────────────────────────────────────────
    ma_sig = compute_ma_signal(close, sma_200) if sma_200 is not None else 0
    rsi_sig = compute_rsi_signal(rsi_current, rsi_previous)
    vol_sig = compute_volume_signal(volume, avg_vol_20, is_bullish_candle)
    sr_sig, sr_details = compute_sr_signal(close, data)

    score = ma_sig + rsi_sig + vol_sig + sr_sig

    # ── Classify ─────────────────────────────────────────────────
    sentiment = _sentiment_label(score)

    # Entry signal logic
    if score >= 2:
        entry_signal = "BULLISH"
    elif score <= -2:
        entry_signal = "BEARISH"
    else:
        entry_signal = "WAIT"

    # ── Signal labels ────────────────────────────────────────────
    def _sig_label(val: int) -> str:
        if val > 0:
            return "bullish"
        elif val < 0:
            return "bearish"
        return "neutral"

    # ── Confidence & labels ───────────────────────────────────────
    confidence = _CONFIDENCE_MAP.get(score, 25)
    confidence_label = _CONFIDENCE_LABEL_MAP.get(score, "Neutral")
    volume_strength = "strong" if (avg_vol_20 and volume > avg_vol_20) else "weak"

    return {
        "trend": _sig_label(ma_sig),
        "trend_score": ma_sig,
        "rsi_signal": _sig_label(rsi_sig),
        "rsi_score": rsi_sig,
        "volume_signal": _sig_label(vol_sig),
        "volume_score": vol_sig,
        "sr_signal": _sig_label(sr_sig),
        "sr_score": sr_sig,
        "score": score,
        "sentiment": sentiment,
        "entry_signal": entry_signal,
        "confidence": confidence,
        "confidence_label": confidence_label,
        "volume_strength": volume_strength,
        "details": {
            "close": round(close, 2),
            "sma_200": round(sma_200, 2) if sma_200 is not None else None,
            "rsi": round(rsi_current, 2) if rsi_current is not None else None,
            "rsi_prev": round(rsi_previous, 2) if rsi_previous is not None else None,
            "volume": int(volume),
            "avg_volume_20": int(avg_vol_20) if avg_vol_20 is not None else None,
            "is_bullish_candle": is_bullish_candle,
            **sr_details,
        },
    }
