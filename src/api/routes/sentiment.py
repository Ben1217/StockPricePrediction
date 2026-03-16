"""
Sentiment API routes — compute and return technical sentiment signals.

This module provides a single endpoint that:
1. Fetches OHLCV data for the given symbol via yfinance.
2. Computes all available sentiment signals (volume-based: VWAP-Z, CMF, OBV;
   momentum: RSI divergence, MACD exhaustion, ROC-Z).
3. Returns individual signal details plus the weighted composite score.

Signals requiring external data feeds (options chain, VIX futures, tick data,
exchange breadth, COT) are returned as stale with reason "data_not_available"
until those feeds are connected.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Query, HTTPException

from src.signals.sentiment.models import SignalOutput
from src.signals.sentiment.composite_score import composite_score
from src.signals.sentiment.signal_processor import get_regime

# Importable signal computers
from src.signals.sentiment.volume_signals import compute_vwap_z, compute_cmf, compute_obv_divergence
from src.signals.sentiment.momentum_signals import compute_rsi_divergence, compute_macd_exhaustion, compute_roc_z

logger = logging.getLogger(__name__)
router = APIRouter()


def _stale_placeholder(name: str, regime: str = "neutral") -> dict:
    """Return a dict for signals whose data feeds are not yet connected."""
    return {
        "name": name, "value": 0.0, "z_score": 0.0, "normalised": 0.0,
        "regime": regime, "confidence": 0.0, "timestamp": 0,
        "lookback_bars": 0, "is_stale": True,
        "meta": {"reason": "data_not_available"},
    }


def _signal_to_dict(s: SignalOutput) -> dict:
    return {
        "name": s.name,
        "value": round(s.value, 4),
        "z_score": round(s.z_score, 4),
        "normalised": round(s.normalised, 4),
        "regime": s.regime,
        "confidence": round(s.confidence, 4),
        "timestamp": s.timestamp,
        "lookback_bars": s.lookback_bars,
        "is_stale": s.is_stale,
        "meta": s.meta,
    }


@router.get("/{symbol}")
async def get_sentiment(
    symbol: str,
    days: int = Query(120, ge=30, le=500),
):
    """
    Compute technical sentiment signals for a symbol.

    Returns a composite score in [-1, +1] and per-signal breakdowns.
    Signals whose data feeds are not yet connected are marked as stale.
    """
    symbol = symbol.upper()

    # Fetch OHLCV
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days + 200)).strftime("%Y-%m-%d")
    try:
        df = yf.download(symbol, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception as e:
        raise HTTPException(502, f"Data fetch failed for {symbol}: {e}")

    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")

    # Normalise column names to lowercase for our signal functions
    ohlcv = df.copy()
    ohlcv.columns = [c.lower() for c in ohlcv.columns]

    # Add timestamp column from index
    if hasattr(ohlcv.index, "dtype") and np.issubdtype(ohlcv.index.dtype, np.datetime64):
        ohlcv["timestamp"] = (ohlcv.index.astype(np.int64) // 10**6).astype(int)

    # ── Compute available signals ─────────────────────────────────
    signals_dict = {}
    signal_details = []

    # P0: VWAP-Z (computable from OHLCV)
    try:
        vwap_z = compute_vwap_z(ohlcv)
        signals_dict["volume.vwap_z"] = vwap_z
        signal_details.append(_signal_to_dict(vwap_z))
    except Exception as e:
        logger.warning("VWAP-Z computation failed: %s", e)
        signal_details.append(_stale_placeholder("volume.vwap_z"))

    # P2: CMF (computable from OHLCV)
    try:
        cmf = compute_cmf(ohlcv)
        signals_dict["volume.cmf_20"] = cmf
        signal_details.append(_signal_to_dict(cmf))
    except Exception as e:
        logger.warning("CMF computation failed: %s", e)
        signal_details.append(_stale_placeholder("volume.cmf_20"))

    # P2: OBV divergence (computable from OHLCV)
    try:
        obv = compute_obv_divergence(ohlcv, window=60)
        signals_dict["volume.obv_divergence"] = obv
        signal_details.append(_signal_to_dict(obv))
    except Exception as e:
        logger.warning("OBV divergence computation failed: %s", e)
        signal_details.append(_stale_placeholder("volume.obv_divergence"))

    # P3: RSI divergence (computable from OHLCV)
    try:
        rsi_div = compute_rsi_divergence(ohlcv)
        signals_dict["momentum.rsi_divergence"] = rsi_div
        signal_details.append(_signal_to_dict(rsi_div))
    except Exception as e:
        logger.warning("RSI divergence computation failed: %s", e)
        signal_details.append(_stale_placeholder("momentum.rsi_divergence"))

    # P3: MACD exhaustion (computable from OHLCV)
    try:
        macd_ex = compute_macd_exhaustion(ohlcv)
        signals_dict["momentum.macd_exhaust"] = macd_ex
        signal_details.append(_signal_to_dict(macd_ex))
    except Exception as e:
        logger.warning("MACD exhaustion failed: %s", e)
        signal_details.append(_stale_placeholder("momentum.macd_exhaust"))

    # P3: ROC Z-score (computable from OHLCV)
    try:
        roc = compute_roc_z(ohlcv)
        signals_dict["momentum.roc_z"] = roc
        signal_details.append(_signal_to_dict(roc))
    except Exception as e:
        logger.warning("ROC-Z computation failed: %s", e)
        signal_details.append(_stale_placeholder("momentum.roc_z"))

    # ── Signals requiring external data (stale placeholders) ──────
    for hook in [
        "micro.ofi", "vol.iv_rank", "vol.term_slope",
        "options.pcr_5d", "breadth.mcclellan", "positioning.cot_z",
    ]:
        signal_details.append(_stale_placeholder(hook))

    # ── Composite score ───────────────────────────────────────────
    score = composite_score(signals_dict)

    # Determine overall regime from available signals
    regime = "neutral"
    if "vol.term_slope" in signals_dict and not signals_dict["vol.term_slope"].is_stale:
        regime = signals_dict["vol.term_slope"].regime

    # Interpret composite score
    if score > 0.3:
        interpretation = "Bullish"
    elif score > 0.1:
        interpretation = "Slightly Bullish"
    elif score < -0.3:
        interpretation = "Bearish"
    elif score < -0.1:
        interpretation = "Slightly Bearish"
    else:
        interpretation = "Neutral"

    # Count active vs stale
    active_count = sum(1 for s in signal_details if not s["is_stale"])
    total_count = len(signal_details)

    return {
        "symbol": symbol,
        "composite_score": round(score, 4),
        "interpretation": interpretation,
        "regime": regime,
        "active_signals": active_count,
        "total_signals": total_count,
        "signals": signal_details,
    }
