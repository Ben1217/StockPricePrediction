"""
Momentum / Divergence Signals — P3

Divergence signals detect when price and an underlying oscillator
decouple — one of the most reliable early-warning signals across
timeframes.

Required input columns (ohlcv_df):
    ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    (case-insensitive)
"""

import logging

import numpy as np
import pandas as pd

from .models import SignalOutput
from .signal_processor import (
    build_signal_output,
    get_config_value,
    z_score_rolling,
)

logger = logging.getLogger(__name__)


def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    col_map = {}
    for c in out.columns:
        if c.lower() in {"open", "high", "low", "close", "volume", "timestamp"}:
            col_map[c] = c.lower()
    out.rename(columns=col_map, inplace=True)
    return out


def _stale(name: str, regime: str) -> SignalOutput:
    return SignalOutput(
        name=name, value=0.0, z_score=0.0, normalised=0.0,
        regime=regime, confidence=0.0, timestamp=0,
        lookback_bars=0, is_stale=True, meta={"reason": "insufficient_data"},
    )


def _simple_slope(arr: np.ndarray) -> float:
    """OLS slope of arr against [0, 1, 2, ...]."""
    n = len(arr)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return 0.0
    x_m, y_m = x[mask], arr[mask]
    x_bar, y_bar = x_m.mean(), y_m.mean()
    denom = ((x_m - x_bar) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x_m - x_bar) * (y_m - y_bar)).sum() / denom)


# ===================================================================
# RSI Divergence
# ===================================================================

def compute_rsi_divergence(
    ohlcv_df: pd.DataFrame,
    rsi_period: int = 14,
    slope_window: int = 20,
    regime: str = "neutral",
) -> SignalOutput:
    """
    RSI Divergence Flag.

    Bearish divergence: price_slope(N) > 0 AND rsi_slope(N) < 0
    Bullish divergence: price_slope(N) < 0 AND rsi_slope(N) > 0

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    rsi_period : int
        RSI calculation period (default: 14).
    slope_window : int
        Window for slope comparison (default: 20).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: momentum.rsi_divergence
        value: +1 (bearish div), -1 (bullish div), 0 (no div)
    """
    df = _normalise_cols(ohlcv_df)
    if "close" not in df.columns:
        return _stale("momentum.rsi_divergence", regime)

    close = df["close"]

    # Compute RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    if len(rsi.dropna()) < slope_window:
        return _stale("momentum.rsi_divergence", regime)

    price_slope = _simple_slope(close.iloc[-slope_window:].values)
    rsi_slope = _simple_slope(rsi.iloc[-slope_window:].values)

    if price_slope > 0 and rsi_slope < 0:
        raw_val = 1.0   # bearish divergence
    elif price_slope < 0 and rsi_slope > 0:
        raw_val = -1.0   # bullish divergence
    else:
        raw_val = 0.0

    ts = _get_ts(df)

    return build_signal_output(
        name="momentum.rsi_divergence",
        raw_value=raw_val,
        series_for_z=rsi.dropna(),
        timestamp=ts,
        lookback_bars=slope_window,
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.momentum.rsi_divergence", 60),
        meta={"price_slope": price_slope, "rsi_slope": rsi_slope},
    )


# ===================================================================
# MACD Histogram Exhaustion
# ===================================================================

def compute_macd_exhaustion(
    ohlcv_df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    regime: str = "neutral",
) -> SignalOutput:
    """
    MACD Histogram Exhaustion.

    Histogram bars shrinking after a trend = momentum exhaustion.

    value:
        +1  = bullish exhaustion (histogram shrinking from positive peak)
        -1  = bearish exhaustion (histogram shrinking from negative trough)
         0  = no exhaustion detected

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    fast, slow, signal : int
        MACD parameters.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: momentum.macd_exhaust
    """
    df = _normalise_cols(ohlcv_df)
    if "close" not in df.columns:
        return _stale("momentum.macd_exhaust", regime)

    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    hist_clean = histogram.dropna()
    if len(hist_clean) < 3:
        return _stale("momentum.macd_exhaust", regime)

    # Check last 3 bars for shrinking histogram
    last3 = hist_clean.iloc[-3:].values
    raw_val = 0.0

    if last3[-3] > 0 and last3[-2] > 0 and last3[-1] > 0:
        # Positive histogram: check if shrinking
        if abs(last3[-1]) < abs(last3[-2]) < abs(last3[-3]):
            raw_val = 1.0  # bullish exhaustion
    elif last3[-3] < 0 and last3[-2] < 0 and last3[-1] < 0:
        # Negative histogram: check if shrinking
        if abs(last3[-1]) < abs(last3[-2]) < abs(last3[-3]):
            raw_val = -1.0  # bearish exhaustion

    ts = _get_ts(df)

    return build_signal_output(
        name="momentum.macd_exhaust",
        raw_value=raw_val,
        series_for_z=histogram.dropna(),
        timestamp=ts,
        lookback_bars=slow + signal,
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.momentum.macd_exhaust", 60),
    )


# ===================================================================
# ROC Z-score
# ===================================================================

def compute_roc_z(
    ohlcv_df: pd.DataFrame,
    roc_period: int = 10,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Rate of Change Z-score.

    Formula:
        ROC = (Close / Close_t-n - 1) * 100
        Z = (ROC - mean(ROC)) / std(ROC)

    Z > +2.5 or Z < -2.5 = extreme momentum.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    roc_period : int
        ROC lookback (default: 10).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: momentum.roc_z
    """
    df = _normalise_cols(ohlcv_df)
    if "close" not in df.columns:
        return _stale("momentum.roc_z", regime)

    close = df["close"]
    roc = (close / close.shift(roc_period) - 1) * 100

    if roc.dropna().empty:
        return _stale("momentum.roc_z", regime)

    raw_val = float(roc.iloc[-1]) if not pd.isna(roc.iloc[-1]) else 0.0
    ts = _get_ts(df)

    return build_signal_output(
        name="momentum.roc_z",
        raw_value=raw_val,
        series_for_z=roc.dropna(),
        timestamp=ts,
        lookback_bars=roc_period,
        regime=regime,
        confidence=min(float(roc.notna().sum()) / max(roc_period, 1), 1.0),
        z_window=get_config_value("lookbacks.momentum.roc_z", 60),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_ts(df: pd.DataFrame) -> int:
    if "timestamp" in df.columns:
        return int(df["timestamp"].iloc[-1])
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return int(df.index[-1].timestamp() * 1000)
    return 0
