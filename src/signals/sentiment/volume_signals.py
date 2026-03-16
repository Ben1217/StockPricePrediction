"""
Volume Flow Signals — P0 / P2

Volume flow signals reveal whether smart money is accumulating or
distributing.  OBV divergence and CMF are the most robust on daily bars.

Required input columns (ohlcv_df):
    ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    Note: column names are case-insensitive; the functions handle both
    capitalised ('Close') and lowercase ('close') conventions.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .models import SignalOutput
from .signal_processor import (
    build_signal_output,
    get_config_value,
    z_score_rolling,
    normalise_to_unit,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column normaliser
# ---------------------------------------------------------------------------

def _normalise_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical lowercase column names."""
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


# ===================================================================
# P0  —  VWAP Deviation Z-score
# ===================================================================

def compute_vwap_z(
    ohlcv_df: pd.DataFrame,
    window: int = 60,
    regime: str = "neutral",
) -> SignalOutput:
    """
    VWAP Deviation Z-score.

    Formula:
        VWAP = cumsum(typical_price * volume) / cumsum(volume)
        deviation = Close - VWAP
        z = (deviation - rolling_mean(deviation)) / rolling_std(deviation)

    Z > +2 or Z < -2 triggers mean-reversion signal.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    window : int
        Rolling window for deviation z-score.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: volume.vwap_z
    """
    df = _normalise_cols(ohlcv_df)
    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return _stale("volume.vwap_z", regime)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    deviation = df["close"] - vwap

    if deviation.dropna().empty:
        return _stale("volume.vwap_z", regime)

    raw_val = float(deviation.iloc[-1]) if not pd.isna(deviation.iloc[-1]) else 0.0

    ts = _get_ts(df)
    lookback = get_config_value("lookbacks.volume.vwap_z", window)

    return build_signal_output(
        name="volume.vwap_z",
        raw_value=raw_val,
        series_for_z=deviation.dropna(),
        timestamp=ts,
        lookback_bars=lookback,
        regime=regime,
        confidence=_completeness(deviation),
        z_window=lookback,
    )


# ===================================================================
# P2  —  Chaikin Money Flow (CMF)
# ===================================================================

def compute_cmf(
    ohlcv_df: pd.DataFrame,
    window: int = 20,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Chaikin Money Flow (CMF).

    Formula:
        MFM = ((2*Close - High - Low) / (High - Low))
        CMF = sum(MFM * Volume, N) / sum(Volume, N)

    > 0 = accumulation;  < 0 = distribution.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    window : int
        Lookback window (default: 20 bars).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: volume.cmf_20
    """
    df = _normalise_cols(ohlcv_df)
    required = {"high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return _stale("volume.cmf_20", regime)

    hl_range = df["high"] - df["low"]
    mfm = ((2 * df["close"] - df["high"] - df["low"]) / hl_range.replace(0, np.nan))
    mfv = mfm * df["volume"]

    cmf = mfv.rolling(window, min_periods=1).sum() / df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)

    if cmf.dropna().empty:
        return _stale("volume.cmf_20", regime)

    raw_val = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0.0
    ts = _get_ts(df)

    return build_signal_output(
        name="volume.cmf_20",
        raw_value=raw_val,
        series_for_z=cmf.dropna(),
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=_completeness(cmf),
        z_window=get_config_value("lookbacks.volume.cmf_20", 20),
    )


# ===================================================================
# P2  —  OBV Divergence
# ===================================================================

def compute_obv_divergence(
    ohlcv_df: pd.DataFrame,
    window: int = 60,
    regime: str = "neutral",
) -> SignalOutput:
    """
    On-Balance Volume (OBV) divergence detector.

    OBV_t = OBV_(t-1) + sign(Close - Prev_Close) * Volume

    Divergence = price slope and OBV slope have opposite signs over
    a trailing window → potential distribution / accumulation mismatch.

    Parameters
    ----------
    ohlcv_df : pd.DataFrame
        OHLCV data.
    window : int
        Lookback window for slope comparison.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: volume.obv_divergence
        value = 1.0 (bearish divergence), -1.0 (bullish divergence), 0 (no divergence)
    """
    df = _normalise_cols(ohlcv_df)
    required = {"close", "volume"}
    if not required.issubset(df.columns):
        return _stale("volume.obv_divergence", regime)

    sign = np.sign(df["close"].diff())
    obv = (sign * df["volume"]).fillna(0).cumsum()

    if len(obv) < window:
        return _stale("volume.obv_divergence", regime)

    # Compute slopes via linear regression over trailing window
    x = np.arange(window, dtype=float)
    price_tail = df["close"].iloc[-window:].values.astype(float)
    obv_tail = obv.iloc[-window:].values.astype(float)

    price_slope = _simple_slope(x, price_tail)
    obv_slope = _simple_slope(x, obv_tail)

    # divergence flag
    if price_slope > 0 and obv_slope < 0:
        raw_val = 1.0   # bearish divergence
    elif price_slope < 0 and obv_slope > 0:
        raw_val = -1.0   # bullish divergence
    else:
        raw_val = 0.0

    ts = _get_ts(df)
    # Build a synthetic series for z-score from trailing OBV
    return build_signal_output(
        name="volume.obv_divergence",
        raw_value=raw_val,
        series_for_z=obv.dropna(),
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=_completeness(obv),
        z_window=get_config_value("lookbacks.volume.obv_divergence", 60),
        meta={"price_slope": price_slope, "obv_slope": obv_slope},
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _simple_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y on x."""
    if len(x) < 2:
        return 0.0
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return 0.0
    x_m, y_m = x[mask], y[mask]
    x_bar = x_m.mean()
    y_bar = y_m.mean()
    denom = ((x_m - x_bar) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x_m - x_bar) * (y_m - y_bar)).sum() / denom)


def _get_ts(df: pd.DataFrame) -> int:
    """Extract unix ms timestamp from the last row."""
    if "timestamp" in df.columns:
        return int(df["timestamp"].iloc[-1])
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return int(df.index[-1].timestamp() * 1000)
    return 0


def _completeness(s: pd.Series) -> float:
    """Fraction of non-NaN values → confidence proxy."""
    if len(s) == 0:
        return 0.0
    return min(float(s.notna().sum()) / len(s), 1.0)
