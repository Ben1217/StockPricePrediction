"""
Volatility Signals — P1 / P3

Volatility signals encode fear and complacency from options pricing.
Among the highest-value signals for regime classification.

Required input columns:
    options_df : ['timestamp', 'strike', 'expiry', 'type', 'iv', 'delta',
                  'gamma', 'oi', 'volume']
    ohlcv_df   : ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    vix_df     : ['timestamp', 'vx_m1', 'vx_m2']  (front two VIX futures)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .models import SignalOutput
from .signal_processor import (
    build_signal_output,
    get_config_value,
    get_regime,
)

logger = logging.getLogger(__name__)


def _stale(name: str, regime: str) -> SignalOutput:
    return SignalOutput(
        name=name, value=0.0, z_score=0.0, normalised=0.0,
        regime=regime, confidence=0.0, timestamp=0,
        lookback_bars=0, is_stale=True, meta={"reason": "insufficient_data"},
    )


# ===================================================================
# P1  —  IV Rank
# ===================================================================

def compute_iv_rank(
    options_df: pd.DataFrame,
    lookback: int = 252,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Implied Volatility Rank.

    Formula:
        IV_Rank = (IV_now - IV_52w_low) / (IV_52w_high - IV_52w_low)

    Range: [0, 1].  >0.8 = extreme fear;  <0.2 = complacency.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain an 'iv' column with a time-indexed average IV per bar.
    lookback : int
        52-week window in bars (default: 252 daily bars).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: vol.iv_rank
    """
    if "iv" not in options_df.columns:
        return _stale("vol.iv_rank", regime)

    iv = options_df["iv"].dropna()
    if len(iv) < 2:
        return _stale("vol.iv_rank", regime)

    window = min(lookback, len(iv))
    iv_window = iv.iloc[-window:]
    iv_high = iv_window.max()
    iv_low = iv_window.min()
    iv_now = iv.iloc[-1]

    if iv_high == iv_low:
        raw_val = 0.5
    else:
        raw_val = float((iv_now - iv_low) / (iv_high - iv_low))

    raw_val = max(0.0, min(1.0, raw_val))
    ts = _get_ts(options_df)

    return build_signal_output(
        name="vol.iv_rank",
        raw_value=raw_val,
        series_for_z=iv,
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=min(float(len(iv)) / lookback, 1.0),
        z_window=get_config_value("lookbacks.vol.iv_rank", 252),
        meta={"iv_52w_high": float(iv_high), "iv_52w_low": float(iv_low)},
    )


# ===================================================================
# P1  —  VIX Term Structure Slope
# ===================================================================

def compute_term_slope(
    vix_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    VIX Term Structure Slope.

    Formula:  slope = VX_M2 / VX_M1

    < 1.0 = backwardation (stress).
    > 1.0 = contango (calm).

    Parameters
    ----------
    vix_df : pd.DataFrame
        Must contain 'vx_m1' and 'vx_m2' columns.
    regime : str
        Ignored — this signal *determines* the regime.

    Returns
    -------
    SignalOutput
        hook: vol.term_slope
    """
    required = {"vx_m1", "vx_m2"}
    if not required.issubset(vix_df.columns):
        return _stale("vol.term_slope", "neutral")

    df = vix_df.dropna(subset=["vx_m1", "vx_m2"])
    if df.empty:
        return _stale("vol.term_slope", "neutral")

    slope = df["vx_m2"] / df["vx_m1"].replace(0, np.nan)
    slope = slope.dropna()
    if slope.empty:
        return _stale("vol.term_slope", "neutral")

    raw_val = float(slope.iloc[-1])
    detected_regime = get_regime(raw_val)
    ts = _get_ts(df)

    return build_signal_output(
        name="vol.term_slope",
        raw_value=raw_val,
        series_for_z=slope,
        timestamp=ts,
        lookback_bars=len(slope),
        regime=detected_regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.vol.term_slope", 60),
        meta={"vx_m1": float(df["vx_m1"].iloc[-1]), "vx_m2": float(df["vx_m2"].iloc[-1])},
    )


# ===================================================================
# P1  —  RV vs IV Spread
# ===================================================================

def compute_rv_iv_spread(
    options_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
    rv_window: int = 20,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Realized Volatility vs Implied Volatility Spread.

    Formula:  spread = IV_30d - RV_20d
    Persistently positive spread = sell-vol bias.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain 'iv' column (30-day IV).
    ohlcv_df : pd.DataFrame
        OHLCV data for RV calculation.
    rv_window : int
        Window for realized volatility (default: 20).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: vol.rv_iv_spread
    """
    if "iv" not in options_df.columns:
        return _stale("vol.rv_iv_spread", regime)

    # Normalise column names
    ohlcv = ohlcv_df.copy()
    col_map = {c: c.lower() for c in ohlcv.columns if c.lower() in {"close", "timestamp"}}
    ohlcv.rename(columns=col_map, inplace=True)

    if "close" not in ohlcv.columns:
        return _stale("vol.rv_iv_spread", regime)

    returns = np.log(ohlcv["close"] / ohlcv["close"].shift(1))
    rv = returns.rolling(rv_window).std() * np.sqrt(252)  # annualised

    iv = options_df["iv"]
    # Align lengths
    min_len = min(len(rv.dropna()), len(iv.dropna()))
    if min_len < 2:
        return _stale("vol.rv_iv_spread", regime)

    spread = iv.iloc[-min_len:].values - rv.dropna().iloc[-min_len:].values
    spread_series = pd.Series(spread)

    raw_val = float(spread[-1]) if len(spread) > 0 else 0.0
    ts = _get_ts(ohlcv)

    return build_signal_output(
        name="vol.rv_iv_spread",
        raw_value=raw_val,
        series_for_z=spread_series.dropna(),
        timestamp=ts,
        lookback_bars=rv_window,
        regime=regime,
        confidence=min(float(min_len) / rv_window, 1.0),
        z_window=get_config_value("lookbacks.vol.rv_iv_spread", 60),
    )


# ===================================================================
# P3  —  Skew (25-delta put − call)
# ===================================================================

def compute_skew_25d(
    options_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    25-Delta Put-Call Skew.

    Formula:  skew = IV_25put - IV_25call
    > 10 pts = elevated tail-risk demand.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain 'delta', 'iv', 'type' columns.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: vol.skew_25d
    """
    required = {"delta", "iv", "type"}
    if not required.issubset(options_df.columns):
        return _stale("vol.skew_25d", regime)

    df = options_df.copy()

    # Find 25-delta puts and calls (within tolerance)
    tol = 0.05
    puts = df[(df["type"] == "put") & ((df["delta"].abs() - 0.25).abs() < tol)]
    calls = df[(df["type"] == "call") & ((df["delta"].abs() - 0.25).abs() < tol)]

    if puts.empty or calls.empty:
        return _stale("vol.skew_25d", regime)

    iv_put = puts["iv"].mean()
    iv_call = calls["iv"].mean()
    raw_val = float(iv_put - iv_call)

    ts = _get_ts(df)

    # For z-score, we build a simple series from the raw value (single snapshot)
    skew_series = pd.Series([raw_val])

    return build_signal_output(
        name="vol.skew_25d",
        raw_value=raw_val,
        series_for_z=skew_series,
        timestamp=ts,
        lookback_bars=1,
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.vol.skew_25d", 60),
        meta={"iv_25put": float(iv_put), "iv_25call": float(iv_call)},
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
