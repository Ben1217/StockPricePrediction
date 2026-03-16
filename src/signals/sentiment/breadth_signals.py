"""
Breadth Signals — P2

Breadth signals measure market-wide participation.  They require
exchange-level data (advancing / declining issues, new highs / lows).
Applicable timeframe: swing to position (daily bars).

Required input columns (breadth_df):
    ['date', 'advancing', 'declining', 'new_highs', 'new_lows']
"""

import logging

import numpy as np
import pandas as pd

from .models import SignalOutput
from .signal_processor import (
    build_signal_output,
    get_config_value,
)

logger = logging.getLogger(__name__)


def _stale(name: str, regime: str) -> SignalOutput:
    return SignalOutput(
        name=name, value=0.0, z_score=0.0, normalised=0.0,
        regime=regime, confidence=0.0, timestamp=0,
        lookback_bars=0, is_stale=True, meta={"reason": "insufficient_data"},
    )


# ===================================================================
# McClellan Oscillator
# ===================================================================

def compute_mcclellan(
    breadth_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    McClellan Oscillator.

    Formula:
        NetAdv = Advancing - Declining
        McClellan = EMA(NetAdv, 19) - EMA(NetAdv, 39)

    Cross above 0 → bullish regime flip.

    Parameters
    ----------
    breadth_df : pd.DataFrame
        Exchange breadth data.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: breadth.mcclellan
    """
    required = {"advancing", "declining"}
    if not required.issubset(breadth_df.columns):
        return _stale("breadth.mcclellan", regime)

    df = breadth_df.copy()
    net_adv = df["advancing"] - df["declining"]

    if len(net_adv) < 39:
        return _stale("breadth.mcclellan", regime)

    ema_19 = net_adv.ewm(span=19, adjust=False).mean()
    ema_39 = net_adv.ewm(span=39, adjust=False).mean()
    mcclellan = ema_19 - ema_39

    raw_val = float(mcclellan.iloc[-1])
    ts = _get_ts(df)

    return build_signal_output(
        name="breadth.mcclellan",
        raw_value=raw_val,
        series_for_z=mcclellan.dropna(),
        timestamp=ts,
        lookback_bars=39,
        regime=regime,
        confidence=min(float(len(mcclellan.dropna())) / 39, 1.0),
        z_window=get_config_value("lookbacks.breadth.mcclellan", 60),
    )


# ===================================================================
# Advance-Decline Line
# ===================================================================

def compute_ad_line(
    breadth_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Advance-Decline Line.

    Formula:
        AD_t = AD_(t-1) + (Advancing_t - Declining_t)

    Divergence from price = breadth warning.

    Parameters
    ----------
    breadth_df : pd.DataFrame
        Exchange breadth data.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: breadth.ad_line
    """
    required = {"advancing", "declining"}
    if not required.issubset(breadth_df.columns):
        return _stale("breadth.ad_line", regime)

    df = breadth_df.copy()
    net = df["advancing"] - df["declining"]
    ad_line = net.cumsum()

    if ad_line.dropna().empty:
        return _stale("breadth.ad_line", regime)

    raw_val = float(ad_line.iloc[-1])
    ts = _get_ts(df)

    return build_signal_output(
        name="breadth.ad_line",
        raw_value=raw_val,
        series_for_z=ad_line.dropna(),
        timestamp=ts,
        lookback_bars=len(ad_line),
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.breadth.ad_line", 60),
    )


# ===================================================================
# NH-NL Ratio
# ===================================================================

def compute_nh_nl_ratio(
    breadth_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    New-Highs / New-Lows Ratio.

    Formula:
        ratio = (NewHighs - NewLows) / (NewHighs + NewLows)

    Ratio shrinking during a price rally = breadth warning.

    Parameters
    ----------
    breadth_df : pd.DataFrame
        Exchange breadth data.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: breadth.nh_nl_ratio
    """
    required = {"new_highs", "new_lows"}
    if not required.issubset(breadth_df.columns):
        return _stale("breadth.nh_nl_ratio", regime)

    df = breadth_df.copy()
    total = df["new_highs"] + df["new_lows"]
    ratio = (df["new_highs"] - df["new_lows"]) / total.replace(0, np.nan)
    ratio = ratio.dropna()

    if ratio.empty:
        return _stale("breadth.nh_nl_ratio", regime)

    raw_val = float(ratio.iloc[-1])
    ts = _get_ts(df)

    return build_signal_output(
        name="breadth.nh_nl_ratio",
        raw_value=raw_val,
        series_for_z=ratio,
        timestamp=ts,
        lookback_bars=len(ratio),
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.breadth.nh_nl_ratio", 60),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_ts(df: pd.DataFrame) -> int:
    for col in ("date", "timestamp"):
        if col in df.columns:
            val = df[col].iloc[-1]
            try:
                return int(pd.Timestamp(val).timestamp() * 1000)
            except Exception:
                pass
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return int(df.index[-1].timestamp() * 1000)
    return 0
