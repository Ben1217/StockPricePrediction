"""
Options Flow Signals — P1

Options flow signals extract market positioning from derivatives activity.
Contrarian in nature — extreme readings mark turning points.

Required input columns (options_df):
    ['timestamp', 'strike', 'expiry', 'type',    # 'call'|'put'
     'iv', 'delta', 'gamma', 'oi', 'volume']
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .models import SignalOutput
from .signal_processor import (
    build_signal_output,
    get_config_value,
    ema_smooth,
)

logger = logging.getLogger(__name__)


def _stale(name: str, regime: str) -> SignalOutput:
    return SignalOutput(
        name=name, value=0.0, z_score=0.0, normalised=0.0,
        regime=regime, confidence=0.0, timestamp=0,
        lookback_bars=0, is_stale=True, meta={"reason": "insufficient_data"},
    )


# ===================================================================
# P1  —  Put/Call Ratio (5-day EMA)
# ===================================================================

def compute_pcr_5d(
    options_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Put/Call Ratio — 5-day EMA.

    Formula:
        daily_pcr = Put_Volume / Call_Volume
        PCR_5d    = EMA(daily_pcr, span=5)

    > 1.3 = extreme fear → mean-reversion buy signal.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain 'type' (call/put), 'volume', and a date key.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: options.pcr_5d
    """
    required = {"type", "volume"}
    if not required.issubset(options_df.columns):
        return _stale("options.pcr_5d", regime)

    df = options_df.copy()

    # Determine date grouping column
    date_col = None
    for col in ("timestamp", "date"):
        if col in df.columns:
            date_col = col
            break
    if date_col is None and hasattr(df.index, "date"):
        df["_date"] = df.index.date
        date_col = "_date"
    if date_col is None:
        # Treat entire frame as one session
        put_vol = df.loc[df["type"] == "put", "volume"].sum()
        call_vol = df.loc[df["type"] == "call", "volume"].sum()
        if call_vol == 0:
            return _stale("options.pcr_5d", regime)
        raw_val = float(put_vol / call_vol)
        return build_signal_output(
            name="options.pcr_5d",
            raw_value=raw_val,
            series_for_z=pd.Series([raw_val]),
            timestamp=0,
            lookback_bars=1,
            regime=regime,
            confidence=0.5,
        )

    # Group by date, compute daily PCR
    daily = df.groupby([date_col, "type"])["volume"].sum().unstack(fill_value=0)
    if "put" not in daily.columns or "call" not in daily.columns:
        return _stale("options.pcr_5d", regime)

    daily_pcr = daily["put"] / daily["call"].replace(0, np.nan)
    daily_pcr = daily_pcr.dropna()

    if daily_pcr.empty:
        return _stale("options.pcr_5d", regime)

    pcr_ema = ema_smooth(daily_pcr, span=5)
    raw_val = float(pcr_ema.iloc[-1])
    ts = _get_ts_from_col(df, date_col)

    return build_signal_output(
        name="options.pcr_5d",
        raw_value=raw_val,
        series_for_z=pcr_ema.dropna(),
        timestamp=ts,
        lookback_bars=min(5, len(pcr_ema)),
        regime=regime,
        confidence=min(float(len(pcr_ema)) / 5, 1.0),
        z_window=get_config_value("lookbacks.options.pcr_5d", 60),
    )


# ===================================================================
# P1  —  Gamma Exposure (GEX)
# ===================================================================

def compute_gex(
    options_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Total Gamma Exposure.

    Formula:
        GEX = sum(gamma * OI * 100) per strike, summed across all strikes.

    Negative GEX = vol-amplification zone (dealers short gamma).

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain 'gamma', 'oi' columns.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: options.gex_total
    """
    required = {"gamma", "oi"}
    if not required.issubset(options_df.columns):
        return _stale("options.gex_total", regime)

    df = options_df.copy()

    # Market-maker gamma sign convention: +gamma for calls, -gamma for puts
    if "type" in df.columns:
        sign = np.where(df["type"] == "call", 1.0, -1.0)
    else:
        sign = 1.0

    gex_per_row = sign * df["gamma"] * df["oi"] * 100
    total_gex = float(gex_per_row.sum())

    ts = _get_ts_from_col(df, "timestamp")

    return build_signal_output(
        name="options.gex_total",
        raw_value=total_gex,
        series_for_z=pd.Series([total_gex]),
        timestamp=ts,
        lookback_bars=1,
        regime=regime,
        confidence=1.0,
        meta={"n_strikes": len(df)},
    )


# ===================================================================
# P1  —  Net Options Delta Flow
# ===================================================================

def compute_delta_flow(
    options_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Net Options Delta Flow.

    Formula:
        flow = sum(signed_delta * volume) per session.

    Positive = net call buying = bullish lean.

    Parameters
    ----------
    options_df : pd.DataFrame
        Must contain 'delta', 'volume' columns.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: options.delta_flow
    """
    required = {"delta", "volume"}
    if not required.issubset(options_df.columns):
        return _stale("options.delta_flow", regime)

    df = options_df.copy()
    delta_flow = (df["delta"] * df["volume"]).sum()
    raw_val = float(delta_flow)

    ts = _get_ts_from_col(df, "timestamp")

    return build_signal_output(
        name="options.delta_flow",
        raw_value=raw_val,
        series_for_z=pd.Series([raw_val]),
        timestamp=ts,
        lookback_bars=1,
        regime=regime,
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_ts_from_col(df: pd.DataFrame, col: str) -> int:
    if col in df.columns:
        val = df[col].iloc[-1]
        if isinstance(val, (int, float, np.integer, np.floating)):
            return int(val)
        try:
            return int(pd.Timestamp(val).timestamp() * 1000)
        except Exception:
            return 0
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return int(df.index[-1].timestamp() * 1000)
    return 0
