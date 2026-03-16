"""
Positioning Signals (COT) — P3

COT data is released weekly by the CFTC.  The z-score of speculator
net positioning is the key derivative.  Most reliable for commodities,
FX, and Treasury futures.

Required input columns (cot_df):
    ['report_date', 'instrument', 'comm_long', 'comm_short',
     'spec_long', 'spec_short', 'nonrep_long', 'nonrep_short']
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


def _stale(name: str, regime: str) -> SignalOutput:
    return SignalOutput(
        name=name, value=0.0, z_score=0.0, normalised=0.0,
        regime=regime, confidence=0.0, timestamp=0,
        lookback_bars=0, is_stale=True, meta={"reason": "insufficient_data"},
    )


# ===================================================================
# Speculator Net Z-score
# ===================================================================

def compute_cot_z(
    cot_df: pd.DataFrame,
    window: int = 52,
    regime: str = "neutral",
) -> SignalOutput:
    """
    COT Speculator Net Z-score.

    Formula:
        NetLong = spec_long - spec_short
        Z = (NetLong - mean(NetLong, 52w)) / std(NetLong, 52w)

    Z > +2 = crowded long → reversal risk.

    Parameters
    ----------
    cot_df : pd.DataFrame
        CFTC COT data.
    window : int
        Rolling z-score window in reports (default: 52 weeks).
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: positioning.cot_z
    """
    required = {"spec_long", "spec_short"}
    if not required.issubset(cot_df.columns):
        return _stale("positioning.cot_z", regime)

    df = cot_df.copy()
    net_long = df["spec_long"] - df["spec_short"]

    if len(net_long.dropna()) < 2:
        return _stale("positioning.cot_z", regime)

    raw_val = float(net_long.iloc[-1])
    ts = _get_ts(df)

    return build_signal_output(
        name="positioning.cot_z",
        raw_value=raw_val,
        series_for_z=net_long.dropna(),
        timestamp=ts,
        lookback_bars=min(window, len(net_long)),
        regime=regime,
        confidence=min(float(len(net_long.dropna())) / window, 1.0),
        z_window=get_config_value("lookbacks.positioning.cot_z", 52),
    )


# ===================================================================
# Commercial Hedger Ratio
# ===================================================================

def compute_commercial_ratio(
    cot_df: pd.DataFrame,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Commercial Hedger Ratio.

    Formula:
        Commercial_Net = comm_long - comm_short
        Ratio = Commercial_Net / (spec_long + spec_short)

    High ratio = smart money fading the crowd.

    Parameters
    ----------
    cot_df : pd.DataFrame
        CFTC COT data.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: positioning.commercial_ratio
    """
    required = {"comm_long", "comm_short", "spec_long", "spec_short"}
    if not required.issubset(cot_df.columns):
        return _stale("positioning.commercial_ratio", regime)

    df = cot_df.copy()
    comm_net = df["comm_long"] - df["comm_short"]
    spec_total = df["spec_long"] + df["spec_short"]
    ratio = comm_net / spec_total.replace(0, np.nan)
    ratio = ratio.dropna()

    if ratio.empty:
        return _stale("positioning.commercial_ratio", regime)

    raw_val = float(ratio.iloc[-1])
    ts = _get_ts(df)

    return build_signal_output(
        name="positioning.commercial_ratio",
        raw_value=raw_val,
        series_for_z=ratio,
        timestamp=ts,
        lookback_bars=len(ratio),
        regime=regime,
        confidence=1.0,
        z_window=get_config_value("lookbacks.positioning.commercial_ratio", 52),
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_ts(df: pd.DataFrame) -> int:
    for col in ("report_date", "timestamp"):
        if col in df.columns:
            val = df[col].iloc[-1]
            try:
                return int(pd.Timestamp(val).timestamp() * 1000)
            except Exception:
                pass
    if hasattr(df.index, "dtype") and np.issubdtype(df.index.dtype, np.datetime64):
        return int(df.index[-1].timestamp() * 1000)
    return 0
