"""
Microstructure Signals — P0

Operate on tick / sub-minute data.  Order Flow Imbalance (OFI) is the
single strongest short-horizon directional predictor and should be
prioritised for any intraday or HFT component.

Required input columns (tick_df):
    ['timestamp', 'price', 'size', 'side',       # 'bid'|'ask'
     'bid_price', 'bid_size', 'ask_price', 'ask_size']
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
    z_score_rolling,
    normalise_to_unit,
)

logger = logging.getLogger(__name__)


def compute_ofi(
    tick_df: pd.DataFrame,
    window: int = 100,
    regime: str = "neutral",
    stale_threshold: Optional[int] = None,
) -> SignalOutput:
    """
    Order Flow Imbalance (OFI).

    Formula:  (Buy_Vol - Sell_Vol) / (Buy_Vol + Sell_Vol)
    Range:    [-1, +1]  (already naturally bounded)

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with 'size' and 'side' columns.
    window : int
        Rolling aggregation window in ticks.
    regime : str
        Current regime label.
    stale_threshold : int, optional
        Override staleness threshold.

    Returns
    -------
    SignalOutput
        hook: micro.ofi
    """
    if stale_threshold is None:
        stale_threshold = get_config_value("staleness.micro.ofi", 2)

    required = {"size", "side", "timestamp"}
    if not required.issubset(tick_df.columns):
        logger.warning("OFI: missing columns %s", required - set(tick_df.columns))
        return _stale_output("micro.ofi", regime)

    df = tick_df.copy()
    df["buy_vol"] = df["size"].where(df["side"] == "ask", 0.0)
    df["sell_vol"] = df["size"].where(df["side"] == "bid", 0.0)

    buy_sum = df["buy_vol"].rolling(window, min_periods=1).sum()
    sell_sum = df["sell_vol"].rolling(window, min_periods=1).sum()
    total = buy_sum + sell_sum
    ofi = (buy_sum - sell_sum) / total.replace(0, np.nan)

    if ofi.dropna().empty:
        return _stale_output("micro.ofi", regime)

    raw_val = float(ofi.iloc[-1]) if not pd.isna(ofi.iloc[-1]) else 0.0
    ts = int(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else 0
    confidence = float(ofi.notna().sum()) / max(len(ofi), 1)

    return build_signal_output(
        name="micro.ofi",
        raw_value=raw_val,
        series_for_z=ofi.dropna(),
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=min(confidence, 1.0),
        is_stale=False,
        z_window=get_config_value("lookbacks.micro.ofi", 100),
    )


def compute_trade_sign_imbalance(
    tick_df: pd.DataFrame,
    window: int = 100,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Trade Sign Imbalance via Lee-Ready classifier.

    sign(trade) = +1 if price > midpoint else -1.
    Sum signs over a rolling window.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with 'price', 'bid_price', 'ask_price', 'timestamp'.
    window : int
        Rolling window.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: micro.trade_sign
    """
    required = {"price", "bid_price", "ask_price", "timestamp"}
    if not required.issubset(tick_df.columns):
        return _stale_output("micro.trade_sign", regime)

    df = tick_df.copy()
    midpoint = (df["bid_price"] + df["ask_price"]) / 2.0
    df["sign"] = np.where(df["price"] > midpoint, 1.0, np.where(df["price"] < midpoint, -1.0, 0.0))
    sign_sum = df["sign"].rolling(window, min_periods=1).sum() / window

    if sign_sum.dropna().empty:
        return _stale_output("micro.trade_sign", regime)

    raw_val = float(sign_sum.iloc[-1])
    ts = int(df["timestamp"].iloc[-1])

    return build_signal_output(
        name="micro.trade_sign",
        raw_value=raw_val,
        series_for_z=sign_sum.dropna(),
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=1.0,
    )


def compute_spread_z(
    tick_df: pd.DataFrame,
    window: int = 100,
    regime: str = "neutral",
) -> SignalOutput:
    """
    Bid-Ask Spread Z-score.

    Z-score of the bid-ask spread over a rolling window.
    Widening spread Z > +2 signals liquidity stress.

    Parameters
    ----------
    tick_df : pd.DataFrame
        Tick data with 'bid_price', 'ask_price', 'timestamp'.
    window : int
        Rolling window.
    regime : str
        Current regime label.

    Returns
    -------
    SignalOutput
        hook: micro.spread_z
    """
    required = {"bid_price", "ask_price", "timestamp"}
    if not required.issubset(tick_df.columns):
        return _stale_output("micro.spread_z", regime)

    df = tick_df.copy()
    spread = df["ask_price"] - df["bid_price"]

    if spread.dropna().empty:
        return _stale_output("micro.spread_z", regime)

    raw_val = float(spread.iloc[-1])
    ts = int(df["timestamp"].iloc[-1])

    return build_signal_output(
        name="micro.spread_z",
        raw_value=raw_val,
        series_for_z=spread.dropna(),
        timestamp=ts,
        lookback_bars=window,
        regime=regime,
        confidence=1.0,
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _stale_output(name: str, regime: str) -> SignalOutput:
    """Return a stale / zero-confidence SignalOutput when data is missing."""
    return SignalOutput(
        name=name,
        value=0.0,
        z_score=0.0,
        normalised=0.0,
        regime=regime,
        confidence=0.0,
        timestamp=0,
        lookback_bars=0,
        is_stale=True,
        meta={"reason": "insufficient_data"},
    )
