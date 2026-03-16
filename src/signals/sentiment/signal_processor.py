"""
Signal Processing Pipeline

Provides normalisation (rolling z-score → tanh mapping), EMA smoothing,
regime classification, and a helper to construct SignalOutput instances.

All rolling window sizes are read from config/sentiment_config.yaml so they
can be tuned without a code change.
"""

import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .models import SignalOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_CACHE: Optional[dict] = None


def _load_config() -> dict:
    """Load and cache sentiment_config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    config_path = Path(__file__).resolve().parents[3] / "config" / "sentiment_config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            _CONFIG_CACHE = yaml.safe_load(f)
    else:
        logger.warning("sentiment_config.yaml not found at %s, using defaults", config_path)
        _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_config_value(key: str, default=None):
    """Get a dot-separated key from the config, e.g. 'processing.z_score_window'."""
    cfg = _load_config()
    parts = key.split(".")
    current = cfg
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def reload_config():
    """Force-reload the config file (useful after hot-updating YAML)."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
    _load_config()


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def z_score_rolling(series: pd.Series, window: int = 60) -> pd.Series:
    """
    Compute a rolling z-score.

    Parameters
    ----------
    series : pd.Series
        Raw signal values.
    window : int
        Rolling lookback window (default: 60 bars).

    Returns
    -------
    pd.Series
        Z-scored series; NaN where std == 0 or insufficient data.
    """
    mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    std = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def normalise_to_unit(z: pd.Series, clip: float = 3.0) -> pd.Series:
    """
    Map z-score to [-1, +1] via soft tanh clip.

    Parameters
    ----------
    z : pd.Series
        Z-scored values.
    clip : float
        Scaling denominator; larger = softer clip.

    Returns
    -------
    pd.Series
        Values in [-1, +1].
    """
    return np.tanh(z / clip)


def ema_smooth(series: pd.Series, span: int = 5) -> pd.Series:
    """
    Exponential moving-average smoother.

    Parameters
    ----------
    series : pd.Series
        Raw or z-scored signal.
    span : int
        EMA span (default: 5).

    Returns
    -------
    pd.Series
        Smoothed series.
    """
    return series.ewm(span=span, adjust=False).mean()


def get_regime(term_slope: float) -> str:
    """
    Classify market regime from VIX term-structure slope.

    Parameters
    ----------
    term_slope : float
        VX_M2 / VX_M1 ratio.

    Returns
    -------
    str
        'risk_off' if slope < 0.95 (backwardation / stress),
        'risk_on'  if slope > 1.05 (contango / calm),
        'neutral'  otherwise.
    """
    low = get_config_value("regime.backwardation_threshold", 0.95)
    high = get_config_value("regime.contango_threshold", 1.05)

    if term_slope < low:
        return "risk_off"
    elif term_slope > high:
        return "risk_on"
    else:
        return "neutral"


# ---------------------------------------------------------------------------
# SignalOutput builder
# ---------------------------------------------------------------------------

def build_signal_output(
    name: str,
    raw_value: float,
    series_for_z: pd.Series,
    timestamp: int,
    lookback_bars: int,
    regime: str = "neutral",
    confidence: float = 1.0,
    is_stale: bool = False,
    meta: Optional[dict] = None,
    z_window: Optional[int] = None,
    clip: float = 3.0,
) -> SignalOutput:
    """
    Convenience builder: computes z-score and normalised value, then
    returns a fully-populated SignalOutput.

    Parameters
    ----------
    name : str
        Signal hook name (e.g. 'micro.ofi').
    raw_value : float
        The latest raw signal value.
    series_for_z : pd.Series
        Historical series used to compute the rolling z-score.
    timestamp : int
        Unix ms timestamp of the current bar.
    lookback_bars : int
        Bars used in calculation.
    regime : str
        Pre-determined regime label.
    confidence : float
        Data quality in [0, 1].
    is_stale : bool
        Whether data is stale.
    meta : dict, optional
        Extra signal-specific metadata.
    z_window : int, optional
        Override z-score lookback (defaults to config or 60).
    clip : float
        Tanh clip value.

    Returns
    -------
    SignalOutput
    """
    if z_window is None:
        z_window = get_config_value("processing.z_score_window", 60)

    z_series = z_score_rolling(series_for_z, window=z_window)
    z_val = float(z_series.iloc[-1]) if not z_series.empty and not pd.isna(z_series.iloc[-1]) else 0.0

    norm_series = normalise_to_unit(z_series, clip=clip)
    norm_val = float(norm_series.iloc[-1]) if not norm_series.empty and not pd.isna(norm_series.iloc[-1]) else 0.0

    # Hard-clip normalised to [-1, +1] for safety
    norm_val = max(-1.0, min(1.0, norm_val))

    return SignalOutput(
        name=name,
        value=raw_value,
        z_score=z_val,
        normalised=norm_val,
        regime=regime,
        confidence=confidence,
        timestamp=timestamp,
        lookback_bars=lookback_bars,
        is_stale=is_stale,
        meta=meta or {},
    )
