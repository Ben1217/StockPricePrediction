"""
Composite Score — Weighted combination of normalised sentiment signals.

Weights are loaded from config/sentiment_config.yaml and can be updated
without a code deploy.  The output is hard-clipped to [-1, +1].
"""

import logging
from typing import Dict

from .models import SignalOutput
from .signal_processor import get_config_value

logger = logging.getLogger(__name__)

# Fallback weights used when sentiment_config.yaml is missing or incomplete
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "micro.ofi": 0.20,
    "vol.iv_rank": 0.15,
    "vol.term_slope": 0.15,
    "options.pcr_5d": 0.10,
    "volume.vwap_z": 0.10,
    "volume.cmf_20": 0.10,
    "breadth.mcclellan": 0.08,
    "momentum.rsi_divergence": 0.07,
    "positioning.cot_z": 0.05,
}


def get_weights() -> Dict[str, float]:
    """
    Return the signal weight map from config, falling back to defaults.

    Returns
    -------
    dict[str, float]
        Mapping of signal hook name → weight.
    """
    cfg_weights = get_config_value("composite.weights", None)
    if cfg_weights and isinstance(cfg_weights, dict):
        return cfg_weights
    return _DEFAULT_WEIGHTS.copy()


def composite_score(signals: Dict[str, SignalOutput]) -> float:
    """
    Compute a weighted composite sentiment score.

    Only non-stale signals contribute.  The result is normalised by the
    sum of active weights so that missing signals reduce confidence but
    do not bias the score.

    Parameters
    ----------
    signals : dict[str, SignalOutput]
        Mapping of signal hook name → SignalOutput.

    Returns
    -------
    float
        Composite score in [-1, +1].  Returns 0.0 when no valid signals
        are available.
    """
    weights = get_weights()

    total_weight = 0.0
    weighted_sum = 0.0

    for key, weight in weights.items():
        if key in signals and not signals[key].is_stale:
            weighted_sum += signals[key].normalised * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    raw = weighted_sum / total_weight

    # Hard-clip to [-1, +1] as a safety net
    return max(-1.0, min(1.0, raw))
