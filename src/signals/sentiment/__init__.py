"""
Technical Sentiment Signal Module

Sits between the raw data pipeline and the alpha generation layer.
Exposes a standardised SignalOutput interface so downstream consumers
(backtester, live execution, risk module) can consume signals without
coupling to implementation details.

Quick-start
-----------
    from src.signals.sentiment import (
        SignalOutput,
        composite_score,
        compute_ofi,
        compute_vwap_z,
        compute_iv_rank,
        compute_term_slope,
        compute_pcr_5d,
        compute_mcclellan,
        compute_cmf,
        compute_cot_z,
        compute_rsi_divergence,
    )
"""

# Core
from .models import SignalOutput
from .signal_processor import (
    z_score_rolling,
    normalise_to_unit,
    ema_smooth,
    get_regime,
    build_signal_output,
    reload_config,
)
from .composite_score import composite_score, get_weights

# P0 — Microstructure
from .microstructure_signals import (
    compute_ofi,
    compute_trade_sign_imbalance,
    compute_spread_z,
)

# P0/P2 — Volume Flow
from .volume_signals import (
    compute_vwap_z,
    compute_cmf,
    compute_obv_divergence,
)

# P1/P3 — Volatility
from .volatility_signals import (
    compute_iv_rank,
    compute_term_slope,
    compute_rv_iv_spread,
    compute_skew_25d,
)

# P1 — Options Flow
from .options_signals import (
    compute_pcr_5d,
    compute_gex,
    compute_delta_flow,
)

# P2 — Breadth
from .breadth_signals import (
    compute_mcclellan,
    compute_ad_line,
    compute_nh_nl_ratio,
)

# P3 — Positioning
from .positioning_signals import (
    compute_cot_z,
    compute_commercial_ratio,
)

# P3 — Momentum / Divergence
from .momentum_signals import (
    compute_rsi_divergence,
    compute_macd_exhaustion,
    compute_roc_z,
)

__all__ = [
    # Core
    "SignalOutput",
    "composite_score",
    "get_weights",
    "z_score_rolling",
    "normalise_to_unit",
    "ema_smooth",
    "get_regime",
    "build_signal_output",
    "reload_config",
    # P0
    "compute_ofi",
    "compute_trade_sign_imbalance",
    "compute_spread_z",
    "compute_vwap_z",
    # P1
    "compute_iv_rank",
    "compute_term_slope",
    "compute_rv_iv_spread",
    "compute_pcr_5d",
    "compute_gex",
    "compute_delta_flow",
    # P2
    "compute_cmf",
    "compute_obv_divergence",
    "compute_mcclellan",
    "compute_ad_line",
    "compute_nh_nl_ratio",
    # P3
    "compute_skew_25d",
    "compute_cot_z",
    "compute_commercial_ratio",
    "compute_rsi_divergence",
    "compute_macd_exhaustion",
    "compute_roc_z",
]
