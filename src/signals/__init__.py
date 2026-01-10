"""
Signals module - Trading signal generation
"""

from .signal_generator import (
    TradingSignalGenerator,
    detect_base_breakout,
    detect_pullback_buy,
    detect_123_continuation,
    detect_base_breakdown,
    check_uptrend,
    check_downtrend
)
from .position_sizing import PositionSizeCalculator, RiskProfile

__all__ = [
    'TradingSignalGenerator',
    'detect_base_breakout',
    'detect_pullback_buy',
    'detect_123_continuation',
    'detect_base_breakdown',
    'check_uptrend',
    'check_downtrend',
    'PositionSizeCalculator',
    'RiskProfile'
]
