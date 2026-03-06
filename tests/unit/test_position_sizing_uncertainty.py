"""
Unit test: Probabilistic Predictions into Position Sizing (Item 8)
Validates that higher MC Dropout uncertainty leads to smaller position sizes.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.signals.position_sizing import PositionSizeCalculator, RiskProfile


def test_adjust_for_uncertainty_thresholds():
    """Test the discrete downscaling bands logic."""
    calc = PositionSizeCalculator()
    base = 100.0
    
    # < 0.2 -> 100%
    assert calc._adjust_for_uncertainty(base, 0.1) == 100.0
    # < 0.4 -> 80%
    assert calc._adjust_for_uncertainty(base, 0.3) == 80.0
    # < 0.6 -> 60%
    assert calc._adjust_for_uncertainty(base, 0.5) == 60.0
    # < 0.8 -> 40%
    assert calc._adjust_for_uncertainty(base, 0.7) == 40.0
    # >= 0.8 -> 20%
    assert calc._adjust_for_uncertainty(base, 0.9) == 20.0
    assert calc._adjust_for_uncertainty(base, 2.5) == 20.0


def test_calculate_shares_with_uncertainty():
    """High uncertainty should yield fewer shares than low uncertainty."""
    profile = RiskProfile(
        risk_per_trade=1000.0, 
        risk_method='fixed_dollar', 
        confidence_scaling=False,
        max_position_size=20000.0  # large enough to not cap
    )
    calc = PositionSizeCalculator(profile)
    
    entry = 100.0
    stop = 90.0
    # base risk = $1000, risk per share = $10
    
    # 0.05 uncertainty (very low -> 100% risk) -> 100 shares
    res_certain = calc.calculate_shares(entry, stop, uncertainty=0.05)
    assert res_certain['shares'] == 100

    # 0.5 uncertainty (med-high -> 60% risk) -> 60 shares
    res_uncertain = calc.calculate_shares(entry, stop, uncertainty=0.5)
    assert res_uncertain['shares'] == 60
    
    # 0.9 uncertainty (very high -> 20% risk) -> 20 shares
    res_very_uncertain = calc.calculate_shares(entry, stop, uncertainty=0.9)
    assert res_very_uncertain['shares'] == 20
    
    # Warnings should be populated on high uncertainty
    assert not any("High prediction uncertainty" in w for w in res_certain['warnings'])
    assert any("High prediction uncertainty" in w for w in res_very_uncertain['warnings'])
