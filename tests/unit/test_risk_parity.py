"""
Unit test: Risk Parity Optimization (Item 7)
Validates Equal Risk Contribution portfolio generation.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimization import optimize_portfolio, _solve_risk_parity


def test_solve_risk_parity_diag_cov():
    """
    With a diagonal covariance matrix, risk parity weights should be 
    inversely proportional to standard deviations.
    """
    # Assets have vols: 10%, 20%, 30% -> Vars: 0.01, 0.04, 0.09
    cov_matrix = np.diag([0.01, 0.04, 0.09])
    names = ['A', 'B', 'C']
    
    weights = _solve_risk_parity(names, cov_matrix, 3)
    
    # Asset A has lowest var -> highest weight. C has highest var -> lowest weight.
    assert weights['A'] > weights['B'] > weights['C']
    assert np.isclose(sum(weights.values()), 1.0)

def test_solve_risk_parity_identical_assets():
    """If all assets have identical risk profiles, weights should be equal."""
    # 2 identical assets
    cov_matrix = np.array([[0.04, 0.01], [0.01, 0.04]])
    names = ['X', 'Y']
    
    weights = _solve_risk_parity(names, cov_matrix, 2)
    
    assert np.isclose(weights['X'], 0.5)
    assert np.isclose(weights['Y'], 0.5)

def test_optimize_portfolio_risk_parity_wrapper():
    """optimize_portfolio should support objective='risk_parity'."""
    np.random.seed(42)
    returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 3)), columns=['A', 'B', 'C'])
    
    weights = optimize_portfolio(returns, objective='risk_parity')
    
    assert isinstance(weights, dict)
    assert 'A' in weights and 'B' in weights and 'C' in weights
    assert np.isclose(sum(weights.values()), 1.0)
