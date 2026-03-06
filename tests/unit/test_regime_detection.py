"""
Unit test: HMM Regime Detection.
Validates that MarketRegimeDetector correctly identifies 3 regimes
and produces properly formatted output.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.models.regime_detection import MarketRegimeDetector


@pytest.fixture
def synthetic_returns():
    """
    Concatenate three very distinct regimes with strong separation:
      - Bull:     strongly positive return, low vol
      - Bear:     strongly negative return, high vol
      - Sideways: ~zero return, low vol
    """
    np.random.seed(42)
    bull = np.random.normal(0.01, 0.003, 200)       # +1% daily avg
    bear = np.random.normal(-0.01, 0.015, 200)      # -1% daily avg, high vol
    sideways = np.random.normal(0.0, 0.002, 200)    #  0% daily avg, low vol

    returns = np.concatenate([bull, bear, sideways])
    idx = pd.date_range('2018-01-01', periods=len(returns), freq='B')
    return pd.Series(returns, index=idx, name='returns')


def test_fit_produces_fitted_model(synthetic_returns):
    """Fitting should mark the detector as fitted."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)
    assert detector.is_fitted


def test_predict_returns_valid_labels(synthetic_returns):
    """Predictions should be integers in {0, 1, 2}."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)
    labels = detector.predict(synthetic_returns)

    unique = set(int(l) for l in labels)
    assert unique.issubset({0, 1, 2}), f"Unexpected labels: {unique}"
    assert len(unique) >= 2, f"Expected at least 2 regimes, got {unique}"


def test_regime_ordering(synthetic_returns):
    """Bull regime should have highest mean return, Bear the lowest."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)

    mean_returns = detector.model.means_[:, 0]
    ordered = mean_returns[detector._state_order]

    # After ordering: index 0 = Bear (lowest), index 2 = Bull (highest)
    assert ordered[0] < ordered[2], (
        f"Regime ordering incorrect: Bear mean={ordered[0]:.4f}, Bull mean={ordered[2]:.4f}"
    )


def test_get_regime_features_columns(synthetic_returns):
    """get_regime_features should return a DataFrame with required columns."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)
    df = detector.get_regime_features(synthetic_returns)

    expected_cols = {'regime_label', 'regime_name', 'regime_probability'}
    assert expected_cols == set(df.columns), (
        f"Expected columns {expected_cols}, got {set(df.columns)}"
    )
    assert len(df) > 0, "Feature DataFrame should not be empty"


def test_regime_probability_range(synthetic_returns):
    """All regime probabilities should be in [0, 1]."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)
    df = detector.get_regime_features(synthetic_returns)

    assert (df['regime_probability'] >= 0).all(), "Probability < 0 found"
    assert (df['regime_probability'] <= 1).all(), "Probability > 1 found"


def test_regime_names_valid(synthetic_returns):
    """All regime names should be one of Bull, Bear, Sideways."""
    detector = MarketRegimeDetector(n_states=3)
    detector.fit(synthetic_returns)
    df = detector.get_regime_features(synthetic_returns)

    valid_names = {'Bull', 'Bear', 'Sideways'}
    actual_names = set(df['regime_name'].unique())
    assert actual_names.issubset(valid_names), (
        f"Unexpected regime names: {actual_names - valid_names}"
    )
