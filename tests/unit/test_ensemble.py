"""
Unit test: Ensemble Predictor (Item 6)
Validates Sharpe-based softmax weighting and prediction aggregation.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.models.ensemble import EnsemblePredictor


def test_ensemble_initial_weights():
    """Initial weights should be uniform across provided models."""
    ens = EnsemblePredictor(model_names=['m1', 'm2'])
    assert ens.weights == {'m1': 0.5, 'm2': 0.5}

def test_ensemble_update_weights_softmax():
    """Softmax weighting should favor the higher Sharpe ratio."""
    ens = EnsemblePredictor(model_names=['xgb', 'lstm'], temperature=1.0)
    
    # xgb Sharpe 2.0, lstm Sharpe 0.0 -> xgb should have > 80% weight
    weights = ens.update_weights({'xgb': 2.0, 'lstm': 0.0})
    
    assert weights['xgb'] > weights['lstm']
    assert np.isclose(weights['xgb'] + weights['lstm'], 1.0)

def test_ensemble_predict_weighted_average():
    """Predict should correctly apply the current weights."""
    ens = EnsemblePredictor(model_names=['m1', 'm2'])
    ens._weights = {'m1': 0.8, 'm2': 0.2}
    
    preds = {
        'm1': np.array([10.0, 20.0]),
        'm2': np.array([0.0,  0.0])
    }
    
    result = ens.predict(preds)
    # Expected: 10 * 0.8 + 0 * 0.2 = 8.0 ; 20 * 0.8 + 0 * 0.2 = 16.0
    np.testing.assert_array_almost_equal(result, np.array([8.0, 16.0]))

def test_ensemble_predict_missing_model():
    """If a model prediction is missing, weights should be renormalised."""
    ens = EnsemblePredictor(model_names=['m1', 'm2', 'm3'])
    ens._weights = {'m1': 0.5, 'm2': 0.3, 'm3': 0.2}
    
    # only m2 and m3 provided
    preds = {
        'm2': np.array([10.0]),
        'm3': np.array([20.0])
    }
    
    result = ens.predict(preds)
    # new sum = 0.5. m2 norm = 0.3/0.5 = 0.6. m3 norm = 0.2/0.5 = 0.4.
    # Expected: 10*0.6 + 20*0.4 = 6 + 8 = 14.0
    np.testing.assert_array_almost_equal(result, np.array([14.0]))
