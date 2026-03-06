"""
Unit test: Optuna Hyperparameter Optimization.
Smoke-tests that optimize_hyperparameters() runs and returns valid results.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.models.model_trainer import ModelTrainer


@pytest.fixture
def synthetic_data():
    """300-sample synthetic returns data."""
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 5).astype(np.float32)
    y = (0.3 * X[:, 0] + 0.2 * X[:, 1] + np.random.normal(0, 0.01, n)).astype(np.float32)
    return X, y


def test_optuna_xgboost_returns_params(synthetic_data):
    """Optuna should return a dict of best params and a Study object."""
    X, y = synthetic_data
    trainer = ModelTrainer()

    best_params, study = trainer.optimize_hyperparameters(
        'xgboost', X, y, n_trials=5, n_cv_splits=2, gap=2
    )

    assert isinstance(best_params, dict), "best_params should be a dict"
    assert 'n_estimators' in best_params, "Missing n_estimators in best_params"
    assert 'max_depth' in best_params, "Missing max_depth in best_params"
    assert 'learning_rate' in best_params, "Missing learning_rate in best_params"


def test_optuna_study_best_value_is_finite(synthetic_data):
    """The study's best Sharpe value should be a finite float."""
    X, y = synthetic_data
    trainer = ModelTrainer()

    _, study = trainer.optimize_hyperparameters(
        'xgboost', X, y, n_trials=3, n_cv_splits=2, gap=2
    )

    assert np.isfinite(study.best_value), f"Best value not finite: {study.best_value}"


def test_optuna_random_forest(synthetic_data):
    """Optuna should also work for random_forest."""
    X, y = synthetic_data
    trainer = ModelTrainer()

    best_params, study = trainer.optimize_hyperparameters(
        'random_forest', X, y, n_trials=3, n_cv_splits=2, gap=2
    )

    assert 'n_estimators' in best_params
    assert 'max_depth' in best_params
    assert len(study.trials) == 3
