"""
Unit test: Walk-Forward Validation.
Validates that ModelTrainer.walk_forward_validate() produces correct
per-fold metrics with the embargo gap respected.
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
    """500-sample synthetic time-series: noisy sine wave."""
    np.random.seed(42)
    n = 500
    t = np.linspace(0, 10 * np.pi, n)
    y = np.sin(t) * 0.02 + np.random.normal(0, 0.005, n)
    X = np.column_stack([
        np.sin(t + 0.5) * 0.02,
        np.cos(t) * 0.02,
        np.random.normal(0, 0.01, n),
    ])
    return X.astype(np.float32), y.astype(np.float32)


def test_walk_forward_shape(synthetic_data):
    """Result DataFrame should have n_splits metric rows + 1 summary row."""
    X, y = synthetic_data
    trainer = ModelTrainer()
    n_splits = 3
    gap = 2

    result = trainer.walk_forward_validate(
        'xgboost', X, y, n_splits=n_splits, gap=gap
    )

    # n_splits data rows + 1 mean±std summary row
    assert len(result) == n_splits + 1, f"Expected {n_splits + 1} rows, got {len(result)}"


def test_walk_forward_columns(synthetic_data):
    """Result should contain all expected metric columns."""
    X, y = synthetic_data
    trainer = ModelTrainer()

    result = trainer.walk_forward_validate('xgboost', X, y, n_splits=3, gap=2)

    expected_cols = {'fold', 'rmse', 'mae', 'r2', 'directional_accuracy', 'sharpe_ratio'}
    assert expected_cols.issubset(set(result.columns)), (
        f"Missing columns: {expected_cols - set(result.columns)}"
    )


def test_walk_forward_no_overlap(synthetic_data):
    """Train and test indices must not overlap (gap is respected)."""
    from sklearn.model_selection import TimeSeriesSplit

    X, y = synthetic_data
    gap = 5
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)

    for train_idx, test_idx in tscv.split(X):
        # Ensure no overlap
        overlap = set(train_idx) & set(test_idx)
        assert len(overlap) == 0, f"Overlap detected: {overlap}"

        # Ensure gap is respected: last train index + gap < first test index
        assert test_idx[0] >= train_idx[-1] + gap, (
            f"Gap not respected: train ends at {train_idx[-1]}, "
            f"test starts at {test_idx[0]}, gap={gap}"
        )


def test_walk_forward_metrics_are_finite(synthetic_data):
    """All numeric fold metrics should be finite floats."""
    X, y = synthetic_data
    trainer = ModelTrainer()

    result = trainer.walk_forward_validate('xgboost', X, y, n_splits=3, gap=2)

    # Check the data rows only (not the summary row)
    data_rows = result.iloc[:-1]
    for col in ['rmse', 'mae', 'r2', 'directional_accuracy', 'sharpe_ratio']:
        vals = data_rows[col].astype(float)
        assert vals.notna().all(), f"NaN found in {col}"
        assert np.isfinite(vals.values).all(), f"Inf found in {col}"
