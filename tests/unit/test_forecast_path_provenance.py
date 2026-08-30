"""
Tests for how the forecast path is built and labelled.

The bundles are trained on a single cumulative horizon-day return, so a 30-day
forecast rests on one model output per model and 29 interpolated points. That is
a legitimate reading of the target, but it must never be reported as a per-step
forecast — these tests pin the labelling and the walk-forward tuning that the
retrain depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.ensemble_predictor import (
    FORECAST_MODE_AUTO,
    FORECAST_MODE_COMPOUNDED,
    FORECAST_MODE_RECURSIVE,
    _bundle_step_horizon,
    _compound_path,
    forecast_mode,
    recursive_path_supported,
)
from src.models.walk_forward import walk_forward_splits, walk_forward_tune


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------

def test_compound_path_lands_on_the_headline_endpoint():
    path = _compound_path(current_price=100.0, total_return=0.10, horizon=30)
    assert len(path) == 30
    assert path[-1] == pytest.approx(110.0)


def test_compound_path_is_smooth_so_it_cannot_be_a_per_step_forecast():
    """
    The compounded path has a vanishing second difference — it is a curve with no
    step-to-step structure. A genuine per-step model path would not be this
    smooth, which is exactly why the straight chart lines were diagnostic.
    """
    path = _compound_path(current_price=100.0, total_return=0.06, horizon=30)
    assert np.max(np.abs(np.diff(path, 2))) < 1e-3


def test_compound_path_is_geometric_not_linear():
    """The interpolation compounds; it is not a straight ramp (though it looks like one)."""
    current, total_return, horizon = 100.0, 0.50, 30
    path = _compound_path(current, total_return, horizon)
    steps = np.arange(1, horizon + 1)
    linear = current + (current * total_return) * steps / horizon
    # At a large return the two mechanisms separate visibly.
    assert np.max(np.abs(path - linear)) > 1.0


# ---------------------------------------------------------------------------
# Recursive-mode gating
# ---------------------------------------------------------------------------

def _bundle(target_col: str, horizon: int):
    return {"meta": {"target_col": target_col, "horizon": horizon}}


def test_bundle_step_horizon_reads_the_trained_target():
    assert _bundle_step_horizon(_bundle("target_return_30d", 30)) == 30
    assert _bundle_step_horizon(_bundle("target_return_1d", 1)) == 1


def test_recursive_mode_refused_for_horizon_day_bundles():
    """
    Rolling a 30-day-return model forward day by day would apply the same 30-day
    forecast 30 times over. The gate must refuse it.
    """
    ok, reason = recursive_path_supported({"random_forest": _bundle("target_return_30d", 30)})
    assert ok is False
    assert "1-day return" in reason


def test_recursive_mode_allowed_for_one_day_bundles():
    ok, reason = recursive_path_supported({"random_forest": _bundle("target_return_1d", 1)})
    assert ok is True
    assert reason is None


def test_step_bundles_are_loaded_from_horizon_one(monkeypatch):
    """
    Recursive mode must fetch the 1-day bundles, not the requested horizon's.

    Loading the requested horizon was the bug that made the mode unreachable:
    the gate correctly refused the 30-day bundles it had just loaded.
    """
    import src.models.ensemble_predictor as ep

    asked: list[int] = []

    def fake_loader(symbol, model_type, horizon):
        asked.append(horizon)
        return {"meta": {"target_col": "target_return_1d", "horizon": 1}}

    monkeypatch.setattr(ep, "_load_regression_bundle", fake_loader)
    bundles = ep.load_step_bundles("GOOGL", ["random_forest", "xgboost"])

    assert asked == [ep.RECURSIVE_STEP_HORIZON, ep.RECURSIVE_STEP_HORIZON]
    assert set(bundles) == {"random_forest", "xgboost"}
    assert recursive_path_supported(bundles)[0] is True


def test_step_bundles_empty_when_none_trained(monkeypatch):
    import src.models.ensemble_predictor as ep

    monkeypatch.setattr(ep, "_load_regression_bundle", lambda s, m, h: None)
    assert ep.load_step_bundles("GOOGL", ["random_forest"]) == {}


def test_forecast_mode_defaults_to_auto(monkeypatch):
    """
    Auto prefers the per-step path whenever step bundles exist.

    The previous default was `compounded`, which meant the recursive path stayed
    off unless someone knew to set an env var — so every chart drew a compounded
    line from a single model output even on symbols whose step bundles were ready.
    """
    monkeypatch.delenv("QUANTVISION_FORECAST_MODE", raising=False)
    assert forecast_mode() == FORECAST_MODE_AUTO


def test_forecast_mode_reads_env(monkeypatch):
    monkeypatch.setenv("QUANTVISION_FORECAST_MODE", "recursive")
    assert forecast_mode() == FORECAST_MODE_RECURSIVE
    monkeypatch.setenv("QUANTVISION_FORECAST_MODE", "compounded")
    assert forecast_mode() == FORECAST_MODE_COMPOUNDED
    monkeypatch.setenv("QUANTVISION_FORECAST_MODE", "nonsense")
    assert forecast_mode() == FORECAST_MODE_AUTO


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------

def test_walk_forward_splits_never_train_on_the_future():
    for train_idx, score_idx in walk_forward_splits(1000, n_splits=4, embargo=30):
        assert train_idx.max() < score_idx.min(), "training rows must precede scored rows"


def test_walk_forward_splits_respect_the_embargo():
    embargo = 30
    for train_idx, score_idx in walk_forward_splits(1000, n_splits=4, embargo=embargo):
        assert score_idx.min() - train_idx.max() > embargo


def test_walk_forward_training_window_expands():
    splits = walk_forward_splits(1000, n_splits=4, embargo=10)
    sizes = [len(train_idx) for train_idx, _ in splits]
    assert sizes == sorted(sizes), "each fold should train on at least as much history"


def test_walk_forward_splits_empty_when_too_short():
    assert walk_forward_splits(2, n_splits=4, embargo=30) == []


class _ConstantModel:
    """Predicts a fixed offset; error is controlled by the `bias` parameter."""

    def __init__(self, params=None):
        self.bias = float((params or {}).get("bias", 0.0))

    def fit(self, X, y, X_val=None, y_val=None):
        self.mean = float(np.mean(y))

    def predict(self, X):
        return np.full(len(X), self.mean + self.bias)

    def evaluate(self, X, y, prev_close=None):
        return {"mae": float(np.mean(np.abs(y - self.predict(X))))}


def test_walk_forward_tune_picks_the_lower_error_candidate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = rng.normal(size=400)

    result = walk_forward_tune(
        model_type="test",
        factory=_ConstantModel,
        X=X,
        y=y,
        param_grid=[{"bias": 0.0}, {"bias": 5.0}],
        n_splits=3,
        embargo=5,
    )
    assert result is not None
    assert result.best_params["bias"] == 0.0
    assert result.n_splits > 0
    assert result.as_metadata()["method"] == "walk_forward_expanding_window"


def test_walk_forward_tune_returns_none_when_data_too_short():
    X = np.zeros((4, 2))
    y = np.zeros(4)
    assert walk_forward_tune(
        model_type="test", factory=_ConstantModel, X=X, y=y,
        param_grid=[{"bias": 0.0}], n_splits=4, embargo=30,
    ) is None
