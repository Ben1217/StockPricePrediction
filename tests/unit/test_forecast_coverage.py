"""
Coverage of every model x horizon, and the shape of what comes back.

Two failures motivated these tests:

* A horizon went dark whenever one ensemble member failed the skill gate, even
  though the other members were servable — the tab showed "Prediction model
  unavailable" for a symbol with two working bundles.
* The forecast drew as a straight line because each model emitted one cumulative
  horizon-day return and the intermediate days were compounded toward it. The
  per-step path that fixes this needs 1-day step bundles, which the training API
  refused to accept as a horizon, so the mode was unreachable in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.schemas.schemas import (
    STEP_FORECAST_HORIZON,
    SUPPORTED_FORECAST_HORIZONS,
    TRAINABLE_HORIZONS,
    EnsembleTrainRequest,
)
from src.models.ensemble_predictor import (
    _build_forecast_points,
    _compound_path,
    recursive_model_path,
)


# ---------------------------------------------------------------------------
# Trainable horizons
# ---------------------------------------------------------------------------

def test_step_horizon_is_trainable_through_the_api():
    """
    The 1-day step bundle must be trainable or per-step forecasting is unreachable.

    The tab's train button is the only path that ever builds bundles, so a
    validator that rejects horizon 1 permanently pins every chart to the
    compounded path regardless of how the forecast mode is configured.
    """
    request = EnsembleTrainRequest(symbol="AAPL", horizons=[STEP_FORECAST_HORIZON])
    assert request.horizons == [STEP_FORECAST_HORIZON]


def test_training_defaults_cover_every_display_horizon_and_the_step_model():
    request = EnsembleTrainRequest(symbol="AAPL")
    for horizon in SUPPORTED_FORECAST_HORIZONS:
        assert horizon in request.horizons, f"horizon {horizon} is offered by the UI but never trained"
    assert STEP_FORECAST_HORIZON in request.horizons


def test_unsupported_horizons_are_still_rejected():
    with pytest.raises(ValueError):
        EnsembleTrainRequest(symbol="AAPL", horizons=[13])


def test_trainable_horizons_are_the_display_horizons_plus_the_step_model():
    assert set(TRAINABLE_HORIZONS) == {STEP_FORECAST_HORIZON, *SUPPORTED_FORECAST_HORIZONS}


# ---------------------------------------------------------------------------
# Per-step inference
# ---------------------------------------------------------------------------

def _ohlcv(rows: int = 320, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, rows)))
    index = pd.bdate_range("2024-01-01", periods=rows)
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, rows),
        },
        index=index,
    )


def test_recursive_path_produces_one_model_output_per_day(monkeypatch):
    """
    The requirement behind the straight line: N days must come from N inferences.

    The compounded path calls the model once and derives the rest, so the step
    deltas are identical by construction. Here each step is a separate call, and
    varying the returned return has to move that day and no other.
    """
    import src.models.ensemble_predictor as ep

    returns = [0.01, -0.008, 0.004, -0.002, 0.012, -0.015, 0.003]
    calls = {"n": 0}

    def fake_inference(bundle, frame, model_type):
        value = returns[calls["n"] % len(returns)]
        calls["n"] += 1
        return value, 0.02

    monkeypatch.setattr(ep, "_run_inference", fake_inference)
    monkeypatch.setattr(ep, "build_regression_feature_frame", lambda history, feature_config=None: history)

    horizon = 7
    raw = _ohlcv()
    path = recursive_model_path(
        bundle={"meta": {"target_col": "target_return_1d", "horizon": 1}},
        model_type="xgboost",
        raw_df=raw,
        horizon=horizon,
        current_price=float(raw["Close"].iloc[-1]),
        feature_config={},
    )

    assert path is not None
    assert len(path) == horizon, "one predicted price per forecast day"
    assert calls["n"] == horizon, "one model call per forecast day, not one for the whole horizon"

    # The supplied returns change sign, so the resulting path must too. A
    # compounded path could not do this — that is the whole distinction.
    deltas = np.diff(np.concatenate([[float(raw["Close"].iloc[-1])], path]))
    assert (deltas > 0).any() and (deltas < 0).any(), "per-step path must follow the per-step returns"


def test_compounded_path_cannot_change_direction():
    """The contrast case: one model output can only ever draw a one-way curve."""
    path = _compound_path(current_price=100.0, total_return=-0.05, horizon=15)
    deltas = np.diff(path)
    assert (deltas < 0).all(), "a single cumulative return compounds in one direction only"


# ---------------------------------------------------------------------------
# Scenario paths — the volatility the chart draws
# ---------------------------------------------------------------------------

def _points_and_paths(horizon: int = 15, n_scenarios: int = 40):
    raw = _ohlcv()
    current = float(raw["Close"].iloc[-1])
    return _build_forecast_points(
        predicted_price=current * 0.98,
        current_price=current,
        horizon=horizon,
        last_date=raw.index[-1],
        avg_mape=3.0,
        weighted_rmse=current * 0.03,
        spread_pct=1.5,
        recent_volatility=0.015,
        raw_predictions={"xgboost": current * 0.98},
        raw_df=raw,
        seed=11,
        n_scenarios=n_scenarios,
    ), current


def test_scenario_paths_align_with_the_forecast_points():
    """
    Each path is [today, ...one value per forecast day].

    The chart pins element 0 to the last historical bar so the fan starts at
    today's price; a length mismatch would silently shift the whole fan by a day.
    """
    horizon = 15
    (points, paths), current = _points_and_paths(horizon=horizon)

    assert len(points) == horizon
    assert paths, "scenario paths must be produced for the fan chart"
    for path in paths:
        assert len(path) == horizon + 1
        assert path[0] == pytest.approx(round(current, 2))


def test_scenario_paths_carry_day_to_day_volatility():
    """
    The fan is what shows realistic volatility, and it must not be smooth.

    The centre line is a conditional expectation and is smooth by construction —
    adding wiggle to it would be inventing detail. The simulated paths are
    resampled from the asset's own recent returns, so they change direction
    repeatedly, which is the honest way to draw the uncertainty.
    """
    horizon = 30
    (points, paths), _ = _points_and_paths(horizon=horizon)

    def direction_changes(series):
        deltas = np.diff(np.asarray(series, dtype=float))
        return int(np.sum(np.sign(deltas[1:]) != np.sign(deltas[:-1])))

    fan_changes = [direction_changes(path) for path in paths]
    assert min(fan_changes) > 0, "every simulated path should reverse at least once"
    assert float(np.mean(fan_changes)) > horizon * 0.2, "the fan should look like a price series"

    centre_changes = direction_changes([p["predicted"] for p in points])
    assert centre_changes == 0, "the compounded centre line is expected to stay smooth"


def test_confidence_bands_widen_with_the_horizon():
    """Uncertainty must grow with distance; a flat band would be the fake-CI bug."""
    (points, _), _ = _points_and_paths(horizon=30)
    widths = [p["upper_95"] - p["lower_95"] for p in points]
    assert widths[-1] > widths[0], "the 95% band must widen as the forecast extends"
    assert all(w > 0 for w in widths), "a zero-width interval asserts precision nobody has"
