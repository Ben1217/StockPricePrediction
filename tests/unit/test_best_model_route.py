"""
Contract tests for GET /api/predict/best/{symbol}.

This is the endpoint the chart overlay draws from, so what is asserted here is
what the chart is allowed to assume: that the two winners are named separately,
that the direction vocabulary is fixed, that every forecast point carries the
sign of its own step, and that an unservable symbol comes back as a 200 saying
so rather than as an exception or an empty chart with no explanation.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import predict as predict_route
from src.api.schemas.schemas import ForecastPoint, PredictResponse
from src.models.model_selection import BestModels, Candidate, rank_candidates


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    return TestClient(app)


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="B")
    close = pd.Series([100 + i * 0.4 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1_000_000 + i for i in range(rows)],
        },
        index=index,
    )


def _candidate(model_type: str, *, price=None, direction=None, horizon: int = 30) -> Candidate:
    return Candidate(
        model_type=model_type,
        label=model_type.replace("_", " ").title(),
        evidence="walk_forward_benchmark",
        horizon=horizon,
        price=price or {},
        direction=direction or {},
        context={"n_test_rows": 189},
    )


def _best_models(price_candidates, direction_candidates, horizon: int = 30) -> BestModels:
    """A real BestModels, so the response is built from the production shapes."""
    return BestModels(
        symbol="TEST",
        horizon=horizon,
        price=rank_candidates(price_candidates, "price"),
        direction=rank_candidates(direction_candidates, "direction"),
        candidates=list({c.model_type: c for c in [*price_candidates, *direction_candidates]}.values()),
    )


def _forecast_response(prices, *, model_type="random_forest", horizon=30,
                       current_price=100.0, probability_up=None) -> PredictResponse:
    points = [
        ForecastPoint(
            date=str(date.date()),
            predicted=price,
            upper95=price * 1.05,
            lower95=price * 0.95,
            upper68=price * 1.02,
            lower68=price * 0.98,
        )
        for date, price in zip(pd.bdate_range("2026-06-16", periods=len(prices)), prices)
    ]
    return PredictResponse(
        symbol="TEST",
        model_type=model_type,
        horizon=horizon,
        current_price=current_price,
        current_price_source="latest_close",
        target_price=prices[-1] if prices else None,
        predicted_price=prices[-1] if prices else None,
        probability_up=probability_up,
        probability_down=None if probability_up is None else 1 - probability_up,
        direction=None if probability_up is None else ("Bullish" if probability_up >= 0.5 else "Bearish"),
        signal="BUY",
        prediction_date=points[0].date if points else None,
        forecasts=points,
        model_info={"path_type": "compounded_interpolation", "per_step_predictions": False},
        status="ok",
    )


def _unavailable_response(model_type: str, message: str) -> PredictResponse:
    return PredictResponse(
        symbol="TEST",
        model_type=model_type,
        horizon=30,
        current_price=100.0,
        model_info={"status": "unavailable"},
        status="unavailable",
        model_available=False,
        reason="missing_bundle",
        message=message,
    )


def _patched(best: BestModels, price_response=None, unified_response=None, gate_candidates=()):
    """The route with its data, its selection and its serving paths stubbed."""
    return (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(100.0, "latest_close")),
        patch.object(predict_route, "select_best_models", return_value=best),
        patch.object(
            predict_route,
            "_predict_regression_or_unavailable",
            return_value=price_response or _unavailable_response("random_forest", "no bundle"),
        ),
        patch.object(
            predict_route,
            "_predict_unified_model",
            return_value=unified_response or _unavailable_response("unified_lstm", "no bundle"),
        ),
        patch.object(predict_route, "direction_report_candidates", return_value=list(gate_candidates)),
    )


def _call(client, best, *, horizon=30, **kwargs):
    patches = _patched(best, **kwargs)
    for entry in patches:
        entry.start()
    try:
        return client.get(f"/api/predict/best/TEST?horizon={horizon}")
    finally:
        for entry in patches:
            entry.stop()


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_two_winners_are_named_separately(client):
    best = _best_models(
        [
            _candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0}),
            _candidate("xgboost", price={"mae": 5.0, "rmse": 6.0, "mape": 2.0}),
        ],
        [
            _candidate("random_forest", direction={"accuracy": 0.49, "f1": 0.48, "roc_auc": 0.50}),
            _candidate("xgboost", direction={"accuracy": 0.57, "f1": 0.56, "roc_auc": 0.59}),
        ],
    )
    response = _call(
        client,
        best,
        price_response=_forecast_response([101.0, 102.0, 101.5]),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["price_model"]["model_type"] == "random_forest"
    assert payload["direction_model"]["model_type"] == "xgboost"
    # Two different winners is the normal case, not a degenerate one.
    assert payload["price_model"]["model_type"] != payload["direction_model"]["model_type"]
    assert payload["status"] == "ok"


def test_every_forecast_point_carries_the_sign_of_its_own_step(client):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("random_forest", direction={"accuracy": 0.55, "f1": 0.54, "roc_auc": 0.56})],
    )
    # Up from 100, up again, then down: the colour of each segment is the move
    # that segment represents, not the shape of the whole path.
    response = _call(
        client,
        best,
        price_response=_forecast_response([101.0, 103.0, 99.0], current_price=100.0),
    )

    points = response.json()["forecast"]
    assert [point["direction"] for point in points] == ["up", "up", "down"]
    assert points[0]["change_pct"] == pytest.approx(1.0)
    assert points[1]["change_pct"] == pytest.approx((103 / 101 - 1) * 100, rel=1e-6)
    # The bands travel with each point, so the chart never has to guess them.
    assert points[0]["upper95"] > points[0]["predicted"] > points[0]["lower95"]


def test_a_step_under_a_basis_point_is_flat_not_a_direction(client):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("random_forest", direction={"accuracy": 0.55, "f1": 0.54, "roc_auc": 0.56})],
    )
    response = _call(
        client,
        best,
        price_response=_forecast_response([100.001], current_price=100.0),
    )

    assert response.json()["forecast"][0]["direction"] == "flat"


def test_direction_is_always_up_or_down_whichever_model_produced_it(client):
    """
    One vocabulary on the chart contract.

    The unified path words its own output as Bullish/Bearish and the regression
    path as UP/DOWN; a client matching on both would be matching on which model
    happened to win.
    """
    best = _best_models(
        [_candidate("unified_lstm", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0}, horizon=1)],
        [_candidate("unified_lstm", direction={"accuracy": 0.55, "f1": 0.54, "roc_auc": 0.56}, horizon=1)],
    )
    response = _call(
        client,
        best,
        horizon=1,
        unified_response=_forecast_response(
            [99.0], model_type="unified_lstm", horizon=1, probability_up=0.11
        ),
    )

    direction = response.json()["direction"]
    assert direction["direction"] == "DOWN"
    assert direction["probability_up"] == pytest.approx(0.11)
    assert direction["probability_down"] == pytest.approx(0.89)
    assert direction["source"] == "unified_bundle"


def test_a_regression_winner_reports_a_sign_and_no_fabricated_probability(client):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("random_forest", direction={"accuracy": 0.55})],
    )
    response = _call(
        client,
        best,
        price_response=_forecast_response([104.0], current_price=100.0),
    )

    direction = response.json()["direction"]
    assert direction["direction"] == "UP"
    assert direction["source"] == "regression_sign"
    assert direction["probability_up"] is None
    assert "no calibrated probability" in direction["message"]


# ---------------------------------------------------------------------------
# Horizon
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("horizon", [7, 15, 30, 60])
def test_the_requested_horizon_scopes_both_the_ranking_and_the_forecast(client, horizon):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0}, horizon=horizon)],
        [_candidate("random_forest", direction={"accuracy": 0.55}, horizon=horizon)],
        horizon=horizon,
    )
    patches = _patched(best, price_response=_forecast_response([101.0] * horizon))
    for entry in patches:
        entry.start()
    try:
        response = client.get(f"/api/predict/best/TEST?horizon={horizon}")
        selection_call = predict_route.select_best_models.call_args
    finally:
        for entry in patches:
            entry.stop()

    assert response.status_code == 200
    # The ranking is asked for the same window the chart is showing, because a
    # 30-day error and a 7-day error are not the same measurement.
    assert selection_call.args[1] == horizon
    assert response.json()["horizon"] == horizon
    assert len(response.json()["forecast"]) == horizon


def test_a_next_bar_winner_reports_the_horizon_it_was_actually_scored_at(client):
    """A one-step model must not look like it forecast thirty days."""
    best = _best_models(
        [_candidate("unified_kronos", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0}, horizon=1)],
        [_candidate("unified_kronos", direction={"accuracy": 0.55, "f1": 0.54, "roc_auc": 0.56}, horizon=1)],
    )
    response = _call(
        client,
        best,
        horizon=30,
        unified_response=_forecast_response(
            [101.0], model_type="unified_kronos", horizon=1, probability_up=0.61
        ),
    )

    payload = response.json()
    assert payload["horizon"] == 30
    assert payload["price_model"]["scored_horizon"] == 1
    assert len(payload["forecast"]) == 1


# ---------------------------------------------------------------------------
# Nothing to draw
# ---------------------------------------------------------------------------

def test_no_servable_model_is_a_200_that_explains_itself(client):
    blocked = Candidate(
        model_type="xgboost",
        label="XGBoost",
        evidence="bundle_holdout",
        horizon=30,
        price={"mae": 1.0, "rmse": 1.0, "mape": 1.0},
        blocked_reason="the bundle does not beat a constant forecast",
    )
    best = _best_models([blocked], [blocked])

    response = _call(client, best)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "no_servable_model"
    assert payload["forecast"] == []
    assert payload["message"]
    # The ranking table still travels, so the UI can say who was excluded and why.
    excluded = payload["selection"]["price"]["excluded"]
    assert excluded[0]["model_type"] == "xgboost"
    assert "constant forecast" in excluded[0]["excluded_because"]


def test_a_winner_that_cannot_be_served_degrades_to_partial(client):
    """
    The bundle was ranked on stored metrics but its artifact is gone.

    Drawing nothing and saying nothing would look identical to a healthy symbol
    with a flat forecast, so the status distinguishes them.
    """
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("random_forest", direction={"accuracy": 0.55})],
    )
    response = _call(
        client,
        best,
        price_response=_unavailable_response("random_forest", "the bundle could not be loaded"),
    )

    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "winners_could_not_be_served"
    assert payload["message"] == "the bundle could not be loaded"


def test_a_price_line_without_a_direction_call_is_partial(client):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("unified_lstm", direction={"accuracy": 0.57, "f1": 0.56, "roc_auc": 0.59}, horizon=1)],
    )
    response = _call(
        client,
        best,
        price_response=_forecast_response([101.0, 102.0]),
        unified_response=_unavailable_response("unified_lstm", "the unified bundle is missing"),
    )

    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["reason"] == "direction_model_unavailable"
    assert len(payload["forecast"]) == 2
    assert payload["direction"]["source"] == "unavailable"


# ---------------------------------------------------------------------------
# Gating
# ---------------------------------------------------------------------------

def test_a_direction_model_that_failed_its_verdict_is_marked_not_tradeable(client):
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("logistic", direction={"accuracy": 0.57, "f1": 0.56, "roc_auc": 0.59}, horizon=1)],
    )
    gated = Candidate(
        model_type="logistic",
        label="Logistic",
        evidence="direction_walk_forward",
        horizon=1,
        direction={"accuracy": 0.57},
        blocked_reason="it did not clear its walk-forward ship criteria",
    )

    patches = _patched(
        best,
        price_response=_forecast_response([101.0]),
        gate_candidates=[gated],
    )
    for entry in patches:
        entry.start()
    try:
        with patch(
            "src.models.direction_pipeline.predict_next_session",
            return_value={"probability_up": 0.58, "target_date": "2026-06-16"},
        ):
            response = client.get("/api/predict/best/TEST?horizon=30")
    finally:
        for entry in patches:
            entry.stop()

    direction = response.json()["direction"]
    assert direction["direction"] == "UP"
    assert direction["source"] == "direction_classifier"
    assert direction["tradeable"] is False
    assert "did not clear" in direction["gate_reason"]


def test_the_forecast_path_provenance_travels_with_the_points(client):
    """
    A compounded path is not thirty predictions, and the chart has to be able
    to say so rather than implying a daily forecast the model never made.
    """
    best = _best_models(
        [_candidate("random_forest", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0})],
        [_candidate("random_forest", direction={"accuracy": 0.55})],
    )
    response = _call(client, best, price_response=_forecast_response([101.0, 102.0]))

    payload = response.json()
    assert payload["path_type"] == "compounded_interpolation"
    assert payload["per_step_predictions"] is False
