from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import predict as predict_route


def _sample_ohlcv(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series([100 + i * 0.4 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1000 + i for i in range(rows)],
        },
        index=index,
    )


class _ConstantModel:
    def __init__(self, probability_up: float):
        self.probability_up = probability_up

    def predict_proba(self, X):
        return [[1 - self.probability_up, self.probability_up] for _ in range(len(X))]

    def predict(self, X):
        return [0.001 for _ in range(len(X))]


class _FakeBundle:
    def __init__(self, probability_up: float = 0.72):
        self.model_type = "xgboost"
        self.feature_columns = ["Daily_Return"]
        self.scaler = None
        self.feature_config = {}
        self.version_id = "bundle_v1"
        self.artifact_dir = "models/bundles/AAPL/xgboost"
        self.horizon = 1
        self.supported_horizons = [1, 7, 15, 30, 60]
        self.bundle_layout = "canonical_symbol_model"
        self.symbol = "AAPL"
        self.model = _ConstantModel(probability_up)
        self.metadata = {
            "objective": "next_day_direction",
            "target_type": "direction",
            "serving_mode": "next_day_direction_classifier",
            "feature_count": 1,
            "trained_at": "2026-04-12T10:00:00",
        }
        self.sequence_length = 60


def test_predict_returns_explicit_unavailable_when_no_bundle_exists():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(147.6, "latest_close")),
        patch.object(predict_route, "load_model_bundle", return_value=None),
    ):
        response = client.post(
            "/api/predict",
            json={"symbol": "AAPL", "model_type": "xgboost", "horizon": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["model_available"] is False
    assert payload["reason"] == "missing_bundle"
    assert payload["model_info"]["status"] == "unavailable"
    assert payload["can_train"] is True
    assert "No trained xgboost bundle found" in payload["model_info"]["message"]


def test_predict_uses_recursive_one_step_bundle_when_available():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(147.6, "latest_close")),
        patch.object(predict_route, "load_model_bundle", return_value=_FakeBundle(probability_up=0.72)),
    ):
        response = client.post(
            "/api/predict",
            json={"symbol": "AAPL", "model_type": "xgboost", "horizon": 5},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model_info"]["artifact_source"] == "canonical_symbol_model_bundle"
    assert payload["model_available"] is True
    assert payload["horizon"] == 5
    assert payload["direction"] == "Bullish"
    assert payload["signal"] == "BUY"
    assert payload["probability_up"] == 0.72
    assert payload["probability_down"] == 0.28
    assert payload["confidence"] == 72.0
    assert payload["prediction_date"]
    assert len(payload["forecasts"]) == 5


def test_ensemble_predict_returns_daily_prediction_series():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    dates = ["2026-04-28", "2026-04-29", "2026-04-30", "2026-05-01", "2026-05-04", "2026-05-05", "2026-05-06"]
    forecast_points = [
        {
            "date": dates[index],
            "predicted": value,
            "xgboost": value - 0.5,
            "random_forest": value + 0.25,
            "lstm": value,
            "lower_95": value - 8,
            "upper_95": value + 8,
            "lower_68": value - 4,
            "upper_68": value + 4,
        }
        for index, value in enumerate([270.0, 267.5, 264.0, 259.0, 261.0, 258.5, 259.5])
    ]
    forecast = SimpleNamespace(
        symbol="SHOP",
        current_price=272.0,
        horizon=7,
        predicted_price=259.5,
        expected_change_pct=-4.6,
        confidence_interval={"lower": 251.5, "upper": 267.5},
        reliability="Medium",
        reason="test forecast",
        signal="Bearish",
        model_predictions=[
            SimpleNamespace(model_type="xgboost", weight=0.35),
            SimpleNamespace(model_type="random_forest", weight=0.25),
            SimpleNamespace(model_type="lstm", weight=0.4),
        ],
        forecast_points=forecast_points,
    )

    class FakePredictor:
        def predict(self, **_kwargs):
            return forecast

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(predict_route, "ensemble_bundles_available", return_value=True),
        patch.object(predict_route, "EnsemblePricePredictor", return_value=FakePredictor()),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    assert response.status_code == 200
    payload = response.json()
    values = [point["prediction"] for point in payload["forecast_points"]]

    assert payload["status"] == "ok"
    assert len(payload["forecast_points"]) == 7
    assert values == [270.0, 267.5, 264.0, 259.0, 261.0, 258.5, 259.5]
    assert payload["forecast_points"][0]["ensemble"] == 270.0
    assert len(set(values)) > 1
