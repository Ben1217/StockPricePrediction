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
        patch.object(predict_route, "load_model_bundle", return_value=None),
    ):
        response = client.post(
            "/api/predict",
            json={"symbol": "AAPL", "model_type": "xgboost", "horizon": 15},
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
