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
    assert "no trained xgboost bundle exists" in payload["model_info"]["message"]
    # The message must not hand the user a command to run. With automatic
    # preparation on, the server starts the training itself; with it off, the
    # remedy named is an API call an operator makes, not a terminal session.
    assert "scripts/" not in payload["model_info"]["message"]


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


def _fake_forecast():
    """A seven-day ensemble forecast with a distinct value on every day."""
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
    return SimpleNamespace(
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
        scenario_paths=[[272.0, 271.0, 268.0], [272.0, 274.0, 269.0]],
    )


def test_ensemble_predict_returns_daily_prediction_series():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    forecast = _fake_forecast()

    class FakePredictor:
        def predict(self, **_kwargs):
            return forecast

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(
            predict_route,
            "ensemble_availability",
            return_value=(["xgboost", "random_forest", "lstm"], {}),
        ),
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
    assert payload["scenario_paths"] == [[272.0, 271.0, 268.0], [272.0, 274.0, 269.0]]


def test_ensemble_predict_explains_why_an_unproven_bundle_is_not_served():
    """A bundle that fails the skill gate must say so, not read as 'not trained yet'."""
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    reason = "the xgboost bundle does not beat a constant forecast (skill score -0.7563)"
    blocked = {mtype: reason for mtype in ("xgboost", "random_forest", "lstm")}
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(predict_route, "ensemble_availability", return_value=([], blocked)),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["model_available"] is False
    assert "does not beat a constant forecast" in payload["message"]
    # A failed skill gate is a measurement, not a missing artifact, so the
    # response must not read as "come back once it is trained" — retraining the
    # same bars reproduces the same verdict, and preparation declines to try.
    assert "not a missing model" in payload["message"]
    assert payload["preparation"] is None
    assert payload["models_unavailable"] == blocked


def test_ensemble_serves_a_partial_ensemble_when_one_member_is_blocked():
    """
    One unservable member must not take the whole horizon offline.

    This is the regression behind "Prediction model unavailable": the gate
    required all three bundles, so a single failing LSTM blanked a tab whose
    XGBoost and Random Forest bundles were both ready to serve.
    """
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    forecast = _fake_forecast()
    requested: dict = {}

    class FakePredictor:
        def predict(self, **kwargs):
            requested.update(kwargs)
            return forecast

    blocked = {"lstm": "the lstm bundle does not beat a constant forecast"}
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(
            predict_route,
            "ensemble_availability",
            return_value=(["xgboost", "random_forest"], blocked),
        ),
        patch.object(predict_route, "EnsemblePricePredictor", return_value=FakePredictor()),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_available"] is True
    assert payload["degraded"] is True
    assert payload["models_available"] == ["xgboost", "random_forest"]
    assert payload["models_unavailable"] == blocked
    assert "lstm" in payload["message"]
    # The blocked member must not be handed to the predictor.
    assert requested["model_types"] == ["xgboost", "random_forest"]


def test_predict_reports_stale_bundle_instead_of_retraining_inline():
    """
    A feature-set mismatch must surface as an actionable `can_train` response.

    This previously deleted the bundle directory and ran a five-year training job
    inside the request, which blocked the event loop for minutes and left the symbol
    with no model at all if training then failed. Retraining belongs on
    POST /api/training/train.
    """
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    def _raise_feature_mismatch(*args, **kwargs):
        raise ValueError("feature columns do not match the stored bundle")

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(147.6, "latest_close")),
        patch.object(predict_route, "load_model_bundle", return_value=_FakeBundle()),
        patch.object(predict_route, "_predict_bundle_probabilities", _raise_feature_mismatch),
        patch("src.models.bundle_training.train_model_bundles") as mocked_train,
        patch("shutil.rmtree") as mocked_rmtree,
    ):
        response = client.post(
            "/api/predict",
            json={"symbol": "AAPL", "model_type": "xgboost", "horizon": 1},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["reason"] == "stale_bundle_requires_retraining"
    assert payload["can_train"] is True
    assert "/api/training/train" in payload["message"]

    # The critical guarantees: no inline training, and the existing bundle is untouched.
    mocked_train.assert_not_called()
    mocked_rmtree.assert_not_called()
