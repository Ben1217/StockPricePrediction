from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
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


class _FakeFoundationMember:
    """One foundation pipeline: a point, a P(up), and a real spread to weight by."""

    def __init__(self, price: float, p_up: float, spread: float = 3.0):
        self.price = price
        self.p_up = p_up
        self.spread = spread

    def predict(self, df, horizon: int = 1, covariates=None):
        samples = np.linspace(self.price - self.spread, self.price + self.spread, 128)
        return {
            "price": self.price,
            "p_up": self.p_up,
            "samples": samples,
            "quantiles": {q: float(np.quantile(samples, q)) for q in (0.1, 0.5, 0.9)},
        }


def _serving(members):
    """A `_get_foundation_pipeline` stand-in over a {model_type: member} map."""
    return lambda model_type: members.get(model_type)


def test_ensemble_predict_serves_the_foundation_aggregate():
    """
    POST /predict/ensemble is the older response shape over the SAME forecast
    the Predictions tab serves -- both go through `run_foundation_forecast`.
    These used to be two implementations of one calculation, which is how they
    could disagree about the same symbol on the same bar.
    """
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    members = {
        "unified_kronos": _FakeFoundationMember(270.0, 0.30),
        "unified_chronos": _FakeFoundationMember(268.0, 0.35),
        "unified_timesfm": _FakeFoundationMember(269.0, 0.40),
    }
    # 430 rows puts the last close at 271.60, just above every member. The
    # quote is the number this assertion used to be written against, back when
    # `change_pct` was measured from it; the anchor is a bar the models read.
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=430)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_serving(members)),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["model_available"] is True
    assert payload["degraded"] is False
    assert payload["models_available"] == ["unified_kronos", "unified_chronos", "unified_timesfm"]

    # Every member answers for the next bar, so there is exactly one point
    # whatever horizon was asked for -- and the horizon is echoed, not applied.
    assert payload["horizon"] == 7
    assert len(payload["forecast_points"]) == 1

    summary = payload["ensemble"]
    assert 268.0 <= summary["target"] <= 270.0
    # P(up) is under 0.5 across every member, so the call is DOWN -- and the
    # forecast is below the 271.60 close those probabilities were computed
    # against, so the change is negative too. The tile prints the two in one
    # string, so they have to be measured from the same price to agree.
    assert payload["anchor_price"] == pytest.approx(271.60)
    assert summary["signal"] == "DOWN"
    assert summary["change_pct"] < 0
    # The quote reading is served beside it and measured from 272.00, so the
    # two differ by however far the quote has drifted off the anchor. Here that
    # is 40 cents and both still fall; the point is that they are separate
    # numbers, so a wider gap cannot silently overwrite the one on the left.
    assert summary["quote_change_pct"] < 0
    assert summary["quote_change_pct"] != summary["change_pct"]

    # The band is named for the coverage it has: q0.05..q0.95 is 90%, not 95%.
    # This is the field the Analysis tab's band card reads.
    assert summary["lower_90"] < summary["target"] < summary["upper_90"]
    assert summary["lower_68"] > summary["lower_90"]
    assert summary["upper_68"] < summary["upper_90"]
    assert summary["upper_95"] is None and summary["lower_95"] is None


def test_ensemble_predict_says_which_member_is_missing():
    """
    One unservable member must not take the whole ensemble offline -- but the
    client has to be able to say the result is partial rather than presenting
    two models as three.
    """
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    class _Broken:
        def predict(self, df, horizon: int = 1, covariates=None):
            raise RuntimeError("model weights are not downloaded")

    members = {
        "unified_kronos": _FakeFoundationMember(270.0, 0.62),
        "unified_chronos": _Broken(),
        "unified_timesfm": None,
    }
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_serving(members)),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_available"] is True
    assert payload["degraded"] is True
    assert payload["models_available"] == ["unified_kronos"]
    assert set(payload["models_unavailable"]) == {"unified_chronos", "unified_timesfm"}
    assert "model weights are not downloaded" in payload["models_unavailable"]["unified_chronos"]
    assert "Chronos-2" in payload["message"] and "TimesFM 2.5" in payload["message"]


def test_ensemble_predict_refuses_to_serve_a_point_with_no_model_behind_it():
    """
    Nothing servable is a 200 saying so, not a fabricated number. A point
    forecast with no interval is what Requirement 11.2 forbids.
    """
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    client = TestClient(app)

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=320)),
        patch.object(predict_route, "_latest_available_price", return_value=(272.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", return_value=None),
    ):
        response = client.post("/api/predict/ensemble", json={"symbol": "SHOP", "horizon": 7})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["model_available"] is False
    assert payload["ensemble"] is None
    assert payload["forecast_points"] == []
    assert payload["message"]
    assert set(payload["models_unavailable"]) == {
        "unified_kronos", "unified_chronos", "unified_timesfm",
    }


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
