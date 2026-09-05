"""
The gate that decides whether a next-day direction bundle may be served.

Two failures are covered separately because they need different fixes and the
serving layer has to be able to tell a reader which one it hit: a model that
ranks no better than a coin flip, and a model that has collapsed onto the base
rate and emits the same probability for every bar. The second is the one ROC-AUC
cannot catch on its own -- a collapsed model can score a high AUC on rank order
while being incapable of ever crossing the BUY/SELL bands.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import predict as predict_route
from src.defaults import ENFORCE_MODEL_SKILL_ENV
from src.models.direction_utils import (
    MIN_PROBABILITY_STD,
    direction_skill_failure,
    direction_skill_passes,
    direction_skill_record,
)


def _sample_ohlcv(rows: int = 200) -> pd.DataFrame:
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


class _GatedBundle:
    """A loadable direction bundle whose recorded verdict the test controls."""

    def __init__(self, *, skill: dict, passes: bool, probability_up: float = 0.72):
        self.model_type = "xgboost"
        self.symbol = "AAPL"
        self.feature_columns = ["Daily_Return"]
        self.scaler = None
        self.feature_config = {}
        self.version_id = "bundle_v1"
        self.horizon = 1
        self.sequence_length = 60
        self.model = _ConstantModel(probability_up)
        self.metadata = {
            "objective": "next_day_direction",
            "target_type": "direction",
            "trained_at": "2026-09-05T10:00:00",
            "passes_baseline": passes,
            "skill": {"test": skill},
        }


SKILFUL = {"roc_auc": 0.58, "skill_score": 0.08, "probability_std": 0.19}
COLLAPSED = {"roc_auc": 0.63, "skill_score": 0.13, "probability_std": 0.0001}
NO_SKILL = {"roc_auc": 0.47, "skill_score": -0.03, "probability_std": 0.18}


# -- the record -------------------------------------------------------------

def test_record_scores_a_ranking_model_above_the_coin_flip():
    y = np.array([0, 1] * 100)
    probs = np.where(y == 1, 0.8, 0.2)
    record = direction_skill_record(y, probs)

    assert record["roc_auc"] == pytest.approx(1.0)
    assert record["skill_score"] == pytest.approx(0.5)
    assert record["probability_std"] > MIN_PROBABILITY_STD
    assert direction_skill_passes(record) is True


def test_record_exposes_a_collapsed_model_through_its_spread():
    y = np.array([0, 1] * 100)
    record = direction_skill_record(y, np.full(200, 0.5297))

    # The spread is the whole signal here: a constant vector has no ranking, so
    # AUC alone would report an unremarkable 0.5 and say nothing about the fault.
    assert record["probability_std"] == pytest.approx(0.0)
    assert direction_skill_passes(record) is False


def test_record_treats_a_single_class_sample_as_no_evidence():
    # A split that landed on one class cannot be ranked. That is a property of
    # the sample, not a verdict the model earned, so it scores the neutral 0.5
    # rather than raising out of roc_auc_score.
    record = direction_skill_record(np.ones(50), np.linspace(0.2, 0.8, 50))

    assert record["roc_auc"] == pytest.approx(0.5)
    assert record["skill_score"] == pytest.approx(0.0)
    assert direction_skill_passes(record) is False


def test_record_handles_empty_input():
    assert direction_skill_record([], []) == {
        "roc_auc": 0.5,
        "skill_score": 0.0,
        "probability_std": 0.0,
        "positive_rate": 0.0,
    }


def test_passes_requires_both_gates():
    assert direction_skill_passes(SKILFUL) is True
    # Ranks well but cannot reach the signal bands.
    assert direction_skill_passes(COLLAPSED) is False
    # Spreads widely but ranks worse than guessing.
    assert direction_skill_passes(NO_SKILL) is False


# -- the verdict a bundle carries -------------------------------------------

def test_failure_is_none_for_a_bundle_that_passed():
    assert direction_skill_failure({"passes_baseline": True}) is None


def test_failure_names_the_missing_record_for_an_ungated_bundle():
    # Bundles trained before the gate existed carry no verdict, and those are
    # precisely the ones that turned out to be predicting the base rate.
    message = direction_skill_failure({})

    assert message is not None
    assert "no evidence" in message


def test_failure_distinguishes_collapse_from_absence_of_skill():
    collapsed = direction_skill_failure({"passes_baseline": False, "skill": {"test": COLLAPSED}})
    no_skill = direction_skill_failure({"passes_baseline": False, "skill": {"test": NO_SKILL}})

    # The remedies differ -- a collapsed model needs a different architecture,
    # one without skill needs a different signal -- so the messages must too.
    assert "same probability for every bar" in collapsed
    assert "coin flip" in no_skill
    assert collapsed != no_skill


def test_enforcement_can_be_switched_off_while_retraining(monkeypatch):
    monkeypatch.setenv(ENFORCE_MODEL_SKILL_ENV, "false")
    assert direction_skill_failure({"passes_baseline": False, "skill": {"test": NO_SKILL}}) is None


# -- the serving contract ---------------------------------------------------

def _client():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    return TestClient(app)


def _get_signals(bundle):
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "load_model_bundle", return_value=bundle),
    ):
        return _client().get("/api/predict/historical-signals/AAPL?days=90&model_type=xgboost")


def test_historical_signals_serves_a_bundle_that_cleared_the_gate():
    response = _get_signals(_GatedBundle(skill=SKILFUL, passes=True, probability_up=0.72))

    assert response.status_code == 200
    signals = response.json()
    assert signals and all(signal["type"] == "BUY" for signal in signals)


@pytest.mark.parametrize(
    "skill, expected",
    [
        (COLLAPSED, "same probability for every bar"),
        (NO_SKILL, "coin flip"),
    ],
)
def test_historical_signals_refuses_a_bundle_that_failed_the_gate(skill, expected):
    # These land on a price chart as BUY and SELL marks, the most credible
    # presentation the app has. A refusal the reader can act on beats a drawing
    # of noise that carries no cue the model failed its own gate.
    response = _get_signals(_GatedBundle(skill=skill, passes=False))

    assert response.status_code == 409
    assert expected in response.json()["detail"]


def test_backtest_skips_a_failed_bundle_so_the_fallback_banner_shows():
    from src.models import model_bundle

    with patch.object(
        model_bundle, "load_model_bundle", return_value=_GatedBundle(skill=NO_SKILL, passes=False)
    ):
        from src.backtesting import backtest_service

        predictions = backtest_service._get_ml_predictions(
            _sample_ohlcv(), symbol="AAPL", model_type="xgboost"
        )

    # None is what makes the caller record `fallback_ta_only`; crediting the
    # model for a run that is really the TA rule's is the claim that banner
    # exists to prevent.
    assert predictions is None
