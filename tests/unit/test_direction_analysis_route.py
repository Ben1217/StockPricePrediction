"""
Contract tests for GET /api/direction/{symbol}/analysis.

Bars are injected rather than downloaded, so the suite stays offline. What is
pinned here is the route's half of the bargain: it answers without needing a
stored walk-forward report, it says so when the classifier is therefore absent,
it folds the classifier in once a report exists, and it never starts a training
run on a request it was able to answer.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.api.routes import direction as direction_route  # noqa: E402
from src.api.security import API_KEY_ENV, limiter  # noqa: E402
from src.data.direction_data import BarLoad  # noqa: E402

SYMBOL = "EVID"


def _bars(n: int = 2000, seed: int = 909) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2016-01-04", periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    close = 45 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(2_000_000, 9_000_000, n).astype(float)},
        index=index,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.setattr(direction_route, "REPORT_DIR", tmp_path)
    limiter.reset()
    direction_route.clear_analysis_cache()
    yield
    limiter.reset()
    direction_route.clear_analysis_cache()


@pytest.fixture
def injected_bars(monkeypatch):
    frame = _bars()
    meta = {
        "ticker": SYMBOL, "first_bar": str(frame.index[0].date()),
        "last_bar": str(frame.index[-1].date()), "clean_rows": len(frame),
        "price_basis": "dividend_and_split_adjusted", "content_sha256": "cafebabe",
    }
    monkeypatch.setattr(
        direction_route, "load_daily_bars", lambda *args, **kwargs: BarLoad(frame=frame, meta=meta)
    )
    return frame


@pytest.fixture
def client():
    return TestClient(app)


def _write_report(directory: Path, *, ship: bool, brier_skill_score: float) -> None:
    """A minimal report carrying the two fields the analysis route reads."""
    payload = {
        "generated_at": "2026-01-01T00:00:00",
        "config": {"model": "logistic", "horizon": 1, "n_folds_run": 4},
        "data": {"ticker": SYMBOL},
        "pooled": {
            "model": {
                "accuracy": 0.56,
                "skill": {"brier_skill_score": brier_skill_score},
            }
        },
        "verdict": {"ship": ship, "failed_criteria": [] if ship else ["beats_best_baseline_accuracy"]},
    }
    path = directory / f"{direction_route.report_stem(SYMBOL, 'logistic')}_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analysis_answers_without_a_stored_report(client, injected_bars):
    """
    The evidence stack evaluates itself, so a symbol nobody has trained still
    gets a direction, a probability and its evidence - and is told plainly that
    the classifier is not part of it.
    """
    response = client.get(f"/api/direction/{SYMBOL}/analysis")
    assert response.status_code == 200
    body = response.json()

    assert body["status"] == "ok"
    assert body["direction"] in {"UP", "DOWN", "NEUTRAL"}
    assert 0.0 < body["probability_up"] < 1.0
    assert body["probability_up"] + body["probability_down"] == pytest.approx(1.0)
    assert len(body["evidence"]) == 7
    assert body["blend"]["classifier"]["weight"] == 0.0
    assert "no walk-forward report" in body["classifier_note"]


def test_analysis_never_serves_a_direction_without_its_confidence(client, injected_bars):
    body = client.get(f"/api/direction/{SYMBOL}/analysis").json()

    assert body["confidence"]["label"] in {"Low", "Moderate", "High"}
    assert body["confidence"]["basis"]
    if body["direction"] == "NEUTRAL":
        assert body["neutral_reason"]


def test_evidence_rows_carry_a_state_and_a_signed_contribution(client, injected_bars):
    body = client.get(f"/api/direction/{SYMBOL}/analysis").json()

    for row in body["evidence"]:
        assert row["label"]
        assert row["state"]
        assert row["leans"] in {"up", "down", "neutral"}
        assert isinstance(row["contribution_pp"], (int, float))
    assert body["evidence_note"]


def test_horizons_and_price_action_travel_with_the_answer(client, injected_bars):
    body = client.get(f"/api/direction/{SYMBOL}/analysis").json()

    assert set(body["horizons"]["directions"]) == {"short", "medium", "long"}
    assert body["price_action"]["available"] is True
    assert body["price_action"]["structure_label"]
    assert body["historical_analogs"]["available"] is True


def test_a_shipped_report_joins_the_blend(client, injected_bars, tmp_path):
    _write_report(tmp_path, ship=True, brier_skill_score=0.05)

    body = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    classifier = body["blend"]["classifier"]

    assert classifier["weight"] == pytest.approx(0.05)
    assert classifier["probability_up"] is not None
    assert classifier["tradeable"] is True
    assert body["classifier_note"] is None


def test_a_failed_report_is_included_at_zero_weight_with_its_reason(client, injected_bars, tmp_path):
    """
    A classifier that did not clear its ship criteria is still reported - the
    panel should be able to show what it said - but it is given no say in the
    blended number.
    """
    _write_report(tmp_path, ship=False, brier_skill_score=-0.02)

    body = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    classifier = body["blend"]["classifier"]

    assert classifier["weight"] == 0.0
    assert classifier["tradeable"] is False
    assert "does not beat the best naive baseline" in classifier["gate_reason"]


def test_the_second_request_is_served_from_cache(client, injected_bars):
    first = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    second = client.get(f"/api/direction/{SYMBOL}/analysis").json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["probability_up"] == first["probability_up"]


def test_refresh_bypasses_the_cache(client, injected_bars):
    client.get(f"/api/direction/{SYMBOL}/analysis")
    refreshed = client.get(f"/api/direction/{SYMBOL}/analysis?refresh=true").json()

    assert refreshed["cached"] is False


def test_an_unknown_model_is_rejected(client, injected_bars):
    response = client.get(f"/api/direction/{SYMBOL}/analysis?model=not_a_model")
    assert response.status_code == 422


def test_a_symbol_with_no_bars_is_a_404(client, monkeypatch):
    def _fail(*args, **kwargs):
        raise ValueError("No daily bars available for NOPE")

    monkeypatch.setattr(direction_route, "load_daily_bars", _fail)
    response = client.get("/api/direction/NOPE/analysis")

    assert response.status_code == 404
    assert "No daily bars" in response.json()["detail"]


def test_too_little_history_is_reported_not_faked(client, monkeypatch):
    frame = _bars(n=200)
    meta = {"first_bar": "2016-01-04", "last_bar": "2016-10-10", "clean_rows": 200,
            "price_basis": "as_returned"}
    monkeypatch.setattr(
        direction_route, "load_daily_bars", lambda *a, **k: BarLoad(frame=frame, meta=meta)
    )

    body = client.get("/api/direction/SHORTY/analysis").json()
    assert body["status"] == "unavailable"
    assert body["message"]
    assert "direction" not in body
