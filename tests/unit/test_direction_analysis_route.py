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
from unittest.mock import patch

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


# ─────────────────────────────────────────────────────────────────────────────
# Timeframes: the analysis has to move with the bar, and the classifier must not
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def long_bars(monkeypatch):
    """
    Twenty years of daily bars — what the route actually downloads for a weekly
    request. The 2,000-bar `injected_bars` fixture aggregates to 400 weekly
    candles, which is under the weekly walk-forward's floor, so a test built on
    it would pass by measuring nothing.
    """
    frame = _bars(n=5100, seed=404)
    meta = {
        "ticker": SYMBOL, "first_bar": str(frame.index[0].date()),
        "last_bar": str(frame.index[-1].date()), "clean_rows": len(frame),
        "price_basis": "dividend_and_split_adjusted", "content_sha256": "deadbeef",
    }
    monkeypatch.setattr(
        direction_route, "load_daily_bars", lambda *args, **kwargs: BarLoad(frame=frame, meta=meta)
    )
    return frame


def test_the_analysis_runs_on_the_bar_that_was_asked_for(client, long_bars):
    """
    Not a relabelling. The weekly answer is measured on weekly candles — its
    `as_of` is a weekly bar's label, its walk-forward scored weekly outcomes,
    and its evidence was read off weekly indicators.
    """
    from src.data.timeframe import next_bar_date, resample_ohlcv, resolve_timeframe

    daily = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    weekly = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week").json()

    assert daily["status"] == weekly["status"] == "ok"
    assert daily["timeframe"] == "day" and weekly["timeframe"] == "week"
    assert daily["bar_noun"] == "session" and weekly["bar_noun"] == "week"
    assert daily["horizon_label"] == "Next 1 Day"
    assert weekly["horizon_label"] == "Next 1 Week"

    week = resolve_timeframe("week")
    expected = resample_ohlcv(long_bars, week)
    assert weekly["as_of"] == str(pd.Timestamp(expected.index[-1]).date())
    assert weekly["bars_analysed"] == len(expected)
    assert weekly["bars_analysed"] < daily["bars_analysed"]
    assert weekly["forecast_bar"] == next_bar_date(expected.index[-1], week)

    # Both are genuinely measured — a weekly walk-forward that scored nothing
    # would satisfy every assertion above while measuring nothing at all.
    weekly_stack = weekly["blend"]["evidence_stack"]
    daily_stack = daily["blend"]["evidence_stack"]
    assert weekly_stack["n_test_rows"] > 0 and daily_stack["n_test_rows"] > 0
    assert weekly_stack["n_test_rows"] < daily_stack["n_test_rows"]
    # A smaller sample has to show up as a wider interval, not as more
    # confidence. This is the property that makes the lower weekly floor honest.
    weekly_ci = weekly_stack["accuracy_ci"]
    daily_ci = daily_stack["accuracy_ci"]
    assert (weekly_ci[1] - weekly_ci[0]) > (daily_ci[1] - daily_ci[0])


def test_the_daily_classifier_does_not_vote_on_a_weekly_call(client, long_bars, tmp_path):
    """
    The stored report is a measurement about *tomorrow*. Carrying its Brier
    skill into a weekly blend would hand it a weight it earned on a different
    question — the one thing the blending rule exists to forbid — so on the
    resampled frames it is excluded, with the reason in the payload.
    """
    _write_report(tmp_path, ship=True, brier_skill_score=0.05)

    daily = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    assert daily["blend"]["classifier"]["weight"] > 0, "the fixture has to put it IN on daily"

    weekly = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week").json()
    assert weekly["status"] == "ok", "this has to exercise the served path, not the refusal"
    assert weekly["blend"]["classifier"]["weight"] == 0.0
    assert weekly["blend"]["classifier"]["model"] is None
    assert "daily bars" in weekly["classifier_note"]
    assert "next-day label" in weekly["classifier_note"]


def test_a_weekly_view_does_not_queue_a_daily_training_run(client, injected_bars):
    """
    Preparation is started only when it would change the answer. On the weekly
    frame the classifier is out because it answers a different question, not
    because its report is missing — so training one would change nothing, and
    kicking one off per chart view is pure waste.
    """
    started = []

    def _spy(symbol, direction_model=None, auto_start=False):
        started.append(auto_start)
        return {"status": "idle"}

    with patch.object(direction_route, "preparation_state", _spy):
        client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week")
        assert started == [False]

        started.clear()
        direction_route.clear_analysis_cache()
        client.get(f"/api/direction/{SYMBOL}/analysis")
        assert started == [True], "the daily path still starts one when the report is missing"


def test_the_analysis_cache_is_keyed_on_the_timeframe(client, long_bars):
    """
    A key without it would serve the daily analysis under a weekly label on any
    Friday, when the two frames end on the same date.
    """
    daily = client.get(f"/api/direction/{SYMBOL}/analysis").json()
    weekly = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week").json()
    assert weekly["cached"] is False, "the weekly request must not hit the daily entry"

    assert daily["timeframe"] == "day" and weekly["timeframe"] == "week"
    assert daily["bars_analysed"] != weekly["bars_analysed"]

    again = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week").json()
    assert again["cached"] is True
    assert again["timeframe"] == "week"
    assert again["bars_analysed"] == weekly["bars_analysed"]


def test_the_level_detail_names_the_bar_it_was_drawn_from(client, long_bars):
    """A 20-bar range on weekly candles is twenty weeks; "20-day range" is a factual error."""
    weekly = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=week").json()
    rows = {row["source"]: row for row in weekly["evidence"]}
    detail = rows["support_resistance"].get("detail")
    if detail:
        assert "-week range" in detail
        assert "-day range" not in detail


def test_an_unknown_timeframe_is_refused(client, injected_bars):
    response = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=quarter")
    assert response.status_code == 422


def test_a_monthly_frame_too_short_to_measure_says_so_rather_than_calling_it(client, injected_bars):
    """
    Eight years of daily bars is 93 monthly candles, and seven evidence
    categories built from 20-, 50- and 60-bar windows cannot all be read on a
    frame that short. The right answer is to decline with the row count, not to
    fit a logistic stack on the handful of rows that survive and print a
    direction from it.
    """
    body = client.get(f"/api/direction/{SYMBOL}/analysis?timeframe=month").json()

    assert body["status"] == "unavailable"
    assert body["timeframe"] == "month"
    assert body["bars_analysed"] < 128
    assert "direction" not in body
    # The message names the size of the frame, so a reader can tell "too little
    # history" from "something is broken".
    assert str(body["bars_analysed"]) in body["message"] or "too short" in body["message"]
