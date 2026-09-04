"""
GET /api/backtest/evidence/{symbol} — the walk-forward record, as the tab reads it.

The route has to serve two artifact generations at once. A file written before
the scoring rewrite carries fold aggregates and nothing else; one written after
carries the null tests, the cost overlay, the split protocol and a per-bar
record. The tests below pin both, because the failure mode that matters is not
a crash on the old file — it is the route quietly presenting an old file as
though the verdicts had been computed and come back empty.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import backtest as backtest_route


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(backtest_route.router, prefix="/api/backtest")
    return TestClient(app)


def _write(tmp_path, payload):
    path = tmp_path / "benchmark_results.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _legacy_row(symbol="AAPL", model="unified_xgboost"):
    """A row as the pre-rewrite benchmark wrote it: aggregates only."""
    return {
        "symbol": symbol,
        "model_name": model,
        "n_splits": 3,
        "total_test_rows": 189,
        "base_rate": 0.5185,
        "direction_accuracy": 0.5291,
        "price_mae": 3.45,
        "price_r2": 0.852,
        "price_r2_return": -0.1505,
        "per_fold": [{"fold": 1, "test_start": "2026-01-02", "test_size": 63}],
    }


def _current_row(symbol="AAPL", model="unified_xgboost"):
    """A row as the benchmark writes it now, per-bar payload included."""
    row = _legacy_row(symbol, model)
    row.update(
        {
            "horizon": 1,
            "direction_eobr": -0.0265,
            "direction_eobr_std": 0.0374,
            "split_protocol": [
                {"fold": 1, "purge_bars": 1, "embargo_bars": 1, "gap_bars": 2, "gap_calendar_days": 6}
            ],
            "evaluation": {
                "available": True,
                "n": 189,
                "effective_n": 189.0,
                "direction": {"eobr": 0.0106, "accuracy": 0.5291},
                "edge_vs_majority": {"p_value_one_sided": 0.3855, "significant": False},
                "economics": {"available": True, "breakeven": {"arithmetic_bps": 12.87}},
                "vs_random_walk": {
                    "r2_vs_random_walk": -0.1772,
                    "diebold_mariano": {"sign": "baseline_better", "p_value": 0.0009, "n": 189},
                },
            },
        }
    )
    row["per_fold"] = [
        {
            "fold": 1,
            "test_start": "2026-01-02",
            "test_size": 63,
            # The heavy part. It must not reach the browser.
            "predictions": {"date": ["2026-01-02"], "p_up": [0.51], "actual_price": [187.4]},
        }
    ]
    return row


def test_missing_artifact_instructs_rather_than_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", tmp_path / "absent.json")
    response = client.get("/api/backtest/evidence/AAPL")

    assert response.status_code == 200
    body = response.json()
    assert body["models"] == []
    assert "unified_benchmark.py" in body["message"]


def test_unknown_symbol_names_the_symbols_that_do_have_records(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", _write(tmp_path, [_legacy_row()]))
    body = client.get("/api/backtest/evidence/ZZZZ").json()

    assert body["models"] == []
    assert "AAPL" in body["message"]


def test_legacy_artifact_reports_absent_verdicts_as_null(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", _write(tmp_path, [_legacy_row()]))
    body = client.get("/api/backtest/evidence/AAPL").json()

    assert len(body["models"]) == 1
    entry = body["models"][0]
    # The aggregates it does have survive.
    assert entry["price"]["r2_return"] == pytest.approx(-0.1505)
    # The verdicts it does not have are absent, not zero and not invented.
    assert entry["vs_random_walk"] is None
    assert entry["economics"] is None
    assert entry["direction"]["eobr"] is None
    assert "predates the null tests" in body["message"]


def test_current_artifact_surfaces_the_verdicts(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", _write(tmp_path, [_current_row()]))
    body = client.get("/api/backtest/evidence/AAPL").json()

    entry = body["models"][0]
    assert entry["horizon"] == 1
    assert entry["vs_random_walk"]["r2"] == pytest.approx(-0.1772)
    assert entry["vs_random_walk"]["verdict"] == "baseline_better"
    assert entry["vs_random_walk"]["p_value"] == pytest.approx(0.0009)
    assert entry["economics"]["breakeven"]["arithmetic_bps"] == pytest.approx(12.87)
    assert entry["effective_n"] == pytest.approx(189.0)
    assert entry["split_protocol"][0]["gap_bars"] == 2
    assert "predates" not in body["message"]


def test_per_bar_arrays_are_stripped_from_the_response(client, tmp_path, monkeypatch):
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", _write(tmp_path, [_current_row()]))
    body = client.get("/api/backtest/evidence/AAPL").json()

    fold = body["models"][0]["folds"][0]
    assert "predictions" not in fold
    # The summary the panel actually draws is still there.
    assert fold["test_start"] == "2026-01-02"
    assert fold["test_size"] == 63


def test_model_filter_selects_one_row(client, tmp_path, monkeypatch):
    payload = [_current_row(model="unified_xgboost"), _current_row(model="unified_lstm")]
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", _write(tmp_path, payload))

    body = client.get("/api/backtest/evidence/AAPL", params={"model": "unified_lstm"}).json()
    assert [entry["model_type"] for entry in body["models"]] == ["unified_lstm"]


def test_corrupt_artifact_is_treated_as_absent(client, tmp_path, monkeypatch):
    path = tmp_path / "benchmark_results.json"
    path.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(backtest_route, "BENCHMARK_ARTIFACT", path)

    body = client.get("/api/backtest/evidence/AAPL").json()
    assert body["models"] == []
    assert "unified_benchmark.py" in body["message"]
