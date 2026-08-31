"""
Contract tests for /api/direction.

The behaviour worth pinning here is the gate. A P(up) gauge is trivial to render
and trivially misleading on its own, so the route must never hand a client a
probability that looks actionable when the walk-forward verdict says it is not.
These tests fabricate a report on disk rather than running a walk-forward, so
they stay offline and fast.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.api.routes import direction as direction_route  # noqa: E402
from src.api.security import API_KEY_ENV, limiter  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def client():
    return TestClient(app)


def _report(ship: bool, failed=None, leakage_passed=True) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00",
        "config": {"model": "logistic", "horizon": 1, "n_folds_run": 4},
        "data": {
            "ticker": "TEST", "first_bar": "2020-01-02", "last_bar": "2025-12-31",
            "clean_rows": 1500, "price_basis": "dividend_and_split_adjusted",
            "content_sha256": "abc123",
        },
        "pooled": {
            "n_test_rows": 252,
            "test_range": ["2025-01-02", "2025-12-31"],
            "model": {
                "accuracy": 0.56, "accuracy_ci_low": 0.4979, "accuracy_ci_high": 0.6195,
                "balanced_accuracy": 0.55, "base_rate": 0.53, "roc_auc": 0.54,
                "brier_score": 0.2481, "log_loss": 0.688, "mcc": 0.06,
                "skill": {"brier_skill_score": 0.004}, "calibration": [],
            },
            "baselines": {
                "majority": {
                    "accuracy": 0.53, "accuracy_ci_low": 0.47, "accuracy_ci_high": 0.59,
                    "balanced_accuracy": 0.5, "mcc": 0.0,
                },
            },
            "best_baseline": "majority",
            "edge_vs_best_baseline": {
                "edge_pp": 3.0, "standard_error_pp": 3.15, "z": 0.95,
                "p_value_one_sided": 0.17, "significant": False, "alpha": 0.05,
                "n_test": 252, "n_required": 1718,
            },
            "backtest": {
                "strategy": {"total_return": 0.05, "sharpe": 0.4},
                "benchmark": {"total_return": 0.12, "sharpe": 0.8},
                "breakeven": {"cost_charged_bps": 10.0, "breakeven_cost_bps_positive": 4.2},
            },
        },
        "leakage_check": {"passed": leakage_passed, "note": "paired test"},
        "verdict": {
            "ship": ship,
            "criteria": {"beats_best_baseline_accuracy": True},
            "failed_criteria": failed or [],
        },
    }


@pytest.fixture
def report_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(direction_route, "REPORT_DIR", tmp_path)
    return tmp_path


def _write(report_dir: Path, report: dict, symbol="TEST", model="logistic"):
    stem = f"{symbol}_{model}"
    (report_dir / f"{stem}_report.json").write_text(json.dumps(report), encoding="utf-8")

    index = pd.bdate_range("2025-01-02", periods=80)
    pd.DataFrame({
        "prediction": [1, 0] * 40,
        "label": [1, 1, 0, 0] * 20,
        "probability_up": [0.55] * 80,
    }, index=index).to_csv(report_dir / f"{stem}_predictions.csv")
    pd.DataFrame({
        "equity": [1.0 + i * 0.001 for i in range(80)],
        "benchmark_equity": [1.0 + i * 0.002 for i in range(80)],
        "position": [1, 0] * 40,
        "drawdown": [0.0] * 80,
    }, index=index).to_csv(report_dir / f"{stem}_equity_curve.csv")


class TestDirectionRoute:
    def test_missing_report_is_404_with_the_command_to_fix_it(self, client, report_dir):
        response = client.get("/api/direction/NOPE?include_gauge=false")
        assert response.status_code == 404
        # The message has to be actionable: a bare 404 leaves the user guessing.
        assert "scripts/direction_backtest.py" in response.json()["detail"]

    def test_evaluation_always_carries_the_interval(self, client, report_dir):
        _write(report_dir, _report(ship=False, failed=["accuracy_edge_is_significant"]))
        payload = client.get("/api/direction/TEST?include_gauge=false").json()

        evaluation = payload["evaluation"]
        assert evaluation["accuracy"] == 0.56
        low, high = evaluation["accuracy_ci"]
        assert low < evaluation["accuracy"] < high
        assert evaluation["edge_vs_best_baseline"]["p_value_one_sided"] == 0.17
        assert evaluation["best_baseline"] == "majority"

    def test_gauge_is_gated_when_the_verdict_says_do_not_ship(self, client, report_dir, monkeypatch):
        _write(report_dir, _report(
            ship=False,
            failed=["accuracy_edge_is_significant", "beats_buy_and_hold_after_costs"],
        ))
        monkeypatch.setattr(
            direction_route, "predict_next_session",
            lambda *a, **k: {"as_of": "2026-01-02", "probability_up": 0.61,
                             "predicted_direction": "up", "train_base_rate": 0.53,
                             "edge_over_base_rate_pp": 8.0, "n_train_rows": 1400,
                             "model": "logistic"},
        )
        monkeypatch.setattr(direction_route, "load_daily_bars", lambda *a, **k: _FakeBars())
        monkeypatch.setattr(direction_route, "build_direction_dataset", lambda *a, **k: None)

        session = client.get("/api/direction/TEST").json()["next_session"]

        assert session["available"] is True
        assert session["probability_up"] == 0.61
        # The number is served, but never as something to act on.
        assert session["tradeable"] is False
        assert "edge is inside the noise" in session["gate_reason"]
        assert "loses to buy and hold" in session["gate_reason"]
        # And the caveat travels with the number so a client cannot drop it.
        assert "95% CI" in session["caveat"]

    def test_gauge_is_tradeable_only_when_the_verdict_ships(self, client, report_dir, monkeypatch):
        _write(report_dir, _report(ship=True))
        monkeypatch.setattr(
            direction_route, "predict_next_session",
            lambda *a, **k: {"as_of": "2026-01-02", "probability_up": 0.58,
                             "predicted_direction": "up", "train_base_rate": 0.53,
                             "edge_over_base_rate_pp": 5.0, "n_train_rows": 1400,
                             "model": "logistic"},
        )
        monkeypatch.setattr(direction_route, "load_daily_bars", lambda *a, **k: _FakeBars())
        monkeypatch.setattr(direction_route, "build_direction_dataset", lambda *a, **k: None)

        session = client.get("/api/direction/TEST").json()["next_session"]
        assert session["tradeable"] is True
        assert session["gate_reason"] is None

    def test_a_failing_gauge_does_not_take_down_the_report(self, client, report_dir, monkeypatch):
        """A stale or offline gauge must degrade, not 500 the evaluation."""
        _write(report_dir, _report(ship=False, failed=["positive_probability_skill"]))

        def boom(*args, **kwargs):
            raise RuntimeError("yahoo is down")

        monkeypatch.setattr(direction_route, "load_daily_bars", boom)
        response = client.get("/api/direction/TEST")

        assert response.status_code == 200
        assert response.json()["next_session"]["available"] is False
        assert response.json()["evaluation"]["accuracy"] == 0.56

    def test_rolling_hit_rate_reports_the_base_rate_beside_it(self, client, report_dir):
        _write(report_dir, _report(ship=False, failed=["x"]))
        points = client.get("/api/direction/TEST?include_gauge=false").json()["rolling_hit_rate"]

        assert points, "a rolling strip should be produced from the predictions CSV"
        for point in points:
            # A hit rate without the base rate beside it is unreadable.
            assert "base_rate" in point and "hit_rate" in point
            assert point["window"] == min(direction_route.ROLLING_HIT_RATE_WINDOW, 80)

    def test_equity_curve_carries_both_series(self, client, report_dir):
        _write(report_dir, _report(ship=False, failed=["x"]))
        curve = client.get("/api/direction/TEST?include_gauge=false").json()["equity_curve"]

        assert len(curve) == 80
        assert curve[0]["strategy"] == 1.0
        assert all("benchmark" in point for point in curve)

    def test_failed_leakage_check_is_surfaced(self, client, report_dir):
        _write(report_dir, _report(ship=False, failed=["passes_leakage_check"], leakage_passed=False))
        payload = client.get("/api/direction/TEST?include_gauge=false").json()
        assert payload["evaluation"]["leakage_check_passed"] is False

    def test_listing_reports(self, client, report_dir):
        _write(report_dir, _report(ship=False, failed=["x"]))
        listing = client.get("/api/direction/").json()
        assert listing["reports"] == [
            {"symbol": "TEST", "model": "logistic", "generated_at": "2026-01-01T00:00:00",
             "accuracy": 0.56, "ship": False}
        ]

    def test_unknown_model_is_rejected(self, client, report_dir):
        assert client.get("/api/direction/TEST?model=transformer").status_code == 422


class _FakeBars:
    frame = pd.DataFrame()
    meta: dict = {}
