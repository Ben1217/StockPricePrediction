"""
The portfolio response fields the dashboard reads, pinned.

The Portfolio and Optimization tabs render these payloads by key —
``metrics.annual_volatility``, ``correlation.avg_correlation``,
``alerts[].message`` and so on. There is no JavaScript test runner in this
project, so nothing on the frontend would fail if one of those keys were
renamed: the tab would quietly draw an em dash where a number belongs, and the
comparison that the whole page is built around would show two blank columns.

These tests are that missing alarm. They assert the *contract* — the keys and
their meaning — rather than the values, which depend on market data.

The most important one is
:func:`test_metrics_honours_the_weights_parameter`. The redesigned Portfolio tab
scores your current split and the optimizer's split through the same endpoint
with weights as the only difference; if that parameter were ever ignored, both
columns would agree exactly and the page would look like it worked while
comparing a thing against itself.

Every test is hermetic: price downloads and the weight-snapshot database are
both patched out, so nothing here touches the network or writes to disk.
"""

import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import portfolio as portfolio_route

# TWIN_A and TWIN_B are built to move together so that `high_corr_pairs` is
# non-empty and its shape gets asserted; SOLO is independent of both.
TICKERS = ["TWIN_A", "TWIN_B", "SOLO"]


@pytest.fixture
def frames():
    """Deterministic daily returns: two near-identical series and one independent."""
    rng = np.random.default_rng(20260905)
    index = pd.date_range("2024-01-01", periods=300, freq="B")

    shared = rng.normal(0.0006, 0.011, len(index))
    returns = pd.DataFrame(
        {
            "TWIN_A": shared + rng.normal(0, 0.001, len(index)),
            "TWIN_B": shared + rng.normal(0, 0.001, len(index)),
            "SOLO": rng.normal(0.0004, 0.017, len(index)),
        },
        index=index,
    )
    prices = (1.0 + returns).cumprod() * 100.0
    return returns, prices


@pytest.fixture
def client(monkeypatch, frames):
    monkeypatch.setattr(portfolio_route, "_fetch_returns", lambda symbols, lookback: frames)
    # Optimizing normally appends a row to the weight-snapshot database. A
    # contract test has no business writing one.
    monkeypatch.setattr(portfolio_route, "save_weights", lambda *args, **kwargs: None)

    app = FastAPI()
    app.include_router(portfolio_route.router, prefix="/api/portfolio")
    return TestClient(app)


def _weights(**overrides):
    base = {"TWIN_A": 0.34, "TWIN_B": 0.33, "SOLO": 0.33}
    base.update(overrides)
    return base


# ── /optimize ────────────────────────────────────────────────────────────────

def test_optimize_returns_the_fields_the_weight_bars_read(client):
    response = client.post("/api/portfolio/optimize", json={"symbols": TICKERS, "method": "max_sharpe"})
    assert response.status_code == 200

    body = response.json()
    for key in ("weights", "method", "expected_return", "volatility", "sharpe_ratio", "metrics"):
        assert key in body, f"optimize response lost '{key}'"

    assert set(body["weights"]) <= set(TICKERS)
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=0.02)
    # The tab labels the run by what the server actually solved, not by the
    # dropdown, so the echoed method has to be the real one.
    assert body["method"] == "max_sharpe"


# ── /metrics ─────────────────────────────────────────────────────────────────

def test_metrics_exposes_every_row_of_the_comparison_table(client):
    response = client.get(
        "/api/portfolio/metrics",
        params={"symbols": ",".join(TICKERS), "lookback": 252, "weights": json.dumps(_weights())},
    )
    assert response.status_code == 200

    metrics = response.json()["metrics"]
    # One assertion per row the Portfolio tab draws.
    for key in ("annual_return", "annual_volatility", "sharpe_ratio", "max_drawdown"):
        assert key in metrics, f"comparison row '{key}' is missing from /metrics"
        assert isinstance(metrics[key], (int, float))

    assert metrics["annual_volatility"] > 0
    # Drawdown is reported as a loss. The tab colours it "higher is better" on
    # that basis, so a sign flip here would invert the verdict.
    assert metrics["max_drawdown"] <= 0


def test_metrics_honours_the_weights_parameter(client):
    """
    Two different splits must score differently.

    This is the load-bearing assumption of the redesigned Portfolio tab. If the
    weights were ignored, "your split" and "the optimizer's split" would print
    identical numbers and the page would look correct while telling the reader
    nothing.
    """
    concentrated = _weights(TWIN_A=0.90, TWIN_B=0.05, SOLO=0.05)
    balanced = _weights()

    def score(weights):
        response = client.get(
            "/api/portfolio/metrics",
            params={"symbols": ",".join(TICKERS), "lookback": 252, "weights": json.dumps(weights)},
        )
        assert response.status_code == 200
        return response.json()["metrics"]

    assert score(concentrated)["annual_volatility"] != score(balanced)["annual_volatility"]


def test_metrics_attribution_breaks_down_by_stock(client):
    response = client.get(
        "/api/portfolio/metrics",
        params={
            "symbols": ",".join(TICKERS),
            "lookback": 252,
            "weights": json.dumps(_weights()),
            "include_attribution": "true",
        },
    )
    assert response.status_code == 200

    by_stock = response.json()["attribution"]["by_stock"]
    assert set(by_stock) == set(TICKERS)
    assert "contribution_to_portfolio" in by_stock["SOLO"]


# ── /correlation ─────────────────────────────────────────────────────────────

def test_correlation_returns_matrix_tickers_and_flagged_pairs(client):
    response = client.get("/api/portfolio/correlation", params={"symbols": ",".join(TICKERS)})
    assert response.status_code == 200

    body = response.json()
    for key in ("matrix", "tickers", "avg_correlation", "high_corr_pairs"):
        assert key in body, f"correlation response lost '{key}'"

    assert set(body["tickers"]) == set(TICKERS)
    # A symbol against itself is 1.0; the tab dims that diagonal.
    assert body["matrix"]["SOLO"]["SOLO"] == pytest.approx(1.0)

    # The twins were constructed to move together, so the flagged-pair shape
    # gets exercised rather than assumed.
    assert body["high_corr_pairs"], "expected the near-identical series to be flagged"
    pair = body["high_corr_pairs"][0]
    # ticker_a / ticker_b, NOT ticker1 / ticker2. Both tabs name these fields
    # directly, and the first version of this file guessed the numeric spelling
    # because a live call happened to return an empty list — which is precisely
    # the mistake an empty fixture lets through and a populated one catches.
    for key in ("ticker_a", "ticker_b", "correlation", "warning"):
        assert key in pair, f"high_corr_pairs entry lost '{key}'"
    assert {pair["ticker_a"], pair["ticker_b"]} == {"TWIN_A", "TWIN_B"}


# ── /alerts ──────────────────────────────────────────────────────────────────

def test_alerts_carry_a_type_severity_and_readable_message(client):
    # A 90% position is over any sane concentration limit, so this reliably
    # produces at least one alert to inspect.
    response = client.get(
        "/api/portfolio/alerts",
        params={
            "symbols": ",".join(TICKERS),
            "weights": json.dumps(_weights(TWIN_A=0.90, TWIN_B=0.05, SOLO=0.05)),
            "lookback_days": 90,
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["alert_count"] == len(body["alerts"])
    assert isinstance(body["critical_count"], int)
    assert body["alerts"], "a 90% single position should raise at least one alert"

    alert = body["alerts"][0]
    for key in ("alert_type", "severity", "message"):
        assert key in alert, f"alert lost '{key}'"
    # The tab prints the message verbatim, so it must be a sentence and not a code.
    assert isinstance(alert["message"], str) and alert["message"].strip()
    assert alert["severity"] in {"INFO", "WARNING", "CRITICAL"}


# ── /frontier ────────────────────────────────────────────────────────────────

def test_frontier_points_carry_the_scatter_axes(client):
    response = client.post("/api/portfolio/frontier", json={"symbols": TICKERS, "method": "max_sharpe"})
    assert response.status_code == 200

    body = response.json()
    assert body["points"], "frontier returned no points to plot"
    point = body["points"][0]
    # x, y and the value the optimal marker is chosen by.
    for key in ("volatility", "return", "sharpe"):
        assert key in point, f"frontier point lost '{key}'"

    optimal = body["optimal_portfolio"]
    assert "volatility" in optimal and "return" in optimal
