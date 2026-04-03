"""
Regression tests for interval-aware request clamping.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.main import app


def _sample_price_df(rows=240, freq="ME"):
    np.random.seed(21)
    dates = pd.date_range("2006-01-31", periods=rows, freq=freq)
    close = np.linspace(100, 180, rows) + np.sin(np.linspace(0, 8, rows))
    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": np.linspace(1_500_000, 4_500_000, rows),
        },
        index=dates,
    )


def test_prices_route_clamps_small_daily_day_requests(monkeypatch):
    sample_df = _sample_price_df(rows=120, freq="D")
    captured = {}

    def fake_fetch(symbol, start, end, interval="1d"):
        captured["interval"] = interval
        captured["start"] = start
        captured["end"] = end
        return sample_df.copy()

    monkeypatch.setattr("src.api.routes.data._fetch_yfinance", fake_fetch)

    client = TestClient(app)
    response = client.get("/api/data/prices/AMZN?source=yfinance&days=5&interval=1d")

    assert response.status_code == 200
    assert response.json()["symbol"] == "AMZN"
    assert captured["interval"] == "1d"


def test_indicators_route_clamps_large_monthly_requests_and_serializes_cleanly(monkeypatch):
    sample_df = _sample_price_df(rows=240, freq="ME")

    def fake_fetch(symbol, start, end, interval="1mo"):
        return sample_df.copy()

    monkeypatch.setattr("src.api.routes.data._fetch_yfinance", fake_fetch)

    client = TestClient(app)
    response = client.get("/api/data/indicators/META?days=7000&interval=1mo")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "META"
    assert payload["count"] <= 180
    assert payload["count"] > 0
    assert "ATR" in payload["indicators"]


def test_support_resistance_route_clamps_large_lookback(monkeypatch):
    sample_df = _sample_price_df(rows=180, freq="W")

    def fake_download(*args, **kwargs):
        return sample_df.copy()

    def fake_detect(df, current_price):
        return {
            "levels": [
                {"price": float(current_price - 5), "type": "support", "strength": "strong", "confirmations": 3, "sources": ["pivot"], "zone_low": float(current_price - 6), "zone_high": float(current_price - 4)},
                {"price": float(current_price + 5), "type": "resistance", "strength": "strong", "confirmations": 3, "sources": ["pivot"], "zone_low": float(current_price + 4), "zone_high": float(current_price + 6)},
            ],
            "trendlines": [],
            "dynamic_levels": [],
        }

    monkeypatch.setattr("src.api.routes.patterns.yf.download", fake_download)
    monkeypatch.setattr("src.features.support_resistance.detect_support_resistance", fake_detect)

    client = TestClient(app)
    response = client.get("/api/patterns/support-resistance/META?interval=1mo&lookback=5600")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "META"
    assert len(payload["levels"]) == 2
