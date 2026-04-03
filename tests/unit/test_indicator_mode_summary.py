"""
Unit tests for indicator-only summary payloads.
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
from src.signals.sentiment.indicator_sentiment import compute_indicator_sentiment


def _sample_ohlcv(rows=260):
    np.random.seed(7)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100, 140, rows) + np.sin(np.linspace(0, 12, rows))
    return pd.DataFrame(
        {
            "Open": close - 0.6,
            "High": close + 1.4,
            "Low": close - 1.2,
            "Close": close,
            "Volume": np.linspace(1_000_000, 2_200_000, rows),
        },
        index=dates,
    )


def test_compute_indicator_sentiment_includes_atr_details():
    result = compute_indicator_sentiment(_sample_ohlcv())

    assert "details" in result
    assert result["details"]["atr"] is not None
    assert result["details"]["atr"] > 0
    assert result["entry_signal"] in {"BULLISH", "BEARISH", "WAIT"}


def test_sentiment_route_returns_indicator_only_summary_fields(monkeypatch):
    sample_df = _sample_ohlcv()

    def fake_download(*args, **kwargs):
        return sample_df.copy()

    monkeypatch.setattr("src.api.routes.sentiment.yf.download", fake_download)

    client = TestClient(app)
    response = client.get("/api/sentiment/META?days=420&interval=1d")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "INDICATOR"
    assert payload["timeframe"] == "1d"
    assert payload["symbol"] == "META"
    assert payload["action"] in {"BUY", "SELL", "WAIT"}
    assert payload["rsi"] is not None
    assert payload["volume"] is not None
    assert payload["atr"] is not None
    assert "support" in payload
    assert "resistance" in payload
