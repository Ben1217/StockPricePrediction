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

    # The route now fetches through the shared cached fetcher rather than yfinance directly.
    monkeypatch.setattr("src.api.routes.patterns.fetch_ohlcv", fake_download)
    monkeypatch.setattr("src.features.support_resistance.detect_support_resistance", fake_detect)

    client = TestClient(app)
    response = client.get("/api/patterns/support-resistance/META?interval=1mo&lookback=5600")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "META"
    assert len(payload["levels"]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Support/resistance lookback is a BAR COUNT
#
# It was applied as calendar days while its callers passed bars. On the long
# intervals that is not a rounding difference: `lookback=240` on `1mo` fetched
# 440 days, which is fifteen candles, and `detect_support_resistance` reads the
# last hundred. Every monthly panel came back empty, for a reason the response
# never mentioned.
# ─────────────────────────────────────────────────────────────────────────────
def _captured_window(monkeypatch, interval, lookback, rows=400, freq="ME"):
    """The download window the S&R route asks for, in days."""
    from datetime import datetime

    sample_df = _sample_price_df(rows=rows, freq=freq)
    seen = {}

    def fake_download(symbol, iv, start=None, end=None, **kwargs):
        seen["start"], seen["end"] = start, end
        return sample_df.copy()

    monkeypatch.setattr("src.api.routes.patterns.fetch_ohlcv", fake_download)
    client = TestClient(app)
    response = client.get(
        f"/api/patterns/support-resistance/META?interval={interval}&lookback={lookback}"
    )
    assert response.status_code == 200
    span = datetime.strptime(seen["end"], "%Y-%m-%d") - datetime.strptime(seen["start"], "%Y-%m-%d")
    return span.days, response.json()


def test_monthly_lookback_buys_months_of_candles_not_months_of_calendar(monkeypatch):
    """240 monthly bars is twenty years of window, not 240 days of one."""
    days, _ = _captured_window(monkeypatch, "1mo", 240)
    # Twenty years, with the safety factor and margin on top. The old reading
    # asked for 440 days here and got fifteen candles for a hundred-bar read.
    assert days > 240 * 28, f"a 240-month lookback fetched only {days} days"


def test_weekly_lookback_buys_weeks_of_candles(monkeypatch):
    days, _ = _captured_window(monkeypatch, "1wk", 260, freq="W")
    assert days > 260 * 6.5, f"a 260-week lookback fetched only {days} days"


def test_the_window_covers_the_bars_it_promises_on_every_interval(monkeypatch):
    """
    Enough calendar to hold the bar count asked for, on all three.

    The conversion is an average with slack on top; this is the property that
    slack exists for, and it is checked per interval because the daily factor
    (365/252, for weekends) is the one that is easy to get wrong.
    """
    from src.api.routes.patterns import _sr_window_days

    for interval, per_bar in (("1d", 365 / 252), ("1wk", 7.0), ("1mo", 30.5)):
        for bars in (120, 200, 240):
            assert _sr_window_days(interval, bars) >= bars * per_bar


def test_the_response_reports_what_was_read_not_what_was_asked_for(monkeypatch):
    """
    `bars_analysed` is the detector's own 100-bar window.

    The panel captions itself from this, so it has to be the read rather than
    the download — a monthly panel claiming twenty years for an eight-year
    analysis is the same class of mistake as the lookback bug itself.
    """
    from src.api.routes.patterns import SR_ANALYSIS_BARS

    _, payload = _captured_window(monkeypatch, "1mo", 240, rows=400)
    assert payload["interval"] == "1mo"
    assert payload["lookback_bars"] == 240
    assert payload["bars_available"] == 400
    assert payload["bars_analysed"] == SR_ANALYSIS_BARS


def test_a_short_history_reports_the_bars_it_actually_had(monkeypatch):
    """`bars_analysed` never claims more candles than the frame held."""
    _, payload = _captured_window(monkeypatch, "1mo", 240, rows=42)
    assert payload["bars_available"] == 42
    assert payload["bars_analysed"] == 42


def test_a_clamped_lookback_still_fills_the_detectors_window(monkeypatch):
    """
    Every interval's floor is above the 100 bars the detector reads.

    The floor is what a too-small request lands on, so if it sat below the
    analysis window a clamp would quietly hand over a thinner read than the
    algorithm was designed for.
    """
    from src.api.limits import limits_for
    from src.api.routes.patterns import SR_ANALYSIS_BARS

    for interval in ("1d", "1wk", "1mo"):
        lower, _ = limits_for(interval, "lookback")
        assert lower >= SR_ANALYSIS_BARS, interval
