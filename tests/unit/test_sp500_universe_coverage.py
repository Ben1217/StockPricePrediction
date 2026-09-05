"""
Every S&P 500 constituent must be reachable by the Predictions tab.

Written after a full sweep of the index found the tab serving 501 of its 503
constituents plus the index itself, and the shortfall coming from two causes
that had nothing to do with the models:

* **Class shares never resolved.** The index lists BRK.B and BF.B with a dot;
  Yahoo indexes them with a dash and answers the dot form with "possibly
  delisted". Both spellings reach the API — from the constituent list, from a
  saved watchlist, from anyone typing the ticker as it is printed — so both
  have to work.
* **The history gate was the legacy one.** ``/predict/forecast`` runs the
  foundation stack, whose longest covariate window is 20 bars, but it shared a
  260-row floor with the regression bundles that read SMA_200. Three
  constituents with complete, current data sat under that floor — Q (216 bars),
  FDXF (71) and HONA (58), all 2025-26 spinoffs — and answered 422 for a
  forecast Chronos-2 and TimesFM 2.5 could produce.

These tests pin the contracts those fixes rest on. They do not hit the network:
the sweep that found the failures is not something to re-run per commit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.routes.predict import (
    FULL_STACK_HISTORY_ROWS,
    MIN_FORECAST_HISTORY_ROWS,
    MIN_LONG_WINDOW_HISTORY_ROWS,
)
from src.data.ohlcv_cache import safe_yf_download, ticker_variants
from src.models.foundation.features import build_technical_features


# ---------------------------------------------------------------------------
# Class-share spellings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol, expected",
    [
        ("BRK.B", ["BRK.B", "BRK-B"]),
        ("BF.B", ["BF.B", "BF-B"]),
        ("brk.b", ["BRK.B", "BRK-B"]),
        (" BRK.B ", ["BRK.B", "BRK-B"]),
    ],
)
def test_class_share_falls_back_to_the_dash_spelling(symbol, expected):
    assert ticker_variants(symbol) == expected


@pytest.mark.parametrize("symbol", ["AAPL", "^GSPC", "RY.TO", "0700.HK", "BRK-B"])
def test_symbols_that_already_resolve_get_no_second_spelling(symbol):
    """
    A rewrite would break more than it fixes.

    ``RY.TO`` and ``0700.HK`` name a foreign listing's exchange, not a share
    class, and Yahoo indexes them with the dot. Only a single-character suffix
    is ambiguous, and even then the dot form is tried first — so the fallback
    can only ever add a request to a symbol that had already failed.
    """
    assert ticker_variants(symbol) == [symbol.upper().strip()]


def test_single_letter_exchange_suffix_still_tries_the_dot_form_first():
    """
    ``SHEL.L`` is London, not a share class, and is shaped exactly like one.

    It is unresolvable by shape, so the ordering carries the guarantee instead:
    the spelling the caller gave is always attempted before any alternative.
    """
    assert ticker_variants("SHEL.L")[0] == "SHEL.L"


def test_download_retries_the_dash_spelling_when_the_dot_form_is_empty(monkeypatch):
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.DatetimeIndex(["2026-01-02"]),
    )
    attempted: list[str] = []

    class _FakeYF:
        @staticmethod
        def download(ticker, **_kwargs):
            attempted.append(ticker)
            # What Yahoo actually returns for the dot spelling: an empty frame.
            return frame if ticker == "BRK-B" else pd.DataFrame()

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _FakeYF)

    result = safe_yf_download("BRK.B", start="2026-01-01", end="2026-01-03")

    assert attempted == ["BRK.B", "BRK-B"], "the requested spelling must be tried first"
    assert result is not None and not result.empty


def test_download_does_not_retry_when_the_first_spelling_works(monkeypatch):
    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.DatetimeIndex(["2026-01-02"]),
    )
    attempted: list[str] = []

    class _FakeYF:
        @staticmethod
        def download(ticker, **_kwargs):
            attempted.append(ticker)
            return frame

    monkeypatch.setitem(__import__("sys").modules, "yfinance", _FakeYF)

    assert safe_yf_download("SHEL.L", start="2026-01-01", end="2026-01-03") is not None
    assert attempted == ["SHEL.L"], "a working symbol must cost exactly one request"


# ---------------------------------------------------------------------------
# History floors
# ---------------------------------------------------------------------------

def _synthetic_bars(rows: int) -> pd.DataFrame:
    """A frame with the shape and variability the feature layer expects."""
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0, 1.0, rows))
    index = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.2, rows),
            "High": close + np.abs(rng.normal(0.5, 0.2, rows)),
            "Low": close - np.abs(rng.normal(0.5, 0.2, rows)),
            "Close": close,
            "Adj Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, rows),
        },
        index=index,
    )


def test_the_foundation_floor_is_below_the_legacy_one():
    """
    The two model families have genuinely different requirements.

    Sharing one number is what silenced the tab for every recent listing in the
    index: the stricter requirement, which belongs to bundles reading SMA_200,
    governed a path that reads nothing longer than 20 bars.
    """
    assert MIN_FORECAST_HISTORY_ROWS < MIN_LONG_WINDOW_HISTORY_ROWS


def test_the_foundation_floor_admits_the_short_history_constituents():
    """
    Q (216 bars), FDXF (71) and HONA (58) are S&P 500 members with current data.

    All three answered 422 under the 260-row gate. The floor has to sit under
    the shortest of them or the index is not covered.
    """
    for observed_rows in (216, 71, 58):
        assert observed_rows >= MIN_FORECAST_HISTORY_ROWS


def test_the_foundation_floor_yields_a_complete_covariate_row():
    """
    The floor is where every Section 4 covariate resolves on the last bar.

    One bar short of it, OBV_Slope_20 is still NaN — which is not a crash but
    does drop Chronos-2 to its univariate arm, silently making the one member
    documented to read the TA features stop reading them.
    """
    at_floor = build_technical_features(_synthetic_bars(MIN_FORECAST_HISTORY_ROWS))
    assert at_floor.iloc[-1].notna().all()


def test_below_the_floor_the_feature_layer_is_not_yet_complete():
    """
    Guards the floor from drifting down without the covariates being re-checked.

    If this starts failing because a shorter frame now resolves cleanly, the
    floor can move — but that is a measurement, not an assumption.
    """
    just_under = build_technical_features(_synthetic_bars(MIN_FORECAST_HISTORY_ROWS - 5))
    assert just_under.iloc[-1].isna().any()


def test_kronos_context_defines_the_thin_history_threshold():
    """
    Below its 128-bar context Kronos cannot run, so the stack is two members.

    That is served, not refused — but the response flags it, so the threshold
    has to be the model's actual context length rather than a round number.
    """
    assert FULL_STACK_HISTORY_ROWS == 128
    assert MIN_FORECAST_HISTORY_ROWS < FULL_STACK_HISTORY_ROWS


# ---------------------------------------------------------------------------
# The constituent endpoint the stock picker is built from
# ---------------------------------------------------------------------------

def test_sp500_endpoint_serves_company_and_sector(monkeypatch):
    """
    A picker cannot filter by sector or search by company name without them.

    Both used to be dropped in a way that looked like success: the handler
    selected the pre-rename ``Security`` / ``GICS Sector`` columns, the
    resulting KeyError went into a bare ``except: pass``, and the ticker-only
    fallback answered 200 with a plausible count of 503.
    """
    from fastapi.testclient import TestClient

    from src.api.main import app
    import src.data.market_data as market_data

    constituents = pd.DataFrame(
        {
            "Symbol": ["MMM", "BRK-B"],
            "Company": ["3M", "Berkshire Hathaway"],
            "Sector": ["Industrials", "Financials"],
            "Industry": ["Industrial Conglomerates", "Multi-Sector Holdings"],
        }
    )
    monkeypatch.setattr(market_data, "get_sp500_constituents", lambda: constituents)

    body = TestClient(app).get("/api/data/sp500").json()

    assert body["count"] == 2
    assert body["symbols"][0] == {
        "symbol": "MMM",
        "company": "3M",
        "sector": "Industrials",
    }
    assert all(row.get("sector") for row in body["symbols"])


def test_sp500_endpoint_falls_back_to_tickers_when_the_scrape_fails(monkeypatch):
    """A degraded picker still beats an empty one, so this stays a fallback."""
    from fastapi.testclient import TestClient

    from src.api.main import app
    import src.api.routes.data as data_routes
    import src.data.market_data as market_data

    def _boom():
        raise RuntimeError("wikipedia unreachable")

    monkeypatch.setattr(market_data, "get_sp500_constituents", _boom)
    monkeypatch.setattr(data_routes, "get_sp500_tickers", lambda: ["AAPL", "MSFT"])

    body = TestClient(app).get("/api/data/sp500").json()

    assert body["count"] == 2
    assert body["symbols"] == [{"symbol": "AAPL"}, {"symbol": "MSFT"}]


# ---------------------------------------------------------------------------
# The generic price route, which is what validates a typed ticker
# ---------------------------------------------------------------------------

def test_generic_price_fetch_also_resolves_the_class_share_spelling(monkeypatch):
    """
    The search box validates against `/data/prices` before adding a ticker.

    That route reads `fetch_ohlcv`, not the prediction downloader, so without
    the same fallback a stock whose forecast serves perfectly well is rejected
    as nonexistent at the point of adding it — the one place the user meets it.
    """
    from src.data import ohlcv

    frame = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10]},
        index=pd.DatetimeIndex(["2026-01-02"]),
    )
    attempted: list[str] = []

    def _download(ticker, **_kwargs):
        attempted.append(ticker)
        return frame if ticker == "BRK-B" else pd.DataFrame()

    monkeypatch.setattr(ohlcv.yf, "download", _download)
    ohlcv.cache_clear()

    result = ohlcv.fetch_ohlcv("BRK.B", interval="1d", use_cache=False)

    assert not result.empty
    assert attempted[0] == "BRK.B", "the requested spelling is always tried first"
    assert "BRK-B" in attempted


# ---------------------------------------------------------------------------
# The Predictions tab's contract with the models behind it
# ---------------------------------------------------------------------------

def test_quote_lookup_rejects_a_payload_that_carries_no_price():
    """
    An unknown spelling does not come back empty, which is what broke this.

    ``yf.Ticker("BRK.B").get_info()`` returns a truthy 15-key stub — symbol,
    region, priceHint, tradeable and friends — with no quote in it, while
    ``BRK-B`` returns 175 keys including regularMarketPrice. Testing the dict
    for truthiness accepted the stub, so the dot spelling silently fell through
    to its own last close and the same stock reported two different current
    prices depending on how it was spelled.
    """
    from src.api.routes.predict import _has_quote

    stub = {
        "symbol": "BRK.B", "region": "US", "priceHint": 2,
        "tradeable": False, "quoteType": "EQUITY", "maxAge": 1,
    }
    assert _has_quote(stub) is False, "a metadata stub is not a quote"
    assert _has_quote({"regularMarketPrice": 506.03}) is True
    assert _has_quote({"preMarketPrice": 12.5}) is True
    assert _has_quote({"postMarketPrice": 12.5}) is True
    assert _has_quote({}) is False
    # A field present but unusable is still not a quote.
    assert _has_quote({"regularMarketPrice": None}) is False
    assert _has_quote({"regularMarketPrice": 0}) is False


def test_forecast_probability_is_served_and_never_claimed_calibrated():
    """
    The direction is a 0.5 threshold of a probability the payload must carry.

    Without it, 0.51 and 0.94 render as the same arrow and a reader has no way
    to tell a coin flip from a strong call. It is served with
    ``probability_is_calibrated`` hardcoded False, because no walk-forward has
    ever been run over the foundation stack — a client that shows it as a
    confidence would be asserting a reliability nothing has measured.
    """
    from src.api.schemas.schemas import SimpleForecastResponse

    fields = SimpleForecastResponse.model_fields
    assert "probability_up" in fields
    assert "probability_is_calibrated" in fields
    assert fields["probability_is_calibrated"].default is False, (
        "the calibration flag must default to the honest answer, not to True"
    )

    served = SimpleForecastResponse(symbol="AAPL")
    assert served.probability_is_calibrated is False
    assert served.probability_up is None
