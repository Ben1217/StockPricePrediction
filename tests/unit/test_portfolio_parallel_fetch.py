"""
Tests for the concurrent per-symbol price fetch in the portfolio routes.

`_fetch_returns` used to download one symbol at a time. It now fans out across a
thread pool, and the invariants that has to preserve are ordering (the caller's
symbol order is the column order of every downstream weight and correlation cell)
and failure handling (a symbol with no data is dropped, not raised).
"""

import threading

import numpy as np
import pandas as pd
import pytest

from fastapi import HTTPException

import src.api.routes.portfolio as portfolio_route


@pytest.fixture
def fake_downloads(monkeypatch):
    """Serve deterministic bars for any symbol except NODATA."""
    index = pd.date_range("2024-01-01", periods=200, freq="B")
    seen = []
    lock = threading.Lock()

    def _cached_download(symbol, start, end, tag, downloader):
        with lock:
            seen.append(symbol)
        if symbol == "NODATA":
            return None
        rng = np.random.default_rng(abs(hash(symbol)) % (2**32))
        return pd.DataFrame(
            {"Close": 100 + np.cumsum(rng.normal(0, 1, len(index)))}, index=index
        )

    monkeypatch.setattr(portfolio_route, "cached_download", _cached_download)
    monkeypatch.setattr(portfolio_route, "normalize_ohlcv_frame", lambda df, s: df)
    return seen


def test_columns_follow_the_requested_order_not_completion_order(fake_downloads):
    order = ["ZZZ", "AAA", "MMM", "QQQ", "BBB"]
    returns, prices = portfolio_route._fetch_returns(order, 120)

    assert list(returns.columns) == order
    assert list(prices.columns) == order


def test_a_symbol_with_no_data_is_dropped_rather_than_raised(fake_downloads):
    returns, _ = portfolio_route._fetch_returns(["AAA", "NODATA", "BBB"], 120)
    assert list(returns.columns) == ["AAA", "BBB"]


def test_symbols_are_deduplicated_and_upper_cased(fake_downloads):
    returns, _ = portfolio_route._fetch_returns(["aaa", "AAA", " bbb ", "aaa"], 120)
    assert list(returns.columns) == ["AAA", "BBB"]
    # Deduplicated before the fan-out, so a repeated ticker is one download.
    assert fake_downloads.count("AAA") == 1


def test_every_symbol_is_fetched_exactly_once(fake_downloads):
    symbols = [f"S{i}" for i in range(20)]
    portfolio_route._fetch_returns(symbols, 120)
    assert sorted(fake_downloads) == sorted(symbols)


def test_no_usable_symbol_is_a_404(fake_downloads):
    with pytest.raises(HTTPException) as excinfo:
        portfolio_route._fetch_returns(["NODATA"], 120)
    assert excinfo.value.status_code == 404


def test_an_empty_request_is_a_400(fake_downloads):
    with pytest.raises(HTTPException) as excinfo:
        portfolio_route._fetch_returns(["", "   "], 120)
    assert excinfo.value.status_code == 400


def test_a_provider_exception_drops_that_symbol_only(monkeypatch, fake_downloads):
    original = portfolio_route.cached_download

    def _explode(symbol, start, end, tag, downloader):
        if symbol == "BOOM":
            raise RuntimeError("provider fell over")
        return original(symbol, start, end, tag, downloader)

    monkeypatch.setattr(portfolio_route, "cached_download", _explode)
    returns, _ = portfolio_route._fetch_returns(["AAA", "BOOM", "BBB"], 120)
    assert list(returns.columns) == ["AAA", "BBB"]


def test_non_positive_prices_are_excluded(monkeypatch, fake_downloads):
    index = pd.date_range("2024-01-01", periods=200, freq="B")

    def _with_a_zero(symbol, start, end, tag, downloader):
        close = np.full(len(index), 100.0)
        if symbol == "ZEROED":
            close[:] = 0.0
        return pd.DataFrame({"Close": close}, index=index)

    monkeypatch.setattr(portfolio_route, "cached_download", _with_a_zero)
    # A zero price makes pct_change return inf, which FastAPI cannot render.
    returns, _ = portfolio_route._fetch_returns(["AAA", "ZEROED"], 120)
    assert list(returns.columns) == ["AAA"]
