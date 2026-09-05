"""
Regressions for the 500s on /api/portfolio/{optimize,frontier,correlation}.

Two independent faults produced them, and both are reproduced here without the
network:

1. ``yfinance.download`` keeps its results in module-level globals that it resets
   on every call, so two concurrent downloads merge into each other. The caller
   for one ticker got a frame carrying a second ticker's columns; flattening it
   left duplicate 'Close' labels, ``df["Close"]`` became a DataFrame, and the
   next ``pd.DataFrame({...})`` raised "Data must be 1-dimensional" -- uncaught,
   so every request in flight 500'd together.

2. FastAPI renders with ``json.dumps(..., allow_nan=False)``. A NaN or Inf
   anywhere in the payload -- a zero-variance symbol makes a whole correlation
   row NaN -- raises after the handler returned, which the browser sees as a
   bare 500.
"""

import json
import math
import threading

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from src.api.routes import portfolio as pf
from src.data import ohlcv_cache as oc


# -- Fixtures ---------------------------------------------------------------

def _bars(n=260, seed=0, start="2024-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {"Open": close, "High": close * 1.01, "Low": close * 0.99,
         "Close": close, "Volume": np.full(n, 1_000_000)},
        index=idx,
    )


# -- 1. Frame normalization (the crash itself) ------------------------------

def test_merged_two_ticker_frame_yields_one_dimensional_close():
    """The exact shape a raced download returns: both tickers in one frame."""
    a, b = _bars(seed=1), _bars(seed=2)
    merged = pd.concat({"AAA": a, "BBB": b}, axis=1)           # (Ticker, Price)
    merged.columns = merged.columns.swaplevel(0, 1)             # -> (Price, Ticker)
    merged = merged.sort_index(axis=1)
    merged.columns.names = ["Price", "Ticker"]

    out = oc.normalize_ohlcv_frame(merged, "BBB")

    assert out is not None
    assert list(out.columns).count("Close") == 1
    assert isinstance(out["Close"], pd.Series), "Close must be 1-D"
    # It must be BBB's data, not whichever column happened to sort first.
    pd.testing.assert_series_equal(
        out["Close"].astype(float), b["Close"].astype(float), check_names=False
    )


def test_group_by_ticker_orientation_is_handled():
    frame = pd.concat({"AAA": _bars(seed=3)}, axis=1)
    frame.columns.names = ["Ticker", "Price"]
    out = oc.normalize_ohlcv_frame(frame, "AAA")
    assert out is not None and "Close" in out.columns
    assert isinstance(out["Close"], pd.Series)


def test_duplicate_timestamps_are_collapsed():
    """Duplicate index labels made every later alignment raise ValueError."""
    df = _bars(n=10)
    df = pd.concat([df, df.iloc[[-1]]])
    out = oc.normalize_ohlcv_frame(df, "AAA")
    assert out is not None
    assert not out.index.has_duplicates
    pd.DataFrame({"AAA": out["Close"], "BBB": out["Close"]})  # must not raise


def test_tz_aware_index_is_normalized():
    df = _bars(n=10)
    df.index = df.index.tz_localize("US/Eastern")
    out = oc.normalize_ohlcv_frame(df, "AAA")
    assert out is not None and out.index.tz is None


def test_flat_frame_passes_through_unchanged():
    df = _bars(n=20)
    out = oc.normalize_ohlcv_frame(df, "AAA")
    assert out is not None
    pd.testing.assert_series_equal(out["Close"], df["Close"])


@pytest.mark.parametrize("value", [None, pd.DataFrame()])
def test_empty_input_returns_none(value):
    assert oc.normalize_ohlcv_frame(value, "AAA") is None


def test_frame_without_close_returns_none():
    assert oc.normalize_ohlcv_frame(pd.DataFrame({"Volume": [1, 2]}), "AAA") is None


# -- 2. The download lock is shared process-wide ----------------------------

def test_every_module_shares_one_download_lock():
    """Two separate locks would each serialise their own callers and still race."""
    from src.data import data_loader, direction_data, market_data, ohlcv
    for mod in (data_loader, direction_data, market_data):
        assert mod.YF_DOWNLOAD_LOCK is oc.YF_DOWNLOAD_LOCK, mod.__name__
    assert ohlcv.download_lock is oc.YF_DOWNLOAD_LOCK


def test_safe_yf_download_serialises_concurrent_callers(monkeypatch):
    overlaps, active, guard = [], [], threading.Lock()

    def fake_download(ticker, **kwargs):
        with guard:
            active.append(ticker)
            overlaps.append(len(active))
        try:
            return _bars(n=30)
        finally:
            with guard:
                active.remove(ticker)

    import yfinance
    monkeypatch.setattr(yfinance, "download", fake_download)

    threads = [threading.Thread(target=oc.safe_yf_download, args=(f"T{i}",))
               for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert overlaps and max(overlaps) == 1, f"downloads overlapped: {overlaps}"


# -- 3. _fetch_returns survives hostile frames ------------------------------

def _patch_fetch(monkeypatch, frames_by_symbol):
    def fake_cached_download(sym, start, end, interval, downloader, **kw):
        return frames_by_symbol.get(sym)
    monkeypatch.setattr(pf, "cached_download", fake_cached_download)


def test_fetch_returns_handles_merged_frame(monkeypatch):
    """Before the fix this raised ValueError: Data must be 1-dimensional -> 500."""
    a, b = _bars(seed=4), _bars(seed=5)
    merged = pd.concat({"AAA": a, "BBB": b}, axis=1)
    merged.columns = merged.columns.swaplevel(0, 1).set_names(["Price", "Ticker"])
    _patch_fetch(monkeypatch, {"AAA": merged, "BBB": b})

    returns, prices = pf._fetch_returns(["AAA", "BBB"], 252)
    assert list(returns.columns) == ["AAA", "BBB"]
    assert np.isfinite(returns.values).all()


def test_fetch_returns_drops_zero_price_inf(monkeypatch):
    """A zero tick makes pct_change return inf, which cannot be rendered."""
    a = _bars(seed=6)
    a.iloc[5, a.columns.get_loc("Close")] = 0.0
    _patch_fetch(monkeypatch, {"AAA": a, "BBB": _bars(seed=7)})

    returns, _ = pf._fetch_returns(["AAA", "BBB"], 252)
    assert np.isfinite(returns.values).all()


def test_fetch_returns_deduplicates_and_uppercases(monkeypatch):
    _patch_fetch(monkeypatch, {"AAA": _bars(seed=8)})
    returns, _ = pf._fetch_returns([" aaa ", "AAA", ""], 252)
    assert list(returns.columns) == ["AAA"]


def test_fetch_returns_rejects_empty_symbol_list():
    with pytest.raises(HTTPException) as exc:
        pf._fetch_returns([], 252)
    assert exc.value.status_code == 400


def test_fetch_returns_404s_when_every_symbol_fails(monkeypatch):
    _patch_fetch(monkeypatch, {})
    with pytest.raises(HTTPException) as exc:
        pf._fetch_returns(["AAA", "BBB"], 252)
    assert exc.value.status_code == 404


# -- 4. Nothing non-finite can reach the renderer ---------------------------

def _renderable(payload):
    """Mirror what FastAPI's JSONResponse does to the handler's return value."""
    json.dumps(payload, allow_nan=False)
    return True


def test_json_safe_neutralises_nan_and_inf():
    payload = pf._json_safe({
        "a": float("nan"),
        "b": [float("inf"), -float("inf"), 1.5],
        "c": {"d": np.float64("nan"), "e": np.int64(3), "f": np.bool_(True)},
        "g": np.array([1.0, float("nan")]),
    })
    assert _renderable(payload)
    assert payload["a"] is None
    assert payload["b"] == [None, None, 1.5]
    assert payload["c"] == {"d": None, "e": 3, "f": True}
    assert payload["g"] == [1.0, None]


@pytest.mark.parametrize("value,expected", [
    (float("nan"), 0.0), (float("inf"), 0.0), (-float("inf"), 0.0),
    ("nope", 0.0), (None, 0.0), (2.5, 2.5), (np.float64(3.5), 3.5),
])
def test_finite_coerces_unrenderable_scalars(value, expected):
    assert pf._finite(value) == expected


def test_correlation_of_a_flat_symbol_is_renderable(monkeypatch):
    """A halted symbol has zero variance, so its whole correlation row is NaN."""
    flat = _bars(seed=9)
    flat["Close"] = 100.0                      # never moves
    _patch_fetch(monkeypatch, {"AAA": _bars(seed=10), "FLAT": flat})

    payload = pf.get_correlation(symbols="AAA,FLAT", lookback_days=90)
    assert _renderable(payload)
    assert payload["matrix"]["FLAT"]["AAA"] is None


def test_optimize_and_frontier_payloads_are_renderable(monkeypatch):
    _patch_fetch(monkeypatch,
                 {s: _bars(seed=i) for i, s in enumerate(["AAA", "BBB", "CCC"])})
    monkeypatch.setattr(pf, "save_weights", lambda *a, **k: None)

    from src.api.schemas.schemas import PortfolioOptimizeRequest
    req = PortfolioOptimizeRequest(symbols=["AAA", "BBB", "CCC"], lookback_days=252)

    assert _renderable(json.loads(pf.optimize(req).model_dump_json()))
    assert _renderable(json.loads(pf.efficient_frontier(req).model_dump_json()))


def test_frontier_drops_non_finite_points(monkeypatch):
    """A degenerate solve yields NaN volatility; such a point cannot be plotted."""
    _patch_fetch(monkeypatch, {s: _bars(seed=i) for i, s in enumerate(["AAA", "BBB"])})
    monkeypatch.setattr(
        pf, "calculate_efficient_frontier",
        lambda returns, n_points=50: (
            np.array([0.1, float("nan"), 0.2]),
            np.array([0.05, 0.06, float("inf")]),
            [{"AAA": 0.5, "BBB": 0.5}] * 3,
        ),
    )
    from src.api.schemas.schemas import PortfolioOptimizeRequest
    result = pf.efficient_frontier(PortfolioOptimizeRequest(symbols=["AAA", "BBB"]))
    assert len(result.points) == 1
    assert all(math.isfinite(p["volatility"]) for p in result.points)


# -- 5. Cache namespaces stay separated by download variant -----------------

def test_cache_key_separates_adjusted_from_raw_bars():
    """Same ticker and window, different auto_adjust: must not share an entry."""
    key = oc.OHLCVCache._key
    assert key("AAPL", "2024-01-01", "2024-06-01", "1d-adj") != \
        key("AAPL", "2024-01-01", "2024-06-01", "1d-raw")
    assert key("AAPL", "2024-01-01", "2024-06-01", "1d-raw-actions") != \
        key("AAPL", "2024-01-01", "2024-06-01", "1d-raw")
