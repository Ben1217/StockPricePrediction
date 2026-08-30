"""
Tests for the on-disk OHLCV cache.

The cache exists so a retrain (12 bundles for one symbol) issues one download
instead of twelve, which is what was earning the 429s.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.ohlcv_cache import OHLCVCache, cached_download


@pytest.fixture
def frame():
    idx = pd.bdate_range("2024-01-01", periods=50)
    return pd.DataFrame(
        {"Open": np.arange(50.0), "High": np.arange(50.0) + 1,
         "Low": np.arange(50.0) - 1, "Close": np.arange(50.0),
         "Adj Close": np.arange(50.0), "Volume": np.arange(50)},
        index=idx,
    )


@pytest.fixture
def cache(tmp_path):
    return OHLCVCache(cache_dir=tmp_path / "ohlcv", ttl_seconds=3600)


def test_store_then_load_round_trips(cache, frame):
    cache.store("AAPL", "2024-01-01", "2024-03-01", "1d", frame)
    loaded = cache.load("AAPL", "2024-01-01", "2024-03-01", "1d")
    assert loaded is not None
    # Parquet does not carry the index `freq` attribute; the bars are what matter.
    pd.testing.assert_frame_equal(loaded, frame, check_freq=False)


def test_load_misses_on_a_different_range(cache, frame):
    cache.store("AAPL", "2024-01-01", "2024-03-01", "1d", frame)
    assert cache.load("AAPL", "2024-01-01", "2024-04-01", "1d") is None


def test_load_misses_on_a_different_symbol(cache, frame):
    cache.store("AAPL", "2024-01-01", "2024-03-01", "1d", frame)
    assert cache.load("MSFT", "2024-01-01", "2024-03-01", "1d") is None


def test_expired_entry_is_not_served(tmp_path, frame):
    cache = OHLCVCache(cache_dir=tmp_path / "ohlcv", ttl_seconds=0)
    cache.store("AAPL", "2024-01-01", "2024-03-01", "1d", frame)
    assert cache.load("AAPL", "2024-01-01", "2024-03-01", "1d") is None


def test_cached_download_calls_the_network_once(tmp_path, frame, monkeypatch):
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache",
                        lambda: OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=3600))
    calls = {"n": 0}

    def downloader():
        calls["n"] += 1
        return frame

    for _ in range(5):
        result = cached_download("AAPL", "2024-01-01", "2024-03-01", "1d", downloader)
        assert result is not None
    assert calls["n"] == 1, "repeat asks must be served from disk"


def test_cached_download_retries_then_serves_stale_on_failure(tmp_path, frame, monkeypatch):
    """A failing provider must not lose a run when usable bars are already on disk."""
    shared = OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=0)  # always expired
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache", lambda: shared)
    monkeypatch.setattr("src.data.ohlcv_cache.time.sleep", lambda _s: None)
    shared.store("AAPL", "2024-01-01", "2024-03-01", "1d", frame)

    def failing():
        raise RuntimeError("429 Too Many Requests")

    result = cached_download("AAPL", "2024-01-01", "2024-03-01", "1d", failing, max_retries=2)
    assert result is not None
    pd.testing.assert_frame_equal(result, frame, check_freq=False)


def test_cached_download_returns_none_when_no_cache_and_download_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache",
                        lambda: OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=3600))
    monkeypatch.setattr("src.data.ohlcv_cache.time.sleep", lambda _s: None)

    def failing():
        raise RuntimeError("429 Too Many Requests")

    assert cached_download("AAPL", "2024-01-01", "2024-03-01", "1d", failing, max_retries=2) is None
