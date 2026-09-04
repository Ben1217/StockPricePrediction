"""
Tests for the on-disk OHLCV cache.

The cache exists so a retrain (12 bundles for one symbol) issues one download
instead of twelve, which is what was earning the 429s.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.ohlcv_cache import (
    PARTIAL_TTL_SECONDS,
    OHLCVCache,
    cached_download,
    is_partial_frame,
)


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


# ---------------------------------------------------------------------------
# Truncated responses
#
# Yahoo answered a five-year ask for NVDA with its last 153 sessions. That frame
# was cached like any other and served for the full six-hour TTL, so every
# prediction in that window failed the route's 260-row minimum with a 422.
# ---------------------------------------------------------------------------

def _frame(periods: int, end: str = "2026-09-04"):
    idx = pd.bdate_range(end=end, periods=periods)
    return pd.DataFrame(
        {"Open": np.arange(float(periods)), "High": np.arange(float(periods)) + 1,
         "Low": np.arange(float(periods)) - 1, "Close": np.arange(float(periods)),
         "Adj Close": np.arange(float(periods)), "Volume": np.arange(periods)},
        index=idx,
    )


def test_is_partial_frame_flags_a_truncated_window():
    assert is_partial_frame(_frame(153), "2021-09-05", "2026-09-04")


def test_is_partial_frame_accepts_a_full_window():
    assert not is_partial_frame(_frame(1254), "2021-09-05", "2026-09-04")


def test_is_partial_frame_accepts_a_late_listing():
    """PLTR listed midway through a ten-year ask; ~59% coverage is the real extent."""
    assert not is_partial_frame(_frame(1490, end="2026-09-05"), "2016-09-07", "2026-09-05")


def test_truncated_response_is_retried_and_the_full_frame_wins(tmp_path, monkeypatch):
    shared = OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=3600)
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache", lambda: shared)
    monkeypatch.setattr("src.data.ohlcv_cache.time.sleep", lambda _s: None)
    frames = iter([_frame(153), _frame(1254)])

    result = cached_download("NVDA", "2021-09-05", "2026-09-04", "1d", lambda: next(frames))

    assert result is not None and len(result) == 1254
    cached = shared.load("NVDA", "2021-09-05", "2026-09-04", "1d")
    assert cached is not None and len(cached) == 1254, "the truncation must not reach disk"


def test_persistently_short_history_is_served_but_expires_quickly(tmp_path, monkeypatch):
    shared = OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=6 * 3600)
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache", lambda: shared)
    monkeypatch.setattr("src.data.ohlcv_cache.time.sleep", lambda _s: None)

    result = cached_download("NEW", "2021-09-05", "2026-09-04", "1d", lambda: _frame(153))

    assert result is not None and len(result) == 153, "short bars still beat no bars"
    assert shared.load("NEW", "2021-09-05", "2026-09-04", "1d") is not None
    # Older than the partial TTL but far inside the normal one: must not be served.
    meta = shared.cache_dir / f"{shared._key('NEW', '2021-09-05', '2026-09-04', '1d')}.meta.json"
    payload = json.loads(meta.read_text())
    assert payload["partial"] is True
    payload["written_at_epoch"] -= PARTIAL_TTL_SECONDS + 60
    meta.write_text(json.dumps(payload))
    assert shared.load("NEW", "2021-09-05", "2026-09-04", "1d") is None


def test_a_truncated_response_does_not_evict_a_fuller_cached_frame(tmp_path, monkeypatch):
    """One bad response must not cost us five years of bars we already had."""
    shared = OHLCVCache(cache_dir=tmp_path / "c", ttl_seconds=0)  # always expired
    monkeypatch.setattr("src.data.ohlcv_cache.get_ohlcv_cache", lambda: shared)
    monkeypatch.setattr("src.data.ohlcv_cache.time.sleep", lambda _s: None)
    shared.store("NVDA", "2021-09-05", "2026-09-04", "1d", _frame(1254))

    result = cached_download("NVDA", "2021-09-05", "2026-09-04", "1d", lambda: _frame(153))

    assert result is not None and len(result) == 1254
