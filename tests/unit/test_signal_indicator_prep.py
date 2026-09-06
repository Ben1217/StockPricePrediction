"""
Tests for the shared indicator preparation in src.signals.signal_generator.

Six detectors each carried their own inline copy of these formulas. The risk in
consolidating them is silent numerical drift, so these pin the exact arithmetic
the copies used — Wilder's true range under a 14-period *simple* mean, and a
simple-mean RSI — rather than whatever a library would compute.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals.signal_generator import ensure_indicators


@pytest.fixture
def bars():
    rng = np.random.default_rng(7)
    n = 300
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame(
        {
            "Open": close + rng.normal(0, 0.3, n),
            "High": close + np.abs(rng.normal(0, 0.8, n)),
            "Low": close - np.abs(rng.normal(0, 0.8, n)),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=pd.date_range("2023-01-01", periods=n, freq="B"),
    )


def test_atr_matches_the_inline_formula_it_replaced(bars):
    high_low = bars["High"] - bars["Low"]
    high_close = abs(bars["High"] - bars["Close"].shift())
    low_close = abs(bars["Low"] - bars["Close"].shift())
    expected = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    assert ensure_indicators(bars, "ATR")["ATR"].equals(expected)


def test_rsi_matches_the_inline_formula_it_replaced(bars):
    delta = bars["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    expected = 100 - (100 / (1 + gain / loss))

    assert ensure_indicators(bars, "RSI")["RSI"].equals(expected)


def test_moving_averages_match_the_inline_formulas(bars):
    prepared = ensure_indicators(bars, "SMA_20", "SMA_200")
    assert prepared["SMA_20"].equals(bars["Close"].rolling(20).mean())
    assert prepared["SMA_200"].equals(bars["Close"].rolling(200).mean())


def test_the_callers_frame_is_never_mutated(bars):
    """
    Two of the six copies took their `df.copy()` before the `if` and two took it
    inside, so whether a detector wrote a column back into its caller's frame
    depended on which detector ran and whether the column already existed.
    """
    before = list(bars.columns)
    ensure_indicators(bars, "ATR", "RSI", "SMA_20", "SMA_200")
    assert list(bars.columns) == before


def test_existing_columns_are_left_alone(bars):
    supplied = bars.copy()
    supplied["ATR"] = 42.0
    assert (ensure_indicators(supplied, "ATR")["ATR"] == 42.0).all()


def test_an_unknown_indicator_is_refused(bars):
    # Silently returning a frame without the column would surface far away, as a
    # KeyError inside a detector loop.
    with pytest.raises(ValueError, match="MACD"):
        ensure_indicators(bars, "MACD")


def test_detectors_still_run_end_to_end(bars):
    from src.signals.signal_generator import (
        check_downtrend,
        check_uptrend,
        detect_123_continuation,
        detect_base_breakdown,
        detect_base_breakout,
        detect_pullback_buy,
    )

    for detector in (detect_base_breakout, detect_pullback_buy):
        result = detector(bars)
        assert len(result) == len(bars)
        assert "buy_signal" in result.columns

    # The breakdown detector is the short side, so it emits sell_signal.
    breakdown = detect_base_breakdown(bars)
    assert len(breakdown) == len(bars)
    assert "sell_signal" in breakdown.columns

    assert len(detect_123_continuation(bars)) == len(bars)
    assert 0.0 <= check_uptrend(bars) <= 100.0
    assert 0.0 <= check_downtrend(bars) <= 100.0
