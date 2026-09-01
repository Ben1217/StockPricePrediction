"""
Tests for the price-action structure read.

Two things are worth pinning here. The first is causality: a structure column
that changes retroactively when tomorrow's bar arrives is a leak, and it is the
single easiest mistake to make when a "swing high" is involved. The second is
that the named structures actually mean what they say - a chart making higher
highs and higher lows has to score +1, or the sentence the panel prints is
decoration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.features.price_action import (  # noqa: E402
    BREAKOUT_WINDOW,
    LONG_SWING_WINDOW,
    MEDIUM_SWING_WINDOW,
    PRICE_ACTION_COLUMNS,
    add_price_action_columns,
    price_action_score,
    read_price_action,
)


def _bars_from_close(close: np.ndarray, *, start: str = "2020-01-01") -> pd.DataFrame:
    """OHLCV around a close path, with a small deterministic intrabar range."""
    index = pd.bdate_range(start, periods=len(close))
    close = np.asarray(close, dtype=float)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.004
    low = np.minimum(open_, close) * 0.996
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": np.full(len(close), 1_000_000.0)},
        index=index,
    )


@pytest.fixture
def random_bars() -> pd.DataFrame:
    rng = np.random.default_rng(20240601)
    n = 500
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, n)))
    return _bars_from_close(close)


def test_every_declared_column_is_produced(random_bars):
    frame = add_price_action_columns(random_bars)
    assert [col for col in PRICE_ACTION_COLUMNS if col not in frame.columns] == []


def test_columns_do_not_change_when_later_bars_arrive(random_bars):
    """
    The leak check. Recomputing on a truncated frame must reproduce the longer
    frame's values exactly over the shared rows - including which rows are NaN,
    because a warm-up that shortens once more data exists is also a leak.
    """
    full = add_price_action_columns(random_bars)[PRICE_ACTION_COLUMNS]
    truncated = add_price_action_columns(random_bars.iloc[:300])[PRICE_ACTION_COLUMNS]
    shared = full.iloc[:300]

    pd.testing.assert_frame_equal(shared, truncated)


def test_rising_staircase_reads_as_higher_highs_and_higher_lows():
    # A monotone advance: every window's high and low beat the prior window's.
    close = np.linspace(100, 200, 400)
    frame = add_price_action_columns(_bars_from_close(close))
    structure = frame[f"PA_Structure_{MEDIUM_SWING_WINDOW}"].dropna()

    assert structure.iloc[-1] == pytest.approx(1.0)
    reading = read_price_action(frame, already_built=True)
    assert reading["structure_label"] == "higher highs and higher lows"
    assert reading["label"] == "bullish"
    assert reading["score"] > 0


def test_falling_staircase_reads_as_lower_highs_and_lower_lows():
    close = np.linspace(200, 100, 400)
    frame = add_price_action_columns(_bars_from_close(close))
    structure = frame[f"PA_Structure_{MEDIUM_SWING_WINDOW}"].dropna()

    assert structure.iloc[-1] == pytest.approx(-1.0)
    reading = read_price_action(frame, already_built=True)
    assert reading["structure_label"] == "lower highs and lower lows"
    assert reading["label"] == "bearish"
    assert reading["score"] < 0


def test_breakout_fires_only_on_a_close_above_the_prior_channel():
    """
    A flat range followed by one decisive close above it. The breakout flag must
    be off through the range and on for the breaking bar - and critically, the
    bar that *sets* the range high must not be flagged, which is what the
    shift(1) on the channel exists to prevent.
    """
    flat = np.full(120, 100.0)
    # A single step up, well clear of the 0.4% synthetic intrabar range.
    close = np.r_[flat, [110.0]]
    frame = add_price_action_columns(_bars_from_close(close))

    breakout = frame["PA_Breakout"].dropna()
    assert breakout.iloc[-1] == 1.0
    assert breakout.iloc[:-1].sum() == 0.0


def test_breakdown_fires_on_a_close_below_the_prior_channel():
    flat = np.full(120, 100.0)
    close = np.r_[flat, [90.0]]
    frame = add_price_action_columns(_bars_from_close(close))

    assert frame["PA_Breakdown"].dropna().iloc[-1] == 1.0
    assert frame["PA_Breakout"].dropna().iloc[-1] == 0.0


def test_score_stays_inside_its_declared_range(random_bars):
    score = price_action_score(random_bars).dropna()
    assert not score.empty
    assert score.min() >= -1.0
    assert score.max() <= 1.0


def test_score_is_missing_rather_than_partial_during_warm_up(random_bars):
    """
    A composite averaged over whatever happened to be available would be a
    different feature at bar 30 than at bar 300. It has to be NaN instead.
    """
    score = price_action_score(random_bars)
    # The longest input compares a 60-bar swing against the 60-bar swing before
    # it, so nothing can be scored until two of them have printed.
    first_scored = int(score.reset_index(drop=True).first_valid_index())
    assert first_scored == 2 * LONG_SWING_WINDOW - 1
    assert score.iloc[:first_scored].isna().all()


def test_reading_reports_the_levels_it_measured_against():
    close = 100 + np.sin(np.linspace(0, 12, 300)) * 5
    frame = add_price_action_columns(_bars_from_close(close))
    reading = read_price_action(frame, already_built=True)

    levels = reading["levels"]
    assert levels["window"] == BREAKOUT_WINDOW
    assert levels["support"] < reading["close"] or levels["support"] == pytest.approx(reading["close"], rel=0.05)
    assert levels["resistance"] > levels["support"]


def test_short_history_is_reported_rather_than_scored():
    frame = _bars_from_close(np.linspace(100, 105, 30))
    reading = read_price_action(frame)
    assert reading["available"] is False
    assert "history" in reading["reason"]


def test_missing_columns_raise_rather_than_silently_skipping():
    frame = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    with pytest.raises(KeyError):
        add_price_action_columns(frame)
