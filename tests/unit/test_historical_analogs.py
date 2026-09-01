"""
Tests for the nearest-neighbour read over historical setups.

The claim this module makes is narrow and checkable: "among the bars that looked
most like today, this share went up next". Two ways that claim can quietly
become false, and both are pinned here - a neighbour whose outcome had not
printed yet, and a z-score computed over the whole series so every past row is
scaled by a future it could not have seen.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.features.historical_analogs import (  # noqa: E402
    ANALOG_FEATURE_COLUMNS,
    MIN_ANALOG_HISTORY,
    analog_matches,
    analog_up_rate_series,
)


@pytest.fixture
def descriptor_and_outcomes():
    rng = np.random.default_rng(4242)
    n = 900
    index = pd.bdate_range("2018-01-01", periods=n)
    features = pd.DataFrame(
        rng.normal(size=(n, len(ANALOG_FEATURE_COLUMNS))),
        index=index,
        columns=ANALOG_FEATURE_COLUMNS,
    )
    outcomes = pd.Series(rng.normal(0.0004, 0.015, n), index=index, name="forward_return")
    return features, outcomes


def test_series_is_causal(descriptor_and_outcomes):
    """Appending later bars must not change any earlier bar's analog read."""
    features, outcomes = descriptor_and_outcomes
    full = analog_up_rate_series(features, outcomes)
    truncated = analog_up_rate_series(features.iloc[:600], outcomes.iloc[:600])

    pd.testing.assert_series_equal(full.iloc[:600], truncated)


def test_series_is_missing_until_there_is_history_to_compare_against(descriptor_and_outcomes):
    features, outcomes = descriptor_and_outcomes
    series = analog_up_rate_series(features, outcomes)

    assert series.iloc[:MIN_ANALOG_HISTORY].isna().all()
    assert series.iloc[MIN_ANALOG_HISTORY:].notna().any()


def test_up_rate_is_a_proportion(descriptor_and_outcomes):
    features, outcomes = descriptor_and_outcomes
    series = analog_up_rate_series(features, outcomes).dropna()

    assert series.min() >= 0.0
    assert series.max() <= 1.0


def test_matches_recover_a_planted_regime():
    """
    Two clearly separated clusters of setups, one always followed by a rise and
    one always by a fall. A query sitting in the rising cluster has to come back
    with an up-rate at or near 1 - if it does not, "most similar" is not
    selecting on similarity at all.
    """
    rng = np.random.default_rng(7)
    n = 600
    index = pd.bdate_range("2018-01-01", periods=n)
    cluster = np.tile([0, 1], n // 2)
    values = rng.normal(size=(n, len(ANALOG_FEATURE_COLUMNS))) * 0.05
    values += cluster[:, None] * 6.0
    features = pd.DataFrame(values, index=index, columns=ANALOG_FEATURE_COLUMNS)
    outcomes = pd.Series(np.where(cluster == 1, 0.02, -0.02), index=index, name="forward_return")

    query = pd.Series(
        np.full(len(ANALOG_FEATURE_COLUMNS), 6.0),
        index=ANALOG_FEATURE_COLUMNS,
        name=pd.Timestamp("2026-01-05"),
    )
    result = analog_matches(features, outcomes, query=query)

    assert result["available"] is True
    assert result["up_rate"] == pytest.approx(1.0)
    assert result["mean_forward_return"] > 0


def test_short_history_reports_rather_than_guesses():
    index = pd.bdate_range("2024-01-01", periods=40)
    features = pd.DataFrame(
        np.zeros((40, len(ANALOG_FEATURE_COLUMNS))), index=index, columns=ANALOG_FEATURE_COLUMNS
    )
    outcomes = pd.Series(np.zeros(40), index=index)

    result = analog_matches(features, outcomes)
    assert result["available"] is False
    assert result["n_matches"] == 0
    assert "floor" in result["reason"]


def test_up_rate_ships_with_its_interval_and_a_reference(descriptor_and_outcomes):
    """
    An up-rate without its sample size and the unconditional rate beside it is
    unreadable: 58% is skill against a 50% base rate and noise against a 57% one.
    """
    features, outcomes = descriptor_and_outcomes
    result = analog_matches(features, outcomes)

    assert result["available"] is True
    low, high = result["up_rate_ci"]
    assert 0.0 <= low <= result["up_rate"] <= high <= 1.0
    assert result["reference_up_rate"] is not None
    assert result["n_matches"] == result["k"]


def test_missing_descriptor_columns_raise(descriptor_and_outcomes):
    features, outcomes = descriptor_and_outcomes
    with pytest.raises(ValueError):
        analog_matches(features.rename(columns=lambda name: f"other_{name}"), outcomes)
