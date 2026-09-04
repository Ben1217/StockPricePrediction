"""
Unit tests for src/evaluation/splitting.py (Addendum A Requirement A4).

Every expected value here is derived by hand from the definition of the split
protocol or from the closed form in the docstring, never copied out of what the
implementation happens to print.
"""

import dataclasses
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.evaluation.splitting import (
    Fold,
    HeldOutSplit,
    UNKNOWN_SECTOR,
    describe_split,
    effective_sample_size,
    held_out_ticker_split,
    non_overlapping_folds,
    non_overlapping_positions,
    purged_walk_forward_splits,
)


# ---------------------------------------------------------------------------
# purged_walk_forward_splits -- the hand-checked reference case
# ---------------------------------------------------------------------------
#
# n_rows=100, horizon=5, test_size=10, n_splits=3, min_train=20, embargo=default=5.
#
# Test blocks cover the tail, oldest first:
#   fold 1: test_start = 100 - 3*10 = 70  -> test 70..79
#   fold 2: test_start = 100 - 2*10 = 80  -> test 80..89
#   fold 3: test_start = 100 - 1*10 = 90  -> test 90..99
# train_end = test_start - embargo - horizon = test_start - 10:
#   fold 1: train 0..59, purge 60..64, embargo 65..69
#   fold 2: train 0..69, purge 70..74, embargo 75..79
#   fold 3: train 0..79, purge 80..84, embargo 85..89
# All three train lengths (60, 70, 80) clear min_train=20, so nothing is dropped.


def test_hand_checked_reference_split():
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20
    )
    assert len(folds) == 3
    assert [f.fold for f in folds] == [1, 2, 3]

    expected = [
        (np.arange(0, 60), np.arange(60, 65), np.arange(65, 70), np.arange(70, 80)),
        (np.arange(0, 70), np.arange(70, 75), np.arange(75, 80), np.arange(80, 90)),
        (np.arange(0, 80), np.arange(80, 85), np.arange(85, 90), np.arange(90, 100)),
    ]
    for fold, (train, purged, embargo, test) in zip(folds, expected):
        assert np.array_equal(fold.train_pos, train), f"fold {fold.fold} train"
        assert np.array_equal(fold.purged_pos, purged), f"fold {fold.fold} purge"
        assert np.array_equal(fold.embargo_pos, embargo), f"fold {fold.fold} embargo"
        assert np.array_equal(fold.test_pos, test), f"fold {fold.fold} test"

    assert [f.n_train for f in folds] == [60, 70, 80]
    assert [f.n_test for f in folds] == [10, 10, 10]

    # The four blocks tile [0, test_end) with no overlap and no hole.
    for fold in folds:
        stitched = np.concatenate(
            [fold.train_pos, fold.purged_pos, fold.embargo_pos, fold.test_pos]
        )
        assert np.array_equal(stitched, np.arange(stitched[0], stitched[-1] + 1))


def test_embargo_larger_than_horizon_widens_only_the_embargo_block():
    # train_end = test_start - embargo - horizon = 70 - 8 - 5 = 57.
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20, embargo=8
    )
    first = folds[0]
    assert np.array_equal(first.train_pos, np.arange(0, 57))
    assert np.array_equal(first.purged_pos, np.arange(57, 62))  # exactly horizon bars
    assert np.array_equal(first.embargo_pos, np.arange(62, 70))  # exactly embargo bars
    assert np.array_equal(first.test_pos, np.arange(70, 80))


def test_embargo_below_horizon_is_an_error():
    with pytest.raises(ValueError, match="below horizon"):
        purged_walk_forward_splits(500, horizon=5, embargo=4, test_size=63, min_train=100)


def test_default_embargo_equals_horizon():
    folds = purged_walk_forward_splits(
        400, horizon=7, test_size=20, n_splits=2, min_train=50
    )
    for fold in folds:
        assert fold.purged_pos.size == 7
        assert fold.embargo_pos.size == 7


def test_short_folds_are_dropped_not_shrunk():
    # min_train=65 kills fold 1 only (train_end 60 < 65); folds 2 and 3 have
    # train_end 70 and 80 and survive, renumbered 1 and 2.
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=65
    )
    assert len(folds) == 2
    assert [f.fold for f in folds] == [1, 2]
    assert [f.n_train for f in folds] == [70, 80]
    assert np.array_equal(folds[0].test_pos, np.arange(80, 90))
    assert np.array_equal(folds[1].test_pos, np.arange(90, 100))


def test_no_fold_fits_returns_empty_and_warns_with_the_exact_shortfall(caplog):
    # One fold needs min_train + horizon + embargo + test_size
    #             = 252 + 5 + 5 + 63 = 325 rows. With 100 rows it is short by 225.
    with caplog.at_level(logging.WARNING, logger="src.evaluation.splitting"):
        folds = purged_walk_forward_splits(
            100, horizon=5, test_size=63, n_splits=4, min_train=252
        )
    assert folds == []
    message = caplog.text
    assert "one fold needs 325 rows" in message
    assert "short by 225" in message


def test_parameter_validation():
    with pytest.raises(ValueError, match="n_rows"):
        purged_walk_forward_splits(-1, horizon=1)
    with pytest.raises(ValueError, match="horizon"):
        purged_walk_forward_splits(500, horizon=0)
    with pytest.raises(ValueError, match="test_size"):
        purged_walk_forward_splits(500, horizon=1, test_size=0)
    with pytest.raises(ValueError, match="n_splits"):
        purged_walk_forward_splits(500, horizon=1, n_splits=0)
    with pytest.raises(ValueError, match="min_train"):
        purged_walk_forward_splits(500, horizon=1, min_train=0)


def test_gap_invariant_holds_across_a_parameter_grid():
    """max(train) + horizon + embargo <= min(test) for every fold, always."""
    checked = 0
    for n_rows in (300, 500, 756, 1260):
        for horizon in (1, 5, 20):
            for test_size in (21, 63):
                for n_splits in (1, 3, 4):
                    for min_train in (60, 252):
                        for extra in (0, 3, 30):
                            embargo = horizon + extra
                            folds = purged_walk_forward_splits(
                                n_rows,
                                horizon=horizon,
                                test_size=test_size,
                                n_splits=n_splits,
                                min_train=min_train,
                                embargo=embargo,
                            )
                            for fold in folds:
                                checked += 1
                                train_last = int(fold.train_pos.max())
                                test_first = int(fold.test_pos.min())
                                assert train_last + horizon + embargo <= test_first

                                # The gap is exactly the purge plus the embargo,
                                # and the two blocks are disjoint from both sides.
                                assert fold.purged_pos.size == horizon
                                assert fold.embargo_pos.size == embargo
                                assert test_first - train_last - 1 == horizon + embargo
                                assert not np.intersect1d(
                                    fold.train_pos, fold.test_pos
                                ).size
                                assert not np.intersect1d(
                                    fold.purged_pos, fold.train_pos
                                ).size
                                assert not np.intersect1d(
                                    fold.embargo_pos, fold.test_pos
                                ).size

                                # Nothing runs off the end of the series, and the
                                # training window meets the floor it promised.
                                assert int(fold.test_pos.max()) < n_rows
                                assert fold.n_train >= min_train
                                assert fold.n_test == test_size
    assert checked > 200, f"grid produced only {checked} folds; it is not exercising much"


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------


def test_fold_is_frozen_and_normalises_positions():
    fold = Fold(fold=1, train_pos=[0, 1, 2], test_pos=[5, 6], purged_pos=[3], embargo_pos=[4])
    assert fold.n_train == 3 and fold.n_test == 2
    assert fold.train_pos.dtype == np.int64
    with pytest.raises(dataclasses.FrozenInstanceError):
        fold.fold = 2


def test_fold_rejects_bad_inputs():
    with pytest.raises(ValueError, match="1-based"):
        Fold(fold=0, train_pos=[0], test_pos=[1], purged_pos=[], embargo_pos=[])
    with pytest.raises(ValueError, match="negative row positions"):
        Fold(fold=1, train_pos=[-1, 0], test_pos=[1], purged_pos=[], embargo_pos=[])


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------


def test_effective_sample_size_horizon_one_is_exact():
    # h=1 -> the sum runs over an empty range -> f = 1 -> n_eff = n.
    assert effective_sample_size(100, 1) == 100.0
    assert effective_sample_size(1, 1) == 1.0
    assert effective_sample_size(2521, 1) == 2521.0


def test_effective_sample_size_horizon_two_hand_computed():
    # n=100, h=2. k runs over {1} only.
    #   (1 - 1/100) * (1 - 1/2) = 0.99 * 0.5 = 0.495
    #   f = 1 + 2 * 0.495 = 1.99
    #   n_eff = 100 / 1.99 = 50.251256281...
    assert effective_sample_size(100, 2) == pytest.approx(100.0 / 1.99, rel=1e-12)
    assert effective_sample_size(100, 2) == pytest.approx(50.25125628140704, rel=1e-12)


def test_effective_sample_size_horizon_five_hand_computed():
    # n=100, h=5. k = 1..4, terms (1 - k/100) * (1 - k/5):
    #   k=1: 0.99 * 0.8 = 0.792
    #   k=2: 0.98 * 0.6 = 0.588
    #   k=3: 0.97 * 0.4 = 0.388
    #   k=4: 0.96 * 0.2 = 0.192
    #   sum = 1.960,  f = 1 + 2 * 1.960 = 4.92
    #   n_eff = 100 / 4.92 = 20.325203252...
    assert effective_sample_size(100, 5) == pytest.approx(100.0 / 4.92, rel=1e-12)
    assert effective_sample_size(100, 5) == pytest.approx(20.32520325203252, rel=1e-12)


def test_effective_sample_size_truncates_at_n_minus_one():
    # n=3, h=10: k can only reach min(9, 2) = 2.
    #   k=1: (1 - 1/3) * (1 - 1/10) = (2/3) * 0.9 = 0.6
    #   k=2: (1 - 2/3) * (1 - 2/10) = (1/3) * 0.8 = 0.266666...
    #   f = 1 + 2 * (0.6 + 4/15) = 1 + 2 * 13/15 = 41/15
    #   n_eff = 3 / (41/15) = 45/41
    assert effective_sample_size(3, 10) == pytest.approx(45.0 / 41.0, rel=1e-12)


def test_effective_sample_size_single_observation_and_monotone_penalty():
    # One observation cannot overlap anything, so it is worth exactly one.
    assert effective_sample_size(1, 20) == 1.0
    # Longer horizons never buy information: n_eff is non-increasing in h and
    # never exceeds n.
    previous = float("inf")
    for horizon in range(1, 40):
        value = effective_sample_size(252, horizon)
        assert value <= 252.0 + 1e-12
        assert value <= previous + 1e-12
        previous = value


def test_effective_sample_size_rejects_undefined_inputs():
    with pytest.raises(ValueError, match="n must be"):
        effective_sample_size(0, 5)
    with pytest.raises(ValueError, match="n must be"):
        effective_sample_size(-3, 5)
    with pytest.raises(ValueError, match="horizon must be"):
        effective_sample_size(100, 0)


# ---------------------------------------------------------------------------
# non_overlapping_positions / non_overlapping_folds
# ---------------------------------------------------------------------------


def test_non_overlapping_positions_hand_checked():
    positions = np.arange(70, 80)
    assert np.array_equal(non_overlapping_positions(positions, 5), np.array([70, 75]))
    assert np.array_equal(
        non_overlapping_positions(positions, 3), np.array([70, 73, 76, 79])
    )
    assert np.array_equal(non_overlapping_positions(positions, 1), positions)
    assert non_overlapping_positions(positions, 20).tolist() == [70]
    assert non_overlapping_positions([], 5).size == 0


def test_non_overlapping_positions_returns_a_copy():
    positions = np.arange(70, 80)
    thinned = non_overlapping_positions(positions, 5)
    thinned[0] = 999
    assert positions[0] == 70


def test_non_overlapping_positions_rejects_bad_horizon():
    with pytest.raises(ValueError, match="horizon must be"):
        non_overlapping_positions(np.arange(10), 0)


def test_non_overlapping_folds_thins_only_the_test_block():
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20
    )
    thinned = non_overlapping_folds(folds, 5)
    assert [f.fold for f in thinned] == [1, 2, 3]
    for original, new in zip(folds, thinned):
        assert np.array_equal(new.train_pos, original.train_pos)
        assert np.array_equal(new.purged_pos, original.purged_pos)
        assert np.array_equal(new.embargo_pos, original.embargo_pos)
        # 10 test rows at horizon 5 -> 2 non-overlapping anchors, 5 bars apart.
        assert new.n_test == 2
        assert np.array_equal(new.test_pos, original.test_pos[::5])
        assert int(np.diff(new.test_pos)[0]) == 5
    # The originals are untouched.
    assert folds[0].n_test == 10


# ---------------------------------------------------------------------------
# held_out_ticker_split
# ---------------------------------------------------------------------------

TECH = [f"T{i:02d}" for i in range(1, 11)]  # 10 members
FIN = [f"F{i}" for i in range(1, 6)]  # 5 members
ENERGY = [f"E{i}" for i in range(1, 4)]  # 3 members
UNIVERSE = TECH + FIN + ENERGY  # 18 tickers
SECTORS = {
    **{t: "Technology" for t in TECH},
    **{t: "Financials" for t in FIN},
    **{t: "Energy" for t in ENERGY},
}


def test_stratification_is_exact_per_sector():
    # holdout_fraction = 0.2:
    #   Technology 10 -> round(2.0) = 2
    #   Financials  5 -> round(1.0) = 1
    #   Energy      3 -> round(0.6) = 1
    # 4 of 18 held out -> 4/18 = 0.222222.
    split = held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.2, seed=42)
    assert isinstance(split, HeldOutSplit)
    assert len(split.by_sector["Technology"]["held_out"]) == 2
    assert len(split.by_sector["Financials"]["held_out"]) == 1
    assert len(split.by_sector["Energy"]["held_out"]) == 1
    assert len(split.by_sector["Technology"]["in_universe"]) == 8
    assert len(split.by_sector["Financials"]["in_universe"]) == 4
    assert len(split.by_sector["Energy"]["in_universe"]) == 2

    assert len(split.held_out) == 4
    assert len(split.in_universe) == 14
    assert split.holdout_fraction_actual == pytest.approx(4 / 18, abs=1e-6)

    # The two sides partition the universe exactly: no overlap, nothing lost.
    assert set(split.held_out) & set(split.in_universe) == set()
    assert set(split.held_out) | set(split.in_universe) == set(UNIVERSE)
    assert split.held_out == sorted(split.held_out)
    assert split.in_universe == sorted(split.in_universe)

    # Every held-out ticker really belongs to the sector it was drawn from.
    for sector, members in split.by_sector.items():
        for ticker in members["held_out"] + members["in_universe"]:
            assert SECTORS[ticker] == sector


def test_split_is_deterministic_and_seed_sensitive():
    first = held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.2, seed=7)
    second = held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.2, seed=7)
    assert first.held_out == second.held_out
    assert first.in_universe == second.in_universe
    assert first.by_sector == second.by_sector
    assert first.holdout_fraction_actual == second.holdout_fraction_actual

    # A different seed must be able to produce a different draw, otherwise the
    # "held out" set is not random at all.
    draws = {
        tuple(held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.2, seed=s).held_out)
        for s in range(10)
    }
    assert len(draws) > 1


def test_split_ignores_input_ordering():
    shuffled_tickers = list(reversed(UNIVERSE))
    shuffled_sectors = {t: SECTORS[t] for t in reversed(UNIVERSE)}
    a = held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.2, seed=42)
    b = held_out_ticker_split(shuffled_tickers, shuffled_sectors, holdout_fraction=0.2, seed=42)
    assert a.held_out == b.held_out
    assert a.in_universe == b.in_universe


def test_small_sectors_get_at_least_one_but_never_all():
    # PAIR has 2 members: round(0.9 * 2) = 2, clamped to 1 so the sector keeps a
    # trainable member. SOLO has 1 member: clamped to 0.
    tickers = ["A1", "A2", "B1"]
    sectors = {"A1": "Pair", "A2": "Pair", "B1": "Solo"}
    split = held_out_ticker_split(tickers, sectors, holdout_fraction=0.9, seed=1)
    assert len(split.by_sector["Pair"]["held_out"]) == 1
    assert len(split.by_sector["Pair"]["in_universe"]) == 1
    assert split.by_sector["Solo"]["held_out"] == []
    assert split.by_sector["Solo"]["in_universe"] == ["B1"]
    assert split.holdout_fraction_actual == pytest.approx(1 / 3, abs=1e-6)

    # A tiny fraction still takes one from a 2-member sector: round(0.01*2) = 0,
    # lifted to 1 by the "at least one" rule.
    tiny = held_out_ticker_split(tickers, sectors, holdout_fraction=0.01, seed=1)
    assert len(tiny.by_sector["Pair"]["held_out"]) == 1


def test_zero_fraction_holds_nothing_out():
    split = held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=0.0, seed=42)
    assert split.held_out == []
    assert split.in_universe == sorted(UNIVERSE)
    assert split.holdout_fraction_actual == 0.0


def test_missing_sectors_fall_into_the_unknown_bucket_and_are_stratified():
    tickers = TECH + ["X1", "X2", "X3", "X4"]
    sectors = {t: "Technology" for t in TECH}
    sectors["X1"] = None
    sectors["X2"] = "   "
    # X3 and X4 are absent from the mapping entirely.
    split = held_out_ticker_split(tickers, sectors, holdout_fraction=0.5, seed=3)
    assert set(split.by_sector) == {"Technology", UNKNOWN_SECTOR}
    assert sorted(
        split.by_sector[UNKNOWN_SECTOR]["held_out"]
        + split.by_sector[UNKNOWN_SECTOR]["in_universe"]
    ) == ["X1", "X2", "X3", "X4"]
    # round(0.5 * 4) = 2 held out of the UNKNOWN bucket, round(0.5 * 10) = 5 of Tech.
    assert len(split.by_sector[UNKNOWN_SECTOR]["held_out"]) == 2
    assert len(split.by_sector["Technology"]["held_out"]) == 5


def test_split_rejects_bad_input():
    with pytest.raises(ValueError, match="duplicate tickers"):
        held_out_ticker_split(["A", "B", "A"], {"A": "S", "B": "S"})
    with pytest.raises(ValueError, match="holdout_fraction"):
        held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=1.0)
    with pytest.raises(ValueError, match="holdout_fraction"):
        held_out_ticker_split(UNIVERSE, SECTORS, holdout_fraction=-0.1)
    with pytest.raises(ValueError, match="empty"):
        held_out_ticker_split([], {})


# ---------------------------------------------------------------------------
# describe_split
# ---------------------------------------------------------------------------


def test_describe_split_hand_checked_dates_and_gaps():
    # A plain calendar-daily index makes every date checkable by counting:
    # index[i] = 2020-01-01 + i days. 2020 is a leap year, so
    #   index[30] = 2020-01-31, index[31] = 2020-02-01, index[59] = 2020-02-29,
    #   index[60] = 2020-03-01, index[70] = 2020-03-11, index[79] = 2020-03-20.
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20
    )
    rows = describe_split(folds, index)
    assert len(rows) == 3

    first = rows[0]
    assert first == {
        "fold": 1,
        "train_start": "2020-01-01",
        "train_end": "2020-02-29",
        "test_start": "2020-03-11",
        "test_end": "2020-03-20",
        "n_train": 60,
        "n_test": 10,
        "purge_bars": 5,
        "embargo_bars": 5,
        "gap_bars": 10,
        # Feb 29 -> Mar 11 is 11 calendar days, one more than the 10 bars of
        # gap, because the boundary bars themselves are one day apart.
        "gap_calendar_days": 11,
    }

    third = rows[2]
    assert third["fold"] == 3
    assert third["n_train"] == 80
    assert third["train_end"] == "2020-03-20"  # index[79]
    assert third["test_start"] == "2020-03-31"  # index[90]
    assert third["test_end"] == "2020-04-09"  # index[99]
    assert third["gap_bars"] == 10
    assert third["gap_calendar_days"] == 11


def test_describe_split_calendar_gap_exceeds_bar_gap_on_a_trading_calendar():
    # Business days only. train_end sits at position 59 (Tue 2020-03-24) and
    # test_start at 70 (Wed 2020-04-08): 10 bars strictly between them, but 11
    # positions apart, and those 11 business days span two weekends. So the
    # calendar gap is 15 days against a 10-bar gap. This is exactly why both
    # numbers are reported -- an embargo counted in bars is not an embargo
    # counted in days, and a reader checking the protocol needs to see both.
    index = pd.bdate_range("2020-01-01", periods=100)
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20
    )
    row = describe_split(folds, index)[0]
    assert row["gap_bars"] == 10
    assert row["gap_calendar_days"] == 15
    assert row["train_end"] == str(index[59].date())
    assert row["test_start"] == str(index[70].date())


def test_describe_split_rejects_an_index_shorter_than_the_folds():
    index = pd.date_range("2020-01-01", periods=50, freq="D")
    folds = purged_walk_forward_splits(
        100, horizon=5, test_size=10, n_splits=3, min_train=20
    )
    with pytest.raises(ValueError, match="row position 79 but the index has 50 rows"):
        describe_split(folds, index)


def test_describe_split_reports_none_rather_than_a_fabricated_date():
    index = pd.date_range("2020-01-01", periods=20, freq="D")
    fold = Fold(fold=1, train_pos=[0, 1, 2], test_pos=[], purged_pos=[3], embargo_pos=[4])
    row = describe_split([fold], index)[0]
    assert row["test_start"] is None
    assert row["test_end"] is None
    assert row["gap_calendar_days"] is None
    assert row["n_test"] == 0
    assert row["train_end"] == "2020-01-03"


def test_describe_split_survives_thinned_folds():
    index = pd.date_range("2020-01-01", periods=100, freq="D")
    folds = non_overlapping_folds(
        purged_walk_forward_splits(100, horizon=5, test_size=10, n_splits=3, min_train=20), 5
    )
    rows = describe_split(folds, index)
    assert [r["n_test"] for r in rows] == [2, 2, 2]
    # The protocol description still reports the real purge and embargo.
    assert [r["gap_bars"] for r in rows] == [10, 10, 10]
    assert rows[0]["test_start"] == "2020-03-11"  # index[70]
    assert rows[0]["test_end"] == "2020-03-16"  # index[75]
