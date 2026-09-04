"""
Tests for the A9 leakage checklist.

The heart of this file is the pair of tests that feed :func:`as_of_invariance`
builders whose leakiness is known by construction. A leakage detector that has
never been shown to *catch* anything is indistinguishable from one that always
returns "pass", which is the exact failure mode the module exists to prevent.
So every leak archetype named in A9 gets a test that proves detection, and every
clean archetype gets one that proves no false alarm.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.leakage import (
    ATTESTED,
    CHECK_NAMES,
    FAIL,
    NOT_APPLICABLE,
    PASS,
    CheckResult,
    LeakageReport,
    as_of_invariance,
    check_02_no_global_scaling,
    check_03_support_resistance_as_of,
    check_04_corporate_action_adjustment,
    check_05_purge_and_embargo,
    check_06_hac_or_non_overlapping,
    check_07_fitted_on_validation_only,
    check_08_timestamp_alignment,
    check_09_universe_independent_of_outcomes,
    check_10_frozen_snapshot,
    record_leakage_report,
    run_leakage_checklist,
)
from src.evaluation.splitting import purged_walk_forward_splits


@pytest.fixture
def price_frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2020-01-01", periods=300)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300)))
    return pd.DataFrame({"Close": close}, index=index)


PROBES = [120, 160, 200, 240, 280]


# ---------------------------------------------------------------------------
# The central instrument
# ---------------------------------------------------------------------------


def test_as_of_invariance_passes_a_trailing_rolling_window(price_frame):
    def builder(frame):
        return pd.DataFrame({"ma20": frame["Close"].rolling(20).mean()}, index=frame.index)

    report = as_of_invariance(builder, price_frame, probe_positions=PROBES)

    assert report["probes_run"] == len(PROBES)
    assert report["invariant"] is True
    assert report["per_column"]["ma20"]["verdict"] == PASS


def test_as_of_invariance_passes_an_expanding_window(price_frame):
    def builder(frame):
        return pd.DataFrame({"em": frame["Close"].expanding(20).mean()}, index=frame.index)

    assert as_of_invariance(builder, price_frame, probe_positions=PROBES)["invariant"] is True


def test_as_of_invariance_catches_a_centred_rolling_window(price_frame):
    """
    A9.1's archetypal violation. A centred window's trailing rows are NaN as-of
    and filled once future bars arrive, so the detector must treat "the full
    build produced a value the as-of build could not" as leakage. Counting it as
    insufficient history let this archetype pass silently.
    """
    def builder(frame):
        return pd.DataFrame(
            {"centred": frame["Close"].rolling(20, center=True).mean()}, index=frame.index
        )

    report = as_of_invariance(builder, price_frame, probe_positions=PROBES)

    assert report["invariant"] is False
    assert "centred" in report["leaky_columns"]
    assert report["per_column"]["centred"]["forward_filled_by_full_build"] == len(PROBES)


def test_as_of_invariance_catches_a_full_sample_z_score(price_frame):
    """A9.2 -- scaling statistics computed over the whole sample."""
    def builder(frame):
        close = frame["Close"]
        return pd.DataFrame({"z": (close - close.mean()) / close.std()}, index=frame.index)

    report = as_of_invariance(builder, price_frame, probe_positions=PROBES)

    assert report["invariant"] is False
    assert "z" in report["leaky_columns"]
    assert report["per_column"]["z"]["max_abs_diff"] > 0


def test_as_of_invariance_catches_a_negative_shift(price_frame):
    """A9.8 -- a feature that reads the next bar outright."""
    def builder(frame):
        return pd.DataFrame({"next_close": frame["Close"].shift(-1)}, index=frame.index)

    report = as_of_invariance(builder, price_frame, probe_positions=PROBES)
    assert report["invariant"] is False
    assert "next_close" in report["leaky_columns"]


def test_as_of_invariance_handles_a_builder_that_trims_its_own_tail(price_frame):
    """
    A builder whose label reads h bars forward emits rows only up to t-h from a
    prefix ending at t. Requiring an exact timestamp match skipped every probe
    and reported "not compared" for every column -- a checklist that ran nothing
    while appearing to have run.
    """
    def builder(frame):
        out = pd.DataFrame({"ma5": frame["Close"].rolling(5).mean()}, index=frame.index)
        return out.iloc[:-3]  # trim three rows, as a forward-looking label would

    report = as_of_invariance(builder, price_frame, probe_positions=PROBES)

    assert report["probes_run"] == len(PROBES), "trimmed rows must still be comparable"
    assert len(report["rows_trimmed_by_builder"]) == len(PROBES)
    assert report["invariant"] is True


def test_as_of_invariance_rejects_a_builder_returning_the_wrong_type(price_frame):
    with pytest.raises(TypeError):
        as_of_invariance(lambda frame: "not a frame", price_frame, probe_positions=[10])


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def test_check_02_fingerprints_a_globally_standardised_column(price_frame):
    """
    A correctly *rolling* z-score is essentially never exactly standardised over
    the full sample, so mean 0 and sd 1 to machine precision is a fingerprint.
    """
    def builder(frame):
        close = frame["Close"]
        return pd.DataFrame({"z": (close - close.mean()) / close.std(ddof=0)}, index=frame.index)

    result = check_02_no_global_scaling(builder, price_frame, PROBES)

    assert result.status == FAIL
    assert "z" in result.evidence["globally_standardised_suspects"]


def test_check_03_flags_a_level_function_that_depends_on_where_the_frame_ends(price_frame):
    def levels(frame):
        window = frame.iloc[-50:]
        return {"support": float(window["Close"].min()), "resistance": float(window["Close"].max())}

    result = check_03_support_resistance_as_of(levels, price_frame, PROBES)

    assert result.status == FAIL
    assert result.evidence["n_differing"] > 0


def test_check_03_passes_a_genuinely_as_of_level_function(price_frame):
    """A level function that returns a constant cannot depend on the future."""
    result = check_03_support_resistance_as_of(lambda frame: {"level": 100.0}, price_frame, PROBES)
    assert result.status == PASS


def test_check_04_never_silently_passes_a_back_adjusted_series():
    adjusted = check_04_corporate_action_adjustment({"adjusted": True, "source": "yfinance"})
    assert adjusted.status == ATTESTED
    assert "look-ahead" in adjusted.evidence["limitation"]

    point_in_time = check_04_corporate_action_adjustment({"point_in_time_adjustment": True})
    assert point_in_time.status == PASS


def test_check_05_verifies_the_gap_on_every_fold_and_names_the_offender():
    folds = purged_walk_forward_splits(600, horizon=5, test_size=60, n_splits=3, min_train=100)
    good = check_05_purge_and_embargo(folds, horizon=5, embargo=5)
    assert good.status == PASS
    assert good.evidence["min_gap_bars"] >= 10

    # A hand-built fold with a deliberately short gap must be caught by number.
    class Fold:
        def __init__(self, fold, train_pos, test_pos):
            self.fold, self.train_pos, self.test_pos = fold, train_pos, test_pos

    # gap = min(test) - max(train) - 1, i.e. bars strictly between the blocks.
    tight = [
        Fold(1, np.arange(0, 100), np.arange(111, 160)),  # 111 - 99 - 1 = 11, fine
        Fold(2, np.arange(0, 200), np.arange(203, 260)),  # 203 - 199 - 1 = 3, too small
    ]
    bad = check_05_purge_and_embargo(tight, horizon=5, embargo=5)
    assert bad.status == FAIL
    assert bad.evidence["worst_fold"] == 2
    assert bad.evidence["min_gap_bars"] == 3
    assert bad.evidence["required_gap_bars"] == 10


def test_check_06_truth_table():
    assert check_06_hac_or_non_overlapping(horizon=1).status == PASS
    assert check_06_hac_or_non_overlapping(horizon=20, non_overlapping=True).status == PASS
    assert check_06_hac_or_non_overlapping(horizon=20, hac_lag=19).status == PASS
    assert check_06_hac_or_non_overlapping(horizon=20, hac_lag=25).status == PASS
    assert check_06_hac_or_non_overlapping(horizon=20, hac_lag=5).status == FAIL
    assert check_06_hac_or_non_overlapping(horizon=20).status == FAIL


def test_check_07_fails_anything_fitted_on_test_or_the_full_sample():
    clean = check_07_fitted_on_validation_only(
        [{"name": "isotonic", "fitted_on": "validation", "fold": 1},
         {"name": "ensemble_weights", "fitted_on": "validation", "fold": 1}]
    )
    assert clean.status == PASS

    leaky = check_07_fitted_on_validation_only(
        [{"name": "isotonic", "fitted_on": "validation", "fold": 1},
         {"name": "ensemble_weights", "fitted_on": "test", "fold": 1}]
    )
    assert leaky.status == FAIL
    assert "ensemble_weights" in leaky.detail


def test_check_08_detects_labels_that_do_not_resolve_after_the_decision():
    calendar = pd.bdate_range("2020-01-01", periods=100)
    features = calendar[:90]
    labels = calendar[1:91]

    good = check_08_timestamp_alignment(features, labels, horizon=1, bar_index=calendar)
    assert good.status == PASS

    # Same-bar labels: the decision and its outcome share a timestamp.
    same_bar = check_08_timestamp_alignment(features, features, horizon=1, bar_index=calendar)
    assert same_bar.status == FAIL
    assert same_bar.evidence["labels_not_strictly_after_decision"] == 90

    # Right ordering, wrong gap: labels are 1 bar out when 5 were declared.
    wrong_gap = check_08_timestamp_alignment(features, labels, horizon=5, bar_index=calendar)
    assert wrong_gap.status == FAIL
    assert wrong_gap.evidence["rows_with_wrong_positional_gap"] == 90


def test_check_09_is_an_attestation_and_fails_when_no_rule_is_stated():
    assert check_09_universe_independent_of_outcomes("").status == FAIL

    attested = check_09_universe_independent_of_outcomes(
        "S&P 500 membership as of 2020-01-01", attested_by="Benjamin"
    )
    assert attested.status == ATTESTED
    assert attested.evidence["verifiable_by_code"] is False


def test_check_10_requires_a_verified_snapshot():
    assert check_10_frozen_snapshot(None).status == NOT_APPLICABLE
    assert check_10_frozen_snapshot({"ok": True}).status == PASS
    assert check_10_frozen_snapshot({"ok": False, "mismatched": ["AAPL"]}).status == FAIL


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def test_checklist_always_returns_all_ten_items_in_order():
    """A ten-row table with nine rows invites the reader to assume the tenth passed."""
    report = run_leakage_checklist()

    assert len(report.results) == 10
    assert [result.check_id for result in report.results] == list(range(1, 11))
    assert [result.name for result in report.results] == list(CHECK_NAMES)
    for result in report.results:
        assert result.detail, f"check {result.check_id} gave no reason for its status"


def test_attestations_do_not_count_as_code_verified():
    report = run_leakage_checklist(
        selection_rule="S&P 500 membership as of 2020-01-01",
        loader_meta={"adjusted": True},
    )
    assert report.n_attested >= 2
    assert report.n_verified < 10
    caveat = report.to_dict()["caveat"]
    assert "not a code-verified property" in caveat


def test_all_passed_is_false_when_any_check_fails():
    failing = run_leakage_checklist(
        horizon=20, hac_lag=2,  # below the required 19
        selection_rule="stated",
    )
    assert failing.all_passed is False
    assert failing.n_failed >= 1
    assert any(result.check_id == 6 for result in failing.failures())


def test_checklist_runs_the_real_feature_builder_end_to_end():
    """
    The as-of invariance of the production feature builder, proven rather than
    assumed. It trims its own tail for the forward label, which is exactly the
    case that previously caused every probe to be skipped.
    """
    pytest.importorskip("ta")
    from src.features.direction_features import build_direction_dataset

    rng = np.random.default_rng(20)
    n = 400
    index = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    spread = np.abs(rng.normal(0, 0.008, n)) * close
    bars = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close + spread,
            "Low": close - spread,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )
    bars["High"] = bars[["Open", "High", "Close"]].max(axis=1)
    bars["Low"] = bars[["Open", "Low", "Close"]].min(axis=1)

    report = as_of_invariance(
        lambda frame: build_direction_dataset(frame, horizon=1).features,
        bars,
        probe_positions=[250, 300, 350],
    )

    assert report["probes_run"] == 3
    assert report["leaky_columns"] == []
    assert report["invariant"] is True


def test_record_leakage_report_round_trips(tmp_path):
    report = run_leakage_checklist(selection_rule="stated", attested_by="Benjamin")
    destination = record_leakage_report(report, tmp_path / "leakage.json")

    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert len(payload["results"]) == 10
    assert payload["n_verified_by_code"] + payload["n_failed"] + payload["n_attested"] + payload[
        "n_not_applicable"
    ] == 10
    assert "generated_at" in payload


def test_report_helpers_are_consistent():
    results = tuple(
        CheckResult(i, CHECK_NAMES[i - 1], status, "detail")
        for i, status in enumerate(
            [PASS, PASS, FAIL, ATTESTED, PASS, PASS, PASS, PASS, ATTESTED, NOT_APPLICABLE], start=1
        )
    )
    report = LeakageReport(results)

    assert report.n_verified == 6
    assert report.n_failed == 1
    assert report.n_attested == 2
    assert report.n_not_applicable == 1
    assert report.all_passed is False
    assert len(report.summary_table()) == 10
