"""
Tests for the best-performing-model ranking.

The selection decides what a user sees drawn on the price chart, so the cases
covered here are the ones where a plausible-looking implementation would name
the wrong winner: metrics in different units, metrics only some candidates were
measured on, horizons that are not the same measurement, and models that scored
well but are not allowed to be served.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.models import model_selection
from src.models.model_selection import (
    Candidate,
    collect_candidates,
    rank_candidates,
    select_best_models,
)


# ---------------------------------------------------------------------------
# Fixtures: on-disk artifacts the selector reads
# ---------------------------------------------------------------------------

def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _unified_metadata(model_type: str, *, mae: float, rmse: float, mape: float,
                      r2_return: float, accuracy: float, horizon: int = 1) -> dict:
    return {
        "model_type": model_type,
        "symbol": "TEST",
        "horizon": horizon,
        "objective": "unified_price_and_direction",
        "trained_at": "2026-01-01T00:00:00",
        "holdout": {
            "n_test": 180,
            "base_rate": 0.52,
            "test_start": "2025-06-01",
            "test_end": "2026-01-01",
            "price_mae": mae,
            "price_rmse": rmse,
            "price_mape": mape,
            "price_r2": 0.95,
            "price_r2_return": r2_return,
            "direction_accuracy": accuracy,
        },
    }


def _regression_metadata(model_type: str, *, horizon: int, mae: float, rmse: float,
                         mape: float, directional_accuracy: float,
                         passes_baseline: bool = True) -> dict:
    return {
        "model_type": model_type,
        "symbol": "TEST",
        "horizon": horizon,
        "target_type": "return_regression",
        "objective": "future_return_pct",
        "trained_at": "2026-01-01T00:00:00",
        "split_sizes": {"train": 800, "val": 150, "test": 175},
        "test_metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "directional_accuracy": directional_accuracy,
        },
        "skill": {"test": {"skill_score": 0.05 if passes_baseline else -0.04}},
        "passes_baseline": passes_baseline,
    }


def _benchmark_row(model_name: str, *, mae: float, rmse: float, mape: float,
                   r2_return: float, accuracy: float, f1: float, roc_auc: float,
                   horizon: int = 1) -> dict:
    return {
        "model_name": model_name,
        "symbol": "TEST",
        "interval": "1d",
        "horizon": horizon,
        "n_splits": 3,
        "total_test_rows": 189,
        "base_rate": 0.52,
        "price_mae": mae,
        "price_rmse": rmse,
        "price_mape": mape,
        "price_r2": 0.9,
        "price_r2_return": r2_return,
        "direction_accuracy": accuracy,
        "direction_f1": f1,
        "direction_roc_auc": roc_auc,
    }


@pytest.fixture
def artifacts(tmp_path: Path):
    """Empty benchmark, bundle and report locations the tests fill in."""
    return {
        "bundles_dir": tmp_path / "bundles",
        "benchmark_artifact": tmp_path / "benchmark_results.json",
        "report_dir": tmp_path / "reports",
    }


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _candidate(model_type: str, price=None, direction=None, blocked=None, n_test=100) -> Candidate:
    return Candidate(
        model_type=model_type,
        label=model_type,
        evidence="walk_forward_benchmark",
        horizon=1,
        price=price or {},
        direction=direction or {},
        context={"n_test_rows": n_test},
        blocked_reason=blocked,
    )


def test_price_winner_is_the_model_that_leads_the_error_metrics():
    candidates = [
        _candidate("a", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0, "r2": 0.10}),
        _candidate("b", price={"mae": 5.0, "rmse": 6.0, "mape": 2.0, "r2": -0.30}),
    ]
    selection = rank_candidates(candidates, "price")

    assert selection.winner.model_type == "a"
    assert selection.metrics_used == ["mae", "rmse", "mape", "r2"]
    assert set(selection.metric_winners.values()) == {"a"}


def test_higher_r2_wins_while_lower_error_wins():
    """The two directions are not the same comparison and must not be conflated."""
    candidates = [
        # Lower error, worse R-squared.
        _candidate("low_error", price={"mae": 1.0, "rmse": 1.0, "mape": 1.0, "r2": -0.5}),
        # Higher R-squared, worse error.
        _candidate("high_r2", price={"mae": 9.0, "rmse": 9.0, "mape": 9.0, "r2": 0.5}),
    ]
    selection = rank_candidates(candidates, "price")

    assert selection.metric_winners["mae"] == "low_error"
    assert selection.metric_winners["r2"] == "high_r2"
    # Three of four metrics went to low_error, so it takes the mean rank.
    assert selection.winner.model_type == "low_error"


def test_direction_winner_is_ranked_on_accuracy_f1_and_auc():
    candidates = [
        _candidate("a", direction={"accuracy": 0.52, "f1": 0.50, "roc_auc": 0.51}),
        _candidate("b", direction={"accuracy": 0.57, "f1": 0.56, "roc_auc": 0.59}),
    ]
    selection = rank_candidates(candidates, "direction")

    assert selection.winner.model_type == "b"
    assert selection.metrics_used == ["accuracy", "f1", "roc_auc"]


def test_metrics_only_some_candidates_carry_are_not_ranked_on():
    """
    A metric one model was measured on and another was not decides nothing.

    Ranking on it would hand the win to whoever happened to be evaluated more
    thoroughly, which is a property of the evaluation run, not of the model.
    """
    candidates = [
        _candidate("scored_everywhere", price={"mae": 5.0, "rmse": 6.0, "mape": 2.0, "r2": 0.4}),
        _candidate("no_r2", price={"mae": 3.0, "rmse": 4.0, "mape": 1.0, "r2": None}),
    ]
    selection = rank_candidates(candidates, "price")

    assert "r2" not in selection.metrics_used
    assert selection.metrics_used == ["mae", "rmse", "mape"]
    assert selection.winner.model_type == "no_r2"


def test_a_model_that_cannot_be_served_never_wins():
    candidates = [
        _candidate("blocked", price={"mae": 0.1, "rmse": 0.1, "mape": 0.1, "r2": 0.9},
                   blocked="it does not beat a constant forecast"),
        _candidate("servable", price={"mae": 5.0, "rmse": 6.0, "mape": 2.0, "r2": 0.1}),
    ]
    selection = rank_candidates(candidates, "price")

    assert selection.winner.model_type == "servable"
    assert [row["model_type"] for row in selection.excluded] == ["blocked"]
    assert selection.excluded[0]["excluded_because"] == "it does not beat a constant forecast"


def test_no_servable_candidate_produces_a_reason_rather_than_a_winner():
    selection = rank_candidates(
        [_candidate("blocked", price={"mae": 1.0}, blocked="it failed its gate")],
        "price",
    )

    assert selection.winner is None
    assert selection.ranked == []
    assert "No servable model" in selection.reason


def test_ties_share_a_rank_and_are_broken_by_the_larger_test_sample():
    candidates = [
        _candidate("small_sample", price={"mae": 2.0, "rmse": 2.0, "mape": 2.0, "r2": 0.1}, n_test=50),
        _candidate("large_sample", price={"mae": 2.0, "rmse": 2.0, "mape": 2.0, "r2": 0.1}, n_test=500),
    ]
    selection = rank_candidates(candidates, "price")

    assert selection.winner.model_type == "large_sample"
    assert {row["mean_rank"] for row in selection.ranked} == {1.5}


# ---------------------------------------------------------------------------
# Collecting candidates from disk
# ---------------------------------------------------------------------------

def test_return_space_errors_are_converted_before_being_compared(artifacts):
    """
    A regression bundle's MAE is a return; a unified bundle's is dollars.

    Compared raw, the regression bundle's 0.02 would beat every dollar figure on
    the board and win every symbol priced above a dollar.
    """
    _write(
        artifacts["bundles_dir"] / "TEST" / "xgboost" / "30" / "metadata.json",
        _regression_metadata("xgboost", horizon=30, mae=0.02, rmse=0.03, mape=2.0,
                             directional_accuracy=0.55),
    )

    candidates = collect_candidates(
        "TEST", 30, reference_price=200.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert len(candidates) == 1
    price = candidates[0].price
    assert price["mae"] == pytest.approx(4.0)      # 0.02 * 200
    assert price["rmse"] == pytest.approx(6.0)     # 0.03 * 200
    # A relative error is the same number in both spaces, so MAPE is untouched.
    assert price["mape"] == pytest.approx(2.0)
    assert price["mae_return"] == pytest.approx(0.02)


def test_errors_stay_unset_without_a_reference_price(artifacts):
    """Reporting a return as though it were dollars is worse than reporting nothing."""
    _write(
        artifacts["bundles_dir"] / "TEST" / "xgboost" / "30" / "metadata.json",
        _regression_metadata("xgboost", horizon=30, mae=0.02, rmse=0.03, mape=2.0,
                             directional_accuracy=0.55),
    )

    candidates = collect_candidates(
        "TEST", 30, reference_price=None,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert candidates[0].price["mae"] is None
    assert candidates[0].price["rmse"] is None
    assert candidates[0].price["mape"] == pytest.approx(2.0)


def test_a_model_scored_at_another_horizon_is_not_a_candidate(artifacts):
    _write(
        artifacts["bundles_dir"] / "TEST" / "xgboost" / "7" / "metadata.json",
        _regression_metadata("xgboost", horizon=7, mae=0.01, rmse=0.02, mape=1.0,
                             directional_accuracy=0.55),
    )
    _write(
        artifacts["bundles_dir"] / "TEST" / "lstm" / "30" / "metadata.json",
        _regression_metadata("lstm", horizon=30, mae=0.05, rmse=0.06, mape=5.0,
                             directional_accuracy=0.51),
    )

    at_30 = collect_candidates(
        "TEST", 30, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert [candidate.model_type for candidate in at_30] == ["lstm"]


def test_the_walk_forward_benchmark_outranks_a_bundles_own_holdout(artifacts):
    """
    Both describe unified_xgboost; the multi-fold comparison is the better evidence.

    A holdout scored on its own split is not comparable with models scored on
    the shared folds, so the benchmark row is the one kept.
    """
    _write(
        artifacts["benchmark_artifact"],
        [_benchmark_row("unified_xgboost", mae=3.0, rmse=4.0, mape=1.0,
                        r2_return=0.05, accuracy=0.55, f1=0.54, roc_auc=0.57)],
    )
    _write(
        artifacts["bundles_dir"] / "TEST" / "unified_xgboost" / "metadata.json",
        _unified_metadata("unified_xgboost", mae=9.9, rmse=9.9, mape=9.9,
                          r2_return=-0.9, accuracy=0.40),
    )

    candidates = collect_candidates(
        "TEST", 1, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert len(candidates) == 1
    assert candidates[0].evidence == "walk_forward_benchmark"
    assert candidates[0].price["mae"] == pytest.approx(3.0)


def test_benchmark_rows_for_another_symbol_or_interval_are_ignored(artifacts):
    _write(
        artifacts["benchmark_artifact"],
        [
            _benchmark_row("unified_xgboost", mae=1.0, rmse=1.0, mape=1.0,
                           r2_return=0.1, accuracy=0.6, f1=0.6, roc_auc=0.6),
            {**_benchmark_row("unified_lstm", mae=1.0, rmse=1.0, mape=1.0,
                              r2_return=0.1, accuracy=0.6, f1=0.6, roc_auc=0.6),
             "symbol": "OTHER"},
            {**_benchmark_row("unified_random_forest", mae=1.0, rmse=1.0, mape=1.0,
                              r2_return=0.1, accuracy=0.6, f1=0.6, roc_auc=0.6),
             "interval": "15m"},
        ],
    )

    candidates = collect_candidates(
        "TEST", 1,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert [candidate.model_type for candidate in candidates] == ["unified_xgboost"]


def test_direction_report_that_failed_its_verdict_is_excluded(artifacts):
    report_dir = artifacts["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    _write(
        report_dir / "TEST_logistic_report.json",
        {
            "config": {"model": "logistic", "horizon": 1, "n_folds_run": 4},
            "pooled": {
                "model": {
                    "n": 252,
                    "accuracy": 0.99,
                    "base_rate": 0.53,
                    "roc_auc": 0.99,
                    "brier_score": 0.01,
                    "mcc": 0.9,
                    "class_up": {"f1": 0.99},
                },
                "edge_vs_best_baseline": {"edge_pp": 45.0},
            },
            "verdict": {"ship": False, "failed_criteria": ["passes_leakage_check"]},
        },
    )
    _write(
        artifacts["bundles_dir"] / "TEST" / "unified_lstm" / "metadata.json",
        _unified_metadata("unified_lstm", mae=3.0, rmse=4.0, mape=1.0,
                          r2_return=0.05, accuracy=0.55),
    )

    best = select_best_models(
        "TEST", 1, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=report_dir,
    )

    assert best.direction.winner.model_type == "unified_lstm"
    excluded = {row["model_type"]: row["excluded_because"] for row in best.direction.excluded}
    assert "logistic" in excluded
    assert "passes_leakage_check" in excluded["logistic"]


def test_price_and_direction_winners_can_be_different_models(artifacts):
    """The whole reason the module returns two winners rather than one."""
    _write(
        artifacts["benchmark_artifact"],
        [
            _benchmark_row("unified_random_forest", mae=3.0, rmse=4.0, mape=1.0,
                           r2_return=0.05, accuracy=0.49, f1=0.48, roc_auc=0.50),
            _benchmark_row("unified_lstm", mae=8.0, rmse=9.0, mape=3.0,
                           r2_return=-0.6, accuracy=0.57, f1=0.57, roc_auc=0.59),
        ],
    )

    best = select_best_models(
        "TEST", 1, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert best.price.winner.model_type == "unified_random_forest"
    assert best.direction.winner.model_type == "unified_lstm"


def test_skill_gate_can_be_switched_off(artifacts, monkeypatch):
    """
    The gate is a setting, and the selection has to observe the same one serving does.

    With enforcement off, a bundle the serving layer will happily run must not
    be filtered out here, or the chart would report "no model" for a symbol that
    can in fact be served.
    """
    _write(
        artifacts["bundles_dir"] / "TEST" / "xgboost" / "30" / "metadata.json",
        _regression_metadata("xgboost", horizon=30, mae=0.02, rmse=0.03, mape=2.0,
                             directional_accuracy=0.55, passes_baseline=False),
    )

    monkeypatch.setenv("QUANTVISION_ENFORCE_MODEL_SKILL", "true")
    gated = select_best_models(
        "TEST", 30, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )
    assert gated.price.winner is None

    monkeypatch.setenv("QUANTVISION_ENFORCE_MODEL_SKILL", "false")
    ungated = select_best_models(
        "TEST", 30, reference_price=100.0,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )
    assert ungated.price.winner.model_type == "xgboost"


def test_missing_artifacts_are_not_an_error(artifacts):
    best = select_best_models(
        "NOBODY", 30,
        bundles_dir=artifacts["bundles_dir"],
        benchmark_artifact=artifacts["benchmark_artifact"],
        report_dir=artifacts["report_dir"],
    )

    assert best.price.winner is None
    assert best.direction.winner is None
    assert best.price.reason


def test_non_finite_scores_are_never_ranked():
    """NaN sorts unpredictably; a model with one has not been measured."""
    candidates = [
        _candidate("nan_mape", price={"mae": 1.0, "rmse": 1.0, "mape": float("nan"), "r2": 0.5}),
        _candidate("measured", price={"mae": 2.0, "rmse": 2.0, "mape": 2.0, "r2": 0.4}),
    ]
    selection = rank_candidates(candidates, "price")

    assert "mape" not in selection.metrics_used
    assert selection.winner.model_type == "nan_mape"


def test_module_defaults_point_at_the_real_artifact_locations():
    """A typo in a default path would silently produce "no model" everywhere."""
    assert model_selection.BENCHMARK_ARTIFACT.name == "benchmark_results.json"
    assert model_selection.BENCHMARK_ARTIFACT.parent.name == "artifacts"
    assert model_selection.BUNDLES_DIR.name == "bundles"
