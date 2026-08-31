"""
Walk-forward evaluation of the next-day direction classifier.

This module is the honest-evaluation harness. It owns the one thing that
decides whether any of the numbers downstream mean anything: what each model was
allowed to see, and when.

Per fold, in order:

1. Split the training window into inner-train and an inner-validation tail,
   with the same embargo that separates train from test.
2. Fit a *throwaway* copy of the model on inner-train only, and pick the trading
   threshold on its inner-validation predictions. Those predictions are
   out-of-sample, which is why the threshold is not chosen on a model's own
   training fit — that would pick the threshold that best fits noise the final
   model has already memorised.
3. Fit the real model on the whole training window, predict on the test window.
4. Fit every baseline on the same training window and predict on the same test
   window, so the comparison is like for like rather than a remembered number.
5. Backtest the test window at the threshold from step 2.

Nothing in steps 1-2 touches a test row, and nothing in step 3-5 refits. The
test window is used exactly once, for scoring.

The folds are then pooled. Pooling is legitimate here because the test windows
are disjoint, contiguous, and chronological, so concatenating them reproduces
one continuous out-of-sample track record — which is also the only sample large
enough for an accuracy edge to clear its own standard error.

Public API:
    run_walk_forward(...) -> DirectionRunResult
    run_shuffled_label_check(...) -> dict
    predict_next_session(...) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..backtesting.direction_backtest import (
    DEFAULT_THRESHOLD_GRID,
    run_long_flat_backtest,
    select_threshold,
)
from ..features.direction_features import (
    DirectionDataset,
    build_direction_dataset,
    latest_feature_row,
)
from ..utils.logger import get_logger
from .direction_bands import (
    BAND_QUANTILES,
    ConditionalReturnBand,
    band_metrics,
    volatility_for,
)
from .direction_metrics import accuracy_edge_test, classification_metrics, one_sided_p_value
from .direction_models import BASELINE_FACTORIES, build_model
from .walk_forward import expanding_window_splits

logger = get_logger(__name__)

DEFAULT_TEST_SIZE = 63          # one trading quarter
DEFAULT_N_FOLDS = 4
DEFAULT_MIN_TRAIN = 252         # one trading year
DEFAULT_COST_BPS = 10.0
DEFAULT_SEED = 42

# Share of a training window reserved, at its end, for threshold selection.
THRESHOLD_VALIDATION_FRACTION = 0.25
MIN_THRESHOLD_VALIDATION_ROWS = 60
# ...and a ceiling on it. A quarter of a ten-year training fold is 600 rows,
# which is free for a tree and hours of forward passes for a sequence model.
# One year is also the better statistic: recent bars resemble the test window
# more closely than bars from five years earlier, and the threshold is being
# fitted to a regime, not to all of history.
MAX_THRESHOLD_VALIDATION_ROWS = 252

# Significance level at which the shuffled-label check declares leakage.
LEAKAGE_ALPHA = 0.01
# Refits per fold in the leakage check. Each shuffled fit lands an arbitrary
# coefficient vector, so a single fit's accuracy swings several points around
# the null; the check averages over fits and needs enough of them for that mean
# to settle. Measured fit-to-fit standard deviation is ~4pp, so 10 repeats over
# 4 folds (40 fits) puts the standard error of the mean near 0.6pp. Three
# repeats is not enough, and produces false alarms on clean data.
DEFAULT_LEAKAGE_REPEATS = 10
MIN_LEAKAGE_FITS = 20


@dataclass
class DirectionRunResult:
    """Everything one walk-forward run produced."""

    report: Dict[str, Any]
    equity_curve: pd.DataFrame
    predictions: pd.DataFrame

    @property
    def ship(self) -> bool:
        return bool(self.report.get("verdict", {}).get("ship", False))


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars so json.dump does not choke."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _threshold_validation_split(
    train_positions: np.ndarray, embargo: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Carve an inner-validation tail out of a training window, embargo included.

    Returns None when the window cannot spare a validation block big enough to
    say anything about a threshold; the caller then falls back to a fixed
    threshold rather than fitting one to a dozen rows.
    """
    n = len(train_positions)
    n_validation = int(round(n * THRESHOLD_VALIDATION_FRACTION))
    n_validation = min(n_validation, MAX_THRESHOLD_VALIDATION_ROWS)
    if n_validation < MIN_THRESHOLD_VALIDATION_ROWS:
        return None
    validation_start = n - n_validation
    inner_train_end = validation_start - int(embargo)
    if inner_train_end < MIN_THRESHOLD_VALIDATION_ROWS:
        return None
    return train_positions[:inner_train_end], train_positions[validation_start:]


def _score(
    dataset: DirectionDataset,
    positions: np.ndarray,
    estimator,
    *,
    reference_accuracy: float,
    reference_rate: float,
) -> Dict[str, Any]:
    """Fit-free scoring of an already-fitted estimator on a positional slice."""
    window = dataset.slice(positions)
    probabilities = estimator.predict_proba_up(window.features)
    predictions = estimator.predict(window.features)
    metrics = classification_metrics(
        window.labels.to_numpy(), predictions, probabilities,
        reference_accuracy=reference_accuracy, reference_rate=reference_rate,
    )
    metrics["_probabilities"] = probabilities
    metrics["_predictions"] = predictions
    price_bands = getattr(estimator, "price_bands_", None)
    if price_bands is not None and len(price_bands) == len(positions):
        metrics["_price_bands"] = price_bands
    return metrics


def _fit_price_band(
    dataset: DirectionDataset,
    model,
    train: DirectionDataset,
    test: DirectionDataset,
) -> Optional[np.ndarray]:
    """
    Derive a price band for a model that only emits a probability.

    The band is fitted on the training fold and applied to the test fold, like
    every other fitted object here. It is fitted against the model's *training*
    probabilities, which are in-sample and therefore a little sharper than the
    test probabilities it will be applied to. That would matter if the bucket
    conditioning carried the band, but it does not: the width comes from today's
    trailing volatility, and the buckets only skew it. The honest check is the
    coverage number in the report, which is measured out-of-sample.

    Returns None when the model already produced its own band (Kronos) or the
    inputs are unusable.
    """
    try:
        volatility_train = volatility_for(train.features, dataset.ohlcv)
        volatility_test = volatility_for(test.features, dataset.ohlcv)
    except KeyError as exc:
        logger.warning("No price band: %s", exc)
        return None

    if dataset.ohlcv is None or "Close" not in dataset.ohlcv.columns:
        logger.warning("No price band: the dataset carries no OHLCV close to anchor it to")
        return None

    try:
        estimator = ConditionalReturnBand().fit(
            model.predict_proba_up(train.features),
            train.forward_return.to_numpy(),
            volatility_train.to_numpy(),
        )
        last_close = dataset.ohlcv["Close"].reindex(test.index).to_numpy()
        return estimator.predict(
            model.predict_proba_up(test.features), last_close, volatility_test.to_numpy()
        )
    except Exception as exc:  # noqa: BLE001 - a missing band must not kill the fold
        logger.warning("Price band could not be fitted: %s", exc)
        return None


def _strip_arrays(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the raw prediction arrays before a metrics dict goes into the report."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


def run_walk_forward(
    dataset: DirectionDataset,
    *,
    model_name: str = "logistic",
    n_folds: int = DEFAULT_N_FOLDS,
    test_size: int = DEFAULT_TEST_SIZE,
    embargo: Optional[int] = None,
    min_train: int = DEFAULT_MIN_TRAIN,
    cost_bps: float = DEFAULT_COST_BPS,
    threshold_grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    threshold_objective: str = "sharpe",
    fixed_threshold: Optional[float] = None,
    seed: int = DEFAULT_SEED,
    risk_free_rate: float = 0.0,
    data_meta: Optional[Dict[str, Any]] = None,
    run_leakage_check: bool = True,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> DirectionRunResult:
    """
    Evaluate one model across expanding-window folds and pool the results.

    Parameters
    ----------
    dataset : DirectionDataset
    model_name : str
        ``logistic``, ``gradient_boosting``, or any baseline name — baselines
        run through the identical harness so their numbers are produced the same
        way as the model's.
    n_folds, test_size, min_train : int
        Fold geometry. ``test_size`` is fixed so per-fold confidence intervals
        are comparable.
    embargo : int, optional
        Rows purged between train and test. Defaults to the dataset's horizon,
        which is the smallest gap that stops the last training label resolving
        inside the test window.
    cost_bps : float
        Round-trip cost charged on every active day.
    fixed_threshold : float, optional
        Skip validation-based threshold selection and use this value. Provided
        for reproducing a specific configuration; the tuned path is the default
        because a hand-picked threshold is a free parameter nobody accounted for.
    run_leakage_check : bool
        Run the shuffled-label check after the folds. On by default.
    model_kwargs : dict, optional
        Options forwarded to the model under test — Kronos' ``sample_count`` and
        ``lookback``, for instance. Baselines never receive them, so the
        comparison rows stay fixed while the model is varied.

    Returns
    -------
    DirectionRunResult
    """
    model_kwargs = dict(model_kwargs or {})
    horizon = int(dataset.meta.get("horizon", 1))
    embargo = horizon if embargo is None else int(embargo)
    if embargo < horizon:
        raise ValueError(
            f"embargo={embargo} is smaller than the target horizon={horizon}; the last "
            f"training label would resolve inside the test window"
        )

    splits = expanding_window_splits(
        len(dataset), test_size=test_size, n_splits=n_folds,
        embargo=embargo, min_train=min_train,
    )
    if not splits:
        raise ValueError(
            f"{len(dataset)} rows cannot support {n_folds} folds of {test_size} test rows "
            f"with min_train={min_train} and embargo={embargo}; "
            f"at least {min_train + embargo + test_size} rows are needed for one fold"
        )

    index = dataset.index
    fold_reports: List[Dict[str, Any]] = []
    pooled_rows: List[pd.DataFrame] = []
    pooled_thresholds: List[np.ndarray] = []
    pooled_baseline_probabilities: Dict[str, List[np.ndarray]] = {n: [] for n in BASELINE_FACTORIES}
    pooled_baseline_predictions: Dict[str, List[np.ndarray]] = {n: [] for n in BASELINE_FACTORIES}
    pooled_reference_rates: List[np.ndarray] = []

    for fold_number, (train_positions, test_positions) in enumerate(splits, start=1):
        train = dataset.slice(train_positions)
        test = dataset.slice(test_positions)
        train_base_rate = float(train.labels.mean())
        # The accuracy a constant majority-class predictor fitted on this
        # training window would score on this test window. This is the number
        # the model has to beat; the test window's own base rate is not
        # knowable in advance, so it is not a fair reference.
        majority_label = int(train_base_rate >= 0.5)
        majority_accuracy = float((test.labels.to_numpy() == majority_label).mean())

        # -- threshold, chosen out-of-sample inside the training window --------
        if fixed_threshold is not None:
            threshold, threshold_report = float(fixed_threshold), {
                "threshold": float(fixed_threshold), "source": "fixed", "candidates": [],
            }
        else:
            inner = _threshold_validation_split(train_positions, embargo)
            if inner is None:
                threshold = float(np.median(threshold_grid))
                threshold_report = {
                    "threshold": threshold,
                    "source": "default_median_grid",
                    "reason": "training window too short for a threshold-validation block",
                    "candidates": [],
                }
            else:
                inner_train_positions, validation_positions = inner
                inner_train = dataset.slice(inner_train_positions)
                validation = dataset.slice(validation_positions)
                probe = build_model(model_name, seed=seed, **model_kwargs)
                if hasattr(probe, "set_ohlcv_context") and dataset.ohlcv is not None:
                    probe.set_ohlcv_context(dataset.ohlcv)
                probe.fit(inner_train.features, inner_train.labels)
                choice = select_threshold(
                    probe.predict_proba_up(validation.features),
                    validation.entry_open.to_numpy(),
                    validation.exit_close.to_numpy(),
                    cost_bps=cost_bps, grid=threshold_grid, objective=threshold_objective,
                )
                threshold = choice.threshold
                threshold_report = {
                    "threshold": choice.threshold,
                    "source": "validation",
                    "objective": choice.objective,
                    "score": choice.score,
                    "fell_back": choice.fell_back,
                    "n_validation_rows": int(len(validation_positions)),
                    "validation_range": [str(validation.index[0].date()), str(validation.index[-1].date())],
                    "candidates": choice.candidates,
                }

        # -- the real fit, on the full training window -------------------------
        model = build_model(model_name, seed=seed, **model_kwargs)
        if hasattr(model, "set_ohlcv_context") and dataset.ohlcv is not None:
            model.set_ohlcv_context(dataset.ohlcv)
        model.fit(train.features, train.labels)
        model_metrics = _score(
            dataset, test_positions, model,
            reference_accuracy=majority_accuracy, reference_rate=train_base_rate,
        )

        baseline_metrics: Dict[str, Any] = {}
        for baseline_name in BASELINE_FACTORIES:
            baseline = build_model(baseline_name, seed=seed).fit(train.features, train.labels)
            scored = _score(
                dataset, test_positions, baseline,
                reference_accuracy=majority_accuracy, reference_rate=train_base_rate,
            )
            pooled_baseline_probabilities[baseline_name].append(scored["_probabilities"])
            pooled_baseline_predictions[baseline_name].append(scored["_predictions"])
            baseline_metrics[baseline_name] = _strip_arrays(scored)

        # -- backtest the test window at the validated threshold ---------------
        backtest = run_long_flat_backtest(
            model_metrics["_probabilities"],
            test.entry_open.to_numpy(),
            test.exit_close.to_numpy(),
            threshold=threshold, cost_bps=cost_bps, index=test.index,
            risk_free_rate=risk_free_rate,
        )

        fold_frame = backtest.equity_curve.copy()
        fold_frame.insert(0, "fold", fold_number)
        fold_frame["label"] = test.labels.to_numpy()
        fold_frame["prediction"] = model_metrics["_predictions"]
        fold_frame["forward_return_close_to_close"] = test.forward_return.to_numpy()

        # Kronos samples whole candles, so it hands back its own band. Every
        # other model gets one derived from its probability and today's
        # volatility, so the report always carries a range and never a bare
        # point forecast.
        bands = model_metrics.get("_price_bands")
        band_source = "model_samples"
        if bands is None:
            bands = _fit_price_band(dataset, model, train, test)
            band_source = "conditional_return_quantiles"
        fold_band_metrics: Optional[Dict[str, Any]] = None
        if bands is not None:
            fold_frame["price_lo_5"] = bands[:, 0]
            fold_frame["price_median"] = bands[:, 1]
            fold_frame["price_hi_95"] = bands[:, 2]
            fold_band_metrics = band_metrics(bands, test.exit_close.to_numpy())
            fold_band_metrics["source"] = band_source
        pooled_rows.append(fold_frame)
        pooled_thresholds.append(np.full(len(test_positions), threshold, dtype=np.float64))
        pooled_reference_rates.append(np.full(len(test_positions), train_base_rate, dtype=np.float64))

        fold_reports.append({
            "fold": fold_number,
            "n_train": int(len(train_positions)),
            "n_test": int(len(test_positions)),
            "train_range": [str(index[train_positions[0]].date()), str(index[train_positions[-1]].date())],
            "test_range": [str(index[test_positions[0]].date()), str(index[test_positions[-1]].date())],
            "embargo_rows": embargo,
            "train_base_rate": round(train_base_rate, 6),
            "test_base_rate": round(float(test.labels.mean()), 6),
            "majority_baseline_accuracy": round(majority_accuracy, 6),
            "threshold": threshold_report,
            "model": _strip_arrays(model_metrics),
            "model_fit_info": model.fit_info_,
            "price_band": fold_band_metrics,
            "baselines": baseline_metrics,
            "backtest": {
                "strategy": backtest.metrics,
                "benchmark": backtest.benchmark_metrics,
                "breakeven": backtest.breakeven,
            },
        })
        logger.info(
            "fold %d/%d  test %s..%s  acc=%.4f (majority %.4f)  threshold=%.2f  "
            "net=%.2f%% vs B&H %.2f%%",
            fold_number, len(splits), fold_reports[-1]["test_range"][0],
            fold_reports[-1]["test_range"][1], model_metrics["accuracy"], majority_accuracy,
            threshold, 100 * backtest.metrics["total_return"],
            100 * backtest.benchmark_metrics["total_return"],
        )

    # -- pool the folds into one continuous out-of-sample record ---------------
    # Fold order is already chronological and the test windows are disjoint, so
    # plain concatenation IS the time-ordered pooled record. It is not re-sorted:
    # the pooled baseline arrays below are concatenated in the same fold order,
    # and a re-sort here would silently break that alignment.
    pooled = pd.concat(pooled_rows)
    if not pooled.index.is_monotonic_increasing:
        raise RuntimeError("Pooled fold index is not chronological; the splitter is misbehaving")
    thresholds = np.concatenate(pooled_thresholds)
    reference_rates = np.concatenate(pooled_reference_rates)
    pooled_labels = pooled["label"].to_numpy()
    pooled_probabilities = pooled["probability_up"].to_numpy()
    pooled_predictions = pooled["prediction"].to_numpy()

    # Same reference as per fold: the majority class each fold's training window
    # implied, evaluated over the whole pooled window.
    pooled_majority_predictions = (reference_rates >= 0.5).astype(np.int8)
    pooled_majority_accuracy = float((pooled_majority_predictions == pooled_labels).mean())

    pooled_model_metrics = classification_metrics(
        pooled_labels, pooled_predictions, pooled_probabilities,
        reference_accuracy=pooled_majority_accuracy, reference_rate=reference_rates,
    )

    pooled_baseline_report: Dict[str, Any] = {}
    for baseline_name in BASELINE_FACTORIES:
        probabilities = np.concatenate(pooled_baseline_probabilities[baseline_name])
        predictions = np.concatenate(pooled_baseline_predictions[baseline_name])
        pooled_baseline_report[baseline_name] = classification_metrics(
            pooled_labels, predictions, probabilities,
            reference_accuracy=pooled_majority_accuracy, reference_rate=reference_rates,
        )

    pooled_backtest = run_long_flat_backtest(
        pooled_probabilities,
        pooled["entry_open"].to_numpy(),
        pooled["exit_close"].to_numpy(),
        threshold=thresholds, cost_bps=cost_bps, index=pooled.index,
        risk_free_rate=risk_free_rate,
    )

    pooled_band: Optional[Dict[str, Any]] = None
    if {"price_lo_5", "price_median", "price_hi_95"}.issubset(pooled.columns):
        pooled_band = band_metrics(
            pooled[["price_lo_5", "price_median", "price_hi_95"]].to_numpy(),
            pooled["exit_close"].to_numpy(),
        )
        pooled_band["quantiles"] = list(BAND_QUANTILES)

    best_baseline_name = max(
        pooled_baseline_report, key=lambda k: pooled_baseline_report[k]["accuracy"]
    )
    best_baseline_accuracy = pooled_baseline_report[best_baseline_name]["accuracy"]
    edge_vs_best_baseline = accuracy_edge_test(
        pooled_model_metrics["accuracy"], best_baseline_accuracy, len(pooled_labels)
    )

    leakage = None
    if run_leakage_check and not _learns_from_labels(model_name, seed, model_kwargs):
        leakage = {
            "passed": None,
            "applicable": False,
            "note": (
                f"not applicable: '{model_name}' is pre-trained and never fits on y, so "
                f"permuting training labels cannot change its predictions. A pass here "
                f"would assert something the test never examined. Run the check against "
                f"a slot that does learn from labels (logistic, gradient_boosting, "
                f"tabpfn) to validate the features and the splitter."
            ),
        }
        logger.info(
            "Skipping the shuffled-label check: %s does not learn from labels", model_name
        )
    elif run_leakage_check:
        # Enough repeats that the mean shuffled accuracy has settled even when
        # the geometry only allowed one or two folds; otherwise the check comes
        # back inconclusive through no fault of the data.
        leakage_repeats = max(
            DEFAULT_LEAKAGE_REPEATS,
            int(np.ceil(MIN_LEAKAGE_FITS / max(1, len(splits)))),
        )
        leakage = run_shuffled_label_check(
            dataset, splits, model_name=model_name, seed=seed,
            n_repeats=leakage_repeats, model_kwargs=model_kwargs,
        )

    verdict = _build_verdict(
        pooled_model_metrics, edge_vs_best_baseline, pooled_backtest, cost_bps, leakage
    )

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "model": model_name,
            "n_folds_requested": n_folds,
            "n_folds_run": len(splits),
            "test_size": test_size,
            "min_train": min_train,
            "embargo": embargo,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "risk_free_rate": risk_free_rate,
            "threshold_grid": list(threshold_grid),
            "threshold_objective": threshold_objective,
            "fixed_threshold": fixed_threshold,
            "seed": seed,
            "model_kwargs": model_kwargs,
            "n_features": int(len(dataset.feature_columns)),
            "split_scheme": "expanding_window_walk_forward",
            "execution": "signal at close(t); enter open(t+1); exit close(t+horizon)",
        },
        "data": data_meta or {},
        "dataset": dataset.meta,
        "folds": fold_reports,
        "pooled": {
            "n_test_rows": int(len(pooled_labels)),
            "test_range": [str(pooled.index[0].date()), str(pooled.index[-1].date())],
            "majority_baseline_accuracy": round(pooled_majority_accuracy, 6),
            "model": pooled_model_metrics,
            "baselines": pooled_baseline_report,
            "best_baseline": best_baseline_name,
            "edge_vs_best_baseline": edge_vs_best_baseline,
            "price_band": pooled_band,
            "backtest": {
                "strategy": pooled_backtest.metrics,
                "benchmark": pooled_backtest.benchmark_metrics,
                "breakeven": pooled_backtest.breakeven,
                "config": pooled_backtest.config,
            },
        },
        "leakage_check": leakage,
        "verdict": verdict,
    }

    pred_cols = [
        "fold", "probability_up", "threshold", "position", "prediction", "label",
        "forward_return_close_to_close", "gross_return", "net_return",
    ]
    for band_col in ["price_lo_5", "price_median", "price_hi_95"]:
        if band_col in pooled.columns:
            pred_cols.append(band_col)

    predictions_frame = pooled[pred_cols].copy()

    return DirectionRunResult(
        report=_jsonable(report),
        equity_curve=pooled_backtest.equity_curve,
        predictions=predictions_frame,
    )


def _learns_from_labels(
    model_name: str, seed: int, model_kwargs: Dict[str, Any]
) -> bool:
    """
    Does this estimator derive anything from ``y``?

    Constructing it is the only reliable way to ask, since the answer can depend
    on which sub-models an ensemble managed to initialise. A construction
    failure is reported as "yes" so the check still runs and the real error
    surfaces from the fold loop rather than being swallowed here.
    """
    try:
        return bool(getattr(build_model(model_name, seed=seed, **model_kwargs),
                            "learns_from_labels", True))
    except Exception:  # noqa: BLE001 - the fold loop reports the real failure
        return True


def _build_verdict(
    model_metrics: Dict[str, Any],
    edge: Dict[str, Any],
    backtest,
    cost_bps: float,
    leakage: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ship or do not ship, with the reason spelled out per criterion.

    Every criterion has to pass. They are not redundant: a model can beat the
    baselines on accuracy and still lose money after costs (it is right on small
    moves and wrong on large ones), and it can make money in the backtest purely
    because the underlying went up (which buy & hold already captured, for one
    round trip instead of hundreds).
    """
    breakeven = backtest.breakeven.get("breakeven_cost_bps_positive")
    criteria = {
        "beats_best_baseline_accuracy": bool(edge.get("edge_pp", 0) > 0),
        "accuracy_edge_is_significant": bool(edge.get("significant", False)),
        "positive_probability_skill": bool(
            model_metrics.get("skill", {}).get("brier_skill_score", 0) > 0
        ),
        "beats_buy_and_hold_after_costs": bool(
            backtest.metrics.get("total_return", 0) > backtest.benchmark_metrics.get("total_return", 0)
        ),
        "survives_the_charged_cost": bool(breakeven is not None and breakeven > float(cost_bps)),
        # `passed` is None when the check could not gather enough fits to judge.
        # That is not evidence of leakage, so it does not fail the criterion, but
        # it is surfaced below so nobody reads silence as a clean bill of health.
        "passes_leakage_check": (leakage.get("passed") is not False) if leakage else True,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    inconclusive_leakage = bool(
        leakage is not None
        and leakage.get("passed") is None
        and leakage.get("applicable", True)
    )
    return {
        "ship": not failed,
        "criteria": criteria,
        "failed_criteria": failed,
        "leakage_check_inconclusive": inconclusive_leakage,
        "summary": (
            "Meets every ship criterion out-of-sample after costs."
            if not failed else
            "fails " + ", ".join(failed) + ". "
            "The honest reading of a failure here is that next-day direction is "
            "not learnable from this feature set on this ticker, which is a "
            "result, not a bug to patch."
        ),
    }


def run_shuffled_label_check(
    dataset: DirectionDataset,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    model_name: str = "logistic",
    seed: int = DEFAULT_SEED,
    n_repeats: int = DEFAULT_LEAKAGE_REPEATS,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    The leakage test: shuffle ``y`` within each training window and refit.

    Shuffling destroys every relationship between features and labels while
    leaving the feature matrix, the class balance, and the fold geometry
    untouched. A model fitted on that has nothing to learn, so its test accuracy
    must collapse to chance. If it does not — if permuted labels still predict
    the test window — then information about the test period is reaching the
    model through the features, the scaler, the split, or the target
    construction, and every other number in the report is void.

    Two details decide whether this check is worth running at all.

    **The right null.** Not the majority baseline: when a test window's base rate
    falls below 50% the majority baseline scores below 50% too, so a coin flip
    beats it and the check fires on clean data. The null is the accuracy of a
    predictor independent of the truth that predicts up at the same rate ``q``
    the shuffled fit happened to::

        E[accuracy | no relationship] = q * p + (1 - q) * (1 - p)

    with ``p`` the test base rate. That is the base rate when q is 1, 1 - p when
    q is 0, and 0.5 when q is 0.5 — "collapses to the base rate", generalised to
    any prediction rate.

    **The right denominator.** Each shuffled fit lands an essentially arbitrary
    coefficient vector, so its accuracy scatters around the null far more widely
    than binomial noise on the test rows would suggest, and repeats re-score the
    *same* rows, so they are not independent observations of those rows. The
    statistic is therefore a paired one-sample test on the per-fit differences
    ``accuracy - null``, with the standard error measured from the fit-to-fit
    spread rather than assumed from the row count. Using the pooled row count
    instead overstates the evidence by roughly sqrt(n_repeats).
    """
    model_kwargs = dict(model_kwargs or {})
    rng = np.random.default_rng(seed)
    per_run: List[Dict[str, Any]] = []
    differences: List[float] = []
    accuracies: List[float] = []
    null_accuracies: List[float] = []

    for repeat in range(int(n_repeats)):
        for fold_number, (train_positions, test_positions) in enumerate(splits, start=1):
            train = dataset.slice(train_positions)
            test = dataset.slice(test_positions)
            truth = test.labels.to_numpy()

            shuffled = train.labels.to_numpy().copy()
            rng.shuffle(shuffled)

            model = build_model(model_name, seed=seed, **model_kwargs)
            if hasattr(model, "set_ohlcv_context") and dataset.ohlcv is not None:
                model.set_ohlcv_context(dataset.ohlcv)
            model.fit(train.features, shuffled)
            predictions = model.predict(test.features)
            accuracy = float((predictions == truth).mean())

            predicted_up_rate = float(np.mean(predictions))
            test_base_rate = float(np.mean(truth))
            null_accuracy = (
                predicted_up_rate * test_base_rate
                + (1.0 - predicted_up_rate) * (1.0 - test_base_rate)
            )

            accuracies.append(accuracy)
            null_accuracies.append(null_accuracy)
            differences.append(accuracy - null_accuracy)
            per_run.append({
                "repeat": repeat + 1,
                "fold": fold_number,
                "shuffled_accuracy": round(accuracy, 6),
                "null_accuracy": round(null_accuracy, 6),
                "difference": round(accuracy - null_accuracy, 6),
                "predicted_up_rate": round(predicted_up_rate, 6),
                "test_base_rate": round(test_base_rate, 6),
            })

    n_fits = len(differences)
    diff_array = np.asarray(differences, dtype=np.float64)
    mean_difference = float(np.mean(diff_array)) if n_fits else float("nan")
    # ddof=1: the fit-to-fit spread is estimated from the fits themselves.
    spread = float(np.std(diff_array, ddof=1)) if n_fits > 1 else 0.0
    standard_error = spread / np.sqrt(n_fits) if n_fits > 1 and spread > 0 else 0.0

    if n_fits < MIN_LEAKAGE_FITS:
        # Too few fits for the mean to have settled. Refusing to judge is the
        # honest outcome; a pass claimed here would be the same false comfort as
        # a failure claimed here would be a false alarm.
        z, p_value, passed = None, None, None
        note = (
            f"inconclusive: {n_fits} fits is below the {MIN_LEAKAGE_FITS} needed for the "
            f"mean shuffled accuracy to settle; raise n_repeats"
        )
    elif standard_error == 0:
        z, p_value = None, None
        passed = bool(mean_difference <= 0)
        note = "every shuffled fit produced an identical difference"
    else:
        z = mean_difference / standard_error
        p_value = one_sided_p_value(z)
        passed = bool(p_value >= LEAKAGE_ALPHA)
        note = "paired one-sample test on per-fit (accuracy - null) differences"

    if passed is False:
        logger.error(
            "LEAKAGE CHECK FAILED: shuffled labels still beat chance by %+.4f "
            "(z=%.2f, p=%s over %d fits). Every other number in this report is void.",
            mean_difference, z, p_value, n_fits,
        )
    elif passed is None:
        logger.warning("Leakage check inconclusive: %s", note)
    else:
        logger.info(
            "Leakage check passed: shuffled labels beat chance by %+.4f "
            "(p=%s over %d fits) — consistent with no relationship",
            mean_difference, p_value, n_fits,
        )

    return {
        "passed": passed,
        "method": (
            "labels permuted within each training window; features, folds, and class "
            "balance unchanged. Null is the accuracy expected of a predictor independent "
            "of the labels that predicts up at the same rate; the test is paired over fits."
        ),
        "note": note,
        "n_repeats": int(n_repeats),
        "n_fits": n_fits,
        "mean_shuffled_accuracy": round(float(np.mean(accuracies)), 6) if n_fits else None,
        "mean_null_accuracy": round(float(np.mean(null_accuracies)), 6) if n_fits else None,
        "mean_difference": round(mean_difference, 6) if n_fits else None,
        "difference_std_across_fits": round(spread, 6),
        "standard_error_of_mean": round(standard_error, 6),
        "z": round(z, 4) if z is not None else None,
        "p_value_one_sided": round(p_value, 6) if p_value is not None else None,
        "alpha": LEAKAGE_ALPHA,
        "runs": per_run,
    }


def predict_next_session(
    bars: pd.DataFrame,
    *,
    model_name: str = "logistic",
    seed: int = DEFAULT_SEED,
    dataset: Optional[DirectionDataset] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    P(up) for the session after the last printed close.

    Fits ``model_name`` on every labelled row and applies it to the one bar the
    dataset had to drop — the latest close, whose next-day move has not happened
    yet. Using the full history is correct here and only here: this is not an
    evaluation, it is the live call, and withholding recent data from it would
    make the served prediction worse than the one the backtest measured.

    The number this returns is **not** on its own a reason to trade. Whether it
    means anything is decided by the walk-forward verdict, which is why the API
    layer serves the two together and refuses to show a bare gauge for a model
    that failed its ship criteria.

    Returns None when no bar has a complete feature vector.
    """
    dataset = dataset if dataset is not None else build_direction_dataset(bars)
    found = latest_feature_row(bars, feature_columns=dataset.feature_columns)
    if found is None:
        return None
    as_of, feature_row = found

    model = build_model(model_name, seed=seed, **dict(model_kwargs or {}))
    if hasattr(model, "set_ohlcv_context"):
        ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in bars.columns]
        if ohlcv_cols:
            model.set_ohlcv_context(bars[ohlcv_cols])
        elif dataset.ohlcv is not None:
            model.set_ohlcv_context(dataset.ohlcv)
    model.fit(dataset.features, dataset.labels)
    probability = float(model.predict_proba_up(feature_row)[0])

    result: Dict[str, Any] = {
        "as_of": str(pd.Timestamp(as_of).date()),
        "model": model_name,
        "probability_up": round(probability, 6),
        "predicted_direction": "up" if probability >= 0.5 else "down",
        "n_train_rows": int(len(dataset)),
        "train_base_rate": round(float(dataset.base_rate), 6),
        # How far the probability sits from simply repeating the training base
        # rate. Near zero means the features moved the model not at all.
        "edge_over_base_rate_pp": round((probability - float(dataset.base_rate)) * 100, 4),
    }

    bands = getattr(model, "price_bands_", None)
    band_source = "model_samples"
    if bands is None or len(bands) == 0 or np.isnan(bands[0][0]):
        # Same construction the walk-forward folds use, fitted on the whole
        # labelled history and applied to the one unlabelled bar. A probability
        # on its own is not a price, and the request is for a range.
        bands, band_source = _live_price_band(bars, dataset, model, as_of, feature_row),             "conditional_return_quantiles"

    if bands is not None and len(bands) > 0 and np.isfinite(bands[0]).all():
        result["price_forecast"] = {
            "price_lo_5": round(float(bands[0][0]), 4),
            "price_median": round(float(bands[0][1]), 4),
            "price_hi_95": round(float(bands[0][2]), 4),
            "quantiles": list(BAND_QUANTILES),
            "source": band_source,
        }

    return result


def _live_price_band(
    bars: pd.DataFrame,
    dataset: DirectionDataset,
    model,
    as_of: pd.Timestamp,
    feature_row: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Price band for the one bar that has no label yet.

    Anchored on the close of ``as_of`` and scaled by the volatility known at
    that close, so it is the same object the backtest reported coverage for —
    which is what makes the reported coverage a claim about *this* number.
    """
    try:
        volatility = volatility_for(dataset.features, dataset.ohlcv)
        estimator = ConditionalReturnBand().fit(
            model.predict_proba_up(dataset.features),
            dataset.forward_return.to_numpy(),
            volatility.to_numpy(),
        )
    except Exception as exc:  # noqa: BLE001 - a missing band must not drop the gauge
        logger.warning("Live price band could not be fitted: %s", exc)
        return None

    if "Volatility" not in feature_row.columns:
        logger.warning("Live price band needs a Volatility feature column")
        return None

    close_series = bars["Close"] if "Close" in bars.columns else None
    if close_series is None or as_of not in close_series.index:
        return None

    return estimator.predict(
        model.predict_proba_up(feature_row),
        [float(close_series.loc[as_of])],
        [float(feature_row["Volatility"].iloc[0])],
    )
