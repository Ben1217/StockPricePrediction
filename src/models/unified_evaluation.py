"""
Walk-forward evaluation for the unified models.

One function scores every model in the comparison, so no model can benefit from
a different split, a different target, or a different metric implementation.

Two objectives, kept apart
--------------------------
Price forecasting is scored with MAE / RMSE / MAPE / R-squared; direction with
accuracy, precision, recall, F1 and ROC-AUC. They are reported side by side and
never merged into one number: a model can nail the level and still be a coin
flip on the sign, and the sign is what a trade is placed on.

Sharpe ratio is deliberately absent. It measures a *strategy* -- entry rule,
sizing, costs -- not a forecast, and using it to pick a forecasting model
rewards whichever model happens to suit the position-sizing rule in front of it.
The backtesting layer is where Sharpe belongs.

On the price R-squared
----------------------
It is computed against the realised price, whose variance is dominated by the
price *level*. Any model that predicts "roughly today's close" scores near 1.0.
It is reported because the brief asks for it, but ``price_r2_return`` -- the
same statistic on the forward return, where the naive forecast scores 0.0 -- is
what actually separates the models, so both are reported.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.evaluation.economics import (
    CostModel,
    breakeven_round_trip_bps,
    long_flat_positions,
    paper_trading_overlay,
)
from src.evaluation.metrics import directional_metrics, probabilistic_metrics
from src.evaluation.splitting import (
    describe_split,
    effective_sample_size,
    purged_walk_forward_splits,
)
from src.evaluation.testing import diebold_mariano_test
from src.models.direction_metrics import accuracy_edge_test, classification_metrics
from src.models.unified_models import UnifiedEstimator, prepare_fold
from src.utils.logger import get_logger

logger = get_logger(__name__)


def price_metrics(y_true: np.ndarray, y_pred: np.ndarray, prev_close: np.ndarray) -> Dict[str, float]:
    """
    Regression metrics on the predicted price, plus the return-space R-squared.

    ``prev_close`` converts both series back into returns so the R-squared can
    be recomputed on the quantity that actually has to be forecast. A model
    scoring below zero there is worse than predicting no change at all.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    prev_close = np.asarray(prev_close, dtype=np.float64)

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    nonzero = y_true != 0
    mape = (
        float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0)
        if nonzero.any()
        else float("nan")
    )

    usable = prev_close != 0
    if usable.sum() > 1:
        true_return = y_true[usable] / prev_close[usable] - 1.0
        predicted_return = y_pred[usable] / prev_close[usable] - 1.0
        r2_return = float(r2_score(true_return, predicted_return))
    else:
        r2_return = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "r2_return": r2_return}


def _flatten_direction(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Lift the up-class precision/recall/F1 out of the nested scorecard into flat columns."""
    flat: Dict[str, float] = {}
    class_up = metrics.get("class_up") or {}
    for key in ("precision", "recall", "f1"):
        value = class_up.get(key)
        if value is not None:
            flat[f"direction_{key}"] = float(value)
    for key in ("accuracy", "balanced_accuracy", "roc_auc", "brier_score"):
        value = metrics.get(key)
        if value is not None:
            flat[f"direction_{key}"] = float(value)
    return flat


def _native(value: Any) -> Any:
    """
    Numpy scalars converted to the Python types ``json`` can write.

    Without this the scorecard serialises through ``default=str`` and a
    ``float64`` accuracy lands in the artifact as the *string* ``"0.412698"``,
    which every downstream reader then has to guess at. Non-finite floats become
    ``null`` for the same reason: ``NaN`` is not JSON, and a reader that accepts
    it is accepting something no other parser will.
    """
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _series(values: np.ndarray, digits: int) -> List[Optional[float]]:
    """One per-bar column, rounded and JSON-safe."""
    return [
        round(float(item), digits) if math.isfinite(float(item)) else None
        for item in np.asarray(values, dtype=np.float64)
    ]


def evaluate_unified_walk_forward(
    model: UnifiedEstimator,
    X: pd.DataFrame,
    y_return: np.ndarray,
    y_direction: np.ndarray,
    prev_close: np.ndarray,
    *,
    test_size: int = 63,
    n_splits: int = 4,
    embargo: Optional[int] = None,
    min_train: int = 252,
    horizon: int = 1,
) -> Dict[str, Any]:
    """
    Score one model with purged, embargoed expanding-window walk-forward validation.

    Chronological order is preserved throughout: every fold trains on a prefix
    of the series and tests on the block that follows it. No shuffling, anywhere.

    The gap between the two is two blocks, not one. ``horizon`` rows are
    *purged* off the training tail because their labels resolve at or after the
    test window opens, and ``embargo`` rows are vacated after that as protection
    against serial correlation across the boundary; it defaults to ``horizon``
    and may not be smaller. At horizon 1 that is a two-bar gap where the older
    splitter left one -- slightly stricter, and the reason a rerun will not
    reproduce a pre-existing artifact to the last decimal.

    Returns the mean of each metric across folds, the fold-level detail under
    ``per_fold``, the protocol itself under ``split_protocol`` so a reader can
    audit the gap without rerunning, and the test-window base rate -- the
    accuracy a model that always guesses the majority class would score, which
    is the number any direction accuracy has to be read against.
    """
    horizon = max(1, int(horizon))
    folds = purged_walk_forward_splits(
        len(X),
        horizon=horizon,
        test_size=test_size,
        n_splits=n_splits,
        min_train=min_train,
        embargo=embargo,
    )
    if not folds:
        logger.warning("Not enough rows to form a walk-forward fold for %s", model.name)
        return {}
    splits = [(spec.train_pos, spec.test_pos) for spec in folds]

    y_return = np.asarray(y_return, dtype=np.float64)
    y_direction = np.asarray(y_direction, dtype=np.int8)
    prev_close = np.asarray(prev_close, dtype=np.float64)

    per_fold: List[Dict[str, Any]] = []

    for fold_number, (train_idx, test_idx) in enumerate(splits, start=1):
        logger.info("%s: fold %d/%d", model.name, fold_number, len(splits))
        fold = prepare_fold(X, train_idx, test_idx)

        try:
            model.fit(fold, y_return, y_direction)
            predicted_price = np.asarray(model.predict_price(fold, prev_close), dtype=np.float64)
            p_up, _ = model.predict_direction_proba(fold)
        except Exception as exc:  # noqa: BLE001 - one bad fold must not lose the whole run
            logger.error("%s failed on fold %d: %s", model.name, fold_number, exc, exc_info=True)
            continue

        p_up = np.asarray(p_up, dtype=np.float64)
        # A sequence model drops test rows without a full lookback window, so
        # the truth arrays are aligned to whatever positions it could predict.
        positions = fold.test_rows(model.sequence_length)[1]
        if len(predicted_price) != len(positions):
            positions = test_idx

        realised_price = prev_close[positions] * (1.0 + y_return[positions])
        valid = np.isfinite(predicted_price) & np.isfinite(p_up) & np.isfinite(realised_price)
        if valid.sum() < 2:
            logger.warning("%s: fold %d produced no usable predictions", model.name, fold_number)
            continue

        positions = positions[valid]
        fold_prev_close = prev_close[positions]
        fold_actual_price = realised_price[valid]
        fold_predicted_price = predicted_price[valid]
        fold_p_up = p_up[valid]

        price_scores = price_metrics(fold_actual_price, fold_predicted_price, fold_prev_close)
        direction_scores = classification_metrics(
            y_direction[positions], (fold_p_up >= 0.5).astype(int), fold_p_up
        )

        base_rate = float(np.mean(y_direction[positions]))
        record: Dict[str, Any] = {
            "fold": fold_number,
            "train_start": str(fold.index[train_idx[0]].date()),
            "train_end": str(fold.index[train_idx[-1]].date()),
            "test_start": str(fold.index[positions[0]].date()),
            "test_end": str(fold.index[positions[-1]].date()),
            "train_size": int(len(train_idx)),
            "test_size": int(len(positions)),
            "base_rate": base_rate,
            **{f"price_{key}": value for key, value in price_scores.items()},
            **_flatten_direction(direction_scores),
        }

        # Excess over base rate, per fold and against *this* fold's own base
        # rate. Averaging the per-fold values is not the same as scoring the
        # averaged accuracy against the averaged base rate -- max() is not
        # linear -- and the per-fold form is the honest one, because each fold
        # is judged against the coin it was actually dealt.
        accuracy = record.get("direction_accuracy")
        if accuracy is not None and math.isfinite(float(accuracy)):
            record["direction_eobr"] = round(
                float(accuracy) - max(base_rate, 1.0 - base_rate), 6
            )

        # The per-bar record. Everything downstream of a mean -- Diebold-Mariano,
        # McNemar, CRPS, the cost overlay -- needs paired observations, and none
        # of it can be recovered from an aggregate after the fact.
        record["predictions"] = {
            "date": [str(pd.Timestamp(stamp).date()) for stamp in fold.index[positions]],
            "prev_close": _series(fold_prev_close, 6),
            "actual_price": _series(fold_actual_price, 6),
            "predicted_price": _series(fold_predicted_price, 6),
            "actual_return": _series(y_return[positions], 8),
            "p_up": _series(fold_p_up, 6),
            "y_direction": [int(item) for item in y_direction[positions]],
        }

        per_fold.append(record)

    if not per_fold:
        logger.error("%s produced no scored folds", model.name)
        return {}

    scorecard = _aggregate(model.name, per_fold, test_size)
    scorecard["horizon"] = int(horizon)
    # The audit trail: where training stopped, where scoring started, and how
    # many bars were purged and embargoed between them. A reader who was not
    # present at the run can check the protocol without rerunning it.
    scorecard["split_protocol"] = _native(describe_split(folds, X.index))
    scorecard["evaluation"] = _pooled_evaluation(per_fold, horizon)
    return scorecard


def _economics(p_up: np.ndarray, period_log_returns: np.ndarray, horizon: int) -> Dict[str, Any]:
    """
    What the forecast is worth once someone has to pay to act on it.

    Deliberately last and deliberately narrow. The overlay tunes nothing -- the
    threshold is 0.5, the notional is fixed, and the rebalance period is the
    horizon -- so this measures the forecast rather than a strategy built around
    it. ``breakeven_bps`` is the number worth quoting: the per-side cost at
    which the gross edge is exactly consumed.

    Only horizon 1 is served. Past that the pooled returns overlap, and running
    a period-by-period equity curve over overlapping windows would compound the
    same move repeatedly -- a wrong number rather than a missing one.
    """
    if horizon != 1:
        return {
            "available": False,
            "reason": f"overlapping returns at horizon {horizon}; needs non-overlapping periods",
        }
    if p_up.size < 3:
        return {"available": False, "reason": f"only {p_up.size} observations"}

    cost = CostModel()
    positions = long_flat_positions(p_up)
    return {
        "available": True,
        "cost_model": cost.to_dict(),
        "breakeven": breakeven_round_trip_bps(positions, period_log_returns),
        "overlay": paper_trading_overlay(
            p_up,
            period_log_returns,
            cost=cost,
            periods_per_year=252,
        ),
    }


def _pooled_evaluation(per_fold: List[Dict[str, Any]], horizon: int) -> Dict[str, Any]:
    """
    The verdict, pooled across folds: does this model beat the martingale null?

    Fold means answer "how did it score"; this answers "is the score separable
    from doing nothing", which is the only question a price forecast has to pass
    before any of the rest is worth reading. Two nulls are tested, because a
    model can fail either independently:

    ``vs_random_walk``
        The primary one. Scoring is in squared log-return space, where the
        random walk forecasts exactly zero -- so the baseline loss is the
        realised squared log return and needs no model to compute. Diebold-
        Mariano on the paired losses says whether the difference is real.
        ``r2_vs_random_walk`` is the same comparison as a fraction of variance:
        below zero means the random walk won.

    ``edge_vs_majority``
        The direction equivalent: accuracy against always guessing the more
        common class, with the one-sided test that says whether a few hundred
        bars can support it.

    Pooling is what makes these testable at all -- three folds of 63 bars are
    189 paired observations together and far too few apiece.
    """
    prev_close: List[Any] = []
    actual_price: List[Any] = []
    predicted_price: List[Any] = []
    probability_up: List[Any] = []

    for fold in per_fold:
        record = fold.get("predictions") or {}
        prev_close.extend(record.get("prev_close") or [])
        actual_price.extend(record.get("actual_price") or [])
        predicted_price.extend(record.get("predicted_price") or [])
        probability_up.extend(record.get("p_up") or [])

    lengths = {len(prev_close), len(actual_price), len(predicted_price), len(probability_up)}
    if len(lengths) != 1:
        return {"available": False, "reason": f"per-bar columns disagree in length: {sorted(lengths)}"}

    def as_array(values: List[Any]) -> np.ndarray:
        return np.asarray([np.nan if item is None else item for item in values], dtype=np.float64)

    prev = as_array(prev_close)
    actual = as_array(actual_price)
    predicted = as_array(predicted_price)
    p_up = as_array(probability_up)

    # Log returns need strictly positive prices on both ends, and the ratio is
    # undefined without a previous close.
    usable = (
        np.isfinite(prev) & np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(p_up)
        & (prev > 0) & (actual > 0) & (predicted > 0)
    )
    n = int(usable.sum())
    if n < 3:
        return {"available": False, "reason": f"only {n} usable paired observations; need at least 3"}

    actual_log = np.log(actual[usable] / prev[usable])
    model_log = np.log(predicted[usable] / prev[usable])
    p_up_usable = p_up[usable]

    loss_model = (model_log - actual_log) ** 2
    loss_random_walk = actual_log ** 2  # the random walk forecasts a zero return

    sse_model = float(np.sum(loss_model))
    sse_random_walk = float(np.sum(loss_random_walk))
    r2_vs_random_walk = (
        1.0 - sse_model / sse_random_walk if sse_random_walk > 0 else float("nan")
    )

    truth = pd.Series(actual_log)
    probabilities = pd.Series(p_up_usable)
    direction = directional_metrics(truth, probabilities)
    probabilistic = probabilistic_metrics(truth, probabilities)

    accuracy = direction.get("accuracy")
    base_rate = direction.get("base_rate")
    edge: Dict[str, Any] = {}
    if accuracy is not None and base_rate is not None and math.isfinite(float(accuracy)):
        reference = max(float(base_rate), 1.0 - float(base_rate))
        edge = accuracy_edge_test(float(accuracy), reference, n)

    return _native(
        {
            "available": True,
            "n": n,
            # Overlapping h-period returns are not h independent observations.
            # At horizon 1 this equals n; past that it is the number every
            # p-value here should really be read against.
            "effective_n": effective_sample_size(n, horizon),
            "horizon": int(horizon),
            "loss_space": "squared_log_return",
            "direction": direction,
            "probabilistic": probabilistic,
            "edge_vs_majority": edge,
            "economics": _economics(p_up_usable, actual_log, horizon),
            "vs_random_walk": {
                "mse_model": sse_model / n,
                "mse_random_walk": sse_random_walk / n,
                "r2_vs_random_walk": r2_vs_random_walk,
                "diebold_mariano": diebold_mariano_test(
                    loss_model, loss_random_walk, horizon=int(horizon)
                ),
            },
        }
    )


def _aggregate(model_name: str, per_fold: List[Dict[str, Any]], test_size: int) -> Dict[str, Any]:
    """
    Mean of each metric across folds, plus its spread.

    The standard deviation is carried alongside every mean on purpose: with
    three or four folds of 63 days, a 2-point accuracy gap that is smaller than
    the fold-to-fold spread is not evidence of anything.
    """
    numeric_keys = {
        key
        for fold in per_fold
        for key, value in fold.items()
        if isinstance(value, (int, float)) and key not in ("fold", "train_size", "test_size")
    }

    aggregated: Dict[str, Any] = {
        "model_name": model_name,
        "n_splits": len(per_fold),
        "test_size": test_size,
        "total_test_rows": int(sum(fold["test_size"] for fold in per_fold)),
    }
    for key in sorted(numeric_keys):
        values = [fold[key] for fold in per_fold if np.isfinite(fold.get(key, np.nan))]
        aggregated[key] = round(float(np.mean(values)), 6) if values else float("nan")
        if key.startswith("direction_") or key == "price_r2_return":
            aggregated[f"{key}_std"] = round(float(np.std(values)), 6) if len(values) > 1 else 0.0

    aggregated["per_fold"] = per_fold
    return aggregated


def summarise_comparison(results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    The head-to-head table, ranked by directional accuracy.

    Three columns carry the actual answer, and they are here rather than left
    to the reader because an accuracy alone invites the wrong conclusion:

    ``edge_pp``
        Percentage points of accuracy bought over always guessing the majority
        class on the same test windows. A model can top the table and still have
        bought nothing.
    ``p_value``
        One-sided, against the hypothesis that the edge is real. A few hundred
        test days is a small sample for a two-point effect, and this is the
        column that says so.
    ``n_required``
        Test days an edge that size would need before it could be called
        significant at 5%. Usually a sobering number.
    """
    if not results:
        return None

    rows = []
    for result in results:
        base_rate = result.get("base_rate")
        accuracy = result.get("direction_accuracy")
        n_rows = result.get("total_test_rows") or 0

        edge = p_value = n_required = None
        significant = None
        if accuracy is not None and base_rate is not None and n_rows:
            # The reference is the majority class, not 0.5: on a window that ran
            # 55% up, a 52%-accurate model has a negative edge, not a positive one.
            reference = max(base_rate, 1.0 - base_rate)
            test = accuracy_edge_test(accuracy, reference, n_rows)
            edge = round(test["edge_pp"], 2)
            p_value = round(test["p_value_one_sided"], 3)
            significant = test["significant"]
            n_required = test["n_required"]

        rows.append(
            {
                "symbol": result.get("symbol", ""),
                "model": result.get("model_name", ""),
                "direction_accuracy": accuracy,
                "base_rate": base_rate,
                "edge_pp": edge,
                "p_value": p_value,
                "significant": significant,
                "n_required": n_required,
                "direction_f1": result.get("direction_f1"),
                "direction_roc_auc": result.get("direction_roc_auc"),
                "price_mape": result.get("price_mape"),
                "price_rmse": result.get("price_rmse"),
                "price_r2_return": result.get("price_r2_return"),
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values("direction_accuracy", ascending=False, na_position="last")
