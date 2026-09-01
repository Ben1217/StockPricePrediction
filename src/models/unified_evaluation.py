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

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.models.direction_metrics import accuracy_edge_test, classification_metrics
from src.models.unified_models import UnifiedEstimator, prepare_fold
from src.models.walk_forward import expanding_window_splits
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


def evaluate_unified_walk_forward(
    model: UnifiedEstimator,
    X: pd.DataFrame,
    y_return: np.ndarray,
    y_direction: np.ndarray,
    prev_close: np.ndarray,
    *,
    test_size: int = 63,
    n_splits: int = 4,
    embargo: int = 1,
    min_train: int = 252,
) -> Dict[str, Any]:
    """
    Score one model with expanding-window walk-forward validation.

    Chronological order is preserved throughout: every fold trains on a prefix
    of the series and tests on the block that follows it, with ``embargo`` rows
    purged in between so the last training label cannot resolve inside the test
    window. No shuffling, anywhere.

    Returns the mean of each metric across folds, the fold-level detail under
    ``per_fold``, and the test-window base rate -- the accuracy a model that
    always guesses the majority class would score, which is the number any
    direction accuracy has to be read against.
    """
    splits = expanding_window_splits(
        len(X), test_size=test_size, n_splits=n_splits, embargo=embargo, min_train=min_train
    )
    if not splits:
        logger.warning("Not enough rows to form a walk-forward fold for %s", model.name)
        return {}

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
        price_scores = price_metrics(
            realised_price[valid], predicted_price[valid], prev_close[positions]
        )
        direction_scores = classification_metrics(
            y_direction[positions], (p_up[valid] >= 0.5).astype(int), p_up[valid]
        )

        per_fold.append(
            {
                "fold": fold_number,
                "train_start": str(fold.index[train_idx[0]].date()),
                "train_end": str(fold.index[train_idx[-1]].date()),
                "test_start": str(fold.index[positions[0]].date()),
                "test_end": str(fold.index[positions[-1]].date()),
                "train_size": int(len(train_idx)),
                "test_size": int(len(positions)),
                "base_rate": float(np.mean(y_direction[positions])),
                **{f"price_{key}": value for key, value in price_scores.items()},
                **_flatten_direction(direction_scores),
            }
        )

    if not per_fold:
        logger.error("%s produced no scored folds", model.name)
        return {}

    return _aggregate(model.name, per_fold, test_size)


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
