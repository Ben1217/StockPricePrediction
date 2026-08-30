"""
Walk-forward hyperparameter tuning for the return regressors.

K-fold CV is invalid here: shuffling folds lets a model fit bars from 2026 and
be scored on 2023. This module instead grows the training window forward in
time, always scoring on the segment that comes *after* the one it fit, which is
the only arrangement that matches how the model is actually used.

Each fold purges `embargo` rows between its train and score segments. With an
h-day forward-return target consecutive rows overlap by h-1 days, so without
that gap the last training rows resolve inside the scoring window and the
reported error is optimistic.

    fold 1:  [--- train ---][gap][score]
    fold 2:  [------ train ------][gap][score]
    fold 3:  [--------- train --------][gap][score]

Public API:
    walk_forward_splits(n_rows, n_splits, embargo) -> list[(train_idx, score_idx)]
    walk_forward_tune(...) -> TuningResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)


# Grids stay deliberately small: these are regularisation knobs, and the point
# is to find whether a *less* flexible model generalises better, not to search
# exhaustively. Every extra point multiplies the training run.
DEFAULT_PARAM_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "random_forest": [
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 20},
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 10},
        {"n_estimators": 300, "max_depth": 10, "min_samples_leaf": 5},
        {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 2},
    ],
    "xgboost": [
        {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.03,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    ],
    # The LSTM costs minutes per fit, so it gets two candidates: a small
    # regularised network and the incumbent.
    "lstm": [
        {"units": 32, "layers": 1, "dropout": 0.3, "epochs": 40, "patience": 8},
        {"units": 64, "layers": 2, "dropout": 0.2, "epochs": 80, "patience": 15},
    ],
}


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    scored_candidates: List[Dict[str, Any]] = field(default_factory=list)
    n_splits: int = 0
    embargo: int = 0
    metric: str = "mae"

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "method": "walk_forward_expanding_window",
            "metric": self.metric,
            "n_splits": self.n_splits,
            "embargo": self.embargo,
            "best_params": self.best_params,
            "best_score": round(float(self.best_score), 6),
            "candidates": self.scored_candidates,
            "note": (
                "k-fold is not used: folds are contiguous and always score the "
                "segment following the one they trained on"
            ),
        }


def walk_forward_splits(
    n_rows: int,
    n_splits: int = 4,
    embargo: int = 0,
    min_train: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window (train_idx, score_idx) pairs over `n_rows` ordered rows.

    The scoring segments are contiguous, equal-sized, and cover the tail of the
    series; the training window grows to meet each one.
    """
    if n_rows < 3:
        return []
    n_splits = max(1, int(n_splits))
    embargo = max(0, int(embargo))

    # Reserve the tail for scoring, split evenly across folds.
    fold_size = max(1, n_rows // (n_splits + 1))
    min_train = min_train if min_train is not None else fold_size

    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(n_splits):
        score_start = n_rows - (n_splits - i) * fold_size
        score_end = score_start + fold_size
        train_end = score_start - embargo
        if train_end < min_train or score_start >= n_rows:
            continue
        splits.append((
            np.arange(0, train_end),
            np.arange(score_start, min(score_end, n_rows)),
        ))
    return splits


def _score_of(metrics: Dict[str, float], metric: str) -> float:
    value = metrics.get(metric)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    return value if np.isfinite(value) else float("inf")


def walk_forward_tune(
    *,
    model_type: str,
    factory,
    X: np.ndarray,
    y: np.ndarray,
    base_params: Optional[Dict[str, Any]] = None,
    param_grid: Optional[Sequence[Dict[str, Any]]] = None,
    n_splits: int = 4,
    embargo: int = 0,
    metric: str = "mae",
    sequence_builder=None,
) -> Optional[TuningResult]:
    """
    Pick hyperparameters by mean walk-forward error.

    `X`/`y` must be the chronologically ordered train+validation region — the
    test set is never touched here. `sequence_builder(X_tr, y_tr, X_sc, y_sc,
    params)` lets the LSTM turn flat rows into sequences per fold; tabular
    models leave it as None.

    Returns None when the data is too short to form a single honest fold, in
    which case the caller should fall back to its default parameters.
    """
    grid = list(param_grid or DEFAULT_PARAM_GRIDS.get(model_type, []))
    if not grid:
        return None

    splits = walk_forward_splits(len(X), n_splits=n_splits, embargo=embargo)
    if not splits:
        logger.warning(
            "Not enough rows (%s) for walk-forward tuning of %s with embargo=%s; "
            "using default parameters",
            len(X), model_type, embargo,
        )
        return None

    base_params = dict(base_params or {})
    scored: List[Dict[str, Any]] = []
    best_params, best_score = None, float("inf")

    for candidate in grid:
        params = {**base_params, **candidate}
        fold_scores: List[float] = []
        for train_idx, score_idx in splits:
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_sc, y_sc = X[score_idx], y[score_idx]
            try:
                if sequence_builder is not None:
                    built = sequence_builder(X_tr, y_tr, X_sc, y_sc, params)
                    if built is None:
                        continue
                    X_tr, y_tr, X_sc, y_sc = built
                model = factory(params)
                model.fit(X_tr, y_tr)
                fold_scores.append(_score_of(model.evaluate(X_sc, y_sc), metric))
            except Exception as exc:  # noqa: BLE001 - a bad candidate must not kill the run
                logger.warning("Walk-forward fold failed for %s %s: %s", model_type, candidate, exc)
                fold_scores.append(float("inf"))

        mean_score = float(np.mean(fold_scores)) if fold_scores else float("inf")
        scored.append({"params": candidate,
                       "mean_score": round(mean_score, 6) if np.isfinite(mean_score) else None,
                       "fold_scores": [round(s, 6) if np.isfinite(s) else None for s in fold_scores]})
        logger.info("  walk-forward %s %s -> mean %s=%.6f", model_type, candidate, metric, mean_score)
        if mean_score < best_score:
            best_params, best_score = params, mean_score

    if best_params is None or not np.isfinite(best_score):
        return None

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        scored_candidates=scored,
        n_splits=len(splits),
        embargo=embargo,
        metric=metric,
    )
