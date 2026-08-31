"""
Classifiers and reference baselines for next-day direction.

Every estimator here implements the same three-method interface:

    fit(X: DataFrame, y: Series) -> self
    predict_proba_up(X: DataFrame) -> np.ndarray   # P(up), one value per row
    predict(X: DataFrame) -> np.ndarray            # hard 0/1 label

``X`` is a DataFrame, not an array, on purpose: the momentum and reversal
baselines need ``Daily_Return`` by name, and a positional index into a feature
matrix is exactly the kind of thing that silently breaks when the feature set
changes.

**All tuning happens inside ``fit``.** A model is only ever handed the training
window, so its scaler, its regularisation strength, and its early-stopping round
count are chosen without the test fold existing. That is enforced structurally
rather than by convention: there is no path by which an estimator here can see a
row the caller did not pass to ``fit``.

Model order is deliberate. Logistic regression is the reference, not a
throwaway — on a problem with a base rate near 53% and a signal-to-noise ratio
this low, a regularised linear model is a genuinely hard baseline, and a
gradient booster that cannot beat it is not learning structure, it is
memorising. The LSTM is not implemented here: it is only worth the runtime once
these two clear the baselines in section 6.

Baselines
---------
``majority``      always the training-set majority class
``momentum_1``    tomorrow repeats the sign of today's return
``reversal_1``    tomorrow reverses the sign of today's return
``base_rate``     Bernoulli(p) draws at the training base rate

The two rule baselines emit a *calibrated* probability — the empirical
P(up tomorrow | today up/down) measured on the training fold — while their hard
label stays the stated rule. Emitting 0.0/1.0 instead would make log loss
infinite and Brier score meaningless, which would flatter the model by
comparison rather than testing it.

Public API:
    build_model(name, seed) -> DirectionEstimator
    MODEL_FACTORIES, BASELINE_FACTORIES, ALL_FACTORIES
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Fraction of a training window held out, chronologically at its end, for
# in-fold model selection (regularisation strength, early-stopping rounds).
INNER_VALIDATION_FRACTION = 0.2
MIN_INNER_VALIDATION_ROWS = 40

# L2 strengths for the logistic reference. Spanning four orders of magnitude
# lets the search land on "almost the intercept" when the features carry
# nothing, which is the honest answer often enough to matter.
LOGISTIC_C_GRID: tuple[float, ...] = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)

# Shallow, heavily regularised trees. Depth beyond 4 at n ~ 1000 fits the noise.
GBM_PARAM_GRID: tuple[Dict[str, Any], ...] = (
    {"max_depth": 2, "learning_rate": 0.03, "min_child_weight": 20,
     "subsample": 0.8, "colsample_bytree": 0.6, "reg_lambda": 10.0},
    {"max_depth": 3, "learning_rate": 0.03, "min_child_weight": 10,
     "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
    {"max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5,
     "subsample": 0.9, "colsample_bytree": 0.8, "reg_lambda": 1.0},
)
GBM_MAX_ROUNDS = 400
GBM_EARLY_STOPPING_ROUNDS = 30

_PROBA_EPS = 1e-6


def _clip_proba(values: np.ndarray) -> np.ndarray:
    """Keep probabilities strictly inside (0, 1) so log loss stays finite."""
    return np.clip(np.asarray(values, dtype=np.float64), _PROBA_EPS, 1.0 - _PROBA_EPS)


def _inner_split(n_rows: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Chronological (inner_train, inner_validation) positions inside a training window.

    Returns None when the window is too short to hold out a meaningful tail, in
    which case the caller must fall back to fixed parameters rather than tune on
    a handful of rows.
    """
    n_validation = int(round(n_rows * INNER_VALIDATION_FRACTION))
    if n_validation < MIN_INNER_VALIDATION_ROWS or n_rows - n_validation < MIN_INNER_VALIDATION_ROWS:
        return None
    cut = n_rows - n_validation
    return np.arange(cut), np.arange(cut, n_rows)


def _log_loss(y_true: np.ndarray, p_up: np.ndarray) -> float:
    """Binary log loss. Local so the model layer does not import the metrics layer."""
    p = _clip_proba(p_up)
    y = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


class DirectionEstimator:
    """Base class: shared fitted-state bookkeeping and the hard-label rule."""

    name = "base"

    # Whether fit() actually derives anything from y. False for a pre-trained
    # zero-shot model, which makes the shuffled-label leakage check vacuous:
    # permuting labels a model never reads cannot change its output, so the
    # check would report a pass without having tested anything. The pipeline
    # reads this and records "not applicable" instead of running it.
    learns_from_labels: bool = True

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.fitted_: bool = False
        self.train_base_rate_: float = 0.5
        self.fit_info_: Dict[str, Any] = {}

    # -- subclass hooks ---------------------------------------------------
    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        raise NotImplementedError

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    # -- public interface -------------------------------------------------
    def fit(self, X: pd.DataFrame, y) -> "DirectionEstimator":
        y_array = np.asarray(y, dtype=np.int8).reshape(-1)
        if len(X) != len(y_array):
            raise ValueError(f"X has {len(X)} rows but y has {len(y_array)}")
        if len(y_array) == 0:
            raise ValueError("Cannot fit on an empty training window")
        self.train_base_rate_ = float(np.mean(y_array))
        self.fit_info_ = {"n_train": int(len(y_array)), "train_base_rate": self.train_base_rate_}
        self._fit(X, y_array)
        self.fitted_ = True
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return _clip_proba(self._predict_proba_up(X))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Hard label at the 0.5 decision boundary. Trading thresholds live in the backtest."""
        return (self.predict_proba_up(X) >= 0.5).astype(np.int8)


class _ConstantProbabilityMixin:
    """Shared fallback for a degenerate training window (one class present)."""

    def _constant(self, n: int, value: float) -> np.ndarray:
        return np.full(int(n), float(value), dtype=np.float64)


class LogisticDirection(DirectionEstimator, _ConstantProbabilityMixin):
    """
    L2 logistic regression on standardised features. The reference model.

    ``C`` is chosen by log loss on a chronological hold-out at the end of the
    training window — a proper scoring rule, because the backtest thresholds a
    probability rather than a hard label, so calibration is what matters.
    The scaler is fitted on the inner-training rows during the search and refit
    on the full training window for the final model; it never sees test rows.
    """

    name = "logistic"

    def __init__(self, seed: int = 42, c_grid: Sequence[float] = LOGISTIC_C_GRID):
        super().__init__(seed=seed)
        self.c_grid = tuple(float(c) for c in c_grid)
        self.scaler_: Optional[StandardScaler] = None
        self.model_: Optional[LogisticRegression] = None
        self.degenerate_: bool = False

    def _make(self, c: float) -> LogisticRegression:
        return LogisticRegression(
            # L2 is sklearn's default; naming it explicitly is deprecated as of
            # sklearn 1.8, so the penalty is left implicit and C carries the
            # regularisation strength.
            C=float(c),
            solver="lbfgs",
            max_iter=2000,
            # Up days outnumber down days; without this the model buys the base
            # rate and its "accuracy" is the majority baseline in disguise.
            class_weight="balanced",
            random_state=self.seed,
        )

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.degenerate_ = len(np.unique(y)) < 2
        if self.degenerate_:
            logger.warning("Training window for %s holds one class only; emitting its base rate", self.name)
            self.fit_info_["degenerate_single_class"] = True
            return

        values = X.to_numpy(dtype=np.float64)
        chosen_c, search = float(self.c_grid[len(self.c_grid) // 2]), []

        split = _inner_split(len(values))
        if split is not None:
            inner_train, inner_validation = split
            if len(np.unique(y[inner_train])) == 2:
                scaler = StandardScaler().fit(values[inner_train])
                x_tr = scaler.transform(values[inner_train])
                x_va = scaler.transform(values[inner_validation])
                best = np.inf
                for c in self.c_grid:
                    model = self._make(c).fit(x_tr, y[inner_train])
                    score = _log_loss(y[inner_validation], model.predict_proba(x_va)[:, 1])
                    search.append({"C": c, "val_log_loss": round(score, 6)})
                    if score < best:
                        best, chosen_c = score, c

        self.scaler_ = StandardScaler().fit(values)
        self.model_ = self._make(chosen_c).fit(self.scaler_.transform(values), y)
        self.fit_info_.update({"C": chosen_c, "c_search": search})

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.degenerate_ or self.model_ is None or self.scaler_ is None:
            return self._constant(len(X), self.train_base_rate_)
        return self.model_.predict_proba(self.scaler_.transform(X.to_numpy(dtype=np.float64)))[:, 1]


class GradientBoostingDirection(DirectionEstimator, _ConstantProbabilityMixin):
    """
    Shallow XGBoost classifier with in-fold early stopping.

    Two stages, both confined to the training window:

    1. For each shallow parameter set, fit on the inner-training rows with early
       stopping on the chronological inner-validation tail. Keep the set with
       the best validation log loss, and the round count it stopped at.
    2. Refit that set on the *whole* training window for exactly that many
       rounds, so the final model uses all the training data rather than 80% of
       it while still not choosing its own capacity from the test fold.

    ``scale_pos_weight`` is set from the training class balance, the tree
    equivalent of ``class_weight='balanced'``.
    """

    name = "gradient_boosting"

    def __init__(self, seed: int = 42, param_grid: Sequence[Dict[str, Any]] = GBM_PARAM_GRID):
        super().__init__(seed=seed)
        self.param_grid = tuple(dict(p) for p in param_grid)
        self.model_ = None
        self.degenerate_: bool = False

    def _base_kwargs(self, y: np.ndarray) -> Dict[str, Any]:
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": self.seed,
            "n_jobs": 1,
            "scale_pos_weight": (n_neg / n_pos) if n_pos > 0 else 1.0,
        }

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        from xgboost import XGBClassifier

        self.degenerate_ = len(np.unique(y)) < 2
        if self.degenerate_:
            logger.warning("Training window for %s holds one class only; emitting its base rate", self.name)
            self.fit_info_["degenerate_single_class"] = True
            return

        values = X.to_numpy(dtype=np.float32)
        base = self._base_kwargs(y)
        best_params: Dict[str, Any] = dict(self.param_grid[0])
        best_rounds = 100
        search: List[Dict[str, Any]] = []

        split = _inner_split(len(values))
        if split is not None and len(np.unique(y[split[0]])) == 2:
            inner_train, inner_validation = split
            best_score = np.inf
            for params in self.param_grid:
                model = XGBClassifier(
                    n_estimators=GBM_MAX_ROUNDS,
                    early_stopping_rounds=GBM_EARLY_STOPPING_ROUNDS,
                    **base,
                    **params,
                )
                model.fit(
                    values[inner_train], y[inner_train],
                    eval_set=[(values[inner_validation], y[inner_validation])],
                    verbose=False,
                )
                proba = model.predict_proba(values[inner_validation])[:, 1]
                score = _log_loss(y[inner_validation], proba)
                # best_iteration is 0-based; +1 converts it to a round count.
                rounds = int(getattr(model, "best_iteration", GBM_MAX_ROUNDS - 1)) + 1
                search.append({"params": params, "val_log_loss": round(score, 6), "rounds": rounds})
                if score < best_score:
                    best_score, best_params, best_rounds = score, dict(params), max(1, rounds)

        final = XGBClassifier(n_estimators=int(best_rounds), **base, **best_params)
        final.fit(values, y, verbose=False)
        self.model_ = final
        self.fit_info_.update({"params": best_params, "n_rounds": int(best_rounds), "search": search})

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        if self.degenerate_ or self.model_ is None:
            return self._constant(len(X), self.train_base_rate_)
        return self.model_.predict_proba(X.to_numpy(dtype=np.float32))[:, 1]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class MajorityBaseline(DirectionEstimator, _ConstantProbabilityMixin):
    """Always the training majority class. Its accuracy is the number to beat."""

    name = "majority"

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.majority_ = int(self.train_base_rate_ >= 0.5)

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return self._constant(len(X), self.train_base_rate_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return np.full(len(X), self.majority_, dtype=np.int8)


class _SignRuleBaseline(DirectionEstimator, _ConstantProbabilityMixin):
    """
    Shared machinery for the momentum and reversal rules.

    ``sign_multiplier`` is +1 for momentum (repeat today's sign) and -1 for
    reversal. The hard label is the rule verbatim; the probability is the
    empirical P(up tomorrow | today's sign) measured on the training fold, so
    Brier and log loss are finite and the rule is scored at its true confidence
    rather than an assumed certainty it never had.
    """

    signal_column = "Daily_Return"
    sign_multiplier = 1

    def _signal(self, X: pd.DataFrame) -> np.ndarray:
        if self.signal_column not in X.columns:
            raise KeyError(
                f"{self.name} needs the '{self.signal_column}' column; "
                f"available: {list(X.columns)[:10]}"
            )
        return np.sign(pd.to_numeric(X[self.signal_column], errors="coerce").to_numpy(dtype=np.float64))

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        signal = self._signal(X)
        up_today = signal > 0
        down_today = signal < 0
        # Fall back to the overall base rate where a branch has no training rows.
        self.p_after_up_ = float(np.mean(y[up_today])) if up_today.any() else self.train_base_rate_
        self.p_after_down_ = float(np.mean(y[down_today])) if down_today.any() else self.train_base_rate_
        self.fit_info_.update({
            "p_up_after_up_day": round(self.p_after_up_, 6),
            "p_up_after_down_day": round(self.p_after_down_, 6),
        })

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        signal = self._signal(X)
        proba = np.where(signal > 0, self.p_after_up_,
                         np.where(signal < 0, self.p_after_down_, self.train_base_rate_))
        return proba.astype(np.float64)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        signal = self._signal(X) * self.sign_multiplier
        # A flat day carries no rule signal, so it defers to the majority class.
        fallback = int(self.train_base_rate_ >= 0.5)
        return np.where(signal > 0, 1, np.where(signal < 0, 0, fallback)).astype(np.int8)


class MomentumBaseline(_SignRuleBaseline):
    """Tomorrow repeats the sign of today's return."""

    name = "momentum_1"
    sign_multiplier = 1


class ReversalBaseline(_SignRuleBaseline):
    """Tomorrow reverses the sign of today's return."""

    name = "reversal_1"
    sign_multiplier = -1


class BaseRateRandomBaseline(DirectionEstimator, _ConstantProbabilityMixin):
    """
    Bernoulli(p) hard labels at the training base rate.

    Distinct from ``majority``: both emit the same constant probability, but
    this one's *labels* are random, which is what makes it the right reference
    for balanced accuracy and MCC. The draws are seeded, so a re-run reproduces
    them exactly.
    """

    name = "base_rate"

    def _fit(self, X: pd.DataFrame, y: np.ndarray) -> None:
        self.rng_ = np.random.default_rng(self.seed)

    def _predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        return self._constant(len(X), self.train_base_rate_)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return (self.rng_.random(len(X)) < self.train_base_rate_).astype(np.int8)


# The foundation-model slots are imported lazily. Neither `tabpfn` nor the
# vendored Kronos checkout is required to run the logistic or gradient-boosting
# path, and importing them at module scope would make a missing optional
# dependency break the whole registry.

def _build_tabpfn(seed: int, **kwargs) -> DirectionEstimator:
    from .tabpfn_direction import TabPFNDirection
    return TabPFNDirection(seed=seed, **kwargs)


def _build_kronos(seed: int, **kwargs) -> DirectionEstimator:
    from .kronos_direction import KronosDirection
    return KronosDirection(seed=seed, **kwargs)


def _build_foundation_ensemble(seed: int, **kwargs) -> DirectionEstimator:
    from .ensemble_direction import FoundationEnsemble
    return FoundationEnsemble(seed=seed, **kwargs)


MODEL_FACTORIES: Dict[str, Callable[..., DirectionEstimator]] = {
    "logistic": lambda seed, **kw: LogisticDirection(seed=seed, **kw),
    "gradient_boosting": lambda seed, **kw: GradientBoostingDirection(seed=seed, **kw),
    "tabpfn": _build_tabpfn,
    "kronos": _build_kronos,
    "foundation_ensemble": _build_foundation_ensemble,
}

BASELINE_FACTORIES: Dict[str, Callable[..., DirectionEstimator]] = {
    "majority": lambda seed, **kw: MajorityBaseline(seed=seed),
    "momentum_1": lambda seed, **kw: MomentumBaseline(seed=seed),
    "reversal_1": lambda seed, **kw: ReversalBaseline(seed=seed),
    "base_rate": lambda seed, **kw: BaseRateRandomBaseline(seed=seed),
}

ALL_FACTORIES: Dict[str, Callable[..., DirectionEstimator]] = {
    **MODEL_FACTORIES,
    **BASELINE_FACTORIES,
}


def build_model(name: str, seed: int = 42, **kwargs) -> DirectionEstimator:
    """
    Instantiate an estimator by name, raising on an unknown name rather than guessing.

    ``kwargs`` are forwarded to the estimator, which is how model-specific
    options (Kronos' sample count and lookback, say) reach it from the CLI.
    Baselines ignore them: they take no options, and a stray keyword should not
    stop the comparison row from being produced.
    """
    key = str(name).strip().lower()
    if key not in ALL_FACTORIES:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(ALL_FACTORIES)}")
    return ALL_FACTORIES[key](seed, **kwargs)

