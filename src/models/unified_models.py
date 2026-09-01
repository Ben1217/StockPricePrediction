"""
Unified model wrappers: one interface that answers both project questions.

Every estimator here answers the two questions the dashboard asks of a single
bar:

    predict_price(...)            -> next-timeframe close, in dollars
    predict_direction_proba(...)  -> (P(up), P(down))

Having both behind one interface is what makes the comparison in
``scripts/unified_benchmark.py`` a fair test: LSTM, XGBoost, Random Forest, the
dynamic ensemble and Kronos see the same rows, the same fold boundaries and the
same targets, and are scored by the same code.

The fold contract
-----------------
Models are handed a :class:`FoldInputs`, not a bare matrix. It carries the whole
feature matrix for the symbol -- already scaled with statistics fitted on the
*training* rows only -- plus the positions of the training and test rows.

That indirection exists for the sequence models. An LSTM asked to predict test
row ``t`` needs the ``sequence_length`` bars ending at ``t``, and most of those
bars live before the fold boundary. Handing it only the test slice would leave
the first rows of every fold with no history; handing it the full matrix and a
set of positions lets it look back into data that was already observed at
decision time. Nothing after ``t`` is ever read, so the lookback is history, not
leakage.

Tabular models ignore the machinery and slice the rows they were given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.direction_models import GradientBoostingDirection
from src.models.lstm_model import LSTMModel
from src.models.random_forest_model import RandomForestModel
from src.models.regression_models import (
    LSTMPriceRegressor,
    RandomForestPriceRegressor,
    XGBoostPriceRegressor,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Bars of history an LSTM sees per prediction. 60 trading days is roughly a
# quarter: long enough to contain the momentum and mean-reversion patterns the
# indicator columns describe, short enough to leave most of the window for
# training rows.
DEFAULT_SEQUENCE_LENGTH = 60


# ---------------------------------------------------------------------------
# Fold plumbing
# ---------------------------------------------------------------------------


@dataclass
class FoldInputs:
    """
    One walk-forward fold, prepared identically for every model.

    ``X_scaled`` covers the *whole* series so sequence models can read the bars
    preceding a test row. The scaler behind it was fitted on ``train_pos`` only.
    """

    X_scaled: np.ndarray
    index: pd.Index
    feature_columns: List[str]
    train_pos: np.ndarray
    test_pos: np.ndarray
    scaler: Any = None

    def train_rows(self, sequence_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Training design matrix, and the positions its rows correspond to."""
        return _slice_rows(self.X_scaled, self.train_pos, sequence_length)

    def test_rows(self, sequence_length: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Test design matrix, and the positions its rows correspond to."""
        return _slice_rows(self.X_scaled, self.test_pos, sequence_length)


def seed_everything(seed: int) -> None:
    """
    Pin every random source the model layer draws from, before a fit.

    Neither LSTM wrapper seeds torch, so weight initialisation and the training
    DataLoader's shuffle differ run to run. Left alone, re-running the benchmark
    moves the LSTM's accuracy by a point or so -- which is the same size as the
    effects the benchmark exists to measure, and would make any comparison
    against it unfalsifiable.
    """
    import random

    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _slice_rows(
    X_scaled: np.ndarray, positions: np.ndarray, sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rows at ``positions``, as a 2D matrix or a stack of lookback windows.

    A window ending at position ``p`` needs rows ``p - L + 1 .. p``; positions
    too close to the start of the series cannot form one and are dropped, which
    is why the kept positions come back alongside the matrix.
    """
    positions = np.asarray(positions, dtype=int)
    if sequence_length <= 1:
        return X_scaled[positions], positions

    keep = positions[positions >= sequence_length - 1]
    if len(keep) == 0:
        return np.empty((0, sequence_length, X_scaled.shape[1]), dtype=np.float32), keep
    windows = np.stack([X_scaled[p - sequence_length + 1 : p + 1] for p in keep])
    return windows.astype(np.float32), keep


def prepare_fold(
    X: pd.DataFrame,
    train_pos: Sequence[int],
    test_pos: Sequence[int],
    *,
    scale: bool = True,
) -> FoldInputs:
    """
    Scale one fold's features and package them for the model layer.

    The scaler is fitted on the training rows and then applied to the whole
    matrix. Fitting on training rows only is what keeps it leakage-free;
    applying it everywhere is what lets a sequence model reach back across the
    fold boundary for its lookback window.
    """
    train_pos = np.asarray(train_pos, dtype=int)
    test_pos = np.asarray(test_pos, dtype=int)
    values = X.to_numpy(dtype=np.float64)

    scaler = None
    if scale:
        scaler = StandardScaler().fit(values[train_pos])
        values = scaler.transform(values)

    return FoldInputs(
        X_scaled=np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        index=X.index,
        feature_columns=list(X.columns),
        train_pos=train_pos,
        test_pos=test_pos,
        scaler=scaler,
    )


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------


class UnifiedEstimator:
    """Predicts an absolute next-timeframe price *and* a directional probability."""

    name = "base_unified"

    #: >1 means the model consumes lookback windows rather than single rows.
    sequence_length: int = 1
    #: Kronos and the other foundation models read raw candles, not the feature matrix.
    requires_ohlcv: bool = False
    #: False for a pre-trained zero-shot model, which does no gradient updates in fit().
    learns_from_labels: bool = True

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.fitted_ = False
        self.fit_info_: Dict[str, Any] = {}

    def fit(
        self, fold: FoldInputs, y_return: np.ndarray, y_direction: np.ndarray
    ) -> "UnifiedEstimator":
        """Fit on ``fold.train_pos``. ``y_*`` are full-length, indexed by position."""
        raise NotImplementedError

    def predict_price(self, fold: FoldInputs, prev_close: np.ndarray) -> np.ndarray:
        """Absolute price per test row. ``prev_close`` is full-length."""
        raise NotImplementedError

    def predict_direction_proba(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        """(P(up), P(down)) per test row."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# The tabular / sequence baselines
# ---------------------------------------------------------------------------


class UnifiedMLModel(UnifiedEstimator):
    """
    An independent regressor and classifier presented as one model.

    Two heads rather than one, deliberately: thresholding a return regression at
    zero gives a hard label but no calibrated probability, and ROC-AUC and Brier
    score are meaningless without one. The regressor answers "how far", the
    classifier answers "which way, and how sure", and each is trained on the
    loss that matches the metric it is scored by.

    Both heads see identical rows, so the split costs nothing in fairness.
    """

    def __init__(
        self,
        name: str,
        regressor_cls: Type,
        classifier_cls: Type,
        reg_params: Optional[Dict[str, Any]] = None,
        clf_params: Optional[Dict[str, Any]] = None,
        seed: int = 42,
        sequence_length: int = 1,
    ):
        super().__init__(seed)
        self.name = name
        self.sequence_length = max(1, int(sequence_length))
        self._regressor_cls = regressor_cls
        self._classifier_cls = classifier_cls
        self._reg_params = dict(reg_params) if reg_params else None
        self._clf_params = dict(clf_params) if clf_params else None
        self._degenerate_base_rate: Optional[float] = None
        self.regressor = self._build_regressor()
        self.classifier = self._build_classifier()

    # -- construction ------------------------------------------------------

    def _build_regressor(self):
        return self._regressor_cls(params=self._reg_params)

    def _build_classifier(self):
        """
        Classifiers come from two families with different constructors.

        ``DirectionEstimator`` subclasses take ``seed=`` and tune themselves;
        ``BaseModel`` subclasses take ``params=``. Try the first, fall back to
        the second, and never overwrite a self-tuning estimator's search grid --
        that would silently disable the tuning the class exists to do.
        """
        try:
            return self._classifier_cls(seed=self.seed)
        except TypeError:
            params = dict(self._clf_params or {})
            params.setdefault("random_state", self.seed)
            return self._classifier_cls(params=params)

    def _reset(self) -> None:
        """Fresh, unfitted heads. Walk-forward refits per fold and must not warm-start."""
        self.regressor = self._build_regressor()
        self.classifier = self._build_classifier()
        self._degenerate_base_rate = None

    # -- head dispatch -----------------------------------------------------

    @property
    def _classifier_is_direction_estimator(self) -> bool:
        return hasattr(self.classifier, "predict_proba_up")

    def _classifier_input(self, X: np.ndarray):
        """DirectionEstimator subclasses index by column; BaseModel subclasses take arrays."""
        if self._classifier_is_direction_estimator:
            flat = X.reshape(len(X), -1) if X.ndim > 2 else X
            return pd.DataFrame(flat)
        return X

    def _fit_heads(self, X: np.ndarray, y_return: np.ndarray, y_direction: np.ndarray) -> None:
        self.regressor.fit(X, y_return)
        if len(np.unique(y_direction)) < 2:
            # A single-class training window. Fitting on it yields a classifier
            # that is right by construction and carries no information; the base
            # rate is the honest answer.
            self._degenerate_base_rate = float(np.mean(y_direction))
            logger.warning("%s: training window holds one class; serving its base rate", self.name)
        else:
            self._degenerate_base_rate = None
            self.classifier.fit(self._classifier_input(X), y_direction)

    def _proba_up(self, X: np.ndarray) -> np.ndarray:
        if self._degenerate_base_rate is not None:
            return np.full(len(X), self._degenerate_base_rate, dtype=np.float64)
        if self._classifier_is_direction_estimator:
            values = self.classifier.predict_proba_up(self._classifier_input(X))
            return np.asarray(values, dtype=np.float64)
        return np.asarray(self.classifier.predict_proba(X), dtype=np.float64)[:, 1]

    # -- UnifiedEstimator --------------------------------------------------

    def fit(self, fold: FoldInputs, y_return: np.ndarray, y_direction: np.ndarray) -> "UnifiedMLModel":
        # Re-seeded per fold rather than once per run, so a fold's result does
        # not depend on how many folds were fitted before it -- otherwise
        # evaluating one model in isolation would not reproduce its row in the
        # full comparison table.
        seed_everything(self.seed + len(fold.train_pos))
        self._reset()
        X_train, positions = fold.train_rows(self.sequence_length)
        if len(X_train) == 0:
            raise ValueError(f"{self.name}: no usable training rows in this fold")
        logger.info(
            "Fitting %s on %d rows (sequence_length=%d)", self.name, len(X_train), self.sequence_length
        )
        self._fit_heads(X_train, y_return[positions], y_direction[positions])
        self.fitted_ = True
        self.fit_info_ = {"n_train": int(len(X_train)), "sequence_length": self.sequence_length}
        return self

    def predict_price(self, fold: FoldInputs, prev_close: np.ndarray) -> np.ndarray:
        X_test, positions = fold.test_rows(self.sequence_length)
        # The regression target is a forward *return*, so the prediction has to
        # be re-based onto the last observed close to become a price.
        returns = np.asarray(self.regressor.predict(X_test), dtype=np.float64)
        return np.asarray(prev_close, dtype=np.float64)[positions] * (1.0 + returns)

    def predict_direction_proba(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        X_test, _ = fold.test_rows(self.sequence_length)
        p_up = np.clip(self._proba_up(X_test), 1e-6, 1 - 1e-6)
        return p_up, 1.0 - p_up

    # -- live serving ------------------------------------------------------

    def predict_latest(self, X_recent: np.ndarray, prev_close: float) -> Tuple[float, float]:
        """
        One prediction from the most recent scaled feature rows.

        ``X_recent`` is (n_rows, n_features), already scaled with the bundle's
        scaler. Sequence models use the trailing ``sequence_length`` rows; the
        tabular ones use the last row.
        """
        X_recent = np.asarray(X_recent, dtype=np.float32)
        if self.sequence_length > 1:
            if len(X_recent) < self.sequence_length:
                raise ValueError(
                    f"{self.name} needs {self.sequence_length} rows of history, got {len(X_recent)}"
                )
            X_input = X_recent[-self.sequence_length :][np.newaxis, :, :]
        else:
            X_input = X_recent[-1:]

        predicted_return = float(np.asarray(self.regressor.predict(X_input)).reshape(-1)[0])
        p_up = float(np.clip(self._proba_up(X_input), 1e-6, 1 - 1e-6)[0])
        return prev_close * (1.0 + predicted_return), p_up

    # -- persistence -------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Write both heads beside a small manifest.

        Each head appends its own extension when the path lacks one (``.json``
        for XGBoost, ``.pt`` for torch), so the sidecar paths are derived rather
        than assumed, and ``load`` rebuilds them by the same rule.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.regressor.save(str(self._reg_path(path)))
        classifier_path = self._clf_path(path)
        if hasattr(self.classifier, "save"):
            self.classifier.save(str(classifier_path))
        else:
            joblib.dump(self.classifier, str(classifier_path))
        path.write_text(
            json.dumps(
                {
                    "type": "UnifiedMLModel",
                    "name": self.name,
                    "sequence_length": self.sequence_length,
                    "degenerate_base_rate": self._degenerate_base_rate,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, filepath: str) -> None:
        path = Path(filepath)
        if path.exists():
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                self.sequence_length = max(1, int(manifest.get("sequence_length", self.sequence_length)))
                self._degenerate_base_rate = manifest.get("degenerate_base_rate")
            except (ValueError, OSError):
                logger.warning("Unreadable unified manifest at %s; keeping defaults", path)
        self.regressor.load(str(self._reg_path(path)))
        classifier_path = self._clf_path(path)
        if hasattr(self.classifier, "load"):
            self.classifier.load(str(classifier_path))
        else:
            self.classifier = joblib.load(str(classifier_path))
        self.fitted_ = True

    @staticmethod
    def _reg_path(path: Path) -> Path:
        return path.with_suffix(".reg" + path.suffix)

    @staticmethod
    def _clf_path(path: Path) -> Path:
        return path.with_suffix(".clf" + path.suffix)


# ---------------------------------------------------------------------------
# The existing dynamic ensemble, as a unified model
# ---------------------------------------------------------------------------


class UnifiedEnsembleModel(UnifiedEstimator):
    """
    The project's dynamic ensemble, kept as the baseline system to beat.

    Weights are not hard-coded. Each member is fitted on the first 80% of the
    training window, scored on the last 20%, and weighted by a softmax over that
    score -- the same "recent validation performance sets the weight" rule the
    existing :class:`~src.models.ensemble.EnsemblePredictor` applies to rolling
    Sharpe. The members are then refitted on the whole training window, so no
    member is scored off a model that saw only 80% of the data.

    Price and direction get separate weights: being the best return regressor
    and being the best-calibrated classifier are different jobs, and one member
    is frequently better at one than at the other.
    """

    name = "unified_ensemble"

    #: Lower = more aggressive concentration on the best member.
    TEMPERATURE = 0.5
    #: Share of the training window held back to score members on.
    INNER_VALIDATION_FRACTION = 0.2

    def __init__(
        self,
        members: Sequence[UnifiedMLModel],
        seed: int = 42,
        temperature: float = TEMPERATURE,
    ):
        super().__init__(seed)
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = list(members)
        self.temperature = float(temperature)
        self.price_weights_: Dict[str, float] = {}
        self.direction_weights_: Dict[str, float] = {}

    def fit(
        self, fold: FoldInputs, y_return: np.ndarray, y_direction: np.ndarray
    ) -> "UnifiedEnsembleModel":
        inner_train, inner_validation = self._inner_split(fold.train_pos)

        if inner_validation is None:
            uniform = 1.0 / len(self.members)
            self.price_weights_ = {member.name: uniform for member in self.members}
            self.direction_weights_ = dict(self.price_weights_)
        else:
            scoring_fold = FoldInputs(
                X_scaled=fold.X_scaled,
                index=fold.index,
                feature_columns=fold.feature_columns,
                train_pos=inner_train,
                test_pos=inner_validation,
                scaler=fold.scaler,
            )
            price_scores: Dict[str, float] = {}
            direction_scores: Dict[str, float] = {}
            for member in self.members:
                member.fit(scoring_fold, y_return, y_direction)
                X_validation, positions = scoring_fold.test_rows(member.sequence_length)
                # Scored in return space, which is comparable across members and
                # across symbols; price-space RMSE would rank mostly by price level.
                predicted = np.asarray(member.regressor.predict(X_validation), dtype=np.float64)
                rmse = float(np.sqrt(np.mean((y_return[positions] - predicted) ** 2)))
                p_up, _ = member.predict_direction_proba(scoring_fold)
                log_loss = _binary_log_loss(y_direction[positions], p_up)
                # Negated so that larger is better for both scores.
                price_scores[member.name] = -rmse
                direction_scores[member.name] = -log_loss

            self.price_weights_ = self._softmax(price_scores)
            self.direction_weights_ = self._softmax(direction_scores)

        for member in self.members:
            member.fit(fold, y_return, y_direction)

        self.fitted_ = True
        self.fit_info_ = {
            "price_weights": dict(self.price_weights_),
            "direction_weights": dict(self.direction_weights_),
        }
        logger.info(
            "Ensemble weights - price: %s | direction: %s",
            _format_weights(self.price_weights_),
            _format_weights(self.direction_weights_),
        )
        return self

    def predict_price(self, fold: FoldInputs, prev_close: np.ndarray) -> np.ndarray:
        return self._blend(
            {member.name: member.predict_price(fold, prev_close) for member in self.members},
            {member.name: fold.test_rows(member.sequence_length)[1] for member in self.members},
            fold.test_pos,
            self.price_weights_,
        )

    def predict_direction_proba(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        p_up = self._blend(
            {member.name: member.predict_direction_proba(fold)[0] for member in self.members},
            {member.name: fold.test_rows(member.sequence_length)[1] for member in self.members},
            fold.test_pos,
            self.direction_weights_,
        )
        p_up = np.clip(np.nan_to_num(p_up, nan=0.5), 1e-6, 1 - 1e-6)
        return p_up, 1.0 - p_up

    # -- internals ---------------------------------------------------------

    def _inner_split(self, train_pos: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Chronological 80/20 split of the training window; None when too short to be meaningful."""
        n = len(train_pos)
        cut = int(n * (1.0 - self.INNER_VALIDATION_FRACTION))
        if n < 120 or cut < 60 or n - cut < 20:
            return train_pos, None
        return train_pos[:cut], train_pos[cut:]

    def _softmax(self, scores: Dict[str, float]) -> Dict[str, float]:
        names = list(scores)
        values = np.array([scores[name] for name in names], dtype=np.float64)
        values = np.nan_to_num(values, nan=-1e6, posinf=-1e6, neginf=-1e6)
        # Standardised before the softmax so the temperature means the same
        # thing whether the scores are log losses (~0.7) or return RMSEs (~0.01).
        spread = float(values.std())
        values = (values - values.mean()) / (spread if spread > 1e-12 else 1.0)
        exponentiated = np.exp(values / max(self.temperature, 1e-6))
        weights = exponentiated / exponentiated.sum()
        return {name: float(weight) for name, weight in zip(names, weights)}

    @staticmethod
    def _blend(
        predictions: Dict[str, np.ndarray],
        positions: Dict[str, np.ndarray],
        test_pos: np.ndarray,
        weights: Dict[str, float],
    ) -> np.ndarray:
        """
        Weighted average over members, renormalised per row.

        Members disagree about which rows they can predict: a sequence member
        drops rows without a full lookback window. Rather than dropping those
        rows from the ensemble, each row is averaged over the members that did
        cover it, with their weights renormalised to sum to one.
        """
        lookup = {int(position): row for row, position in enumerate(test_pos)}
        total = np.zeros(len(test_pos), dtype=np.float64)
        weight_sum = np.zeros(len(test_pos), dtype=np.float64)
        for name, values in predictions.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            for value, position in zip(values, positions[name]):
                row = lookup.get(int(position))
                if row is not None and np.isfinite(value):
                    total[row] += weight * float(value)
                    weight_sum[row] += weight
        covered = weight_sum > 0
        total[covered] /= weight_sum[covered]
        total[~covered] = np.nan
        return total


def _binary_log_loss(y_true: np.ndarray, p_up: np.ndarray) -> float:
    p = np.clip(np.asarray(p_up, dtype=np.float64), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _format_weights(weights: Dict[str, float]) -> str:
    return ", ".join(f"{name}={weight:.3f}" for name, weight in sorted(weights.items()))


# ---------------------------------------------------------------------------
# Kronos
# ---------------------------------------------------------------------------


class UnifiedKronosModel(UnifiedEstimator):
    """
    Kronos: the candlestick foundation model, and the point of the comparison.

    Kronos does not consume the engineered feature matrix. It samples a
    distribution over tomorrow's candle from the raw OHLCV context, which yields
    both required outputs from one pass:

        price  = median of the sampled closes
        P(up)  = share of sampled closes above today's close

    ``fit`` performs no gradient updates -- the model is pre-trained -- but it
    is not a no-op: it records the training base rate the fallback path emits,
    and asserts the OHLCV side channel was supplied.
    """

    name = "unified_kronos"
    requires_ohlcv = True
    learns_from_labels = False

    def __init__(self, seed: int = 42, **kwargs):
        super().__init__(seed)
        from src.models.kronos_direction import KronosDirection

        self.model = KronosDirection(seed=seed, **kwargs)
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cached_p_up: Optional[np.ndarray] = None
        self._cached_bands: Optional[np.ndarray] = None

    def set_ohlcv_context(self, ohlcv: pd.DataFrame) -> None:
        """Supply the full raw bars. Only bars at or before each decision date are read."""
        self.model.set_ohlcv_context(ohlcv)

    def fit(self, fold: FoldInputs, y_return: np.ndarray, y_direction: np.ndarray) -> "UnifiedKronosModel":
        frame = pd.DataFrame(index=fold.index[fold.train_pos])
        self.model.fit(frame, y_direction[fold.train_pos])
        self._invalidate()
        self.fitted_ = True
        self.fit_info_ = dict(self.model.fit_info_)
        return self

    def _invalidate(self) -> None:
        self._cache_key = None
        self._cached_p_up = None
        self._cached_bands = None

    def _run(self, fold: FoldInputs) -> None:
        """
        One inference pass per fold, cached so price and direction share it.

        The cache is keyed on the actual test dates rather than on ``id()`` of
        the input frame: CPython reuses object ids after garbage collection, so
        an id-keyed cache can silently return a previous fold's probabilities.
        """
        dates = fold.index[fold.test_pos]
        key = (len(dates), pd.Timestamp(dates[0]).value, pd.Timestamp(dates[-1]).value)
        if self._cache_key == key:
            return

        logger.info("Running Kronos over %d rows (%s to %s)", len(dates), dates[0], dates[-1])
        frame = pd.DataFrame(index=dates)
        self._cached_p_up = np.asarray(self.model.predict_proba_up(frame), dtype=np.float64)
        self._cached_bands = self.model.price_bands_
        self._cache_key = key

    def predict_price(self, fold: FoldInputs, prev_close: np.ndarray) -> np.ndarray:
        """The median sampled close, falling back to the last close where sampling failed."""
        self._run(fold)
        last_close = np.asarray(prev_close, dtype=np.float64)[fold.test_pos]
        if self._cached_bands is None:
            return last_close
        # Columns are the 5th / 50th / 95th percentiles of the sampled closes.
        median = np.asarray(self._cached_bands, dtype=np.float64)[:, 1]
        # A chunk that failed to sample leaves NaN. A random-walk forecast --
        # today's close -- is the honest stand-in; NaN would poison the fold's
        # MAE and make one bad chunk look like a missing model.
        return np.where(np.isfinite(median), median, last_close)

    def predict_direction_proba(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        self._run(fold)
        p_up = self._cached_p_up
        if p_up is None:
            p_up = np.full(len(fold.test_pos), 0.5)
        p_up = np.clip(np.nan_to_num(p_up, nan=0.5), 1e-6, 1 - 1e-6)
        return p_up, 1.0 - p_up

    def predict_latest(self, as_of: pd.Timestamp, prev_close: float) -> Tuple[float, float]:
        """Serve one live prediction for the session after the bar ending at ``as_of``."""
        frame = pd.DataFrame(index=pd.DatetimeIndex([pd.Timestamp(as_of)]))
        p_up = float(np.clip(self.model.predict_proba_up(frame)[0], 1e-6, 1 - 1e-6))
        bands = self.model.price_bands_
        usable = bands is not None and len(bands) and np.isfinite(bands[0, 1])
        price = float(bands[0, 1]) if usable else float(prev_close)
        return price, p_up


# ---------------------------------------------------------------------------
# Optional comparison foundation models
# ---------------------------------------------------------------------------


class _UnivariateForecastModel(UnifiedEstimator):
    """
    Shared shape for the univariate foundation models (TimesFM, Chronos).

    Both are general time-series models rather than financial ones: they take a
    context window of closes and return a distribution over the next value.
    Neither is fine-tuned here, so both are zero-shot like Kronos, and both
    derive P(up) from the share of their forecast distribution above the last
    close. That keeps all three foundation models on the same footing.
    """

    requires_ohlcv = True
    learns_from_labels = False
    lookback = 256

    def __init__(self, seed: int = 42):
        super().__init__(seed)
        self._ohlcv: Optional[pd.DataFrame] = None
        self._pipeline: Optional[Any] = None
        self._base_rate = 0.5
        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_value: Tuple[np.ndarray, np.ndarray] = (np.empty(0), np.empty(0))

    def set_ohlcv_context(self, ohlcv: pd.DataFrame) -> None:
        if "Close" not in ohlcv.columns:
            raise KeyError("set_ohlcv_context requires a Close column")
        frame = ohlcv.copy()
        frame.index = pd.to_datetime(frame.index)
        self._ohlcv = frame.sort_index()

    def _load_pipeline(self):
        raise NotImplementedError

    def _forecast_samples(self, context: np.ndarray) -> np.ndarray:
        """Draws from the predictive distribution for the next step."""
        raise NotImplementedError

    def fit(
        self, fold: FoldInputs, y_return: np.ndarray, y_direction: np.ndarray
    ) -> "_UnivariateForecastModel":
        if self._ohlcv is None:
            raise RuntimeError(f"{self.name}.set_ohlcv_context() must be called before fit()")
        self._base_rate = float(np.mean(y_direction[fold.train_pos]))
        self._cache_key = None
        self.fitted_ = True
        self.fit_info_ = {"pretrained": True, "gradient_updates": 0, "train_base_rate": self._base_rate}
        return self

    def _predict(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        """(median price, P(up)) per test row, computed one decision date at a time."""
        assert self._ohlcv is not None
        self._load_pipeline()
        dates = fold.index[fold.test_pos]
        closes = self._ohlcv["Close"]

        prices = np.full(len(dates), np.nan)
        probabilities = np.full(len(dates), self._base_rate)

        for i, date in enumerate(dates):
            history = closes.loc[closes.index <= date]
            if len(history) < 32:
                continue
            context = history.tail(self.lookback).to_numpy(dtype=np.float64)
            try:
                samples = self._forecast_samples(context)
            except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
                logger.warning("%s failed on %s: %s", self.name, date, exc)
                continue
            prices[i] = float(np.median(samples))
            probabilities[i] = float(np.mean(samples > float(context[-1])))

        return prices, probabilities

    def _cached_predict(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        dates = fold.index[fold.test_pos]
        key = (len(dates), pd.Timestamp(dates[0]).value, pd.Timestamp(dates[-1]).value)
        if self._cache_key != key:
            self._cache_value = self._predict(fold)
            self._cache_key = key
        return self._cache_value

    def predict_latest(self, closes: pd.Series, prev_close: float) -> Tuple[float, float]:
        """
        Serve one live forecast from a Close series, for the API route.

        Same two estimators the walk-forward path uses -- the median of the
        forecast distribution, and the share of it above the last close -- so a
        served number means what the benchmarked number meant.
        """
        context = np.asarray(closes.tail(self.lookback), dtype=np.float64)
        if len(context) < 32:
            raise ValueError(f"{self.name} needs at least 32 bars of context, got {len(context)}")
        self._load_pipeline()
        samples = np.asarray(self._forecast_samples(context), dtype=np.float64).reshape(-1)
        p_up = float(np.clip(np.mean(samples > float(context[-1])), 1e-6, 1 - 1e-6))
        median = float(np.median(samples))
        return (median if np.isfinite(median) else float(prev_close)), p_up

    def predict_price(self, fold: FoldInputs, prev_close: np.ndarray) -> np.ndarray:
        prices, _ = self._cached_predict(fold)
        last_close = np.asarray(prev_close, dtype=np.float64)[fold.test_pos]
        return np.where(np.isfinite(prices), prices, last_close)

    def predict_direction_proba(self, fold: FoldInputs) -> Tuple[np.ndarray, np.ndarray]:
        _, probabilities = self._cached_predict(fold)
        p_up = np.clip(np.nan_to_num(probabilities, nan=self._base_rate), 1e-6, 1 - 1e-6)
        return p_up, 1.0 - p_up


class UnifiedTimesFMModel(_UnivariateForecastModel):
    """
    TimesFM 2.5, Google's decoder-only time-series foundation model.

    Optional: install with ``pip install -e ".[comparison]"``. It emits
    quantiles rather than samples, so the "share above last close" probability
    is read off the quantile curve rather than counted over draws.
    """

    name = "unified_timesfm"

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import timesfm

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        model.compile(
            timesfm.ForecastConfig(
                max_context=self.lookback, max_horizon=1, use_continuous_quantile_head=True
            )
        )
        self._pipeline = model
        return model

    def _forecast_samples(self, context: np.ndarray) -> np.ndarray:
        _, quantile_forecast = self._pipeline.forecast(horizon=1, inputs=[context])
        # (batch, horizon, quantile) -> the quantile levels for our single step.
        return np.asarray(quantile_forecast)[0, 0, :]


class UnifiedChronosModel(_UnivariateForecastModel):
    """
    Chronos-2, Amazon's tokenised time-series foundation model.

    Optional: install with ``pip install -e ".[comparison]"``. Chronos samples
    directly, so P(up) is the raw share of draws above the last close -- the
    same estimator Kronos uses.
    """

    name = "unified_chronos"
    num_samples = 128

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        import torch
        from chronos import BaseChronosPipeline

        self._pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float32,
        )
        return self._pipeline

    def _forecast_samples(self, context: np.ndarray) -> np.ndarray:
        import torch

        forecast = self._pipeline.predict(
            context=torch.tensor(context, dtype=torch.float32),
            prediction_length=1,
            num_samples=self.num_samples,
        )
        return np.asarray(forecast[0]).reshape(-1)


def foundation_model_availability(name: str) -> Tuple[bool, Optional[str]]:
    """Whether an optional foundation model can be constructed, and why not if it cannot."""
    if name in ("kronos", "unified_kronos"):
        from src.models.kronos_direction import kronos_availability

        return kronos_availability()
    module = {
        "timesfm": "timesfm",
        "unified_timesfm": "timesfm",
        "chronos": "chronos",
        "unified_chronos": "chronos",
    }.get(name)
    if module is None:
        return False, f"unknown foundation model {name!r}"
    try:
        __import__(module)
    except Exception as exc:  # noqa: BLE001 - any import failure means unavailable
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _build_xgboost(params: Optional[Dict[str, Any]] = None) -> UnifiedMLModel:
    return UnifiedMLModel(
        name="unified_xgboost",
        regressor_cls=XGBoostPriceRegressor,
        classifier_cls=GradientBoostingDirection,
        reg_params=params,
        clf_params=params,
    )


def _build_random_forest(params: Optional[Dict[str, Any]] = None) -> UnifiedMLModel:
    return UnifiedMLModel(
        name="unified_random_forest",
        regressor_cls=RandomForestPriceRegressor,
        classifier_cls=RandomForestModel,
        reg_params=params,
        clf_params=params,
    )


def _build_lstm(params: Optional[Dict[str, Any]] = None) -> UnifiedMLModel:
    settings = dict(params or {})
    sequence_length = int(settings.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))
    settings["sequence_length"] = sequence_length
    return UnifiedMLModel(
        name="unified_lstm",
        regressor_cls=LSTMPriceRegressor,
        classifier_cls=LSTMModel,
        reg_params=settings,
        clf_params=settings,
        sequence_length=sequence_length,
    )


# Deliberately modest, and shared by the benchmark, the training pipeline and
# the ensemble's own members. One source of truth matters here: an ensemble
# whose LSTM member is configured differently from the standalone LSTM is not
# comparable to it, and the difference is invisible in the results table.
DEFAULT_MODEL_PARAMS: Dict[str, Dict[str, Any]] = {
    "unified_xgboost": {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
    "unified_random_forest": {"n_estimators": 300, "max_depth": 10},
    "unified_lstm": {
        "epochs": 40,
        "batch_size": 32,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "units": 64,
    },
}


#: Names that can be trained into a saved bundle and served by the API.
UNIFIED_FACTORIES: Dict[str, Callable[..., UnifiedMLModel]] = {
    "unified_xgboost": _build_xgboost,
    "unified_random_forest": _build_random_forest,
    "unified_lstm": _build_lstm,
}

#: Every model the benchmark can evaluate, including those that are never persisted.
BENCHMARK_MODELS: Tuple[str, ...] = (
    "unified_xgboost",
    "unified_random_forest",
    "unified_lstm",
    "unified_ensemble",
    "unified_kronos",
    "unified_timesfm",
    "unified_chronos",
)

#: The set that runs by default: the four existing baselines plus Kronos.
DEFAULT_BENCHMARK_MODELS: Tuple[str, ...] = BENCHMARK_MODELS[:5]


def build_unified_model(
    name: str, params: Optional[Dict[str, Any]] = None, seed: int = 42
) -> UnifiedEstimator:
    """
    Instantiate a benchmark model by name, raising on an unknown name rather than guessing.

    ``params`` overrides :data:`DEFAULT_MODEL_PARAMS` for that model. The
    ensemble ignores it and builds each member from that member's own defaults,
    so its components stay identical to the standalone runs they are compared
    against.
    """
    if name in UNIFIED_FACTORIES:
        return UNIFIED_FACTORIES[name](params if params is not None else DEFAULT_MODEL_PARAMS.get(name))
    if name == "unified_ensemble":
        return UnifiedEnsembleModel(
            members=[
                UNIFIED_FACTORIES[member](DEFAULT_MODEL_PARAMS.get(member))
                for member in ("unified_xgboost", "unified_random_forest", "unified_lstm")
            ],
            seed=seed,
        )
    if name == "unified_kronos":
        return UnifiedKronosModel(seed=seed, **(params or {}))
    if name == "unified_timesfm":
        return UnifiedTimesFMModel(seed=seed)
    if name == "unified_chronos":
        return UnifiedChronosModel(seed=seed)
    raise ValueError(f"Unknown unified model {name!r}; known: {sorted(BENCHMARK_MODELS)}")


_KRONOS_SINGLETON: Optional[UnifiedKronosModel] = None


def get_kronos_singleton() -> UnifiedKronosModel:
    """
    A process-wide Kronos instance for the API.

    Loading the tokeniser and weights takes seconds; doing it per request would
    dominate the response time. Zero-shot means there is no per-symbol state to
    keep separate, so one instance serves every request.
    """
    global _KRONOS_SINGLETON
    if _KRONOS_SINGLETON is None:
        logger.info("Initialising the Kronos singleton")
        model = UnifiedKronosModel()
        # No training window exists at serve time, so mark it ready directly and
        # leave the fallback base rate at the uninformative 0.5.
        model.fitted_ = True
        model.model.fitted_ = True
        model.model.train_base_rate_ = 0.5
        _KRONOS_SINGLETON = model
    return _KRONOS_SINGLETON
