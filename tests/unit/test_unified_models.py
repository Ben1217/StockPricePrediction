"""
Unit tests for the unified price-and-direction models.

The tests that matter here are the structural ones. A benchmark that runs and
prints plausible numbers is not evidence of anything if the fold plumbing under
it leaks, mislabels, or silently drops rows -- and none of those failures show
up as an exception. So the checks below pin the properties the comparison rests
on:

* ``TestFoldPreparation`` proves the scaler never sees a test row, and that a
  sequence model's lookback window is drawn from bars that precede its
  prediction date rather than following it.
* ``TestAlignment`` pins predictions to the rows they are scored against, which
  is the failure that would quietly inflate or destroy every metric at once.
* ``TestKronosCaching`` covers the ``id()``-reuse bug: a cache keyed on object
  identity returns a previous fold's answers once CPython recycles an address.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.models.unified_evaluation import (  # noqa: E402
    evaluate_unified_walk_forward,
    price_metrics,
)
from src.models.unified_models import (  # noqa: E402
    DEFAULT_MODEL_PARAMS,
    FoldInputs,
    UnifiedEnsembleModel,
    UnifiedEstimator,
    build_unified_model,
    prepare_fold,
)


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """A deterministic feature matrix on a business-day index."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=600)
    return pd.DataFrame(
        rng.normal(size=(len(index), 6)),
        index=index,
        columns=[f"f{i}" for i in range(6)],
    )


@pytest.fixture
def targets(feature_frame: pd.DataFrame):
    """Forward returns and their signs, plus the close each return is applied to."""
    rng = np.random.default_rng(11)
    y_return = rng.normal(0.0, 0.01, size=len(feature_frame))
    y_direction = (y_return > 0).astype(np.int8)
    prev_close = 100.0 + np.cumsum(rng.normal(0.0, 0.5, size=len(feature_frame)))
    return y_return, y_direction, prev_close


class TestFoldPreparation:
    """The scaling and windowing every model shares."""

    def test_scaler_is_fitted_on_training_rows_only(self, feature_frame):
        train_pos = np.arange(0, 400)
        test_pos = np.arange(401, 500)
        fold = prepare_fold(feature_frame, train_pos, test_pos)

        # The scaler's parameters must be reproducible from the training rows
        # alone. If a test row contributed, these would not match.
        expected_mean = feature_frame.to_numpy()[train_pos].mean(axis=0)
        np.testing.assert_allclose(fold.scaler.mean_, expected_mean, rtol=1e-9)

        # And the training rows must standardise to roughly zero mean.
        assert np.abs(fold.X_scaled[train_pos].mean(axis=0)).max() < 1e-4

    def test_scaling_a_test_row_ignores_every_other_test_row(self, feature_frame):
        """
        Changing one test row must not move any other row's scaled value.

        This is the structural proof that the scaler is a fixed transform at
        test time rather than a statistic refitted over the test block.
        """
        train_pos = np.arange(0, 400)
        test_pos = np.arange(401, 500)
        baseline = prepare_fold(feature_frame, train_pos, test_pos)

        perturbed_frame = feature_frame.copy()
        perturbed_frame.iloc[450] += 1000.0
        perturbed = prepare_fold(perturbed_frame, train_pos, test_pos)

        untouched = [p for p in test_pos if p != 450]
        np.testing.assert_allclose(
            baseline.X_scaled[untouched], perturbed.X_scaled[untouched], rtol=1e-6
        )

    def test_sequence_window_ends_at_its_own_position(self, feature_frame):
        """Window for position p covers p-L+1..p -- history, never the future."""
        fold = prepare_fold(feature_frame, np.arange(0, 400), np.arange(401, 500))
        windows, positions = fold.test_rows(sequence_length=60)

        assert windows.shape == (len(positions), 60, feature_frame.shape[1])
        for row, position in enumerate(positions):
            np.testing.assert_allclose(
                windows[row], fold.X_scaled[position - 59 : position + 1], rtol=1e-6
            )
            # The last bar of the window is the decision bar itself.
            np.testing.assert_allclose(windows[row][-1], fold.X_scaled[position], rtol=1e-6)

    def test_warmup_rows_without_a_full_window_are_dropped(self, feature_frame):
        """Positions closer to the series start than the lookback cannot form a window."""
        fold = prepare_fold(feature_frame, np.arange(0, 400), np.arange(401, 500))
        _, positions = fold.train_rows(sequence_length=60)

        assert positions.min() == 59
        assert len(positions) == 400 - 59

    def test_tabular_rows_are_returned_unwindowed(self, feature_frame):
        fold = prepare_fold(feature_frame, np.arange(0, 400), np.arange(401, 500))
        rows, positions = fold.test_rows(sequence_length=1)

        assert rows.ndim == 2
        np.testing.assert_array_equal(positions, np.arange(401, 500))


class _CountingModel(UnifiedEstimator):
    """Records what it was handed, and predicts the row index it saw."""

    name = "counting"

    def __init__(self, sequence_length: int = 1):
        super().__init__()
        self.sequence_length = sequence_length
        self.train_positions = None
        self.test_positions = None

    def fit(self, fold, y_return, y_direction):
        _, self.train_positions = fold.train_rows(self.sequence_length)
        self.fitted_ = True
        return self

    def predict_price(self, fold, prev_close):
        _, positions = fold.test_rows(self.sequence_length)
        self.test_positions = positions
        return np.asarray(prev_close)[positions]

    def predict_direction_proba(self, fold):
        _, positions = fold.test_rows(self.sequence_length)
        p_up = np.full(len(positions), 0.6)
        return p_up, 1.0 - p_up


class TestAlignment:
    """Predictions must line up with the rows they are scored against."""

    def test_sequence_model_scores_only_the_rows_it_predicted(self, feature_frame, targets):
        y_return, y_direction, prev_close = targets
        model = _CountingModel(sequence_length=60)

        result = evaluate_unified_walk_forward(
            model,
            feature_frame,
            y_return,
            y_direction,
            prev_close,
            test_size=50,
            n_splits=2,
            min_train=200,
        )

        assert result["n_splits"] == 2
        for fold in result["per_fold"]:
            assert fold["test_size"] == 50

    def test_a_no_change_forecast_separates_the_two_r_squareds(self, feature_frame, targets):
        """
        "Tomorrow equals today" scores near 1.0 on price and near 0.0 on return.

        That gap is the whole reason both are reported. The price-level figure
        is dominated by the price level itself, so it flatters a forecast that
        has predicted nothing; the return figure is measured against the mean
        return, where a no-change forecast is worth approximately zero.
        """
        y_return, y_direction, prev_close = targets
        result = evaluate_unified_walk_forward(
            _CountingModel(),
            feature_frame,
            y_return,
            y_direction,
            prev_close,
            test_size=50,
            n_splits=2,
            min_train=200,
        )

        # Not exactly zero: R-squared is measured against the mean realised
        # return, which is only approximately zero over a 50-day window.
        assert abs(result["price_r2_return"]) < 0.05
        # The price-level figure is far higher for the very same forecast. The
        # size of the gap is what the two-column report exists to expose.
        assert result["price_r2"] - result["price_r2_return"] > 0.5
        assert result["price_mae"] > 0

    def test_folds_are_chronological_and_do_not_overlap(self, feature_frame, targets):
        y_return, y_direction, prev_close = targets
        result = evaluate_unified_walk_forward(
            _CountingModel(),
            feature_frame,
            y_return,
            y_direction,
            prev_close,
            test_size=50,
            n_splits=3,
            min_train=200,
        )

        folds = result["per_fold"]
        starts = [fold["test_start"] for fold in folds]
        assert starts == sorted(starts)
        for earlier, later in zip(folds, folds[1:]):
            assert earlier["test_end"] < later["test_start"]
            # Expanding window: each fold trains on more data than the last.
            assert earlier["train_size"] < later["train_size"]


class TestPriceMetrics:
    def test_perfect_forecast_scores_zero_error(self):
        truth = np.array([100.0, 101.0, 99.0, 102.0])
        prev = np.array([99.0, 100.0, 101.0, 99.0])
        scores = price_metrics(truth, truth, prev)

        assert scores["mae"] == pytest.approx(0.0)
        assert scores["rmse"] == pytest.approx(0.0)
        assert scores["mape"] == pytest.approx(0.0)
        assert scores["r2"] == pytest.approx(1.0)
        assert scores["r2_return"] == pytest.approx(1.0)

    def test_mape_is_a_percentage(self):
        truth = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        scores = price_metrics(truth, predicted, np.array([100.0, 200.0]))

        assert scores["mape"] == pytest.approx(10.0)


class TestEnsembleWeighting:
    """The dynamic ensemble picks its weights from held-out performance."""

    def test_weights_are_normalised_and_favour_the_better_member(self):
        ensemble = UnifiedEnsembleModel(members=[build_unified_model("unified_xgboost")])
        scores = {"good": -0.1, "bad": -10.0}
        weights = ensemble._softmax(scores)

        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["good"] > weights["bad"]

    def test_identical_scores_give_uniform_weights(self):
        ensemble = UnifiedEnsembleModel(members=[build_unified_model("unified_xgboost")])
        weights = ensemble._softmax({"a": -1.0, "b": -1.0, "c": -1.0})

        for weight in weights.values():
            assert weight == pytest.approx(1 / 3)

    def test_blend_renormalises_over_the_members_that_covered_a_row(self):
        """
        A row a sequence member could not predict is averaged over the rest.

        Without the renormalisation the row would be scaled down by the missing
        member's weight and read as a forecast of a lower price.
        """
        test_pos = np.array([10, 11, 12])
        blended = UnifiedEnsembleModel._blend(
            predictions={"tabular": np.array([1.0, 1.0, 1.0]), "sequence": np.array([3.0, 3.0])},
            positions={"tabular": test_pos, "sequence": np.array([11, 12])},
            test_pos=test_pos,
            weights={"tabular": 0.5, "sequence": 0.5},
        )

        assert blended[0] == pytest.approx(1.0)  # only the tabular member covered it
        assert blended[1] == pytest.approx(2.0)  # both, evenly weighted
        assert blended[2] == pytest.approx(2.0)


class TestKronosCaching:
    """
    A fold cache must key on the dates it answered for, not on object identity.

    CPython reuses ``id()`` values once an object is collected, so an id-keyed
    cache can hand fold two the probabilities it computed for fold one.
    """

    def test_cache_key_distinguishes_folds_with_identical_shapes(self):
        pytest.importorskip("einops")
        from src.models.unified_models import UnifiedKronosModel

        calls = []

        class _StubKronos:
            price_bands_ = None
            fit_info_: dict = {}
            sample_count = 8
            lookback = 32

            def set_ohlcv_context(self, ohlcv):
                pass

            def fit(self, X, y):
                return self

            def predict_proba_up(self, X):
                calls.append(tuple(X.index))
                self.price_bands_ = np.tile([99.0, 100.0, 101.0], (len(X), 1))
                return np.full(len(X), 0.5 + 0.01 * len(calls))

        model = UnifiedKronosModel.__new__(UnifiedKronosModel)
        UnifiedEstimator.__init__(model, seed=42)
        model.model = _StubKronos()
        model._cache_key = None
        model._cached_p_up = None
        model._cached_bands = None

        index = pd.bdate_range("2020-01-01", periods=200)
        base = FoldInputs(
            X_scaled=np.zeros((200, 3), dtype=np.float32),
            index=index,
            feature_columns=["a", "b", "c"],
            train_pos=np.arange(0, 100),
            test_pos=np.arange(100, 120),
        )
        # Same length, same shapes, different dates.
        other = FoldInputs(
            X_scaled=base.X_scaled,
            index=index,
            feature_columns=base.feature_columns,
            train_pos=np.arange(0, 120),
            test_pos=np.arange(120, 140),
        )

        first = model.predict_direction_proba(base)[0]
        # A second call on the same fold must reuse the cached pass.
        model.predict_direction_proba(base)
        assert len(calls) == 1

        second = model.predict_direction_proba(other)[0]
        assert len(calls) == 2, "second fold reused the first fold's cached predictions"
        assert not np.allclose(first, second)

    def test_failed_samples_fall_back_to_the_last_close(self):
        """A NaN band must become a random-walk forecast, never a NaN metric."""
        pytest.importorskip("einops")
        from src.models.unified_models import UnifiedKronosModel

        class _StubKronos:
            price_bands_ = None

            def predict_proba_up(self, X):
                bands = np.full((len(X), 3), np.nan)
                bands[0] = [98.0, 100.0, 102.0]
                self.price_bands_ = bands
                return np.full(len(X), 0.5)

        model = UnifiedKronosModel.__new__(UnifiedKronosModel)
        UnifiedEstimator.__init__(model, seed=42)
        model.model = _StubKronos()
        model._cache_key = None
        model._cached_p_up = None
        model._cached_bands = None

        index = pd.bdate_range("2020-01-01", periods=50)
        fold = FoldInputs(
            X_scaled=np.zeros((50, 2), dtype=np.float32),
            index=index,
            feature_columns=["a", "b"],
            train_pos=np.arange(0, 40),
            test_pos=np.arange(40, 43),
        )
        prev_close = np.arange(50, dtype=float) + 200.0

        prices = model.predict_price(fold, prev_close)

        assert np.isfinite(prices).all()
        assert prices[0] == pytest.approx(100.0)
        np.testing.assert_allclose(prices[1:], prev_close[41:43])


class TestRegistry:
    def test_ensemble_members_match_the_standalone_configurations(self):
        """
        An ensemble whose members are tuned differently is not comparable to them.

        This is the check that keeps the "existing ensemble" row in the results
        table an honest baseline rather than a fourth, differently configured
        model.
        """
        ensemble = build_unified_model("unified_ensemble")
        for member in ensemble.members:
            standalone = build_unified_model(member.name)
            assert member._reg_params == standalone._reg_params
            assert member.sequence_length == standalone.sequence_length
            assert member._reg_params == DEFAULT_MODEL_PARAMS[member.name]

    def test_unknown_model_name_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown unified model"):
            build_unified_model("unified_crystal_ball")

    def test_lstm_is_the_only_sequence_model(self):
        assert build_unified_model("unified_lstm").sequence_length > 1
        assert build_unified_model("unified_xgboost").sequence_length == 1
        assert build_unified_model("unified_random_forest").sequence_length == 1


class TestReproducibility:
    """
    Two runs of the same fold must produce the same numbers.

    Neither LSTM wrapper seeds torch on its own, so without the per-fold seeding
    in ``UnifiedMLModel.fit`` the LSTM's accuracy drifts by around a point
    between runs -- the same magnitude as the differences the benchmark exists
    to measure. A comparison that cannot be reproduced cannot be argued with.
    """

    def _fold_and_targets(self):
        index = pd.bdate_range("2020-01-01", periods=320)
        rng = np.random.default_rng(17)
        signal = rng.normal(size=len(index))
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(index))}, index=index)
        y_return = 0.02 * signal + rng.normal(0.0, 0.002, size=len(index))
        return (
            prepare_fold(X, np.arange(0, 250), np.arange(250, 320)),
            y_return,
            (y_return > 0).astype(np.int8),
        )

    def test_lstm_refit_reproduces_its_predictions(self):
        fold, y_return, y_direction = self._fold_and_targets()
        prev_close = np.full(320, 100.0)
        params = {"epochs": 3, "sequence_length": 20, "units": 16}

        runs = []
        for _ in range(2):
            model = build_unified_model("unified_lstm", params)
            model.fit(fold, y_return, y_direction)
            runs.append(
                (model.predict_price(fold, prev_close), model.predict_direction_proba(fold)[0])
            )

        np.testing.assert_allclose(runs[0][0], runs[1][0], rtol=1e-6)
        np.testing.assert_allclose(runs[0][1], runs[1][1], rtol=1e-6)

    def test_a_fold_does_not_depend_on_folds_fitted_before_it(self):
        """
        Seeding is per fold, not per run.

        Otherwise evaluating a model on its own would not reproduce its row in
        the full comparison, because the RNG would be at a different point.
        """
        fold, y_return, y_direction = self._fold_and_targets()
        prev_close = np.full(320, 100.0)
        params = {"epochs": 3, "sequence_length": 20, "units": 16}

        alone = build_unified_model("unified_lstm", params)
        alone.fit(fold, y_return, y_direction)
        expected = alone.predict_price(fold, prev_close)

        after_others = build_unified_model("unified_lstm", params)
        # Two unrelated fits first, advancing every global RNG.
        for _ in range(2):
            after_others.fit(fold, y_return, y_direction)
        actual = after_others.predict_price(fold, prev_close)

        np.testing.assert_allclose(expected, actual, rtol=1e-6)


class TestPersistence:
    """
    A saved bundle has to come back as the same model.

    Both heads persist to sidecar files beside the manifest, and each appends
    its own extension when the path lacks one. A mismatch between the write path
    and the read path is silent until inference, where it surfaces as a missing
    file rather than as a wrong number -- but the sequence length lives in that
    manifest, and getting *it* wrong is silent all the way through.
    """

    def _fitted_model(self, name: str, tmp_path: Path):
        index = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(13)
        signal = rng.normal(size=len(index))
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(index))}, index=index)
        y_return = 0.02 * signal + rng.normal(0.0, 0.002, size=len(index))
        y_direction = (y_return > 0).astype(np.int8)

        model = build_unified_model(name)
        fold = prepare_fold(X, np.arange(0, 300), np.arange(300, 400))
        model.fit(fold, y_return, y_direction)
        return model, fold, X

    @pytest.mark.parametrize("name", ["unified_xgboost", "unified_random_forest"])
    def test_round_trip_reproduces_predictions(self, name, tmp_path):
        model, fold, _ = self._fitted_model(name, tmp_path)
        prev_close = np.full(400, 100.0)

        before_price = model.predict_price(fold, prev_close)
        before_p_up = model.predict_direction_proba(fold)[0]

        path = tmp_path / "model.joblib"
        model.save(str(path))
        assert path.exists(), "the manifest itself must be written"

        reloaded = build_unified_model(name)
        reloaded.load(str(path))

        np.testing.assert_allclose(reloaded.predict_price(fold, prev_close), before_price, rtol=1e-5)
        np.testing.assert_allclose(reloaded.predict_direction_proba(fold)[0], before_p_up, rtol=1e-5)

    def test_manifest_carries_the_sequence_length(self, tmp_path):
        """
        Serving reads the lookback from here, so a wrong value is silent.

        A model saved with a 60-bar lookback and reloaded as a 1-bar one would
        keep answering, on a window of the wrong length, with no error.
        """
        import json

        model = build_unified_model("unified_lstm")
        path = tmp_path / "model.joblib"
        # No fit needed: the manifest records configuration, not learned state.
        model.regressor.save = lambda p: Path(p).write_text("stub", encoding="utf-8")
        model.classifier.save = lambda p: Path(p).write_text("stub", encoding="utf-8")
        model.save(str(path))

        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["sequence_length"] == model.sequence_length > 1


class TestTabularModelsEndToEnd:
    """The real estimators, on synthetic data with a signal they should find."""

    def test_xgboost_learns_a_planted_signal(self):
        index = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(3)
        signal = rng.normal(size=len(index))
        X = pd.DataFrame({"signal": signal, "noise": rng.normal(size=len(index))}, index=index)

        # The target is the signal column plus a little noise, so a working
        # model must beat the base rate by a wide margin.
        y_return = 0.02 * signal + rng.normal(0.0, 0.002, size=len(index))
        y_direction = (y_return > 0).astype(np.int8)
        prev_close = np.full(len(index), 100.0)

        result = evaluate_unified_walk_forward(
            build_unified_model("unified_xgboost"),
            X,
            y_return,
            y_direction,
            prev_close,
            test_size=60,
            n_splits=2,
            min_train=200,
        )

        assert result["direction_accuracy"] > 0.85
        assert result["price_r2_return"] > 0.8

    def test_a_pure_noise_target_lands_near_the_base_rate(self):
        """No planted signal, no edge. A model that "finds" one here is leaking."""
        index = pd.bdate_range("2020-01-01", periods=500)
        rng = np.random.default_rng(5)
        X = pd.DataFrame(rng.normal(size=(len(index), 4)), index=index, columns=list("abcd"))
        y_return = rng.normal(0.0, 0.01, size=len(index))
        y_direction = (y_return > 0).astype(np.int8)
        prev_close = np.full(len(index), 100.0)

        result = evaluate_unified_walk_forward(
            build_unified_model("unified_xgboost"),
            X,
            y_return,
            y_direction,
            prev_close,
            test_size=60,
            n_splits=2,
            min_train=200,
        )

        assert 0.3 < result["direction_accuracy"] < 0.7
        assert result["price_r2_return"] < 0.2
