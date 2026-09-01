"""
Unit tests for the next-day direction pipeline.

The tests that matter most here are not the ones checking a number comes out;
they are the ones checking that a number could not have come out dishonestly:

* ``TestFeatureCausality`` perturbs a future bar and asserts no earlier feature
  row moves. That is a structural proof of no look-ahead, stronger than any
  accuracy-based check, because it holds whatever the model does.
* ``TestLeakageCheck`` runs the shuffled-label check on clean synthetic data and
  requires it to pass — the check itself has to be trustworthy before its
  verdict on real data means anything.
* ``TestExecutionTiming`` pins the backtest to the open of the bar *after* the
  signal. A one-bar slip there is the difference between a study that reports an
  edge and a study that reports a fill it never had.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.backtesting.direction_backtest import (  # noqa: E402
    _tradeable_returns,
    run_long_flat_backtest,
    select_threshold,
)
from src.data.direction_data import (  # noqa: E402
    apply_dividend_adjustment,
    clean_daily_bars,
    frame_content_hash,
)
from src.features.chart_patterns import CHART_PATTERN_FEATURE_COLUMNS  # noqa: E402
from src.features.direction_features import (  # noqa: E402
    DIRECTION_EXTRA_FEATURE_COLUMNS,
    DIRECTION_FEATURE_COLUMNS,
    build_direction_dataset,
)
from src.features.feature_engineering import (  # noqa: E402
    STATIONARY_REGRESSION_FEATURE_COLUMNS,
)
from src.models.direction_metrics import (  # noqa: E402
    accuracy_edge_test,
    calibration_bins,
    classification_metrics,
    classifier_skill,
    matthews_corrcoef,
    required_sample_size,
    roc_auc,
    wilson_interval,
)
from src.models.direction_models import ALL_FACTORIES, BASELINE_FACTORIES, build_model  # noqa: E402
from src.models.direction_pipeline import (  # noqa: E402
    run_shuffled_label_check,
    run_walk_forward,
)
from src.models.walk_forward import expanding_window_splits  # noqa: E402


def make_bars(n=900, seed=0, drift=0.0004, vol=0.011, reversion=0.0):
    """Synthetic OHLCV. ``reversion`` plants a genuine next-day mean-reversion effect."""
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = rng.normal(drift, vol) - reversion * returns[t - 1]
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1_000_000, 9_000_000, n)},
        index=pd.bdate_range("2016-01-04", periods=n),
    )


@pytest.fixture(scope="module")
def bars():
    return make_bars()


@pytest.fixture(scope="module")
def dataset(bars):
    return build_direction_dataset(bars)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

class TestDataLayer:
    def test_dividend_adjustment_preserves_intraday_ratios(self):
        """
        The whole point of adjusting all four price columns by one factor: the
        shape of the bar is unchanged, only its level moves.
        """
        frame = pd.DataFrame({
            "Open": [100.0, 101.0], "High": [102.0, 103.0], "Low": [99.0, 100.0],
            "Close": [101.0, 102.0], "Adj Close": [99.0, 102.0], "Volume": [1e6, 1e6],
        }, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

        adjusted = apply_dividend_adjustment(frame)

        assert "Adj Close" not in adjusted.columns
        # Close now equals the old Adj Close: everything is on one basis.
        assert np.allclose(adjusted["Close"], frame["Adj Close"])
        # And every scale-free quantity is untouched.
        for original, new in [(frame, adjusted)]:
            assert np.allclose(
                (original["High"] - original["Low"]) / original["Close"],
                (new["High"] - new["Low"]) / new["Close"],
            )
            assert np.allclose(original["Close"] / original["Open"], new["Close"] / new["Open"])
        # Volume is not a price and must not be scaled.
        assert np.allclose(adjusted["Volume"], frame["Volume"])

    def test_mixing_bases_would_corrupt_the_range(self):
        """
        Guards the bug the module exists to prevent: an unadjusted High/Low over
        an adjusted Close gives a different range on any dividend date.
        """
        frame = pd.DataFrame({
            "Open": [100.0], "High": [102.0], "Low": [99.0],
            "Close": [101.0], "Adj Close": [99.0], "Volume": [1e6],
        }, index=pd.to_datetime(["2024-01-02"]))

        mixed = (frame["High"] - frame["Low"]) / frame["Adj Close"]
        consistent = (frame["High"] - frame["Low"]) / frame["Close"]
        assert not np.allclose(mixed, consistent)

    def test_clean_daily_bars_hygiene(self):
        index = pd.to_datetime([
            "2024-01-02", "2024-01-02", "2024-01-04", "2024-01-03", "2024-01-05",
        ]).tz_localize("America/New_York")
        frame = pd.DataFrame({
            "Open": [100.0, 100.5, 101.0, 100.0, 102.0],
            "High": [101.0, 101.5, 102.0, 101.0, 103.0],
            "Low": [99.0, 99.5, 100.0, 99.0, 101.0],
            "Close": [100.5, 101.0, 101.5, 100.5, 102.5],
            "Volume": [1e6, 2e6, 3e6, 4e6, 0],  # last row has zero volume
        }, index=index)

        cleaned = clean_daily_bars(frame)

        assert cleaned.index.tz is None, "index must be timezone-naive"
        assert cleaned.index.is_monotonic_increasing, "index must be sorted"
        assert not cleaned.index.duplicated().any(), "duplicates must be dropped"
        assert pd.Timestamp("2024-01-05") not in cleaned.index, "zero-volume bar must be dropped"
        # keep='last' on the duplicate pair.
        assert cleaned.loc[pd.Timestamp("2024-01-02"), "Close"] == 101.0

    def test_content_hash_is_deterministic_and_sensitive(self, bars):
        assert frame_content_hash(bars) == frame_content_hash(bars.copy())
        nudged = bars.copy()
        nudged.iloc[10, nudged.columns.get_loc("Close")] += 0.01
        assert frame_content_hash(bars) != frame_content_hash(nudged)


# ---------------------------------------------------------------------------
# Leakage: the structural proof
# ---------------------------------------------------------------------------

class TestFeatureCausality:
    def test_future_bars_cannot_change_earlier_features(self, bars):
        """
        Perturb the tail of the series and assert every feature row before the
        perturbation is bit-identical.

        This is the definitive no-look-ahead test. A centred rolling window, a
        reversed series, a full-series scaler fit, or a ``bfill`` would all move
        earlier rows and fail here, regardless of what any model then scores.
        """
        cut = 600
        # float throughout so the in-place scaling does not trip pandas' int64
        # downcast warning on Volume.
        perturbed = bars.astype(float)
        perturbed.iloc[cut:] *= 1.25  # a violent change to everything after `cut`
        bars = bars.astype(float)

        original = build_direction_dataset(bars)
        modified = build_direction_dataset(perturbed)

        shared = original.index.intersection(modified.index)
        # Only rows whose entire feature window sits before the perturbation.
        safe = shared[shared < bars.index[cut]]
        assert len(safe) > 200, "need a meaningful number of rows to make this test mean anything"

        pd.testing.assert_frame_equal(
            original.features.loc[safe], modified.features.loc[safe], check_exact=True,
        )

    def test_label_uses_only_the_next_bar(self, bars):
        """The target is the sign of the next close-to-close return, exactly."""
        dataset = build_direction_dataset(bars)
        close = bars["Close"].reindex(dataset.index)
        next_close = bars["Close"].shift(-1).reindex(dataset.index)
        expected_return = next_close / close - 1.0

        assert np.allclose(dataset.forward_return, expected_return)
        assert np.array_equal(dataset.labels.to_numpy(), (expected_return > 0).astype(np.int8))

    def test_execution_prices_are_the_next_bar(self, bars):
        """``entry_open``/``exit_close`` must be bar t+1, never bar t."""
        dataset = build_direction_dataset(bars)
        assert np.allclose(dataset.entry_open, bars["Open"].shift(-1).reindex(dataset.index))
        assert np.allclose(dataset.exit_close, bars["Close"].shift(-1).reindex(dataset.index))
        # And never the signal bar itself.
        assert not np.allclose(dataset.entry_open, bars["Open"].reindex(dataset.index))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

class TestDatasetConstruction:
    def test_feature_set_is_stationary_plus_directional_plus_chart(self, dataset):
        """
        The feature set is exactly its three declared groups, and nothing else.

        The count is derived from those groups rather than written out, so
        adding a column to one of them updates this test instead of breaking it
        — which is what a hardcoded total did when the chart-pattern group was
        introduced.
        """
        assert dataset.feature_columns == DIRECTION_FEATURE_COLUMNS

        groups = (
            STATIONARY_REGRESSION_FEATURE_COLUMNS,
            DIRECTION_EXTRA_FEATURE_COLUMNS,
            CHART_PATTERN_FEATURE_COLUMNS,
        )
        for group in groups:
            for column in group:
                assert column in dataset.feature_columns

        # No duplicates across the groups, and no column from anywhere else.
        expected = [column for group in groups for column in group]
        assert len(expected) == len(set(expected))
        assert sorted(dataset.feature_columns) == sorted(expected)

    def test_no_missing_values_and_nothing_filled(self, dataset, bars):
        assert not dataset.features.isna().any().any()
        assert not dataset.labels.isna().any()
        # Rows were dropped, not imputed: the dataset is strictly shorter than
        # the bar count by the warm-up plus the unresolvable final row.
        assert len(dataset) < len(bars)
        assert dataset.meta["rows_dropped_incomplete"] > 0
        # The last bar has no next bar, so it cannot carry a label.
        assert bars.index[-1] not in dataset.index

    def test_directional_features_match_their_definitions(self, bars):
        dataset = build_direction_dataset(bars)
        index = dataset.index
        expected_gap = (bars["Open"] / bars["Close"].shift(1) - 1.0).reindex(index)
        expected_intraday = (bars["Close"] / bars["Open"] - 1.0).reindex(index)
        expected_position = (
            (bars["Close"] - bars["Low"]) / (bars["High"] - bars["Low"])
        ).reindex(index)

        assert np.allclose(dataset.features["Overnight_Gap"], expected_gap)
        assert np.allclose(dataset.features["Intraday_Return"], expected_intraday)
        assert np.allclose(dataset.features["Close_Position_In_Range"], expected_position)
        for lag in (1, 2, 3):
            expected_sign = np.sign(bars["Close"].pct_change().shift(lag)).reindex(index)
            assert np.allclose(dataset.features[f"Return_Sign_Lag{lag}"], expected_sign)

    def test_deadband_drops_the_middle_and_reports_the_shift(self, bars):
        plain = build_direction_dataset(bars)
        banded = build_direction_dataset(bars, deadband_sigma_multiple=0.5)

        assert len(banded) < len(plain)
        assert banded.meta["rows_dropped_deadband"] > 0
        # Every surviving row is outside the band, so no near-zero moves remain.
        assert banded.forward_return.abs().min() > 0
        # The base rate changed, and the metadata says so rather than hiding it.
        assert banded.meta["base_rate"] == pytest.approx(banded.base_rate)
        assert banded.meta["deadband_sigma_multiple"] == 0.5

    def test_regime_features_are_rejected(self, bars):
        with pytest.raises(ValueError, match="include_regime"):
            build_direction_dataset(bars, feature_config={"include_regime": True})

    def test_slice_keeps_every_series_aligned(self, dataset):
        window = dataset.slice(np.arange(10, 40))
        assert len(window) == 30
        assert window.features.index.equals(window.labels.index)
        assert window.features.index.equals(window.entry_open.index)
        assert window.features.index.equals(window.exit_close.index)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

class TestExpandingWindowSplits:
    def test_folds_are_chronological_disjoint_and_embargoed(self):
        splits = expanding_window_splits(1000, test_size=63, n_splits=4, embargo=1, min_train=252)
        assert len(splits) == 4

        previous_test_end = -1
        for train_positions, test_positions in splits:
            assert len(test_positions) == 63, "test windows must be a fixed size"
            assert set(train_positions).isdisjoint(set(test_positions))
            assert train_positions[-1] < test_positions[0], "training must precede the test window"
            # Embargo of 1 leaves exactly one purged row between them.
            assert test_positions[0] - train_positions[-1] - 1 == 1
            assert test_positions[0] > previous_test_end, "test windows must not overlap"
            previous_test_end = test_positions[-1]

    def test_training_window_expands(self):
        splits = expanding_window_splits(1000, test_size=63, n_splits=4, embargo=1, min_train=252)
        sizes = [len(train) for train, _ in splits]
        assert sizes == sorted(sizes) and len(set(sizes)) == len(sizes)

    def test_embargo_scales_with_horizon(self):
        splits = expanding_window_splits(1000, test_size=63, n_splits=2, embargo=5, min_train=252)
        for train_positions, test_positions in splits:
            assert test_positions[0] - train_positions[-1] - 1 == 5

    def test_short_series_yields_no_folds(self):
        assert expanding_window_splits(100, test_size=63, n_splits=4, min_train=252) == []

    def test_folds_below_min_train_are_dropped_not_shrunk(self):
        splits = expanding_window_splits(500, test_size=63, n_splits=4, embargo=1, min_train=252)
        assert all(len(train) >= 252 for train, _ in splits)
        assert len(splits) < 4


# ---------------------------------------------------------------------------
# Models and baselines
# ---------------------------------------------------------------------------

    @pytest.mark.parametrize("name", sorted(ALL_FACTORIES))
    def test_interface_and_probability_range(self, dataset, name):
        train, test = dataset.slice(np.arange(500)), dataset.slice(np.arange(500, 700))
        try:
            model = build_model(name, seed=7)
            if hasattr(model, "set_ohlcv_context") and dataset.ohlcv is not None:
                model.set_ohlcv_context(dataset.ohlcv)
            model.fit(train.features, train.labels)
        except (ImportError, Exception) as exc:
            if "TabPFN" in str(type(exc).__name__) or "tabpfn" in str(exc).lower() or isinstance(exc, ImportError):
                pytest.skip(f"Optional model {name} requires authentication/dependency: {exc}")
            raise

        probabilities = model.predict_proba_up(test.features)
        predictions = model.predict(test.features)

        assert probabilities.shape == (len(test),)
        assert np.all((probabilities > 0) & (probabilities < 1)), "log loss must stay finite"
        assert set(np.unique(predictions)).issubset({0, 1})
        assert model.fit_info_["n_train"] == 500

    def test_unfitted_model_refuses_to_predict(self, dataset):
        with pytest.raises(RuntimeError, match="must be fitted"):
            build_model("logistic").predict_proba_up(dataset.features)

    def test_unknown_model_name_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("transformer")

    def test_majority_baseline_is_constant(self, dataset):
        train, test = dataset.slice(np.arange(500)), dataset.slice(np.arange(500, 700))
        model = build_model("majority", seed=7).fit(train.features, train.labels)
        predictions = model.predict(test.features)
        assert len(np.unique(predictions)) == 1
        assert len(np.unique(model.predict_proba_up(test.features))) == 1

    def test_momentum_and_reversal_are_opposites(self, dataset):
        train, test = dataset.slice(np.arange(500)), dataset.slice(np.arange(500, 700))
        momentum = build_model("momentum_1", seed=7).fit(train.features, train.labels)
        reversal = build_model("reversal_1", seed=7).fit(train.features, train.labels)

        moves = test.features["Daily_Return"].to_numpy() != 0
        assert np.all(
            momentum.predict(test.features)[moves] != reversal.predict(test.features)[moves]
        )

    def test_degenerate_training_window_falls_back(self, dataset):
        """A single-class training window must not raise; it emits its base rate."""
        train = dataset.slice(np.arange(300))
        single_class = np.ones(len(train), dtype=np.int8)
        model = build_model("logistic", seed=7).fit(train.features, single_class)
        probabilities = model.predict_proba_up(train.features)
        assert model.fit_info_.get("degenerate_single_class") is True
        assert np.allclose(probabilities, probabilities[0])

    def test_seeding_is_reproducible(self, dataset):
        train, test = dataset.slice(np.arange(500)), dataset.slice(np.arange(500, 700))
        first = build_model("gradient_boosting", seed=11).fit(train.features, train.labels)
        second = build_model("gradient_boosting", seed=11).fit(train.features, train.labels)
        assert np.allclose(first.predict_proba_up(test.features), second.predict_proba_up(test.features))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_reproduces_the_worked_significance_example(self):
        """252 test days, 53% base rate, 56.0% accuracy -> z ~ 0.95, p ~ 0.17."""
        result = accuracy_edge_test(0.560, 0.530, 252)
        assert result["edge_pp"] == pytest.approx(3.0, abs=1e-6)
        assert result["standard_error_pp"] == pytest.approx(3.15, abs=0.01)
        assert result["z"] == pytest.approx(0.95, abs=0.01)
        assert result["p_value_one_sided"] == pytest.approx(0.17, abs=0.005)
        assert result["significant"] is False

    def test_a_3pp_edge_needs_about_a_thousand_days(self):
        needed = required_sample_size(0.03, alpha=0.05, power=0.80)
        assert needed > 1000, "a 3pp edge cannot be established on one year of test data"
        # Bare significance at 50% power still needs several hundred days.
        assert required_sample_size(0.03, alpha=0.05, power=0.50) > 700

    def test_wilson_interval_brackets_and_narrows(self):
        low_n = wilson_interval(35, 63)
        high_n = wilson_interval(560, 1008)  # same proportion, 16x the data
        assert low_n[0] < 35 / 63 < low_n[1]
        assert (high_n[1] - high_n[0]) < (low_n[1] - low_n[0])
        assert 0.0 <= low_n[0] and low_n[1] <= 1.0

    def test_metrics_match_sklearn(self):
        rng = np.random.default_rng(3)
        y = (rng.random(400) < 0.53).astype(int)
        probabilities = np.clip(0.5 + 0.15 * (y - 0.5) + rng.normal(0, 0.12, 400), 1e-6, 1 - 1e-6)
        predictions = (probabilities >= 0.5).astype(int)

        sklearn_metrics = pytest.importorskip("sklearn.metrics")
        assert roc_auc(y, probabilities) == pytest.approx(
            sklearn_metrics.roc_auc_score(y, probabilities)
        )
        assert matthews_corrcoef(y, predictions) == pytest.approx(
            sklearn_metrics.matthews_corrcoef(y, predictions)
        )
        result = classification_metrics(y, predictions, probabilities)
        assert result["balanced_accuracy"] == pytest.approx(
            sklearn_metrics.balanced_accuracy_score(y, predictions), abs=1e-6
        )
        assert result["brier_score"] == pytest.approx(
            sklearn_metrics.brier_score_loss(y, probabilities), abs=1e-6
        )
        assert result["log_loss"] == pytest.approx(
            sklearn_metrics.log_loss(y, probabilities), abs=1e-6
        )

    def test_constant_predictor_scores_zero_mcc_and_zero_skill(self):
        rng = np.random.default_rng(5)
        y = (rng.random(300) < 0.55).astype(int)
        base_rate = float(y.mean())
        constant = np.full(300, base_rate)

        assert matthews_corrcoef(y, np.ones(300, dtype=int)) == 0.0
        skill = classifier_skill(y, constant, base_rate)
        assert skill["brier_skill_score"] == pytest.approx(0.0, abs=1e-9)
        assert skill["log_loss_skill_score"] == pytest.approx(0.0, abs=1e-9)
        assert skill["prediction_std"] == pytest.approx(0.0, abs=1e-9)

    def test_skill_is_negative_when_probabilities_are_worse_than_the_base_rate(self):
        y = np.array([1, 1, 1, 0, 0, 0] * 20)
        # Confidently backwards.
        probabilities = np.where(y == 1, 0.2, 0.8)
        assert classifier_skill(y, probabilities, 0.5)["brier_skill_score"] < 0

    def test_auc_is_none_for_a_single_class_window(self):
        assert roc_auc(np.ones(20, dtype=int), np.linspace(0.1, 0.9, 20)) is None

    def test_calibration_bins_cover_every_observation(self):
        rng = np.random.default_rng(9)
        y = (rng.random(500) < 0.5).astype(int)
        probabilities = np.clip(rng.normal(0.52, 0.03, 500), 1e-6, 1 - 1e-6)
        bins = calibration_bins(y, probabilities, n_bins=10)
        assert sum(b["count"] for b in bins) == 500
        for point in bins:
            assert point["observed_ci_low"] <= point["observed_frequency"] <= point["observed_ci_high"]


# ---------------------------------------------------------------------------
# Backtest execution and costs
# ---------------------------------------------------------------------------

class TestExecutionTiming:
    def test_captured_return_is_open_to_close_not_close_to_close(self, dataset):
        """
        The signal is formed at the close of t and filled at the open of t+1, so
        the overnight gap is NOT captured. If these two ever agree, the backtest
        has silently started paying itself a fill it never got.
        """
        window = dataset.slice(np.arange(200, 400))
        tradeable = _tradeable_returns(window.entry_open, window.exit_close)

        assert np.allclose(tradeable, window.exit_close / window.entry_open - 1.0)
        assert not np.allclose(tradeable, window.forward_return.to_numpy())

    def test_no_position_is_taken_on_the_signal_bar(self, dataset):
        window = dataset.slice(np.arange(200, 400))
        result = run_long_flat_backtest(
            np.full(len(window), 0.9), window.entry_open, window.exit_close,
            threshold=0.5, cost_bps=0.0, index=window.index,
        )
        # Every row transacts at the *next* bar's prices, carried on the dataset.
        assert np.allclose(result.equity_curve["entry_open"], window.entry_open)
        assert np.allclose(result.equity_curve["exit_close"], window.exit_close)


class TestBacktestArithmetic:
    def test_breakeven_equals_a_constant_edge(self):
        """With a constant +10bps gross move per day, the edge dies at 10 bps."""
        opens = np.full(100, 100.0)
        closes = opens * 1.001
        result = run_long_flat_backtest(
            np.full(100, 0.9), opens, closes, threshold=0.5, cost_bps=0.0
        )
        assert result.breakeven["mean_gross_return_per_trade_bps"] == pytest.approx(10.0, abs=1e-6)
        assert result.breakeven["breakeven_cost_bps_positive"] == pytest.approx(10.0, abs=1e-3)

    def test_at_breakeven_the_strategy_returns_nothing(self):
        opens = np.full(100, 100.0)
        closes = opens * 1.001
        result = run_long_flat_backtest(
            np.full(100, 0.9), opens, closes, threshold=0.5, cost_bps=10.0
        )
        assert result.metrics["total_return"] == pytest.approx(0.0, abs=1e-9)

    def test_returns_fall_monotonically_with_cost(self):
        opens = np.full(100, 100.0)
        closes = opens * 1.002
        totals = [
            run_long_flat_backtest(
                np.full(100, 0.9), opens, closes, threshold=0.5, cost_bps=cost
            ).metrics["total_return"]
            for cost in (0.0, 5.0, 10.0, 20.0)
        ]
        assert totals == sorted(totals, reverse=True)

    def test_costs_are_charged_only_on_active_days(self):
        opens = np.full(10, 100.0)
        closes = np.full(10, 100.5)
        probabilities = np.array([0.9, 0.1] * 5)
        result = run_long_flat_backtest(
            probabilities, opens, closes, threshold=0.5, cost_bps=10.0
        )
        curve = result.equity_curve
        assert np.all(curve.loc[curve["position"] == 0, "cost"] == 0)
        assert np.allclose(curve.loc[curve["position"] == 1, "cost"], 10.0 / 1e4)
        assert result.metrics["round_trips"] == 5

    def test_benchmark_is_buy_at_first_open_hold_to_last_close(self):
        rng = np.random.default_rng(4)
        opens = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50)))
        closes = opens * (1 + rng.normal(0, 0.01, 50))
        result = run_long_flat_backtest(
            np.full(50, 0.9), opens, closes, threshold=0.5, cost_bps=0.0
        )
        assert result.equity_curve["benchmark_equity"].iloc[-1] == pytest.approx(
            closes[-1] / opens[0], rel=1e-9
        )

    def test_per_row_thresholds_are_applied_row_by_row(self):
        probabilities = np.linspace(0.40, 0.70, 10)
        thresholds = np.concatenate([np.full(5, 0.45), np.full(5, 0.65)])
        result = run_long_flat_backtest(
            probabilities, np.full(10, 100.0), np.full(10, 100.5),
            threshold=thresholds, cost_bps=0.0,
        )
        expected = (probabilities > thresholds).astype(float)
        assert np.array_equal(result.equity_curve["position"].to_numpy(), expected)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length mismatch"):
            run_long_flat_backtest(np.full(5, 0.6), np.full(4, 100.0), np.full(4, 101.0))


class TestThresholdSelection:
    def test_thin_candidates_are_rejected(self):
        # 90 rows just above 0.51, 10 rows at 0.99: only the loosest threshold
        # clears the minimum trade count.
        probabilities = np.concatenate([np.full(90, 0.51), np.full(10, 0.99)])
        choice = select_threshold(
            probabilities, np.full(100, 100.0), np.full(100, 100.5),
            cost_bps=5.0, min_trades=15,
        )
        assert choice.threshold == 0.50
        assert not choice.fell_back
        thin = [c for c in choice.candidates if c["threshold"] > 0.51]
        assert thin and all(not c["eligible"] for c in thin)

    def test_falls_back_visibly_when_nothing_is_eligible(self):
        choice = select_threshold(
            np.full(50, 0.30), np.full(50, 100.0), np.full(50, 100.5),
            cost_bps=5.0, min_trades=15,
        )
        assert choice.fell_back is True

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError, match="objective"):
            select_threshold(
                np.full(50, 0.6), np.full(50, 100.0), np.full(50, 100.5), objective="profit",
            )


# ---------------------------------------------------------------------------
# The leakage check itself
# ---------------------------------------------------------------------------

class TestLeakageCheck:
    def test_shuffled_labels_collapse_to_chance_on_clean_data(self, dataset):
        splits = expanding_window_splits(
            len(dataset), test_size=63, n_splits=3, embargo=1, min_train=252
        )
        result = run_shuffled_label_check(
            dataset, splits, model_name="logistic", seed=1, n_repeats=10
        )
        assert result["passed"] is True, (
            f"permuted labels still predict: {result['mean_shuffled_accuracy']:.4f} vs "
            f"null {result['mean_null_accuracy']:.4f} (p={result['p_value_one_sided']})"
        )
        assert result["mean_difference"] == pytest.approx(0.0, abs=0.03)

    def test_it_passes_on_data_with_real_signal_too(self):
        """
        Shuffling must destroy the relationship even when one genuinely exists.
        A check that fires here would be useless: it would flag every dataset
        the model can actually learn from.
        """
        dataset = build_direction_dataset(make_bars(n=900, seed=2, reversion=0.45))
        splits = expanding_window_splits(
            len(dataset), test_size=63, n_splits=3, embargo=1, min_train=252
        )
        result = run_shuffled_label_check(
            dataset, splits, model_name="logistic", seed=1, n_repeats=10
        )
        assert result["passed"] is True

    def test_too_few_fits_is_inconclusive_not_a_pass(self, dataset):
        splits = expanding_window_splits(
            len(dataset), test_size=63, n_splits=2, embargo=1, min_train=252
        )
        result = run_shuffled_label_check(
            dataset, splits, model_name="logistic", seed=1, n_repeats=2
        )
        assert result["passed"] is None
        assert "inconclusive" in result["note"]


# ---------------------------------------------------------------------------
# End-to-end walk-forward
# ---------------------------------------------------------------------------

class TestWalkForwardRun:
    @pytest.fixture(scope="class")
    def run_result(self, dataset):
        return run_walk_forward(
            dataset, model_name="logistic", n_folds=3, test_size=63,
            min_train=252, cost_bps=10.0, seed=42,
        )

    def test_report_is_json_serializable(self, run_result):
        import json
        json.dumps(run_result.report)  # raises on numpy scalars or NaN

    def test_every_baseline_is_scored_on_every_fold(self, run_result):
        for fold in run_result.report["folds"]:
            assert set(fold["baselines"]) == set(BASELINE_FACTORIES)
            for metrics in fold["baselines"].values():
                assert 0.0 <= metrics["accuracy"] <= 1.0
        assert set(run_result.report["pooled"]["baselines"]) == set(BASELINE_FACTORIES)

    def test_constant_classifier_comparison_is_always_present(self, run_result):
        """Section 6's requirement: the constant-classifier reference is automatic."""
        pooled = run_result.report["pooled"]
        assert "majority" in pooled["baselines"]
        assert pooled["edge_vs_best_baseline"]["standard_error_pp"] is not None
        assert "brier_skill_score" in pooled["model"]["skill"]
        assert pooled["model"]["skill"]["reference_rate"] > 0

    def test_every_accuracy_carries_an_interval(self, run_result):
        pooled = run_result.report["pooled"]["model"]
        assert pooled["accuracy_ci_low"] <= pooled["accuracy"] <= pooled["accuracy_ci_high"]
        for fold in run_result.report["folds"]:
            metrics = fold["model"]
            assert metrics["accuracy_ci_low"] <= metrics["accuracy"] <= metrics["accuracy_ci_high"]

    def test_threshold_validation_window_precedes_the_test_window(self, run_result):
        """The threshold is a fitted parameter and must not see the scored rows."""
        for fold in run_result.report["folds"]:
            threshold = fold["threshold"]
            if threshold.get("source") != "validation":
                continue
            assert threshold["validation_range"][1] < fold["test_range"][0]

    def test_folds_are_chronological_and_do_not_overlap(self, run_result):
        ranges = [fold["test_range"] for fold in run_result.report["folds"]]
        for earlier, later in zip(ranges, ranges[1:]):
            assert earlier[1] < later[0]

    def test_pooled_row_count_matches_the_folds(self, run_result):
        folds = run_result.report["folds"]
        assert run_result.report["pooled"]["n_test_rows"] == sum(f["n_test"] for f in folds)
        assert len(run_result.equity_curve) == run_result.report["pooled"]["n_test_rows"]

    def test_verdict_enumerates_every_criterion(self, run_result):
        verdict = run_result.report["verdict"]
        expected = {
            "beats_best_baseline_accuracy", "accuracy_edge_is_significant",
            "positive_probability_skill", "beats_buy_and_hold_after_costs",
            "survives_the_charged_cost", "passes_leakage_check",
        }
        assert set(verdict["criteria"]) == expected
        assert verdict["ship"] == (not verdict["failed_criteria"])

    def test_noise_data_does_not_ship(self, run_result):
        """
        The synthetic series is a near-random walk. Anything that "ships" on it
        is a bug in the harness, not a discovery.
        """
        assert run_result.report["verdict"]["ship"] is False

    def test_embargo_below_the_horizon_is_refused(self, bars):
        dataset = build_direction_dataset(bars, horizon=5)
        with pytest.raises(ValueError, match="smaller than the target horizon"):
            run_walk_forward(dataset, n_folds=2, embargo=1, run_leakage_check=False)

    def test_too_little_data_fails_loudly(self, bars):
        small = build_direction_dataset(bars.iloc[:300])
        with pytest.raises(ValueError, match="cannot support"):
            run_walk_forward(small, n_folds=4, test_size=63, min_train=252, run_leakage_check=False)

    def test_run_is_reproducible(self, dataset):
        first = run_walk_forward(
            dataset, n_folds=2, cost_bps=10.0, seed=42, run_leakage_check=False
        )
        second = run_walk_forward(
            dataset, n_folds=2, cost_bps=10.0, seed=42, run_leakage_check=False
        )
        assert (
            first.report["pooled"]["model"]["accuracy"]
            == second.report["pooled"]["model"]["accuracy"]
        )
        pd.testing.assert_frame_equal(first.equity_curve, second.equity_curve)
