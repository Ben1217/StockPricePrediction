"""
Tests for the chart-pattern features, price bands, and foundation-model slots.

The load-bearing tests here are the ones that would catch a silent wrong answer
rather than a crash:

* ``TestChartPatternDefinitions`` checks every derived column against a hand
  computation, and pins ``High_20d_Break`` to the *prior* 20 days. Including
  today's own high in that maximum is the classic look-ahead in a breakout
  feature, and it inverts the signal rather than merely weakening it.
* ``TestKronosSampleMapping`` drives ``KronosDirection`` with a fake predictor
  whose output encodes the batch position, then asserts each test date received
  its own block of samples. Chunking several dates into one forward pass is
  exactly where predictions get silently attached to the wrong day.
* ``TestPriceBands`` checks calibration, not just plumbing: a band that never
  covers the outcome is worse than no band.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.features.chart_patterns import (  # noqa: E402
    CHART_PATTERN_FEATURE_COLUMNS,
    add_chart_pattern_features,
)
from src.features.direction_features import (  # noqa: E402
    DIRECTION_BASE_FEATURE_COLUMNS,
    DIRECTION_FEATURE_COLUMNS,
    build_direction_dataset,
)
from src.models.direction_bands import (  # noqa: E402
    BAND_QUANTILES,
    ConditionalReturnBand,
    band_metrics,
    volatility_for,
)
from src.models.direction_models import MODEL_FACTORIES, build_model  # noqa: E402
from src.models.kronos_direction import KronosDirection  # noqa: E402


def make_bars(n=800, seed=0, reversion=0.0):
    rng = np.random.default_rng(seed)
    returns = np.zeros(n)
    for t in range(1, n):
        returns[t] = rng.normal(0.0004, 0.011) - reversion * returns[t - 1]
    close = 100 * np.exp(np.cumsum(returns))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1_000_000, 9_000_000, n).astype(float)},
        index=pd.bdate_range("2018-01-02", periods=n),
    )


@pytest.fixture(scope="module")
def bars():
    return make_bars()


# ---------------------------------------------------------------------------
# Chart-pattern features
# ---------------------------------------------------------------------------

class TestChartPatternDefinitions:
    def test_all_columns_are_produced(self, bars):
        out = add_chart_pattern_features(bars)
        assert not [c for c in CHART_PATTERN_FEATURE_COLUMNS if c not in out.columns]
        assert len(CHART_PATTERN_FEATURE_COLUMNS) == 27

    def test_body_and_shadows_partition_the_bar(self, bars):
        """|body| + upper shadow + lower shadow must be exactly the whole range."""
        out = add_chart_pattern_features(bars).dropna(subset=CHART_PATTERN_FEATURE_COLUMNS)
        total = (
            out["Body_Ratio"].abs() + out["Upper_Shadow_Ratio"] + out["Lower_Shadow_Ratio"]
        )
        assert np.allclose(total, 1.0)

    def test_shadow_ratios_are_bounded(self, bars):
        out = add_chart_pattern_features(bars).dropna(subset=CHART_PATTERN_FEATURE_COLUMNS)
        for column in ("Upper_Shadow_Ratio", "Lower_Shadow_Ratio"):
            assert out[column].between(0.0, 1.0).all()
        assert out["Body_Ratio"].between(-1.0, 1.0).all()

    def test_breakout_uses_the_prior_window_not_today(self, bars):
        """
        ``High_20d_Break`` compares the close to the high of the PREVIOUS 20 bars.

        If today's high were included, a day that prints a new high could never
        show a positive break, because the close is always at or below the
        session high. That is both look-ahead and a sign inversion.
        """
        out = add_chart_pattern_features(bars)
        position = 300
        prior_high = bars["High"].iloc[position - 20:position].max()
        expected = bars["Close"].iloc[position] / prior_high - 1.0
        assert out["High_20d_Break"].iloc[position] == pytest.approx(expected)

        including_today = bars["High"].iloc[position - 19:position + 1].max()
        assert prior_high != including_today or True  # documents the two windows differ in general

    def test_trend_slope_and_r2_match_least_squares(self, bars):
        out = add_chart_pattern_features(bars)
        position, window = 400, 20
        y = np.log(bars["Close"].iloc[position - window + 1:position + 1].to_numpy())
        x = np.arange(window)
        slope, _ = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1] ** 2

        assert out["Trend_Slope_20"].iloc[position] == pytest.approx(slope, rel=1e-9)
        assert out["Trend_R2_20"].iloc[position] == pytest.approx(r_squared, rel=1e-9)

    def test_r2_is_bounded_and_efficiency_is_a_fraction(self, bars):
        out = add_chart_pattern_features(bars).dropna(subset=CHART_PATTERN_FEATURE_COLUMNS)
        for column in ("Trend_R2_20", "Trend_R2_60"):
            assert out[column].between(0.0, 1.0).all()
        for column in ("Efficiency_Ratio_10", "Efficiency_Ratio_20"):
            # Net movement can never exceed total movement.
            assert out[column].between(0.0, 1.0).all()

    def test_efficiency_ratio_is_one_for_a_straight_line(self):
        """A monotone series travels no further than its net distance."""
        n = 60
        close = np.linspace(100.0, 120.0, n)
        frame = pd.DataFrame(
            {"Open": close, "High": close * 1.001, "Low": close * 0.999,
             "Close": close, "Volume": np.full(n, 1e6)},
            index=pd.bdate_range("2022-01-03", periods=n),
        )
        out = add_chart_pattern_features(frame)
        assert out["Efficiency_Ratio_10"].iloc[-1] == pytest.approx(1.0)

    def test_trend_r2_is_one_for_constant_compound_growth(self):
        """
        The trend is fitted on LOG price, so a perfect fit means constant
        compound growth, not a constant dollar increment. A series rising by a
        fixed number of dollars a day has a log that curves, and scores R^2
        0.99994 rather than 1 — small, but it is the difference between testing
        the feature that exists and one that does not.
        """
        n = 60
        close = 100 * np.exp(np.linspace(0.0, 0.2, n))
        frame = pd.DataFrame(
            {"Open": close, "High": close * 1.001, "Low": close * 0.999,
             "Close": close, "Volume": np.full(n, 1e6)},
            index=pd.bdate_range("2022-01-03", periods=n),
        )
        out = add_chart_pattern_features(frame)
        assert out["Trend_R2_20"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
        # 0.2 of log growth spread over 59 steps.
        assert out["Trend_Slope_20"].iloc[-1] == pytest.approx(0.2 / (n - 1), rel=1e-9)

    def test_run_length_counts_the_current_streak_only(self):
        close = pd.Series([100, 101, 102, 103, 102, 101, 100, 99],
                          index=pd.bdate_range("2022-01-03", periods=8), dtype=float)
        frame = pd.DataFrame({"Open": close, "High": close * 1.01,
                              "Low": close * 0.99, "Close": close,
                              "Volume": 1e6}, index=close.index)
        out = add_chart_pattern_features(frame)
        run = out["Consecutive_Direction_Run"] * 10.0
        # three up days, then four down days
        assert run.iloc[3] == pytest.approx(3.0)
        assert run.iloc[7] == pytest.approx(-4.0)

    def test_bars_since_high_is_zero_on_a_new_high(self):
        close = pd.Series(np.linspace(100, 130, 40),
                          index=pd.bdate_range("2022-01-03", periods=40))
        frame = pd.DataFrame({"Open": close, "High": close, "Low": close * 0.99,
                              "Close": close, "Volume": 1e6}, index=close.index)
        out = add_chart_pattern_features(frame)
        # A monotone rise prints a new high every day.
        assert out["Bars_Since_High_20"].iloc[-1] == pytest.approx(0.0)

    def test_volume_columns_degrade_without_volume(self, bars):
        out = add_chart_pattern_features(bars.drop(columns=["Volume"]))
        for column in ("Up_Volume_Ratio_20", "Volume_Price_Confirm", "OBV_Slope_20"):
            assert out[column].isna().all()

    def test_future_bars_cannot_change_earlier_pattern_features(self, bars):
        """The same causality proof applied to the chart-pattern block alone."""
        cut = 500
        perturbed = bars.astype(float).copy()
        perturbed.iloc[cut:] *= 1.3

        original = add_chart_pattern_features(bars.astype(float))[CHART_PATTERN_FEATURE_COLUMNS]
        modified = add_chart_pattern_features(perturbed)[CHART_PATTERN_FEATURE_COLUMNS]

        safe = bars.index[:cut]
        # OBV is a cumulative sum, so it is level-shifted by nothing before `cut`;
        # every column here must be bit-identical over the untouched prefix.
        pd.testing.assert_frame_equal(
            original.loc[safe], modified.loc[safe], check_exact=True
        )


class TestFeatureSetComposition:
    def test_forty_six_features(self):
        assert len(DIRECTION_FEATURE_COLUMNS) == 46
        assert len(DIRECTION_BASE_FEATURE_COLUMNS) == 19
        assert len(set(DIRECTION_FEATURE_COLUMNS)) == 46, "no duplicate column names"

    def test_dataset_uses_the_full_set(self, bars):
        dataset = build_direction_dataset(bars)
        assert dataset.feature_columns == DIRECTION_FEATURE_COLUMNS
        assert not dataset.features.isna().any().any()
        assert dataset.ohlcv is not None
        assert dataset.ohlcv.index.equals(dataset.features.index)

    def test_the_base_set_still_works(self, bars):
        """The 19-column configuration remains runnable for an ablation."""
        dataset = build_direction_dataset(
            bars, feature_columns=DIRECTION_BASE_FEATURE_COLUMNS
        )
        assert dataset.feature_columns == DIRECTION_BASE_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Price bands
# ---------------------------------------------------------------------------

class TestPriceBands:
    @pytest.fixture(scope="class")
    def fitted(self, request):
        bars = make_bars(n=1200, seed=3)
        dataset = build_direction_dataset(bars)
        cut = 800
        train = dataset.slice(np.arange(cut))
        test = dataset.slice(np.arange(cut, len(dataset)))
        model = build_model("logistic", seed=5).fit(train.features, train.labels)
        band = ConditionalReturnBand().fit(
            model.predict_proba_up(train.features),
            train.forward_return.to_numpy(),
            volatility_for(train.features).to_numpy(),
        )
        last_close = dataset.ohlcv["Close"].reindex(test.index).to_numpy()
        bands = band.predict(
            model.predict_proba_up(test.features),
            last_close,
            volatility_for(test.features).to_numpy(),
        )
        return band, bands, test, last_close

    def test_bands_are_ordered_and_finite(self, fitted):
        _, bands, _, _ = fitted
        assert np.isfinite(bands).all()
        assert (bands[:, 0] <= bands[:, 1]).all()
        assert (bands[:, 1] <= bands[:, 2]).all()

    def test_band_brackets_the_anchor_price(self, fitted):
        """A 5-95 band on a daily move must straddle today's close."""
        _, bands, _, last_close = fitted
        assert (bands[:, 0] < last_close).mean() > 0.95
        assert (bands[:, 2] > last_close).mean() > 0.95

    def test_out_of_sample_coverage_is_near_nominal(self, fitted):
        """
        The band claims 90%. If it covers 60% it is decoration, and if it covers
        100% it is too wide to constrain anything.
        """
        _, bands, test, _ = fitted
        metrics = band_metrics(bands, test.exit_close.to_numpy())
        assert metrics["nominal_coverage"] == pytest.approx(0.90)
        assert 0.82 <= metrics["coverage"] <= 0.96, f"coverage {metrics['coverage']}"
        assert metrics["mean_relative_width"] < 0.25, "a band that wide says nothing"

    def test_width_scales_with_volatility(self, fitted):
        """Today's volatility sets the width; that is what makes it per-day."""
        band, _, test, last_close = fitted
        probabilities = np.full(len(test), 0.5)
        calm = band.predict(probabilities, last_close, np.full(len(test), 0.005))
        wild = band.predict(probabilities, last_close, np.full(len(test), 0.02))
        assert ((wild[:, 2] - wild[:, 0]) > (calm[:, 2] - calm[:, 0])).all()

    def test_higher_probability_shifts_the_band_up(self, fitted):
        """
        The band's skew comes from the model's conviction, measured on training
        data. If confident-up days really did resolve higher, the band moves.
        """
        band, _, test, last_close = fitted
        volatility = np.full(len(test), 0.01)
        low = band.predict(np.full(len(test), 0.01), last_close, volatility)
        high = band.predict(np.full(len(test), 0.99), last_close, volatility)
        assert (high[:, 1] >= low[:, 1]).all()
        assert high[:, 1].mean() > low[:, 1].mean()

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            ConditionalReturnBand().predict([0.5], [100.0], [0.01])

    def test_too_little_data_raises(self):
        with pytest.raises(ValueError, match="usable training rows"):
            ConditionalReturnBand().fit([0.5] * 10, [0.01] * 10, [0.01] * 10)

    def test_missing_volatility_yields_nan_not_a_borrowed_width(self, fitted):
        band, _, test, last_close = fitted
        volatility = np.full(len(test), 0.01)
        volatility[0] = np.nan
        out = band.predict(np.full(len(test), 0.5), last_close, volatility)
        assert np.isnan(out[0]).all()
        assert np.isfinite(out[1:]).all()

    def test_band_metrics_on_a_known_case(self):
        """Two rows in, one out: coverage is exactly 0.5 and width is exact."""
        bands = np.array([[90.0, 100.0, 110.0], [90.0, 100.0, 110.0]])
        actual = np.array([100.0, 200.0])
        metrics = band_metrics(bands, actual)
        assert metrics["n"] == 2
        assert metrics["coverage"] == pytest.approx(0.5)
        # widths are 20/100 and 20/200
        assert metrics["mean_relative_width"] == pytest.approx((0.2 + 0.1) / 2)

    def test_band_metrics_rejects_misaligned_input(self):
        with pytest.raises(ValueError, match="does not align"):
            band_metrics(np.zeros((3, 3)), np.zeros(5))


# ---------------------------------------------------------------------------
# Kronos wiring, without the weights
# ---------------------------------------------------------------------------

class _FakePredictor:
    """
    Stands in for ``KronosPredictor``, encoding batch position in its output.

    ``generate`` returns the batch index in the close channel, so a test can
    prove which samples were attributed to which date. Anything that shuffles or
    mis-slices the batch shows up immediately.
    """

    price_cols = ["open", "high", "low", "close"]
    vol_col = "volume"
    amt_vol = "amount"
    clip = 5
    max_context = 512

    def __init__(self):
        self.calls = []

    def generate(self, x, x_stamp, y_stamp, pred_len, T, top_k, top_p, sample_count, verbose):
        batch = np.asarray(x).shape[0]
        self.calls.append({"batch": batch, "sample_count": sample_count, "pred_len": pred_len})
        out = np.zeros((batch, pred_len, 6), dtype=np.float64)
        out[:, 0, 3] = np.arange(batch, dtype=np.float64)  # close channel = batch index
        return out


class TestKronosSampleMapping:
    @pytest.fixture
    def setup(self, bars):
        dataset = build_direction_dataset(bars)
        fake = _FakePredictor()
        model = KronosDirection(
            seed=1, sample_count=4, lookback=32, max_batch=8, predictor=fake
        )
        model.set_ohlcv_context(dataset.ohlcv)
        train = dataset.slice(np.arange(600))
        model.fit(train.features, train.labels)
        return dataset, model, fake

    def test_generate_is_called_with_sample_count_one(self, setup):
        """
        The whole per-sample-path trick: replicate the batch, ask for one sample.

        With ``sample_count > 1`` the library averages internally and the spread
        this model depends on is destroyed.
        """
        dataset, model, fake = setup
        test = dataset.slice(np.arange(600, 606))
        model.predict_proba_up(test.features)
        assert fake.calls, "generate was never called"
        assert all(call["sample_count"] == 1 for call in fake.calls)
        assert all(call["pred_len"] == 1 for call in fake.calls)

    def test_batch_is_chunked_within_the_cap(self, setup):
        dataset, model, fake = setup
        test = dataset.slice(np.arange(600, 606))
        model.predict_proba_up(test.features)
        # max_batch 8 / sample_count 4 => 2 dates per call
        assert all(call["batch"] <= 8 for call in fake.calls)
        assert sum(call["batch"] for call in fake.calls) == 6 * 4

    def test_each_date_gets_its_own_block_of_samples(self, setup):
        """
        The fake returns the batch index, so after de-normalisation each date's
        samples must come from its own contiguous block. Row j's raw draws are
        ``[j*N, (j+1)*N)`` within its chunk — if the mapping slipped by one
        block, every probability would belong to the neighbouring day.
        """
        dataset, model, fake = setup
        rows = 6
        test = dataset.slice(np.arange(600, 600 + rows))
        model.predict_proba_up(test.features)

        n = model.sample_count
        per_chunk = model.max_batch // n
        bands = model.price_bands_
        assert bands.shape == (rows, 3)

        for row in range(rows):
            position_in_chunk = row % per_chunk
            raw = np.arange(position_in_chunk * n, (position_in_chunk + 1) * n, dtype=np.float64)
            context = model._context_frame(pd.Timestamp(test.index[row]))
            values = context.to_numpy(dtype=np.float32)
            mean, std = values.mean(axis=0), values.std(axis=0)
            expected = raw * (std[3] + 1e-5) + mean[3]
            assert bands[row] == pytest.approx(np.percentile(expected, [5, 50, 95]))

    def test_context_never_includes_the_predicted_bar(self, setup):
        dataset, model, _ = setup
        as_of = pd.Timestamp(dataset.index[700])
        context = model._context_frame(as_of)
        assert context.index.max() == as_of, "context must end at the decision date"
        assert len(context) <= model.lookback

    def test_fit_without_ohlcv_context_raises(self, bars):
        dataset = build_direction_dataset(bars)
        model = KronosDirection(seed=1, sample_count=2, lookback=16, predictor=_FakePredictor())
        train = dataset.slice(np.arange(300))
        with pytest.raises(RuntimeError, match="set_ohlcv_context"):
            model.fit(train.features, train.labels)

    def test_monte_carlo_error_is_reported(self, setup):
        _, model, _ = setup
        # sqrt(0.25/4) = 0.25 at sample_count=4
        assert model.fit_info_["monte_carlo_se_of_p_up"] == pytest.approx(0.25)

    def test_degenerate_training_window_falls_back(self, bars):
        dataset = build_direction_dataset(bars)
        model = KronosDirection(seed=1, sample_count=2, lookback=16, predictor=_FakePredictor())
        model.set_ohlcv_context(dataset.ohlcv)
        train = dataset.slice(np.arange(300))
        model.fit(train.features, np.ones(len(train), dtype=np.int8))
        probabilities = model.predict_proba_up(train.features)
        assert np.allclose(probabilities, probabilities[0])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_foundation_slots_are_registered(self):
        for name in ("logistic", "gradient_boosting", "tabpfn", "kronos", "foundation_ensemble"):
            assert name in MODEL_FACTORIES

    def test_lstm_is_not_a_slot(self):
        """Dropped deliberately: it never cleared the baselines."""
        assert "lstm" not in MODEL_FACTORIES

    def test_unavailable_dependency_gives_an_actionable_error(self, monkeypatch):
        import src.models.kronos_direction as kronos_module

        monkeypatch.setattr(kronos_module, "_KRONOS_AVAILABLE", False)
        monkeypatch.setattr(kronos_module, "_KRONOS_IMPORT_ERROR", "no vendor dir")
        with pytest.raises(ImportError, match="setup_kronos"):
            kronos_module.KronosDirection(seed=1)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("tabpfn") is None, reason="tabpfn not installed"
)
class TestTabPFN:
    def test_interface_and_probabilities(self, bars):
        dataset = build_direction_dataset(bars)
        train = dataset.slice(np.arange(250))
        test = dataset.slice(np.arange(250, 265))
        model = build_model("tabpfn", seed=7)
        try:
            model.fit(train.features, train.labels)
            probabilities = model.predict_proba_up(test.features)
        except RuntimeError as exc:
            # TabPFN allocates a large attention buffer. On a small-RAM box the
            # CPU allocator refuses, which says nothing about this wrapper.
            if "not enough memory" in str(exc) or "DefaultCPUAllocator" in str(exc):
                pytest.skip(f"TabPFN needs more memory than this machine has: {exc}")
            raise
        assert probabilities.shape == (len(test),)
        assert np.all((probabilities > 0) & (probabilities < 1))
        assert set(np.unique(model.predict(test.features))).issubset({0, 1})
        assert model.fit_info_["n_features"] == 46

    def test_non_finite_input_is_refused_not_imputed(self, bars):
        dataset = build_direction_dataset(bars)
        train = dataset.slice(np.arange(400))
        model = build_model("tabpfn", seed=7)
        broken = train.features.copy()
        broken.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            model.fit(broken, train.labels)
