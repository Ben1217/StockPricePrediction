"""
The shapes the installed Chronos-2 and TimesFM 2.5 builds actually return.

Both pipelines were written against a different API from the one that ships, and
both failed the first time they were run for real:

  * ``Chronos2Pipeline.predict`` takes ``inputs`` positionally and reads
    covariates out of the input schema, not out of a ``past_covariates``
    keyword. Probing the signature for that keyword returned False on a build
    that supports covariates perfectly well, so every run silently took the
    univariate branch -- the feature-informed arm of the comparison had never
    once run as one.
  * TimesFM 2.5 returns ten values for nine configured quantile levels: a
    leading mean, then the deciles. Reading the ten as though they were the
    levels mislabels the mean as q0.1 and shifts every level after it.

The fakes below mimic those APIs exactly, so the wiring is pinned without
downloading weights. If either upstream changes shape, these fail with a clear
reason rather than the tab quietly serving a mislabelled distribution.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.foundation.aggregator import FoundationAggregator
from src.models.foundation.chronos_pipeline import ChronosPipeline
from src.models.foundation.timesfm_pipeline import TimesFMPipeline


def _ohlcv(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series(100 + np.linspace(0, 40, rows), index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.arange(rows) + 1_000_000,
        },
        index=index,
    )


# ── Chronos-2 ────────────────────────────────────────────────


class _FakeChronos2:
    """The chronos-forecasting 2.x surface: dict inputs, 21 native quantiles."""

    quantiles = [0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99]

    def __init__(self, centre: float = 150.0, spread: float = 4.0):
        self.centre = centre
        self.spread = spread
        self.seen = None

    def predict(self, inputs, prediction_length=None, **kwargs):
        self.seen = inputs
        # (n_variates, n_quantiles, prediction_length)
        values = [
            self.centre + self.spread * (level - 0.5) * 2 for level in self.quantiles
        ]
        return [np.asarray(values, dtype=np.float64).reshape(1, len(self.quantiles), 1)]


def _chronos_with(model):
    pipeline = ChronosPipeline(lookback=128)
    pipeline.model = model  # bypass _ensure_model, which would download weights
    return pipeline


def test_chronos_passes_covariates_through_the_input_schema():
    model = _FakeChronos2()
    result = _chronos_with(model).predict(_ohlcv(), horizon=1)

    assert result["covariates_used"] is True
    entry = model.seen[0]
    assert set(entry) == {"target", "past_covariates"}
    assert len(entry["target"]) == 128
    # Every Section 4 covariate reached the model, each aligned to the target.
    assert set(entry["past_covariates"]) == set(result["covariate_columns"])
    assert all(len(series) == 128 for series in entry["past_covariates"].values())


def test_chronos_reads_the_levels_off_the_model_not_off_an_assumption():
    model = _FakeChronos2()
    result = _chronos_with(model).predict(_ohlcv(), horizon=1)

    assert result["quantile_levels"] == [float(q) for q in model.quantiles]
    assert len(result["quantiles"]) == 21
    assert result["price"] == pytest.approx(150.0)
    # A quantile model, so no invented sample paths.
    assert result["samples"] is None


def test_chronos_p_up_is_the_interpolated_cdf_not_the_sign_of_the_median():
    df = _ohlcv()
    last_close = float(df["Close"].iloc[-1])

    # Centred exactly on the last close: half the mass above it.
    centred = _chronos_with(_FakeChronos2(centre=last_close)).predict(df, horizon=1)
    assert centred["p_up"] == pytest.approx(0.5, abs=0.02)

    # Shifted well above it: P(up) high, but strictly below 1 -- the model said
    # nothing beyond its outermost knot, so the tail is clamped, not assumed empty.
    above = _chronos_with(_FakeChronos2(centre=last_close + 20.0)).predict(df, horizon=1)
    assert 0.95 <= above["p_up"] <= 1.0
    assert above["tail_clamped"] is True


def test_chronos_reports_a_shape_it_does_not_understand():
    class _Wrong(_FakeChronos2):
        def predict(self, inputs, prediction_length=None, **kwargs):
            return [np.zeros((21,))]  # not (n_variates, n_quantiles, horizon)

    with pytest.raises(ValueError, match="n_variates"):
        _chronos_with(_Wrong()).predict(_ohlcv(), horizon=1)


# ── TimesFM 2.5 ──────────────────────────────────────────────


class _FakeTimesFM:
    """Ten outputs for nine levels: a leading mean, then q0.1 .. q0.9."""

    def __init__(self, deciles):
        self.deciles = list(deciles)

    def forecast(self, horizon, inputs):
        mean = float(np.mean(self.deciles))
        quantiles = np.asarray([mean] + self.deciles, dtype=np.float64).reshape(1, 1, 10)
        point = np.asarray([[self.deciles[4]]], dtype=np.float64)  # decode_index 5 == q0.5
        return point, quantiles


def _timesfm_with(model):
    pipeline = TimesFMPipeline()
    pipeline.model = model
    return pipeline


def test_timesfm_strips_the_leading_mean_before_labelling_the_deciles():
    deciles = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0]
    result = _timesfm_with(_FakeTimesFM(deciles)).predict(_ohlcv(), horizon=1)

    # q0.1 is the first decile, not the mean that preceded it.
    assert result["quantiles"][0.1] == pytest.approx(100.0)
    assert result["quantiles"][0.5] == pytest.approx(104.0)
    assert result["quantiles"][0.9] == pytest.approx(108.0)
    assert result["price"] == pytest.approx(104.0)
    assert result["mean"] == pytest.approx(float(np.mean(deciles)))
    assert result["samples"] is None


def test_timesfm_still_rejects_a_count_it_cannot_place():
    class _Eight(_FakeTimesFM):
        def forecast(self, horizon, inputs):
            return np.zeros((1, 1)), np.zeros((1, 1, 8))

    with pytest.raises(ValueError, match="8 values"):
        _timesfm_with(_Eight([])).predict(_ohlcv(), horizon=1)


# ── Aggregation across the two kinds of member ───────────────


def test_aggregator_handles_a_member_that_reports_no_samples():
    """
    `samples: None` is how a quantile model says it has no paths. The aggregator
    used to test `'samples' in result`, which is True for that key, and then
    called len(None) -- so the whole ensemble raised TypeError as soon as
    Chronos-2 or TimesFM 2.5 joined it.
    """
    quantile_member = {
        "price": 100.0,
        "p_up": 0.6,
        "samples": None,
        "quantiles": {0.1: 98.0, 0.5: 100.0, 0.9: 102.0},
    }
    sample_member = {
        "price": 101.0,
        "p_up": 0.55,
        "samples": np.linspace(99.0, 103.0, 64),
        "quantiles": {0.1: 99.4, 0.5: 101.0, 0.9: 102.6},
    }

    aggregated = FoundationAggregator.aggregate(
        {"unified_chronos": quantile_member, "unified_kronos": sample_member}
    )

    assert aggregated["method"] == "inverse_variance"
    assert set(aggregated["weights"]) == {"unified_chronos", "unified_kronos"}
    assert aggregated["models_excluded"] == {}
    assert 100.0 <= aggregated["price"] <= 101.0
    assert 0.55 <= aggregated["p_up"] <= 0.6
