"""
The Predictions tab's endpoint and the service behind it.

Two things are worth pinning down here, and they are the two the tab depends on
being true:

  * the response carries the answer and *only* the answer -- a price, a
    direction, candles, one forecast point. If the pipeline ever leaks back into
    the payload the UI will grow panels to render it again;
  * direction comes from the aggregated probability, not from the sign of the
    price move. Those are different statistics and they are allowed to disagree.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import predict as predict_route
from src.models.foundation.forecast_service import (
    ForecastUnavailable,
    run_foundation_forecast,
)


def _sample_ohlcv(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    close = pd.Series([100 + i * 0.4 for i in range(rows)], index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1000 + i for i in range(rows)],
        },
        index=index,
    )


class _FakePipeline:
    """A member with a real spread, so inverse-variance weighting has something to weigh."""

    def __init__(self, price: float, p_up: float, spread: float = 2.0, name: str = "fake"):
        self.name = name
        self.price = price
        self.p_up = p_up
        self.spread = spread
        self.seen_covariates = None

    def predict(self, df, horizon: int = 1, covariates=None):
        self.seen_covariates = covariates
        samples = np.linspace(self.price - self.spread, self.price + self.spread, 128)
        return {
            "price": self.price,
            "p_up": self.p_up,
            "samples": samples,
            "quantiles": {q: float(np.quantile(samples, q)) for q in (0.1, 0.5, 0.9)},
        }


def _factory(members):
    return lambda model_type: members.get(model_type)


def test_only_chronos_is_handed_the_technical_features():
    """
    Kronos reads candles and TimesFM 2.5 is univariate; handing either the TA
    frame would be a false claim about what the model saw.
    """
    members = {
        "unified_kronos": _FakePipeline(140.0, 0.7),
        "unified_chronos": _FakePipeline(141.0, 0.8),
        "unified_timesfm": _FakePipeline(139.0, 0.6),
    }
    result = run_foundation_forecast(
        _sample_ohlcv(), pipeline_factory=_factory(members), symbol="TEST"
    )

    assert members["unified_chronos"].seen_covariates is not None
    assert members["unified_kronos"].seen_covariates is None
    assert members["unified_timesfm"].seen_covariates is None
    # Every Section 4 category contributed at least one column.
    assert all(count > 0 for count in result.features_built.values())
    assert set(result.members_used) == {"Kronos", "Chronos-2", "TimesFM 2.5"}


def test_one_failing_member_does_not_take_the_forecast_offline():
    class _Broken:
        def predict(self, df, horizon: int = 1, covariates=None):
            raise RuntimeError("model weights are not downloaded")

    members = {
        "unified_kronos": _FakePipeline(140.0, 0.7),
        "unified_chronos": _Broken(),
        "unified_timesfm": None,
    }
    result = run_foundation_forecast(_sample_ohlcv(), pipeline_factory=_factory(members))

    assert result.members_used == ["Kronos"]
    assert "model weights are not downloaded" in result.members_failed["unified_chronos"]
    assert result.members_failed["unified_timesfm"] == "pipeline not available"


def test_no_member_at_all_is_unavailable_rather_than_a_fabricated_number():
    with pytest.raises(ForecastUnavailable):
        run_foundation_forecast(_sample_ohlcv(), pipeline_factory=lambda _: None)


def test_direction_follows_the_probability_and_flags_a_split():
    """
    A member set whose aggregated price falls while P(up) stays above 0.5 must
    still call UP -- Requirement 5.1 thresholds the probability, and sign(price -
    close) is not a substitute for it. The disagreement is reported, not hidden.
    """
    df = _sample_ohlcv()
    last_close = float(df["Close"].iloc[-1])
    members = {
        "unified_kronos": _FakePipeline(last_close - 3.0, 0.62),
        "unified_chronos": _FakePipeline(last_close - 2.0, 0.58),
        "unified_timesfm": _FakePipeline(last_close - 1.0, 0.61),
    }
    result = run_foundation_forecast(df, pipeline_factory=_factory(members))

    assert result.p_up > 0.5
    assert result.direction == "UP"
    assert result.price < last_close
    assert result.split is True


def test_the_anchor_is_the_close_the_members_measured_against():
    """
    Nothing outside this function can move the anchor, and that is the point.

    Every member computes p_up internally as P(next bar > last close), so a
    `split` measured against any other price reports a disagreement about a
    number the probability never saw -- and hides the one that is real. This
    function used to accept a `reference_price` for exactly that, and the route
    handed it a live quote.
    """
    df = _sample_ohlcv()
    last_close = float(df["Close"].iloc[-1])
    members = {"unified_kronos": _FakePipeline(last_close - 0.06, 0.47)}

    result = run_foundation_forecast(df, pipeline_factory=_factory(members))

    assert result.anchor_price == pytest.approx(last_close)
    assert result.direction == "DOWN"
    # Forecast and call are both below the close, so the two heads agree. What
    # a quote does relative to them is the caller's business, not this one's.
    assert result.split is False

    with pytest.raises(TypeError):
        run_foundation_forecast(
            df, pipeline_factory=_factory(members), reference_price=last_close - 1.0
        )


def test_interval_is_wider_than_a_point():
    members = {"unified_kronos": _FakePipeline(140.0, 0.7, spread=5.0)}
    result = run_foundation_forecast(_sample_ohlcv(), pipeline_factory=_factory(members))

    assert result.lower_90 < result.lower_68 <= result.price <= result.upper_68 < result.upper_90


# ── The route ────────────────────────────────────────────────


def _client():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    return TestClient(app)


def _clear_forecast_cache():
    """The route memoises by (symbol, bar); tests must not read each other's runs."""
    predict_route._FORECAST_CACHE.clear()


def test_forecast_endpoint_returns_the_answer_and_nothing_else():
    _clear_forecast_cache()
    members = {
        "unified_kronos": _FakePipeline(140.0, 0.7),
        "unified_chronos": _FakePipeline(141.0, 0.8),
        "unified_timesfm": _FakePipeline(139.0, 0.6),
    }
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(138.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_factory(members)),
    ):
        response = _client().get("/api/predict/forecast/META")

    assert response.status_code == 200
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["direction"] == "UP"
    assert payload["forecast_price"] == pytest.approx(140.0, abs=1.5)
    assert payload["current_price"] == 138.0
    assert payload["horizon_label"] == "Next 1 Day"
    # Named in pipeline order, not sorted by dict key.
    assert payload["models"] == ["Kronos", "Chronos-2", "TimesFM 2.5"]

    assert len(payload["forecast"]) == 1
    point = payload["forecast"][0]
    assert point["date"] == payload["forecast_date"]
    # Named for the coverage the bounds have: q0.05..q0.95 brackets 90%, so the
    # point must NOT carry a `95` field claiming five points more than that.
    assert point["lower_90"] < point["predicted"] < point["upper_90"]
    assert point["lower_68"] > point["lower_90"] and point["upper_68"] < point["upper_90"]
    assert "upper95" not in point and "lower95" not in point

    # The candles are NOT here. They come from /predict/history, because they
    # are ready in milliseconds and this response is not -- bundling them is
    # what made the chart wait on the models.
    assert "history" not in payload

    # Nothing from the pipeline reaches the client: no weights, no per-model
    # numbers, no feature inventory, no aggregation method. If a key like these
    # ever appears the tab will grow a panel for it again.
    for leaked in ("weights", "method", "features_built", "model_predictions", "p_up", "evidence"):
        assert leaked not in payload


def test_a_quote_from_a_later_session_does_not_flip_the_expected_change():
    """
    The PLTR report this came from: current price $174.94, forecast price
    $181.62, direction DOWN.

    Nothing was wrong with the models. They read a 182.53 close and forecast
    181.62 from it -- a fall, and P(up) agreed. But the stock had already
    dropped ~4% in a session the frame does not contain (the download window
    ends at today's date and yfinance treats that end as exclusive), so the
    route divided 181.62 by a 174.94 quote and published +3.8% as "Expected
    Change" next to the DOWN arrow.

    Expected Change is now the move away from the close the models read, the
    quote figure is served separately under its own name, and the note that
    explains the gap is told which of the two things happened.
    """
    _clear_forecast_cache()
    df = _sample_ohlcv()
    anchor = float(df["Close"].iloc[-1])
    quote = anchor * 0.9601  # the -3.99% session the frame does not contain
    members = {"unified_kronos": _FakePipeline(anchor * 0.995, 0.42)}

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=df),
        patch.object(predict_route, "_latest_available_price", return_value=(quote, "post_market")),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_factory(members)),
    ):
        payload = _client().get("/api/predict/forecast/PLTR").json()

    assert payload["direction"] == "DOWN"
    assert payload["anchor_price"] == pytest.approx(round(anchor, 2))
    assert payload["current_price"] == pytest.approx(round(quote, 2))

    # Measured from the close the models read, the forecast falls -- which is
    # what the DOWN arrow says...
    assert payload["expected_change_pct"] < 0
    # ...while against a quote from a session they never saw the same number
    # rises. Both are served; only the first is called "Expected Change".
    assert payload["quote_change_pct"] > 0

    # The heads never disagreed here, so saying they did would be wrong twice.
    assert payload["split"] is True
    assert payload["split_reason"] == "quote"

    # The chart hangs its segment off the last candle, so the segment's own
    # direction is the move from the anchor. Against the quote it drew a
    # falling line in the colour of a rise.
    assert payload["forecast"][0]["direction"] == "down"
    assert payload["forecast"][0]["change_pct"] == payload["expected_change_pct"]


def test_history_endpoint_serves_candles_without_running_a_model():
    """
    The point of the split: the chart's request must touch neither a pipeline
    nor the live-quote lookup, which cost ~7s and ~1.25s respectively.
    """
    _clear_forecast_cache()

    def _explode(model_type):
        raise AssertionError(f"the history route must not load {model_type}")

    def _explode_quote(*args, **kwargs):
        raise AssertionError("the history route must not fetch a live quote")

    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=400)),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_explode),
        patch.object(predict_route, "_latest_available_price", side_effect=_explode_quote),
    ):
        response = _client().get("/api/predict/history/META?days=60")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "META"
    assert len(payload["bars"]) == 60
    assert set(payload["bars"][0]) == {"date", "open", "high", "low", "close", "volume"}
    # `as_of` is the last bar, so a client can check that a forecast it holds
    # was built on the same frame these candles came from.
    assert payload["as_of"] == payload["bars"][-1]["date"]


def test_forecast_is_computed_once_per_bar():
    """
    Re-selecting a symbol must not pay for the models again. The forecast is a
    function of the bars, so it is cached on (symbol, last bar) -- but the live
    quote is not, because it moves intraday.
    """
    _clear_forecast_cache()
    runs = []

    class _Counting(_FakePipeline):
        def predict(self, df, horizon: int = 1, covariates=None):
            runs.append(1)
            return super().predict(df, horizon=horizon, covariates=covariates)

    members = {"unified_kronos": _Counting(140.0, 0.7)}
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_factory(members)),
        patch.object(predict_route, "_latest_available_price", return_value=(138.0, "latest_close")),
    ):
        first = _client().get("/api/predict/forecast/META").json()
        second = _client().get("/api/predict/forecast/META").json()

    assert len(runs) == 1, "the second request re-ran the models"
    assert first["forecast_price"] == second["forecast_price"]

    # A fresh quote still moves everything derived from it, so the cache must
    # not freeze the current price or the change measured against it.
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_factory(members)),
        patch.object(predict_route, "_latest_available_price", return_value=(150.0, "pre_market")),
    ):
        moved = _client().get("/api/predict/forecast/META").json()

    assert len(runs) == 1
    assert moved["current_price"] == 150.0
    # The cached run holds nothing quote-derived, so the anchor and the change
    # measured from it are identical across the two requests...
    assert moved["anchor_price"] == first["anchor_price"]
    assert moved["expected_change_pct"] == first["expected_change_pct"]
    # ...while the figure that is measured against the quote moves with it.
    assert moved["quote_change_pct"] != first["quote_change_pct"]
    # A 140 forecast against a 227.60 close is a fall while P(up) 0.7 says UP:
    # the two heads genuinely disagree, on both requests, because that is a
    # property of the bars rather than of the quote.
    assert moved["direction"] == "UP"
    assert moved["split_reason"] == first["split_reason"] == "heads"


def test_forecast_endpoint_is_unavailable_rather_than_wrong_when_no_model_runs():
    """
    An unservable forecast is a normal 200. The chart is unaffected -- it is
    drawn from its own request.
    """
    _clear_forecast_cache()
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv()),
        patch.object(predict_route, "_latest_available_price", return_value=(138.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", return_value=None),
    ):
        response = _client().get("/api/predict/forecast/META")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "unavailable"
    assert payload["forecast_price"] is None
    assert payload["direction"] is None
    assert payload["forecast"] == []
    assert payload["message"]


def test_models_read_the_full_download_not_the_chart_window():
    """
    The chart window never reaches the models. The feature layer needs long
    windows (SMA_200 among them), so a short chart range must not truncate them.
    """
    _clear_forecast_cache()
    seen = {}

    class _Recorder(_FakePipeline):
        def predict(self, df, horizon: int = 1, covariates=None):
            seen["rows"] = len(df)
            return super().predict(df, horizon=horizon, covariates=covariates)

    members = {"unified_kronos": _Recorder(140.0, 0.7)}
    with (
        patch.object(predict_route, "_download_prediction_data", return_value=_sample_ohlcv(rows=400)),
        patch.object(predict_route, "_latest_available_price", return_value=(138.0, "latest_close")),
        patch.object(predict_route, "_get_foundation_pipeline", side_effect=_factory(members)),
    ):
        response = _client().get("/api/predict/forecast/META")

    assert response.status_code == 200
    assert seen["rows"] == 400
