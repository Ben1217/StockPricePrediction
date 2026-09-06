"""
DAY / WEEK / MONTH: that the timeframe is a horizon and not a label.

The selector this covers used to read 3M / 6M / 1Y and changed only how many
daily candles were drawn — the forecast under it was the same number under all
three. The whole point of the change is that it is no longer cosmetic, so these
tests are about the things that would make it cosmetic again:

  * the aggregation is a real OHLCV bar (first / max / min / last / sum), not a
    resampled close with the other columns carried along;
  * the models are handed the aggregated frame, so "the next bar" moves with
    the selection;
  * the forecast date is the next *period*, not the next business day;
  * nothing is shared between timeframes that should not be — the response
    caches are keyed on it, and a Friday, when the daily and weekly frames end
    on the same date, is where a missing key would show;
  * DAY is byte-for-byte what the route did before any of this existed.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import predict as predict_route
from src.data.timeframe import (
    TIMEFRAMES,
    last_bar_is_forming,
    next_bar_date,
    period_bounds,
    period_end,
    period_sessions,
    resample_ohlcv,
    resolve_timeframe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
def _daily(rows: int = 2600, start: str = "2016-01-04") -> pd.DataFrame:
    """A deterministic daily frame long enough to aggregate to months."""
    index = pd.bdate_range(start, periods=rows)
    rng = np.random.default_rng(3)
    close = pd.Series(80 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, rows))), index=index)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.008,
            "Low": close * 0.992,
            "Close": close,
            "Volume": np.arange(1, rows + 1, dtype=float) * 1000.0,
        },
        index=index,
    )


class _FakePipeline:
    """
    A member with a real spread, and a memory of the frame it was handed.

    ``runs`` counts *forecasts*, not calls. The stack has three members and the
    factory below hands the same object to all of them, so a single request
    reaches ``predict`` three times; counting calls would make "computed once"
    read as three.
    """

    def __init__(self):
        self.seen = []
        self.members = 0

    def predict(self, df, horizon: int = 1, covariates=None):
        self.seen.append(df)
        self.members += 1
        price = float(df["Close"].iloc[-1]) * 1.01
        return {
            "price": price,
            "p_up": 0.6,
            "samples": np.linspace(price - 2.0, price + 2.0, 128),
        }

    @property
    def runs(self) -> int:
        assert self.members % len(predict_route.FOUNDATION_MEMBERS) == 0
        return self.members // len(predict_route.FOUNDATION_MEMBERS)


#: A live quote that never moves, so nothing in these tests depends on the
#: market being open. The forecast is anchored on the last close regardless;
#: this only fills the "Current Price" box.
_QUOTE = (100.0, "regular_market")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(predict_route.router, prefix="/api/predict")
    predict_route._FORECAST_CACHE.clear()
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# The aggregation itself
# ─────────────────────────────────────────────────────────────────────────────
def test_a_weekly_bar_is_an_ohlcv_bar_and_not_five_closes():
    """
    Open from the first day, High/Low from the extremes, Close from the last,
    Volume summed. A weekly frame that gets any one of these wrong still charts
    and still forecasts; it is simply a different instrument.
    """
    index = pd.bdate_range("2024-01-01", periods=5)  # Mon-Fri, one whole week
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0, 14.0],
            "High": [11.0, 12.0, 18.0, 14.0, 15.0],
            "Low": [9.0, 10.0, 11.0, 3.0, 13.0],
            "Close": [10.5, 11.5, 12.5, 13.5, 14.5],
            "Volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        },
        index=index,
    )
    weekly = resample_ohlcv(frame, resolve_timeframe("week"))

    assert len(weekly) == 1
    bar = weekly.iloc[0]
    assert bar["Open"] == 10.0, "Open is the first day's open"
    assert bar["Close"] == 14.5, "Close is the last day's close"
    assert bar["High"] == 18.0, "High is the week's extreme, not the last day's"
    assert bar["Low"] == 3.0, "Low is the week's extreme, not the last day's"
    assert bar["Volume"] == 1500.0, "Volume is summed over the period"
    # Exactly the daily frame's columns, so the result can be handed to
    # anything that takes daily bars without a filtering step to remember.
    assert list(weekly.columns) == list(frame.columns)
    # The bar is indexed by the last session it holds, not by the period label.
    assert weekly.index[-1] == pd.Timestamp("2024-01-05")
    assert int(period_sessions(frame, resolve_timeframe("week")).iloc[-1]) == 5


def test_a_week_with_no_trading_is_dropped_rather_than_filled():
    """
    A gap is a gap. Filling it forward would hand the models a flat candle with
    a zero range that never happened, which the volatility features read as a
    real quiet week.
    """
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-16", "2024-01-17"])
    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0, 12.0, 13.0],
            "High": [11.0, 12.0, 13.0, 14.0],
            "Low": [9.0, 10.0, 11.0, 12.0],
            "Close": [10.5, 11.5, 12.5, 13.5],
            "Volume": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )
    week = resolve_timeframe("week")
    weekly = resample_ohlcv(frame, week)
    assert len(weekly) == 2, "the two empty weeks between are not bars"
    assert list(period_sessions(frame, week)) == [2, 2]


def test_daily_resampling_is_the_identity():
    """
    DAY must not go through the resampler at all. Aggregating daily bars onto a
    daily rule would introduce bins for market holidays and change the frame the
    route has always served.
    """
    frame = _daily(rows=300)
    assert resample_ohlcv(frame, resolve_timeframe("day")) is frame


def test_the_forecast_date_is_the_next_period_not_the_next_business_day():
    """
    A weekly forecast is about next week's bar and has to carry that bar's own
    label, or the chart hangs the point one day after the last candle instead of
    one week after it.
    """
    friday = pd.Timestamp("2026-09-04")
    assert next_bar_date(friday, resolve_timeframe("day")) == "2026-09-07"
    assert next_bar_date(friday, resolve_timeframe("week")) == "2026-09-11"
    assert next_bar_date(pd.Timestamp("2026-09-30"), resolve_timeframe("month")) == "2026-10-31"


def test_period_bounds_name_the_days_a_bar_covers():
    start, end = period_bounds(pd.Timestamp("2026-09-04"), resolve_timeframe("week"))
    assert (start, end) == ("2026-08-29", "2026-09-04")
    start, end = period_bounds(pd.Timestamp("2026-09-30"), resolve_timeframe("month"))
    assert (start, end) == ("2026-09-01", "2026-09-30")


def test_a_forming_bar_is_reported_against_the_calendar_not_the_data():
    """
    A week whose Friday was a holiday is finished even though no bar printed on
    the label date. Comparing the label to the last daily row would call that
    week forming for ever, so the test is against today.
    """
    week = resolve_timeframe("week")
    frame = resample_ohlcv(_daily(rows=40, start="2026-08-03"), week)
    closes_on = period_end(frame, week)
    assert last_bar_is_forming(frame, week, today=closes_on - pd.Timedelta(days=2))
    assert not last_bar_is_forming(frame, week, today=closes_on + pd.Timedelta(days=3))


def test_a_bar_is_dated_by_the_last_session_it_holds_not_by_its_period():
    """
    A monthly bar read on the 6th holds four sessions and closes on the 30th.
    Labelling it 2026-09-30 puts a candle three weeks into the future on the
    chart and reports `as_of` as a date that has not happened; the honest label
    is the last session the bar actually contains.

    The period is not lost by this — `period_end` derives it from the index —
    and it is the period, not the label, that decides what the next bar is.
    """
    month = resolve_timeframe("month")
    daily = _daily(rows=200, start="2026-01-01").loc[: pd.Timestamp("2026-09-04")]
    monthly = resample_ohlcv(daily, month)

    assert monthly.index[-1] == pd.Timestamp("2026-09-04"), "dated by its last session"
    assert period_end(monthly, month) == pd.Timestamp("2026-09-30"), "but closes at month end"
    assert monthly.index[-1] <= daily.index[-1], "no bar is dated after the data ends"
    assert last_bar_is_forming(monthly, month, today=pd.Timestamp("2026-09-06"))
    # The forecast is still the *next period*, measured from the period end.
    assert next_bar_date(period_end(monthly, month), month) == "2026-10-31"
    # And the last bar's close is the latest close there is, which is what
    # makes it a usable anchor.
    assert monthly["Close"].iloc[-1] == daily["Close"].iloc[-1]


def test_an_unknown_timeframe_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="day, week, month"):
        resolve_timeframe("quarter")
    # Absent means daily, so callers that predate the concept are unaffected.
    assert resolve_timeframe(None).key == "day"
    assert resolve_timeframe("").key == "day"
    assert resolve_timeframe("1wk").key == "week"


# ─────────────────────────────────────────────────────────────────────────────
# The routes
# ─────────────────────────────────────────────────────────────────────────────
def test_the_models_are_handed_the_aggregated_frame(client):
    """
    The load-bearing assertion of the whole feature. If the pipeline still sees
    daily candles on a weekly request then the timeframe is a caption, and the
    weekly "forecast" is tomorrow's number under next week's name.
    """
    frame = _daily()
    pipeline = _FakePipeline()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: pipeline), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        client.get("/api/predict/forecast/AAA?timeframe=week")

    seen = pipeline.seen[0]
    weekly = resample_ohlcv(frame, resolve_timeframe("week"))
    assert len(seen) == len(weekly) < len(frame)
    assert seen.index[-1] == weekly.index[-1]
    assert float(seen["Close"].iloc[-1]) == pytest.approx(float(weekly["Close"].iloc[-1]))


def test_each_timeframe_answers_about_its_own_bar(client):
    """
    Same symbol, same download, three horizons: the anchor, the forecast date
    and the label all move together, and the label is the server's own wording
    rather than something the client assembles.
    """
    frame = _daily()
    expected_label = {"day": "Next 1 Day", "week": "Next 1 Week", "month": "Next 1 Month"}

    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: _FakePipeline()), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        for key, timeframe in TIMEFRAMES.items():
            payload = client.get(f"/api/predict/forecast/AAA?timeframe={key}").json()

            assert payload["status"] == "ok"
            assert payload["timeframe"] == key
            assert payload["horizon_label"] == expected_label[key]
            assert payload["bar_noun"] == timeframe.bar_noun
            assert payload["interval"] == timeframe.interval

            bars = resample_ohlcv(frame, timeframe)
            assert payload["as_of"] == str(pd.Timestamp(bars.index[-1]).date())
            assert payload["forecast_date"] == next_bar_date(bars.index[-1], timeframe)
            assert payload["forecast"][0]["date"] == payload["forecast_date"]
            # Bars of this timeframe, and the daily rows behind them, are
            # different numbers and both are reported.
            assert payload["bars_available"] == len(bars)
            assert payload["history_days"] == len(frame)


def test_as_of_never_runs_ahead_of_the_data(client):
    """
    `as_of` is read as "data through". On a forming monthly bar the period label
    is weeks away, so serving that would claim history nobody has.
    """
    frame = _daily()
    last_daily = str(pd.Timestamp(frame.index[-1]).date())
    with patch.object(predict_route, "_download_prediction_data", return_value=frame):
        for key in TIMEFRAMES:
            payload = client.get(f"/api/predict/history/AAA?timeframe={key}&bars=20").json()
            assert payload["as_of"] <= last_daily
            assert payload["bars"][-1]["date"] <= last_daily


def test_the_forecast_cache_does_not_serve_one_timeframe_under_another(client):
    """
    On a Friday the daily and weekly frames end on the same date, so a cache
    keyed on (symbol, bar) alone would return whichever ran first for both — and
    would do it silently, with the correct label on the wrong number.
    """
    # Truncated to a Friday, so both frames end on the same bar.
    frame = _daily().loc[: pd.Timestamp("2024-08-09")]
    assert frame.index[-1].dayofweek == 4, "the fixture has to end on a Friday"

    pipeline = _FakePipeline()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: pipeline), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        daily = client.get("/api/predict/forecast/AAA?timeframe=day").json()
        weekly = client.get("/api/predict/forecast/AAA?timeframe=week").json()

    assert daily["as_of"] == weekly["as_of"], "the premise: both frames end on the same bar"
    assert daily["horizon_label"] != weekly["horizon_label"]
    assert daily["forecast_date"] != weekly["forecast_date"]
    assert daily["bars_available"] != weekly["bars_available"]
    # Two runs, not one served twice.
    assert pipeline.runs == 2
    assert len(pipeline.seen[0]) != len(pipeline.seen[-1])


def test_the_same_timeframe_twice_is_computed_once(client):
    """The cache still does its job within a timeframe: ~7s of Kronos per bar."""
    frame = _daily()
    pipeline = _FakePipeline()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: pipeline), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        client.get("/api/predict/forecast/AAA?timeframe=month")
        client.get("/api/predict/forecast/AAA?timeframe=month")
    assert pipeline.runs == 1


def test_history_serves_the_same_candles_the_forecast_was_built_on(client):
    """
    The chart and the box are two requests and have to land on one frame. A
    weekly chart under a daily forecast would hang the estimate off a candle the
    models never saw.
    """
    frame = _daily()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: _FakePipeline()), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        history = client.get("/api/predict/history/AAA?timeframe=week&bars=40").json()
        forecast = client.get("/api/predict/forecast/AAA?timeframe=week").json()

    assert history["timeframe"] == "week"
    assert history["as_of"] == forecast["as_of"]
    assert len(history["bars"]) == 40
    assert history["bars"][-1]["date"] == forecast["as_of"]
    assert history["bars"][-1]["close"] == pytest.approx(forecast["anchor_price"], abs=0.01)
    # One bar per week, strictly increasing. The gap is seven days on a clean
    # calendar and shorter when a holiday moved the last session, so the
    # invariant is the ordering and the count, not the spacing.
    dates = [pd.Timestamp(bar["date"]) for bar in history["bars"]]
    assert all(a < b for a, b in zip(dates, dates[1:]))
    assert all(0 < (b - a).days <= 7 for a, b in zip(dates, dates[1:]))


def test_bars_counts_candles_of_the_selected_timeframe(client):
    """60 on the monthly view is five years, not two months."""
    frame = _daily()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame):
        monthly = client.get("/api/predict/history/AAA?timeframe=month&bars=60").json()
        daily = client.get("/api/predict/history/AAA?timeframe=day&bars=60").json()

    assert len(monthly["bars"]) == len(daily["bars"]) == 60

    def span(payload):
        first, last = payload["bars"][0]["date"], payload["bars"][-1]["date"]
        return pd.Timestamp(last) - pd.Timestamp(first)

    assert span(monthly) > span(daily) * 15


def test_the_deprecated_days_parameter_still_asks_for_the_same_thing(client):
    """A caller sending `days=` wanted this; the noun changing is not a reason to fail them."""
    frame = _daily()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame):
        assert len(client.get("/api/predict/history/AAA?days=30").json()["bars"]) == 30


def test_an_unknown_timeframe_is_refused_rather_than_silently_daily(client):
    frame = _daily()
    with patch.object(predict_route, "_download_prediction_data", return_value=frame):
        for path in ("history", "forecast"):
            response = client.get(f"/api/predict/{path}/AAA?timeframe=quarter")
            assert response.status_code == 422


def test_too_little_history_is_reported_in_the_bar_the_user_chose(client):
    """
    "40 monthly bars" and "40 trading days" are the same row count and different
    amounts of evidence. Reporting the daily number would say a monthly forecast
    failed for want of 40 rows while the frame held 900.
    """
    frame = _daily(rows=400)  # ~19 months
    with patch.object(predict_route, "_download_prediction_data", return_value=frame):
        response = client.get("/api/predict/forecast/AAA?timeframe=month")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "months of history" in detail
    assert "400 trading days" in detail


def test_thin_history_is_measured_in_the_selected_bar(client):
    """
    Kronos needs a 128-bar context, and the bars it counts are the ones it was
    handed. A monthly frame with 1,200 daily rows behind it has 57 bars, and it
    is the 57 that decides whether Kronos ran.
    """
    frame = _daily(rows=1200)
    with patch.object(predict_route, "_download_prediction_data", return_value=frame), \
         patch.object(predict_route, "_get_foundation_pipeline", lambda _: _FakePipeline()), \
         patch.object(predict_route, "_latest_available_price", return_value=_QUOTE):
        daily = client.get("/api/predict/forecast/AAA?timeframe=day").json()
        monthly = client.get("/api/predict/forecast/AAA?timeframe=month").json()

    assert daily["bars_available"] == 1200 and not daily["thin_history"]
    assert monthly["bars_available"] < 128 and monthly["thin_history"]


def test_each_timeframe_downloads_the_window_its_bars_need(client):
    """
    Five years of daily bars is sixty monthly ones, which is under Kronos's
    context. Each timeframe asks for its own history rather than sharing the
    daily default.
    """
    asked = {}

    def _record(symbol, min_rows=0, lookback_days=0):
        asked[lookback_days] = symbol
        return _daily()

    with patch.object(predict_route, "_download_prediction_data", side_effect=_record):
        for key in TIMEFRAMES:
            client.get(f"/api/predict/history/AAA?timeframe={key}&bars=30")

    windows = sorted(asked)
    assert windows == [TIMEFRAMES[key].lookback_days for key in ("day", "week", "month")]
    assert windows[0] < windows[1] < windows[2]
