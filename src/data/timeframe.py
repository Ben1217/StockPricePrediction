"""
What DAY, WEEK and MONTH mean, in one place.

The Predictions tab lets a reader choose the bar the models forecast. That
choice is not a display filter: every member of the foundation stack is a
one-step model, so the *bar it is handed* is the horizon it predicts. Feed it
daily candles and it answers "tomorrow"; feed it weekly candles built from the
same downloads and it answers "next week", because the next bar now is a week.
Nothing inside the models changes, and nothing in them needs to.

That makes the resampler the load-bearing part, and it is why the aggregation
lives here rather than being spelled out at each call site::

    Open   first of the period        High  max of the period
    Close  last  of the period        Low   min of the period
    Volume sum   over the period

Getting one of those wrong is not a visible bug. A weekly frame whose Open is
the period's *last* open still charts, still forecasts, and is simply a
different instrument from the one the user asked about.

Bar counts stay bar counts
--------------------------
RSI_14 on weekly candles is the weekly RSI(14) a trader means by the name --
fourteen weeks, not fourteen days -- so the indicator windows downstream are
left alone. What does scale with the timeframe is anything expressing
*calendar* time or *statistical power*: how far back to download, how many bars
a walk-forward needs, the trailing window an evidence score is normalised
against. Those are the fields on :class:`Timeframe`, and they are why this is a
table rather than three string constants.

What does not scale is honesty about history. A monthly walk-forward wants
decades of bars and most tickers do not have them; the floors here are set to
what is genuinely available, and a symbol that misses them is told so rather
than served a number fitted on thirty rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

#: How each OHLCV column collapses over a period. The whole point of the module.
OHLCV_AGGREGATION = {
    "Open": "first",
    "High": "max",
    "Low": "min",
    "Close": "last",
    "Volume": "sum",
}


@dataclass(frozen=True)
class Timeframe:
    """One selectable bar size, and every number that depends on it."""

    key: str
    label: str
    #: Singular noun for one bar of this size, as the UI writes it in a
    #: sentence: "the next session", "the next week".
    bar_noun: str
    #: Plural, for counts. English is irregular enough that deriving it is not
    #: worth the one field.
    bar_noun_plural: str
    #: ``DataFrame.resample`` rule, or None for daily -- daily bars *are* the
    #: download, and resampling them onto themselves would only invent holidays.
    resample_rule: Optional[str]
    #: How the rest of the API spells this interval (patterns, prices).
    interval: str
    #: Calendar days of daily history to download before resampling. Sized so
    #: the resampled frame clears ``full_stack_bars`` for an established name:
    #: 128 weekly bars is two and a half years, 128 monthly bars is over ten.
    lookback_days: int
    #: Resampled bars below which no forecast is served at all.
    min_bars: int
    #: Below this Kronos loses its 128-bar context and drops out, leaving a
    #: two-member forecast. Still served; the caller is told.
    full_stack_bars: int
    #: "Next 1 Day" / "Next 1 Week" / "Next 1 Month", for the forecast box.
    horizon_label: str
    #: Trailing window the evidence scores are normalised against, in bars --
    #: about a year of them, so the scale tracks a regime rather than a decade.
    scale_window: int
    #: Minimum observations before that window reports a scale at all.
    min_scale_observations: int
    #: Rows the evidence stack's own walk-forward trains on, and the smallest
    #: test block it will score. Both in bars of this timeframe.
    stack_train_rows: int
    stack_test_rows: int
    #: Bars of resolved history before the nearest-neighbour analog read is
    #: allowed to contribute.
    analog_min_history: int
    #: Lookback in bars for the support/resistance route, inside the limits
    #: ``src.api.routes.patterns`` clamps each interval to.
    sr_lookback: int

    @property
    def min_stack_rows(self) -> int:
        """Bars needed before a walk-forward is attempted at all."""
        return self.stack_train_rows + self.stack_test_rows


#: The three selectable timeframes, keyed by what the API accepts.
#:
#: The daily row restates today's behaviour exactly -- 1825 calendar days, a
#: 40-bar floor, a 128-bar full stack, a 252-bar scale window and the 400/40
#: walk-forward split -- so selecting DAY runs the pipeline that was already
#: shipping rather than a re-derivation of it.
TIMEFRAMES: Dict[str, Timeframe] = {
    "day": Timeframe(
        key="day",
        label="Day",
        bar_noun="session",
        bar_noun_plural="sessions",
        resample_rule=None,
        interval="1d",
        lookback_days=1825,
        min_bars=40,
        full_stack_bars=128,
        horizon_label="Next 1 Day",
        scale_window=252,
        min_scale_observations=60,
        stack_train_rows=400,
        stack_test_rows=40,
        analog_min_history=250,
        sr_lookback=180,
    ),
    "week": Timeframe(
        key="week",
        label="Week",
        bar_noun="week",
        bar_noun_plural="weeks",
        resample_rule="W-FRI",
        interval="1wk",
        # Twenty years of daily bars is a little over a thousand weekly ones:
        # enough for the 128-bar context, the 52-bar scale window and a
        # walk-forward, with room for a name that listed part-way through.
        lookback_days=365 * 20,
        min_bars=40,
        full_stack_bars=128,
        horizon_label="Next 1 Week",
        scale_window=52,
        min_scale_observations=26,
        # Five years of training rows and a half-year test block. Lower than
        # the daily floor because no ticker has 440 weekly bars to spare that
        # is not already twenty years old -- and the walk-forward still reports
        # the width of its own confidence interval, so a thin sample answers
        # "no edge" rather than answering confidently.
        stack_train_rows=260,
        stack_test_rows=26,
        analog_min_history=104,
        sr_lookback=260,
    ),
    "month": Timeframe(
        key="month",
        label="Month",
        bar_noun="month",
        bar_noun_plural="months",
        resample_rule="ME",
        interval="1mo",
        # Yahoo clamps this to the listing date, so asking for forty years
        # simply takes everything the ticker has -- which is what a monthly bar
        # needs and what most tickers cannot supply.
        lookback_days=365 * 40,
        min_bars=36,
        full_stack_bars=128,
        horizon_label="Next 1 Month",
        scale_window=36,
        min_scale_observations=18,
        # Ten years of training rows. A monthly walk-forward is a small-sample
        # exercise however it is arranged; this is the floor below which the
        # folds stop being separable from each other, not a threshold that
        # makes them precise.
        stack_train_rows=120,
        stack_test_rows=12,
        analog_min_history=60,
        sr_lookback=240,
    ),
}

DEFAULT_TIMEFRAME = "day"

TIMEFRAME_KEYS: Tuple[str, ...] = tuple(TIMEFRAMES)

#: Accepted spellings that are not the canonical key. The frontend sends the
#: canonical one; these exist so a hand-written request, and the interval names
#: the rest of the API already speaks, land on the same table.
_ALIASES: Dict[str, str] = {
    "d": "day", "1d": "day", "daily": "day", "days": "day",
    "w": "week", "1wk": "week", "1w": "week", "weekly": "week", "weeks": "week",
    "mo": "month", "1mo": "month", "monthly": "month", "months": "month",
}


def resolve_timeframe(value: Optional[str]) -> Timeframe:
    """
    The :class:`Timeframe` for ``value``, or a ValueError naming the choices.

    ``None`` and the empty string resolve to daily, so every existing caller
    that has never heard of a timeframe keeps the behaviour it has today.
    """
    if value is None or not str(value).strip():
        return TIMEFRAMES[DEFAULT_TIMEFRAME]
    key = str(value).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in TIMEFRAMES:
        raise ValueError(
            f"Unknown timeframe '{value}'. Available: {', '.join(TIMEFRAME_KEYS)}."
        )
    return TIMEFRAMES[key]


def resample_ohlcv(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """
    Daily bars aggregated onto ``timeframe``, keeping the forming period.

    Empty bins are dropped rather than filled. A week the market never opened
    is not a candle with last week's close repeated four times; leaving it in
    would hand the models a flat bar and the volatility features a zero range
    that never happened.

    The frame that comes back has exactly the columns that went in, so it can
    be handed to anything that accepts the daily frame without a filtering step
    the caller has to remember. The two facts a caller needs *about* the
    periods -- how many sessions the last bar holds, and when its period closes
    -- come from :func:`period_sessions` and :func:`period_end` rather than
    from columns riding along inside the model input.

    The final bar is deliberately kept even when its period has not finished.
    That bar is what every live chart draws -- the candle currently forming --
    and it carries the most recent close, which is the anchor the forecast is
    measured against. Callers that need to say so read
    :func:`last_bar_is_forming`; the honest move is to label it, not to drop
    the newest information the models have.
    """
    if timeframe.resample_rule is None:
        return df
    if df is None or df.empty:
        return df

    frame = df.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    columns = {name: how for name, how in OHLCV_AGGREGATION.items() if name in frame.columns}
    if "Close" not in columns:
        raise ValueError("Cannot resample a frame that has no Close column")

    resampled = frame.resample(timeframe.resample_rule).agg(columns)
    # `sum` over an empty bin returns 0 rather than NaN, so Volume cannot be
    # used to find the gaps. Close is a `last` and is NaN exactly when the bin
    # held no trading days, which is the definition being applied.
    resampled = resampled[resampled["Close"].notna()]

    # Index each bar by the last session it actually contains, not by the
    # period label pandas puts there.
    #
    # The calendar label put a September monthly candle at 2026-09-30 on a
    # chart drawn on the 6th: a bar three weeks in the future, under an `as_of`
    # claiming data through a date that had not happened. The bar holds four
    # sessions ending on the 4th, so the 4th is when its data ends and where it
    # belongs on a time axis. The period it covers is still recoverable --
    # `period_end` derives it from this index -- so nothing is lost by not
    # carrying it.
    last_session = frame.index.to_series().resample(timeframe.resample_rule).last()
    resampled.index = pd.DatetimeIndex(last_session.reindex(resampled.index).to_numpy())
    resampled.index.name = frame.index.name

    if "Volume" in resampled.columns:
        resampled["Volume"] = resampled["Volume"].fillna(0)
    return resampled


def period_end(bars, timeframe: Timeframe) -> pd.Timestamp:
    """
    The calendar close of the period the last bar covers.

    Resampled frames are indexed by the last session inside each bar, so on a
    forming bar -- and on any bar whose period ended over a weekend or a
    holiday -- the index and the period end are different dates. Everything
    that reasons about *periods* starts here: whether the bar has closed, what
    the next one will be, which days it spans.

    Derived from the index rather than stored, because ``date_range`` rolls a
    date forward to the first grid point at or after it, which is the
    definition of the period containing that date. A column would have to ride
    along inside the frame handed to the models.
    """
    last = bars.index[-1] if hasattr(bars, "index") else bars
    timestamp = pd.Timestamp(last)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    if timeframe.resample_rule is None:
        return timestamp
    return pd.Timestamp(pd.date_range(start=timestamp, periods=1, freq=timeframe.resample_rule)[0])


def period_sessions(daily: pd.DataFrame, timeframe: Timeframe) -> pd.Series:
    """
    Daily bars inside each period, on the same index :func:`resample_ohlcv` uses.

    The numerator of "4 of 21 sessions" on a forming month, and the only way to
    tell a holiday-shortened week from a full one. Served separately rather than
    as a column so the resampled frame stays shaped exactly like the daily one.
    """
    if timeframe.resample_rule is None:
        return pd.Series(1, index=daily.index, dtype=int)
    counts = daily["Close"].resample(timeframe.resample_rule).count()
    counts = counts[counts > 0]
    last_session = daily.index.to_series().resample(timeframe.resample_rule).last()
    counts.index = pd.DatetimeIndex(last_session.reindex(counts.index).to_numpy())
    return counts.astype(int)


def last_bar_is_forming(
    resampled: pd.DataFrame,
    timeframe: Timeframe,
    *,
    today: Optional[pd.Timestamp] = None,
) -> bool:
    """
    True when the final bar's period has not closed yet.

    Measured against the calendar rather than against the daily frame: a week
    whose Friday was a holiday is complete even though no bar printed on the
    label date, and comparing the two indices would call that week forming for
    ever.
    """
    if timeframe.resample_rule is None or resampled is None or resampled.empty:
        return False
    reference = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    return period_end(resampled, timeframe).normalize() >= reference.normalize()


def next_bar_date(last_index, timeframe: Timeframe) -> str:
    """
    The label the bar being forecast will carry, as a date string.

    For daily that is the next business day. For the resampled frames it is the
    next period end -- the Friday after this one, the last day of next month --
    which is the label the candle prints under, so a client can align the
    forecast point with the bar it belongs to.
    """
    timestamp = pd.Timestamp(last_index)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)

    if timeframe.resample_rule is None:
        return str(pd.bdate_range(start=timestamp, periods=2)[1].date())

    # `date_range` anchored on the current label yields that label first, so the
    # second element is the next period end under the same rule that produced
    # the index. Deriving it any other way -- adding seven days, adding a month
    # -- reimplements the offset and drifts at month ends.
    following = pd.date_range(start=timestamp, periods=2, freq=timeframe.resample_rule)
    return str(pd.Timestamp(following[-1]).date())


def period_bounds(last_index, timeframe: Timeframe) -> Tuple[str, str]:
    """The first and last calendar day covered by the bar labelled ``last_index``."""
    end = pd.Timestamp(last_index)
    if end.tzinfo is not None:
        end = end.tz_localize(None)
    if timeframe.resample_rule is None:
        return str(end.date()), str(end.date())
    previous = pd.date_range(end=end, periods=2, freq=timeframe.resample_rule)
    start = pd.Timestamp(previous[0]) + pd.Timedelta(days=1)
    return str(start.date()), str(end.date())


def describe_timeframe(timeframe: Timeframe) -> Dict[str, object]:
    """The timeframe fields a client needs to label what it is showing."""
    return {
        "timeframe": timeframe.key,
        "timeframe_label": timeframe.label,
        "bar_noun": timeframe.bar_noun,
        "bar_noun_plural": timeframe.bar_noun_plural,
        "interval": timeframe.interval,
        "horizon_label": timeframe.horizon_label,
    }


def timeframe_options() -> List[Dict[str, object]]:
    """Every selectable timeframe, in the order a selector should show them."""
    return [describe_timeframe(timeframe) for timeframe in TIMEFRAMES.values()]
