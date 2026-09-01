"""
Chart-pattern features: reading candlestick shape as a technical analyst would.

The 13 stationary columns describe *where price is* (above its 20-day mean, RSI
at 60). They say almost nothing about *what the chart looks like* — and the shape
of the recent candles is the thing a chartist actually reads. This module adds
that vocabulary, in four families:

**Candle geometry** — the alphabet. Where the close sits inside the bar, how long
the shadows are, how big the body is relative to a normal day. A long lower
shadow with the close near the high is a rejection of lower prices; the same
range with the close near the low is the opposite. ``High_Low_Range`` alone
cannot tell those two apart.

**Multi-bar shape** — the words. The previous candle's geometry, the run length
of same-signed days, whether the range is expanding or contracting. Two-candle
patterns (engulfing, harami) live here.

**Breakout and channel position** — the sentences. Did the close just take out
the prior 20-day high? Where does it sit inside the 20- and 60-day Donchian
channel? How long since the last new high?

**Trend cleanliness and volume confirmation** — the punctuation. A trend that
fits a straight line with R^2 0.9 is a different object from one that drifts to
the same place through chop, and the Kaufman efficiency ratio measures exactly
that: net movement divided by total movement. Volume features ask whether the
move was participated in or drifted.

Causality
---------
Every column at row ``t`` reads only bars at or before ``t``. Two places where
that is easy to get wrong, and how they are handled here:

* ``High_20d_Break`` compares today's close against the prior-20-day high, using
  ``.shift(1)`` so today's own high is excluded. Without the shift the feature is
  bounded above by 0 whenever today prints the high, which both leaks and inverts
  the signal.
* The rolling OLS trend uses ``sliding_window_view`` over trailing windows only.
  No centred windows, no full-series fits.

Public API:
    add_chart_pattern_features(df) -> DataFrame
    CHART_PATTERN_FEATURE_COLUMNS
    safe_divide(numerator, denominator) -> Series
    true_range(high, low, close) -> Series
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

ATR_WINDOW = 14
TREND_WINDOWS: Tuple[int, ...] = (20, 60)
EFFICIENCY_WINDOWS: Tuple[int, ...] = (10, 20)
DONCHIAN_WINDOWS: Tuple[int, ...] = (20, 60)
BREAKOUT_WINDOW = 20
VOLUME_WINDOW = 20

# Run lengths beyond this are vanishingly rare and would dominate a scaled
# feature; the column saturates instead of letting one 14-day streak set the
# scale for every other row.
MAX_RUN_LENGTH = 10

CHART_PATTERN_FEATURE_COLUMNS: List[str] = [
    # -- candle geometry -----------------------------------------------------
    "Body_Ratio",
    "Upper_Shadow_Ratio",
    "Lower_Shadow_Ratio",
    "Body_To_ATR",
    "Range_To_ATR",
    "Gap_To_ATR",
    # -- multi-bar shape -----------------------------------------------------
    "Body_Ratio_Lag1",
    "Close_Position_Lag1",
    "Close_Position_Mean_5",
    "Consecutive_Direction_Run",
    "Range_Expansion_5_20",
    # -- breakout and channel position ---------------------------------------
    "High_20d_Break",
    "Low_20d_Break",
    "Donchian_Position_20",
    "Donchian_Position_60",
    "Bars_Since_High_20",
    # -- trend cleanliness ---------------------------------------------------
    "Trend_Slope_20",
    "Trend_R2_20",
    "Trend_Slope_60",
    "Trend_R2_60",
    "Efficiency_Ratio_10",
    "Efficiency_Ratio_20",
    # -- volume confirmation -------------------------------------------------
    "Up_Volume_Ratio_20",
    "Volume_Price_Confirm",
    "OBV_Slope_20",
    # -- volatility texture --------------------------------------------------
    "ATR_Ratio",
    "Parkinson_Vol_Ratio",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping a zero or non-finite denominator to NaN rather than inf."""
    denominator = denominator.where(np.isfinite(denominator) & (denominator != 0))
    return numerator / denominator


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: the largest of today's span and the two gap spans."""
    previous_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)


def _rolling_trend(values: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Trailing OLS of ``values`` on time: per-bar slope and R^2.

    Fitted on log price, so the slope is a per-day log return and is comparable
    across price levels. R^2 is the cleanliness of the trend — how much of the
    window's movement a straight line explains. A 20-day advance with R^2 0.95
    is a trend; the same advance with R^2 0.15 is chop that happened to end
    higher, and a chartist treats those as different charts.

    Implemented with ``sliding_window_view`` so the whole series is one
    vectorised pass. Because the regressor is a fixed 0..w-1 ramp, the slope and
    R^2 reduce to covariance identities rather than a per-window ``lstsq``.
    """
    series = pd.to_numeric(values, errors="coerce")
    array = series.to_numpy(dtype=np.float64)
    n = array.size

    slope = np.full(n, np.nan)
    r_squared = np.full(n, np.nan)
    if n < window:
        return pd.Series(slope, index=series.index), pd.Series(r_squared, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(array, window)  # (n-window+1, window)

    x = np.arange(window, dtype=np.float64)
    x_centred = x - x.mean()
    x_variance = float(np.sum(x_centred ** 2))

    y_mean = windows.mean(axis=1, keepdims=True)
    y_centred = windows - y_mean

    covariance = y_centred @ x_centred          # sum((x-xbar)(y-ybar)) per window
    y_variance = np.sum(y_centred ** 2, axis=1)

    window_slope = covariance / x_variance
    # R^2 of a simple linear fit is the squared correlation. A flat window has
    # zero variance to explain, so R^2 is undefined rather than 1.
    with np.errstate(divide="ignore", invalid="ignore"):
        window_r2 = np.where(
            y_variance > 0, (covariance ** 2) / (x_variance * y_variance), np.nan
        )

    # Each window ends at position window-1 + i, which is the row it describes.
    slope[window - 1:] = window_slope
    r_squared[window - 1:] = window_r2
    return (
        pd.Series(slope, index=series.index),
        pd.Series(r_squared, index=series.index),
    )


def _bars_since_high(high: pd.Series, window: int) -> pd.Series:
    """
    Bars since the highest high of the trailing window, scaled to [0, 1].

    0.0 means today printed the window high; 1.0 means the high is the oldest
    bar in the window and price has been falling away from it since.
    """
    array = pd.to_numeric(high, errors="coerce").to_numpy(dtype=np.float64)
    n = array.size
    result = np.full(n, np.nan)
    if n < window:
        return pd.Series(result, index=high.index)

    windows = np.lib.stride_tricks.sliding_window_view(array, window)
    # argmax returns the FIRST maximum; on ties the older bar wins, which is the
    # conservative reading (the high is older than it might be).
    position_of_high = np.argmax(windows, axis=1)
    result[window - 1:] = (window - 1 - position_of_high) / float(window - 1)
    # A window containing a NaN gives a meaningless argmax; mask those rows.
    has_nan = np.isnan(windows).any(axis=1)
    result[window - 1:][has_nan] = np.nan
    return pd.Series(result, index=high.index)


def _signed_run_length(daily_return: pd.Series) -> pd.Series:
    """
    Signed length of the current same-direction streak, saturated and scaled.

    +1.0 is ten or more consecutive up days, -1.0 ten or more down days, 0.0 a
    flat day. Causal: the run is counted within the current streak only, using a
    cumulative count that cannot see past the current row.
    """
    sign = np.sign(pd.to_numeric(daily_return, errors="coerce"))
    # A new group starts whenever the sign changes (NaN compares unequal, which
    # correctly starts a fresh run after a gap).
    group = (sign != sign.shift()).cumsum()
    run_length = sign.groupby(group).cumcount() + 1
    signed = sign * run_length.clip(upper=MAX_RUN_LENGTH)
    return signed / float(MAX_RUN_LENGTH)


def add_chart_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the chart-pattern columns listed in ``CHART_PATTERN_FEATURE_COLUMNS``.

    Requires ``Open``/``High``/``Low``/``Close``; ``Volume`` enables the three
    volume-confirmation columns and is required for the full set. Uses
    ``Daily_Return`` and ``Volume_Zscore`` when the frame already carries them,
    deriving them otherwise, so the function works on a raw OHLCV frame or on
    the enriched regression feature frame.
    """
    data = df.copy()
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"add_chart_pattern_features requires {sorted(missing)}")

    open_ = pd.to_numeric(data["Open"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")
    safe_close = close.replace(0, np.nan)

    span = high - low
    body = close - open_
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low

    # -- candle geometry -----------------------------------------------------
    # Signed body as a share of the day's range: +1 is a close on the high after
    # opening on the low (a marubozu), -1 the mirror image, 0 a doji.
    data["Body_Ratio"] = safe_divide(body, span).where(span > 0, 0.0).where(span.notna())
    data["Upper_Shadow_Ratio"] = safe_divide(upper_shadow, span).where(span > 0, 0.0).where(span.notna())
    data["Lower_Shadow_Ratio"] = safe_divide(lower_shadow, span).where(span > 0, 0.0).where(span.notna())

    tr = true_range(high, low, close)
    atr = tr.rolling(ATR_WINDOW).mean()
    data["Body_To_ATR"] = safe_divide(body, atr)
    data["Range_To_ATR"] = safe_divide(span, atr)
    data["Gap_To_ATR"] = safe_divide(open_ - close.shift(1), atr)
    data["ATR_Ratio"] = safe_divide(atr, safe_close)

    # -- multi-bar shape -----------------------------------------------------
    close_position = safe_divide(close - low, span).where(span > 0, 0.5).where(span.notna())
    data["Body_Ratio_Lag1"] = data["Body_Ratio"].shift(1)
    data["Close_Position_Lag1"] = close_position.shift(1)
    # Persistent closing strength: five days of closing near the high is a
    # different chart from one strong day and four weak ones.
    data["Close_Position_Mean_5"] = close_position.rolling(5).mean()

    daily_return = data["Daily_Return"] if "Daily_Return" in data.columns else close.pct_change()
    daily_return = pd.to_numeric(daily_return, errors="coerce")
    data["Consecutive_Direction_Run"] = _signed_run_length(daily_return)

    # Expanding ranges mark breakouts and capitulation; contracting ranges mark
    # coiling. The ratio is scale-free in both price and volatility regime.
    data["Range_Expansion_5_20"] = safe_divide(span.rolling(5).mean(), span.rolling(VOLUME_WINDOW).mean())

    # -- breakout and channel position ---------------------------------------
    # shift(1) excludes today's own bar, so this asks "did the close take out the
    # level that existed before today", which is the question a chartist asks.
    prior_high = high.rolling(BREAKOUT_WINDOW).max().shift(1)
    prior_low = low.rolling(BREAKOUT_WINDOW).min().shift(1)
    data["High_20d_Break"] = safe_divide(close, prior_high) - 1.0
    data["Low_20d_Break"] = safe_divide(close, prior_low) - 1.0

    for window in DONCHIAN_WINDOWS:
        channel_high = high.rolling(window).max()
        channel_low = low.rolling(window).min()
        channel_span = channel_high - channel_low
        data[f"Donchian_Position_{window}"] = (
            safe_divide(close - channel_low, channel_span)
            .where(channel_span > 0, 0.5)
            .where(channel_span.notna())
        )

    data["Bars_Since_High_20"] = _bars_since_high(high, BREAKOUT_WINDOW)

    # -- trend cleanliness ---------------------------------------------------
    log_close = np.log(safe_close)
    for window in TREND_WINDOWS:
        slope, r_squared = _rolling_trend(log_close, window)
        data[f"Trend_Slope_{window}"] = slope
        data[f"Trend_R2_{window}"] = r_squared

    # Kaufman efficiency: net distance travelled over total distance travelled.
    # 1.0 is a straight line, near 0 is pure chop. This is "how clean is the
    # trend line" stated as a number.
    absolute_step = close.diff().abs()
    for window in EFFICIENCY_WINDOWS:
        net_move = (close - close.shift(window)).abs()
        total_move = absolute_step.rolling(window).sum()
        data[f"Efficiency_Ratio_{window}"] = safe_divide(net_move, total_move)

    # -- volume confirmation -------------------------------------------------
    if "Volume" in data.columns:
        volume = pd.to_numeric(data["Volume"], errors="coerce")
        up_day = (daily_return > 0).astype(float)
        up_volume = (volume * up_day).rolling(VOLUME_WINDOW).sum()
        total_volume = volume.rolling(VOLUME_WINDOW).sum()
        # Above 0.5 means the last month's volume arrived mostly on up days:
        # accumulation rather than distribution.
        data["Up_Volume_Ratio_20"] = safe_divide(up_volume, total_volume)

        if "Volume_Zscore" in data.columns:
            volume_zscore = pd.to_numeric(data["Volume_Zscore"], errors="coerce")
        else:
            volume_mean = volume.rolling(VOLUME_WINDOW).mean()
            volume_std = volume.rolling(VOLUME_WINDOW).std()
            volume_zscore = safe_divide(volume - volume_mean, volume_std)
        # Positive when an unusually heavy day moved up, negative when it moved
        # down: the sign of the move weighted by how much participation it drew.
        data["Volume_Price_Confirm"] = np.sign(daily_return) * volume_zscore

        on_balance_volume = (np.sign(daily_return).fillna(0.0) * volume.fillna(0.0)).cumsum()
        obv_slope, _ = _rolling_trend(on_balance_volume, VOLUME_WINDOW)
        # OBV is a cumulative share count, so its raw slope scales with the
        # ticker's turnover. Dividing by average volume makes it comparable
        # across names and across a decade of volume growth in one name.
        data["OBV_Slope_20"] = safe_divide(obv_slope, volume.rolling(VOLUME_WINDOW).mean())
    else:
        logger.warning("No Volume column: the three volume-confirmation columns will be NaN")
        for column in ("Up_Volume_Ratio_20", "Volume_Price_Confirm", "OBV_Slope_20"):
            data[column] = np.nan

    # -- volatility texture --------------------------------------------------
    # Parkinson uses the high-low range; close-to-close uses only the close. The
    # ratio separates a market that moves intraday and closes flat from one that
    # gaps. Both are "volatile"; they are not the same chart.
    log_high_low = np.log(safe_divide(high, low))
    parkinson = np.sqrt(
        (log_high_low ** 2).rolling(VOLUME_WINDOW).mean() / (4.0 * np.log(2.0))
    )
    close_to_close = daily_return.rolling(VOLUME_WINDOW).std()
    data["Parkinson_Vol_Ratio"] = safe_divide(parkinson, close_to_close)

    return data
