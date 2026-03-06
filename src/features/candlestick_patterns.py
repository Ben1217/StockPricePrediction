"""
Candlestick Pattern Detection Module.

Detects the top 8 candlestick patterns from OHLCV data and returns
binary features suitable for model input.

Patterns detected:
  1. Doji          – indecision (body < 10% of range)
  2. Hammer        – bullish reversal at bottom
  3. Shooting Star – bearish reversal at top
  4. Bullish Engulfing
  5. Bearish Engulfing
  6. Morning Star  – 3-bar bullish reversal
  7. Evening Star  – 3-bar bearish reversal
  8. Harami        – inside bar (bullish / bearish)
"""

import numpy as np
import pandas as pd
from typing import List

from ..utils.logger import get_logger

logger = get_logger(__name__)


def _body(o, c):
    return abs(c - o)


def _upper_wick(h, o, c):
    return h - max(o, c)


def _lower_wick(l, o, c):
    return min(o, c) - l


def _range(h, l):
    return h - l


def detect_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect 8 candlestick patterns and return binary feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Open, High, Low, Close columns.

    Returns
    -------
    pd.DataFrame
        Columns: cdl_doji, cdl_hammer, cdl_shooting_star,
        cdl_bullish_engulfing, cdl_bearish_engulfing,
        cdl_morning_star, cdl_evening_star,
        cdl_bullish_harami, cdl_bearish_harami.
        Values are 1 (detected) or 0.
    """
    o = df['Open'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    c = df['Close'].values.astype(float)
    n = len(df)

    patterns = {
        'cdl_doji': np.zeros(n, dtype=np.int8),
        'cdl_hammer': np.zeros(n, dtype=np.int8),
        'cdl_shooting_star': np.zeros(n, dtype=np.int8),
        'cdl_bullish_engulfing': np.zeros(n, dtype=np.int8),
        'cdl_bearish_engulfing': np.zeros(n, dtype=np.int8),
        'cdl_morning_star': np.zeros(n, dtype=np.int8),
        'cdl_evening_star': np.zeros(n, dtype=np.int8),
        'cdl_bullish_harami': np.zeros(n, dtype=np.int8),
        'cdl_bearish_harami': np.zeros(n, dtype=np.int8),
    }

    for i in range(n):
        body_i = _body(o[i], c[i])
        range_i = _range(h[i], l[i])
        upper_i = _upper_wick(h[i], o[i], c[i])
        lower_i = _lower_wick(l[i], o[i], c[i])

        if range_i == 0:
            continue

        # 1. Doji: body < 10% of range
        if body_i < 0.1 * range_i:
            patterns['cdl_doji'][i] = 1

        # 2. Hammer: small body at top, long lower wick (>= 2x body)
        if (lower_i >= 2 * body_i and
                upper_i < body_i * 0.5 and
                body_i > 0 and
                c[i] >= o[i]):  # bullish body
            patterns['cdl_hammer'][i] = 1

        # 3. Shooting Star: small body at bottom, long upper wick (>= 2x body)
        if (upper_i >= 2 * body_i and
                lower_i < body_i * 0.5 and
                body_i > 0 and
                c[i] <= o[i]):  # bearish body
            patterns['cdl_shooting_star'][i] = 1

        # 2-bar patterns (need i >= 1)
        if i >= 1:
            body_prev = _body(o[i - 1], c[i - 1])
            prev_bearish = c[i - 1] < o[i - 1]
            prev_bullish = c[i - 1] > o[i - 1]
            curr_bullish = c[i] > o[i]
            curr_bearish = c[i] < o[i]

            # 4. Bullish Engulfing: prev bearish, curr bullish body engulfs prev body
            if (prev_bearish and curr_bullish and
                    o[i] <= c[i - 1] and c[i] >= o[i - 1] and
                    body_i > body_prev):
                patterns['cdl_bullish_engulfing'][i] = 1

            # 5. Bearish Engulfing: prev bullish, curr bearish body engulfs prev body
            if (prev_bullish and curr_bearish and
                    o[i] >= c[i - 1] and c[i] <= o[i - 1] and
                    body_i > body_prev):
                patterns['cdl_bearish_engulfing'][i] = 1

            # 8. Bullish Harami: prev bearish wide, curr small bullish inside prev body
            if (prev_bearish and curr_bullish and
                    body_i < body_prev * 0.5 and
                    c[i] < o[i - 1] and o[i] > c[i - 1]):
                patterns['cdl_bullish_harami'][i] = 1

            # 8b. Bearish Harami: prev bullish wide, curr small bearish inside prev body
            if (prev_bullish and curr_bearish and
                    body_i < body_prev * 0.5 and
                    c[i] > o[i - 1] and o[i] < c[i - 1]):
                patterns['cdl_bearish_harami'][i] = 1

        # 3-bar patterns (need i >= 2)
        if i >= 2:
            body_2 = _body(o[i - 2], c[i - 2])
            body_1 = _body(o[i - 1], c[i - 1])
            range_1 = _range(h[i - 1], l[i - 1])

            # 6. Morning Star: bar0 bearish, bar1 small (star), bar2 bullish
            if (c[i - 2] < o[i - 2] and        # bar0 bearish
                    body_1 < 0.3 * body_2 and   # bar1 is small star
                    c[i] > o[i] and              # bar2 bullish
                    c[i] > (o[i - 2] + c[i - 2]) / 2):  # bar2 closes above bar0 midpoint
                patterns['cdl_morning_star'][i] = 1

            # 7. Evening Star: bar0 bullish, bar1 small (star), bar2 bearish
            if (c[i - 2] > o[i - 2] and        # bar0 bullish
                    body_1 < 0.3 * body_2 and   # bar1 is small star
                    c[i] < o[i] and              # bar2 bearish
                    c[i] < (o[i - 2] + c[i - 2]) / 2):  # bar2 closes below bar0 midpoint
                patterns['cdl_evening_star'][i] = 1

    result = pd.DataFrame(patterns, index=df.index)

    detected = {k: int(v.sum()) for k, v in patterns.items() if v.sum() > 0}
    logger.info(f"Candlestick patterns detected: {detected if detected else 'none'}")

    return result
