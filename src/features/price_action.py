"""
Price action: reading market structure the way a chartist reads a chart.

:mod:`src.features.chart_patterns` already turns candle geometry into numbers a
model can fit on. What it does not do is *name* what the chart is doing, and
naming it is half the request: "higher high plus higher low" is a sentence a
person can check against the chart in front of them, where ``Donchian_Position_20
= 0.87`` is not.

This module produces both halves of that, from the same bars:

* a small set of **structure and event columns** - swing structure over three
  windows, breakout/breakdown, retest, rejection, consolidation, continuation
  and reversal - every one of them a causal function of bars at or before ``t``;
* a **composite score in [-1, 1]** per bar, positive for conventionally bullish
  structure, that the direction stack fits a weight on rather than assuming one.

The score is a plain unweighted mean of its component sub-scores. That is
deliberate: nothing here decides how much price action moves a probability. The
logistic stack in :mod:`src.models.direction_evidence` fits that weight against
this symbol's own realised next-day moves, so a symbol whose breakouts have
historically failed gets a negative coefficient rather than the positive one a
hand-tuned weight would have baked in.

Structure, concretely
---------------------
A swing high over window ``w`` is ``High.rolling(w).max()``; the prior swing high
is that same series shifted ``w`` bars back. Comparing the two pairs gives the
four cases a chartist reads::

    higher high + higher low   ->  +1.0   bullish structure
    higher high + lower low    ->   0.0   expanding range, no structure
    lower high  + higher low   ->   0.0   contracting range (coil)
    lower high  + lower low    ->  -1.0   bearish structure

Reading rolling extrema rather than confirmed pivots is a deliberate trade. A
pivot is only *confirmed* ``w`` bars after it prints, so a pivot-based structure
read at ``t`` describes the chart as it looked ``w`` bars ago. The rolling
extremum is knowable at ``t`` and is what the next bar actually reacts to.

Causality
---------
Every column at row ``t`` reads only bars at or before ``t``. Comparisons
against a channel exclude the current bar via ``.shift(1)`` wherever the current
bar would otherwise be part of the level it is being tested against - a close
compared to a 20-day high that includes today's high can never break out, which
both leaks and inverts the signal.

Public API:
    add_price_action_columns(df) -> DataFrame
    price_action_score(df) -> Series
    read_price_action(df) -> dict
    PRICE_ACTION_COLUMNS
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .chart_patterns import safe_divide, true_range

logger = get_logger(__name__)

# Swing windows. Two weeks, a month, a quarter - the three scales a chart is
# normally read at, and the three the direction panel reports separately.
SHORT_SWING_WINDOW = 10
MEDIUM_SWING_WINDOW = 20
LONG_SWING_WINDOW = 60
SWING_WINDOWS: tuple = (SHORT_SWING_WINDOW, MEDIUM_SWING_WINDOW, LONG_SWING_WINDOW)

# The channel a breakout is measured against, and how long a break stays
# "recent" for the purpose of calling a pullback a retest.
BREAKOUT_WINDOW = 20
RETEST_WINDOW = 10

ATR_WINDOW = 14

# A 5-bar range under this share of the 20-bar range is a coil, not a trend.
CONSOLIDATION_RATIO = 0.75

# How close to a level counts as "at" it, in ATR. Half a day's true range is the
# distance a single session can cover, so a close inside it is a level the next
# bar is still interacting with.
LEVEL_PROXIMITY_ATR = 0.5

PRICE_ACTION_COLUMNS: List[str] = [
    f"PA_Structure_{window}" for window in SWING_WINDOWS
] + [
    "PA_Breakout",
    "PA_Breakdown",
    "PA_Retest_Hold",
    "PA_Failed_Break",
    "PA_Consolidation",
    "PA_Continuation",
    "PA_Reversal",
    "PA_Close_Strength",
    "PA_Support_Distance_ATR",
    "PA_Resistance_Distance_ATR",
    "PA_Level_Reaction",
    "PA_Break_Bias",
]

# The sub-scores averaged into the composite. Each is already oriented so that
# positive means conventionally bullish and each is bounded in [-1, 1].
_SCORE_COMPONENTS: List[str] = [
    f"PA_Structure_{window}" for window in SWING_WINDOWS
] + [
    "PA_Break_Bias",
    "PA_Close_Strength",
    "PA_Level_Reaction",
]


def _atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Trailing ATR over ``ATR_WINDOW`` bars."""
    return true_range(high, low, close).rolling(ATR_WINDOW).mean()


def _structure(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Swing structure over ``window`` bars, in [-1, 1].

    ``+1`` is a higher high *and* a higher low, ``-1`` a lower high and a lower
    low, and the two mixed cases land on 0 because an expanding or contracting
    range is not a direction.
    """
    recent_high = high.rolling(window).max()
    recent_low = low.rolling(window).min()
    prior_high = recent_high.shift(window)
    prior_low = recent_low.shift(window)

    high_leg = np.sign(recent_high - prior_high)
    low_leg = np.sign(recent_low - prior_low)
    return (0.5 * high_leg + 0.5 * low_leg).rename(f"PA_Structure_{window}")


def add_price_action_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the structure, event and level columns named in ``PRICE_ACTION_COLUMNS``.

    Requires ``Open``/``High``/``Low``/``Close``; ``Volume`` is not read here -
    participation is the volume evidence category's job, not price action's.

    Event columns are emitted as floats so they can sit in the same matrix as
    the continuous ones, with NaN preserved through the warm-up rather than
    filled with a "no event" that was never observed.
    """
    data = df.copy()
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"add_price_action_columns requires {sorted(missing)}")

    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")

    atr = _atr(high, low, close)
    # A warm-up ATR of zero would divide every distance to infinity; keep those
    # rows missing so they are dropped rather than scored.
    atr = atr.where(atr > 0)

    for window in SWING_WINDOWS:
        data[f"PA_Structure_{window}"] = _structure(high, low, window)

    # -- the channel the break is measured against ---------------------------
    # shift(1) excludes today: a close cannot break a high it helped set.
    prior_high = high.rolling(BREAKOUT_WINDOW).max().shift(1)
    prior_low = low.rolling(BREAKOUT_WINDOW).min().shift(1)
    channel_known = prior_high.notna() & prior_low.notna()

    broke_out = (close > prior_high) & channel_known
    broke_down = (close < prior_low) & channel_known
    data["PA_Breakout"] = broke_out.astype(float).where(channel_known)
    data["PA_Breakdown"] = broke_down.astype(float).where(channel_known)

    # A failed break: the bar traded through the level and closed back inside
    # it. This is the rejection a chartist reads as evidence *against* the
    # direction of the poke, which is why it carries the opposite sign to the
    # break itself.
    poked_high = (high > prior_high) & (close <= prior_high)
    poked_low = (low < prior_low) & (close >= prior_low)
    data["PA_Failed_Break"] = pd.Series(
        np.where(poked_high, -1.0, np.where(poked_low, 1.0, 0.0)),
        index=data.index,
    ).where(channel_known)

    # A retest that held: price broke out within the last RETEST_WINDOW bars,
    # came back to within LEVEL_PROXIMITY_ATR of the level it broke, and is
    # still above it. The mirror case - a lost support retested from below and
    # rejected - scores -1.
    broke_out_recently = broke_out.rolling(RETEST_WINDOW, min_periods=1).max().fillna(0) > 0
    broke_down_recently = broke_down.rolling(RETEST_WINDOW, min_periods=1).max().fillna(0) > 0
    # The level that was broken, carried forward from the bar that broke it.
    broken_up_level = prior_high.where(broke_out).ffill(limit=RETEST_WINDOW)
    broken_down_level = prior_low.where(broke_down).ffill(limit=RETEST_WINDOW)

    retest_up = (
        broke_out_recently
        & (safe_divide(close - broken_up_level, atr).abs() <= LEVEL_PROXIMITY_ATR)
        & (close >= broken_up_level)
    )
    retest_down = (
        broke_down_recently
        & (safe_divide(broken_down_level - close, atr).abs() <= LEVEL_PROXIMITY_ATR)
        & (close <= broken_down_level)
    )
    data["PA_Retest_Hold"] = pd.Series(
        np.where(retest_up, 1.0, np.where(retest_down, -1.0, 0.0)),
        index=data.index,
    ).where(atr.notna() & channel_known)

    # -- range texture -------------------------------------------------------
    span = high - low
    contraction = safe_divide(span.rolling(5).mean(), span.rolling(BREAKOUT_WINDOW).mean())
    data["PA_Consolidation"] = (contraction < CONSOLIDATION_RATIO).astype(float).where(contraction.notna())

    # Continuation vs reversal, read off the medium structure against where it
    # stood a full swing ago. A structure that held its sign is a continuation;
    # one that flipped is a reversal, and the sign says which way.
    medium = data[f"PA_Structure_{MEDIUM_SWING_WINDOW}"]
    prior_medium = medium.shift(MEDIUM_SWING_WINDOW)
    both_known = medium.notna() & prior_medium.notna()
    same_sign = np.sign(medium) == np.sign(prior_medium)
    data["PA_Continuation"] = pd.Series(
        np.where(same_sign, np.sign(medium), 0.0), index=data.index
    ).where(both_known)
    data["PA_Reversal"] = pd.Series(
        np.where(~same_sign & (np.sign(prior_medium) != 0), np.sign(medium), 0.0),
        index=data.index,
    ).where(both_known)

    # -- where the close landed inside its own bar ---------------------------
    # A close on the high after a wide range is buyers finishing in control;
    # the same range closing on the low is the opposite. Rescaled to [-1, 1] so
    # it sits on the same axis as the structure columns.
    close_position = safe_divide(close - low, span).where(span > 0, 0.5).where(span.notna())
    data["PA_Close_Strength"] = 2.0 * close_position - 1.0

    # -- distance to the nearest level, in ATR -------------------------------
    data["PA_Support_Distance_ATR"] = safe_divide(close - prior_low, atr)
    data["PA_Resistance_Distance_ATR"] = safe_divide(prior_high - close, atr)

    # How hard the bar reacted at whichever level it is nearest. Proximity
    # decays with distance in ATR, so a bar in the middle of its range
    # contributes almost nothing and a bar sitting on a level contributes the
    # full sign of where it closed. A strong close at resistance is a break in
    # progress and a weak close there is a rejection - the same arithmetic
    # covers both, and support is its mirror.
    support_proximity = np.exp(-data["PA_Support_Distance_ATR"].clip(lower=0))
    resistance_proximity = np.exp(-data["PA_Resistance_Distance_ATR"].clip(lower=0))
    data["PA_Level_Reaction"] = (
        (support_proximity + resistance_proximity).clip(upper=1.0) * data["PA_Close_Strength"]
    )

    # The net directional read of the break/retest/rejection family, averaged
    # rather than summed so it stays inside [-1, 1] like its neighbours.
    data["PA_Break_Bias"] = pd.concat(
        [
            data["PA_Breakout"] - data["PA_Breakdown"],
            data["PA_Retest_Hold"],
            data["PA_Failed_Break"],
        ],
        axis=1,
    ).mean(axis=1).clip(-1.0, 1.0)

    return data


def price_action_score(df: pd.DataFrame, *, already_built: bool = False) -> pd.Series:
    """
    Composite price-action read per bar, in [-1, 1], positive for bullish.

    The unweighted mean of the three structure columns, the break/retest bias,
    the closing strength and the level reaction. Rows missing any component are
    NaN rather than partially scored - a structure read that silently drops its
    long window during the warm-up is a different feature from bar to bar.

    ``already_built`` skips recomputation when the caller has already run
    :func:`add_price_action_columns` on the frame.
    """
    data = df if already_built else add_price_action_columns(df)
    missing = [col for col in _SCORE_COMPONENTS if col not in data.columns]
    if missing:
        raise KeyError(f"price_action_score is missing {missing}")
    components = data[_SCORE_COMPONENTS]
    score = components.mean(axis=1).where(components.notna().all(axis=1))
    return score.clip(-1.0, 1.0).rename("price_action_score")


def _structure_label(value: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "unknown"
    if value >= 0.99:
        return "higher highs and higher lows"
    if value <= -0.99:
        return "lower highs and lower lows"
    if value > 0:
        return "higher highs, lower lows (expanding range)"
    if value < 0:
        return "lower highs, higher lows (contracting range)"
    return "no clear swing structure"


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def read_price_action(df: pd.DataFrame, *, already_built: bool = False) -> Dict[str, Any]:
    """
    The named price-action reading for the most recent bar.

    Returns the structure sentence at each swing window, the events currently
    active, the levels those events reference, and the composite score. This is
    the descriptive half - what the chart is doing. How much it should move a
    probability is decided by the fitted stack, not here.
    """
    data = df if already_built else add_price_action_columns(df)
    if data.empty:
        return {"available": False, "reason": "no bars"}

    score_series = price_action_score(data, already_built=True)
    usable = score_series.dropna()
    if usable.empty:
        return {"available": False, "reason": "not enough history for a structure read"}

    as_of = usable.index[-1]
    row = data.loc[as_of]

    structures = {}
    for window in SWING_WINDOWS:
        value = _finite(row.get(f"PA_Structure_{window}"))
        structures[str(window)] = {"score": value, "label": _structure_label(value)}

    events: List[str] = []
    if _finite(row.get("PA_Breakout")):
        events.append(f"breakout above the {BREAKOUT_WINDOW}-day high")
    if _finite(row.get("PA_Breakdown")):
        events.append(f"breakdown below the {BREAKOUT_WINDOW}-day low")
    retest = _finite(row.get("PA_Retest_Hold")) or 0.0
    if retest > 0:
        events.append("pullback holding the level it broke")
    elif retest < 0:
        events.append("pullback rejected at the level it lost")
    failed = _finite(row.get("PA_Failed_Break")) or 0.0
    if failed < 0:
        events.append("resistance rejection (traded through, closed back below)")
    elif failed > 0:
        events.append("support rejection (traded through, closed back above)")
    if _finite(row.get("PA_Consolidation")):
        events.append("range contracting (consolidation)")
    continuation = _finite(row.get("PA_Continuation")) or 0.0
    reversal = _finite(row.get("PA_Reversal")) or 0.0
    if continuation > 0:
        events.append("bullish trend continuation")
    elif continuation < 0:
        events.append("bearish trend continuation")
    if reversal > 0:
        events.append("structure flipped bullish (reversal)")
    elif reversal < 0:
        events.append("structure flipped bearish (reversal)")

    close = _finite(row.get("Close"))
    high_window = pd.to_numeric(data["High"], errors="coerce").rolling(BREAKOUT_WINDOW).max().shift(1)
    low_window = pd.to_numeric(data["Low"], errors="coerce").rolling(BREAKOUT_WINDOW).min().shift(1)
    resistance = _finite(high_window.get(as_of))
    support = _finite(low_window.get(as_of))

    score = float(usable.iloc[-1])
    return {
        "available": True,
        "as_of": str(pd.Timestamp(as_of).date()),
        "score": round(score, 6),
        "label": "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral",
        "structure": structures,
        "structure_label": structures[str(MEDIUM_SWING_WINDOW)]["label"],
        "events": events,
        "close": close,
        "levels": {
            "support": support,
            "resistance": resistance,
            "support_distance_atr": _finite(row.get("PA_Support_Distance_ATR")),
            "resistance_distance_atr": _finite(row.get("PA_Resistance_Distance_ATR")),
            "support_distance_pct": (
                round((close - support) / support * 100, 4)
                if close is not None and support not in (None, 0) else None
            ),
            "resistance_distance_pct": (
                round((resistance - close) / close * 100, 4)
                if close is not None and resistance is not None and close != 0 else None
            ),
            "window": BREAKOUT_WINDOW,
        },
        "close_strength": _finite(row.get("PA_Close_Strength")),
    }
