"""
Direction from combined evidence, with the evidence kept visible.

The next-day classifier in :mod:`src.models.direction_pipeline` answers "up or
down" from a 46-column feature vector. It answers well or badly, and the
walk-forward report says which - but it cannot say *why*, and "why" is most of
what a reader wants from a direction call. A single probability with no account
of what produced it is the thing that makes a model look like an oracle.

This module produces the account. It reduces the same bars to seven named
categories of evidence - the ones a person actually names when they explain a
chart::

    trend                where price sits against its own moving averages
    momentum             RSI, MACD, multi-week rate of change
    volume               participation: up-volume share, OBV slope, confirmation
    price_action         swing structure and closing strength
    support_resistance   what happened at the levels: breaks, retests, rejections
    volatility           the regime the move is happening in
    historical_analogs   what followed the most similar setups in this symbol's past

It fits a logistic stack on those seven scores and reports each one's
*contribution to the answer in percentage points*. The seven scores are
descriptive and hand-oriented (positive = conventionally bullish); the weight
each one carries is fitted against this symbol's own realised next-day moves.
That split is the point. Nothing here decides that a breakout is worth five
points of probability - the data decides, and for a symbol whose breakouts have
historically failed the fitted coefficient comes out negative.

Blending, and the right to be uncertain
---------------------------------------
Two probabilities are available for the same session: this evidence stack's, and
the existing classifier's. They are combined in log-odds, weighted by each
one's **measured out-of-sample Brier skill** - not by a chosen ratio. A source
that scored no better than always predicting the base rate gets weight zero, and
when *both* score zero the blend does not fall back to whichever number looks
more interesting: it returns the base rate and calls the direction NEUTRAL. A
model with no measured skill is not entitled to an opinion.

The confidence label is likewise read off data. ``|p - 0.5|`` is compared
against the terciles of the *same stack's own out-of-sample* conviction, so
"Moderate" means "this model is more sure than it usually is, but not much", and
a stack whose probabilities never leave 0.48-0.52 can never report High.

Public API:
    analyse_direction(bars, ...) -> dict
    build_evidence_frame(bars) -> DataFrame
    EVIDENCE_CATEGORIES
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..features.chart_patterns import add_chart_pattern_features
from ..features.direction_features import (
    DIRECTION_FEATURE_CONFIG,
    add_direction_features,
)
from ..features.feature_engineering import build_regression_feature_frame
from ..features.historical_analogs import (
    ANALOG_FEATURE_COLUMNS,
    DEFAULT_K,
    MIN_ANALOG_HISTORY,
    analog_matches,
    analog_up_rate_series,
)
from ..features.price_action import (
    LONG_SWING_WINDOW,
    MEDIUM_SWING_WINDOW,
    SHORT_SWING_WINDOW,
    add_price_action_columns,
    read_price_action,
)
from ..utils.logger import get_logger
from .direction_metrics import classifier_skill, wilson_interval
from .direction_models import LogisticDirection

logger = get_logger(__name__)

EVIDENCE_CATEGORIES: Tuple[str, ...] = (
    "trend",
    "momentum",
    "volume",
    "price_action",
    "support_resistance",
    "volatility",
    "historical_analogs",
)

# Readable names for the panel, kept next to the definitions they label.
CATEGORY_LABELS: Dict[str, str] = {
    "trend": "Trend",
    "momentum": "Momentum",
    "volume": "Volume",
    "price_action": "Price action",
    "support_resistance": "Support / resistance",
    "volatility": "Volatility regime",
    "historical_analogs": "Historical analogs",
}

# Trailing window for the scale each unbounded column is squashed against. One
# trading year is long enough to be stable and short enough to track a regime.
SCALE_WINDOW = 252
MIN_SCALE_OBSERVATIONS = 60

# Walk-forward settings for the stack's own evaluation. Seven features need far
# less training data than the 46-column model, so a shorter minimum is honest
# here; the folds still never train on a row that follows their test block.
MIN_STACK_TRAIN_ROWS = 400
STACK_TEST_FOLDS = 5
MIN_STACK_TEST_ROWS = 40

# Probabilities are clipped before any log-odds arithmetic. A blended logit is
# undefined at exactly 0 or 1, and no honest daily direction model reaches
# either.
_PROBABILITY_EPS = 1e-6

# Horizon reads. Each is (label, structure window, return column, trend ratio).
_HORIZON_SPECS: Tuple[Tuple[str, str, int, str, str], ...] = (
    ("short", "recent price action", SHORT_SWING_WINDOW, "Return_5d", "Close_SMA20_Ratio"),
    ("medium", "trend and momentum", MEDIUM_SWING_WINDOW, "Return_20d", "Close_SMA50_Ratio"),
    ("long", "broader structure", LONG_SWING_WINDOW, "Return_60d", "Close_SMA200_Ratio"),
)


# ─────────────────────────────────────────────────────────────────────────────
# Evidence construction
# ─────────────────────────────────────────────────────────────────────────────
def _bounded(series: pd.Series) -> pd.Series:
    """
    Squash an unbounded, naturally zero-centred column into [-1, 1].

    ``tanh(x / rms)`` where ``rms`` is the trailing root-mean-square of the same
    column. Dividing by the RMS rather than the standard deviation preserves the
    column's own zero - the point at which "price equals its moving average" or
    "MACD equals its signal" stops being directional - which subtracting a
    trailing mean would move. The window is trailing, so row ``t`` is never
    scaled by a spread that includes ``t+1``.
    """
    values = pd.to_numeric(series, errors="coerce")
    rms = np.sqrt(
        (values ** 2).rolling(SCALE_WINDOW, min_periods=MIN_SCALE_OBSERVATIONS).mean()
    )
    return np.tanh(values / rms.where(rms > 0))


def _centred(series: pd.Series, centre: float, half_range: float) -> pd.Series:
    """Rescale a column with known bounds onto [-1, 1] (RSI, ratios in [0, 1])."""
    values = pd.to_numeric(series, errors="coerce")
    return ((values - centre) / half_range).clip(-1.0, 1.0)


def _mean_of(columns: Sequence[pd.Series]) -> pd.Series:
    """
    Unweighted mean of sub-scores, NaN where any of them is missing.

    Averaging over whatever happens to be present would silently change the
    definition of the category from bar to bar - a "trend" score built from six
    inputs in 2024 and two in 2019 is two different features wearing one name.
    """
    stacked = pd.concat(columns, axis=1)
    return stacked.mean(axis=1).where(stacked.notna().all(axis=1)).clip(-1.0, 1.0)


def build_evidence_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """
    Indicators, chart-pattern geometry and price-action structure on one frame.

    This is the single feature build the evidence engine reads. It runs the same
    causal chain the direction dataset uses, plus the price-action columns and a
    ``Close_SMA200_Ratio`` the long-horizon read needs.
    """
    frame = build_regression_feature_frame(bars, feature_config=dict(DIRECTION_FEATURE_CONFIG))
    if frame.empty:
        raise ValueError("Feature frame is empty; check the input bars")
    frame = add_direction_features(frame)
    frame = add_chart_pattern_features(frame)
    frame = add_price_action_columns(frame)

    if "SMA_200" in frame.columns:
        sma200 = pd.to_numeric(frame["SMA_200"], errors="coerce")
        # Same convention as its siblings in the stationary set: despite the
        # name, Close_SMA20_Ratio and Close_SMA50_Ratio are `close / sma - 1`,
        # a deviation centred on zero rather than a ratio centred on one.
        # Defining this one any other way would have it read backwards next to
        # them the moment anything squashes it around its own zero.
        frame["Close_SMA200_Ratio"] = (
            pd.to_numeric(frame["Close"], errors="coerce") / sma200.where(sma200 > 0) - 1.0
        )
    return frame


def _category_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """
    The seven evidence scores per bar, each in [-1, 1], positive = bullish.

    Sub-scores are oriented by convention and averaged unweighted. The
    orientations are the textbook ones and are *claims about direction that the
    fit is free to reject*: ``volatility``, for instance, is oriented negative
    for expanding range because equity vol and returns are negatively correlated
    on average, but a symbol where that is untrue gets a coefficient near zero.
    """
    scores: Dict[str, pd.Series] = {}

    # These three are already zero-centred deviations (`close / sma - 1`), so
    # they are squashed as they stand. Subtracting another 1 would put "price at
    # its moving average" at -1 and invert the whole category.
    trend_parts = [
        _bounded(frame["Close_SMA20_Ratio"]),
        _bounded(frame["Close_SMA50_Ratio"]),
        _bounded(frame["SMA20_SMA50_Ratio"]),
        _bounded(frame["Trend_Slope_20"]),
        _bounded(frame["Trend_Slope_60"]),
    ]
    # The 200-day read is the "above the long-term trend" line the panel prints.
    # It is appended only when it exists: a column of NaN would make the whole
    # trend category missing for every bar rather than for the warm-up.
    if "Close_SMA200_Ratio" in frame.columns:
        trend_parts.append(_bounded(frame["Close_SMA200_Ratio"]))
    scores["trend"] = _mean_of(trend_parts)

    scores["momentum"] = _mean_of([
        _centred(frame["RSI_14"], centre=50.0, half_range=50.0),
        _bounded(frame["MACD_Norm"]),
        _bounded(frame["Return_5d"]),
        _bounded(frame["Return_20d"]),
    ])

    scores["volume"] = _mean_of([
        _centred(frame["Up_Volume_Ratio_20"], centre=0.5, half_range=0.5),
        _bounded(frame["OBV_Slope_20"]),
        _bounded(frame["Volume_Price_Confirm"]),
    ])

    # Structure and closing strength: what the candles themselves are doing.
    scores["price_action"] = _mean_of([
        frame[f"PA_Structure_{SHORT_SWING_WINDOW}"],
        frame[f"PA_Structure_{MEDIUM_SWING_WINDOW}"],
        frame[f"PA_Structure_{LONG_SWING_WINDOW}"],
        frame["PA_Close_Strength"],
    ])

    # What happened *at the levels*: breaks, retests, rejections, and how hard
    # the bar reacted where it met one. Kept separate from structure so the two
    # can disagree - a bullish structure rejected at resistance is exactly the
    # case a single blended "technicals" score would hide.
    scores["support_resistance"] = _mean_of([
        frame["PA_Break_Bias"],
        frame["PA_Level_Reaction"],
    ])

    # Expanding range and rising realised vol, oriented negative. Volatility_Ratio
    # and Parkinson_Vol_Ratio *are* true ratios centred on one, unlike the trend
    # columns above, so these do subtract it.
    scores["volatility"] = _mean_of([
        -_bounded(frame["Volatility_Ratio"] - 1.0),
        -_bounded(frame["ATR_Ratio"] - frame["ATR_Ratio"].rolling(SCALE_WINDOW, min_periods=MIN_SCALE_OBSERVATIONS).mean()),
        -_bounded(frame["Parkinson_Vol_Ratio"] - 1.0),
    ])

    return pd.DataFrame(scores, index=frame.index)


@dataclass
class EvidenceStack:
    """Aligned evidence scores, labels and outcomes for one symbol."""

    scores: pd.DataFrame
    labels: pd.Series
    forward_return: pd.Series
    latest_scores: Optional[pd.Series] = None
    latest_as_of: Optional[pd.Timestamp] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.scores)

    @property
    def base_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else float("nan")


def build_evidence_stack(
    frame: pd.DataFrame,
    *,
    horizon: int = 1,
    analog_k: int = DEFAULT_K,
    min_analog_history: int = MIN_ANALOG_HISTORY,
) -> EvidenceStack:
    """
    Turn a feature frame into labelled evidence rows plus the unlabelled latest one.

    The label is the sign of the ``horizon``-bar forward return, exactly as
    :func:`src.features.direction_features.build_direction_dataset` defines it,
    so the stack and the classifier are answering the same question.

    The ``historical_analogs`` column is filled from a causal k-NN pass, whose
    up-rate at row ``t`` reads only bars whose outcome had printed by ``t``.
    """
    close = pd.to_numeric(frame["Close"], errors="coerce")
    forward_return = close.shift(-horizon) / close.replace(0, np.nan) - 1.0

    scores = _category_scores(frame)

    # The analog column needs its descriptor and its outcomes on the same index,
    # so it is computed over the rows where the descriptor is complete.
    descriptor_columns = [col for col in ANALOG_FEATURE_COLUMNS if col in frame.columns]
    descriptor = frame[descriptor_columns].dropna()
    analog_rate = analog_up_rate_series(
        descriptor,
        forward_return.reindex(descriptor.index),
        k=analog_k,
        horizon=horizon,
        min_history=min_analog_history,
    ).reindex(frame.index)
    # Centred on the trailing unconditional up-rate rather than 0.5: an analog
    # up-rate of 0.55 is no evidence at all in a symbol that rises 55% of the
    # time anyway, and this is the column that says so.
    label_series = (forward_return > 0).astype(float).where(forward_return.notna())
    trailing_base = label_series.shift(1).expanding(min_periods=min_analog_history).mean()
    scores["historical_analogs"] = (
        2.0 * (analog_rate - trailing_base)
    ).clip(-1.0, 1.0)

    scores = scores[list(EVIDENCE_CATEGORIES)]

    complete = scores.dropna()
    if complete.empty:
        raise ValueError("No bar has a complete evidence vector")

    # The latest complete row has no resolved outcome yet; it is the one the
    # question is actually about, so it is held out rather than dropped.
    latest_as_of = complete.index[-1]
    labelled_index = complete.index.intersection(forward_return.dropna().index).difference([latest_as_of])
    if len(labelled_index) == 0:
        raise ValueError("No evidence row has a resolved outcome to train on")

    labels = (forward_return.loc[labelled_index] > 0).astype(np.int8)
    labels.name = "direction_up"

    return EvidenceStack(
        scores=complete.loc[labelled_index],
        labels=labels,
        forward_return=forward_return.loc[labelled_index].rename("forward_return"),
        latest_scores=complete.loc[latest_as_of],
        latest_as_of=pd.Timestamp(latest_as_of),
        meta={
            "horizon": int(horizon),
            "n_rows": int(len(labelled_index)),
            "analog_k": int(analog_k),
            "analog_rows": int(analog_rate.notna().sum()),
            "first_date": str(pd.Timestamp(labelled_index[0]).date()),
            "last_date": str(pd.Timestamp(labelled_index[-1]).date()),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fitting, evaluating and reading the stack
# ─────────────────────────────────────────────────────────────────────────────
def _logit(probability: float) -> float:
    p = float(np.clip(probability, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def evaluate_evidence_stack(stack: EvidenceStack, *, seed: int = 42) -> Dict[str, Any]:
    """
    Expanding-window walk-forward over the evidence stack.

    Five contiguous test blocks, each scored by a model fitted only on the rows
    before it. The pooled out-of-sample probabilities are what every downstream
    number leans on: the blend weight, the confidence terciles, and the accuracy
    interval printed beside the answer. Nothing here is fitted on its own test
    rows, and nothing downstream is allowed to use the in-sample probabilities
    in their place.
    """
    n = len(stack)
    if n < MIN_STACK_TRAIN_ROWS + MIN_STACK_TEST_ROWS:
        return {
            "available": False,
            "reason": f"{n} labelled rows, below the "
                      f"{MIN_STACK_TRAIN_ROWS + MIN_STACK_TEST_ROWS}-row floor for a "
                      "walk-forward evaluation",
            "n_test_rows": 0,
            "brier_skill_score": 0.0,
        }

    testable = n - MIN_STACK_TRAIN_ROWS
    fold_size = max(MIN_STACK_TEST_ROWS, testable // STACK_TEST_FOLDS)

    probabilities: List[float] = []
    outcomes: List[int] = []
    reference_rates: List[float] = []
    dates: List[pd.Timestamp] = []
    folds = 0

    start = MIN_STACK_TRAIN_ROWS
    while start < n:
        stop = min(start + fold_size, n)
        if stop - start < MIN_STACK_TEST_ROWS and folds > 0:
            break
        train_y = stack.labels.iloc[:start]
        if train_y.nunique() < 2:
            start = stop
            continue
        model = LogisticDirection(seed=seed)
        model.fit(stack.scores.iloc[:start], train_y)
        fold_probabilities = model.predict_proba_up(stack.scores.iloc[start:stop])
        probabilities.extend(float(value) for value in fold_probabilities)
        outcomes.extend(int(value) for value in stack.labels.iloc[start:stop])
        reference_rates.extend([float(train_y.mean())] * (stop - start))
        dates.extend(stack.scores.index[start:stop])
        folds += 1
        start = stop

    if not probabilities:
        return {
            "available": False,
            "reason": "no walk-forward fold produced predictions",
            "n_test_rows": 0,
            "brier_skill_score": 0.0,
        }

    probability_array = np.asarray(probabilities, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=int)
    correct = int(((probability_array >= 0.5).astype(int) == outcome_array).sum())
    accuracy = correct / outcome_array.size
    low, high = wilson_interval(correct, int(outcome_array.size))
    skill = classifier_skill(outcome_array, probability_array, reference_rates)

    # Terciles of this stack's own out-of-sample conviction. These are what turn
    # |p - 0.5| into a word, and they are the reason a flat stack can never
    # report high confidence.
    conviction = np.abs(probability_array - 0.5)
    return {
        "available": True,
        "n_folds": folds,
        "n_test_rows": int(outcome_array.size),
        "test_range": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-1]).date())],
        "accuracy": round(accuracy, 6),
        "accuracy_ci": [round(low, 6), round(high, 6)],
        "test_base_rate": round(float(outcome_array.mean()), 6),
        "brier_score": skill["model_brier"],
        "brier_skill_score": skill["brier_skill_score"],
        "log_loss_skill_score": skill["log_loss_skill_score"],
        "prediction_std": skill["prediction_std"],
        "conviction_terciles": [
            round(float(np.quantile(conviction, 1 / 3)), 6),
            round(float(np.quantile(conviction, 2 / 3)), 6),
        ],
        "oos_probabilities": probability_array,
    }


def _contributions(
    model: LogisticDirection,
    row: pd.Series,
    categories: Sequence[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Log-odds of the fitted stack for one row, split by category.

    ``intercept + sum(contributions)`` reconstructs the logit exactly, so the
    per-category numbers the panel prints are the actual decomposition of the
    answer rather than a plausible-looking attribution computed some other way.
    """
    if model.model_ is None or model.scaler_ is None:
        return _logit(model.train_base_rate_), {name: 0.0 for name in categories}

    values = row[list(categories)].to_numpy(dtype=float).reshape(1, -1)
    standardised = model.scaler_.transform(values)[0]
    coefficients = np.asarray(model.model_.coef_, dtype=float).reshape(-1)
    intercept = float(np.asarray(model.model_.intercept_, dtype=float).reshape(-1)[0])
    terms = coefficients * standardised
    return intercept + float(terms.sum()), {
        name: float(term) for name, term in zip(categories, terms)
    }


def _horizon_reads(frame: pd.DataFrame) -> Dict[str, Any]:
    """
    A separate directional read at three lookbacks, plus how well they agree.

    Each read is the mean of a structure column, a return column and a trend
    ratio at its own scale. The dead zone that turns a score into UP / DOWN /
    NEUTRAL is this symbol's own trailing standard deviation of that score, so a
    quiet name is not read as directional on a move a volatile one would ignore.
    """
    reads: Dict[str, Any] = {}
    directions: List[str] = []

    for key, description, window, return_column, ratio_column in _HORIZON_SPECS:
        parts: List[pd.Series] = [frame[f"PA_Structure_{window}"]]
        if return_column in frame.columns:
            parts.append(_bounded(frame[return_column]))
        if ratio_column in frame.columns:
            # Deviation columns, already centred on zero - see _category_scores.
            parts.append(_bounded(frame[ratio_column]))
        series = _mean_of(parts).dropna()
        if series.empty:
            reads[key] = {"available": False, "window": window, "description": description}
            continue

        value = float(series.iloc[-1])
        spread = float(series.tail(SCALE_WINDOW).std())
        # Half a standard deviation of its own history: below that the read is
        # not distinguishable from this score's normal wandering.
        dead_zone = 0.5 * spread if np.isfinite(spread) and spread > 0 else 0.0
        direction = "UP" if value > dead_zone else "DOWN" if value < -dead_zone else "NEUTRAL"
        directions.append(direction)
        reads[key] = {
            "available": True,
            "window": int(window),
            "description": description,
            "score": round(value, 6),
            "direction": direction,
            "dead_zone": round(dead_zone, 6),
        }

    decided = [d for d in directions if d != "NEUTRAL"]
    if not decided:
        agreement = "undecided"
    elif len(set(decided)) == 1:
        agreement = "aligned" if len(decided) == len(directions) else "partly aligned"
    else:
        agreement = "conflicting"

    reads["agreement"] = agreement
    reads["directions"] = {
        spec[0]: reads[spec[0]].get("direction") for spec in _HORIZON_SPECS
    }
    return reads


def _confidence(
    probability: float,
    terciles: Optional[Sequence[float]],
    agreement: str,
    total_weight: float,
) -> Dict[str, Any]:
    """
    Turn conviction into a word, against this stack's own historical range.

    Downgraded one step when the three horizon reads conflict, because a
    probability produced by evidence pulling in opposite directions is a fragile
    number however far from 0.5 it lands, and forced to Low when no source has
    measurable skill.
    """
    conviction = abs(float(probability) - 0.5)
    if total_weight <= 0:
        return {
            "label": "Low",
            "score": round(conviction, 6),
            "basis": "no source cleared a measurable out-of-sample edge",
        }
    if not terciles or len(terciles) != 2:
        return {
            "label": "Low",
            "score": round(conviction, 6),
            "basis": "the evidence stack has no walk-forward record to read a conviction range from",
        }

    low_cut, high_cut = float(terciles[0]), float(terciles[1])
    if conviction >= high_cut:
        label, rank = "High", 2
    elif conviction >= low_cut:
        label, rank = "Moderate", 1
    else:
        label, rank = "Low", 0

    basis = (
        f"|p-0.5| = {conviction:.3f} against this model's own out-of-sample "
        f"terciles {low_cut:.3f} / {high_cut:.3f}"
    )
    if agreement == "conflicting" and rank > 0:
        rank -= 1
        label = ["Low", "Moderate", "High"][rank]
        basis += "; downgraded one step because the horizon reads conflict"

    return {"label": label, "score": round(conviction, 6), "basis": basis}


def _evidence_rows(
    stack: EvidenceStack,
    contributions: Dict[str, float],
    logit_total: float,
    price_action: Dict[str, Any],
    analogs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    One row per category: its state, its fitted pull, and what it is reading.

    ``contribution_pp`` is the honest counterfactual - the probability with this
    category's term in the sum minus the probability without it - so the numbers
    describe the answer that was actually given.
    """
    full = _sigmoid(logit_total)
    rows: List[Dict[str, Any]] = []

    for category in EVIDENCE_CATEGORIES:
        score = stack.latest_scores.get(category)
        score = float(score) if score is not None and np.isfinite(score) else None
        term = float(contributions.get(category, 0.0))
        without = _sigmoid(logit_total - term)
        contribution_pp = (full - without) * 100.0

        rows.append({
            "source": category,
            "label": CATEGORY_LABELS[category],
            "score": round(score, 6) if score is not None else None,
            "state": _state_label(category, score),
            "leans": "up" if term > 0 else "down" if term < 0 else "neutral",
            "contribution_pp": round(contribution_pp, 4),
            "detail": _category_detail(category, score, price_action, analogs),
        })

    rows.sort(key=lambda row: abs(row["contribution_pp"]), reverse=True)
    return rows


def _state_label(category: str, score: Optional[float]) -> str:
    """The descriptive state of one category, before any fitted weight."""
    if score is None:
        return "unavailable"
    vocabulary = {
        "trend": ("Bullish", "Bearish", "Flat"),
        "momentum": ("Positive", "Negative", "Flat"),
        "volume": ("Confirming", "Distributing", "Neutral"),
        "price_action": ("Bullish structure", "Bearish structure", "No clear structure"),
        "support_resistance": ("Holding / breaking up", "Rejecting / breaking down", "Mid-range"),
        "volatility": ("Calm", "Expanding", "Normal"),
        "historical_analogs": ("Rose more often", "Fell more often", "No tendency"),
    }
    up, down, flat = vocabulary[category]
    # A tenth of the [-1, 1] range: below that the category is not saying
    # anything a reader should be told a direction about.
    if score > 0.1:
        return up
    if score < -0.1:
        return down
    return flat


def _category_detail(
    category: str,
    score: Optional[float],
    price_action: Dict[str, Any],
    analogs: Dict[str, Any],
) -> Optional[str]:
    """The one concrete fact behind a category, where there is one to give."""
    if category == "price_action" and price_action.get("available"):
        events = price_action.get("events") or []
        structure = price_action.get("structure_label")
        return "; ".join([structure] + events[:2]) if structure else "; ".join(events[:3])
    if category == "support_resistance" and price_action.get("available"):
        levels = price_action.get("levels") or {}
        support, resistance = levels.get("support"), levels.get("resistance")
        if support is not None and resistance is not None:
            # The two percentages are None when the level itself is degenerate,
            # and formatting None with a numeric spec raises. The range is still
            # worth printing without them.
            above = levels.get("support_distance_pct")
            below = levels.get("resistance_distance_pct")
            detail = f"{levels.get('window')}-day range {support:,.2f} - {resistance:,.2f}"
            if above is not None and below is not None:
                detail += f"; {above:+.1f}% above support, {below:.1f}% below resistance"
            return detail
    if category == "historical_analogs" and analogs.get("available"):
        low, high = (analogs.get("up_rate_ci") or [None, None])
        interval = f" (95% CI {low:.0%}-{high:.0%})" if low is not None and high is not None else ""
        return (
            f"{analogs['n_matches']} most-similar past setups rose "
            f"{analogs['up_rate']:.0%} of the time{interval}, against "
            f"{analogs['reference_up_rate']:.0%} for all history"
            if analogs.get("reference_up_rate") is not None
            else f"{analogs['n_matches']} similar setups rose {analogs['up_rate']:.0%} of the time"
        )
    return None


def _reasoning(rows: Sequence[Dict[str, Any]], horizons: Dict[str, Any]) -> List[str]:
    """
    Plain sentences, ordered by how much each one moved the answer.

    Only categories that actually moved it appear. A category sitting at zero
    contribution is not evidence for anything and listing it as a bullet would
    imply it was.
    """
    sentences: List[str] = []
    for row in rows:
        if abs(row["contribution_pp"]) < 0.05 or row["state"] == "unavailable":
            continue
        pushes = "up" if row["contribution_pp"] > 0 else "down"
        # The descriptive state and the fitted pull can disagree, and when they
        # do the sentence has to say so rather than quietly reporting the fitted
        # sign as though the indicator itself had that reading.
        conventional = "up" if (row["score"] or 0) > 0.1 else "down" if (row["score"] or 0) < -0.1 else None
        if conventional is not None and conventional != pushes:
            sentence = (
                f"{row['label']}: {row['state'].lower()} - reads {conventional}, but the weight fitted "
                f"on this symbol's own history pulls the answer {pushes} by {abs(row['contribution_pp']):.1f}pp"
            )
        else:
            sentence = (
                f"{row['label']}: {row['state'].lower()} - pushes the answer {pushes} "
                f"by {abs(row['contribution_pp']):.1f}pp"
            )
        if row.get("detail"):
            sentence += f" ({row['detail']})"
        sentences.append(sentence)

    agreement = horizons.get("agreement")
    if agreement == "aligned":
        sentences.append("Short, medium and long-term reads all point the same way")
    elif agreement == "conflicting":
        sentences.append("Short, medium and long-term reads disagree, which is why confidence is held down")
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# The public entry point
# ─────────────────────────────────────────────────────────────────────────────
def analyse_direction(
    bars: pd.DataFrame,
    *,
    symbol: Optional[str] = None,
    horizon: int = 1,
    seed: int = 42,
    model_probability: Optional[float] = None,
    model_skill: Optional[float] = None,
    model_name: Optional[str] = None,
    model_tradeable: Optional[bool] = None,
    model_gate_reason: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Direction, probability, confidence and the evidence behind them.

    Parameters
    ----------
    bars : DataFrame
        Adjusted daily OHLCV, ideally from
        :func:`src.data.direction_data.load_daily_bars`.
    model_probability, model_skill : float, optional
        The existing classifier's live P(up) and its **measured** out-of-sample
        Brier skill from the walk-forward report. Both or neither: a probability
        with no measured skill beside it cannot be given a blend weight, and
        weighting it by anything other than its measured skill is the guess this
        function exists to avoid. When omitted the answer rests on the evidence
        stack alone and says so.

    Returns
    -------
    dict
        Direction, the two probabilities, the confidence with its basis, one row
        per evidence category with its contribution in percentage points, the
        three horizon reads, the price-action reading, the analog match, and the
        blend that produced the number.
    """
    frame = build_evidence_frame(bars)
    stack = build_evidence_stack(frame, horizon=horizon)

    evaluation = evaluate_evidence_stack(stack, seed=seed)

    # The live fit uses every labelled row. That is correct here and only here:
    # this is the served call, not an evaluation, and the evaluation above is
    # what licenses it.
    model = LogisticDirection(seed=seed)
    model.fit(stack.scores, stack.labels)
    logit_evidence, contributions = _contributions(model, stack.latest_scores, EVIDENCE_CATEGORIES)
    probability_evidence = _sigmoid(logit_evidence)

    price_action = read_price_action(frame, already_built=True)

    descriptor_columns = [col for col in ANALOG_FEATURE_COLUMNS if col in frame.columns]
    descriptor = frame[descriptor_columns].dropna()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    forward_return = close.shift(-horizon) / close.replace(0, np.nan) - 1.0
    labelled_descriptor = descriptor.loc[
        descriptor.index.intersection(forward_return.dropna().index).difference([stack.latest_as_of])
    ]
    analogs: Dict[str, Any]
    if len(labelled_descriptor) and stack.latest_as_of in descriptor.index:
        analogs = analog_matches(
            labelled_descriptor,
            forward_return.reindex(labelled_descriptor.index),
            query=descriptor.loc[stack.latest_as_of],
            horizon=horizon,
        )
    else:
        analogs = {"available": False, "reason": "the current bar has an incomplete setup descriptor", "n_matches": 0}

    # ── Blend ────────────────────────────────────────────────────────────────
    # Weights are measured skill, floored at zero. A source that could not beat
    # a constant base-rate forecast out of sample does not get a vote.
    evidence_weight = max(float(evaluation.get("brier_skill_score") or 0.0), 0.0)
    classifier_weight = (
        max(float(model_skill), 0.0)
        if model_skill is not None and model_probability is not None
        else 0.0
    )
    total_weight = evidence_weight + classifier_weight
    base_rate = stack.base_rate

    if total_weight > 0:
        blended_logit = (
            evidence_weight * _logit(probability_evidence)
            + classifier_weight * _logit(model_probability if model_probability is not None else base_rate)
        ) / total_weight
        probability_up = _sigmoid(blended_logit)
        blend_note = "log-odds average weighted by each source's measured out-of-sample Brier skill"
    else:
        # Neither source demonstrated skill. The honest answer is the base rate,
        # and the honest label for it is NEUTRAL.
        probability_up = float(base_rate) if np.isfinite(base_rate) else 0.5
        blend_note = (
            "no source cleared a positive out-of-sample Brier skill, so the answer "
            "falls back to the historical base rate and is reported as uncertain"
        )

    horizons = _horizon_reads(frame)
    confidence = _confidence(
        probability_up,
        evaluation.get("conviction_terciles"),
        horizons.get("agreement", "undecided"),
        total_weight,
    )

    # Two independent reasons to refuse a direction. The first is relative -
    # today's conviction is in the bottom third of this stack's own range. The
    # second is absolute: if the stack's out-of-sample accuracy interval still
    # covers its base rate *and* the classifier failed its ship criteria, then
    # nothing on the page has demonstrated an edge, and a direction label would
    # be an assertion no measurement supports.
    accuracy_ci = evaluation.get("accuracy_ci") or []
    stack_edge_proven = bool(
        len(accuracy_ci) == 2
        and evaluation.get("test_base_rate") is not None
        and accuracy_ci[0] > float(evaluation["test_base_rate"])
    )
    no_source_proven = not stack_edge_proven and not bool(model_tradeable)
    if confidence["label"] == "Low" or no_source_proven:
        direction = "NEUTRAL"
    else:
        direction = "UP" if probability_up >= 0.5 else "DOWN"

    # Which of the two refusals fired, in words. "No edge anywhere" and "the
    # classifier scored well but failed its ship criteria" are different
    # situations, and collapsing them would tell a user their model found
    # nothing when it in fact found something it is not allowed to trade on.
    if direction != "NEUTRAL":
        neutral_reason = None
    elif no_source_proven and model_gate_reason:
        # The gate reason arrives prefixed for standalone display ("Not
        # tradeable: ..."); inside this sentence the prefix reads as a stutter.
        detail = model_gate_reason.replace("Not tradeable:", "").strip().rstrip(".")
        neutral_reason = (
            "the evidence stack has not beaten its own base rate out of sample, and the "
            f"classifier did not clear its ship criteria — {detail}"
        )
    elif no_source_proven:
        neutral_reason = "no source has shown an out-of-sample edge over the base rate"
    else:
        neutral_reason = (
            "today's conviction is in the bottom third of this model's own historical range"
        )

    rows = _evidence_rows(stack, contributions, logit_evidence, price_action, analogs)

    expected_range = _expected_range(close, stack.latest_as_of, analogs)

    evaluation_public = {key: value for key, value in evaluation.items() if key != "oos_probabilities"}

    return {
        "symbol": symbol,
        "as_of": str(pd.Timestamp(stack.latest_as_of).date()),
        "horizon_days": int(horizon),
        "direction": direction,
        # Why a direction was withheld, when it was. A NEUTRAL with no reason
        # beside it reads as a model with no opinion rather than one that has
        # one and cannot justify acting on it.
        "neutral_reason": neutral_reason,
        "probability_up": round(float(probability_up), 6),
        "probability_down": round(float(1.0 - probability_up), 6),
        "confidence": confidence,
        "base_rate": round(float(base_rate), 6) if np.isfinite(base_rate) else None,
        "last_close": float(close.loc[stack.latest_as_of]) if stack.latest_as_of in close.index else None,
        "evidence": rows,
        # The contributions decompose the evidence stack's own probability, not
        # the blend. Saying so in the payload keeps a panel from adding them up
        # against the headline number and finding they do not reconcile.
        "evidence_note": (
            "contribution_pp decomposes the evidence stack's probability of "
            f"{probability_evidence:.1%}, which is then blended with the classifier"
        ),
        "reasoning": _reasoning(rows, horizons),
        "horizons": horizons,
        "price_action": price_action,
        "historical_analogs": analogs,
        "expected_range": expected_range,
        "blend": {
            "note": blend_note,
            "evidence_stack": {
                "probability_up": round(probability_evidence, 6),
                "weight": round(evidence_weight, 6),
                "brier_skill_score": evaluation.get("brier_skill_score"),
                "accuracy": evaluation.get("accuracy"),
                "accuracy_ci": evaluation.get("accuracy_ci"),
                "n_test_rows": evaluation.get("n_test_rows"),
            },
            "classifier": {
                "model": model_name,
                "probability_up": round(float(model_probability), 6) if model_probability is not None else None,
                "weight": round(classifier_weight, 6),
                "brier_skill_score": model_skill,
                "tradeable": model_tradeable,
                "gate_reason": model_gate_reason,
            },
        },
        "evaluation": evaluation_public,
        "stack": {
            "categories": list(EVIDENCE_CATEGORIES),
            "scores": {
                category: (
                    round(float(stack.latest_scores[category]), 6)
                    if np.isfinite(stack.latest_scores[category]) else None
                )
                for category in EVIDENCE_CATEGORIES
            },
            "n_train_rows": int(len(stack)),
            "train_base_rate": round(float(base_rate), 6) if np.isfinite(base_rate) else None,
            "meta": stack.meta,
        },
    }


def _expected_range(
    close: pd.Series,
    as_of: Optional[pd.Timestamp],
    analogs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    The range the matched historical setups actually produced, priced off today.

    This is deliberately not a forecast path. It is the 10th-to-90th percentile
    of what followed the most similar past setups, applied to the last close -
    a statement about spread, with the sample it came from named, and nothing
    about which day inside the horizon any particular price is reached.
    """
    if not analogs.get("available") or as_of is None or as_of not in close.index:
        return {"available": False, "reason": analogs.get("reason", "no analog sample")}

    last_close = float(close.loc[as_of])
    low = analogs.get("forward_return_p10")
    median = analogs.get("median_forward_return")
    high = analogs.get("forward_return_p90")
    if low is None or high is None:
        return {"available": False, "reason": "analog sample has no return distribution"}

    return {
        "available": True,
        "basis": "10th-90th percentile of the outcomes that followed the matched historical setups",
        "n_samples": analogs.get("n_matches"),
        "quantiles": [0.10, 0.50, 0.90],
        "price_low": round(last_close * (1.0 + float(low)), 4),
        "price_median": round(last_close * (1.0 + float(median or 0.0)), 4),
        "price_high": round(last_close * (1.0 + float(high)), 4),
        "return_low": low,
        "return_median": median,
        "return_high": high,
    }
