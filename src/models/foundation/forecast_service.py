"""
The Predictions tab's backend, end to end, in one call.

    OHLCV
      -> price action + momentum + volatility + support/resistance + volume
      -> Kronos + Chronos-2 + TimesFM 2.5
      -> inverse-variance aggregation
      -> one price, one UP/DOWN call, one interval

Every part of that already existed; what did not exist was a single entry point
producing the *one* answer the tab shows. The route used to fan out to a model
per request and leave the frontend to reconcile three payloads, which is how the
pipeline ended up on screen: the UI could not present one forecast because it
was never given one.

Two properties this module holds and the caller must not undo:

**Direction comes from the aggregated probability, never from the price.**
Requirement 5.1 is explicit -- combine ``p_up`` across members and threshold at
0.5. ``sign(forecast - close)`` is a different statistic and is not a substitute
for it, so :attr:`FoundationForecast.split` reports the (real, informative) case
where the two disagree rather than quietly forcing them to agree.

**A point is never served without an interval.** No member exposing a predictive
distribution means :class:`ForecastUnavailable`, not a band of width zero
(Requirement 11.2): zero width asserts a precision none of these models has.

Nothing in :class:`FoundationForecast` is meant to be rendered as an
explanation. ``weights``, ``method``, ``features_built`` and ``members_failed``
exist for logs, tests and an admin view. The tab shows a price and an arrow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .aggregator import FoundationAggregator
from .features import SPEC_V2_COVARIATES, build_technical_features

logger = logging.getLogger(__name__)

#: The three members, in the order the tab names them. Keys are the model types
#: the rest of the API already speaks; values are how a human writes them.
FOUNDATION_MEMBERS: Dict[str, str] = {
    "unified_kronos": "Kronos",
    "unified_chronos": "Chronos-2",
    "unified_timesfm": "TimesFM 2.5",
}

#: The quantile levels the served interval is built from.
#:
#: [q0.05, q0.95] spans 90% of the mass and [q0.16, q0.84] spans 68%, so the
#: bounds are named for those coverages and not for the levels that bracket
#: them -- a q0.95 upper bound paired with a q0.05 lower one is a 90% interval,
#: and calling it 95% overstates the coverage by five points.
#:
#: A genuine 95% interval is not derivable from this member set: TimesFM 2.5
#: exposes deciles only, so q0.025 and q0.975 fall outside its outermost knots
#: and would be clipped back to q0.1 and q0.9 -- a "95%" band no wider than the
#: model's deciles, and silently so. 90% is the widest honest claim here.
INTERVAL_LEVELS = (0.05, 0.16, 0.84, 0.95)


class ForecastUnavailable(Exception):
    """No servable forecast, with the reason a user can be shown."""

    def __init__(self, message: str, members_failed: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.message = message
        self.members_failed = members_failed or {}


@dataclass(frozen=True)
class FoundationForecast:
    """One aggregated next-bar forecast. The whole answer, and nothing else."""

    price: float
    p_up: float
    direction: str  # "UP" | "DOWN"
    #: The close ``p_up`` and ``direction`` were measured against: the last bar
    #: the members read. Served because a caller showing a live quote as
    #: "current price" is holding a DIFFERENT number, and the expected move has
    #: to be reported against the one the models actually used.
    anchor_price: float
    lower_90: float
    lower_68: float
    upper_68: float
    upper_90: float
    #: Display names of the members that actually produced this number.
    members_used: List[str]
    #: Model type -> why it did not contribute. Diagnostics, not UI copy.
    members_failed: Dict[str, str] = field(default_factory=dict)
    #: Aggregation weights by model type, and which rule produced them.
    weights: Dict[str, float] = field(default_factory=dict)
    method: str = "inverse_variance"
    #: True when the aggregated probability and the aggregated price point in
    #: opposite directions *against the same anchor*. Rare, real, and worth one
    #: line rather than a panel.
    #:
    #: NOT the far commoner case of a live quote having moved away from
    #: ``anchor_price`` since the last bar printed. Only the caller holds a
    #: quote, so only the caller can detect that -- and conflating the two is
    #: what made a -0.26% forecast read as +3.8% beside a DOWN arrow.
    split: bool = False
    #: How many TA feature columns were assembled, by Section 4 category.
    features_built: Dict[str, int] = field(default_factory=dict)


def _member_quantiles(result: Dict[str, Any], levels) -> Optional[np.ndarray]:
    """
    One member's predictive quantiles at ``levels``, or None if it has none.

    Prefers raw sample paths (Kronos, Chronos-2) and falls back to interpolating
    the member's own quantile knots -- TimesFM 2.5 is a quantile model and emits
    no samples at all.
    """
    samples = result.get("samples")
    if samples is not None:
        array = np.asarray(samples, dtype=np.float64).reshape(-1)
        array = array[np.isfinite(array)]
        if array.size:
            return np.quantile(array, levels)

    quantiles = result.get("quantiles")
    if quantiles:
        knots = sorted((float(q), float(v)) for q, v in quantiles.items() if v is not None)
        if len(knots) >= 2:
            qs = np.array([knot for knot, _ in knots], dtype=np.float64)
            vs = np.array([value for _, value in knots], dtype=np.float64)
            # np.interp clips outside the knot range, which is the honest
            # reading: beyond the outermost quantile the model has said nothing.
            return np.interp(levels, qs, vs)
        if len(knots) == 1:
            return np.full(len(levels), knots[0][1], dtype=np.float64)
    return None


def _aggregate_interval(
    predictions: Dict[str, Dict[str, Any]],
    weights: Dict[str, float],
    levels=INTERVAL_LEVELS,
) -> Optional[np.ndarray]:
    """
    Quantile averaging across members, on the weights that produced the point.

    The interval and the point therefore describe the same combined
    distribution. Returns None when no member exposed one.
    """
    rows, row_weights = [], []
    for name, result in predictions.items():
        quantiles = _member_quantiles(result, levels)
        if quantiles is None:
            continue
        rows.append(quantiles)
        row_weights.append(float(weights.get(name, 0.0)))

    if not rows:
        return None
    weight_array = np.asarray(row_weights, dtype=np.float64)
    if weight_array.sum() <= 0:
        weight_array = np.ones_like(weight_array)
    weight_array = weight_array / weight_array.sum()
    return np.average(np.vstack(rows), axis=0, weights=weight_array)


def run_foundation_forecast(
    df: pd.DataFrame,
    *,
    pipeline_factory: Callable[[str], Any],
    symbol: str = "",
) -> FoundationForecast:
    """
    Run the three foundation models over ``df`` and return the single forecast.

    ``pipeline_factory`` maps a model type to a loaded pipeline (or None), so the
    route keeps ownership of the process-wide model cache -- these are multi-
    hundred-megabyte transformers and must not be reloaded per request.

    Everything here is measured against the last close, and there is no hook to
    override that. Each member computes ``p_up`` internally as P(next bar > last
    close) -- Kronos over its sample paths, Chronos-2 and TimesFM 2.5 through
    their quantile CDFs -- so an anchor handed in from outside would move
    :attr:`FoundationForecast.split` onto a price the probability was never
    computed against: it reports a disagreement that is not happening and hides
    the one that is. This function used to take a ``reference_price`` for
    exactly that, and its callers passed a live quote.

    A caller that shows a quote rather than the last close is therefore holding
    two prices, not one. :attr:`FoundationForecast.anchor_price` is served so it
    can keep them apart.

    The horizon is the next bar, and only the next bar. All three members are
    built for one step: TimesFM compiles with ``max_horizon=1``, Kronos samples a
    single chunk, and Chronos-2's sample reshape assumes one prediction length.
    Serving a longer horizon from here would mean reading numbers those calls do
    not produce.
    """
    features = build_technical_features(df)
    features_built = {
        category: sum(1 for column in columns if column in features.columns)
        for category, columns in SPEC_V2_COVARIATES.items()
    }
    logger.debug(
        "%s: assembled %s TA feature columns for the foundation stack (%s)",
        symbol or "forecast", features.shape[1], features_built,
    )

    predictions: Dict[str, Dict[str, Any]] = {}
    members_failed: Dict[str, str] = {}
    for model_type in FOUNDATION_MEMBERS:
        pipeline = pipeline_factory(model_type)
        if pipeline is None:
            members_failed[model_type] = "pipeline not available"
            continue
        try:
            # Chronos-2 is the only member with a covariate channel, so it is the
            # only one handed the frame. Kronos reads the OHLCV candles directly
            # and TimesFM 2.5 is univariate by construction (Spec v2 Section 2.4)
            # -- passing features to either would be a claim about their input
            # that is not true.
            if model_type == "unified_chronos":
                predictions[model_type] = pipeline.predict(df, horizon=1, covariates=features)
            else:
                predictions[model_type] = pipeline.predict(df, horizon=1)
        except Exception as exc:  # one member failing must not lose the other two
            logger.warning("%s: %s did not produce a forecast: %s", symbol, model_type, exc)
            members_failed[model_type] = str(exc)

    if not predictions:
        raise ForecastUnavailable(
            "None of the forecast models could run for this symbol.",
            members_failed,
        )

    aggregated = FoundationAggregator.aggregate(predictions)
    # The aggregator drops members with no usable spread; they contributed to
    # nothing, so they are reported as excluded rather than counted as used.
    for name, reason in (aggregated.get("models_excluded") or {}).items():
        members_failed.setdefault(name, reason)

    weights = aggregated.get("weights", {}) or {}
    bounds = _aggregate_interval(predictions, weights)
    if bounds is None:
        raise ForecastUnavailable(
            "The models produced no prediction interval, so no forecast is served.",
            members_failed,
        )
    lower_90, lower_68, upper_68, upper_90 = (float(bound) for bound in bounds)

    price = float(aggregated["price"])
    p_up = float(aggregated["p_up"])
    # Requirement 5.1: threshold the aggregated probability. Never sign(price - close).
    direction = "UP" if p_up > 0.5 else "DOWN"
    # The close every member's p_up was computed against, and therefore the only
    # price on which a disagreement between the two heads means anything.
    anchor = float(df["Close"].iloc[-1])
    split = (price >= anchor) != (direction == "UP")

    return FoundationForecast(
        price=price,
        p_up=p_up,
        direction=direction,
        anchor_price=anchor,
        lower_90=lower_90,
        lower_68=lower_68,
        upper_68=upper_68,
        upper_90=upper_90,
        # In FOUNDATION_MEMBERS order, which is the order the tab names them.
        # Sorting instead put Chronos-2 first on the strength of its dict key.
        members_used=[
            label for name, label in FOUNDATION_MEMBERS.items() if name in weights
        ],
        members_failed=members_failed,
        weights=weights,
        method=aggregated.get("method", "inverse_variance"),
        split=split,
        features_built=features_built,
    )
