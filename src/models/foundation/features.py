"""
The technical-analysis feature set the foundation stack is allowed to see.

Spec v2 Section 4 names five categories -- momentum, volatility, price action,
support/resistance and volume -- and Section 9 forbids anything beyond them.
This module is the single place those five are assembled, so every consumer
sees the identical frame:

    OHLCV
      -> price action  (candle body / shadow geometry, direction runs)
      -> momentum      (RSI)
      -> volatility    (ATR ratio, realised-volatility ratio)
      -> support / resistance (Donchian position, 20-day break flags)
      -> volume        (volume z-score, OBV slope)
      -> model input

It used to live inside ``ChronosPipeline.build_covariates``, which meant the
forecast service could not compute the frame once and hand it to the one member
that consumes it -- the pipeline always rebuilt it. Chronos-2 still owns the
decision to *use* these as past covariates; it no longer owns their definition.

Nothing here reaches the API surface. The Predictions tab shows a price and a
direction; which columns produced them is a backend concern, kept for logs and
diagnostics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from ...features.chart_patterns import add_chart_pattern_features
from ...features.direction_features import (
    DIRECTION_FEATURE_CONFIG,
    add_direction_features,
)
from ...features.feature_engineering import build_regression_feature_frame

#: Spec v2 Section 4, mapped onto the columns that implement each item. Anything
#: not named here is out of scope by Section 9 and must not be passed.
SPEC_V2_COVARIATES: Dict[str, List[str]] = {
    "momentum": ["RSI_14"],
    "volatility": ["ATR_Ratio", "Volatility_Ratio"],
    "price_action": [
        "Body_Ratio",
        "Upper_Shadow_Ratio",
        "Lower_Shadow_Ratio",
        "Consecutive_Direction_Run",
    ],
    "support_resistance": ["Donchian_Position_20", "High_20d_Break", "Low_20d_Break"],
    "volume": ["Volume_Zscore", "OBV_Slope_20"],
}

#: Section 4 items with no exact implementation in the feature layer. Recorded
#: openly instead of being quietly replaced by a near neighbour, because which
#: features the models actually saw is a claim the report has to make precisely.
#:
#:   * Bollinger band width -- ``Volatility_Ratio`` is used in the volatility
#:     slot as the closest available normalised-dispersion measure, but it is not
#:     the same statistic.
#:   * explicit higher-high / higher-low / lower-high / lower-low flags --
#:     ``Consecutive_Direction_Run`` and the Donchian position together carry
#:     related information, but the four discrete flags do not exist as columns.
SPEC_V2_UNIMPLEMENTED: Tuple[str, ...] = (
    "Bollinger band width (volatility category)",
    "higher-high / higher-low / lower-high / lower-low flags (price action category)",
)


def spec_v2_covariate_columns() -> List[str]:
    """The flat, ordered covariate list Section 4 permits."""
    return [column for columns in SPEC_V2_COVARIATES.values() for column in columns]


def build_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Section 4 features, on the same index as the input bars.

    Assembled without the supervised label path, so no row is dropped for want
    of a forward label and the features end on the same bar as the Close series.
    ``build_direction_dataset`` attaches an h-step-ahead label and drops the rows
    whose label cannot resolve; at inference that would leave the features
    ``horizon`` bars short of the series they are meant to accompany.
    """
    # The same three assembly steps build_direction_dataset performs, minus the
    # forward label and the dropna that follows it. Regime detection stays off in
    # DIRECTION_FEATURE_CONFIG because it fits an HMM over the whole series,
    # which would leak future information into every earlier row (Spec v2
    # Requirement 4.1).
    frame = build_regression_feature_frame(df, feature_config=DIRECTION_FEATURE_CONFIG)
    if frame.empty:
        raise ValueError("Feature frame is empty; check the input bars")
    frame = add_direction_features(frame)
    featured = add_chart_pattern_features(frame)

    wanted = spec_v2_covariate_columns()
    missing = [column for column in wanted if column not in featured.columns]
    if missing:
        raise KeyError(
            f"Section 4 covariates missing from the feature layer: {missing}. "
            f"Update SPEC_V2_COVARIATES rather than passing a different set."
        )
    features = featured[wanted].reindex(df.index)
    if len(features) != len(df):
        raise AssertionError(
            f"features ({len(features)}) must align 1:1 with the target "
            f"series ({len(df)})"
        )
    return features
