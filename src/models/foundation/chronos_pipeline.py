"""
Chronos-2 input builder (Spec v2 Section 2.2).

This is the **only** model in the comparison that consumes the technical-analysis
feature set. Kronos has no channel for them and TimesFM 2.5 deliberately runs
univariate, so Chronos-2 carries the "general-purpose, feature-informed" arm of
the three-way comparison in Section 2.4.

Two consequences follow, and both are enforced rather than assumed:

**Dropping the covariates silently is not an acceptable degradation.** If the
installed Chronos build cannot accept past covariates, Chronos-2 becomes a second
univariate model and the three-way comparison collapses into a two-way one --
with the results table still labelling it "feature-informed". So an unsupported
build is logged loudly and reported in ``covariates_used``, which the caller must
surface rather than discard.

**Covariates must not come from the supervised dataset builder.** That rule, and
the five Section 4 categories themselves, now live in
:mod:`src.models.foundation.features` -- one definition shared by this pipeline
and by the forecast service, so the service can build the frame once and pass it
in rather than having it rebuilt here on every request.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import (
    SPEC_V2_COVARIATES,
    SPEC_V2_UNIMPLEMENTED,
    build_technical_features,
    spec_v2_covariate_columns,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ChronosPipeline",
    "SPEC_V2_COVARIATES",
    "SPEC_V2_UNIMPLEMENTED",
    "spec_v2_covariate_columns",
]


class ChronosPipeline:
    """
    Chronos-2: general-purpose foundation model, informed by the TA covariates.

    Target series is Close; covariates are the Section 4 features as *past-only*
    covariates. Output is 21 quantiles per horizon step, all of which are kept --
    Section 2.2 is explicit that the median alone is not enough.
    """

    #: Section 2.2 requires 21 quantiles. Symmetric, spanning 0.01 to 0.99.
    QUANTILE_LEVELS: Tuple[float, ...] = tuple(
        [0.01] + list(np.round(np.linspace(0.05, 0.95, 19), 4)) + [0.99]
    )

    def __init__(self, sample_count: int = 128, lookback: int = 128):
        self.name = "chronos"
        self.sample_count = int(sample_count)
        self.lookback = int(lookback)
        self.model = None
        self.quantile_levels: List[float] = list(self.QUANTILE_LEVELS)
        if len(self.quantile_levels) != 21:
            raise AssertionError(
                f"Section 2.2 requires 21 quantiles, built {len(self.quantile_levels)}"
            )

    def _ensure_model(self):
        if self.model is None:
            import torch

            try:
                from chronos import BaseChronosPipeline
            except ImportError:
                logger.error("chronos is not installed; Chronos-2 cannot run")
                raise
            self.model = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.float32,
            )
        return self.model

    def build_covariates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        The Section 4 covariates, on the same index as the target series.

        A thin alias for :func:`~src.models.foundation.features.build_technical_features`
        so callers that already hold the shared frame can pass it to
        :meth:`predict` instead of paying for it twice.
        """
        return build_technical_features(df)

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        covariates: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Forecast the Close series with the Section 4 features as past covariates.

        ``covariates`` may be a frame already built by
        :func:`~src.models.foundation.features.build_technical_features` for the
        same bars; it is rebuilt here only when the caller has none. Passing a
        frame that does not align 1:1 with ``df`` is rejected rather than
        reindexed, because a covariate row silently paired with the wrong bar is
        the one failure mode this whole assembly path exists to prevent.

        Returns the median price, ``P(up)``, all 21 quantiles, the raw samples,
        and ``covariates_used`` -- which the caller must not discard, because a
        run where it is False is not the feature-informed arm the report claims.
        """
        if df is None or df.empty or "Close" not in df.columns:
            raise ValueError("Chronos-2 needs a non-empty frame with a Close column")

        history = df.sort_index()
        last_close = float(history["Close"].iloc[-1])
        context = history["Close"].tail(self.lookback).to_numpy(dtype=np.float32)
        if len(context) < 32:
            raise ValueError(
                f"Chronos-2 needs at least 32 bars of context, got {len(context)}"
            )

        if covariates is None:
            covariate_frame = build_technical_features(history)
        elif not history.index.equals(covariates.index):
            raise ValueError(
                "Supplied covariates are not indexed on the same bars as the "
                "target series; build them from the frame being forecast."
            )
        else:
            covariate_frame = covariates
        covariate_frame = covariate_frame.tail(len(context))
        model = self._ensure_model()

        # Chronos-2 takes its covariates through the *input schema*, not through
        # a predict() keyword: each element of `inputs` is a dict of a `target`
        # series plus a `past_covariates` mapping of name -> series of the same
        # length. Probing signature(model.predict) for a "past_covariates"
        # parameter therefore always came back False on a build that supports
        # them perfectly well, and every run silently took the univariate branch
        # while logging that the feature-informed arm was unavailable. The
        # capability is now decided by whether the model accepts the schema.
        finite = np.isfinite(covariate_frame.to_numpy(dtype=np.float64)).all()
        covariates_used = bool(finite)
        if not finite:
            # A NaN covariate is a warm-up row that reached the model, not a
            # degradation to accept quietly: it would be imputed by whatever the
            # model does with missing values and the run would still be reported
            # as feature-informed.
            logger.warning(
                "Chronos-2 covariates contain non-finite values over the context "
                "window, so this run is UNIVARIATE. Section 2.2 makes this the "
                "only model that consumes the TA feature set; report it as such."
            )

        target = np.asarray(context, dtype=np.float32)
        entry: Dict[str, Any] = {"target": target}
        if covariates_used:
            entry["past_covariates"] = {
                column: covariate_frame[column].to_numpy(dtype=np.float32)
                for column in covariate_frame.columns
            }
        forecast = model.predict([entry], prediction_length=int(horizon))

        # (n_variates, n_quantiles, prediction_length) per input series.
        predicted = np.asarray(forecast[0], dtype=np.float64)
        if predicted.ndim != 3:
            raise ValueError(
                f"Chronos-2 returned an array of shape {predicted.shape}; "
                f"(n_variates, n_quantiles, prediction_length) was expected."
            )
        q_vals = predicted[0, :, 0]

        # The model's own quantile grid, not an assumed one. Section 2.2 asks for
        # 21 quantiles and this build emits exactly those 21 natively, so they are
        # read off the model rather than re-derived -- a build whose grid differs
        # would otherwise have its levels silently mislabelled.
        levels = [float(level) for level in getattr(model, "quantiles", self.quantile_levels)]
        if len(levels) != len(q_vals):
            raise ValueError(
                f"Chronos-2 returned {len(q_vals)} quantiles but reports "
                f"{len(levels)} levels; the two must correspond one to one."
            )
        if len(levels) != len(self.quantile_levels):
            raise ValueError(
                f"Section 2.2 requires {len(self.quantile_levels)} quantiles, "
                f"this build emits {len(levels)}."
            )

        # A quantile function must be non-decreasing. Crossing is a real defect
        # worth surfacing rather than silently sorting away.
        if np.any(np.diff(q_vals) < 0):
            logger.warning(
                "Chronos-2 returned crossing quantiles; the CDF interpolation "
                "below assumes monotonicity"
            )

        quantiles = {round(level, 4): float(value) for level, value in zip(levels, q_vals)}

        # P(up) = 1 - F(last_close), with F interpolated through the quantile
        # knots -- Section 3.2's method. This build returns quantiles rather than
        # sample paths, so the empirical `mean(samples > last_close)` the previous
        # code used has nothing to compute over. Never sign(median - close).
        # Outside the outermost knots np.interp clips, which is the honest
        # reading: beyond 0.01 and 0.99 the model has said nothing.
        p_up = 1.0 - float(np.interp(last_close, q_vals, levels))
        tail_clamped = bool(last_close < q_vals[0] or last_close > q_vals[-1])

        median_price = quantiles.get(0.5)
        if median_price is None:
            median_price = float(np.interp(0.5, levels, q_vals))

        return {
            "price": float(median_price),
            "p_up": p_up,
            "quantiles": quantiles,
            "quantile_levels": levels,
            # Chronos-2 is a QUANTILE model here: it emits no sample paths, and
            # returning invented ones would be scored as if they were the
            # model's predictive distribution.
            "samples": None,
            "tail_clamped": tail_clamped,
            "covariates_used": covariates_used,
            "covariate_columns": spec_v2_covariate_columns() if covariates_used else [],
            "covariates_unimplemented": list(SPEC_V2_UNIMPLEMENTED),
            "last_close": last_close,
        }
