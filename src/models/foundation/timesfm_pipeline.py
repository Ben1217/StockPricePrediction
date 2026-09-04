import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TimesFMPipeline:
    """
    Dedicated pipeline for TimesFM 2.5.
    Consumes Close price only. No XReg.
    Output: Point forecast + 9 quantiles.
    Price = 0.5 quantile / median.
    P(up) = 1 - CDF evaluated at last observed close (interpolated).
    """
    def __init__(self, lookback: int = 512):
        self.name = "timesfm"
        self.lookback = lookback
        self.model = None
        # Default quantiles for TimesFM continuous quantile head
        # It typically emits deciles if configured for it, but we can query specific quantiles
        self.quantile_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _ensure_model(self):
        if self.model is None:
            try:
                import timesfm
                self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
                self.model.compile(
                    timesfm.ForecastConfig(
                        max_context=self.lookback, 
                        max_horizon=1, 
                        use_continuous_quantile_head=True
                    )
                )
            except ImportError:
                logger.error("timesfm is not installed.")
                raise
        return self.model

    def predict(self, df: pd.DataFrame, horizon: int = 1) -> Dict[str, Any]:
        """
        Predict future price, probability of up move, and quantiles.
        
        Args:
            df: OHLCV dataframe. Must contain 'Close'.
            horizon: forecast horizon.
            
        Returns:
            Dict containing:
            - 'price': Predicted future price (median)
            - 'p_up': Probability of price going up
            - 'quantiles': Dictionary of quantiles
        """
        if df.empty or 'Close' not in df.columns:
            raise ValueError("Dataframe must contain 'Close'")
            
        context = df["Close"].tail(self.lookback).to_numpy(dtype=np.float64)
        last_close = float(context[-1])
        
        model = self._ensure_model()
        
        # TimesFM forecast API returns (point_forecast, quantile_forecast)
        point_forecast, quantile_forecast = model.forecast(horizon=1, inputs=[context])
        
        # quantile_forecast shape: (batch, horizon, 1 + n_quantiles).
        #
        # TimesFM 2.5 emits one slot per configured quantile plus a LEADING MEAN.
        # `TimesFM_2p5_200M_Definition` sets quantiles=[0.1 ... 0.9] (nine) and
        # decode_index=5, and slot 5 of the ten returned equals the point
        # forecast -- which is only consistent with slot 0 being the mean and
        # slots 1..9 being the deciles. Reading the raw ten as though they were
        # the levels would silently label the mean as q0.1 and shift every
        # subsequent level by one, so the offset is resolved explicitly.
        raw = np.asarray(quantile_forecast)[0, 0, :]
        mean_forecast = None
        if len(raw) == len(self.quantile_levels) + 1:
            mean_forecast = float(raw[0])
            q_vals = raw[1:]
        else:
            q_vals = raw

        if len(q_vals) != len(self.quantile_levels):
            # Previously this fabricated a distribution with
            # np.random.normal(point_forecast, 1.0, 100) and returned it as the
            # model's quantile forecast. That silently fed invented numbers into
            # CRPS, pinball loss, interval coverage and the PIT histogram -- the
            # A6 metrics exist precisely to characterise the model's real
            # distribution, so a synthetic stand-in makes them meaningless
            # rather than approximate. A shape mismatch is a genuine integration
            # failure and must surface as one.
            raise ValueError(
                f"TimesFM returned {len(raw)} values (read as {len(q_vals)} "
                f"quantiles) but {len(self.quantile_levels)} levels are "
                f"configured ({self.quantile_levels}). Align quantile_levels "
                f"with the model's ForecastConfig rather than substituting a "
                f"synthetic distribution."
            )

        # A quantile function must be non-decreasing. Crossing is a real defect
        # worth surfacing rather than silently sorting away.
        if np.any(np.diff(q_vals) < 0):
            logger.warning(
                "TimesFM returned crossing quantiles at %s; the CDF interpolation "
                "below assumes monotonicity", self.quantile_levels
            )

        quantiles = {q: float(v) for q, v in zip(self.quantile_levels, q_vals)}
        median_price = quantiles.get(0.5, float(point_forecast[0, 0]))
        
        # P(up) = 1 - F(last_close), with F interpolated between the quantile
        # knots. Outside the outermost knots the model has told us nothing, so
        # np.interp's flat clipping to [0.1, 0.9] is the honest reading: the
        # true tail mass is unknown, not small.
        #
        # The previous +/-0.05 nudge applied there was an invented tail
        # adjustment with no basis in the model's output. A6.3 exists to MEASURE
        # tail calibration; hand-adjusting the tail corrupts the very quantity
        # being measured. The clamp is reported instead so the caller can mark
        # the row.
        cdf_val = float(np.interp(last_close, q_vals, self.quantile_levels))
        tail_clamped = bool(last_close < q_vals[0] or last_close > q_vals[-1])
        if tail_clamped:
            logger.debug(
                "last close %.4f falls outside the [%.4f, %.4f] quantile range; "
                "P(up) is bounded by the outermost level",
                last_close, float(q_vals[0]), float(q_vals[-1]),
            )

        p_up = 1.0 - cdf_val

        # TimesFM 2.5 is a QUANTILE model: it does not emit sample paths. The
        # previous code returned np.random.normal(...) here as "samples", which
        # any sample-based CRPS would have scored as if it were the model's
        # predictive distribution. Returning None makes the absence explicit so
        # callers use the quantile path (which is what A6.2 specifies for
        # cross-model CRPS comparability).
        return {
            "price": median_price,
            "p_up": p_up,
            "quantiles": quantiles,
            "quantile_levels": list(self.quantile_levels),
            "samples": None,
            "tail_clamped": tail_clamped,
            # The model's mean, which is not its median and is not the point it
            # is scored on. Carried so a caller can see the skew rather than
            # having to infer it; `price` stays the median throughout.
            "mean": mean_forecast,
        }
