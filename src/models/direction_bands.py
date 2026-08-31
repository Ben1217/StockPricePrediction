"""
Per-day price bands for models that only emit a probability.

Kronos gets a price range for free: it samples whole candles, so the percentiles
of the sampled closes *are* the band. The tabular slots (logistic, XGBoost,
TabPFN) emit a single number, P(up), and a probability is not a price. This
module turns one into the other without inventing anything.

The construction
----------------
Fit, on the training fold only:

1. Standardise each realised next-day return by the volatility known at the time
   it was predicted::

       z_t = r_t / sigma_t          sigma_t = trailing 20-day return std at t

   Standardising is what makes the band adapt to regime. A +-2% range is wide in
   a calm month and narrow in a panic; ``z`` removes that, so quantiles fitted
   across a whole training window remain meaningful in both.

2. Bucket the training rows by the model's own predicted probability, and take
   the empirical quantiles of ``z`` inside each bucket.

Predict, for a row with probability ``p`` and trailing volatility ``sigma_t``::

       band = close_t * (1 + sigma_t * quantile_of_bucket(p))

So the width comes from today's volatility and the *skew* comes from the model's
own conviction — measured, not assumed. If the model's high-probability days
genuinely resolved higher in training, the upper band shifts up; if they did
not, the buckets collapse onto each other and the band is honestly symmetric.
A model with no signal produces a band that is simply the unconditional return
distribution scaled to today's volatility, which is the correct answer for a
model with no signal.

Why not a normal distribution
-----------------------------
Fitting ``mu +- 1.645 sigma`` would assume daily returns are Gaussian. They are
not — they are fat-tailed and left-skewed, so a Gaussian 90% interval
under-covers on exactly the days anyone cares about. Empirical quantiles of
``z`` inherit the real shape, including the fat left tail.

Leakage
-------
``sigma_t`` is a trailing window and ``fit()`` sees only training rows. The
quantiles are fitted on the training fold's realised returns and applied
unchanged to the test fold, like any other fitted parameter.

Public API:
    ConditionalReturnBand(...)  — fit / predict
    band_metrics(...)           — coverage, width, pinball loss
    BAND_QUANTILES
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Lower, median, upper. Matches the percentiles Kronos reports, so the two
# band sources are directly comparable in the report.
BAND_QUANTILES: tuple[float, ...] = (0.05, 0.50, 0.95)

# Probability buckets. Three is deliberate: at ~1000 training rows, more buckets
# means fewer rows each, and a 5th-percentile estimated from 60 observations is
# noise wearing a decimal point.
DEFAULT_N_BUCKETS = 3

# A bucket needs at least this many training rows to get its own quantiles;
# below it the bucket falls back to the pooled distribution.
MIN_BUCKET_ROWS = 60

# Floor on volatility, so a freak run of identical closes cannot produce a
# zero-width band.
MIN_VOLATILITY = 1e-6


@dataclass
class ConditionalReturnBand:
    """
    Empirical, volatility-scaled, probability-conditioned return quantiles.

    Parameters
    ----------
    n_buckets : int
        Number of probability buckets. Bucket edges are training-set quantiles
        of the predicted probability, so buckets are equally populated whatever
        range a given model's probabilities happen to occupy — logistic output
        spanning 0.47-0.53 buckets as usefully as Kronos output spanning
        0.2-0.8.
    quantiles : sequence of float
        Quantiles of the standardised return to report.
    """

    n_buckets: int = DEFAULT_N_BUCKETS
    quantiles: Sequence[float] = BAND_QUANTILES
    fitted_: bool = False
    edges_: Optional[np.ndarray] = None
    bucket_quantiles_: Optional[np.ndarray] = None
    pooled_quantiles_: Optional[np.ndarray] = None
    info_: Dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        probabilities: Sequence[float],
        forward_return: Sequence[float],
        volatility: Sequence[float],
    ) -> "ConditionalReturnBand":
        """Learn the conditional quantiles of the volatility-standardised return."""
        p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        r = np.asarray(forward_return, dtype=np.float64).reshape(-1)
        sigma = np.asarray(volatility, dtype=np.float64).reshape(-1)
        if not (p.size == r.size == sigma.size):
            raise ValueError(
                f"length mismatch: probabilities={p.size}, forward_return={r.size}, "
                f"volatility={sigma.size}"
            )

        usable = np.isfinite(p) & np.isfinite(r) & np.isfinite(sigma) & (sigma > MIN_VOLATILITY)
        if usable.sum() < MIN_BUCKET_ROWS:
            raise ValueError(
                f"only {int(usable.sum())} usable training rows for the price band; "
                f"need at least {MIN_BUCKET_ROWS}"
            )

        z = r[usable] / sigma[usable]
        p_usable = p[usable]
        quantiles = np.asarray(self.quantiles, dtype=np.float64)

        self.pooled_quantiles_ = np.quantile(z, quantiles)

        # Equally populated buckets, so every bucket has rows to estimate from.
        interior = np.quantile(p_usable, np.linspace(0, 1, self.n_buckets + 1)[1:-1])
        self.edges_ = np.unique(interior)

        assignments = np.digitize(p_usable, self.edges_, right=False)
        n_buckets = len(self.edges_) + 1
        bucket_quantiles = np.empty((n_buckets, quantiles.size), dtype=np.float64)
        counts: List[int] = []
        for bucket in range(n_buckets):
            mask = assignments == bucket
            count = int(mask.sum())
            counts.append(count)
            if count >= MIN_BUCKET_ROWS:
                bucket_quantiles[bucket] = np.quantile(z[mask], quantiles)
            else:
                # Not enough rows to say anything bucket-specific; the pooled
                # distribution is the honest fallback.
                bucket_quantiles[bucket] = self.pooled_quantiles_

        self.bucket_quantiles_ = bucket_quantiles
        self.fitted_ = True
        self.info_ = {
            "n_train_rows": int(usable.sum()),
            "n_buckets": n_buckets,
            "bucket_edges": [round(float(e), 6) for e in self.edges_],
            "bucket_counts": counts,
            "quantiles": [float(q) for q in quantiles],
            "pooled_z_quantiles": [round(float(v), 6) for v in self.pooled_quantiles_],
            "bucket_z_quantiles": [[round(float(v), 6) for v in row] for row in bucket_quantiles],
        }
        return self

    def predict(
        self,
        probabilities: Sequence[float],
        last_close: Sequence[float],
        volatility: Sequence[float],
    ) -> np.ndarray:
        """
        Price band for the next close, one row per input.

        Returns an ``(n, len(quantiles))`` array in price units, ordered low to
        high. Rows whose volatility is missing come back as NaN rather than
        silently borrowing another row's width.
        """
        if not self.fitted_:
            raise RuntimeError("ConditionalReturnBand must be fitted before predicting")

        p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
        close = np.asarray(last_close, dtype=np.float64).reshape(-1)
        sigma = np.asarray(volatility, dtype=np.float64).reshape(-1)
        if not (p.size == close.size == sigma.size):
            raise ValueError(
                f"length mismatch: probabilities={p.size}, last_close={close.size}, "
                f"volatility={sigma.size}"
            )

        assert self.bucket_quantiles_ is not None and self.edges_ is not None
        assignments = np.digitize(p, self.edges_, right=False)
        assignments = np.clip(assignments, 0, self.bucket_quantiles_.shape[0] - 1)

        z_quantiles = self.bucket_quantiles_[assignments]          # (n, n_quantiles)
        bands = close[:, None] * (1.0 + sigma[:, None] * z_quantiles)

        invalid = ~(np.isfinite(close) & np.isfinite(sigma) & (sigma > MIN_VOLATILITY))
        bands[invalid] = np.nan
        # Quantiles are monotone by construction, but floating-point noise on a
        # near-zero sigma could invert them; sorting makes the contract explicit.
        return np.sort(bands, axis=1)


def band_metrics(
    bands: np.ndarray,
    actual_close: Sequence[float],
    *,
    quantiles: Sequence[float] = BAND_QUANTILES,
) -> Dict[str, Any]:
    """
    Score a price band: does it cover, how wide is it, and is it well placed?

    * ``coverage`` — share of actual closes inside the outer interval. For a 5/95
      band this should be 0.90. Below that the band is too narrow and its
      confidence is fictional; far above it, the band is too wide to be useful.
    * ``mean_relative_width`` — outer width as a fraction of the actual close.
      A band can always reach 100% coverage by being enormous, so width is the
      other half of the score.
    * ``pinball_loss`` — the proper scoring rule for quantiles, penalising a
      quantile that sits on the wrong side of the outcome in proportion to how
      far and how extreme the quantile is. Lower is better, and it is the number
      to compare two band methods on, because it cannot be gamed by widening.
    """
    bands = np.asarray(bands, dtype=np.float64)
    actual = np.asarray(actual_close, dtype=np.float64).reshape(-1)
    quantiles = np.asarray(quantiles, dtype=np.float64)

    if bands.ndim != 2 or bands.shape[0] != actual.size:
        raise ValueError(f"bands {bands.shape} does not align with {actual.size} actual closes")
    if bands.shape[1] != quantiles.size:
        raise ValueError(f"bands has {bands.shape[1]} columns but {quantiles.size} quantiles given")

    usable = np.isfinite(bands).all(axis=1) & np.isfinite(actual)
    n = int(usable.sum())
    if n == 0:
        return {"n": 0, "coverage": None, "mean_relative_width": None, "pinball_loss": None}

    valid_bands = bands[usable]
    valid_actual = actual[usable]

    lower, upper = valid_bands[:, 0], valid_bands[:, -1]
    inside = (valid_actual >= lower) & (valid_actual <= upper)
    nominal = float(quantiles[-1] - quantiles[0])

    losses = []
    for column, q in enumerate(quantiles):
        error = valid_actual - valid_bands[:, column]
        losses.append(np.mean(np.maximum(q * error, (q - 1.0) * error)))

    median_column = int(np.argmin(np.abs(quantiles - 0.5)))
    median_band = valid_bands[:, median_column]

    return {
        "n": n,
        "nominal_coverage": round(nominal, 4),
        "coverage": round(float(inside.mean()), 6),
        "coverage_gap": round(float(inside.mean() - nominal), 6),
        "mean_relative_width": round(float(np.mean((upper - lower) / valid_actual)), 6),
        "median_relative_width": round(float(np.median((upper - lower) / valid_actual)), 6),
        "pinball_loss": round(float(np.mean(losses)), 6),
        "pinball_loss_per_quantile": [round(float(v), 6) for v in losses],
        # A median that is systematically above or below the outcome is a biased
        # centre, which coverage alone would not reveal.
        "median_bias_relative": round(
            float(np.mean((median_band - valid_actual) / valid_actual)), 6
        ),
    }


def volatility_for(dataset_features: pd.DataFrame, ohlcv: Optional[pd.DataFrame] = None) -> pd.Series:
    """
    Trailing return volatility for band scaling.

    Prefers the ``Volatility`` feature column (20-day standard deviation of daily
    returns, already trailing and already leakage-checked). Falls back to
    recomputing it from ``ohlcv`` when a caller has supplied a feature set that
    omits it.
    """
    if "Volatility" in dataset_features.columns:
        return pd.to_numeric(dataset_features["Volatility"], errors="coerce")
    if ohlcv is not None and "Close" in ohlcv.columns:
        logger.info("Volatility column absent; recomputing from OHLCV for the price band")
        returns = pd.to_numeric(ohlcv["Close"], errors="coerce").pct_change()
        return returns.rolling(20).std().reindex(dataset_features.index)
    raise KeyError(
        "Price bands need a 'Volatility' feature column or an OHLCV frame to derive one from"
    )
