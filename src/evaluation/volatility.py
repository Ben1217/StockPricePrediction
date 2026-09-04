"""
Realised-volatility estimators and losses for the volatility arm (A8).

Why this arm exists
-------------------
The signal-to-noise ratio in realised volatility is far better than in signed
returns, and volatility is strongly persistent and clustered -- exactly the
structure a pretrained time-series model can exploit. It is therefore the arm
most likely to produce a *positive* result, which materially de-risks the
project. It is also the only arm that uses High, Low and Open as information
rather than decoration.

A8.1 requires it to ship behind a feature flag, independently switchable from
the direction arm; see :func:`volatility_arm_enabled`. The estimators themselves
are not gated -- they are pure functions and are useful regardless.

Units
-----
Every estimator returns a **per-bar variance** of log price, not a standard
deviation and not an annualised figure. The caller sums over a window to get an
h-bar realised variance and annualises if it wants to.

Why QLIKE is primary
--------------------
MSE on a variance is dominated by a handful of crisis observations: one 2020
bar can outweigh a year of ordinary ones, so an MSE ranking mostly reports who
did best in March 2020. QLIKE is robust to the multiplicative error structure
volatility actually has. Both are reported, and when they disagree the
disagreement is reported too (A8.4) rather than quietly resolved in favour of
whichever flatters the model.

Public API:
    volatility_arm_enabled() -> bool
    parkinson_variance(high, low) -> Series
    garman_klass_variance(open_price, high, low, close) -> Series
    close_to_close_variance(close) -> Series
    realized_variance(df, estimator, window) -> Series
    forward_realized_variance(df, estimator, horizon) -> Series
    qlike_loss(y_true, y_pred) -> Series
    mse_loss(y_true, y_pred) -> Series
    evaluate_volatility(y_true, y_pred) -> dict
    volatility_loss_comparison(y_true, forecasts) -> dict
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Mapping

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: A8.1 -- the volatility arm is switchable independently of the direction arm.
VOLATILITY_ARM_ENABLED_ENV = "QV_VOLATILITY_ARM"


def volatility_arm_enabled() -> bool:
    """Whether the volatility arm is switched on. Defaults to off (A8.1)."""
    return os.environ.get(VOLATILITY_ARM_ENABLED_ENV, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _positive(series: pd.Series, name: str) -> pd.Series:
    """
    Prices must be strictly positive for a log ratio to exist.

    Non-positive bars become NaN and are counted, rather than being clamped to a
    small epsilon -- a clamp would turn a data error into a plausible-looking
    huge variance that no one would notice downstream.
    """
    values = pd.Series(series, dtype="float64")
    bad = int((~(values > 0)).sum())
    if bad:
        logger.warning("%s: %d non-positive values set to NaN", name, bad)
    return values.where(values > 0)


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Parkinson (1980) range estimator: ``(1 / (4 ln 2)) * ln(H/L)^2``.

    Uses the intraday range, so it extracts far more information from one bar
    than a close-to-close square does -- roughly five times more efficient under
    a driftless diffusion.
    """
    high, low = _positive(high, "High"), _positive(low, "Low")
    return (1.0 / (4.0 * np.log(2.0))) * np.log(high / low) ** 2


def garman_klass_variance(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series
) -> pd.Series:
    """
    Garman-Klass (1980): ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``.

    Adds the open-to-close move to the range, which makes it more efficient
    again than Parkinson. It can return small negative values on a bar whose
    close-to-open move exceeds its range implication; those are left as computed
    rather than floored, and :func:`realized_variance` handles them at the
    aggregation step where the sum is what matters.
    """
    open_price, close = _positive(open_price, "Open"), _positive(close, "Close")
    high, low = _positive(high, "High"), _positive(low, "Low")
    return (
        0.5 * np.log(high / low) ** 2
        - (2.0 * np.log(2.0) - 1.0) * np.log(close / open_price) ** 2
    )


def close_to_close_variance(close: pd.Series) -> pd.Series:
    """Close-to-close squared log return -- the simple, noisy reference."""
    close = _positive(close, "Close")
    return np.log(close / close.shift(1)) ** 2


_ESTIMATORS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "parkinson": lambda df: parkinson_variance(df["High"], df["Low"]),
    "garman_klass": lambda df: garman_klass_variance(
        df["Open"], df["High"], df["Low"], df["Close"]
    ),
    "close_to_close": lambda df: close_to_close_variance(df["Close"]),
}


def _per_bar(df: pd.DataFrame, estimator: str) -> pd.Series:
    if estimator not in _ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}; known: {sorted(_ESTIMATORS)}")
    required = {"parkinson": ["High", "Low"],
                "garman_klass": ["Open", "High", "Low", "Close"],
                "close_to_close": ["Close"]}[estimator]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{estimator} needs columns {required}; missing {missing}")
    return _ESTIMATORS[estimator](df)


def realized_variance(
    df: pd.DataFrame, estimator: str = "garman_klass", window: int = 1
) -> pd.Series:
    """Trailing ``window``-bar realised variance: the sum of per-bar variances."""
    per_bar = _per_bar(df, estimator)
    if window <= 1:
        return per_bar
    return per_bar.rolling(int(window), min_periods=int(window)).sum()


def forward_realized_variance(
    df: pd.DataFrame, estimator: str = "garman_klass", horizon: int = 1
) -> pd.Series:
    """
    The **target**: realised variance over the next ``horizon`` bars.

    The value at index ``t`` covers bars ``t+1 .. t+h``, so it is strictly in the
    future of the decision made at ``t`` -- the same alignment the direction arm
    uses. The final ``h`` rows are NaN because their window is not complete;
    they are left NaN rather than partially summed, which would score a model
    against a shorter window than it forecast.
    """
    horizon = max(1, int(horizon))
    per_bar = _per_bar(df, estimator)
    # ``rolling`` looks backward, so a forward window is built by shifting the
    # series one bar into the past, summing the trailing h bars, then shifting
    # the result back by h-1. Rolling the shifted series directly would lose the
    # first h-1 rows to an incomplete window even though their forward windows
    # are perfectly well defined.
    return per_bar.shift(-1).rolling(horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def _clean_pair(y_true: pd.Series, y_pred: pd.Series, *, require_positive: bool):
    true_values = pd.Series(y_true, dtype="float64")
    pred_values = pd.Series(y_pred, dtype="float64").reindex(true_values.index)
    usable = true_values.notna() & pred_values.notna()
    if require_positive:
        usable &= (true_values > 0) & (pred_values > 0)
    return true_values[usable], pred_values[usable], int((~usable).sum())


def qlike_loss(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    QLIKE on variances: ``r/f - ln(r/f) - 1``, which is exactly 0 at ``f == r``.

    Both arguments are **variances** and must be strictly positive. Non-positive
    or non-finite pairs are dropped, not clamped: the previous behaviour floored
    the forecast at 1e-8, which turned a missing forecast into a loss of order
    1e4 and made an absent model look like a catastrophically bad one.

    The uncentred form ``ln f + r/f`` ranks models identically; the centred one
    is used here so a perfect forecast reads as zero.
    """
    true_values, pred_values, _ = _clean_pair(y_true, y_pred, require_positive=True)
    ratio = true_values / pred_values
    return ratio - np.log(ratio) - 1.0


def mse_loss(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Squared error on the variance scale. Secondary to QLIKE (A8.4)."""
    true_values, pred_values, _ = _clean_pair(y_true, y_pred, require_positive=False)
    return (true_values - pred_values) ** 2


def evaluate_volatility(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """Mean QLIKE and MSE for one variance forecast, with the dropped-pair count."""
    _, _, dropped_qlike = _clean_pair(y_true, y_pred, require_positive=True)
    qlike_values = qlike_loss(y_true, y_pred)
    mse_values = mse_loss(y_true, y_pred)

    if dropped_qlike:
        logger.info("volatility scoring: dropped %d pairs that were not strictly positive",
                    dropped_qlike)

    return {
        "qlike": round(float(qlike_values.mean()), 8) if len(qlike_values) else None,
        "mse": round(float(mse_values.mean()), 12) if len(mse_values) else None,
        "n_qlike": int(len(qlike_values)),
        "n_mse": int(len(mse_values)),
        "n_dropped_non_positive": dropped_qlike,
    }


def volatility_loss_comparison(
    y_true: pd.Series, forecasts: Mapping[str, pd.Series]
) -> Dict[str, Any]:
    """
    Rank every model under both losses and say so when the rankings disagree.

    A8.4 requires the disagreement to be *reported*, not resolved. QLIKE and MSE
    genuinely measure different things, and a model that wins on MSE while
    losing on QLIKE is usually one that fits the crisis tail at the expense of
    everything else -- which is a finding, not a tie to be broken.
    """
    if not forecasts:
        raise ValueError("no forecasts to compare")

    scores = {name: evaluate_volatility(y_true, series) for name, series in forecasts.items()}
    scorable = {n: s for n, s in scores.items() if s["qlike"] is not None and s["mse"] is not None}

    if len(scorable) < 2:
        return {
            "scores": scores,
            "qlike_ranking": sorted(scorable, key=lambda n: scorable[n]["qlike"]),
            "mse_ranking": sorted(scorable, key=lambda n: scorable[n]["mse"]),
            "losses_disagree": False,
            "note": "fewer than two models could be scored; no ranking comparison is possible",
        }

    qlike_ranking = sorted(scorable, key=lambda n: scorable[n]["qlike"])
    mse_ranking = sorted(scorable, key=lambda n: scorable[n]["mse"])
    disagree = qlike_ranking != mse_ranking

    return {
        "scores": scores,
        "qlike_ranking": qlike_ranking,
        "mse_ranking": mse_ranking,
        "qlike_best": qlike_ranking[0],
        "mse_best": mse_ranking[0],
        "losses_disagree": disagree,
        "note": (
            "QLIKE and MSE rank these models differently; both rankings are reported "
            "and neither is treated as decisive"
            if disagree
            else "QLIKE and MSE agree on the ranking"
        ),
    }
