"""
The mandatory baseline suite (Requirement A2).

These are not a footnote. The null hypothesis of the whole module is the
martingale ``E[P(t+h) | F(t)] = P(t)``, and a foundation model that does not
beat the random walk has produced no forecasting skill however good its charts
look. So the baselines are scored through the identical code path as the
models, on identical windows, and land in the identical results schema.

Probabilities, not labels
-------------------------
Every baseline that has a direction emits a *probability*, never a hard 0/1.
A 0/1 "probability" makes Brier, log-loss and ROC-AUC meaningless -- log-loss
is infinite the first time a 1.0 call is wrong, and AUC degenerates because the
scores have no ordering within a class. The drift and AR(1) baselines therefore
convert their point forecast to ``P(up) = Phi(mu_h / sigma_h)`` using their own
predictive standard deviation, and the always-up / always-down rules are
clipped just inside the open interval so the scoring code stays finite.

Always-up is non-negotiable
---------------------------
It ignores its input entirely, and on a rising sample it will often match a
model's headline accuracy. That is the base-rate artifact A3 exists to expose,
which is why the rule is here rather than in a footnote.

Public API:
    random_walk_forecast(prices, horizon) -> DataFrame
    random_walk_drift_forecast(returns, horizon, lookback, min_observations) -> DataFrame
    always_directional_forecast(index, direction) -> DataFrame
    ar1_log_return_forecast(train_returns, test_returns, horizon, min_train) -> DataFrame
    har_rv_forecast(train_rv, test_rv, horizon) -> DataFrame
    base_rate(direction_or_returns) -> float
    climatology_p_up(pi, n, index) -> Series
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Probabilities are clipped into this open interval. A hard 0.0 or 1.0 makes
#: log-loss infinite, which would let one wrong call from a constant rule
#: dominate every aggregate it appears in.
_EPS = 1e-6

#: Below this many observations a rolling drift estimate is noise, so the row
#: falls back to the uninformative 0.5 and is counted as degenerate rather than
#: quietly reported as a forecast.
_MIN_DRIFT_OBSERVATIONS = 20


def _clip_probability(values) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), _EPS, 1.0 - _EPS)


def _normal_p_up(point: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    P(r_h > 0) under a Gaussian predictive distribution.

    Rows with a non-finite or non-positive sigma have no usable predictive
    spread, so they fall back to 0.5 -- the honest statement that the baseline
    has nothing to say about that row, not a forecast of a coin flip.
    """
    point = np.asarray(point, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    usable = np.isfinite(point) & np.isfinite(sigma) & (sigma > 0)
    probabilities = np.full(point.shape, 0.5, dtype=np.float64)
    probabilities[usable] = norm.cdf(point[usable] / sigma[usable])
    return _clip_probability(probabilities)


def random_walk_forecast(prices: pd.Series, horizon: int) -> pd.DataFrame:
    """
    The primary null: tomorrow's best forecast is today's close.

    Price forecast = last observed close, return forecast = 0, P(up) = 0.5.
    This is *the* benchmark every other row in the results table is measured
    against.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    prices = pd.Series(prices, dtype="float64")

    return pd.DataFrame(
        {
            "pred_price": prices.to_numpy(dtype=np.float64),
            "pred_return": np.zeros(len(prices), dtype=np.float64),
            "p_up": np.full(len(prices), 0.5, dtype=np.float64),
            "sigma": np.full(len(prices), np.nan, dtype=np.float64),
        },
        index=prices.index,
    )


def random_walk_drift_forecast(
    returns: pd.Series,
    horizon: int,
    lookback: int = 252,
    min_observations: int = _MIN_DRIFT_OBSERVATIONS,
) -> pd.DataFrame:
    """
    Random walk with drift: does the apparent skill just capture a trend?

    The drift is the mean one-bar log return over a **trailing** window, scaled
    to the horizon, and the predictive spread is the trailing standard deviation
    scaled by ``sqrt(horizon)`` under i.i.d. accumulation. Rolling and as-of
    throughout -- a full-sample mean here would be look-ahead of the crudest
    kind, since it would know the sample's overall direction in advance.

    ``degenerate`` counts rows that fell back to P(up) = 0.5 for want of enough
    history; it is returned as a column so the count survives into the report.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    returns = pd.Series(returns, dtype="float64")
    floor = max(2, int(min_observations))
    window = returns.rolling(window=lookback, min_periods=floor)

    drift_per_bar = window.mean()
    sigma_per_bar = window.std(ddof=1)

    pred_return = drift_per_bar * horizon
    sigma_h = sigma_per_bar * np.sqrt(horizon)
    p_up = _normal_p_up(pred_return.to_numpy(), sigma_h.to_numpy())

    degenerate = int(np.sum(~np.isfinite(sigma_h.to_numpy()) | (sigma_h.to_numpy() <= 0)))
    if degenerate:
        logger.info(
            "drift baseline: %d/%d rows had no usable predictive spread and fell back to 0.5",
            degenerate,
            len(returns),
        )

    return pd.DataFrame(
        {
            "pred_return": pred_return.to_numpy(dtype=np.float64),
            "p_up": p_up,
            "sigma": sigma_h.to_numpy(dtype=np.float64),
            "degenerate": np.full(len(returns), degenerate, dtype=np.int64),
        },
        index=returns.index,
    )


def always_directional_forecast(index: pd.Index, direction: str) -> pd.DataFrame:
    """
    The base-rate artifact (always-up) and its symmetry check (always-down).

    It ignores its input entirely -- only the index length is read -- which is
    exactly the point: on a rising window this rule can match a foundation
    model's headline accuracy without using any information at all.

    P(up) is clipped to within ``_EPS`` of the bound rather than set to a hard
    1.0 / 0.0, so log-loss stays finite when the rule is wrong.
    """
    normalised = str(direction).upper()
    if normalised not in ("UP", "DOWN"):
        raise ValueError(f"direction must be UP or DOWN, got {direction!r}")

    p_up = 1.0 - _EPS if normalised == "UP" else _EPS
    return pd.DataFrame(
        {
            "p_up": np.full(len(index), p_up, dtype=np.float64),
            "pred_return": np.full(len(index), np.nan, dtype=np.float64),
            "sigma": np.full(len(index), np.nan, dtype=np.float64),
        },
        index=index,
    )


def ar1_log_return_forecast(
    train_returns: pd.Series,
    test_returns: pd.Series,
    horizon: int,
    min_train: int = 60,
) -> pd.DataFrame:
    """
    AR(1) on one-bar log returns, refit per fold: is there weak linear serial
    dependence the foundation models should at minimum match?

    The target is the **cumulative** h-period log return, so the forecast is the
    sum of the iterated one-step forecasts, not the level forecast of
    ``r_{t+h}``. For an AR(1) with ``mu = c / (1 - phi)``:

        E[ sum_{i=1..h} r_{t+i} | r_t ] = h*mu + (r_t - mu) * phi * (1 - phi^h) / (1 - phi)

    Forecasting ``r_{t+h}`` alone instead would score a different quantity from
    the one the evaluator realises, which is a silent target mismatch rather
    than a visible error.

    The h-step forecast-error variance follows from the MA representation: the
    cumulative error is ``sum_{k=1..h} e_{t+k} * (1 - phi^(h-k+1)) / (1 - phi)``,
    so its variance is ``sigma_e^2 * sum_{k=1..h} ((1 - phi^k) / (1 - phi))^2``.

    When the fit is non-stationary (``|phi| >= 1``) or the fold is too short,
    the baseline falls back to the random walk and says so in ``fell_back`` --
    it does not quietly emit zeros as if they were a fitted forecast.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    train_clean = (
        pd.Series(train_returns, dtype="float64")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    test_returns = pd.Series(test_returns, dtype="float64")

    def _fallback(reason: str) -> pd.DataFrame:
        logger.info("AR(1) baseline fell back to the random walk: %s", reason)
        return pd.DataFrame(
            {
                "pred_return": np.zeros(len(test_returns), dtype=np.float64),
                "p_up": np.full(len(test_returns), 0.5, dtype=np.float64),
                "sigma": np.full(len(test_returns), np.nan, dtype=np.float64),
                "fell_back": np.ones(len(test_returns), dtype=bool),
            },
            index=test_returns.index,
        )

    if len(train_clean) < max(2, int(min_train)):
        return _fallback(f"only {len(train_clean)} usable training rows, need {min_train}")

    # OLS of r_i on a constant and r_{i-1}. numpy.linalg.lstsq keeps the
    # arithmetic visible; the model is two parameters and needs no more.
    y = train_clean.to_numpy(dtype=np.float64)[1:]
    x = train_clean.to_numpy(dtype=np.float64)[:-1]
    design = np.column_stack([np.ones_like(x), x])
    try:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError as exc:  # noqa: BLE001 - a singular fold must not kill the run
        return _fallback(f"least squares failed: {exc}")

    constant, phi = float(coefficients[0]), float(coefficients[1])
    if not np.isfinite(phi) or abs(phi) >= 1.0:
        return _fallback(f"non-stationary estimate phi={phi:.4f}")

    residuals = y - design @ coefficients
    # Two parameters estimated, so the residual variance loses two degrees of freedom.
    dof = max(1, len(residuals) - 2)
    sigma_e = float(np.sqrt(np.sum(residuals**2) / dof))

    mu = constant / (1.0 - phi)
    # The lagged input for a decision at test bar t is the return realised AT t,
    # which is known at that bar's close. The last training return seeds the
    # first test row so no test row is dropped.
    lagged = np.concatenate(
        ([train_clean.to_numpy(dtype=np.float64)[-1]], test_returns.to_numpy(dtype=np.float64)[:-1])
    )

    persistence = phi * (1.0 - phi**horizon) / (1.0 - phi)
    pred_return = horizon * mu + (lagged - mu) * persistence

    k = np.arange(1, horizon + 1, dtype=np.float64)
    variance_factor = float(np.sum(((1.0 - phi**k) / (1.0 - phi)) ** 2))
    sigma_h = np.full(len(test_returns), sigma_e * np.sqrt(variance_factor), dtype=np.float64)

    return pd.DataFrame(
        {
            "pred_return": pred_return,
            "p_up": _normal_p_up(pred_return, sigma_h),
            "sigma": sigma_h,
            "fell_back": np.zeros(len(test_returns), dtype=bool),
        },
        index=test_returns.index,
    )


def har_rv_forecast(train_rv: pd.Series, test_rv: pd.Series, horizon: int = 1) -> pd.DataFrame:
    """
    Corsi's HAR-RV: the standard econometric benchmark for the volatility arm.

    Realised variance is regressed on its own daily, weekly (5-bar) and monthly
    (22-bar) trailing averages. Beating a random walk on volatility is trivial
    and proves nothing; this is the model the volatility literature is actually
    judged against (A8.3).

    Returns **NaN** rather than 0.0 when the fold is too short to fit. A zero
    variance forecast is not a conservative default: QLIKE divides by the
    forecast, so a zero would produce an astronomically large loss and make a
    missing model look like a catastrophically bad one.
    """
    train_rv = pd.Series(train_rv, dtype="float64")
    test_rv = pd.Series(test_rv, dtype="float64")

    def har_features(series: pd.Series) -> pd.DataFrame:
        # min_periods matches each window so a partial window is NaN rather than
        # a mean over three bars presented as a monthly component.
        return pd.DataFrame(
            {
                "rv_daily": series,
                "rv_weekly": series.rolling(5, min_periods=5).mean(),
                "rv_monthly": series.rolling(22, min_periods=22).mean(),
            }
        )

    # Features are lagged one bar, so the row at t predicts the target at t.
    train_features = har_features(train_rv).shift(1)
    target = train_rv.reindex(train_features.index)
    usable = train_features.notna().all(axis=1) & target.notna()
    train_features, target = train_features[usable], target[usable]

    nan_result = pd.DataFrame(
        {"pred_rv": np.full(len(test_rv), np.nan, dtype=np.float64)}, index=test_rv.index
    )
    if len(train_features) < 30:
        logger.warning(
            "HAR-RV: only %d usable training rows (need 30); returning NaN forecasts",
            len(train_features),
        )
        return nan_result

    design = np.column_stack(
        [np.ones(len(train_features)), train_features.to_numpy(dtype=np.float64)]
    )
    try:
        coefficients, *_ = np.linalg.lstsq(design, target.to_numpy(dtype=np.float64), rcond=None)
    except np.linalg.LinAlgError as exc:  # noqa: BLE001 - a singular fold must not kill the run
        logger.warning("HAR-RV least squares failed: %s", exc)
        return nan_result

    # The first test rows need trailing history that lives in the training fold,
    # so the windows are built on the concatenation and then sliced back.
    combined = pd.concat([train_rv.iloc[-22:], test_rv])
    test_features = har_features(combined).shift(1).reindex(test_rv.index)
    test_design = np.column_stack(
        [np.ones(len(test_features)), test_features.to_numpy(dtype=np.float64)]
    )
    predictions = test_design @ coefficients

    # Realised variance cannot be negative; an OLS fit can extrapolate below
    # zero, and a negative variance forecast is not a forecast.
    negative = int(np.sum(predictions <= 0))
    if negative:
        logger.warning("HAR-RV produced %d non-positive variance forecasts; set to NaN", negative)
        predictions = np.where(predictions > 0, predictions, np.nan)

    return pd.DataFrame({"pred_rv": predictions}, index=test_rv.index)


def base_rate(values) -> float:
    """
    Share of up moves -- the number every accuracy figure must be read against.

    Accepts either 0/1 direction labels or raw returns (ties, ``r == 0``, are
    counted as **down**, matching the label convention used throughout).
    """
    array = np.asarray(pd.Series(values, dtype="float64").dropna(), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute a base rate from an empty series")
    labels = array if np.all(np.isin(array, (0.0, 1.0))) else (array > 0).astype(np.float64)
    return float(np.mean(labels))


def climatology_p_up(pi: float, n: int, index: Optional[pd.Index] = None) -> pd.Series:
    """
    The constant forecast ``P(up) = pi`` that the Brier Skill Score is measured
    against (A3.2). Its Brier score is exactly ``pi * (1 - pi)``.
    """
    value = float(np.clip(pi, _EPS, 1.0 - _EPS))
    return pd.Series(np.full(int(n), value, dtype=np.float64), index=index)
