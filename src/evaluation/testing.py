"""
Statistical significance testing (Requirement A5).

Without these tests the results section is a table of numbers with no
inferential claim. With them it is a study. A model "beating" a baseline by a
raw margin, with no test, no HAC correction and no FDR adjustment, is not a
finding and must not be phrased as one (A5.4).

Sign convention, which every caller depends on
----------------------------------------------
Loss differentials are always ``d_t = L_model(t) - L_baseline(t)``. A
**negative** mean differential therefore means the model is **better**. This is
stated on every function that produces or consumes one, because getting it
backwards inverts the conclusion of the entire chapter.

Why the long-run variance is not the sample variance
----------------------------------------------------
For a horizon ``h > 1`` consecutive h-step returns overlap by ``h - 1`` bars, so
their loss differentials are autocorrelated *by construction*. The naive
standard error is then far too small and the test over-rejects. A4.3 makes a
HAC/Newey-West estimator with a lag of at least ``h - 1`` mandatory, and this
module defaults to exactly that rather than leaving it to the caller.

Why both a normal and a t p-value are reported
-----------------------------------------------
The asymptotic DM statistic over-rejects in small samples, which is the regime
a few hundred test days sits in. The Harvey-Leybourne-Newbold correction
rescales the statistic and refers it to a t distribution; that is the one to
quote in the report. Both are returned so the difference is visible.

Public API:
    newey_west_variance(d, lag) -> float | None
    diebold_mariano_test(loss_model, loss_baseline, horizon, lag) -> dict
    mcnemar_test(y_true, y_pred1, y_pred2, exact) -> dict
    bh_fdr_control(p_values, alpha) -> (reject, adjusted)
    benjamini_hochberg(p_values, alpha) -> dict
    describe_family(n_tests, family) -> str
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)


def newey_west_variance(d: Sequence[float], lag: Optional[int] = None) -> Optional[float]:
    """
    HAC long-run variance of the mean of ``d``, with Bartlett kernel weights.

        LRV = gamma_0 + 2 * sum_{k=1..L} (1 - k/(L+1)) * gamma_k

    where ``gamma_k = (1/n) sum_t (d_t - dbar)(d_{t-k} - dbar)``.

    Returns **None** when the estimate is non-positive. A Bartlett-weighted sum
    can come out negative on short samples, and square-rooting it silently would
    manufacture a NaN test statistic that reads as "no result" rather than
    "the variance estimate failed".

    ``lag=None`` uses the Newey-West automatic rule ``floor(4 (n/100)^(2/9))``.
    Callers testing an h-step forecast must pass ``h - 1`` or more (A4.3).
    """
    values = np.asarray(d, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = values.size
    if n < 2:
        return None

    max_lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lag is None else int(lag)
    max_lag = int(np.clip(max_lag, 0, n - 1))

    centred = values - values.mean()
    variance = float(np.dot(centred, centred) / n)
    for k in range(1, max_lag + 1):
        gamma_k = float(np.dot(centred[k:], centred[:-k]) / n)
        variance += 2.0 * (1.0 - k / (max_lag + 1.0)) * gamma_k

    return variance if variance > 0 else None


def diebold_mariano_test(
    loss_model: Sequence[float],
    loss_baseline: Sequence[float],
    horizon: int = 1,
    lag: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Diebold-Mariano test of equal expected loss (A5.1).

    ``d = loss_model - loss_baseline``, so ``mean_differential < 0`` means the
    model beat the baseline. The default lag is ``horizon - 1``, the minimum
    A4.3 permits for overlapping returns.

    Every unavailable result carries a ``reason``. Returning a bare NaN p-value
    would be indistinguishable from a test that ran and found nothing.
    """
    model = np.asarray(loss_model, dtype=np.float64).reshape(-1)
    baseline = np.asarray(loss_baseline, dtype=np.float64).reshape(-1)
    if model.size != baseline.size:
        raise ValueError(f"length mismatch: model={model.size}, baseline={baseline.size}")

    horizon = max(1, int(horizon))
    usable = np.isfinite(model) & np.isfinite(baseline)
    n_dropped = int(np.sum(~usable))
    d = model[usable] - baseline[usable]
    n = d.size

    result: Dict[str, Any] = {
        "n": n,
        "n_dropped": n_dropped,
        "horizon": horizon,
        "available": False,
        "reason": None,
        "mean_differential": None,
        "sign": None,
        "lag_used": None,
        "long_run_variance": None,
        "dm_stat": None,
        "p_value": None,
        "dm_stat_hln": None,
        "p_value_hln": None,
    }

    if n < 3:
        result["reason"] = f"only {n} usable paired observations; need at least 3"
        return result

    mean_differential = float(np.mean(d))
    result["mean_differential"] = round(mean_differential, 10)
    result["sign"] = (
        "model_better" if mean_differential < 0
        else "baseline_better" if mean_differential > 0
        else "tie"
    )

    lag_used = horizon - 1 if lag is None else int(lag)
    lag_used = int(np.clip(lag_used, 0, n - 1))
    result["lag_used"] = lag_used

    long_run_variance = newey_west_variance(d, lag=lag_used)
    if long_run_variance is None:
        result["reason"] = (
            "the HAC long-run variance estimate was not positive; the test cannot be formed"
        )
        return result
    result["long_run_variance"] = round(long_run_variance, 12)

    dm_stat = mean_differential / np.sqrt(long_run_variance / n)
    if not np.isfinite(dm_stat):
        result["reason"] = "the DM statistic was not finite"
        return result

    # Harvey-Leybourne-Newbold small-sample correction, referred to t(n-1).
    hln_factor = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    if not np.isfinite(hln_factor) or hln_factor <= 0:
        result["reason"] = (
            f"the HLN correction is undefined for n={n}, h={horizon} "
            "(the horizon is too long relative to the sample)"
        )
        return result

    dm_hln = dm_stat * hln_factor
    result.update(
        {
            "available": True,
            "dm_stat": round(float(dm_stat), 6),
            "p_value": round(float(2.0 * stats.norm.sf(abs(dm_stat))), 8),
            "dm_stat_hln": round(float(dm_hln), 6),
            "p_value_hln": round(float(2.0 * stats.t.sf(abs(dm_hln), df=n - 1)), 8),
            "hln_factor": round(float(hln_factor), 6),
        }
    )
    return result


def mcnemar_test(
    y_true: Sequence,
    y_pred1: Sequence,
    y_pred2: Sequence,
    exact: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    McNemar's paired test for directional classification (A5.2).

    The two arms see the identical bars, so the comparison must be paired; an
    unpaired two-proportion test on the same data overstates the uncertainty and
    will fail to detect a real difference.

    Only the discordant pairs carry information:

        b = model 1 right, model 2 wrong
        c = model 1 wrong, model 2 right

    ``exact=None`` picks the exact binomial test when ``b + c < 25`` and the
    continuity-corrected chi-square otherwise, which is the usual rule. Both
    p-values are returned regardless.

    When there are no discordant pairs the test has no information at all, so
    ``available`` is False. Reporting ``p = 1`` there would imply the test ran
    and found the arms equivalent.
    """
    truth = np.asarray(y_true).reshape(-1)
    first = np.asarray(y_pred1).reshape(-1)
    second = np.asarray(y_pred2).reshape(-1)
    if not (truth.size == first.size == second.size):
        raise ValueError(
            f"length mismatch: y_true={truth.size}, y_pred1={first.size}, y_pred2={second.size}"
        )

    correct1 = truth == first
    correct2 = truth == second
    b = int(np.sum(correct1 & ~correct2))
    c = int(np.sum(~correct1 & correct2))
    discordant = b + c

    result: Dict[str, Any] = {
        "n": int(truth.size),
        "b_model1_only_correct": b,
        "c_model2_only_correct": c,
        "n_discordant": discordant,
        "accuracy_model1": round(float(np.mean(correct1)), 6),
        "accuracy_model2": round(float(np.mean(correct2)), 6),
        "accuracy_difference": round(float(np.mean(correct1) - np.mean(correct2)), 6),
        "available": False,
        "reason": None,
        "statistic": None,
        "p_value": None,
        "p_value_exact": None,
        "p_value_chi2": None,
        "test_used": None,
    }

    if discordant == 0:
        result["reason"] = (
            "no discordant pairs: the two arms were right and wrong on exactly the "
            "same bars, so the test carries no information"
        )
        return result

    p_exact = float(stats.binomtest(min(b, c), n=discordant, p=0.5).pvalue)
    chi2 = (abs(b - c) - 1.0) ** 2 / discordant if discordant > 0 else np.nan
    p_chi2 = float(stats.chi2.sf(chi2, df=1)) if np.isfinite(chi2) else np.nan

    use_exact = (discordant < 25) if exact is None else bool(exact)
    result.update(
        {
            "available": True,
            "statistic": round(float(chi2), 6),
            "p_value_exact": round(p_exact, 8),
            "p_value_chi2": round(p_chi2, 8) if np.isfinite(p_chi2) else None,
            "p_value": round(p_exact if use_exact else p_chi2, 8),
            "test_used": "exact_binomial" if use_exact else "chi2_continuity_corrected",
        }
    )
    return result


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Benjamini-Hochberg FDR control across the whole family of tests (A5.3).

    The grid of {models} x {horizons} x {tickers} x {metrics} will produce false
    positives with near certainty if left unadjusted, so the adjustment runs
    across the full family and both raw and adjusted p-values are reported.

    NaN p-values -- tests that could not be formed -- are **excluded from the
    family size**, not replaced by 1.0. Padding the family with unavailable
    tests inflates ``m`` and makes the adjustment needlessly conservative: with
    two real tests among five slots, a raw p of 0.01 was being reported as 0.05
    instead of 0.02.

    The adjusted values enforce monotonicity by a cumulative minimum taken from
    the largest p downwards. The naive ``m * p / i`` is *not* the BH adjusted
    p-value and can be non-monotone.
    """
    raw = np.asarray(p_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(raw)
    m = int(np.sum(finite))

    adjusted = np.full(raw.shape, np.nan, dtype=np.float64)
    rejected = np.zeros(raw.shape, dtype=bool)

    if m == 0:
        logger.warning("Benjamini-Hochberg: no finite p-values in the family")
        return {
            "raw_p": raw.tolist(),
            "adjusted_p": adjusted.tolist(),
            "rejected": rejected.tolist(),
            "alpha": float(alpha),
            "family_size": 0,
            "n_excluded": int(raw.size),
            "n_rejected": 0,
        }

    positions = np.nonzero(finite)[0]
    order = positions[np.argsort(raw[positions], kind="mergesort")]
    ranks = np.arange(1, m + 1, dtype=np.float64)

    scaled = m * raw[order] / ranks
    # Step-up: enforce monotonicity from the largest p downwards, then clip.
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted[order] = np.clip(monotone, 0.0, 1.0)
    rejected[order] = adjusted[order] <= float(alpha)

    return {
        "raw_p": raw.tolist(),
        "adjusted_p": adjusted.tolist(),
        "rejected": rejected.tolist(),
        "alpha": float(alpha),
        "family_size": m,
        "n_excluded": int(raw.size - m),
        "n_rejected": int(np.sum(rejected)),
    }


def bh_fdr_control(
    p_values: Sequence[float], alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Array form of :func:`benjamini_hochberg`: ``(reject, adjusted_p)``.

    Kept for callers that only want the two arrays. NaN inputs come back as NaN
    adjusted p-values and are not rejected.
    """
    report = benjamini_hochberg(p_values, alpha=alpha)
    return np.asarray(report["rejected"], dtype=bool), np.asarray(
        report["adjusted_p"], dtype=np.float64
    )


def describe_family(n_tests: int, family: str) -> str:
    """
    The plain-English family statement A5.3 requires under the results table.

    A BH adjustment is only interpretable next to the definition of the family
    it was computed over, so this string is written into the artifact rather
    than left for the reader to reconstruct.
    """
    return (
        f"Benjamini-Hochberg FDR control was applied across the family "
        f"'{family}', comprising {int(n_tests)} tests. Both raw and adjusted "
        f"p-values are reported; a claim of significance refers to the adjusted "
        f"value."
    )
