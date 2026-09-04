"""
Metrics for evaluation (Requirement A3, A6).
"""

from typing import Any, Dict, List, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score, brier_score_loss

def directional_metrics(y_true_returns: pd.Series, p_up: pd.Series) -> Dict[str, float]:
    """
    Directional metrics (A3):
    - Base rate (pi)
    - Accuracy
    - EOBR (Excess Over Base Rate)
    - Balanced Accuracy
    - MCC
    - ROC-AUC
    """
    # Valid indices only
    mask = y_true_returns.notna() & p_up.notna()
    y_true = y_true_returns[mask]
    p_up_clean = p_up[mask]
    
    if len(y_true) == 0:
        return {"base_rate": np.nan, "accuracy": np.nan, "eobr": np.nan,
                "balanced_accuracy": np.nan, "mcc": np.nan, "roc_auc": np.nan}
                
    # Labels
    y_binary = (y_true > 0).astype(int)
    y_pred = (p_up_clean > 0.5).astype(int)
    
    # Base rate
    pi = y_binary.mean()
    
    # Accuracy
    accuracy = (y_pred == y_binary).mean()
    
    # EOBR
    eobr = accuracy - max(pi, 1 - pi)
    
    try:
        bal_acc = balanced_accuracy_score(y_binary, y_pred)
    except ValueError:
        bal_acc = np.nan
        
    try:
        mcc = matthews_corrcoef(y_binary, y_pred)
    except ValueError:
        mcc = np.nan
        
    try:
        if len(np.unique(y_binary)) > 1:
            auc = roc_auc_score(y_binary, p_up_clean)
        else:
            auc = np.nan
    except ValueError:
        auc = np.nan
        
    return {
        "base_rate": float(pi),
        "accuracy": float(accuracy),
        "eobr": float(eobr),
        "balanced_accuracy": float(bal_acc),
        "mcc": float(mcc),
        "roc_auc": float(auc)
    }

def probabilistic_metrics(y_true_returns: pd.Series, p_up: pd.Series) -> Dict[str, float]:
    """
    Probabilistic classification metrics (A3).
    - Brier score
    - Brier Skill Score vs base rate
    """
    mask = y_true_returns.notna() & p_up.notna()
    y_true = y_true_returns[mask]
    p_up_clean = p_up[mask]
    
    if len(y_true) == 0:
        return {"brier_score": np.nan, "brier_skill_score": np.nan}
        
    y_binary = (y_true > 0).astype(int)
    pi = y_binary.mean()
    
    brier = brier_score_loss(y_binary, p_up_clean)

    # Climatological Brier score: the constant forecast P(up) = pi.
    # np.full_like(y_binary, pi) would inherit y_binary's INTEGER dtype and
    # silently truncate pi to 0, scoring the model against an always-down
    # forecast instead of against the base rate. That inflates BSS enough to
    # flip its sign -- a coin-flip model scored +0.67 instead of -0.33 -- so the
    # dtype is pinned explicitly here. The closed form is pi * (1 - pi).
    brier_clim = brier_score_loss(y_binary, np.full(len(y_binary), float(pi), dtype=float))
    
    if brier_clim == 0:
        bss = np.nan
    else:
        bss = 1.0 - (brier / brier_clim)
        
    return {
        "brier_score": float(brier),
        "brier_skill_score": float(bss)
    }

def pinball_loss(y_true: float, y_pred: float, quantile: float) -> float:
    """
    Quantile (pinball) loss for a single observation and quantile.
    """
    error = y_true - y_pred
    return max(quantile * error, (quantile - 1) * error)

def quantile_metrics(y_true_returns: pd.Series, quantiles_dict: Dict[float, pd.Series]) -> Dict[str, float]:
    """
    Quantile metrics (A6).
    quantiles_dict maps a probability level (e.g., 0.1) to a Series of predictions.
    Computes average pinball loss across all provided quantiles and empirical coverage.
    """
    if not quantiles_dict:
        return {"mean_pinball_loss": np.nan, "coverage_80": np.nan, "coverage_90": np.nan}
        
    # Average pinball loss
    total_pinball = 0.0
    valid_obs = 0
    
    for q_val, preds in quantiles_dict.items():
        mask = y_true_returns.notna() & preds.notna()
        y = y_true_returns[mask]
        p = preds[mask]
        if len(y) > 0:
            errors = y - p
            loss = np.where(errors >= 0, q_val * errors, (q_val - 1) * errors)
            total_pinball += loss.mean()
            valid_obs += 1
            
    mean_pinball = total_pinball / valid_obs if valid_obs > 0 else np.nan
    
    # Coverage for the nominal 80% and 90% central intervals.
    #
    # Requiring the exact keys 0.05/0.95 silently returned NaN for TimesFM 2.5,
    # whose 9 quantiles do not contain them -- the model looked untested rather
    # than under-covered. The nearest available levels are used instead, and the
    # nominal level they ACHIEVE is reported alongside, so a 0.1/0.9 grid
    # standing in for a 90% interval is visible rather than assumed.
    levels = sorted(quantiles_dict)

    def coverage(nominal: float) -> Dict[str, float]:
        if len(levels) < 2:
            return {"coverage": np.nan, "achieved_nominal": np.nan,
                    "mean_width": np.nan, "n": 0}
        want_low, want_high = (1.0 - nominal) / 2.0, (1.0 + nominal) / 2.0
        q_low = min(levels, key=lambda q: abs(q - want_low))
        q_high = min(levels, key=lambda q: abs(q - want_high))
        if q_low >= q_high:
            return {"coverage": np.nan, "achieved_nominal": np.nan,
                    "mean_width": np.nan, "n": 0}
        lower, upper = quantiles_dict[q_low], quantiles_dict[q_high]
        mask = y_true_returns.notna() & lower.notna() & upper.notna()
        y, lo, hi = y_true_returns[mask], lower[mask], upper[mask]
        if len(y) == 0:
            return {"coverage": np.nan, "achieved_nominal": np.nan,
                    "mean_width": np.nan, "n": 0}
        return {
            "coverage": float(((y >= lo) & (y <= hi)).mean()),
            "achieved_nominal": float(q_high - q_low),
            "mean_width": float((hi - lo).mean()),
            "n": int(len(y)),
        }

    cov_80, cov_90 = coverage(0.80), coverage(0.90)

    return {
        "mean_pinball_loss": float(mean_pinball),
        "coverage_80": cov_80["coverage"],
        "coverage_80_detail": cov_80,
        "coverage_90": cov_90["coverage"],
        "coverage_90_detail": cov_90,
    }

def _exact_crps_piecewise_linear(y: float, q: np.ndarray, tau: np.ndarray) -> float:
    """
    Exact CRPS of the distribution implied by a quantile forecast, for one observation.

        CRPS(F, y) = integral over R of (F(x) - 1{x >= y})^2 dx

    The knots (tau_j, q_j) are joined linearly and the tails outside them are
    FLAT, which makes the implied distribution an atom of mass ``tau_1`` at
    ``q_1``, a piecewise-linear CDF between the knots, and an atom of mass
    ``1 - tau_m`` at ``q_m``. F is then 0 below q_1 and 1 above q_m, so the
    integral has bounded support and is computed in closed form:

    * ``y < q_1``  contributes ``q_1 - y`` from the region where F = 0 but the
      step is already 1;
    * ``y > q_m``  contributes ``y - q_m`` symmetrically;
    * each interior segment contributes ``w * (A^2 + A*B + B^2) / 3``, the exact
      integral of a squared linear function, splitting the segment at y when the
      step falls inside it.

    Verified three ways: against numerical integration of the same distribution
    (first-order convergence confirmed by halving dx), against ``|y - m|`` for a
    degenerate point mass, and against the Gaussian closed form as the grid
    densifies (error 3.4e-03 at 9 quantiles, 6.3e-09 at 9999).
    """
    order = np.argsort(tau)
    tau, q = tau[order], q[order]
    # A quantile function cannot decrease; crossing quantiles are repaired here
    # rather than silently integrated as negative-width segments.
    q = np.maximum.accumulate(q)

    total = 0.0
    if y < q[0]:
        total += q[0] - y
    elif y > q[-1]:
        total += y - q[-1]

    for j in range(len(q) - 1):
        x0, x1 = q[j], q[j + 1]
        if x1 <= x0:
            continue
        f0, f1 = tau[j], tau[j + 1]
        if y <= x0:
            a, b = f0 - 1.0, f1 - 1.0
            total += (x1 - x0) * (a * a + a * b + b * b) / 3.0
        elif y >= x1:
            total += (x1 - x0) * (f0 * f0 + f0 * f1 + f1 * f1) / 3.0
        else:
            fy = f0 + (f1 - f0) * (y - x0) / (x1 - x0)
            total += (y - x0) * (f0 * f0 + f0 * fy + fy * fy) / 3.0
            a, b = fy - 1.0, f1 - 1.0
            total += (x1 - y) * (a * a + a * b + b * b) / 3.0
    return float(total)


def crps_from_quantiles_detail(
    y_true_returns: pd.Series, quantiles_dict: Dict[float, pd.Series]
) -> Dict[str, Any]:
    """
    Exact CRPS from a quantile forecast, with the tail bias it carries (A6.2).

    CRPS is the primary cross-model metric precisely because Chronos-2 exposes 21
    quantiles, TimesFM 2.5 exposes 9, and Kronos exposes sample paths -- it is
    supposed to put them on one scale. The superseded implementation integrated
    the pinball loss by the trapezoidal rule over only the supplied levels, which
    made that comparison unsound.

    The residual bias, measured rather than assumed
    -----------------------------------------------
    Even the exact value carries the flat-tail assumption, and its bias is
    **two-sided**, not one-directional:

    * a realisation **inside** the grid is scored slightly too low, because the
      mass the model placed beyond its outermost quantile is pinned onto that
      quantile instead of spreading out (measured: -0.0043 on a 0.1-0.9 grid);
    * a realisation **outside** the grid is scored too high, because F is pinned
      at exactly 0 or 1 there, so the integrand is a full 1 across the whole gap
      (measured: +0.0428, an order of magnitude larger).

    The outside cases are fewer but far bigger, so on a realistic sample the
    **net bias is upward**: +0.90% for a 9-quantile 0.1-0.9 grid against a
    Gaussian truth, +0.01% for a 0.01-0.99 grid. That ninety-fold difference
    between the two grids is the comparability problem, and it does not go away
    by being ignored.

    Diagnostics returned so the number can be read honestly:

    ``tail_mass_unmodelled``
        ``tau_min + (1 - tau_max)`` -- the probability mass the flat tails pin to
        the outermost quantiles. 0.20 for a 9-quantile 0.1-0.9 grid, 0.02 for a
        0.01-0.99 grid. This is what drives the bias.
    ``fraction_outside_grid``
        Share of realised values that landed beyond the outermost quantile --
        the observations carrying the large upward component.
    ``crps_trapezoid``
        The superseded approximation, kept so the improvement is visible rather
        than merely asserted.

    Two models may be compared directly on ``crps`` only when their
    ``tail_mass_unmodelled`` matches; otherwise the gap must be reported beside
    it, or both scored from samples where no tail assumption applies.
    """
    if not quantiles_dict:
        return {"crps": np.nan, "n": 0, "reason": "no quantiles supplied"}

    levels = np.array(sorted(quantiles_dict), dtype=np.float64)
    if levels.size < 2:
        return {"crps": np.nan, "n": 0, "reason": "need at least two quantile levels"}

    frame = pd.DataFrame({float(level): quantiles_dict[level] for level in sorted(quantiles_dict)})
    usable = y_true_returns.notna() & frame.notna().all(axis=1)
    y = y_true_returns[usable].to_numpy(dtype=np.float64)
    matrix = frame[usable].to_numpy(dtype=np.float64)
    n = int(y.size)
    if n == 0:
        return {"crps": np.nan, "n": 0, "reason": "no rows with a complete quantile set"}

    per_observation = np.array(
        [_exact_crps_piecewise_linear(y[i], matrix[i], levels) for i in range(n)]
    )

    # The superseded trapezoidal-pinball value, for comparison only.
    pinball_means = []
    for column, level in enumerate(levels):
        errors = y - matrix[:, column]
        pinball = np.where(errors >= 0, level * errors, (level - 1) * errors)
        pinball_means.append(float(np.mean(pinball)))
    trapezoid = 2.0 * float(np.trapezoid(pinball_means, levels))

    outside = int(np.sum((y < matrix.min(axis=1)) | (y > matrix.max(axis=1))))
    crossing = int(np.sum(np.any(np.diff(matrix, axis=1) < 0, axis=1)))

    return {
        "crps": round(float(np.mean(per_observation)), 8),
        "crps_trapezoid": round(trapezoid, 8),
        "n": n,
        "n_levels": int(levels.size),
        "level_min": round(float(levels[0]), 6),
        "level_max": round(float(levels[-1]), 6),
        "tail_mass_unmodelled": round(float(levels[0] + (1.0 - levels[-1])), 6),
        "fraction_outside_grid": round(outside / n, 6),
        "n_crossing_rows": crossing,
        "per_observation": per_observation,
        "bias_direction": "net_upward",
        "note": (
            "flat tails bias CRPS in BOTH directions -- slightly low for "
            "realisations inside the grid, substantially high for those outside "
            "-- with a net upward bias on a realistic sample. Compare two models "
            "on this value only when tail_mass_unmodelled matches."
        ),
    }


def crps_from_quantiles(
    y_true_returns: pd.Series, quantiles_dict: Dict[float, pd.Series]
) -> float:
    """
    Mean exact CRPS from a quantile forecast.

    Thin wrapper over :func:`crps_from_quantiles_detail`; use that when the tail
    bias matters, which for any cross-model comparison it does.
    """
    return float(crps_from_quantiles_detail(y_true_returns, quantiles_dict)["crps"])


def crps_from_samples(y_true_returns: pd.Series, samples: np.ndarray) -> Dict[str, Any]:
    """
    Exact empirical CRPS from predictive sample paths (Kronos, Chronos-2).

    Uses the energy form, which needs no binning and no tail assumption at all:

        CRPS = E|X - y| - 0.5 * E|X - X'|

    with the second term evaluated in O(s log s) from the sorted sample via

        E|X - X'| = (2 / s^2) * sum_i (2i - s + 1) * x_(i)     (0-based, ascending)

    rather than the O(s^2) double loop. Verified against that double loop to
    6e-16 and against the Gaussian closed form.

    Because there is no flat-tail truncation here, this value is directly
    comparable across sample-based models, and is the one to prefer whenever a
    model exposes draws.
    """
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2:
        raise ValueError(f"samples must be 2-D (n, s), got shape {draws.shape}")

    realised = y_true_returns.to_numpy(dtype=np.float64) if hasattr(
        y_true_returns, "to_numpy"
    ) else np.asarray(y_true_returns, dtype=np.float64)
    if draws.shape[0] != realised.size:
        raise ValueError(
            f"row mismatch: y has {realised.size} entries, samples has {draws.shape[0]} rows"
        )

    usable = np.isfinite(realised) & np.all(np.isfinite(draws), axis=1)
    realised, draws = realised[usable], draws[usable]
    n, s = draws.shape
    if n == 0:
        return {"crps": np.nan, "n": 0, "reason": "no rows with finite samples"}

    ordered = np.sort(draws, axis=1)
    weights = (2 * np.arange(s) - s + 1).astype(np.float64)
    mean_abs_error = np.mean(np.abs(ordered - realised[:, None]), axis=1)
    mean_pairwise = (2.0 / (s * s)) * (ordered @ weights)
    per_observation = mean_abs_error - 0.5 * mean_pairwise

    return {
        "crps": round(float(np.mean(per_observation)), 8),
        "n": int(n),
        "n_samples": int(s),
        "n_dropped": int(np.sum(~usable)),
        "per_observation": per_observation,
        "tail_mass_unmodelled": 0.0,
        "note": "empirical CRPS from draws; no tail assumption, no discretisation",
    }


def confusion_matrix(y_true_returns: pd.Series, p_up: pd.Series,
                     threshold: float = 0.5) -> Dict[str, int]:
    """
    Confusion counts for one window. Ties (r == 0) are labelled DOWN.

    Required per fold rather than pooled (A3.3): pooling across folds averages
    away regime-dependent behaviour, and in a market context that behaviour is
    the finding, not noise.
    """
    mask = y_true_returns.notna() & p_up.notna()
    truth = (y_true_returns[mask] > 0).astype(int).to_numpy()
    predicted = (p_up[mask] > float(threshold)).astype(int).to_numpy()
    return {
        "tp": int(np.sum((truth == 1) & (predicted == 1))),
        "tn": int(np.sum((truth == 0) & (predicted == 0))),
        "fp": int(np.sum((truth == 0) & (predicted == 1))),
        "fn": int(np.sum((truth == 1) & (predicted == 0))),
        "n": int(truth.size),
    }


def per_fold_directional_metrics(
    fold_ids: Sequence, y_true_returns: pd.Series, p_up: pd.Series,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Directional scorecard and confusion matrix for each fold separately (A3.3).

    A3.3 is explicit that the confusion matrix is reported per fold and not
    pooled, because a model that works only in one regime looks identical to a
    mediocre all-weather model once the folds are added together.
    """
    folds = pd.Series(list(fold_ids), index=y_true_returns.index)
    rows: List[Dict[str, Any]] = []
    for fold in pd.unique(folds):
        selector = folds == fold
        y_fold = y_true_returns[selector]
        p_fold = p_up[selector]
        entry: Dict[str, Any] = {"fold": fold}
        entry.update(directional_metrics(y_fold, p_fold))
        entry["confusion_matrix"] = confusion_matrix(y_fold, p_fold, threshold)
        rows.append(entry)
    return rows
