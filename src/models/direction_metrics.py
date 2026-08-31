"""
Scoring for the next-day direction classifier.

Accuracy on its own is theatre. On 252 test days the standard error of a
proportion is about 3.1 percentage points, so a model that scores 56.0% against
a 53% base rate has a 3pp edge and a z of 0.95 — p about 0.17, one-sided. That
is not a finding, it is a coin landing the way you hoped. Every accuracy figure
this module emits therefore carries an interval and a test against the reference
it claims to beat, and :func:`required_sample_size` says how much test data an
edge of a given size would actually need.

Two intervals are reported for one accuracy, and they answer different questions:

* ``wilson_interval`` — where the model's *true* accuracy plausibly sits. Wilson
  rather than the textbook normal interval because the normal one misbehaves near
  0 and 1 and undercovers at these sample sizes.
* ``accuracy_edge_test`` — whether the *gap* to a reference is separable from
  noise. Its standard error uses ``sqrt(0.25 / n)``, the largest a proportion's
  standard error can be, so the test never flatters the model by assuming a
  tighter spread than the data supports.

The skill scores mirror the regression pipeline's ``_baseline_skill`` (see
:mod:`src.models.ensemble_training`) with the constant *forecast* swapped for a
constant *classifier*: the reference always emits the training base rate.
Positive means the model added information; zero means it reproduced the base
rate; negative means it was worse than knowing nothing but the class balance.

Public API:
    classification_metrics(y_true, y_pred, p_up, ...) -> dict
    wilson_interval(successes, n, confidence) -> (lo, hi)
    accuracy_edge_test(model_accuracy, reference_accuracy, n) -> dict
    one_sided_p_value(z) -> float
    classifier_skill(y_true, p_up, reference_rate) -> dict
    calibration_bins(y_true, p_up, n_bins, strategy) -> list[dict]
    required_sample_size(edge, alpha, power) -> int
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-6


def _as_int_array(values: Sequence) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _as_prob_array(values: Sequence) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)


def _normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal. erfc avoids a scipy import."""
    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _normal_ppf(p: float) -> float:
    """
    Inverse standard normal CDF, Acklam's rational approximation.

    Accurate to ~1e-9 over the whole range, which is far tighter than anything
    that matters for a confidence level. Kept local so the metrics module has no
    scipy dependency.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    p_low, p_high = 0.02425, 1.0 - 0.02425

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def one_sided_p_value(z: float) -> float:
    """Upper-tail normal probability for a z statistic. P(Z > z)."""
    return float(_normal_sf(z))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.

    Preferred over the normal approximation: it stays inside [0, 1], keeps close
    to nominal coverage at the sample sizes a single walk-forward fold produces
    (63 rows), and does not collapse to zero width when the proportion does.
    """
    n = int(n)
    if n <= 0:
        return (float("nan"), float("nan"))
    z = _normal_ppf(0.5 + confidence / 2.0)
    phat = successes / n
    denominator = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denominator
    half_width = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denominator
    return (max(0.0, centre - half_width), min(1.0, centre + half_width))


def accuracy_edge_test(
    model_accuracy: float,
    reference_accuracy: float,
    n: int,
    *,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Is the accuracy gap over a reference separable from noise?

    Worked example from the handoff brief, reproduced exactly by this function:
    252 test days, 53% base rate, 56.0% model accuracy gives a 3.0pp edge, a
    standard error of sqrt(0.25/252) = 3.15pp, z = 0.95 and a one-sided p of
    about 0.17. Not significant.

    The test is one-sided because the claim being tested is directional — the
    model is asserted to be *better* than the reference, not merely different.

    Returns
    -------
    dict with edge_pp, standard_error_pp, z, p_value_one_sided, significant,
    and n_required (test days an edge this size would need at ``alpha``).
    """
    n = int(n)
    edge = float(model_accuracy) - float(reference_accuracy)
    if n <= 0:
        return {
            "edge_pp": round(edge * 100, 4), "standard_error_pp": None, "z": None,
            "p_value_one_sided": None, "significant": False, "alpha": alpha,
            "n_test": 0, "n_required": None,
        }

    # sqrt(0.25/n) is the maximum standard error of a proportion, attained at
    # p = 0.5. Using it rather than the reference's own p(1-p) keeps the test
    # conservative: it cannot manufacture significance by assuming a tight spread.
    standard_error = math.sqrt(0.25 / n)
    z = edge / standard_error if standard_error > 0 else 0.0
    p_value = _normal_sf(z)
    return {
        "edge_pp": round(edge * 100, 4),
        "standard_error_pp": round(standard_error * 100, 4),
        "z": round(z, 4),
        "p_value_one_sided": round(p_value, 6),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "n_test": n,
        "n_required": required_sample_size(edge, alpha=alpha) if edge > 0 else None,
    }


def required_sample_size(edge: float, alpha: float = 0.05, power: float = 0.80) -> Optional[int]:
    """
    Test observations needed to detect an accuracy edge of ``edge`` (as a fraction).

    Uses the same conservative sqrt(0.25/n) standard error as
    :func:`accuracy_edge_test`, so ``n = 0.25 * (z_alpha + z_power)^2 / edge^2``.
    A 3pp edge at alpha=0.05, power=0.80 needs about 1720 test days (roughly
    750 for bare significance at 50% power) — which is why a single 252-day year
    cannot settle the question however the number happens to land.
    """
    edge = float(edge)
    if edge <= 0:
        return None
    z_alpha = _normal_ppf(1.0 - alpha)
    z_power = _normal_ppf(power)
    return int(math.ceil(0.25 * (z_alpha + z_power) ** 2 / (edge ** 2)))


def brier_score(y_true: Sequence, p_up: Sequence) -> float:
    """Mean squared error of the probability. Lower is better; 0.25 is a coin."""
    y = _as_int_array(y_true).astype(np.float64)
    p = _as_prob_array(p_up)
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: Sequence, p_up: Sequence) -> float:
    """Binary cross-entropy. Punishes confident errors the way a bet does."""
    y = _as_int_array(y_true).astype(np.float64)
    p = _as_prob_array(p_up)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def matthews_corrcoef(y_true: Sequence, y_pred: Sequence) -> float:
    """
    Matthews correlation coefficient, computed from the confusion counts.

    Reported because accuracy on an imbalanced binary problem is dominated by
    the majority class; MCC is 0 for any constant predictor, whichever class it
    constantly picks.
    """
    y = _as_int_array(y_true)
    p = _as_int_array(y_pred)
    tp = float(np.sum((y == 1) & (p == 1)))
    tn = float(np.sum((y == 0) & (p == 0)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        # A predictor that never varies has no correlation with anything.
        return 0.0
    return float((tp * tn - fp * fn) / denominator)


def roc_auc(y_true: Sequence, p_up: Sequence) -> Optional[float]:
    """
    Area under the ROC curve via the rank (Mann-Whitney) identity, ties averaged.

    Returns None when the test window holds a single class, where AUC is
    undefined — reporting 0.5 there would invent a result.
    """
    y = _as_int_array(y_true)
    p = np.asarray(p_up, dtype=np.float64).reshape(-1)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = average_rank
        i = j + 1
    rank_sum_positive = float(np.sum(ranks[y == 1]))
    return float((rank_sum_positive - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classifier_skill(
    y_true: Sequence,
    p_up: Sequence,
    reference_rate: float | Sequence[float],
) -> Dict[str, float]:
    """
    Skill of the probabilities against a constant classifier at ``reference_rate``.

    The regression pipeline scores a model against the constant forecast
    ``train_mean``; this is the same construction with the constant classifier
    that always emits the training base rate:

        brier_skill_score = 1 - Brier(model) / Brier(constant)

    Positive means the probabilities carry information. Zero means they
    reproduce the base rate. Negative means they are worse than the base rate,
    which is the classification analogue of the -0.69 and -0.87 skill scores the
    30-day regressors currently report.

    ``prediction_std`` is the spread of the emitted probabilities. A collapsed
    model returns near-identical values for every input, and that is the
    cheapest way to spot it from a report file.

    ``reference_rate`` may be a single rate or one rate per row. Per-row is what
    pooled walk-forward folds need: the constant classifier a trader would
    actually have run on a given day knows only the base rate of the data
    available up to that day, and that rate differs from fold to fold.
    """
    y = _as_int_array(y_true)
    p = _as_prob_array(p_up)
    rates = np.asarray(reference_rate, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return {
            "reference_rate": round(float(np.mean(rates)), 6) if rates.size else 0.5,
            "model_brier": 0.0, "reference_brier": 0.0,
            "brier_skill_score": 0.0, "model_log_loss": 0.0, "reference_log_loss": 0.0,
            "log_loss_skill_score": 0.0, "prediction_std": 0.0,
        }
    if rates.size == 1:
        rates = np.repeat(rates, y.size)
    elif rates.size != y.size:
        raise ValueError(f"reference_rate has {rates.size} entries but y_true has {y.size}")

    constant = np.clip(rates, _EPS, 1.0 - _EPS)
    model_brier, reference_brier = brier_score(y, p), brier_score(y, constant)
    model_ll, reference_ll = log_loss(y, p), log_loss(y, constant)
    return {
        "reference_rate": round(float(np.mean(constant)), 6),
        "model_brier": round(model_brier, 6),
        "reference_brier": round(reference_brier, 6),
        "brier_skill_score": round(1.0 - model_brier / reference_brier, 6) if reference_brier > 0 else 0.0,
        "model_log_loss": round(model_ll, 6),
        "reference_log_loss": round(reference_ll, 6),
        "log_loss_skill_score": round(1.0 - model_ll / reference_ll, 6) if reference_ll > 0 else 0.0,
        "prediction_std": round(float(np.std(p)), 6),
    }


def calibration_bins(
    y_true: Sequence,
    p_up: Sequence,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> List[Dict[str, Any]]:
    """
    Reliability-curve points: predicted probability vs realised frequency.

    ``strategy='quantile'`` (the default) puts an equal count in each bin, which
    is what these probabilities need — they cluster tightly around 0.5, so
    equal-width bins leave most of the range empty and the two occupied bins
    carry everything. Pass ``'uniform'`` for equal-width bins over [0, 1].

    For a trading layer this matters more than the hard label: a threshold rule
    only means something if "0.58" is actually right 58% of the time. Bins with
    no observations are omitted rather than reported as zero.
    """
    y = _as_int_array(y_true).astype(np.float64)
    p = _as_prob_array(p_up)
    n_bins = max(2, int(n_bins))
    if y.size == 0:
        return []

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 2:
            # Every probability is identical — one bin is the whole truth here.
            edges = np.array([p[0] - _EPS, p[0] + _EPS])
    else:
        raise ValueError(f"strategy must be 'quantile' or 'uniform', got {strategy!r}")

    # np.digitize with right=False puts x == edges[-1] in an overflow bin; clip
    # it back into the last real bin so the top observation is not dropped.
    assignments = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    points: List[Dict[str, Any]] = []
    for b in range(len(edges) - 1):
        mask = assignments == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        observed = float(np.mean(y[mask]))
        lo, hi = wilson_interval(int(np.sum(y[mask])), count)
        points.append({
            "bin": b,
            "bin_lower": round(float(edges[b]), 6),
            "bin_upper": round(float(edges[b + 1]), 6),
            "count": count,
            "mean_predicted": round(float(np.mean(p[mask])), 6),
            "observed_frequency": round(observed, 6),
            "observed_ci_low": round(lo, 6),
            "observed_ci_high": round(hi, 6),
        })
    return points


def _per_class_scores(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> Dict[str, Any]:
    predicted = y_pred == label
    actual = y_true == label
    tp = float(np.sum(predicted & actual))
    n_predicted = float(np.sum(predicted))
    n_actual = float(np.sum(actual))
    precision = tp / n_predicted if n_predicted > 0 else None
    recall = tp / n_actual if n_actual > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "support": int(n_actual),
        "predicted": int(n_predicted),
        "precision": round(precision, 6) if precision is not None else None,
        "recall": round(recall, 6) if recall is not None else None,
        "f1": round(f1, 6) if f1 is not None else None,
    }


def classification_metrics(
    y_true: Sequence,
    y_pred: Sequence,
    p_up: Sequence,
    *,
    reference_accuracy: Optional[float] = None,
    reference_rate: Optional[float | Sequence[float]] = None,
    confidence: float = 0.95,
    calibration_bin_count: int = 10,
) -> Dict[str, Any]:
    """
    The full scorecard for one test window.

    Parameters
    ----------
    y_true, y_pred : sequences of 0/1
    p_up : sequence of P(up) in (0, 1)
    reference_accuracy : float, optional
        Accuracy of the baseline being claimed beaten — usually the majority
        baseline's, or the test-window base rate. Drives ``edge_vs_reference``.
    reference_rate : float or sequence of float, optional
        Training base rate for the constant classifier the skill scores are
        measured against; a sequence supplies one rate per row. Defaults to
        ``reference_accuracy``, then to 0.5.
    """
    y = _as_int_array(y_true)
    predictions = _as_int_array(y_pred)
    probabilities = _as_prob_array(p_up)
    n = int(y.size)
    if not (len(predictions) == len(probabilities) == n):
        raise ValueError(
            f"length mismatch: y_true={n}, y_pred={len(predictions)}, p_up={len(probabilities)}"
        )

    correct = int(np.sum(predictions == y))
    accuracy = correct / n if n else float("nan")
    accuracy_ci = wilson_interval(correct, n, confidence)

    class_1 = _per_class_scores(y, predictions, 1)
    class_0 = _per_class_scores(y, predictions, 0)
    # Balanced accuracy is the mean of the two recalls, so a model that wins by
    # always predicting the majority class scores 0.5 here however high its
    # plain accuracy is.
    recalls = [c["recall"] for c in (class_0, class_1) if c["recall"] is not None]
    balanced_accuracy = float(np.mean(recalls)) if recalls else None

    rate = reference_rate if reference_rate is not None else (
        reference_accuracy if reference_accuracy is not None else 0.5
    )

    metrics: Dict[str, Any] = {
        "n": n,
        "base_rate": round(float(np.mean(y)), 6) if n else None,
        "accuracy": round(accuracy, 6),
        "accuracy_ci_low": round(accuracy_ci[0], 6),
        "accuracy_ci_high": round(accuracy_ci[1], 6),
        "accuracy_ci_confidence": confidence,
        "balanced_accuracy": round(balanced_accuracy, 6) if balanced_accuracy is not None else None,
        "predicted_up_rate": round(float(np.mean(predictions)), 6) if n else None,
        "class_up": class_1,
        "class_down": class_0,
        "roc_auc": roc_auc(y, probabilities),
        "brier_score": round(brier_score(y, probabilities), 6),
        "log_loss": round(log_loss(y, probabilities), 6),
        "mcc": round(matthews_corrcoef(y, predictions), 6),
        "skill": classifier_skill(y, probabilities, rate),
        "calibration": calibration_bins(y, probabilities, calibration_bin_count),
    }
    if metrics["roc_auc"] is not None:
        metrics["roc_auc"] = round(metrics["roc_auc"], 6)
    if reference_accuracy is not None:
        metrics["edge_vs_reference"] = accuracy_edge_test(accuracy, float(reference_accuracy), n)
    return metrics
