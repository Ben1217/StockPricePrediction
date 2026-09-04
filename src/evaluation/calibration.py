"""
Calibration diagnostics and fold-honest recalibration (A3.2, A6.4-A6.6).

Two different failures hide behind one bad Brier score, and separating them is
the point of this module. A probability can be **miscalibrated** -- it says 70%
when the event happens 55% of the time -- or merely **uninformative** -- it says
53% for every bar, and is right about that. The first is fixable by
recalibration; the second is an absence of signal that no post-processing can
manufacture. Murphy's decomposition (A3.2) is what tells them apart:
reliability measures the first, resolution the second.

The invariant this module exists to enforce
-------------------------------------------
A recalibration mapping may be fitted **only on validation data**, and must be
refit inside each walk-forward fold. Fitting it on the test fold is leakage and
voids the evaluation (A9 item 7, which names calibration explicitly). That is
enforced through the API shape rather than by convention: :meth:`Recalibrator.fit`
and :meth:`Recalibrator.transform` take separate arguments, there is no
``fit_transform`` that could be handed a test array by accident, and
:func:`fold_honest_recalibration` never sees test labels at all.

A6.5 also requires that **both** raw and recalibrated results be reported.
:class:`RecalibrationResult` therefore carries both and overwrites neither.

Why the identity fallback is explicit
-------------------------------------
Isotonic regression on 30 validation points is a step function that memorises
noise. Rather than fit it anyway, the recalibrators **refuse** below a minimum
sample size (or on a one-class validation window, or a zero-variance score) and
fall back to the identity, flagged. A silent bad fit would be indistinguishable
from a good one in the results table; a flagged identity is not.

Public API:
    brier_decomposition(y_true, p_up, n_bins, strategy) -> dict
    reliability_diagram(y_true, p_up, n_bins, strategy) -> dict
    pit_values_from_quantiles(y, quantile_preds, levels) -> ndarray
    pit_values_from_samples(y, samples, seed) -> ndarray
    pit_histogram(pit, n_bins) -> dict
    IsotonicRecalibrator / PlattRecalibrator
    carve_validation_tail(train_pos, validation_fraction, min_validation) -> tuple
    FoldProbabilities / RecalibrationResult
    fold_honest_recalibration(folds, method, ...) -> RecalibrationResult
    calibration_verdict(coverage_reports, pit_report, ...) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from ..models.direction_metrics import brier_score, calibration_bins
from ..utils.logger import get_logger

logger = get_logger(__name__)

_EPS = 1e-6


def _as_labels(values: Sequence) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_probabilities(values: Sequence) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64).reshape(-1), _EPS, 1.0 - _EPS)


# ---------------------------------------------------------------------------
# A3.2 -- Murphy's decomposition
# ---------------------------------------------------------------------------


def brier_decomposition(
    y_true: Sequence,
    p_up: Sequence,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, Any]:
    """
    Split the Brier score into reliability, resolution and uncertainty.

        BS = RELIABILITY - RESOLUTION + UNCERTAINTY

        REL = (1/n) * sum_k n_k * (pbar_k - obar_k)^2
        RES = (1/n) * sum_k n_k * (obar_k - obar)^2
        UNC = obar * (1 - obar)

    with ``k`` indexing forecast bins, ``n_k`` the bin count, ``pbar_k`` the mean
    forecast in the bin, ``obar_k`` the observed frequency in it, and ``obar``
    the overall base rate.

    How to read it, which is the whole reason A3.2 asks for it:

    * **REL near 0** -- the probabilities mean what they say.
      **REL large** -- they are wrong, and recalibration can help.
    * **RES near 0** -- the probabilities are the same whatever the input, so
      they carry no information. Recalibration cannot fix this; there is nothing
      to calibrate. A model can be perfectly reliable and completely useless.
    * **UNC** depends only on the base rate, not on the model at all.

    The identity is exact only for a partition on *distinct* forecast values.
    Binning continuous forecasts leaves a within-bin variance term, so the
    residual is computed and reported rather than hidden -- a decomposition that
    silently fails to reconstruct its own Brier score is not evidence of
    anything.
    """
    y = _as_labels(y_true)
    p = _as_probabilities(p_up)
    if y.size != p.size:
        raise ValueError(f"length mismatch: y_true={y.size}, p_up={p.size}")
    n = int(y.size)
    if n == 0:
        raise ValueError("cannot decompose the Brier score of an empty sample")

    base_rate = float(np.mean(y))
    bins = calibration_bins(y, p, n_bins, strategy)

    reliability = 0.0
    resolution = 0.0
    for entry in bins:
        count = int(entry["count"])
        mean_predicted = float(entry["mean_predicted"])
        observed = float(entry["observed_frequency"])
        reliability += count * (mean_predicted - observed) ** 2
        resolution += count * (observed - base_rate) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base_rate * (1.0 - base_rate)

    actual = brier_score(y, p)
    reconstructed = reliability - resolution + uncertainty

    return {
        "n": n,
        "n_bins_occupied": len(bins),
        "strategy": strategy,
        "base_rate": round(base_rate, 6),
        "reliability": round(reliability, 8),
        "resolution": round(resolution, 8),
        "uncertainty": round(uncertainty, 8),
        "brier_score": round(actual, 8),
        "brier_reconstructed": round(reconstructed, 8),
        # Non-negative by construction (it is a within-bin variance), so a large
        # value means the bins are too coarse for the decomposition to be read
        # literally.
        "binning_residual": round(actual - reconstructed, 8),
        "bins": bins,
    }


# ---------------------------------------------------------------------------
# A6.4 -- reliability diagram and PIT
# ---------------------------------------------------------------------------


def reliability_diagram(
    y_true: Sequence,
    p_up: Sequence,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> Dict[str, Any]:
    """
    Binned predicted-vs-observed frequencies, plus ECE and MCE.

    ``ECE`` is the count-weighted mean absolute gap between the mean forecast in
    a bin and the frequency actually observed there; ``MCE`` is the worst single
    bin. Both are in probability units, so an ECE of 0.08 means the probabilities
    are off by eight percentage points on average.

    Returns plot-ready **data** only -- bin edges, counts, coordinates and the
    Wilson interval on each observed frequency. No figure is constructed here,
    and matplotlib is deliberately not imported anywhere in this package.
    """
    y = _as_labels(y_true)
    p = _as_probabilities(p_up)
    if y.size != p.size:
        raise ValueError(f"length mismatch: y_true={y.size}, p_up={p.size}")

    bins = calibration_bins(y, p, n_bins, strategy)
    n = int(y.size)
    if n == 0 or not bins:
        return {"n": n, "bins": [], "ece": None, "mce": None, "strategy": strategy}

    gaps = [
        (
            int(entry["count"]),
            abs(float(entry["mean_predicted"]) - float(entry["observed_frequency"])),
        )
        for entry in bins
    ]
    ece = sum(count * gap for count, gap in gaps) / n
    mce = max(gap for _, gap in gaps)

    return {
        "n": n,
        "strategy": strategy,
        "n_bins_occupied": len(bins),
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "bins": bins,
    }


def pit_values_from_quantiles(
    y: Sequence, quantile_preds: Sequence, levels: Sequence
) -> np.ndarray:
    """
    Probability integral transform F(y) read off a quantile forecast.

    ``quantile_preds`` is ``(n, q)`` and ``levels`` is ``(q,)``, strictly
    ascending in (0, 1). The CDF is interpolated linearly between the quantile
    knots; outside them ``np.interp`` clips to the outermost level, which is the
    honest reading -- beyond its outer quantile the model has said nothing about
    how much mass lies further out.

    A well-calibrated forecast gives PIT values uniform on (0, 1).
    """
    realised = _as_labels(y)
    predictions = np.asarray(quantile_preds, dtype=np.float64)
    taus = np.asarray(levels, dtype=np.float64).reshape(-1)

    if predictions.ndim != 2:
        raise ValueError(f"quantile_preds must be 2-D (n, q), got shape {predictions.shape}")
    if predictions.shape[0] != realised.size:
        raise ValueError(
            f"row mismatch: y has {realised.size} entries, quantile_preds has "
            f"{predictions.shape[0]} rows"
        )
    if predictions.shape[1] != taus.size:
        raise ValueError(
            f"column mismatch: quantile_preds has {predictions.shape[1]} columns, "
            f"levels has {taus.size}"
        )
    if np.any(np.diff(taus) <= 0) or taus[0] <= 0 or taus[-1] >= 1:
        raise ValueError("levels must be strictly ascending within the open interval (0, 1)")

    pit = np.empty(realised.size, dtype=np.float64)
    for i in range(realised.size):
        row = predictions[i]
        order = np.argsort(row, kind="mergesort")
        pit[i] = float(np.interp(realised[i], row[order], taus[order]))
    return pit


def pit_values_from_samples(
    y: Sequence, samples: Sequence, *, seed: Optional[int] = None
) -> np.ndarray:
    """
    Randomised PIT for a sample-based predictive distribution.

        PIT = ( #{x < y} + U * #{x == y} ) / s

    The randomisation matters: a discrete predictive distribution gives a PIT
    that is uniform only if ties are broken at random. With ``seed=None`` the tie
    term uses U = 0.5 deterministically, which keeps a run reproducible at the
    cost of a slight granularity artefact in the histogram; pass a seed for the
    properly randomised version.
    """
    realised = _as_labels(y)
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != 2:
        raise ValueError(f"samples must be 2-D (n, s), got shape {draws.shape}")
    if draws.shape[0] != realised.size:
        raise ValueError(
            f"row mismatch: y has {realised.size} entries, samples has {draws.shape[0]} rows"
        )

    n_samples = draws.shape[1]
    below = np.sum(draws < realised[:, None], axis=1)
    equal = np.sum(draws == realised[:, None], axis=1)
    if seed is None:
        uniform = np.full(realised.size, 0.5)
    else:
        uniform = np.random.default_rng(int(seed)).random(realised.size)
    return (below + uniform * equal) / float(n_samples)


def pit_histogram(
    pit: Sequence, *, n_bins: int = 10, alpha: float = 0.05, tie_tolerance: float = 0.01
) -> Dict[str, Any]:
    """
    Histogram of PIT values with two tests of uniformity.

    A flat histogram means calibrated. A **U shape** means the predictive
    distribution is too narrow -- reality lands in the tails more often than the
    model allows, which is the documented failure mode of foundation models on
    financial data (A6.3). A **hump** means it is too wide. A **slope** means the
    forecasts are biased.

    Why the KS test is not always trusted
    -------------------------------------
    Kolmogorov-Smirnov assumes a **continuous** null. A PIT read off a finite
    quantile grid is not continuous: values outside the outermost levels are
    clipped onto them, producing point masses at the grid boundary. On a correct
    forecast quoted over a 5%-95% grid, about 5% of observations land in each
    clipped tail *by construction* -- and KS duly returns p < 1e-9 for a model
    that is perfectly calibrated. Measured on a correctly specified Gaussian
    example: 4.7% and 5.5% of values pinned to the boundary, whole-sample
    KS p = 0.0000, but interior-only KS p = 0.667.

    Trusting KS blindly would therefore mark **every** quantile model
    miscalibrated, and that verdict feeds the A6.6/A11.5 "uncalibrated" badge.
    So the largest tie fraction is measured; above ``tie_tolerance`` the KS
    p-value is reported but flagged ``ks_valid: False`` and excluded from the
    verdict, which then rests on the chi-square test -- binned counts are valid
    for a discrete distribution, and the bins are what reveal the U shape anyway.
    """
    values = np.asarray(pit, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    n = int(values.size)
    n_bins = max(2, int(n_bins))

    if n == 0:
        return {"n": 0, "n_bins": n_bins, "counts": [], "edges": [],
                "expected_per_bin": None, "chi2_statistic": None, "chi2_p_value": None,
                "ks_statistic": None, "ks_p_value": None, "ks_valid": None,
                "max_tie_fraction": None, "uniform": None,
                "reason": "no finite PIT values"}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(np.clip(values, 0.0, 1.0), bins=edges)
    expected = n / n_bins

    chi2_statistic = float(np.sum((counts - expected) ** 2 / expected))
    chi2_p = float(stats.chi2.sf(chi2_statistic, df=n_bins - 1))
    ks = stats.kstest(values, "uniform")

    _, tie_counts = np.unique(values, return_counts=True)
    max_tie_fraction = float(tie_counts.max()) / n
    ks_valid = bool(max_tie_fraction < float(tie_tolerance))

    if ks_valid:
        uniform = bool(ks.pvalue >= alpha and chi2_p >= alpha)
        reason = (
            f"chi-square p = {chi2_p:.4f}, KS p = {float(ks.pvalue):.4f}"
        )
    else:
        uniform = bool(chi2_p >= alpha)
        reason = (
            f"chi-square p = {chi2_p:.4f}; the KS test was excluded because "
            f"{max_tie_fraction:.1%} of PIT values share a single value (a point "
            f"mass, usually quantile-grid clipping), which violates its "
            f"continuous-null assumption"
        )

    return {
        "n": n,
        "n_bins": n_bins,
        "counts": counts.tolist(),
        "edges": [round(float(edge), 6) for edge in edges],
        "expected_per_bin": round(float(expected), 6),
        "chi2_statistic": round(chi2_statistic, 6),
        "chi2_dof": n_bins - 1,
        "chi2_p_value": round(chi2_p, 8),
        "ks_statistic": round(float(ks.statistic), 6),
        "ks_p_value": round(float(ks.pvalue), 8),
        "ks_valid": ks_valid,
        "max_tie_fraction": round(max_tie_fraction, 6),
        "alpha": float(alpha),
        "uniform": uniform,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# A6.5 -- recalibration, fitted on validation folds only
# ---------------------------------------------------------------------------


class Recalibrator:
    """Base class. ``transform`` before ``fit`` is a programming error, not a default."""

    name = "identity"
    min_samples = 1

    def __init__(self, min_samples: Optional[int] = None):
        if min_samples is not None:
            self.min_samples = int(min_samples)
        self.fitted_ = False
        self.fit_info_: Dict[str, Any] = {}
        self._model: Any = None

    def _refuse(self, reason: str) -> "Recalibrator":
        self.fitted_ = False
        self.fit_info_ = {"fitted": False, "reason": reason, "method": self.name}
        logger.info("%s recalibrator declined to fit: %s", self.name, reason)
        return self

    def _check(self, p_val: np.ndarray, y_val: np.ndarray) -> Optional[str]:
        if p_val.size != y_val.size:
            raise ValueError(f"length mismatch: p_val={p_val.size}, y_val={y_val.size}")
        if p_val.size < self.min_samples:
            return f"only {p_val.size} validation rows, need {self.min_samples}"
        if np.unique(y_val).size < 2:
            return "the validation window holds a single class"
        if float(np.std(p_val)) < 1e-12:
            return "the validation scores have no variance"
        return None

    def fit(self, p_val: Sequence, y_val: Sequence) -> "Recalibrator":
        raise NotImplementedError

    def transform(self, p: Sequence) -> np.ndarray:
        """Map raw probabilities through the fitted mapping, or return them unchanged."""
        probabilities = _as_probabilities(p)
        if not self.fitted_:
            if not self.fit_info_:
                raise RuntimeError(
                    f"{self.name} recalibrator has not been fitted; call fit() on the "
                    f"VALIDATION fold first"
                )
            # Documented identity fallback: the fit was declined and said why.
            return probabilities
        return np.clip(self._apply(probabilities), _EPS, 1.0 - _EPS)

    def _apply(self, probabilities: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IsotonicRecalibrator(Recalibrator):
    """
    Monotone, non-parametric recalibration -- the default choice.

    Isotonic regression makes no shape assumption beyond monotonicity, which is
    exactly right for a probability that is ordered but miscalibrated. It is also
    greedy: on a small validation window it becomes a step function that
    memorises noise, so ``min_samples`` defaults to 50 and the fit is declined
    below that rather than performed badly.
    """

    name = "isotonic"
    min_samples = 50

    def fit(self, p_val: Sequence, y_val: Sequence) -> "IsotonicRecalibrator":
        probabilities = _as_probabilities(p_val)
        labels = _as_labels(y_val)
        refusal = self._check(probabilities, labels)
        if refusal:
            return self._refuse(refusal)

        from sklearn.isotonic import IsotonicRegression

        self._model = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True
        ).fit(probabilities, labels)
        self.fitted_ = True
        self.fit_info_ = {
            "fitted": True,
            "method": self.name,
            "n_validation": int(probabilities.size),
            "validation_base_rate": round(float(np.mean(labels)), 6),
        }
        return self

    def _apply(self, probabilities: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(probabilities), dtype=np.float64)


class PlattRecalibrator(Recalibrator):
    """
    Platt scaling: a logistic regression on the **logit** of the raw score.

    Regressing on the raw probability would fit a logistic curve to something
    already squashed into (0, 1), which cannot represent the identity mapping and
    so distorts an already-calibrated model. On the logit scale the identity is
    ``slope = 1, intercept = 0``, so a well-calibrated input passes through
    essentially untouched -- which is the property that makes the transform safe
    to apply unconditionally.

    Two parameters rather than isotonic's step function, so it needs far less
    validation data; ``min_samples`` is 20.
    """

    name = "platt"
    min_samples = 20

    def fit(self, p_val: Sequence, y_val: Sequence) -> "PlattRecalibrator":
        probabilities = _as_probabilities(p_val)
        labels = _as_labels(y_val)
        refusal = self._check(probabilities, labels)
        if refusal:
            return self._refuse(refusal)

        from sklearn.linear_model import LogisticRegression

        design = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
        self._model = LogisticRegression(solver="lbfgs", C=1e6).fit(design, labels.astype(int))
        self.fitted_ = True
        self.fit_info_ = {
            "fitted": True,
            "method": self.name,
            "n_validation": int(probabilities.size),
            "validation_base_rate": round(float(np.mean(labels)), 6),
            "slope": round(float(self._model.coef_[0][0]), 6),
            "intercept": round(float(self._model.intercept_[0]), 6),
        }
        return self

    def _apply(self, probabilities: np.ndarray) -> np.ndarray:
        design = np.log(probabilities / (1.0 - probabilities)).reshape(-1, 1)
        return np.asarray(self._model.predict_proba(design)[:, 1], dtype=np.float64)


RECALIBRATORS = {"isotonic": IsotonicRecalibrator, "platt": PlattRecalibrator}


def build_recalibrator(method: str, min_samples: Optional[int] = None) -> Recalibrator:
    if method not in RECALIBRATORS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(RECALIBRATORS)}")
    return RECALIBRATORS[method](min_samples=min_samples)


def carve_validation_tail(
    train_pos: Sequence[int],
    *,
    validation_fraction: float = 0.3,
    min_validation: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a training fold into an inner training block and a validation tail.

    The validation block is the **last** ``validation_fraction`` of the fold, so
    it sits chronologically after the rows the model fits on. A random slice
    would put validation rows before training rows and let the calibrator be
    tuned on a period the model had already seen -- the same ordering violation
    the walk-forward split exists to prevent, reintroduced one level down.

    Returns ``(train_pos, empty)`` with a warning when the fold is too short to
    carve, which the caller must treat as "no recalibration for this fold".
    """
    positions = np.asarray(train_pos, dtype=int).reshape(-1)
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    n_validation = int(round(len(positions) * float(validation_fraction)))
    if n_validation < int(min_validation) or len(positions) - n_validation < int(min_validation):
        logger.warning(
            "Training fold of %d rows cannot yield a %d-row validation tail; "
            "no recalibration will be fitted for it",
            len(positions), int(min_validation),
        )
        return positions, np.empty(0, dtype=int)

    return positions[:-n_validation], positions[-n_validation:]


@dataclass(frozen=True)
class FoldProbabilities:
    """One fold's validation and test probabilities. Test labels are for SCORING only."""

    fold: int
    p_up_val: np.ndarray
    y_val: np.ndarray
    p_up_test: np.ndarray
    y_test: Optional[np.ndarray] = None


@dataclass
class RecalibrationResult:
    """
    Raw and recalibrated probabilities, per fold and pooled.

    A6.5 requires both to be reported, so the raw arrays are carried alongside
    rather than replaced. ``n_folds_identity`` counts folds whose calibrator
    declined to fit -- those folds' "recalibrated" values are the raw ones, and
    a result where that count equals ``n_folds`` has recalibrated nothing.
    """

    method: str
    per_fold: List[Dict[str, Any]] = field(default_factory=list)
    n_folds_fitted: int = 0
    n_folds_identity: int = 0

    @property
    def n_folds(self) -> int:
        return len(self.per_fold)

    def pooled_raw(self) -> np.ndarray:
        if not self.per_fold:
            return np.empty(0)
        return np.concatenate([entry["p_up_test_raw"] for entry in self.per_fold])

    def pooled_recalibrated(self) -> np.ndarray:
        if not self.per_fold:
            return np.empty(0)
        return np.concatenate([entry["p_up_test_recalibrated"] for entry in self.per_fold])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_folds": self.n_folds,
            "n_folds_fitted": self.n_folds_fitted,
            "n_folds_identity": self.n_folds_identity,
            "recalibrated_anything": self.n_folds_fitted > 0,
            "per_fold": [
                {k: v for k, v in entry.items() if not isinstance(v, np.ndarray)}
                for entry in self.per_fold
            ],
        }


def fold_honest_recalibration(
    folds: Sequence[FoldProbabilities],
    *,
    method: str = "isotonic",
    min_samples: Optional[int] = None,
) -> RecalibrationResult:
    """
    Fit a recalibration mapping inside each fold, on that fold's validation rows only.

    The mapping is **refit per fold** (A6.5): a mapping fitted once on the first
    fold and reused later would carry that fold's information forward into every
    subsequent test window.

    Test labels are never read here. They are accepted on
    :class:`FoldProbabilities` purely so the caller can score raw against
    recalibrated afterwards, and this function does not touch them -- which is
    what makes the leakage claim checkable rather than asserted.
    """
    result = RecalibrationResult(method=method)

    for entry in folds:
        recalibrator = build_recalibrator(method, min_samples=min_samples)
        recalibrator.fit(entry.p_up_val, entry.y_val)
        raw = _as_probabilities(entry.p_up_test)
        recalibrated = recalibrator.transform(raw)

        if recalibrator.fitted_:
            result.n_folds_fitted += 1
        else:
            result.n_folds_identity += 1

        result.per_fold.append(
            {
                "fold": int(entry.fold),
                "fitted": bool(recalibrator.fitted_),
                "fit_info": dict(recalibrator.fit_info_),
                "n_validation": int(np.asarray(entry.p_up_val).size),
                "n_test": int(raw.size),
                "p_up_test_raw": raw,
                "p_up_test_recalibrated": recalibrated,
            }
        )

    logger.info(
        "fold-honest recalibration (%s): %d/%d folds fitted, %d fell back to identity",
        method, result.n_folds_fitted, result.n_folds, result.n_folds_identity,
    )
    return result


# ---------------------------------------------------------------------------
# A6.6 / A11.5 -- the verdict the UI keys its badge off
# ---------------------------------------------------------------------------


def calibration_verdict(
    coverage_reports: Sequence[Dict[str, Any]],
    pit_report: Optional[Dict[str, Any]] = None,
    *,
    coverage_tolerance: float = 0.05,
    ks_alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Turn the A6.3/A6.4 diagnostics into the single boolean A6.6 and A11.5 need.

    Deliberately **conservative**: any nominal level whose empirical coverage
    misses by more than ``coverage_tolerance``, or a PIT uniformity test below
    ``ks_alpha``, marks the model uncalibrated. The cost of wrongly labelling a
    calibrated model is a cautious badge; the cost of wrongly clearing a
    miscalibrated one is a dashboard asserting "Confidence: 73%" about a number
    that means nothing, which is precisely what A6.6 forbids.

    ``coverage_reports`` are dicts carrying ``nominal`` and ``coverage`` (the
    shape :func:`src.evaluation.metrics.quantile_metrics` returns in its
    ``coverage_*_detail`` entries).
    """
    failed: List[str] = []
    checked = 0

    for report in coverage_reports or []:
        nominal = report.get("nominal", report.get("achieved_nominal"))
        empirical = report.get("coverage")
        if nominal is None or empirical is None or not np.isfinite(float(empirical)):
            continue
        checked += 1
        error = float(empirical) - float(nominal)
        if abs(error) > float(coverage_tolerance):
            failed.append(
                f"{float(nominal):.0%} interval covered {float(empirical):.1%} "
                f"({error:+.1%} against a +/-{coverage_tolerance:.0%} tolerance)"
            )

    if pit_report:
        # Which uniformity test to believe depends on whether the PIT is
        # continuous. A quantile-grid PIT has point masses at the clipped
        # boundary, and KS then rejects a perfectly calibrated model -- see
        # pit_histogram. When pit_histogram has flagged ks_valid=False the
        # chi-square test is used instead; reading ks_p_value unconditionally
        # would mark every quantile model uncalibrated.
        ks_valid = pit_report.get("ks_valid")
        ks_p = pit_report.get("ks_p_value")
        chi2_p = pit_report.get("chi2_p_value")

        if ks_valid is False and chi2_p is not None:
            checked += 1
            if float(chi2_p) < float(ks_alpha):
                failed.append(
                    f"PIT values are not uniform (chi-square p = {float(chi2_p):.4f} "
                    f"< {ks_alpha}; KS excluded, "
                    f"{pit_report.get('max_tie_fraction', 0):.1%} of values are a point mass)"
                )
        elif ks_p is not None:
            checked += 1
            if float(ks_p) < float(ks_alpha):
                failed.append(
                    f"PIT values are not uniform (KS p = {float(ks_p):.4f} < {ks_alpha})"
                )

    if checked == 0:
        return {
            "is_calibrated": None,
            "checked": 0,
            "failed_checks": [],
            "reason": (
                "no coverage or PIT diagnostic was supplied, so calibration is "
                "unknown -- display the uncertainty as unverified, not as calibrated"
            ),
        }

    return {
        "is_calibrated": not failed,
        "checked": checked,
        "failed_checks": failed,
        "coverage_tolerance": float(coverage_tolerance),
        "ks_alpha": float(ks_alpha),
        "reason": (
            "; ".join(failed)
            if failed
            else f"all {checked} calibration checks passed"
        ),
    }
