"""
Tests for calibration diagnostics and fold-honest recalibration (A3.2, A6.4-A6.6).

Two design notes, because both were learned the hard way while building this:

**No knife-edge assertions on a p-value.** A calibrated forecast is rejected at
roughly the nominal alpha by construction, so a test asserting ``uniform is True``
on one seed is flaky at ~5-8%. Measured over 200 seeds, this module rejects a
calibrated sample-based PIT 8.0% of the time and a calibrated quantile-grid PIT
1.5% of the time, against 100% power on a genuinely too-narrow forecast. The
tests below therefore assert either a structural property (which is
deterministic) or a strongly-powered rejection, never a marginal acceptance.

**Expected values are derived, not observed.** Every number here comes from a
closed form or an explicitly written-out reference computation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from src.evaluation.calibration import (
    FoldProbabilities,
    IsotonicRecalibrator,
    PlattRecalibrator,
    brier_decomposition,
    build_recalibrator,
    calibration_verdict,
    carve_validation_tail,
    fold_honest_recalibration,
    pit_histogram,
    pit_values_from_quantiles,
    pit_values_from_samples,
    reliability_diagram,
)
from src.models.direction_metrics import brier_score

LEVELS = np.linspace(0.05, 0.95, 19)


# ---------------------------------------------------------------------------
# A3.2 -- Murphy's decomposition
# ---------------------------------------------------------------------------


def _three_value_sample(seed: int = 0):
    """Forecasts on three distinct values, so uniform bins form an exact partition."""
    rng = np.random.default_rng(seed)
    p = np.array([0.2] * 100 + [0.5] * 100 + [0.8] * 100)
    y = np.concatenate(
        [
            (rng.random(100) < 0.2).astype(float),
            (rng.random(100) < 0.5).astype(float),
            (rng.random(100) < 0.8).astype(float),
        ]
    )
    return y, p


def test_murphy_identity_is_exact_on_a_distinct_value_partition():
    """BS = REL - RES + UNC holds exactly when each bin holds one forecast value."""
    y, p = _three_value_sample()
    result = brier_decomposition(y, p, n_bins=10, strategy="uniform")

    assert result["brier_reconstructed"] == pytest.approx(result["brier_score"], abs=1e-9)
    assert abs(result["binning_residual"]) < 1e-9


def test_reliability_resolution_and_uncertainty_match_their_definitions():
    """Recomputed here from the bin table, independently of the implementation."""
    y, p = _three_value_sample()
    result = brier_decomposition(y, p, n_bins=10, strategy="uniform")

    n = len(y)
    base_rate = float(np.mean(y))
    reliability = sum(
        b["count"] * (b["mean_predicted"] - b["observed_frequency"]) ** 2 for b in result["bins"]
    ) / n
    resolution = sum(
        b["count"] * (b["observed_frequency"] - base_rate) ** 2 for b in result["bins"]
    ) / n

    assert result["reliability"] == pytest.approx(reliability, abs=1e-7)
    assert result["resolution"] == pytest.approx(resolution, abs=1e-7)
    assert result["uncertainty"] == pytest.approx(base_rate * (1 - base_rate), abs=1e-7)


def test_a_constant_forecast_has_zero_resolution():
    """
    Perfectly reliable and completely useless. This is the distinction A3.2 exists
    to surface: recalibration cannot rescue a forecast with nothing to calibrate.
    """
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.6).astype(float)
    p = np.full(4000, 0.6)

    result = brier_decomposition(y, p, n_bins=10, strategy="uniform")

    assert result["resolution"] < 1e-3
    assert result["reliability"] < 1e-3
    assert result["uncertainty"] == pytest.approx(float(np.mean(y)) * (1 - float(np.mean(y))))


def test_a_skilful_but_shifted_forecast_shows_high_reliability_error():
    """Informative (high resolution) yet wrong (high reliability) -- recalibration's case."""
    rng = np.random.default_rng(2)
    latent = rng.random(4000)
    y = (rng.random(4000) < latent).astype(float)
    shifted = np.clip(latent + 0.25, 0.01, 0.99)

    result = brier_decomposition(y, shifted, n_bins=10, strategy="uniform")

    assert result["resolution"] > 0.02, "a skilful forecast must have real resolution"
    assert result["reliability"] > 0.01, "a shifted forecast must be unreliable"


def test_brier_decomposition_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length mismatch"):
        brier_decomposition([1, 0, 1], [0.5, 0.5])


# ---------------------------------------------------------------------------
# A6.4 -- PIT and reliability diagram
# ---------------------------------------------------------------------------


def _gaussian_case(seed: int, n: int = 2000, quoted_sd: float = 1.0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0, 1, n)
    y = mu + rng.normal(0, 1, n)
    quantiles = mu[:, None] + stats.norm.ppf(LEVELS)[None, :] * quoted_sd
    return y, quantiles


def test_quantile_pit_is_clipped_to_the_level_grid():
    """
    Structural and deterministic: np.interp cannot return a value outside the
    supplied levels, so a 5%-95% grid pins about 5% of a CORRECT forecast's PIT
    to each boundary. This is the point mass that invalidates the KS test.
    """
    y, quantiles = _gaussian_case(seed=3)
    pit = pit_values_from_quantiles(y, quantiles, LEVELS)

    assert pit.min() >= LEVELS[0] - 1e-12
    assert pit.max() <= LEVELS[-1] + 1e-12
    at_boundary = np.mean((pit == LEVELS[0]) | (pit == LEVELS[-1]))
    assert 0.02 < at_boundary < 0.20, "a correct forecast should clip roughly 10% in total"


def test_ks_is_disabled_when_the_pit_has_a_point_mass():
    """
    Without this, KS rejects EVERY quantile model -- a calibrated Gaussian gave
    KS p = 6e-07 purely from grid clipping -- and that verdict feeds the A11.5
    'uncalibrated' badge.
    """
    y, quantiles = _gaussian_case(seed=4)
    report = pit_histogram(pit_values_from_quantiles(y, quantiles, LEVELS), n_bins=10)

    assert report["ks_valid"] is False
    assert report["max_tie_fraction"] >= 0.01
    assert "point mass" in report["reason"]
    # The KS p-value is still reported, just not used for the verdict.
    assert report["ks_p_value"] is not None


def test_ks_stays_enabled_when_the_pit_is_effectively_continuous():
    rng = np.random.default_rng(5)
    report = pit_histogram(rng.random(3000), n_bins=10)
    assert report["ks_valid"] is True


def test_a_too_narrow_predictive_distribution_gives_a_u_shaped_pit():
    """
    A6.3's documented failure mode, and the one the diagnostic must catch. Power
    was measured at 100% over 200 seeds, so this assertion is not marginal.
    """
    y, quantiles = _gaussian_case(seed=6, quoted_sd=0.5)
    pit = pit_values_from_quantiles(y, quantiles, LEVELS)
    report = pit_histogram(pit, n_bins=10)
    counts = np.array(report["counts"])

    assert report["uniform"] is False
    assert report["chi2_p_value"] < 1e-6
    assert counts[0] + counts[-1] > counts[4] + counts[5], "mass must pile into the tails"


def test_a_calibrated_forecast_is_not_u_shaped():
    """
    Asserts the SHAPE rather than the binary verdict: a calibrated forecast is
    rejected at roughly the nominal alpha, so a `uniform is True` assertion here
    would be flaky by construction.
    """
    y, quantiles = _gaussian_case(seed=7)
    counts = np.array(pit_histogram(pit_values_from_quantiles(y, quantiles, LEVELS))["counts"])
    edges = counts[0] + counts[-1]
    middle = counts[4] + counts[5]
    assert edges < 2 * middle, "a calibrated PIT must not pile up in the tails"


def test_sample_based_pit_is_deterministic_without_a_seed():
    y = np.array([0.0, 1.0])
    samples = np.array([[-1.0, 0.0, 1.0, 2.0], [-1.0, 0.0, 1.0, 2.0]])
    first = pit_values_from_samples(y, samples)
    second = pit_values_from_samples(y, samples)
    assert np.array_equal(first, second)
    # y=0.0: one draw below, one equal -> (1 + 0.5*1)/4 = 0.375
    assert first[0] == pytest.approx(0.375)
    # y=1.0: two draws below, one equal -> (2 + 0.5*1)/4 = 0.625
    assert first[1] == pytest.approx(0.625)


def test_pit_from_quantiles_validates_its_inputs():
    y = np.zeros(3)
    good = np.tile(np.array([-1.0, 0.0, 1.0]), (3, 1))
    levels = np.array([0.25, 0.5, 0.75])

    with pytest.raises(ValueError, match="2-D"):
        pit_values_from_quantiles(y, np.zeros(3), levels)
    with pytest.raises(ValueError, match="row mismatch"):
        pit_values_from_quantiles(np.zeros(4), good, levels)
    with pytest.raises(ValueError, match="column mismatch"):
        pit_values_from_quantiles(y, good, np.array([0.5, 0.9]))
    with pytest.raises(ValueError, match="strictly ascending"):
        pit_values_from_quantiles(y, good, np.array([0.75, 0.5, 0.25]))


def test_pit_histogram_on_an_empty_sample_reports_rather_than_raises():
    report = pit_histogram(np.array([]))
    assert report["n"] == 0
    assert report["uniform"] is None
    assert report["reason"]


def test_reliability_diagram_ece_and_mce_match_their_definitions():
    y, p = _three_value_sample(seed=8)
    diagram = reliability_diagram(y, p, n_bins=10, strategy="uniform")

    gaps = [
        (b["count"], abs(b["mean_predicted"] - b["observed_frequency"])) for b in diagram["bins"]
    ]
    expected_ece = sum(count * gap for count, gap in gaps) / len(y)
    expected_mce = max(gap for _, gap in gaps)

    assert diagram["ece"] == pytest.approx(expected_ece, abs=1e-6)
    assert diagram["mce"] == pytest.approx(expected_mce, abs=1e-6)
    assert diagram["mce"] >= diagram["ece"], "the worst bin cannot beat the average"


# ---------------------------------------------------------------------------
# A6.5 -- recalibration, validation folds only
# ---------------------------------------------------------------------------


def _miscalibrated_split(seed: int = 9):
    """A monotone but squashed score: ordered correctly, wrong in level."""
    rng = np.random.default_rng(seed)
    latent_val = rng.random(3000)
    y_val = (rng.random(3000) < latent_val).astype(float)
    latent_test = rng.random(3000)
    y_test = (rng.random(3000) < latent_test).astype(float)
    return latent_val**2, y_val, latent_test**2, y_test


@pytest.mark.parametrize("method", ["isotonic", "platt"])
def test_recalibration_improves_the_brier_score_on_held_out_data(method):
    p_val, y_val, p_test, y_test = _miscalibrated_split()
    recalibrator = build_recalibrator(method).fit(p_val, y_val)
    recalibrated = recalibrator.transform(p_test)

    assert recalibrator.fitted_ is True
    assert brier_score(y_test, recalibrated) < brier_score(y_test, p_test)


def test_platt_is_near_identity_on_an_already_calibrated_score():
    """
    Fitting on the LOGIT is what makes this true: the identity mapping is
    slope 1, intercept 0, so a calibrated input passes through untouched.
    Regressing on the raw probability could not represent the identity at all.
    """
    rng = np.random.default_rng(10)
    latent_val = rng.random(3000)
    y_val = (rng.random(3000) < latent_val).astype(float)
    latent_test = rng.random(3000)

    recalibrator = PlattRecalibrator().fit(latent_val, y_val)
    out = recalibrator.transform(latent_test)

    assert recalibrator.fit_info_["slope"] == pytest.approx(1.0, abs=0.15)
    assert recalibrator.fit_info_["intercept"] == pytest.approx(0.0, abs=0.15)
    assert np.max(np.abs(out - latent_test)) < 0.05


@pytest.mark.parametrize(
    "label, p_val, y_val",
    [
        ("one class", np.linspace(0.2, 0.8, 100), np.ones(100)),
        ("too few rows", np.linspace(0.2, 0.8, 10), (np.arange(10) % 2).astype(float)),
        ("no score variance", np.full(100, 0.5), (np.arange(100) % 2).astype(float)),
    ],
)
def test_a_declined_fit_returns_the_input_unchanged_and_says_why(label, p_val, y_val):
    """A silent bad fit is indistinguishable from a good one; a flagged identity is not."""
    recalibrator = IsotonicRecalibrator().fit(p_val, y_val)
    probe = np.array([0.1, 0.5, 0.9])

    assert recalibrator.fitted_ is False
    assert recalibrator.fit_info_["reason"]
    assert np.allclose(recalibrator.transform(probe), probe)


def test_transform_before_fit_raises_rather_than_defaulting():
    with pytest.raises(RuntimeError, match="has not been fitted"):
        IsotonicRecalibrator().transform([0.5])


def test_build_recalibrator_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        build_recalibrator("sigmoid")


def test_carve_validation_tail_keeps_validation_after_training():
    """
    A random slice would put validation rows before training rows and let the
    calibrator be tuned on a period the model had already seen -- the ordering
    violation the walk-forward split exists to prevent, one level down.
    """
    train = np.arange(1000)
    inner, validation = carve_validation_tail(train, validation_fraction=0.3)

    assert len(validation) == 300
    assert inner[-1] < validation[0]
    assert np.intersect1d(inner, validation).size == 0
    assert len(inner) + len(validation) == len(train)
    assert np.array_equal(np.concatenate([inner, validation]), train)


def test_carve_validation_tail_refuses_a_fold_that_is_too_short():
    inner, validation = carve_validation_tail(np.arange(20), validation_fraction=0.3)
    assert validation.size == 0
    assert len(inner) == 20


def test_carve_validation_tail_rejects_an_out_of_range_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        carve_validation_tail(np.arange(100), validation_fraction=1.5)


def test_fold_honest_recalibration_never_reads_the_test_labels():
    """
    The leakage claim made checkable: the test labels are all NaN, so any code
    path that used them would produce NaN probabilities or raise. It does neither.
    """
    p_val, y_val, p_test, _ = _miscalibrated_split(seed=11)
    folds = [
        FoldProbabilities(
            fold=1,
            p_up_val=p_val,
            y_val=y_val,
            p_up_test=p_test,
            y_test=np.full(p_test.size, np.nan),
        )
    ]

    result = fold_honest_recalibration(folds, method="isotonic")

    assert result.n_folds_fitted == 1
    assert np.all(np.isfinite(result.pooled_recalibrated()))


def test_fold_honest_recalibration_keeps_both_raw_and_recalibrated():
    """A6.5 requires both to be reported, so neither may overwrite the other."""
    p_val, y_val, p_test, _ = _miscalibrated_split(seed=12)
    folds = [
        FoldProbabilities(fold=i, p_up_val=p_val, y_val=y_val, p_up_test=p_test)
        for i in (1, 2)
    ]

    result = fold_honest_recalibration(folds, method="isotonic")

    assert result.n_folds == 2
    assert result.pooled_raw().size == 2 * p_test.size
    assert result.pooled_recalibrated().size == 2 * p_test.size
    assert not np.allclose(result.pooled_raw(), result.pooled_recalibrated())
    assert result.to_dict()["recalibrated_anything"] is True


def test_folds_that_cannot_be_fitted_are_counted_as_identity():
    p_test = np.linspace(0.1, 0.9, 50)
    folds = [
        FoldProbabilities(
            fold=1,
            p_up_val=np.linspace(0.2, 0.8, 10),   # below isotonic's minimum
            y_val=(np.arange(10) % 2).astype(float),
            p_up_test=p_test,
        )
    ]

    result = fold_honest_recalibration(folds, method="isotonic")

    assert result.n_folds_fitted == 0
    assert result.n_folds_identity == 1
    assert np.allclose(result.pooled_recalibrated(), result.pooled_raw())
    assert result.to_dict()["recalibrated_anything"] is False


# ---------------------------------------------------------------------------
# A6.6 / A11.5 -- the verdict
# ---------------------------------------------------------------------------


def test_verdict_flags_under_coverage():
    verdict = calibration_verdict([{"nominal": 0.90, "coverage": 0.76}], {"ks_p_value": 0.4})
    assert verdict["is_calibrated"] is False
    assert "76" in verdict["failed_checks"][0]


def test_verdict_accepts_coverage_within_tolerance():
    verdict = calibration_verdict([{"nominal": 0.90, "coverage": 0.89}], {"ks_p_value": 0.4})
    assert verdict["is_calibrated"] is True


def test_verdict_uses_chi_square_when_ks_is_invalid():
    """
    A quantile-grid PIT has point masses, so its KS p-value is meaningless. Using
    it anyway would mark every quantile model uncalibrated.
    """
    pit_report = {
        "ks_p_value": 1e-12,
        "ks_valid": False,
        "chi2_p_value": 0.40,
        "max_tie_fraction": 0.05,
    }
    verdict = calibration_verdict([{"nominal": 0.90, "coverage": 0.89}], pit_report)
    assert verdict["is_calibrated"] is True

    pit_report["chi2_p_value"] = 1e-9
    failing = calibration_verdict([{"nominal": 0.90, "coverage": 0.89}], pit_report)
    assert failing["is_calibrated"] is False
    assert "chi-square" in failing["failed_checks"][0]


def test_verdict_is_none_when_nothing_was_checked():
    """Unknown is not the same as calibrated, and must not be reported as it."""
    verdict = calibration_verdict([], None)
    assert verdict["is_calibrated"] is None
    assert verdict["checked"] == 0
    assert "unknown" in verdict["reason"]


def test_verdict_ignores_unusable_coverage_entries():
    verdict = calibration_verdict(
        [{"nominal": 0.90, "coverage": float("nan")}, {"nominal": None, "coverage": 0.9}], None
    )
    assert verdict["is_calibrated"] is None
    assert verdict["checked"] == 0
