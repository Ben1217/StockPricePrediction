"""
Regression tests for the evaluation package (Addendum A).

Every test here pins a bug that was actually present and shipped, or an
invariant Addendum A makes non-negotiable. Expected values are derived
independently -- from a closed form, a hand-computed worked example, or a
brute-force reference written in the test -- never copied from what the
implementation happens to emit. A test whose expectation came from the code it
tests proves only that the code is deterministic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.evaluation.baselines import (
    always_directional_forecast,
    ar1_log_return_forecast,
    base_rate,
    climatology_p_up,
    har_rv_forecast,
    random_walk_drift_forecast,
    random_walk_forecast,
)
from src.evaluation.cross_sectional import compute_ic, quintile_spread_portfolio
from src.evaluation.economics import (
    CostModel,
    breakeven_round_trip_bps,
    long_flat_positions,
    long_short_positions,
    max_drawdown,
    performance_summary,
    strategy_returns,
)
from src.evaluation.metrics import (
    confusion_matrix,
    crps_from_quantiles,
    crps_from_quantiles_detail,
    crps_from_samples,
    directional_metrics,
    per_fold_directional_metrics,
    probabilistic_metrics,
    quantile_metrics,
)
from src.evaluation.testing import (
    benjamini_hochberg,
    diebold_mariano_test,
    mcnemar_test,
    newey_west_variance,
)
from src.evaluation.volatility import (
    evaluate_volatility,
    forward_realized_variance,
    garman_klass_variance,
    parkinson_variance,
    volatility_loss_comparison,
)


# ---------------------------------------------------------------------------
# A3 -- base-rate honesty
# ---------------------------------------------------------------------------


def test_brier_skill_score_is_negative_for_a_coin_flip_on_a_rising_window():
    """
    The bug: the climatological forecast was built with ``np.full_like(y_binary, pi)``,
    which inherited y_binary's integer dtype and truncated pi to 0. That scored
    the model against an always-down forecast, giving BS_clim = 0.75 instead of
    pi*(1-pi) = 0.1875 -- so a skill-less model reported BSS = +0.67 rather than
    -0.33. The sign flipped, which is the exact failure A0 forbids.
    """
    returns = pd.Series([0.01, 0.01, 0.01, -0.01, 0.01, -0.01, 0.01, 0.01])
    p_up = pd.Series([0.5] * 8)

    scores = probabilistic_metrics(returns, p_up)

    pi = 0.75
    assert scores["brier_score"] == pytest.approx(0.25)
    # BS_clim must be the closed form pi*(1-pi), so BSS = 1 - 0.25/0.1875.
    assert scores["brier_skill_score"] == pytest.approx(1.0 - 0.25 / (pi * (1 - pi)))
    assert scores["brier_skill_score"] < 0, "a coin flip must show no skill"


def test_eobr_is_negative_when_accuracy_trails_the_majority_rule():
    """A 68% model on a 70%-up window is worse than always guessing up."""
    returns = pd.Series([0.01] * 70 + [-0.01] * 30)
    # Correct on 68 of 100: right on 68 up-days, wrong on the other 2 up and all 30 down.
    p_up = pd.Series([0.9] * 68 + [0.1] * 2 + [0.9] * 30)

    scores = directional_metrics(returns, p_up)

    assert scores["base_rate"] == pytest.approx(0.70)
    assert scores["accuracy"] == pytest.approx(0.68)
    assert scores["eobr"] == pytest.approx(0.68 - 0.70)
    assert scores["eobr"] < 0


def test_constant_predictor_scores_zero_mcc_and_half_balanced_accuracy():
    """Both metrics must be immune to base-rate inflation, on any class balance."""
    returns = pd.Series([0.01] * 80 + [-0.01] * 20)
    always_up = pd.Series([0.99] * 100)

    scores = directional_metrics(returns, always_up)

    assert scores["accuracy"] == pytest.approx(0.80)  # looks good
    assert scores["mcc"] == pytest.approx(0.0)  # is not
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
    assert scores["eobr"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# A2 -- baselines emit probabilities, not labels
# ---------------------------------------------------------------------------


def test_directional_baselines_never_emit_a_hard_zero_or_one():
    """
    The bug: drift, AR(1) and always-up returned p_up in {0.0, 1.0}. Log-loss is
    then infinite the first time the rule is wrong, and ROC-AUC degenerates
    because the scores carry no within-class ordering.
    """
    index = pd.RangeIndex(50)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 300))

    for frame in (
        always_directional_forecast(index, "UP"),
        always_directional_forecast(index, "DOWN"),
        random_walk_drift_forecast(returns, horizon=5),
        ar1_log_return_forecast(returns.iloc[:200], returns.iloc[200:], horizon=5),
    ):
        p_up = frame["p_up"].to_numpy()
        assert np.all(p_up > 0.0) and np.all(p_up < 1.0)
        assert np.all(np.isfinite(np.log(p_up))) and np.all(np.isfinite(np.log1p(-p_up)))


def test_random_walk_is_the_martingale():
    prices = pd.Series([100.0, 101.0, 99.5, 102.0])
    frame = random_walk_forecast(prices, horizon=1)

    assert np.all(frame["pred_return"].to_numpy() == 0.0)
    assert np.all(frame["p_up"].to_numpy() == 0.5)
    assert np.allclose(frame["pred_price"].to_numpy(), prices.to_numpy())


def test_always_up_ignores_its_input_entirely():
    """The base-rate artifact must not depend on the data in any way."""
    index = pd.RangeIndex(20)
    first = always_directional_forecast(index, "UP")
    second = always_directional_forecast(index, "UP")
    assert first["p_up"].equals(second["p_up"])


def test_ar1_h_step_forecast_matches_an_iterated_brute_force_simulation():
    """
    The bug: the h-step forecast used the AR(1) *level* forecast of r_{t+h},
    while the evaluator realises the *cumulative* h-period return. That is a
    silent target mismatch. The closed form is checked here against an explicit
    iterated simulation driven by the same fitted parameters.
    """
    rng = np.random.default_rng(7)
    c, phi, sd, n = 0.0004, 0.35, 0.01, 3000
    series = np.zeros(n)
    for t in range(1, n):
        series[t] = c + phi * series[t - 1] + rng.normal(0, sd)
    data = pd.Series(series)
    train, test = data.iloc[:2500], data.iloc[2500:]

    # Refit exactly as the module does so the comparison isolates the formula.
    y, x = train.to_numpy()[1:], train.to_numpy()[:-1]
    design = np.column_stack([np.ones_like(x), x])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    c_hat, phi_hat = float(coefficients[0]), float(coefficients[1])
    residuals = y - design @ coefficients
    sd_hat = float(np.sqrt(np.sum(residuals**2) / (len(residuals) - 2)))

    horizon = 5
    out = ar1_log_return_forecast(train, test, horizon=horizon)

    start = float(train.iloc[-1])
    draws = 40_000
    simulated = np.empty(draws)
    for j in range(draws):
        value, total = start, 0.0
        for _ in range(horizon):
            value = c_hat + phi_hat * value + rng.normal(0, sd_hat)
            total += value
        simulated[j] = total

    standard_error = simulated.std(ddof=1) / np.sqrt(draws)
    assert float(out["pred_return"].iloc[0]) == pytest.approx(
        simulated.mean(), abs=5 * standard_error
    )
    assert float(out["sigma"].iloc[0]) == pytest.approx(simulated.std(ddof=1), rel=0.02)


def test_ar1_falls_back_visibly_rather_than_emitting_silent_zeros():
    short_train = pd.Series([0.001, -0.002, 0.003])
    test = pd.Series([0.001] * 5)
    out = ar1_log_return_forecast(short_train, test, horizon=1)

    assert bool(out["fell_back"].all())
    assert np.all(out["pred_return"].to_numpy() == 0.0)
    assert np.all(out["p_up"].to_numpy() == 0.5)


def test_har_rv_returns_nan_not_zero_when_the_fold_is_too_short():
    """
    A zero variance forecast is not a conservative default: QLIKE divides by the
    forecast, so a zero makes a missing model look catastrophically bad instead
    of absent.
    """
    rv = pd.Series(np.abs(np.random.default_rng(1).normal(1e-4, 2e-5, 40)))
    out = har_rv_forecast(rv.iloc[:10], rv.iloc[10:])
    assert out["pred_rv"].isna().all()


def test_base_rate_counts_ties_as_down():
    assert base_rate(pd.Series([0.01, -0.01, 0.0, 0.01])) == pytest.approx(0.5)
    assert climatology_p_up(0.6, 3).tolist() == pytest.approx([0.6, 0.6, 0.6])


# ---------------------------------------------------------------------------
# A5 -- significance testing
# ---------------------------------------------------------------------------


def test_newey_west_at_lag_zero_is_the_plain_sample_variance():
    d = np.random.default_rng(11).normal(size=200)
    assert newey_west_variance(d, lag=0) == pytest.approx(float(np.var(d)), rel=1e-12)


def test_newey_west_bartlett_weights_by_hand_at_lag_two():
    d = np.array([1.0, -2.0, 3.0, -1.0, 0.5, 2.0, -0.5, 1.5])
    n, centred = d.size, d - d.mean()
    gamma = lambda k: float(np.dot(centred[k:], centred[: n - k]) / n)  # noqa: E731
    expected = gamma(0) + 2 * ((1 - 1 / 3) * gamma(1) + (1 - 2 / 3) * gamma(2))
    assert newey_west_variance(d, lag=2) == pytest.approx(expected, rel=1e-12)


def test_diebold_mariano_sign_convention_and_unavailability():
    """d = L_model - L_baseline, so a negative mean means the MODEL is better."""
    rng = np.random.default_rng(5)
    baseline = np.abs(rng.normal(size=300)) + 0.5
    model = baseline - 0.05 + rng.normal(0, 0.01, 300)

    result = diebold_mariano_test(model, baseline, horizon=1)
    assert result["available"] is True
    assert result["mean_differential"] < 0
    assert result["sign"] == "model_better"

    identical = diebold_mariano_test(baseline, baseline, horizon=1)
    assert identical["available"] is False
    assert identical["reason"]
    assert identical["p_value"] is None, "an unavailable test must not report a p-value"


def test_diebold_mariano_defaults_its_hac_lag_to_horizon_minus_one():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=200), rng.normal(size=200)
    assert diebold_mariano_test(a, b, horizon=10)["lag_used"] == 9


def test_harvey_leybourne_newbold_factor_matches_the_published_formula():
    rng = np.random.default_rng(9)
    a, b = np.abs(rng.normal(size=100)), np.abs(rng.normal(size=100))
    result = diebold_mariano_test(a, b, horizon=1)
    n, h = result["n"], 1
    expected = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    assert result["hln_factor"] == pytest.approx(expected, rel=1e-6)
    assert result["dm_stat_hln"] == pytest.approx(result["dm_stat"] * expected, rel=1e-5)


def test_mcnemar_exact_binomial_against_a_hand_computed_value():
    """b=10, c=2 -> two-sided exact p = 2 * P(X <= 2) under Binom(12, 0.5)."""
    truth = np.ones(12, dtype=int)
    first = np.array([1] * 10 + [0] * 2)
    second = np.array([0] * 10 + [1] * 2)

    result = mcnemar_test(truth, first, second)
    expected = 2.0 * stats.binom.cdf(2, 12, 0.5)

    assert result["b_model1_only_correct"] == 10
    assert result["c_model2_only_correct"] == 2
    assert result["p_value_exact"] == pytest.approx(expected, rel=1e-7)
    assert result["test_used"] == "exact_binomial"


def test_mcnemar_reports_no_information_when_there_are_no_discordant_pairs():
    predictions = np.array([1, 0, 1, 1])
    result = mcnemar_test(predictions, predictions, predictions)
    assert result["available"] is False
    assert result["p_value"] is None, "p = 1 would imply the test ran and found equivalence"


def test_benjamini_hochberg_excludes_unavailable_tests_from_the_family_size():
    """
    The bug: NaN p-values were replaced by 1.0 and left in the family, inflating
    m. With two real tests among five slots, a raw p of 0.01 was adjusted to
    0.05 instead of 0.02 -- needlessly conservative, and the family definition
    reported alongside it was wrong.
    """
    report = benjamini_hochberg([0.01, 0.04, np.nan, np.nan, np.nan])

    assert report["family_size"] == 2
    assert report["n_excluded"] == 3
    assert report["adjusted_p"][0] == pytest.approx(0.02)
    assert report["adjusted_p"][1] == pytest.approx(0.04)
    assert np.isnan(report["adjusted_p"][2])
    assert report["rejected"][2] is False or report["rejected"][2] == False  # noqa: E712


def test_benjamini_hochberg_enforces_monotonicity():
    """The naive m*p/i is not the BH adjusted p-value and can be non-monotone."""
    p_values = [0.001, 0.30, 0.031, 0.04, 0.20]
    adjusted = np.array(benjamini_hochberg(p_values)["adjusted_p"])

    # Sorted ascending the family is 0.001, 0.031, 0.04, 0.20, 0.30, so 0.031
    # holds rank 2 and 0.04 holds rank 3. The naive m*p/i for 0.031 is therefore
    # 0.0775, which is LARGER than the 0.0667 of the next-ranked test -- a
    # non-monotone sequence that the step-up cumulative minimum must repair.
    naive_for_0031 = 5 * 0.031 / 2
    naive_for_004 = 5 * 0.04 / 3
    assert naive_for_0031 == pytest.approx(0.0775)
    assert naive_for_0031 > naive_for_004, "this example must be non-monotone to be a test"
    assert adjusted[2] == pytest.approx(naive_for_004)
    order = np.argsort(p_values)
    assert np.all(np.diff(adjusted[order]) >= -1e-12), "adjusted p must be monotone in rank"


# ---------------------------------------------------------------------------
# A6 -- interval coverage
# ---------------------------------------------------------------------------


def test_coverage_uses_the_nearest_levels_and_reports_what_it_achieved():
    """
    The bug: coverage required the exact keys 0.05/0.95, so TimesFM 2.5 -- whose
    nine quantiles do not contain them -- silently returned NaN and read as
    untested rather than under-covered.
    """
    rng = np.random.default_rng(3)
    y = pd.Series(rng.normal(0, 1, 2000))
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = {q: pd.Series(np.full(2000, stats.norm.ppf(q))) for q in levels}

    scores = quantile_metrics(y, quantiles)

    assert scores["coverage_90"] is not None and not np.isnan(scores["coverage_90"])
    # The nearest available pair to a 90% interval is 0.1/0.9, achieving only 80%.
    assert scores["coverage_90_detail"]["achieved_nominal"] == pytest.approx(0.8)
    assert scores["coverage_80_detail"]["achieved_nominal"] == pytest.approx(0.8)
    assert scores["coverage_80"] == pytest.approx(0.8, abs=0.03)


# ---------------------------------------------------------------------------
# A7 -- economics
# ---------------------------------------------------------------------------


def test_cost_arithmetic_charges_a_flip_twice_an_entry():
    cost = CostModel(commission_bps_per_side=1.0, spread_bps_per_side=2.5)  # 3.5 bps/side
    log_returns = np.log1p(np.array([0.01, -0.02, 0.03]))
    positions = np.array([1.0, -1.0, 1.0])  # flat->long, long->short, short->long

    result = strategy_returns(positions, log_returns, cost)

    assert result["traded"].tolist() == [1.0, 2.0, 2.0]
    assert (result["cost"] * 1e4).tolist() == pytest.approx([3.5, 7.0, 7.0])
    assert result["gross"].tolist() == pytest.approx([0.01, 0.02, 0.03])


def test_cost_model_always_states_the_market_impact_exclusion():
    payload = CostModel().to_dict()
    assert payload["market_impact_included"] is False
    assert "market impact is excluded" in payload["limitation"]


def test_max_drawdown_and_recovery_on_a_known_equity_path():
    """Equity 1.0 -> 1.2 -> 0.9 -> 1.3: a 25% drawdown from the peak at index 0."""
    returns = np.array([0.2, -0.25, 1.3 / 0.9 - 1.0])
    result = max_drawdown(returns)

    assert result["max_drawdown"] == pytest.approx(-0.25)
    assert result["trough_index"] == 1
    assert result["recovered"] is True
    assert result["time_to_recovery_periods"] == 2


def test_time_to_recovery_is_none_when_the_curve_never_recovers():
    """Filling in the sample end would report a recovery that did not happen."""
    result = max_drawdown(np.array([0.2, -0.5, -0.1]))
    assert result["recovered"] is False
    assert result["time_to_recovery_periods"] is None


def test_sortino_uses_downside_deviation_over_all_observations():
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    summary = performance_summary(returns, periods_per_year=252)

    shortfall = np.minimum(returns, 0.0)
    expected_downside = np.sqrt(np.mean(shortfall**2))
    expected = returns.mean() / expected_downside * np.sqrt(252)

    assert summary["sortino_ratio"] == pytest.approx(expected, rel=1e-6)


def test_annualised_return_is_geometric():
    returns = np.array([0.10, -0.05, 0.07, 0.02])
    summary = performance_summary(returns, periods_per_year=4)
    growth = float(np.prod(1 + returns))
    assert summary["annualised_return"] == pytest.approx(growth ** (4 / 4) - 1)
    assert summary["total_return"] == pytest.approx(growth - 1)


def test_breakeven_is_none_when_there_is_no_gross_alpha():
    positions = np.ones(5)
    losing = np.log1p(np.full(5, -0.01))
    result = breakeven_round_trip_bps(positions, losing)

    assert result["arithmetic_bps"] is None
    assert "no edge to erode" in result["reason"]


def test_breakeven_arithmetic_matches_the_closed_form():
    positions = np.ones(10)  # enter once, hold: one side of turnover in total
    log_returns = np.log1p(np.full(10, 0.002))
    result = breakeven_round_trip_bps(positions, log_returns)

    # sum(gross) = 10 * 0.002 = 0.02 against 1.0 unit traded -> 200 bps per side.
    assert result["arithmetic_bps"] == pytest.approx(200.0, rel=1e-6)
    assert result["compounded_bps"] == pytest.approx(200.0, rel=0.02)


def test_signal_rules_are_fixed_at_the_half_threshold():
    p_up = np.array([0.49, 0.50, 0.51])
    assert long_flat_positions(p_up).tolist() == [0.0, 0.0, 1.0]
    assert long_short_positions(p_up).tolist() == [-1.0, -1.0, 1.0]


# ---------------------------------------------------------------------------
# A7.5 -- cross-sectional
# ---------------------------------------------------------------------------


def _panel(rows, dates, tickers):
    return pd.DataFrame(rows, index=dates, columns=tickers, dtype=float)


def test_skipped_dates_are_counted_not_recorded_as_zero_returns():
    """
    The bug: a date with too few names appended a 0.0 spread. That is a
    fabricated observation -- it drags the mean toward zero and adds a
    zero-variance point that inflates the Sharpe ratio of everything around it.
    """
    dates = pd.date_range("2024-01-01", periods=3)
    tickers = list("ABCDE")
    predictions = _panel(
        [[1, 2, 3, 4, 5], [1, 2, np.nan, np.nan, np.nan], [5, 4, 3, 2, 1]], dates, tickers
    )
    realised = _panel(
        [[1, 2, 3, 4, 5], [1, 2, np.nan, np.nan, np.nan], [5, 4, 3, 2, 1]], dates, tickers
    )

    spread = quintile_spread_portfolio(predictions, realised, n_quantiles=5)
    assert spread["n_dates_skipped"] == 1
    assert spread["n_dates_used"] == 2
    assert len(spread["per_date"]) == 2

    ic = compute_ic(predictions, realised)
    assert ic["n_dates_skipped"] == 1
    assert ic["ic_mean"] == pytest.approx(1.0)


def test_rank_ic_uses_ranks_not_levels():
    """A monotone non-linear map must give Spearman 1.0 but Pearson below 1.0."""
    date = pd.date_range("2024-01-01", periods=1)
    tickers = list("ABCDE")
    predictions = _panel([[1, 2, 3, 4, 5]], date, tickers)
    realised = _panel([[1, 8, 27, 64, 125]], date, tickers)

    scores = compute_ic(predictions, realised)
    assert scores["rank_ic_mean"] == pytest.approx(1.0)
    assert scores["ic_mean"] < 1.0


# ---------------------------------------------------------------------------
# A8 -- volatility
# ---------------------------------------------------------------------------


def test_parkinson_and_garman_klass_against_a_hand_computed_bar():
    bar = pd.DataFrame({"Open": [100.0], "High": [101.0], "Low": [99.0], "Close": [100.5]})
    log_range = np.log(101 / 99)
    log_close_open = np.log(100.5 / 100)

    assert parkinson_variance(bar["High"], bar["Low"]).iloc[0] == pytest.approx(
        log_range**2 / (4 * np.log(2))
    )
    assert garman_klass_variance(
        bar["Open"], bar["High"], bar["Low"], bar["Close"]
    ).iloc[0] == pytest.approx(0.5 * log_range**2 - (2 * np.log(2) - 1) * log_close_open**2)


def test_range_estimators_are_more_efficient_than_close_to_close():
    """This efficiency is the entire reason A8.2 asks for OHLC estimators."""
    rng = np.random.default_rng(3)
    n, sigma, steps = 2000, 0.02, 400
    paths = np.cumsum(rng.normal(0, sigma, (n, steps)), axis=1)
    bars = pd.DataFrame(
        {
            "Open": np.full(n, 100.0),
            "High": np.exp(paths.max(axis=1)) * 100,
            "Low": np.exp(paths.min(axis=1)) * 100,
            "Close": np.exp(paths[:, -1]) * 100,
        }
    )

    parkinson = parkinson_variance(bars["High"], bars["Low"])
    garman_klass = garman_klass_variance(
        bars["Open"], bars["High"], bars["Low"], bars["Close"]
    )
    close_to_close = np.log(bars["Close"] / 100.0) ** 2

    assert parkinson.std() < close_to_close.std()
    assert garman_klass.std() < parkinson.std()


def test_forward_realized_variance_is_strictly_forward_looking():
    bars = pd.DataFrame(
        {
            "Open": np.full(10, 100.0),
            "High": np.linspace(101, 110, 10),
            "Low": np.linspace(99, 90, 10),
            "Close": np.full(10, 100.0),
        }
    )
    per_bar = parkinson_variance(bars["High"], bars["Low"])
    forward = forward_realized_variance(bars, estimator="parkinson", horizon=3)

    assert forward.iloc[0] == pytest.approx(per_bar.iloc[1:4].sum())
    assert forward.iloc[4] == pytest.approx(per_bar.iloc[5:8].sum())
    assert int(forward.isna().sum()) == 3, "the final h rows have no complete window"


def test_qlike_is_zero_only_at_a_perfect_forecast():
    realised = pd.Series([1e-4, 2e-4, 3e-4])
    assert evaluate_volatility(realised, realised)["qlike"] == pytest.approx(0.0)
    assert evaluate_volatility(realised, realised * 1.2)["qlike"] > 0
    assert evaluate_volatility(realised, realised * 0.8)["qlike"] > 0


def test_non_positive_variance_forecasts_are_dropped_not_clamped():
    """
    Clamping a zero forecast to 1e-8 turned a missing forecast into a QLIKE of
    order 1e4, making an absent model look catastrophically bad.
    """
    realised = pd.Series([1e-4, 2e-4, 3e-4])
    scores = evaluate_volatility(realised, pd.Series([1e-4, 0.0, -1.0]))

    assert scores["n_qlike"] == 1
    assert scores["n_dropped_non_positive"] == 2
    assert scores["qlike"] == pytest.approx(0.0)


def test_qlike_and_mse_disagreement_is_reported_not_resolved():
    """
    QLIKE penalises relative error, MSE absolute. A model that fits the crisis
    bar at the expense of every calm bar wins on MSE and loses on QLIKE, and
    A8.4 requires that be surfaced.
    """
    realised = pd.Series([1e-4] * 20 + [9e-3])
    proportional = pd.Series([1e-4] * 20 + [4.5e-3])
    crisis_fitter = pd.Series([3e-4] * 20 + [9e-3])

    comparison = volatility_loss_comparison(
        realised, {"proportional": proportional, "crisis_fitter": crisis_fitter}
    )

    assert comparison["qlike_best"] == "proportional"
    assert comparison["mse_best"] == "crisis_fitter"
    assert comparison["losses_disagree"] is True


# ---------------------------------------------------------------------------
# A6.2 -- CRPS, exact rather than approximated
# ---------------------------------------------------------------------------


def _gaussian_crps(y, mu=0.0, sigma=1.0):
    """Closed form: sigma * [ z(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]."""
    z = (y - mu) / sigma
    return sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )


def test_crps_of_a_point_mass_is_the_absolute_error():
    """
    The random-walk baseline's predictive distribution IS a point mass, so its
    CRPS must reduce exactly to mean |y - m|. Both representations must agree.
    """
    rng = np.random.default_rng(3)
    y = pd.Series(rng.normal(0, 1, 500))
    m = 0.25
    expected = float(np.mean(np.abs(y - m)))

    quantiles = {float(t): pd.Series(np.full(500, m)) for t in np.linspace(0.1, 0.9, 9)}
    assert crps_from_quantiles(y, quantiles) == pytest.approx(expected, abs=1e-8)

    from_samples = crps_from_samples(y, np.full((500, 40), m))
    assert from_samples["crps"] == pytest.approx(expected, abs=1e-8)


def test_sample_crps_matches_the_quadratic_reference_implementation():
    """
    The O(s log s) sorted identity must equal the obviously-correct O(s^2)
    double sum of E|X - X'|.
    """
    rng = np.random.default_rng(4)
    draws = rng.normal(0.5, 2.0, (12, 400))
    y = pd.Series(rng.normal(0, 1, 12))

    fast = crps_from_samples(y, draws)["per_observation"]
    reference = np.array(
        [
            np.mean(np.abs(draws[i] - y.iloc[i]))
            - 0.5 * np.mean(np.abs(draws[i][:, None] - draws[i][None, :]))
            for i in range(12)
        ]
    )
    assert np.allclose(fast, reference, atol=1e-12)


def test_sample_crps_matches_the_gaussian_closed_form():
    rng = np.random.default_rng(5)
    draws = rng.normal(0.0, 1.0, (1, 400_000))
    for y in (-1.0, 0.0, 0.8):
        got = crps_from_samples(pd.Series([y]), draws)["crps"]
        assert got == pytest.approx(_gaussian_crps(y), abs=5e-3)


def test_quantile_crps_converges_to_the_gaussian_closed_form():
    """A denser grid must monotonically approach the truth."""
    y_value = 0.37
    truth = _gaussian_crps(y_value)
    errors = []
    for k in (9, 99, 999):
        levels = np.linspace(1 / (k + 1), k / (k + 1), k)
        quantiles = {float(t): pd.Series([float(stats.norm.ppf(t))]) for t in levels}
        errors.append(abs(crps_from_quantiles(pd.Series([y_value]), quantiles) - truth))

    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 1e-5


def test_the_exact_crps_beats_the_superseded_trapezoid_rule():
    """
    The trapezoidal pinball integral over only the supplied levels was the old
    implementation. It must be strictly worse against a known truth.
    """
    rng = np.random.default_rng(6)
    n = 4000
    mu = rng.normal(0, 1, n)
    y = pd.Series(mu + rng.normal(0, 1, n))
    truth = float(np.mean([_gaussian_crps(v, m, 1.0) for v, m in zip(y.to_numpy(), mu)]))

    levels = np.linspace(0.1, 0.9, 9)
    quantiles = {float(t): pd.Series(mu + stats.norm.ppf(t)) for t in levels}
    detail = crps_from_quantiles_detail(y, quantiles)

    assert abs(detail["crps"] - truth) < abs(detail["crps_trapezoid"] - truth)


def test_crps_reports_the_tail_mass_its_flat_tails_pin_to_the_boundary():
    """
    ``tail_mass_unmodelled`` is what makes two grids comparable-or-not, so it
    must be exact: tau_min + (1 - tau_max).
    """
    y = pd.Series([0.0] * 50)
    for levels, expected in (
        (np.linspace(0.1, 0.9, 9), 0.2),
        (np.array([0.01] + list(np.linspace(0.05, 0.95, 19)) + [0.99]), 0.02),
    ):
        quantiles = {float(t): pd.Series(np.full(50, float(stats.norm.ppf(t)))) for t in levels}
        detail = crps_from_quantiles_detail(y, quantiles)
        assert detail["tail_mass_unmodelled"] == pytest.approx(expected, abs=1e-9)
        assert detail["n_levels"] == len(levels)


def test_the_flat_tail_bias_is_two_sided():
    """
    Documented behaviour, and the correction to an earlier wrong claim that the
    bias was uniformly downward: a realisation INSIDE the grid is scored a little
    low, one OUTSIDE is scored substantially high, and the outside cases dominate
    the mean.
    """
    levels = np.linspace(0.1, 0.9, 9)
    grid_low, grid_high = float(stats.norm.ppf(0.1)), float(stats.norm.ppf(0.9))

    def crps_at(value):
        quantiles = {float(t): pd.Series([float(stats.norm.ppf(t))]) for t in levels}
        return crps_from_quantiles(pd.Series([value]), quantiles)

    inside = 0.0
    outside = 3.0
    assert grid_low < inside < grid_high and outside > grid_high

    assert crps_at(inside) < _gaussian_crps(inside), "inside the grid: understated"
    assert crps_at(outside) > _gaussian_crps(outside), "outside the grid: overstated"
    # The outside error is far the larger of the two.
    assert abs(crps_at(outside) - _gaussian_crps(outside)) > 5 * abs(
        crps_at(inside) - _gaussian_crps(inside)
    )


def test_crossing_quantiles_are_repaired_and_counted():
    """A quantile function cannot decrease; a crossing row must not silently integrate."""
    y = pd.Series([0.0, 0.0])
    quantiles = {
        0.25: pd.Series([-1.0, 1.0]),
        0.5: pd.Series([0.0, 0.0]),
        0.75: pd.Series([1.0, -1.0]),
    }
    detail = crps_from_quantiles_detail(y, quantiles)

    assert detail["n_crossing_rows"] == 1
    assert np.all(np.isfinite(detail["per_observation"]))
    assert np.all(detail["per_observation"] >= 0), "CRPS is non-negative by definition"


def test_crps_from_samples_validates_its_input_shape():
    with pytest.raises(ValueError, match="2-D"):
        crps_from_samples(pd.Series([0.0, 1.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="row mismatch"):
        crps_from_samples(pd.Series([0.0, 1.0]), np.zeros((3, 10)))


def test_crps_handles_an_empty_or_degenerate_quantile_set():
    y = pd.Series([0.1, 0.2])
    assert np.isnan(crps_from_quantiles_detail(y, {})["crps"])
    single = {0.5: pd.Series([0.0, 0.0])}
    assert np.isnan(crps_from_quantiles_detail(y, single)["crps"])


# ---------------------------------------------------------------------------
# A3.3 -- confusion matrices per fold, never pooled
# ---------------------------------------------------------------------------


def test_confusion_matrix_counts_by_hand():
    returns = pd.Series([0.01, 0.01, -0.01, -0.01, 0.0])
    p_up = pd.Series([0.9, 0.1, 0.9, 0.1, 0.9])
    # ties (r == 0) are labelled DOWN, so the last row is a false positive
    assert confusion_matrix(returns, p_up) == {"tp": 1, "tn": 1, "fp": 2, "fn": 1, "n": 5}


def test_pooling_folds_conceals_regime_dependence():
    """
    Exactly why A3.3 forbids pooling. The same always-up model scores 80% in the
    bull fold and 20% in the bear fold; pooled it reads a bland 50% and the
    regime dependence -- the actual finding -- disappears.
    """
    returns = pd.Series([0.01] * 40 + [-0.01] * 10 + [0.01] * 10 + [-0.01] * 40)
    p_up = pd.Series([0.9] * 100)
    folds = ["bull"] * 50 + ["bear"] * 50

    per_fold = per_fold_directional_metrics(folds, returns, p_up)
    by_name = {row["fold"]: row for row in per_fold}

    assert by_name["bull"]["accuracy"] == pytest.approx(0.80)
    assert by_name["bear"]["accuracy"] == pytest.approx(0.20)
    assert by_name["bull"]["confusion_matrix"] == {"tp": 40, "tn": 0, "fp": 10, "fn": 0, "n": 50}
    assert by_name["bear"]["confusion_matrix"] == {"tp": 10, "tn": 0, "fp": 40, "fn": 0, "n": 50}

    pooled = confusion_matrix(returns, p_up)
    assert pooled == {"tp": 50, "tn": 0, "fp": 50, "fn": 0, "n": 100}
    # Every fold carries its own base rate, as A3.1 requires.
    assert by_name["bull"]["base_rate"] == pytest.approx(0.80)
    assert by_name["bear"]["base_rate"] == pytest.approx(0.20)
