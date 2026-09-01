"""
Tests for the combined-evidence direction call.

The behaviour worth pinning is what the module refuses to do. On a random walk
there is no direction to find, and a stack that reports UP with 68% confidence
on one is exactly the failure the whole design is meant to prevent - so the
first assertion here is that noise produces NEUTRAL. The rest pin the
arithmetic a reader is entitled to check: the contributions have to reconstruct
the probability they claim to explain, and the blend weights have to be the
measured skill numbers rather than anything chosen.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.models.direction_evidence import (  # noqa: E402
    EVIDENCE_CATEGORIES,
    _category_scores,
    _contributions,
    _sigmoid,
    analyse_direction,
    build_evidence_frame,
    build_evidence_stack,
    evaluate_evidence_stack,
)
from src.models.direction_models import LogisticDirection  # noqa: E402


def _random_walk_bars(n: int = 1800, seed: int = 2026) -> pd.DataFrame:
    """Bars with no exploitable structure: geometric Brownian motion."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2017-01-02", periods=n)
    returns = rng.normal(0.0004, 0.015, n)
    close = 60 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.007, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.007, n)))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(2_000_000, 9_000_000, n).astype(float)},
        index=index,
    )


@pytest.fixture(scope="module")
def random_walk_bars() -> pd.DataFrame:
    return _random_walk_bars()


@pytest.fixture(scope="module")
def random_walk_analysis(random_walk_bars) -> dict:
    return analyse_direction(random_walk_bars, symbol="NOISE")


def test_a_random_walk_is_reported_as_neutral(random_walk_analysis):
    """
    The headline requirement. With no classifier and no findable edge, the
    answer must be NEUTRAL with a stated reason - not a confident-looking arrow.
    """
    assert random_walk_analysis["direction"] == "NEUTRAL"
    assert random_walk_analysis["neutral_reason"]
    assert random_walk_analysis["confidence"]["label"] == "Low"


def test_probabilities_are_complementary_and_bounded(random_walk_analysis):
    up = random_walk_analysis["probability_up"]
    down = random_walk_analysis["probability_down"]
    assert 0.0 < up < 1.0
    assert up + down == pytest.approx(1.0)


def test_every_category_is_reported_with_a_contribution(random_walk_analysis):
    sources = [row["source"] for row in random_walk_analysis["evidence"]]
    assert sorted(sources) == sorted(EVIDENCE_CATEGORIES)
    assert all("contribution_pp" in row for row in random_walk_analysis["evidence"])
    # Ordered by how much each moved the answer, so the panel can read top-down.
    magnitudes = [abs(row["contribution_pp"]) for row in random_walk_analysis["evidence"]]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_contributions_reconstruct_the_probability_they_explain(random_walk_bars):
    """
    ``intercept + sum(contributions)`` has to be the fitted logit exactly. If it
    is not, the percentage points printed next to each evidence row are an
    attribution invented after the fact rather than the decomposition of the
    answer that was actually given.
    """
    frame = build_evidence_frame(random_walk_bars)
    stack = build_evidence_stack(frame)
    model = LogisticDirection(seed=42).fit(stack.scores, stack.labels)

    logit, contributions = _contributions(model, stack.latest_scores, EVIDENCE_CATEGORIES)
    direct = float(model.predict_proba_up(stack.latest_scores.to_frame().T)[0])

    assert _sigmoid(logit) == pytest.approx(direct, abs=1e-9)
    assert set(contributions) == set(EVIDENCE_CATEGORIES)


def test_blend_weights_are_the_measured_skill_numbers(random_walk_bars):
    """
    Nothing chooses how much each source counts for. The evidence stack's weight
    is its own walk-forward Brier skill, the classifier's is the skill handed in
    from its stored report, and a negative skill is floored at zero rather than
    inverted into a contrarian vote.
    """
    analysis = analyse_direction(
        random_walk_bars,
        symbol="NOISE",
        model_name="logistic",
        model_probability=0.62,
        model_skill=0.02,
        model_tradeable=True,
    )
    blend = analysis["blend"]

    assert blend["evidence_stack"]["weight"] == pytest.approx(
        max(blend["evidence_stack"]["brier_skill_score"], 0.0)
    )
    assert blend["classifier"]["weight"] == pytest.approx(0.02)

    negative = analyse_direction(
        random_walk_bars,
        symbol="NOISE",
        model_name="logistic",
        model_probability=0.62,
        model_skill=-0.05,
        model_tradeable=False,
    )
    assert negative["blend"]["classifier"]["weight"] == 0.0


def _fixed_evaluation(brier_skill_score: float):
    """A stand-in walk-forward result, so blend arithmetic can be checked exactly."""
    def _evaluate(stack, *, seed=42):
        return {
            "available": True, "n_folds": 5, "n_test_rows": 400,
            "test_range": ["2021-01-04", "2022-12-30"],
            "accuracy": 0.55, "accuracy_ci": [0.53, 0.60], "test_base_rate": 0.51,
            "brier_score": 0.24, "brier_skill_score": brier_skill_score,
            "log_loss_skill_score": brier_skill_score, "prediction_std": 0.04,
            "conviction_terciles": [0.01, 0.02],
            "oos_probabilities": np.full(400, 0.5),
        }
    return _evaluate


def test_a_skilled_classifier_pulls_the_blend_towards_itself(random_walk_bars, monkeypatch):
    """
    Same bars, same evidence, one classifier leaning hard up. The blend has to
    move that way - and further when the classifier's measured skill is larger,
    because the weight *is* the skill. The stack's own skill is pinned here so
    the comparison is between the two classifier weights and nothing else.
    """
    import src.models.direction_evidence as evidence_module

    monkeypatch.setattr(evidence_module, "evaluate_evidence_stack", _fixed_evaluation(0.01))
    weak = analyse_direction(
        random_walk_bars, symbol="NOISE", model_name="logistic",
        model_probability=0.75, model_skill=0.002, model_tradeable=True,
    )
    strong = analyse_direction(
        random_walk_bars, symbol="NOISE", model_name="logistic",
        model_probability=0.75, model_skill=0.20, model_tradeable=True,
    )
    evidence_only = weak["blend"]["evidence_stack"]["probability_up"]

    assert evidence_only < 0.75, "the fixture needs an evidence read below the classifier's"
    assert strong["probability_up"] > weak["probability_up"] > evidence_only


def test_the_blend_is_the_skill_weighted_log_odds_average(random_walk_bars, monkeypatch):
    """The published formula, checked against the published weights."""
    import src.models.direction_evidence as evidence_module

    monkeypatch.setattr(evidence_module, "evaluate_evidence_stack", _fixed_evaluation(0.03))
    analysis = analyse_direction(
        random_walk_bars, symbol="NOISE", model_name="logistic",
        model_probability=0.62, model_skill=0.05, model_tradeable=True,
    )

    stack_probability = analysis["blend"]["evidence_stack"]["probability_up"]
    logit = lambda p: float(np.log(p / (1.0 - p)))  # noqa: E731
    expected = 1.0 / (1.0 + np.exp(-(0.03 * logit(stack_probability) + 0.05 * logit(0.62)) / 0.08))

    assert analysis["probability_up"] == pytest.approx(expected, abs=1e-6)


def test_no_measured_skill_anywhere_falls_back_to_the_base_rate(random_walk_bars, monkeypatch):
    """
    When the stack scores no better than a constant base-rate forecast and there
    is no classifier, the answer is the base rate itself - not whichever of the
    two numbers happens to look more decisive.
    """
    import src.models.direction_evidence as evidence_module

    def _no_skill(stack, *, seed=42):
        return {
            "available": True, "n_folds": 1, "n_test_rows": 200,
            "test_range": ["2020-01-01", "2020-12-31"],
            "accuracy": 0.50, "accuracy_ci": [0.43, 0.57], "test_base_rate": 0.52,
            "brier_score": 0.25, "brier_skill_score": -0.01,
            "log_loss_skill_score": -0.01, "prediction_std": 0.01,
            "conviction_terciles": [0.01, 0.02],
            "oos_probabilities": np.full(200, 0.5),
        }

    monkeypatch.setattr(evidence_module, "evaluate_evidence_stack", _no_skill)
    analysis = analyse_direction(random_walk_bars, symbol="NOISE")

    assert analysis["direction"] == "NEUTRAL"
    assert analysis["probability_up"] == pytest.approx(analysis["base_rate"])
    assert "base rate" in analysis["blend"]["note"]


def test_the_evaluation_is_out_of_sample(random_walk_bars):
    """
    The walk-forward has to actually walk forward: more than one fold, a test
    range that starts after the training floor, and an accuracy interval rather
    than a bare point estimate.
    """
    frame = build_evidence_frame(random_walk_bars)
    stack = build_evidence_stack(frame)
    evaluation = evaluate_evidence_stack(stack)

    assert evaluation["available"] is True
    assert evaluation["n_folds"] >= 2
    assert evaluation["n_test_rows"] > 0
    assert evaluation["n_test_rows"] < len(stack)
    low, high = evaluation["accuracy_ci"]
    assert low <= evaluation["accuracy"] <= high


def test_the_latest_bar_is_never_trained_on(random_walk_bars):
    """
    The bar being predicted has no resolved outcome, so it cannot be a training
    row - and must not sneak in as one via the index intersection either.
    """
    frame = build_evidence_frame(random_walk_bars)
    stack = build_evidence_stack(frame)

    assert stack.latest_as_of not in stack.scores.index
    assert stack.latest_as_of not in stack.labels.index
    assert stack.latest_as_of > stack.scores.index[-1]


def test_horizon_reads_cover_short_medium_and_long(random_walk_analysis):
    horizons = random_walk_analysis["horizons"]
    assert set(horizons["directions"]) == {"short", "medium", "long"}
    assert horizons["agreement"] in {"aligned", "partly aligned", "conflicting", "undecided"}
    for key in ("short", "medium", "long"):
        assert horizons[key]["direction"] in {"UP", "DOWN", "NEUTRAL"}


def test_expected_range_is_a_spread_not_a_path(random_walk_analysis):
    """
    The range is three numbers and a sample size. There is deliberately no
    per-day series here: the model has no per-day claim to make.
    """
    expected = random_walk_analysis["expected_range"]
    assert expected["available"] is True
    assert expected["price_low"] <= expected["price_median"] <= expected["price_high"]
    assert expected["n_samples"] > 0
    assert "path" not in expected


def test_too_little_history_raises_rather_than_inventing_an_answer():
    bars = _random_walk_bars(n=150)
    with pytest.raises(ValueError):
        analyse_direction(bars, symbol="SHORT")


def _trending_bars(direction: int, n: int = 900, seed: int = 5) -> pd.DataFrame:
    """A clean, persistent trend with mild noise, so the sign of the read is not in doubt."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2019-01-02", periods=n)
    returns = rng.normal(direction * 0.0025, 0.008, n)
    close = 80 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.003
    low = np.minimum(open_, close) * 0.997
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float)},
        index=index,
    )


@pytest.mark.parametrize("direction", [1, -1])
def test_trend_and_momentum_scores_face_the_direction_of_the_trend(direction):
    """
    Orientation regression.

    ``Close_SMA20_Ratio`` and its siblings are named like ratios but hold
    ``close / sma - 1`` — a deviation already centred on zero. Subtracting one
    from them puts "price sitting on its moving average" at -1 and inverts the
    whole trend category, which is a bug no aggregate assertion catches: the
    probability still looks plausible, the sentence beside it just says the
    opposite of what the chart shows. So the sign is pinned directly.
    """
    frame = build_evidence_frame(_trending_bars(direction))
    scores = _category_scores(frame).dropna()

    assert not scores.empty
    latest = scores.iloc[-1]
    assert np.sign(latest["trend"]) == direction
    assert np.sign(latest["momentum"]) == direction
    assert np.sign(latest["price_action"]) == direction


def test_the_reported_state_matches_the_chart_in_a_clean_uptrend():
    """The words the panel prints, not just the numbers behind them."""
    analysis = analyse_direction(_trending_bars(1), symbol="UPTREND")
    states = {row["source"]: row["state"] for row in analysis["evidence"]}

    assert states["trend"] == "Bullish"
    assert states["momentum"] == "Positive"
    assert states["price_action"] == "Bullish structure"
    assert analysis["price_action"]["structure_label"] == "higher highs and higher lows"


@pytest.mark.parametrize(
    "probability, tradeable, expected",
    [
        (0.62, True, "UP"),
        (0.38, True, "DOWN"),
        # Positive Brier skill but a failed ship verdict: the probability is
        # still shown and still weighted, but no direction is called on it.
        (0.62, False, "NEUTRAL"),
    ],
)
def test_a_proven_classifier_produces_a_directional_call(random_walk_bars, probability, tradeable, expected):
    """
    The counterpart to the random-walk test. NEUTRAL is the honest answer when
    nothing has an edge — but a model that *has* shown one has to be able to
    produce a direction, or the panel is just permanently non-committal.
    """
    analysis = analyse_direction(
        random_walk_bars, symbol="NOISE", model_name="logistic",
        model_probability=probability, model_skill=0.05,
        model_tradeable=tradeable,
        model_gate_reason=None if tradeable else "Not tradeable: it loses to buy and hold after costs.",
    )

    assert analysis["direction"] == expected
    assert analysis["probability_up"] == pytest.approx(probability, abs=1e-6)
    if expected == "NEUTRAL":
        assert "ship criteria" in analysis["neutral_reason"]
    else:
        assert analysis["neutral_reason"] is None
