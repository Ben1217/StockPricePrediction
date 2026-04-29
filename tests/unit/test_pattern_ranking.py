"""
Unit tests for chart pattern ranking and best-setup selection.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.pattern_detector import rank_patterns, select_best_pattern, evaluate_best_setup


def _base_pattern(**overrides):
    pattern = {
        "pattern_name": "Double Bottom",
        "direction": "bullish",
        "status": "forming",
        "start_date": "2026-01-01",
        "end_date": "2026-01-20",
        "timeframe": "1d",
        "weight": 3,
        "key_levels": [
            {"date": "2026-01-05", "price": 560.0},
            {"date": "2026-01-10", "price": 575.0},
            {"date": "2026-01-20", "price": 561.0},
        ],
        "trendlines": None,
        "neckline": 575.0,
        "breakout_price": None,
        "target_price": 610.0,
        "stop_loss": 560.0,
    }
    pattern.update(overrides)
    return pattern


def test_rank_patterns_prefers_higher_confidence_setup():
    patterns = [
        _base_pattern(
            pattern_name="Double Bottom",
            neckline=575.0,
            target_price=610.0,
            stop_loss=560.0,
        ),
        _base_pattern(
            pattern_name="Bull Flag",
            start_date="2026-02-01",
            end_date="2026-02-18",
            entry_price=575.0,
            neckline=None,
            key_levels=[
                {"date": "2026-02-01", "price": 545.0},
                {"date": "2026-02-10", "price": 575.0},
                {"date": "2026-02-18", "price": 571.0},
            ],
            trendlines=[
                [{"date": "2026-02-10", "price": 575.0}, {"date": "2026-02-18", "price": 572.0}],
                [{"date": "2026-02-10", "price": 566.0}, {"date": "2026-02-18", "price": 563.0}],
            ],
        ),
    ]

    ranked = rank_patterns(patterns)

    assert ranked[0]["pattern_name"] == "Bull Flag"
    assert ranked[0]["confidence"] > ranked[1]["confidence"]
    assert ranked[0]["entry_price"] == 575.0
    assert ranked[1]["entry_price"] == 575.0


def test_select_best_pattern_prefers_actionable_trade_over_neutral_pattern():
    patterns = [
        _base_pattern(
            pattern_name="Symmetrical Triangle",
            direction="neutral",
            timeframe="1wk",
            weight=4,
            entry_price=None,
            neckline=None,
            breakout_price=None,
            target_price=None,
            stop_loss=None,
            key_levels=[
                {"date": "2026-03-01", "price": 580.0},
                {"date": "2026-03-08", "price": 575.0},
            ],
            trendlines=[
                [{"date": "2026-03-01", "price": 590.0}, {"date": "2026-03-20", "price": 582.0}],
                [{"date": "2026-03-01", "price": 560.0}, {"date": "2026-03-20", "price": 568.0}],
            ],
        ),
        _base_pattern(
            pattern_name="Head & Shoulders",
            direction="bearish",
            start_date="2026-04-01",
            end_date="2026-04-25",
            neckline=420.0,
            target_price=390.0,
            stop_loss=430.0,
        ),
    ]

    best = select_best_pattern(patterns)

    assert best is not None
    assert best["pattern_name"] == "Head & Shoulders"
    assert best["entry_price"] == 420.0
    assert best["direction"] == "bearish"


def test_evaluate_best_setup_rejects_low_confidence_candidate():
    result = evaluate_best_setup(
        rank_patterns([
            _base_pattern(
                pattern_name="Double Bottom",
                timeframe="1m",
                weight=1,
                target_price=578.0,
                stop_loss=572.0,
            )
        ]),
        candle_count=120,
    )

    assert result["setup_available"] is False
    assert result["status"] == "NO_SETUP"
    assert result["reason_code"] == "LOW_CONFIDENCE"
    assert result["has_detected_pattern"] is True
    assert result["confidence_ok"] is False


def test_evaluate_best_setup_rejects_conflicting_actionable_signals():
    patterns = rank_patterns([
        _base_pattern(
            pattern_name="Bull Flag",
            start_date="2026-05-01",
            end_date="2026-05-20",
            entry_price=575.0,
            neckline=None,
            target_price=620.0,
            stop_loss=560.0,
        ),
        _base_pattern(
            pattern_name="Head & Shoulders",
            direction="bearish",
            start_date="2026-06-01",
            end_date="2026-06-25",
            timeframe="1d",
            weight=3,
            neckline=420.0,
            entry_price=420.0,
            target_price=390.0,
            stop_loss=430.0,
            key_levels=[
                {"date": "2026-06-01", "price": 430.0},
                {"date": "2026-06-10", "price": 420.0},
                {"date": "2026-06-25", "price": 429.0},
            ],
        ),
    ])

    result = evaluate_best_setup(patterns, candle_count=320, timeframe="1d")

    assert result["setup_available"] is False
    assert result["status"] == "NO_SETUP"
    assert result["reason_code"] == "CONFLICTING_SIGNALS"
    assert result["no_conflicting_filters"] is False
    assert "Head & Shoulders" in result["conflicting_pattern_names"] or "Bull Flag" in result["conflicting_pattern_names"]


def test_evaluate_best_setup_returns_trade_setup_payload():
    patterns = rank_patterns([
        _base_pattern(
            pattern_name="Bull Flag",
            status="confirmed",
            timeframe="1wk",
            weight=4,
            start_date="2026-07-01",
            end_date="2026-08-20",
            entry_price=575.0,
            neckline=None,
            target_price=650.0,
            stop_loss=550.0,
        )
    ])

    result = evaluate_best_setup(patterns, candle_count=320, timeframe="1wk")

    assert result["setup_available"] is True
    assert result["status"] == "VALID_SETUP"
    assert result["reason_code"] == "VALID_SETUP"
    assert result["best_setup"] is not None
    assert result["best_setup"]["pattern_name"] == "Bull Flag"
    assert result["best_setup"]["timeframe"] == "1wk"
    assert result["best_setup"]["entry_price"] == 575.0
    assert result["best_setup"]["primary_target"] == 650.0
    assert result["best_setup"]["risk_reward_ratio"] == 3.0
    assert len(result["best_setup"]["secondary_targets"]) == 2


def test_evaluate_best_setup_marks_bullish_setup_completed_when_current_price_above_target():
    market_context = {"close": 152.51}
    patterns = rank_patterns([
        _base_pattern(
            pattern_name="Double Bottom",
            status="forming",
            neckline=54.39,
            entry_price=54.39,
            target_price=70.08,
            stop_loss=45.74,
        )
    ], market_context=market_context)

    result = evaluate_best_setup(
        patterns,
        candle_count=320,
        timeframe="1d",
        market_context=market_context,
    )

    assert result["setup_available"] is False
    assert result["status"] == "NO_SETUP"
    assert result["reason_code"] == "SETUP_COMPLETED"
    assert result["levels_ok"] is True
    assert result["price_relevance_ok"] is False
    assert result["candidate_relevance_status"] == "completed"
    assert result["best_setup"] is None


def test_evaluate_best_setup_rejects_bullish_entry_too_far_from_current_price():
    market_context = {"close": 152.51}
    patterns = rank_patterns([
        _base_pattern(
            pattern_name="Bull Flag",
            status="confirmed",
            entry_price=125.0,
            neckline=None,
            target_price=180.0,
            stop_loss=115.0,
        )
    ], market_context=market_context)

    result = evaluate_best_setup(
        patterns,
        candle_count=320,
        timeframe="1d",
        market_context=market_context,
    )

    assert result["setup_available"] is False
    assert result["reason_code"] == "SETUP_STALE"
    assert result["price_relevance_ok"] is False
    assert result["candidate_relevance_status"] == "stale"
    assert result["candidate_entry_distance_pct"] > 15.0
