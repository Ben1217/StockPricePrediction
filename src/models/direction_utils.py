"""
Shared helpers for the simplified next-day direction pipeline.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

NEXT_DAY_HORIZON = 1
BUY_PROBABILITY_THRESHOLD = 0.55
SELL_PROBABILITY_THRESHOLD = 0.45


def normalize_supported_horizons(_: Iterable[int] | None = None) -> list[int]:
    """The simplified pipeline only supports next-day direction prediction."""
    return [NEXT_DAY_HORIZON]


def probability_up(proba) -> np.ndarray:
    values = np.asarray(proba)
    if values.ndim == 1:
        return values.astype(np.float32)
    if values.shape[1] < 2:
        raise ValueError("Binary classifier probabilities must include two columns")
    return values[:, 1].astype(np.float32)


def confidence_from_probability(prob_up: float) -> float:
    return float(max(prob_up, 1.0 - prob_up) * 100.0)


def direction_from_probability(prob_up: float) -> str:
    return "Bullish" if float(prob_up) >= 0.5 else "Bearish"


def expected_move_from_probability(prob_up: float) -> str:
    return "up" if float(prob_up) >= 0.5 else "down"


def signal_from_probability(prob_up: float) -> str:
    value = float(prob_up)
    if value >= BUY_PROBABILITY_THRESHOLD:
        return "BUY"
    if value <= SELL_PROBABILITY_THRESHOLD:
        return "SELL"
    return "HOLD"


def simple_long_flat_backtest(prob_up, forward_returns) -> Dict[str, float]:
    """
    Evaluate a simple long/flat rule on realised next-day returns.

    Rule:
    - `BUY` threshold => long for the next session
    - otherwise stay flat
    """
    probs = np.asarray(prob_up, dtype=np.float32).reshape(-1)
    realised = np.asarray(forward_returns, dtype=np.float32).reshape(-1)

    if len(probs) == 0 or len(realised) == 0:
        return {
            "strategy_return": 0.0,
            "benchmark_return": 0.0,
            "trade_days": 0,
            "long_ratio": 0.0,
            "win_rate": 0.0,
        }

    n = min(len(probs), len(realised))
    probs = probs[:n]
    realised = realised[:n]

    active_mask = probs >= BUY_PROBABILITY_THRESHOLD
    strategy_returns = np.where(active_mask, realised, 0.0)
    win_rate = float(np.mean(strategy_returns[active_mask] > 0)) if np.any(active_mask) else 0.0

    return {
        "strategy_return": float(np.prod(1.0 + strategy_returns) - 1.0),
        "benchmark_return": float(np.prod(1.0 + realised) - 1.0),
        "trade_days": int(np.sum(active_mask)),
        "long_ratio": float(np.mean(active_mask)),
        "win_rate": win_rate,
    }
