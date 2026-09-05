"""
Shared helpers for the simplified next-day direction pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from src.defaults import ENFORCE_MODEL_SKILL_ENV
from src.utils.config_loader import get_env_bool

NEXT_DAY_HORIZON = 1
BUY_PROBABILITY_THRESHOLD = 0.55
SELL_PROBABILITY_THRESHOLD = 0.45

# A direction bundle earns the right to be served by beating the coin flip out of
# sample. Zero means "AUC strictly above 0.5" -- the direct analogue of
# MIN_SKILL_SCORE for the regression bundles.
MIN_DIRECTION_SKILL_SCORE = 0.0

# The companion check, and the one AUC cannot make on its own. A classifier that
# collapses onto the base rate emits the same probability for every bar; it scores
# an unremarkable AUC while being incapable of ever crossing the BUY/SELL bands.
# The bands sit 0.05 either side of 0.5, so a spread this narrow cannot reach them
# from the middle no matter which way it leans.
MIN_PROBABILITY_STD = 0.01


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


def direction_skill_record(y_true, prob_up) -> Dict[str, float]:
    """
    Score a direction classifier against the coin flip, out of sample.

    ``skill_score`` is ROC-AUC minus 0.5: positive means the ranking carries
    information, zero means it is indistinguishable from guessing, negative means
    it is actively worse. It is the direction-side counterpart of
    :func:`src.models.ensemble_training._baseline_skill`, and it exists for the
    same reason -- a bundle has to carry the evidence for its own serving.

    ``probability_std`` is recorded alongside it because AUC alone cannot see a
    collapsed model. A classifier that has learned nothing but the base rate
    returns one number for every bar; its AUC lands wherever the tie-breaking
    falls, while the spread goes to zero and the signal bands become unreachable.
    """
    y_true = np.asarray(y_true).reshape(-1)
    probs = np.asarray(prob_up, dtype=np.float64).reshape(-1)
    if y_true.size == 0 or probs.size == 0:
        return {
            "roc_auc": 0.5,
            "skill_score": 0.0,
            "probability_std": 0.0,
            "positive_rate": 0.0,
        }

    n = min(len(y_true), len(probs))
    y_true, probs = y_true[:n], probs[:n]

    # A split that landed on one class carries no ranking to score. That is a
    # property of the sample rather than of the model, so it is reported as "no
    # evidence" (0.5) rather than as a failure the model earned.
    if len(np.unique(y_true)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_true, probs))

    return {
        "roc_auc": round(auc, 6),
        "skill_score": round(auc - 0.5, 6),
        "probability_std": round(float(np.std(probs)), 6),
        "positive_rate": round(float(np.mean(probs >= BUY_PROBABILITY_THRESHOLD)), 6),
    }


def direction_skill_passes(record: Dict[str, float]) -> bool:
    """Whether a :func:`direction_skill_record` clears both gates."""
    return bool(
        float(record.get("skill_score", 0.0)) > MIN_DIRECTION_SKILL_SCORE
        and float(record.get("probability_std", 0.0)) >= MIN_PROBABILITY_STD
    )


def direction_skill_enforcement_enabled() -> bool:
    """Whether direction bundles must prove out-of-sample skill before serving."""
    return get_env_bool(ENFORCE_MODEL_SKILL_ENV, True)


def direction_skill_failure(meta: Dict[str, Any]) -> Optional[str]:
    """
    Return why a direction bundle must not be served, or None when it is fit.

    Mirrors :func:`src.models.ensemble_predictor.bundle_skill_failure`. A bundle
    qualifies by recording ``passes_baseline: true``, which training sets when the
    model beats the coin flip out of sample *and* emits a usable spread of
    probabilities. Bundles trained before the gate existed carry no such record
    and are treated as unproven, because those are exactly the ones that turned
    out to be predicting the base rate.
    """
    if not direction_skill_enforcement_enabled():
        return None

    if "passes_baseline" not in meta:
        return (
            "was trained before out-of-sample direction skill was recorded, so "
            "there is no evidence it beats a coin flip"
        )

    if meta.get("passes_baseline"):
        return None

    record = (meta.get("skill") or {}).get("test") or {}
    auc = record.get("roc_auc")
    spread = record.get("probability_std")

    # The two failures have different remedies, so they are named separately: a
    # collapsed model needs a different architecture or features, while a model
    # that ranks worse than chance needs a different signal altogether.
    if isinstance(spread, (int, float)) and spread < MIN_PROBABILITY_STD:
        detail = f"probability spread {spread:.4f}"
        if isinstance(auc, (int, float)):
            detail += f", ROC-AUC {auc:.4f}"
        return f"predicts the same probability for every bar, so it can never signal ({detail})"

    detail = f"ROC-AUC {auc:.4f}" if isinstance(auc, (int, float)) else "no measured skill"
    if isinstance(spread, (int, float)):
        detail += f", probability spread {spread:.4f}"
    return f"does not beat a coin flip out of sample ({detail})"
