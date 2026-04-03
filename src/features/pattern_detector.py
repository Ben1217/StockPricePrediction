"""
Chart Pattern Detection Module (Multi-Timeframe Spec)

Detects 4 specific patterns with strict coordinate and targets logic:
  - Head & Shoulders (Bearish)
  - Double Bottom (Bullish)
  - Bull Flag (Bullish)
  - Symmetrical Triangle (Neutral / Breakout)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

from .technical_indicators import add_all_technical_indicators
from ..utils.logger import get_logger

logger = get_logger(__name__)

_PATTERN_BASE_CONFIDENCE = {
    "Bull Flag": 73.0,
    "Head & Shoulders": 70.0,
    "Double Bottom": 62.0,
    "Symmetrical Triangle": 58.0,
}
_TIMEFRAME_CONFIDENCE_BONUS = {
    "1m": 0.0,
    "1h": 1.5,
    "1d": 3.0,
    "1wk": 5.0,
    "1mo": 6.5,
}
_STATUS_CONFIDENCE_BONUS = {
    "forming": 0.0,
    "confirmed": 3.0,
    "broken": -20.0,
}
_STATUS_SORT_ORDER = {
    "broken": 0,
    "forming": 1,
    "confirmed": 2,
}
_PATTERN_QUALITY_PRIORS = {
    "Bull Flag": 0.84,
    "Head & Shoulders": 0.8,
    "Double Bottom": 0.74,
    "Symmetrical Triangle": 0.68,
}
_TIMEFRAME_REQUIRED_CANDLES = {
    "1m": 120,
    "1h": 160,
    "1d": 260,
    "1wk": 280,
    "1mo": 150,
}
MIN_DECISION_CANDLES = 120
MIN_SETUP_CONFIDENCE = 65.0
MIN_SETUP_RISK_REWARD = 1.5


# ── Pivot Detection ─────────────────────────────────────────────

def _find_pivots(df: pd.DataFrame, window: int = 5) -> tuple:
    """Find local high/low pivot points using rolling window."""
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    pivot_highs = []
    pivot_lows = []

    for i in range(window, n - window):
        if highs[i] == max(highs[i - window : i + window + 1]):
            pivot_highs.append(i)
        if lows[i] == min(lows[i - window : i + window + 1]):
            pivot_lows.append(i)

    return pivot_highs, pivot_lows


def _date_str(df: pd.DataFrame, idx: int) -> str:
    dt = df.index[idx]
    if hasattr(dt, "date") and dt.time() == dt.time().replace(hour=0, minute=0, second=0):
        return str(dt.date())
    return str(dt)


def _price_at(df: pd.DataFrame, idx: int, col: str = "Close") -> float:
    return float(df.iloc[idx][col])


def _make_key_level(df, idx, price=None):
    return {"date": _date_str(df, idx), "price": price or _price_at(df, idx)}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def get_required_candles(timeframe: Optional[str]) -> int:
    return _TIMEFRAME_REQUIRED_CANDLES.get(timeframe or "", MIN_DECISION_CANDLES)


def _derive_entry_price(pattern: Dict[str, Any]) -> Optional[float]:
    """Prefer explicit entry price, then breakout, then neckline."""
    for field in ("entry_price", "breakout_price", "neckline"):
        value = pattern.get(field)
        if value is not None:
            return round(float(value), 2)
    return None


def _is_actionable_pattern(pattern: Dict[str, Any]) -> bool:
    return (
        pattern.get("direction") in {"bullish", "bearish"}
        and pattern.get("status") != "broken"
        and _derive_entry_price(pattern) is not None
        and pattern.get("target_price") is not None
        and pattern.get("stop_loss") is not None
    )


def calculate_risk_reward(pattern: Dict[str, Any]) -> Optional[float]:
    """Return reward/risk for actionable patterns."""
    if not _is_actionable_pattern(pattern):
        return None

    entry_price = _derive_entry_price(pattern)
    target_price = pattern.get("target_price")
    stop_loss = pattern.get("stop_loss")
    if entry_price is None or target_price is None or stop_loss is None:
        return None

    risk = abs(float(entry_price) - float(stop_loss))
    reward = abs(float(target_price) - float(entry_price))
    if risk <= 0:
        return None
    return reward / risk


def build_market_context(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Build a lightweight indicator snapshot from the full history."""
    if df is None or df.empty:
        return {}

    try:
        enriched = add_all_technical_indicators(df.copy())
        latest = enriched.iloc[-1]
        return {
            "close": _safe_float(latest.get("Close")),
            "sma_20": _safe_float(latest.get("SMA_20")),
            "sma_50": _safe_float(latest.get("SMA_50")),
            "sma_200": _safe_float(latest.get("SMA_200")),
            "ema_12": _safe_float(latest.get("EMA_12")),
            "ema_26": _safe_float(latest.get("EMA_26")),
            "rsi": _safe_float(latest.get("RSI")),
            "macd": _safe_float(latest.get("MACD")),
            "macd_signal": _safe_float(latest.get("MACD_Signal")),
            "macd_histogram": _safe_float(latest.get("MACD_Histogram")),
            "adx": _safe_float(latest.get("ADX")),
            "volume_ratio": _safe_float(latest.get("Volume_Ratio")),
            "cmf": _safe_float(latest.get("CMF")),
            "history_bars": int(len(enriched)),
        }
    except Exception as exc:
        logger.warning(f"Failed to build market context: {exc}")
        latest = df.iloc[-1]
        return {
            "close": _safe_float(latest.get("Close")),
            "history_bars": int(len(df)),
        }


def _score_ml_probability(pattern: Dict[str, Any]) -> float:
    """Heuristic proxy for ML conviction used by the decision score."""
    base = _PATTERN_BASE_CONFIDENCE.get(pattern.get("pattern_name"), 60.0)
    timeframe_bonus = _TIMEFRAME_CONFIDENCE_BONUS.get(
        pattern.get("timeframe"),
        max(float(pattern.get("weight", 1)) - 1.0, 0.0) * 1.5,
    )
    status_bonus = _STATUS_CONFIDENCE_BONUS.get(pattern.get("status"), 0.0)
    actionable_bonus = 3.0 if _is_actionable_pattern(pattern) else -6.0

    probability = base + timeframe_bonus + status_bonus + actionable_bonus

    risk_reward = calculate_risk_reward(pattern)
    if risk_reward is not None:
        probability += min(max(risk_reward - 1.0, 0.0) * 8.0, 14.0)

    return round(_clamp(probability / 100.0, 0.35, 0.95), 4)


def _score_pattern_quality(pattern: Dict[str, Any]) -> float:
    prior = _PATTERN_QUALITY_PRIORS.get(pattern.get("pattern_name"), 0.7)
    status_score = {
        "confirmed": 0.95,
        "forming": 0.78,
        "broken": 0.12,
    }.get(pattern.get("status"), 0.6)
    actionable_score = 1.0 if _is_actionable_pattern(pattern) else 0.35
    risk_reward = calculate_risk_reward(pattern)
    rr_score = 0.35 if risk_reward is None else _clamp(risk_reward / 3.0)
    quality = (
        0.35 * prior
        + 0.25 * status_score
        + 0.2 * actionable_score
        + 0.2 * rr_score
    )
    return round(_clamp(quality), 4)


def _score_indicator_alignment(pattern: Dict[str, Any], market_context: Optional[Dict[str, Any]]) -> float:
    if not market_context:
        return 0.5

    direction = pattern.get("direction")
    if direction not in {"bullish", "bearish"}:
        return 0.5

    bullish = direction == "bullish"
    close = market_context.get("close")
    scores: List[float] = []

    for key in ("sma_20", "sma_50", "sma_200"):
        baseline = market_context.get(key)
        if close is not None and baseline is not None:
            scores.append(1.0 if ((close >= baseline) == bullish) else 0.15)

    ema_12 = market_context.get("ema_12")
    ema_26 = market_context.get("ema_26")
    if ema_12 is not None and ema_26 is not None:
        scores.append(0.9 if ((ema_12 >= ema_26) == bullish) else 0.2)

    rsi = market_context.get("rsi")
    if rsi is not None:
        if bullish:
            if 52 <= rsi <= 72:
                scores.append(1.0)
            elif 45 <= rsi < 52 or 72 < rsi <= 78:
                scores.append(0.65)
            else:
                scores.append(0.2)
        else:
            if 28 <= rsi <= 48:
                scores.append(1.0)
            elif 48 < rsi <= 55 or 22 <= rsi < 28:
                scores.append(0.65)
            else:
                scores.append(0.2)

    macd_histogram = market_context.get("macd_histogram")
    if macd_histogram is not None:
        scores.append(0.9 if ((macd_histogram >= 0) == bullish) else 0.2)

    adx = market_context.get("adx")
    if adx is not None:
        scores.append(1.0 if adx >= 20 else 0.55)

    if not scores:
        return 0.5
    return round(sum(scores) / len(scores), 4)


def _score_volume_confirmation(pattern: Dict[str, Any], market_context: Optional[Dict[str, Any]]) -> float:
    if not market_context:
        return 0.5

    scores: List[float] = []
    volume_ratio = market_context.get("volume_ratio")
    if volume_ratio is not None:
        scores.append(_clamp((volume_ratio - 0.75) / 0.75))

    cmf = market_context.get("cmf")
    direction = pattern.get("direction")
    if cmf is not None and direction in {"bullish", "bearish"}:
        bullish = direction == "bullish"
        scores.append(0.9 if ((cmf >= 0) == bullish) else 0.25)

    if not scores:
        return 0.5
    return round(sum(scores) / len(scores), 4)


def _derive_secondary_targets(pattern: Dict[str, Any]) -> List[float]:
    entry_price = _derive_entry_price(pattern)
    primary_target = _safe_float(pattern.get("target_price"))
    if entry_price is None or primary_target is None:
        return []

    reward_distance = abs(primary_target - entry_price)
    if reward_distance <= 0:
        return []

    sign = 1 if primary_target >= entry_price else -1
    derived: List[float] = []
    for multiplier in (1.5, 2.0):
        target = round(entry_price + sign * reward_distance * multiplier, 2)
        if target != round(primary_target, 2) and target not in derived:
            derived.append(target)
    return derived


def _derive_strength_label(confidence: Optional[float], risk_reward: Optional[float]) -> str:
    conf = confidence or 0.0
    rr = risk_reward or 0.0
    if conf >= 80.0 and rr >= 2.0:
        return "Strong"
    if conf >= 65.0 and rr >= 1.5:
        return "Moderate"
    return "Weak"


def _build_setup_action(pattern: Dict[str, Any]) -> str:
    entry_price = _derive_entry_price(pattern)
    if entry_price is None:
        return "Wait for confirmation"
    if pattern.get("direction") == "bullish":
        return f"Buy above {entry_price:.2f}"
    if pattern.get("direction") == "bearish":
        return f"Sell below {entry_price:.2f}"
    return "Wait for breakout"


def build_best_trade_setup(pattern: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pattern or not _is_actionable_pattern(pattern):
        return None

    entry_price = _derive_entry_price(pattern)
    stop_loss = _safe_float(pattern.get("stop_loss"))
    primary_target = _safe_float(pattern.get("target_price"))
    risk_reward = _safe_float(pattern.get("risk_reward_ratio"))
    if risk_reward is None:
        risk_reward = calculate_risk_reward(pattern)
    confidence = _safe_float(pattern.get("confidence"))

    if entry_price is None or stop_loss is None or primary_target is None:
        return None

    target_move_pct = None
    if entry_price > 0:
        target_move_pct = round(abs(primary_target - entry_price) / entry_price * 100.0, 2)

    secondary_targets = _derive_secondary_targets(pattern)
    strength_label = _derive_strength_label(confidence, risk_reward)
    return {
        "pattern_name": pattern.get("pattern_name"),
        "timeframe": pattern.get("timeframe"),
        "direction": pattern.get("direction"),
        "pattern_status": pattern.get("status"),
        "confidence_score": round(confidence, 1) if confidence is not None else None,
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "primary_target": round(primary_target, 2),
        "secondary_targets": secondary_targets,
        "risk_reward_ratio": round(risk_reward, 2) if risk_reward is not None else None,
        "strength_label": strength_label,
        "target_move_pct": target_move_pct,
        "action": _build_setup_action(pattern),
        "score_components": dict(pattern.get("score_components") or {}),
    }


def rank_patterns(
    patterns: List[Dict[str, Any]],
    market_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Enrich detected patterns with decision fields and sort strongest-first."""
    ranked: List[Dict[str, Any]] = []

    for pattern in patterns:
        enriched = dict(pattern)
        enriched["entry_price"] = _derive_entry_price(enriched)
        risk_reward = calculate_risk_reward(enriched)
        ml_probability = _score_ml_probability(enriched)
        pattern_quality = _score_pattern_quality(enriched)
        indicator_alignment = _score_indicator_alignment(enriched, market_context)
        volume_confirmation = _score_volume_confirmation(enriched, market_context)
        composite_score = (
            0.4 * ml_probability
            + 0.3 * pattern_quality
            + 0.2 * indicator_alignment
            + 0.1 * volume_confirmation
        )
        confidence = round(composite_score * 100.0, 1)

        enriched["risk_reward_ratio"] = round(risk_reward, 2) if risk_reward is not None else None
        enriched["confidence"] = confidence
        enriched["strength_label"] = _derive_strength_label(confidence, risk_reward)
        enriched["secondary_targets"] = _derive_secondary_targets(enriched)
        enriched["score_components"] = {
            "ml_probability": round(ml_probability, 4),
            "pattern_quality": round(pattern_quality, 4),
            "indicator_alignment": round(indicator_alignment, 4),
            "volume_confirmation": round(volume_confirmation, 4),
            "composite_score": round(composite_score, 4),
        }
        ranked.append(enriched)

    ranked.sort(
        key=lambda p: (
            1 if _is_actionable_pattern(p) else 0,
            p.get("score_components", {}).get("composite_score", 0.0),
            p.get("confidence", 0.0),
            _STATUS_SORT_ORDER.get(p.get("status"), 0),
            p.get("weight", 0),
            p.get("end_date", ""),
        ),
        reverse=True,
    )
    return ranked


def select_best_pattern(
    patterns: List[Dict[str, Any]],
    market_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Select the single best decision-worthy setup, preferring actionable trades."""
    if not patterns:
        return None

    needs_rerank = (
        market_context is not None
        or "confidence" not in patterns[0]
        or "score_components" not in patterns[0]
    )
    ranked = rank_patterns(patterns, market_context=market_context) if needs_rerank else patterns
    actionable = [pattern for pattern in ranked if _is_actionable_pattern(pattern)]
    candidates = actionable or ranked
    return dict(candidates[0]) if candidates else None


def evaluate_best_setup(
    patterns: List[Dict[str, Any]],
    candle_count: int,
    timeframe: Optional[str] = None,
    market_context: Optional[Dict[str, Any]] = None,
    min_confidence: float = MIN_SETUP_CONFIDENCE,
    min_risk_reward: float = MIN_SETUP_RISK_REWARD,
    min_candles: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply best-setup decision rules and explain rejection reasons."""
    resolved_timeframe = timeframe or (patterns[0].get("timeframe") if patterns else None)
    required_candles = min_candles or get_required_candles(resolved_timeframe)
    needs_rerank = (
        market_context is not None
        or not patterns
        or "confidence" not in patterns[0]
        or "score_components" not in patterns[0]
    )
    ranked = rank_patterns(patterns, market_context=market_context) if needs_rerank else patterns
    candidate = select_best_pattern(ranked)
    candidate_risk_reward = calculate_risk_reward(candidate) if candidate else None
    candidate_setup = build_best_trade_setup(candidate)

    sufficient_data = candle_count >= required_candles
    has_detected_pattern = len(ranked) > 0
    levels_ok = bool(candidate and _is_actionable_pattern(candidate))
    confidence_ok = bool(candidate and float(candidate.get("confidence", 0.0)) >= min_confidence)
    risk_reward_ok = bool(candidate_risk_reward is not None and candidate_risk_reward >= min_risk_reward)

    conflicting_patterns: List[str] = []
    if candidate and candidate.get("direction") in {"bullish", "bearish"}:
        candidate_name = candidate.get("pattern_name")
        candidate_direction = candidate.get("direction")
        for pattern in ranked:
            if pattern.get("pattern_name") == candidate_name and pattern.get("start_date") == candidate.get("start_date"):
                continue
            if pattern.get("direction") == candidate_direction:
                continue
            if pattern.get("direction") not in {"bullish", "bearish"}:
                continue
            if not _is_actionable_pattern(pattern):
                continue
            if float(pattern.get("confidence", 0.0)) < min_confidence:
                continue
            name = pattern.get("pattern_name")
            if name and name not in conflicting_patterns:
                conflicting_patterns.append(name)

    no_conflicting_filters = len(conflicting_patterns) == 0

    if not sufficient_data:
        reason_code = "INSUFFICIENT_DATA"
        reason = "Insufficient candles/data"
    elif not has_detected_pattern:
        reason_code = "NO_PATTERN"
        reason = "No pattern detected"
    elif not confidence_ok:
        reason_code = "LOW_CONFIDENCE"
        reason = "Confidence below threshold"
    elif not levels_ok:
        reason_code = "INVALID_LEVELS"
        reason = "Missing entry / stop / target"
    elif not risk_reward_ok:
        reason_code = "LOW_RR"
        reason = "Risk/reward too weak"
    elif not no_conflicting_filters:
        reason_code = "CONFLICTING_SIGNALS"
        reason = "Conflicting signals"
    else:
        reason_code = "VALID_SETUP"
        reason = "Best setup ready"

    setup_available = reason_code == "VALID_SETUP"
    return {
        "status": "VALID_SETUP" if setup_available else "NO_SETUP",
        "setup_available": setup_available,
        "reason_code": reason_code,
        "reason": reason,
        "sufficient_data": sufficient_data,
        "has_detected_pattern": has_detected_pattern,
        "confidence_ok": confidence_ok,
        "levels_ok": levels_ok,
        "risk_reward_ok": risk_reward_ok,
        "no_conflicting_filters": no_conflicting_filters,
        "candle_count": candle_count,
        "min_candles": required_candles,
        "min_confidence": min_confidence,
        "min_risk_reward": min_risk_reward,
        "candidate_pattern_name": candidate.get("pattern_name") if candidate else None,
        "candidate_confidence": float(candidate.get("confidence")) if candidate and candidate.get("confidence") is not None else None,
        "candidate_risk_reward": round(candidate_risk_reward, 2) if candidate_risk_reward is not None else None,
        "candidate_strength_label": candidate_setup.get("strength_label") if candidate_setup else None,
        "conflicting_pattern_names": conflicting_patterns,
        "best_setup": candidate_setup if setup_available else None,
        "best_pattern": dict(candidate) if setup_available and candidate else None,
    }


# ── Pattern Matchers ────────────────────────────────────────────

def _detect_head_shoulders(df, pivot_highs, tolerance=0.03):
    """Detect Head & Shoulders pattern (Bearish)."""
    patterns = []
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)

    for i in range(len(pivot_highs) - 2):
        ls, hd, rs = pivot_highs[i], pivot_highs[i + 1], pivot_highs[i + 2]
        h_ls, h_hd, h_rs = highs[ls], highs[hd], highs[rs]

        # Head must be highest
        if h_hd <= h_ls or h_hd <= h_rs:
            continue
        # Shoulders roughly equal
        if abs(h_ls - h_rs) / max(h_ls, h_rs) > tolerance:
            continue

        # Neckline from troughs between peaks
        t1_idx = ls + np.argmin(lows[ls:hd])
        t2_idx = hd + np.argmin(lows[hd:rs])
        neckline = (float(lows[t1_idx]) + float(lows[t2_idx])) / 2
        
        target = neckline - (h_hd - neckline)
        stop_loss = h_rs

        last_close = closes[min(rs + 5, len(df) - 1)]
        status = "forming"
        breakout_price = None
        
        if last_close < neckline:
            status = "confirmed"
            breakout_price = neckline

        patterns.append({
            "pattern_name": "Head & Shoulders",
            "direction": "bearish",
            "start_date": _date_str(df, ls),
            "end_date": _date_str(df, rs),
            "key_levels": [
                _make_key_level(df, ls, h_ls),
                _make_key_level(df, t1_idx, float(lows[t1_idx])),
                _make_key_level(df, hd, h_hd),
                _make_key_level(df, t2_idx, float(lows[t2_idx])),
                _make_key_level(df, rs, h_rs),
            ],
            "neckline": round(neckline, 2),
            "entry_price": round(neckline, 2),
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })

    return patterns


def _detect_double_bottom(df, pivot_lows, tolerance=0.02):
    """Detect Double Bottom pattern (Bullish)."""
    patterns = []
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)

    for i in range(len(pivot_lows) - 1):
        p1, p2 = pivot_lows[i], pivot_lows[i + 1]
        if p2 - p1 < 5:
            continue

        l1, l2 = lows[p1], lows[p2]
        if abs(l1 - l2) / max(l1, l2) > tolerance:
            continue

        peak_idx = p1 + np.argmax(highs[p1:p2])
        resistance = float(highs[peak_idx])
        trough_avg = (l1 + l2) / 2
        height = resistance - trough_avg
        
        target = resistance + height
        stop_loss = trough_avg * 0.995 # lightly below

        last_close = closes[min(p2 + 5, len(df) - 1)]
        status = "forming"
        breakout_price = None

        if last_close > resistance:
            status = "confirmed"
            breakout_price = resistance

        patterns.append({
            "pattern_name": "Double Bottom",
            "direction": "bullish",
            "start_date": _date_str(df, p1),
            "end_date": _date_str(df, p2),
            "key_levels": [
                _make_key_level(df, p1, l1),
                _make_key_level(df, peak_idx, resistance),
                _make_key_level(df, p2, l2),
            ],
            "neckline": round(resistance, 2),
            "entry_price": round(resistance, 2),
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })

    return patterns


def _detect_bull_flag(df, pivot_highs, pivot_lows):
    """Detect Bull Flag pattern (Bullish)."""
    patterns = []
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    n = len(df)

    for i in range(20, n - 5):
        # Look for a sharp bullish move (flagpole)
        lookback = 15
        start = max(0, i - lookback)
        move = (closes[i] - closes[start]) / closes[start]

        if move < 0.05: # Need > 5% move for pole
            continue

        pole_length = highs[start:i].max() - lows[start:i].min()

        # Check for consolidation channel
        flag_end = min(i + 15, n)
        
        # Fit trendlines for the flag
        flag_highs = highs[i:flag_end]
        flag_lows = lows[i:flag_end]
        
        if len(flag_highs) < 5: continue
        
        upper_channel = np.polyfit(np.arange(len(flag_highs)), flag_highs, 1)[0]
        lower_channel = np.polyfit(np.arange(len(flag_lows)), flag_lows, 1)[0]
        
        # Upper and lower should be sloping down or flat
        if upper_channel > 0.01 or lower_channel > 0.01:
             continue
             
        breakout = float(flag_highs.max())
        flag_bottom = float(flag_lows.min())

        target = breakout + pole_length
        stop_loss = flag_bottom

        status = "forming"
        breakout_p = None
        if closes[flag_end - 1] > breakout:
            status = "confirmed"
            breakout_p = breakout

        patterns.append({
            "pattern_name": "Bull Flag",
            "direction": "bullish",
            "start_date": _date_str(df, start),
            "end_date": _date_str(df, min(flag_end - 1, n - 1)),
            "key_levels": [
                _make_key_level(df, start, closes[start]),
                _make_key_level(df, i, highs[i]),
                _make_key_level(df, min(flag_end - 1, n - 1), closes[flag_end - 1]),
            ],
            "trendlines": [
                [_make_key_level(df, i, highs[i]), _make_key_level(df, flag_end - 1, flag_highs[-1])],
                [_make_key_level(df, i, lows[i]), _make_key_level(df, flag_end - 1, flag_lows[-1])]
            ],
            "neckline": None,
            "entry_price": round(breakout, 2),
            "breakout_price": round(breakout_p, 2) if breakout_p else None,
            "target_price": round(target, 2),
            "stop_loss": round(stop_loss, 2),
            "status": status,
        })
        break # Only report the most recent

    return patterns


def _detect_symmetrical_triangle(df, pivot_highs, pivot_lows):
    """Detect Symmetrical Triangle (Neutral until breakout)."""
    patterns = []
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    closes = df["Close"].values.astype(float)
    n = len(df)

    for start_idx in range(0, n - 20, 10):
        end_idx = min(start_idx + 60, n)
        
        # Get pivots strictly within timeline window
        wh = [p for p in pivot_highs if start_idx <= p < end_idx]
        wl = [p for p in pivot_lows if start_idx <= p < end_idx]

        if len(wh) < 2 or len(wl) < 2:
            continue

        h_vals = highs[wh]
        l_vals = lows[wl]

        h_slope = np.polyfit(wh, h_vals, 1)[0]
        l_slope = np.polyfit(wl, l_vals, 1)[0]

        # Both must converge: Upper trendline falls, lower rises
        if not (h_slope < -0.01 and l_slope > 0.01):
            continue

        height = h_vals[0] - l_vals[0]
        
        last_idx = max(max(wh), max(wl))
        last_close = closes[min(n - 1, last_idx + 5)]
        
        # Check breakout
        # Project lines
        proj_h = np.polyval(np.polyfit(wh, h_vals, 1), min(n-1, last_idx + 5))
        proj_l = np.polyval(np.polyfit(wl, l_vals, 1), min(n-1, last_idx + 5))
        
        status = "forming"
        direction = "neutral"
        target = None
        stop_loss = None
        breakout_price = None
        
        if last_close > proj_h:
            status = "confirmed"
            direction = "bullish"
            breakout_price = proj_h
            target = breakout_price + height
            stop_loss = proj_l # inside triangle
        elif last_close < proj_l:
            status = "confirmed"
            direction = "bearish"
            breakout_price = proj_l
            target = breakout_price - height
            stop_loss = proj_h # inside triangle

        key_levels = []
        for idx in sorted(list(wh) + list(wl)):
            key_levels.append(_make_key_level(df, idx, float(closes[idx])))

        trendlines = [
            [_make_key_level(df, wh[0], h_vals[0]), _make_key_level(df, wh[-1], h_vals[-1])],
            [_make_key_level(df, wl[0], l_vals[0]), _make_key_level(df, wl[-1], l_vals[-1])]
        ]

        patterns.append({
            "pattern_name": "Symmetrical Triangle",
            "direction": direction,
            "start_date": _date_str(df, min(min(wh), min(wl))),
            "end_date": _date_str(df, last_idx),
            "key_levels": key_levels[:8],
            "trendlines": trendlines,
            "neckline": None,
            "entry_price": round(breakout_price, 2) if breakout_price else None,
            "breakout_price": round(breakout_price, 2) if breakout_price else None,
            "target_price": round(target, 2) if target else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "status": status,
        })

    return patterns


# ── Main Entry Point ────────────────────────────────────────────

def detect_chart_patterns(df: pd.DataFrame, lookback: int = 120, timeframe: str = "1d", weight: int = 1) -> List[Dict[str, Any]]:
    """
    Detect exactly 4 chart patterns.
    """
    data = df.tail(lookback).copy()
    if len(data) < 30:
        return []

    pivot_highs, pivot_lows = _find_pivots(data, window=5)
    all_patterns = []

    try:
        all_patterns.extend(_detect_head_shoulders(data, pivot_highs))
    except Exception as e:
        logger.warning(f"H&S detection failed: {e}")

    try:
        all_patterns.extend(_detect_double_bottom(data, pivot_lows))
    except Exception as e:
        logger.warning(f"Double bottom detection failed: {e}")

    try:
        all_patterns.extend(_detect_bull_flag(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Flag detection failed: {e}")
        
    try:
        all_patterns.extend(_detect_symmetrical_triangle(data, pivot_highs, pivot_lows))
    except Exception as e:
        logger.warning(f"Symmetrical triangle failed: {e}")

    # Deduplicate overlapping patterns
    seen = set()
    unique = []
    
    # Enrich with weight and timeframe
    for p in all_patterns:
        key = (p["pattern_name"], p["start_date"])
        if key not in seen:
            seen.add(key)
            p['timeframe'] = timeframe
            p['weight'] = weight
            unique.append(p)

    return rank_patterns(unique)
