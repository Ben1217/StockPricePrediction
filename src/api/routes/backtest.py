"""
Backtest API routes — run and retrieve backtests.
"""

import uuid
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import BacktestRequest, BacktestResponse
from src.backtesting.backtest_engine import BacktestEngine
from src.defaults import DEFAULT_INDEX_SYMBOL
from src.features.feature_engineering import (
    build_feature_frame,
    build_supervised_dataset,
    create_sequences,
    select_feature_columns,
    transform_feature_frame,
)
from src.features.technical_indicators import add_all_technical_indicators
from src.models.direction_utils import (
    BUY_PROBABILITY_THRESHOLD,
    NEXT_DAY_HORIZON,
    SELL_PROBABILITY_THRESHOLD,
    confidence_from_probability,
    direction_from_probability,
    probability_up,
    signal_from_probability,
)
from src.models.model_bundle import load_model_bundle
from src.models.model_trainer import ModelTrainer
from src.signals.signal_generator import TradingSignalGenerator

router = APIRouter()

# Store results in memory
_backtest_results = {}

STRATEGY_HYBRID = "hybrid_ml_ta"
STRATEGY_TECHNICAL = "technical_only"
STRATEGY_HOLD = "buy_and_hold"
MODEL_TYPES = ["xgboost", "random_forest", "lstm"]
MIN_SIGNAL_HISTORY = 60


def _download_price_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()

    required_cols = {"Open", "High", "Low", "Close", "Volume"}
    if df.empty or not required_cols.issubset(df.columns):
        raise HTTPException(404, f"Insufficient data for {symbol} ({start_date} to {end_date})")

    return df


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, pd.Series):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def _format_patterns(patterns: List[str]) -> str:
    if not patterns:
        return ""
    return ", ".join(p.replace("_", " ").title() for p in patterns)


def _to_signal_value(action: str) -> int:
    action = (action or "HOLD").upper()
    if "BUY" in action:
        return 1
    if "SELL" in action:
        return -1
    return 0


def _classify_direction(prob_up: float) -> str:
    if prob_up >= 0.70:
        return "STRONG_BUY"
    if prob_up >= BUY_PROBABILITY_THRESHOLD:
        return "BUY"
    if prob_up <= 0.30:
        return "STRONG_SELL"
    if prob_up <= SELL_PROBABILITY_THRESHOLD:
        return "SELL"
    return "HOLD"


def _prediction_confidence(prob_up: float) -> float:
    return float(min(0.98, max(0.0, confidence_from_probability(prob_up) / 100.0)))


def _load_feature_columns(feat_df: pd.DataFrame, bundle=None) -> List[str]:
    if bundle is not None and bundle.feature_columns:
        return list(bundle.feature_columns)
    return select_feature_columns(feat_df)


def _load_trained_model(model_type: str, symbol: str, horizon: int = 1):
    try:
        bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=horizon)
    except Exception as exc:  # pragma: no cover - best effort loading
        return None, None, f"Failed to load {model_type} model bundle: {exc}"

    if bundle is None:
        return None, None, f"No trained {model_type} horizon-{horizon} model bundle found for {symbol}"
    return bundle, bundle.version_id, None


def _bundle_uncertainty(bundle) -> float:
    metrics = bundle.metadata.get("metrics", {})
    nested = metrics.get("test", metrics)
    rmse = nested.get("rmse")
    try:
        return max(0.0, float(rmse))
    except Exception:
        return 0.0


def _build_indicator_snapshot(row: pd.Series) -> Dict[str, Any]:
    close = float(row.get("Close", np.nan)) if not pd.isna(row.get("Close", np.nan)) else None
    sma20 = float(row.get("SMA_20", np.nan)) if not pd.isna(row.get("SMA_20", np.nan)) else None
    sma50 = float(row.get("SMA_50", np.nan)) if not pd.isna(row.get("SMA_50", np.nan)) else None
    rsi = float(row.get("RSI", np.nan)) if not pd.isna(row.get("RSI", np.nan)) else None
    return {
        "close": close,
        "rsi": rsi,
        "sma20": sma20,
        "sma50": sma50,
    }


def _compose_technical_reason(tech: Dict[str, Any], indicator_snapshot: Dict[str, Any]) -> str:
    signal = tech.get("signal", "NEUTRAL")
    patterns = tech.get("patterns", [])
    pattern_text = _format_patterns(patterns)
    bits = []
    if signal == "BUY":
        bits.append("Bullish technical confirmation")
    elif signal == "SELL":
        bits.append("Bearish technical confirmation")
    else:
        bits.append("No clear technical setup")
    if pattern_text:
        bits.append(f"patterns: {pattern_text}")

    rsi = indicator_snapshot.get("rsi")
    close = indicator_snapshot.get("close")
    sma20 = indicator_snapshot.get("sma20")
    sma50 = indicator_snapshot.get("sma50")
    if rsi is not None:
        bits.append(f"RSI {rsi:.1f}")
    if close is not None and sma20 is not None:
        bits.append(f"price {'above' if close >= sma20 else 'below'} SMA20")
    if close is not None and sma50 is not None:
        bits.append(f"price {'above' if close >= sma50 else 'below'} SMA50")
    return "; ".join(bits)


def _compose_hybrid_reason(
    tech: Dict[str, Any],
    probability_up: Optional[float],
    indicator_snapshot: Dict[str, Any],
) -> str:
    bits = []
    if probability_up is not None:
        bits.append(
            f"ML probability up {float(probability_up):.1%} "
            f"({direction_from_probability(float(probability_up)).lower()})"
        )

    tech_signal = tech.get("signal", "NEUTRAL")
    pattern_text = _format_patterns(tech.get("patterns", []))
    if tech_signal == "BUY":
        bits.append("bullish technical confirmation")
    elif tech_signal == "SELL":
        bits.append("bearish technical confirmation")
    else:
        bits.append("neutral technical backdrop")
    if pattern_text:
        bits.append(f"patterns: {pattern_text}")

    rsi = indicator_snapshot.get("rsi")
    close = indicator_snapshot.get("close")
    sma20 = indicator_snapshot.get("sma20")
    sma50 = indicator_snapshot.get("sma50")
    if rsi is not None:
        bits.append(f"RSI {rsi:.1f}")
    if close is not None and sma20 is not None:
        bits.append(f"price {'above' if close >= sma20 else 'below'} SMA20")
    if close is not None and sma50 is not None:
        bits.append(f"price {'above' if close >= sma50 else 'below'} SMA50")
    return "; ".join(bits)


def _combine_hybrid_action(tech: Dict[str, Any], prob_up: float) -> Dict[str, Any]:
    tech_signal = tech.get("signal", "NEUTRAL")
    tech_confidence = float(tech.get("confidence", 0.0) or 0.0)
    ml_signal = signal_from_probability(prob_up)
    ml_confidence = confidence_from_probability(prob_up)

    if ml_signal == "BUY" and tech_signal == "BUY":
        action = "STRONG_BUY" if prob_up >= 0.70 and tech_confidence >= 75 else "BUY"
        return {
            "action": action,
            "confidence": (tech_confidence + ml_confidence) / 2,
        }
    if ml_signal == "SELL" and tech_signal == "SELL":
        action = "STRONG_SELL" if prob_up <= 0.30 and tech_confidence >= 75 else "SELL"
        return {
            "action": action,
            "confidence": (tech_confidence + ml_confidence) / 2,
        }
    return {
        "action": "HOLD",
        "confidence": 0.0,
    }


def _shift_for_execution(symbol: str, signals: pd.DataFrame, details: pd.DataFrame):
    shifted_signals = signals.shift(1).fillna(0).astype(int)
    shifted_details = details.shift(1)
    if len(shifted_details.index) > 0:
        shifted_details.at[shifted_details.index[0], symbol] = {
            "action": "HOLD",
            "confidence": 0.0,
            "patterns": [],
            "predicted_return": None,
            "reason": "Waiting for the first fully observed signal.",
        }
    return shifted_signals, shifted_details


def _build_technical_signal_history(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    enriched = add_all_technical_indicators(df.copy())
    generator = TradingSignalGenerator(mode=TradingSignalGenerator.MODE_TECHNICAL_ONLY)
    signals = pd.DataFrame(0, index=df.index, columns=[symbol], dtype=int)
    details = pd.DataFrame(index=df.index, columns=[symbol], dtype=object)

    for idx, date in enumerate(df.index):
        indicator_snapshot = _build_indicator_snapshot(enriched.iloc[idx])
        if idx < MIN_SIGNAL_HISTORY:
            details.at[date, symbol] = {
                "action": "HOLD",
                "confidence": 0.0,
                "patterns": [],
                "predicted_return": None,
                "reason": "Insufficient lookback for technical validation.",
                "technical_details": {"signal": "NEUTRAL", "confidence": 0.0, "patterns": []},
                "indicator_snapshot": indicator_snapshot,
            }
            continue

        window = enriched.iloc[: idx + 1].copy()
        tech = generator._get_technical_signals(window)
        technical_only = generator._create_technical_only_signal(tech)

        signals.at[date, symbol] = _to_signal_value(technical_only.get("action", "HOLD"))
        details.at[date, symbol] = {
            "action": technical_only.get("action", "HOLD"),
            "confidence": float(technical_only.get("confidence", 0.0) or 0.0),
            "patterns": list(tech.get("patterns", []) or []),
            "predicted_return": None,
            "reason": _compose_technical_reason(tech, indicator_snapshot),
            "technical_details": tech,
            "indicator_snapshot": indicator_snapshot,
        }

    exec_signals, exec_details = _shift_for_execution(symbol, signals, details)
    return {
        "signals": exec_signals,
        "details": exec_details,
        "indicator_data": enriched,
    }


def _build_model_prediction_history(symbol: str, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    bundle, model_file, error_message = _load_trained_model(model_type, symbol, horizon=NEXT_DAY_HORIZON)
    if bundle is None:
        return {
            "status": "unavailable",
            "model_type": model_type,
            "message": error_message,
        }
    bundle_target_type = getattr(bundle, "target_type", bundle.metadata.get("target_type", "direction"))
    if bundle_target_type != "direction":
        return {
            "status": "unavailable",
            "model_type": model_type,
            "message": f"{model_type} bundle for {symbol} must be retrained for next-day direction.",
        }

    feat_df = build_feature_frame(df, feature_config=bundle.feature_config)
    if feat_df.empty:
        return {
            "status": "unavailable",
            "model_type": model_type,
            "message": f"Unable to build features for {model_type}.",
        }

    feature_cols = _load_feature_columns(feat_df, bundle=bundle)
    if not feature_cols:
        return {
            "status": "unavailable",
            "model_type": model_type,
            "message": f"No feature columns available for {model_type}.",
        }

    prediction_frame = pd.DataFrame(
        index=df.index,
        columns=["probability_up", "confidence_score", "uncertainty", "directional_signal"],
        dtype=object,
    )

    try:
        aligned, X = transform_feature_frame(feat_df, feature_cols, scaler=bundle.scaler)
        if aligned.empty:
            return {
                "status": "unavailable",
                "model_type": model_type,
                "message": f"No aligned inference rows available for {model_type}.",
            }

        if model_type == "lstm":
            sequence_length = int(bundle.sequence_length)
            if len(X) < sequence_length:
                return {
                    "status": "unavailable",
                    "model_type": model_type,
                    "message": f"Need at least {sequence_length} feature rows for LSTM comparison.",
                }
            X_seq = np.array([X[i - sequence_length + 1:i + 1] for i in range(sequence_length - 1, len(X))])
            probs = probability_up(bundle.model.predict_proba(X_seq))
            pred_index = aligned.index[sequence_length - 1:]
        else:
            probs = probability_up(bundle.model.predict_proba(X))
            pred_index = aligned.index
    except Exception as exc:
        return {
            "status": "unavailable",
            "model_type": model_type,
            "message": f"{model_type} prediction failed: {exc}",
        }

    for date, prob_up in zip(pred_index, probs):
        probability_value = float(prob_up)
        prediction_frame.loc[date, "probability_up"] = probability_value
        prediction_frame.loc[date, "confidence_score"] = _prediction_confidence(probability_value)
        prediction_frame.loc[date, "uncertainty"] = 0.0
        prediction_frame.loc[date, "directional_signal"] = _classify_direction(probability_value)

    return {
        "status": "ok",
        "model_type": model_type,
        "model_file": model_file,
        "message": f"Loaded trained {model_type} model from {model_file}",
        "predictions": prediction_frame,
        "feature_rows": len(feat_df),
    }


def _build_hybrid_signal_history(
    symbol: str,
    df: pd.DataFrame,
    technical_history: Dict[str, Any],
    model_history: Dict[str, Any],
) -> Dict[str, Any]:
    if model_history.get("status") != "ok":
        return model_history

    signals = pd.DataFrame(0, index=df.index, columns=[symbol], dtype=int)
    details = pd.DataFrame(index=df.index, columns=[symbol], dtype=object)
    prediction_frame = model_history["predictions"]

    for date in df.index:
        tech_detail = technical_history["details"].loc[date, symbol]
        tech = (tech_detail or {}).get("technical_details", {"signal": "NEUTRAL", "confidence": 0.0, "patterns": []})
        indicator_snapshot = (tech_detail or {}).get("indicator_snapshot", {})
        pred_row = prediction_frame.loc[date] if date in prediction_frame.index else None
        prob_up = None if pred_row is None else pred_row.get("probability_up")

        if pred_row is None or pd.isna(prob_up):
            details.at[date, symbol] = {
                "action": "HOLD",
                "confidence": 0.0,
                "patterns": list(tech.get("patterns", []) or []),
                "predicted_return": None,
                "probability_up": None,
                "reason": "No model prediction available for this date.",
                "technical_details": tech,
                "indicator_snapshot": indicator_snapshot,
            }
            continue

        combined = _combine_hybrid_action(tech, float(prob_up))

        signals.at[date, symbol] = _to_signal_value(combined.get("action", "HOLD"))
        details.at[date, symbol] = {
            "action": combined.get("action", "HOLD"),
            "confidence": float(combined.get("confidence", 0.0) or 0.0),
            "patterns": list(tech.get("patterns", []) or []),
            "predicted_return": None,
            "probability_up": float(prob_up),
            "reason": _compose_hybrid_reason(tech, float(prob_up), indicator_snapshot),
            "technical_details": tech,
            "indicator_snapshot": indicator_snapshot,
        }

    exec_signals, exec_details = _shift_for_execution(symbol, signals, details)
    return {
        "status": "ok",
        "model_type": model_history["model_type"],
        "model_file": model_history.get("model_file"),
        "message": model_history.get("message"),
        "signals": exec_signals,
        "details": exec_details,
    }


def _build_buy_hold_history(symbol: str, index: pd.Index, label: str) -> Dict[str, Any]:
    signals = pd.DataFrame(0, index=index, columns=[symbol], dtype=int)
    details = pd.DataFrame(index=index, columns=[symbol], dtype=object)
    for date in index:
        details.at[date, symbol] = {
            "action": "HOLD",
            "confidence": 100.0,
            "patterns": [],
            "predicted_return": None,
            "reason": f"{label} benchmark remains fully invested.",
        }
    if len(index) > 0:
        signals.at[index[0], symbol] = 1
        details.at[index[0], symbol] = {
            "action": "BUY",
            "confidence": 100.0,
            "patterns": [],
            "predicted_return": None,
            "reason": f"{label} benchmark enters on the first available date.",
        }
    return {"signals": signals, "details": details}


def _format_trade(trade) -> Dict[str, Any]:
    return {
        "date": str(trade.date.date() if hasattr(trade.date, "date") else trade.date),
        "symbol": trade.symbol,
        "type": trade.trade_type,
        "quantity": round(float(trade.quantity), 4),
        "price": round(float(trade.price), 2),
        "commission": round(float(trade.commission), 2),
        "slippage": round(float(trade.slippage), 2),
        "reason": trade.reason,
        "confidence": round(float(trade.confidence), 2),
        "predicted_return": round(float(trade.predicted_return), 4) if trade.predicted_return is not None else None,
        "probability_up": round(float(trade.probability_up), 4) if trade.probability_up is not None else None,
        "patterns": list(trade.patterns or []),
        "strategy": trade.strategy,
        "model_type": trade.model_type,
        "entry_date": str(trade.entry_date.date()) if trade.entry_date is not None else None,
        "holding_days": trade.holding_days,
        "realized_pnl": round(float(trade.realized_pnl), 2) if trade.realized_pnl is not None else None,
        "return_pct": round(float(trade.return_pct), 4) if trade.return_pct is not None else None,
    }


def _format_run(
    *,
    key: str,
    label: str,
    strategy: str,
    model_type: Optional[str],
    result: Optional[Dict[str, Any]],
    signal_details: Optional[pd.DataFrame],
    symbol: str,
    status: str = "ok",
    message: Optional[str] = None,
    benchmark_type: Optional[str] = None,
) -> Dict[str, Any]:
    if status != "ok" or result is None:
        return {
            "key": key,
            "label": label,
            "strategy": strategy,
            "model_type": model_type,
            "benchmark_type": benchmark_type,
            "status": status,
            "message": message or "Run unavailable",
            "metrics": {},
            "equity_curve": [],
            "trades": [],
            "markers": [],
            "signal_summary": {"buy_signals": 0, "sell_signals": 0, "hold_signals": 0},
        }

    equity_curve = []
    equity = result.get("portfolio_values", pd.Series(dtype=float))
    if isinstance(equity, pd.Series) and not equity.empty:
        for dt, val in equity.items():
            equity_curve.append({"date": str(dt.date()), "value": round(float(val), 2)})

    trades = [_format_trade(t) for t in result.get("trades", [])]
    markers = [
        {
            "date": trade["date"],
            "type": trade["type"],
            "price": trade["price"],
            "label": trade["type"],
            "reason": trade["reason"],
            "confidence": trade["confidence"],
            "predicted_return": trade["predicted_return"],
            "probability_up": trade["probability_up"],
            "patterns": trade["patterns"],
        }
        for trade in trades
    ]

    if signal_details is not None:
        detail_series = signal_details[symbol]
        buy_signals = int(sum(1 for value in detail_series if isinstance(value, dict) and _to_signal_value(value.get("action")) == 1))
        sell_signals = int(sum(1 for value in detail_series if isinstance(value, dict) and _to_signal_value(value.get("action")) == -1))
        hold_signals = int(len(detail_series) - buy_signals - sell_signals)
    else:
        buy_signals = sell_signals = hold_signals = 0

    return {
        "key": key,
        "label": label,
        "strategy": strategy,
        "model_type": model_type,
        "benchmark_type": benchmark_type,
        "status": status,
        "message": message or "Completed",
        "metrics": _json_safe(result.get("metrics", {})),
        "equity_curve": equity_curve,
        "trades": trades,
        "markers": markers,
        "signal_summary": {
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
        },
    }


def _run_engine(
    *,
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    signal_details: pd.DataFrame,
    initial_capital: float,
    commission_rate: float,
    slippage_rate: float,
    position_size: float,
    strategy_name: str,
    model_type: Optional[str] = None,
) -> Dict[str, Any]:
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )
    return engine.run(
        prices,
        signals,
        position_size=position_size,
        signal_details=signal_details,
        strategy_name=strategy_name,
        model_type=model_type,
    )


def _build_relative_performance(primary_run: Dict[str, Any], benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
    primary_metrics = primary_run.get("metrics", {})
    if not primary_metrics:
        return {}

    comparisons = {}
    for benchmark in benchmarks:
        if benchmark.get("status") != "ok":
            continue
        bench_metrics = benchmark.get("metrics", {})
        comparisons[benchmark["key"]] = {
            "label": benchmark["label"],
            "total_return_delta": float(primary_metrics.get("total_return", 0.0) - bench_metrics.get("total_return", 0.0)),
            "cagr_delta": float(primary_metrics.get("cagr", 0.0) - bench_metrics.get("cagr", 0.0)),
            "sharpe_delta": float(primary_metrics.get("sharpe_ratio", 0.0) - bench_metrics.get("sharpe_ratio", 0.0)),
            "sortino_delta": float(primary_metrics.get("sortino_ratio", 0.0) - bench_metrics.get("sortino_ratio", 0.0)),
            "drawdown_improvement": float(abs(bench_metrics.get("max_drawdown", 0.0)) - abs(primary_metrics.get("max_drawdown", 0.0))),
            "outperformed_return": bool(primary_metrics.get("total_return", 0.0) > bench_metrics.get("total_return", 0.0)),
            "outperformed_sharpe": bool(primary_metrics.get("sharpe_ratio", 0.0) > bench_metrics.get("sharpe_ratio", 0.0)),
        }
    return comparisons


def _build_price_series(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{"date": str(idx.date()), "close": round(float(close), 2)} for idx, close in df["Close"].items()]


def _run_walk_forward_validation(symbol: str, df: pd.DataFrame, model_type: str, n_splits: int, gap: int) -> Dict[str, Any]:
    bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=1)
    feature_config = bundle.feature_config if bundle is not None else None
    preferred_feature_cols = bundle.feature_columns if bundle is not None else None

    dataset, feature_cols = build_supervised_dataset(
        df,
        horizon=NEXT_DAY_HORIZON,
        target_type="direction",
        feature_config=feature_config,
        feature_columns=preferred_feature_cols,
    )
    if dataset.empty or not feature_cols:
        return {
            "mode": "walk_forward",
            "status": "unavailable",
            "model_type": model_type,
            "message": "No features available for walk-forward validation.",
        }

    X = dataset[feature_cols].values.astype(np.float32)
    y = dataset["Target"].values.astype(np.float32)
    if model_type == "lstm":
        sequence_length = int(bundle.sequence_length) if bundle is not None else 60
        if len(X) < sequence_length + n_splits - 1:
            return {
                "mode": "walk_forward",
                "status": "unavailable",
                "model_type": model_type,
                "message": f"Need at least {sequence_length + n_splits - 1} rows for LSTM walk-forward validation.",
            }
        X, y = create_sequences(X, y, sequence_length=sequence_length)

    trainer = ModelTrainer()
    try:
        wf_df = trainer.walk_forward_validate(model_type, X, y, n_splits=n_splits, gap=gap)
    except Exception as exc:
        return {
            "mode": "walk_forward",
            "status": "unavailable",
            "model_type": model_type,
            "message": f"Walk-forward validation failed: {exc}",
        }

    folds = []
    summary = {}
    for row in wf_df.to_dict(orient="records"):
        if isinstance(row.get("fold"), str):
            summary = _json_safe(row)
        else:
            folds.append(_json_safe(row))

    return {
        "mode": "walk_forward",
        "status": "ok",
        "model_type": model_type,
        "n_splits": n_splits,
        "gap": gap,
        "folds": folds,
        "summary": summary,
    }


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """Run comparison-focused backtests for the requested symbol."""
    symbol = req.symbol.upper()
    primary_model = (req.primary_model or req.model_type).value
    benchmark_symbol = (req.benchmark_symbol or DEFAULT_INDEX_SYMBOL).upper()

    df = _download_price_data(symbol, req.start_date, req.end_date)
    if len(df) < 30:
        raise HTTPException(404, f"Insufficient data for {symbol} ({req.start_date} to {req.end_date})")

    prices = pd.DataFrame({symbol: df["Close"]})
    price_series = _build_price_series(df)

    technical_history = _build_technical_signal_history(symbol, df)
    model_histories = {
        model_type: _build_model_prediction_history(symbol, df, model_type)
        for model_type in MODEL_TYPES
    }
    hybrid_histories = {
        model_type: _build_hybrid_signal_history(symbol, df, technical_history, model_history)
        for model_type, model_history in model_histories.items()
    }
    buy_hold_history = _build_buy_hold_history(symbol, df.index, f"{symbol} buy-and-hold")

    primary_hybrid_history = hybrid_histories.get(primary_model, {})
    if primary_hybrid_history.get("status") == "ok":
        hybrid_result = _run_engine(
            prices=prices,
            signals=primary_hybrid_history["signals"],
            signal_details=primary_hybrid_history["details"],
            initial_capital=req.initial_capital,
            commission_rate=req.commission_rate,
            slippage_rate=req.slippage_rate,
            position_size=req.position_size,
            strategy_name=STRATEGY_HYBRID,
            model_type=primary_model,
        )
        hybrid_run = _format_run(
            key=STRATEGY_HYBRID,
            label=f"Hybrid ML + TA ({primary_model.replace('_', ' ').title()})",
            strategy=STRATEGY_HYBRID,
            model_type=primary_model,
            result=hybrid_result,
            signal_details=primary_hybrid_history["details"],
            symbol=symbol,
            message=primary_hybrid_history.get("message"),
        )
    else:
        hybrid_run = _format_run(
            key=STRATEGY_HYBRID,
            label=f"Hybrid ML + TA ({primary_model.replace('_', ' ').title()})",
            strategy=STRATEGY_HYBRID,
            model_type=primary_model,
            result=None,
            signal_details=None,
            symbol=symbol,
            status="unavailable",
            message=primary_hybrid_history.get("message", f"{primary_model} hybrid run unavailable"),
        )

    technical_result = _run_engine(
        prices=prices,
        signals=technical_history["signals"],
        signal_details=technical_history["details"],
        initial_capital=req.initial_capital,
        commission_rate=req.commission_rate,
        slippage_rate=req.slippage_rate,
        position_size=req.position_size,
        strategy_name=STRATEGY_TECHNICAL,
    )
    technical_run = _format_run(
        key=STRATEGY_TECHNICAL,
        label="Technical Analysis Only",
        strategy=STRATEGY_TECHNICAL,
        model_type=None,
        result=technical_result,
        signal_details=technical_history["details"],
        symbol=symbol,
    )

    stock_hold_result = _run_engine(
        prices=prices,
        signals=buy_hold_history["signals"],
        signal_details=buy_hold_history["details"],
        initial_capital=req.initial_capital,
        commission_rate=req.commission_rate,
        slippage_rate=req.slippage_rate,
        position_size=1.0,
        strategy_name=STRATEGY_HOLD,
    )
    stock_hold_run = _format_run(
        key=f"{symbol.lower()}_buy_hold",
        label=f"{symbol} Buy-and-Hold",
        strategy=STRATEGY_HOLD,
        model_type=None,
        result=stock_hold_result,
        signal_details=buy_hold_history["details"],
        symbol=symbol,
        benchmark_type="selected_symbol",
    )

    strategy_runs = [
        hybrid_run,
        technical_run,
        {**stock_hold_run, "key": STRATEGY_HOLD, "label": "Buy-and-Hold Baseline"},
    ]

    model_runs = []
    for model_type in MODEL_TYPES:
        hybrid_history = hybrid_histories[model_type]
        if hybrid_history.get("status") != "ok":
            model_runs.append(_format_run(
                key=f"model_{model_type}",
                label=model_type.replace("_", " ").title(),
                strategy=STRATEGY_HYBRID,
                model_type=model_type,
                result=None,
                signal_details=None,
                symbol=symbol,
                status="unavailable",
                message=hybrid_history.get("message", f"{model_type} comparison unavailable"),
            ))
            continue

        model_result = _run_engine(
            prices=prices,
            signals=hybrid_history["signals"],
            signal_details=hybrid_history["details"],
            initial_capital=req.initial_capital,
            commission_rate=req.commission_rate,
            slippage_rate=req.slippage_rate,
            position_size=req.position_size,
            strategy_name=STRATEGY_HYBRID,
            model_type=model_type,
        )
        model_runs.append(_format_run(
            key=f"model_{model_type}",
            label=model_type.replace("_", " ").title(),
            strategy=STRATEGY_HYBRID,
            model_type=model_type,
            result=model_result,
            signal_details=hybrid_history["details"],
            symbol=symbol,
            message=hybrid_history.get("message"),
        ))

    benchmarks = [stock_hold_run]
    if req.include_market_benchmark:
        try:
            benchmark_df = _download_price_data(benchmark_symbol, req.start_date, req.end_date)
            benchmark_close = benchmark_df["Close"].reindex(df.index).ffill().bfill()
            benchmark_prices = pd.DataFrame({benchmark_symbol: benchmark_close})
            benchmark_history = _build_buy_hold_history(benchmark_symbol, benchmark_prices.index, f"{benchmark_symbol} benchmark")
            benchmark_result = _run_engine(
                prices=benchmark_prices,
                signals=benchmark_history["signals"],
                signal_details=benchmark_history["details"],
                initial_capital=req.initial_capital,
                commission_rate=req.commission_rate,
                slippage_rate=req.slippage_rate,
                position_size=1.0,
                strategy_name=STRATEGY_HOLD,
            )
            benchmarks.append(_format_run(
                key=f"{benchmark_symbol.lower()}_benchmark",
                label=f"{benchmark_symbol} Market Benchmark",
                strategy=STRATEGY_HOLD,
                model_type=None,
                result=benchmark_result,
                signal_details=benchmark_history["details"],
                symbol=benchmark_symbol,
                benchmark_type="market",
            ))
        except HTTPException as exc:
            benchmarks.append(_format_run(
                key=f"{benchmark_symbol.lower()}_benchmark",
                label=f"{benchmark_symbol} Market Benchmark",
                strategy=STRATEGY_HOLD,
                model_type=None,
                result=None,
                signal_details=None,
                symbol=benchmark_symbol,
                status="unavailable",
                message=str(exc.detail),
                benchmark_type="market",
            ))

    primary_run = hybrid_run if hybrid_run.get("status") == "ok" else {**technical_run, "fallback_reason": hybrid_run.get("message")}
    primary_run["relative_performance"] = _build_relative_performance(primary_run, benchmarks)

    validation = {
        "mode": "single_period",
        "status": "skipped",
        "message": "Single-period backtest only.",
    }
    if req.validation_mode.value == "walk_forward":
        validation = _run_walk_forward_validation(symbol, df, primary_model, req.walk_forward_splits, req.walk_forward_gap)

    backtest_id = str(uuid.uuid4())[:8]
    response = {
        "backtest_id": backtest_id,
        "summary": {
            "symbol": symbol,
            "start_date": req.start_date,
            "end_date": req.end_date,
            "initial_capital": req.initial_capital,
            "position_size": req.position_size,
            "commission_rate": req.commission_rate,
            "slippage_rate": req.slippage_rate,
            "primary_model": primary_model,
            "primary_strategy": STRATEGY_HYBRID,
            "effective_primary_strategy": primary_run.get("strategy"),
            "benchmark_symbol": benchmark_symbol if req.include_market_benchmark else None,
            "include_market_benchmark": req.include_market_benchmark,
            "validation_mode": req.validation_mode.value,
            "signal_execution_lag_days": 1,
        },
        "price_series": price_series,
        "primary_run": primary_run,
        "strategy_runs": strategy_runs,
        "model_runs": model_runs,
        "benchmarks": benchmarks,
        "validation": validation,
        "metrics": primary_run.get("metrics", {}),
        "equity_curve": primary_run.get("equity_curve", []),
        "trades": primary_run.get("trades", []),
        "message": f"Backtest comparison completed for {symbol}. Primary strategy: {primary_run.get('label', 'N/A')}",
    }

    response = _json_safe(response)
    _backtest_results[backtest_id] = response
    return BacktestResponse(**response)


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Retrieve a previous backtest result."""
    if backtest_id not in _backtest_results:
        raise HTTPException(404, f"Backtest {backtest_id} not found")
    return _backtest_results[backtest_id]


@router.get("/results")
async def list_backtests():
    """List all backtest results."""
    return {
        "backtests": [
            {"backtest_id": bid, "metrics": data.get("metrics", {}), "summary": data.get("summary", {})}
            for bid, data in _backtest_results.items()
        ]
    }
