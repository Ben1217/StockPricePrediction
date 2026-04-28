"""
Simplified backtesting service for the QuantVision dashboard.

This module keeps the FYP1 backtest path intentionally small: one symbol, one
strategy, one buy-and-hold benchmark, and a compact trade log. Indicator
calculation is delegated to the existing feature pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.data_loader import download_stock_data
from src.features.technical_indicators import add_all_technical_indicators


SIMPLE_STRATEGIES = {"ta_only", "ml_hybrid", "buy_hold"}


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    strategy: str,
    ml_predictions: Optional[pd.Series] = None,
    model_type: str = "xgboost",
) -> Dict[str, Any]:
    """
    Run a simplified backtest and return metrics, equity curves, and trades.

    Strategies:
    - ta_only: SMA20/SMA50 crossover with RSI confirmation.
    - ml_hybrid: TA signal must agree with a best-effort ML direction signal.
    - buy_hold: fully invested from the first available bar.
    """
    strategy = str(strategy or "").strip().lower()
    if strategy not in SIMPLE_STRATEGIES:
        raise ValueError("strategy must be one of: ta_only, ml_hybrid, buy_hold")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than zero")

    df = _fetch_ohlcv(symbol, start_date, end_date)
    enriched = add_all_technical_indicators(df)

    ml_status = "not_requested"
    if strategy == "buy_hold":
        enriched["signal"] = 0
        enriched.iloc[0, enriched.columns.get_loc("signal")] = 1
    elif strategy == "ta_only":
        enriched["signal"] = _generate_ta_signals(enriched)
    else:
        if ml_predictions is None:
            ml_predictions = _get_ml_predictions(enriched, symbol=symbol, model_type=model_type)

        aligned_predictions = _align_ml_predictions(ml_predictions, enriched.index)
        if aligned_predictions is None:
            enriched["signal"] = _generate_ta_signals(enriched)
            ml_status = "fallback_ta_only"
        else:
            enriched["ml_pred"] = aligned_predictions
            enriched["signal"] = _generate_hybrid_signals(enriched)
            ml_status = "available"

    trades, equity_curve = _simulate_portfolio(enriched, initial_capital, strategy)
    bh_curve = _buy_and_hold_curve(enriched, initial_capital)
    metrics = _compute_metrics(equity_curve, bh_curve, trades, initial_capital)
    benchmark_metrics = _compute_benchmark_metrics(bh_curve, initial_capital)

    return {
        "summary": {
            "symbol": symbol.upper(),
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital),
            "strategy": strategy,
            "position_size": 1.0 if strategy == "buy_hold" else 0.10,
            "ml_status": ml_status,
            "signal_execution_lag_days": 1 if strategy != "buy_hold" else 0,
        },
        "metrics": metrics,
        "benchmark_metrics": benchmark_metrics,
        "equity_curve": _series_to_records(equity_curve),
        "bh_curve": _series_to_records(bh_curve),
        "price_series": _price_series_records(enriched),
        "trades": trades,
    }


def _fetch_ohlcv(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = download_stock_data(symbol.upper(), start_date, end_date, interval="1d")
    if df is None or df.empty:
        raise ValueError(f"No data found for {symbol}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {', '.join(missing)}")

    data = df.copy()
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data[~data.index.duplicated(keep="last")]
    data = data.sort_index()
    data = data.dropna(subset=["Open", "High", "Low", "Close"])
    data = data[data["Close"] > 0]

    if len(data) < 2:
        raise ValueError(f"Insufficient data for {symbol} ({start_date} to {end_date})")

    return data[required_cols]


def _align_ml_predictions(predictions: Optional[pd.Series], index: pd.Index) -> Optional[pd.Series]:
    if predictions is None or len(predictions) == 0:
        return None

    series = pd.Series(predictions).copy()
    if len(series) == len(index):
        series.index = index
    else:
        series = series.reindex(index)

    series = pd.to_numeric(series, errors="coerce")
    if series.notna().sum() == 0:
        return None
    return (series.ffill().fillna(0) > 0).astype(int)


def _get_ml_predictions(df: pd.DataFrame, symbol: str, model_type: str = "xgboost") -> Optional[pd.Series]:
    """
    Best-effort server-side ML direction predictions.

    This uses the existing trained bundle as a simple FYP1 shortcut. It is not a
    walk-forward retrain, so callers should treat it as a convenience signal.
    """
    try:
        from src.features.feature_engineering import (
            build_feature_frame,
            select_feature_columns,
            transform_feature_frame,
        )
        from src.models.direction_utils import probability_up
        from src.models.model_bundle import load_model_bundle

        bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=1)
        if bundle is None:
            return None

        target_type = getattr(bundle, "target_type", bundle.metadata.get("target_type", "direction"))
        if target_type != "direction":
            return None

        feat_df = build_feature_frame(df, feature_config=bundle.feature_config)
        feature_cols = list(bundle.feature_columns or select_feature_columns(feat_df))
        if not feature_cols:
            return None

        aligned, x_values = transform_feature_frame(feat_df, feature_cols, scaler=bundle.scaler)
        if aligned.empty or len(x_values) == 0:
            return None

        if model_type == "lstm":
            sequence_length = int(getattr(bundle, "sequence_length", 60) or 60)
            if len(x_values) < sequence_length:
                return None
            x_values = np.array([
                x_values[i - sequence_length + 1:i + 1]
                for i in range(sequence_length - 1, len(x_values))
            ])
            prediction_index = aligned.index[sequence_length - 1:]
        else:
            prediction_index = aligned.index

        probabilities = probability_up(bundle.model.predict_proba(x_values))
        predictions = pd.Series(np.nan, index=df.index, dtype=float)
        predictions.loc[prediction_index] = (np.asarray(probabilities, dtype=float) >= 0.5).astype(int)
        return predictions
    except Exception:
        return None


def _generate_ta_signals(df: pd.DataFrame) -> pd.Series:
    """SMA20/SMA50 crossover with RSI confirmation."""
    signals = pd.Series(0, index=df.index, dtype=int)
    required = {"SMA_20", "SMA_50", "RSI"}
    if not required.issubset(df.columns):
        return signals

    in_position = False
    for i in range(1, len(df)):
        sma20 = df["SMA_20"].iloc[i]
        sma50 = df["SMA_50"].iloc[i]
        rsi = df["RSI"].iloc[i]
        sma20_prev = df["SMA_20"].iloc[i - 1]
        sma50_prev = df["SMA_50"].iloc[i - 1]

        if pd.isna(sma20) or pd.isna(sma50) or pd.isna(rsi):
            continue
        if pd.isna(sma20_prev) or pd.isna(sma50_prev):
            continue

        if not in_position and sma20 > sma50 and sma20_prev <= sma50_prev and rsi < 70:
            signals.iloc[i] = 1
            in_position = True
        elif in_position and (sma20 < sma50 or rsi > 75):
            signals.iloc[i] = -1
            in_position = False

    return signals


def _generate_hybrid_signals(df: pd.DataFrame) -> pd.Series:
    """Buy only when TA and ML direction agree; sell when TA exits."""
    ta_signals = _generate_ta_signals(df)
    signals = pd.Series(0, index=df.index, dtype=int)

    for i in range(len(df)):
        ta_signal = int(ta_signals.iloc[i])
        ml_signal = int(df["ml_pred"].iloc[i]) if "ml_pred" in df.columns else 0

        if ta_signal == 1 and ml_signal == 1:
            signals.iloc[i] = 1
        elif ta_signal == -1:
            signals.iloc[i] = -1

    return signals


def _simulate_portfolio(
    df: pd.DataFrame,
    initial_capital: float,
    strategy: str,
) -> tuple[List[Dict[str, Any]], pd.Series]:
    """
    Simulate executions.

    TA and hybrid signals execute on the next bar's open. Buy-and-hold enters on
    the first available bar to remain a true baseline strategy.
    """
    capital = float(initial_capital)
    shares = 0.0
    entry_price = 0.0
    entry_date = None
    trades: List[Dict[str, Any]] = []
    equity: Dict[pd.Timestamp, float] = {}
    position_pct = 1.0 if strategy == "buy_hold" else 0.10

    for i, date in enumerate(df.index):
        close = float(df["Close"].iloc[i])
        open_price = float(df["Open"].iloc[i]) if not pd.isna(df["Open"].iloc[i]) else close

        signal = 0
        signal_idx = i
        execution_price = open_price

        if strategy == "buy_hold" and i == 0:
            signal = 1
        elif strategy != "buy_hold" and i > 0:
            signal = int(df["signal"].iloc[i - 1])
            signal_idx = i - 1

        if signal == 1 and shares == 0:
            invest = capital * position_pct
            if invest > 0 and execution_price > 0:
                shares = invest / execution_price
                capital -= invest
                entry_price = execution_price
                entry_date = date
                trades.append({
                    "date": _date_str(date),
                    "type": "BUY",
                    "shares": round(float(shares), 4),
                    "price": round(execution_price, 2),
                    "pnl": None,
                    "return_pct": None,
                    "reason": _signal_reason(df, signal_idx, strategy),
                })

        elif signal == -1 and shares > 0:
            proceeds = shares * execution_price
            cost_basis = shares * entry_price
            pnl = proceeds - cost_basis
            ret_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
            capital += proceeds
            trades.append({
                "date": _date_str(date),
                "type": "SELL",
                "shares": round(float(shares), 4),
                "price": round(execution_price, 2),
                "pnl": round(float(pnl), 2),
                "return_pct": round(float(ret_pct), 2),
                "reason": _signal_reason(df, signal_idx, strategy),
            })
            shares = 0.0
            entry_price = 0.0
            entry_date = None

        equity[date] = round(float(capital + shares * close), 2)

    if shares > 0:
        last_date = df.index[-1]
        last_close = float(df["Close"].iloc[-1])
        proceeds = shares * last_close
        cost_basis = shares * entry_price
        pnl = proceeds - cost_basis
        ret_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
        capital += proceeds
        trades.append({
            "date": _date_str(last_date),
            "type": "SELL",
            "shares": round(float(shares), 4),
            "price": round(last_close, 2),
            "pnl": round(float(pnl), 2),
            "return_pct": round(float(ret_pct), 2),
            "reason": "End of backtest period - position closed",
        })
        equity[last_date] = round(float(capital), 2)

    equity_curve = pd.Series(equity, dtype=float)
    equity_curve.index.name = "Date"
    return trades, equity_curve


def _buy_and_hold_curve(df: pd.DataFrame, initial_capital: float) -> pd.Series:
    start_price = float(df["Close"].iloc[0])
    shares = float(initial_capital) / start_price
    curve = (df["Close"].astype(float) * shares).round(2)
    curve.index.name = "Date"
    return curve


def _signal_reason(df: pd.DataFrame, i: int, strategy: str) -> str:
    if strategy == "buy_hold":
        return "Buy-and-hold baseline entry"

    parts = []
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        s20 = df["SMA_20"].iloc[i]
        s50 = df["SMA_50"].iloc[i]
        if not (pd.isna(s20) or pd.isna(s50)):
            rel = "above" if s20 > s50 else "below"
            parts.append(f"SMA20 {rel} SMA50")
    if "RSI" in df.columns:
        rsi = df["RSI"].iloc[i]
        if not pd.isna(rsi):
            parts.append(f"RSI {float(rsi):.1f}")
    if strategy == "ml_hybrid" and "ml_pred" in df.columns:
        ml_text = "up" if int(df["ml_pred"].iloc[i]) == 1 else "down"
        parts.append(f"ML direction {ml_text}")
    return "; ".join(parts) if parts else "Technical signal"


def _compute_metrics(
    equity: pd.Series,
    bh: pd.Series,
    trades: List[Dict[str, Any]],
    initial_capital: float,
) -> Dict[str, Any]:
    final_value = float(equity.iloc[-1])
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    bh_return = ((float(bh.iloc[-1]) - initial_capital) / initial_capital) * 100

    n_years = max((equity.index[-1] - equity.index[0]).days / 365.25, 0.01)
    cagr = ((final_value / initial_capital) ** (1 / n_years) - 1) * 100

    daily_returns = equity.pct_change().dropna()
    daily_std = float(daily_returns.std()) if len(daily_returns) else 0.0
    sharpe = (float(daily_returns.mean()) / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min() * 100) if len(drawdown) else 0.0

    sell_trades = [trade for trade in trades if trade["type"] == "SELL" and trade["pnl"] is not None]
    wins = [trade for trade in sell_trades if float(trade["pnl"]) >= 0]
    win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0.0

    return {
        "total_return": round(float(total_return), 2),
        "cagr": round(float(cagr), 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown": round(max_dd, 2),
        "win_rate": round(float(win_rate), 1),
        "n_trades": len(sell_trades),
        "final_value": round(final_value, 2),
        "bh_return": round(float(bh_return), 2),
    }


def _compute_benchmark_metrics(curve: pd.Series, initial_capital: float) -> Dict[str, Any]:
    final_value = float(curve.iloc[-1])
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    return {
        "total_return": round(float(total_return), 2),
        "final_value": round(final_value, 2),
    }


def _series_to_records(series: pd.Series) -> List[Dict[str, Any]]:
    return [
        {"date": _date_str(idx), "value": round(float(value), 2)}
        for idx, value in series.items()
    ]


def _price_series_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {"date": _date_str(idx), "close": round(float(row["Close"]), 2)}
        for idx, row in df.iterrows()
    ]


def _date_str(value: Any) -> str:
    return str(value.date() if hasattr(value, "date") else value)
