"""
Backtest API routes — run and retrieve backtests.
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import BacktestRequest, BacktestResponse
from src.backtesting.backtest_engine import BacktestEngine

router = APIRouter()

# Store results in memory
_backtest_results = {}


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest):
    """Run a backtest with the given parameters."""
    import yfinance as yf

    # Fetch data
    df = yf.download(req.symbol, start=req.start_date, end=req.end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty or len(df) < 30:
        raise HTTPException(404, f"Insufficient data for {req.symbol} ({req.start_date} to {req.end_date})")

    # Create a simple mean-reversion signal based on technical indicators
    close = df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    rsi = _compute_rsi(close, 14)

    # Signal: buy when RSI < 30 and price < SMA20; sell when RSI > 70 and price > SMA50
    signals = pd.DataFrame(0, index=df.index, columns=[req.symbol])
    buy_mask = (rsi < 30) & (close < sma20)
    sell_mask = (rsi > 70) & (close > sma50)
    signals.loc[buy_mask, req.symbol] = 1
    signals.loc[sell_mask, req.symbol] = -1

    # If a trained model exists, use model-based signals instead
    model_dir = Path("models/saved_models") / req.model_type.value
    if model_dir.exists():
        try:
            from src.features.technical_indicators import add_all_technical_indicators
            from src.features.feature_engineering import create_features
            from src.models.model_trainer import ModelTrainer
            import json

            feat_df = create_features(df)
            feat_df = feat_df.dropna()
            feature_cols_file = Path("models/model_metadata/feature_columns.json")
            if feature_cols_file.exists():
                with open(feature_cols_file) as f:
                    feature_cols = json.load(f)
                feature_cols = [c for c in feature_cols if c in feat_df.columns]
            else:
                feature_cols = [c for c in feat_df.columns
                                if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Target"]]

            if feature_cols:
                trainer = ModelTrainer()
                model = trainer.create_model(req.model_type.value)
                for p in model_dir.iterdir():
                    try:
                        model.load(str(p))
                        break
                    except Exception:
                        continue

                if model.is_fitted:
                    X = feat_df[feature_cols].values
                    preds = model.predict(X)
                    # Signal: buy when predicted return > 0, sell when < 0
                    pred_signals = pd.Series(0, index=feat_df.index)
                    pred_signals[preds > 0.001] = 1
                    pred_signals[preds < -0.001] = -1
                    signals = pd.DataFrame({req.symbol: pred_signals})
        except Exception:
            pass  # Fall back to technical signals

    # Run backtest
    prices = pd.DataFrame({req.symbol: df["Close"]})
    engine = BacktestEngine(
        initial_capital=req.initial_capital,
        commission_rate=req.commission_rate,
        slippage_rate=req.slippage_rate,
    )
    result = engine.run(prices, signals, position_size=req.position_size)

    # Format response
    backtest_id = str(uuid.uuid4())[:8]

    equity = result.get("portfolio_values", pd.Series())
    equity_curve = []
    if not equity.empty:
        for dt, val in equity.items():
            equity_curve.append({"date": str(dt.date()), "value": round(float(val), 2)})

    trades = []
    for t in result.get("trades", []):
        trades.append({
            "date": str(t.date.date() if hasattr(t.date, "date") else t.date),
            "symbol": t.symbol,
            "type": t.trade_type,
            "quantity": round(t.quantity, 4),
            "price": round(t.price, 2),
            "commission": round(t.commission, 2),
        })

    metrics = result.get("metrics", {})
    # Convert numpy types for JSON
    clean_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, (np.integer, np.floating)):
            clean_metrics[k] = float(v)
        elif isinstance(v, (int, float, str)):
            clean_metrics[k] = v

    _backtest_results[backtest_id] = {"metrics": clean_metrics, "equity_curve": equity_curve, "trades": trades}

    return BacktestResponse(
        backtest_id=backtest_id,
        metrics=clean_metrics,
        equity_curve=equity_curve,
        trades=trades,
        message=f"Backtest completed: {clean_metrics.get('total_return', 0):.2%} return",
    )


@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Retrieve a previous backtest result."""
    if backtest_id not in _backtest_results:
        raise HTTPException(404, f"Backtest {backtest_id} not found")
    return _backtest_results[backtest_id]


@router.get("/results")
async def list_backtests():
    """List all backtest results."""
    return {"backtests": [
        {"backtest_id": bid, "metrics": data["metrics"]}
        for bid, data in _backtest_results.items()
    ]}


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Simple RSI computation."""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
