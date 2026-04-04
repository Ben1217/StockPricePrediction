import pandas as pd

from src.backtesting.backtest_engine import BacktestEngine


def test_backtest_engine_reports_closed_trade_metrics():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"AAPL": [100.0, 110.0, 100.0, 90.0, 95.0]}, index=dates)
    signals = pd.DataFrame({"AAPL": [1, -1, 1, -1, 0]}, index=dates)
    details = pd.DataFrame(index=dates, columns=["AAPL"], dtype=object)
    details.loc[:, "AAPL"] = [
        {"reason": "Entry 1", "confidence": 75.0},
        {"reason": "Exit 1", "confidence": 80.0},
        {"reason": "Entry 2", "confidence": 70.0},
        {"reason": "Exit 2", "confidence": 78.0},
        {"reason": "Hold", "confidence": 0.0},
    ]

    engine = BacktestEngine(initial_capital=1000, commission_rate=0.0, slippage_rate=0.0)
    result = engine.run(
        prices,
        signals,
        position_size=1.0,
        signal_details=details,
        strategy_name="hybrid_ml_ta",
        model_type="xgboost",
    )

    metrics = result["metrics"]
    assert metrics["closed_trades"] == 2
    assert metrics["profit_factor"] > 0
    assert metrics["average_win"] > 0
    assert metrics["average_loss"] < 0
    assert metrics["max_consecutive_wins"] == 1
    assert metrics["max_consecutive_losses"] == 1
    assert "cagr" in metrics


def test_backtest_engine_handles_zero_trade_metrics():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0, 103.0]}, index=dates)
    signals = pd.DataFrame({"AAPL": [0, 0, 0, 0]}, index=dates)
    details = pd.DataFrame(index=dates, columns=["AAPL"], dtype=object)
    details.loc[:, "AAPL"] = [{"reason": "Hold", "confidence": 0.0}] * len(dates)

    engine = BacktestEngine(initial_capital=1000, commission_rate=0.0, slippage_rate=0.0)
    result = engine.run(prices, signals, position_size=1.0, signal_details=details)

    metrics = result["metrics"]
    assert metrics["total_trades"] == 0
    assert metrics["closed_trades"] == 0
    assert metrics["profit_factor"] == 0.0
    assert metrics["expectancy"] == 0.0
    assert metrics["max_consecutive_wins"] == 0
    assert metrics["max_consecutive_losses"] == 0
