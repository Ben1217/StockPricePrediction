from unittest.mock import patch

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes import backtest as backtest_route


def _make_signal_history(symbol: str, index: pd.Index):
    signals = pd.DataFrame(0, index=index, columns=[symbol], dtype=int)
    details = pd.DataFrame(index=index, columns=[symbol], dtype=object)
    details.loc[:, symbol] = [
        {"action": "HOLD", "confidence": 0.0, "reason": "Hold", "patterns": []}
        for _ in index
    ]
    return {"signals": signals, "details": details}


def _make_result(index: pd.Index, total_return: float, total_trades: int):
    return {
        "metrics": {
            "total_return": total_return,
            "cagr": total_return,
            "sharpe_ratio": 1.1 + total_return,
            "sortino_ratio": 1.0 + total_return,
            "max_drawdown": -0.08,
            "total_trades": total_trades,
            "final_value": 100000 * (1 + total_return),
        },
        "portfolio_values": pd.Series(
            [100000, 101000, 102000 * (1 + total_return)],
            index=index[:3],
        ),
        "trades": [],
    }


def test_backtest_run_returns_comparison_payload():
    app = FastAPI()
    app.include_router(backtest_route.router, prefix="/api/backtest")
    client = TestClient(app)

    index = pd.date_range("2024-01-01", periods=40, freq="D")
    df = pd.DataFrame(
        {
            "Open": [100 + i for i in range(40)],
            "High": [101 + i for i in range(40)],
            "Low": [99 + i for i in range(40)],
            "Close": [100 + i for i in range(40)],
            "Volume": [1000 for _ in range(40)],
        },
        index=index,
    )

    def fake_run_engine(*, prices, strategy_name, model_type=None, position_size=None, **kwargs):
        column = prices.columns[0]
        if strategy_name == backtest_route.STRATEGY_HYBRID and model_type == "xgboost":
            return _make_result(prices.index, 0.18, 4)
        if strategy_name == backtest_route.STRATEGY_HYBRID:
            return _make_result(prices.index, 0.12, 3)
        if strategy_name == backtest_route.STRATEGY_TECHNICAL:
            return _make_result(prices.index, 0.09, 2)
        if column == "^GSPC":
            return _make_result(prices.index, 0.05, 1)
        return _make_result(prices.index, 0.07, 1)

    with (
        patch.object(backtest_route, "_download_price_data", return_value=df),
        patch.object(backtest_route, "_build_technical_signal_history", return_value=_make_signal_history("AAPL", index)),
        patch.object(
            backtest_route,
            "_build_model_prediction_history",
            side_effect=[
                {"status": "ok", "model_type": "xgboost", "predictions": pd.DataFrame(index=index)},
                {"status": "unavailable", "model_type": "random_forest", "message": "missing random forest"},
                {"status": "unavailable", "model_type": "lstm", "message": "missing lstm"},
            ],
        ),
        patch.object(
            backtest_route,
            "_build_hybrid_signal_history",
            side_effect=[
                {"status": "ok", "model_type": "xgboost", "signals": _make_signal_history("AAPL", index)["signals"], "details": _make_signal_history("AAPL", index)["details"], "message": "xgboost ready"},
                {"status": "unavailable", "model_type": "random_forest", "message": "missing random forest"},
                {"status": "unavailable", "model_type": "lstm", "message": "missing lstm"},
            ],
        ),
        patch.object(backtest_route, "_build_buy_hold_history", side_effect=lambda symbol, idx, label: _make_signal_history(symbol, idx)),
        patch.object(backtest_route, "_run_engine", side_effect=fake_run_engine),
        patch.object(backtest_route, "_run_walk_forward_validation", return_value={"mode": "walk_forward", "status": "ok", "folds": [], "summary": {}}),
    ):
        response = client.post(
            "/api/backtest/run",
            json={
                "symbol": "AAPL",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
                "primary_model": "xgboost",
                "model_type": "xgboost",
                "include_market_benchmark": True,
                "validation_mode": "walk_forward",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert "summary" in payload
    assert "primary_run" in payload
    assert "strategy_runs" in payload and len(payload["strategy_runs"]) == 3
    assert "model_runs" in payload and len(payload["model_runs"]) == 3
    assert "benchmarks" in payload and len(payload["benchmarks"]) == 2
    assert payload["metrics"] == payload["primary_run"]["metrics"]
    assert payload["validation"]["mode"] == "walk_forward"
    assert payload["summary"]["benchmark_symbol"] == "^GSPC"
    unavailable_models = [run for run in payload["model_runs"] if run["status"] == "unavailable"]
    assert len(unavailable_models) == 2


def test_backtest_run_accepts_simplified_strategy_payload():
    app = FastAPI()
    app.include_router(backtest_route.router, prefix="/api/backtest")
    client = TestClient(app)

    simple_result = {
        "summary": {
            "symbol": "MSFT",
            "start_date": "2022-01-01",
            "end_date": "2022-03-01",
            "initial_capital": 100000,
            "strategy": "ta_only",
        },
        "metrics": {
            "total_return": 4.2,
            "cagr": 9.1,
            "sharpe": 1.25,
            "max_drawdown": -3.5,
            "win_rate": 50.0,
            "n_trades": 2,
            "final_value": 104200,
            "bh_return": 3.1,
        },
        "benchmark_metrics": {"total_return": 3.1, "final_value": 103100},
        "equity_curve": [
            {"date": "2022-01-01", "value": 100000},
            {"date": "2022-03-01", "value": 104200},
        ],
        "bh_curve": [
            {"date": "2022-01-01", "value": 100000},
            {"date": "2022-03-01", "value": 103100},
        ],
        "price_series": [{"date": "2022-01-01", "close": 300.0}],
        "trades": [
            {"date": "2022-01-10", "type": "BUY", "shares": 30.0, "price": 310.0, "pnl": None, "return_pct": None, "reason": "SMA20 above SMA50"},
            {"date": "2022-02-10", "type": "SELL", "shares": 30.0, "price": 322.0, "pnl": 360.0, "return_pct": 3.87, "reason": "RSI 76.0"},
        ],
    }

    with patch.object(backtest_route, "run_simple_backtest", return_value=simple_result) as run_mock:
        response = client.post(
            "/api/backtest/run",
            json={
                "symbol": "MSFT",
                "start_date": "2022-01-01",
                "end_date": "2022-03-01",
                "initial_capital": 100000,
                "strategy": "ta_only",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["metrics"]["total_return"] == 4.2
    assert payload["bh_curve"] == simple_result["bh_curve"]
    assert payload["trades"][0]["shares"] == 30.0
    assert payload["strategy_runs"][0]["key"] == "ta_only"
    assert "Buy-and-hold for MSFT returned 3.10%" in payload["benchmark_notice"]
    run_mock.assert_called_once()
