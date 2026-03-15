"""
Unit tests for QuantVision AI agent tools.

Uses mocked httpx responses — no running server or API key needed.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Test: get_prediction ──────────────────────────────────────────────────────

@patch("src.agents.tools.httpx.post")
def test_get_prediction_success(mock_post):
    """Test that get_prediction calls the correct endpoint and returns data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "symbol": "AAPL",
        "model_type": "xgboost",
        "horizon": 30,
        "current_price": 185.50,
        "forecasts": [
            {"date": "2024-01-02", "predicted": 186.0, "upper95": 190.0,
             "lower95": 182.0, "upper68": 188.0, "lower68": 184.0}
        ],
        "model_info": {"type": "xgboost", "source": "trained"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from src.agents.tools import get_prediction
    result = get_prediction.run(ticker="AAPL", model="xgboost")

    assert result["symbol"] == "AAPL"
    assert "forecasts" in result
    assert result["current_price"] == 185.50
    mock_post.assert_called_once()


@patch("src.agents.tools.httpx.post")
def test_get_prediction_error(mock_post):
    """Test that get_prediction handles errors gracefully."""
    mock_post.side_effect = Exception("Connection refused")

    from src.agents.tools import get_prediction
    result = get_prediction.run(ticker="INVALID")

    assert "error" in result


# ── Test: get_technical_signals ───────────────────────────────────────────────

@patch("src.agents.tools.httpx.get")
def test_get_technical_signals_success(mock_get):
    """Test that get_technical_signals returns pattern and confluence data."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "symbol": "AAPL",
        "candlestick_patterns": [],
        "chart_patterns": [],
        "confluence": {
            "rsi_signal": "neutral",
            "rsi_value": 55.0,
            "macd_signal": "bullish",
            "pattern_signal": "neutral",
            "ml_direction": "up",
            "ml_confidence": 65.0,
            "overall": "Buy",
            "strength": 62.0,
        },
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    from src.agents.tools import get_technical_signals
    result = get_technical_signals.run(ticker="AAPL")

    assert result["symbol"] == "AAPL"
    assert "confluence" in result
    assert result["confluence"]["overall"] == "Buy"


# ── Test: optimize_portfolio ──────────────────────────────────────────────────

@patch("src.agents.tools.httpx.post")
def test_optimize_portfolio_success(mock_post):
    """Test that optimize_portfolio returns weights and metrics."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "method": "max_sharpe",
        "weights": {"AAPL": 0.3, "MSFT": 0.4, "GOOGL": 0.3},
        "expected_return": 0.15,
        "volatility": 0.12,
        "sharpe_ratio": 0.92,
        "metrics": {},
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from src.agents.tools import optimize_portfolio
    result = optimize_portfolio.run(tickers=["AAPL", "MSFT", "GOOGL"])

    assert "weights" in result
    assert result["sharpe_ratio"] == 0.92


# ── Test: run_backtest ────────────────────────────────────────────────────────

@patch("src.agents.tools.httpx.post")
def test_run_backtest_success(mock_post):
    """Test that run_backtest returns metrics and equity curve."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "backtest_id": "abc123",
        "metrics": {"total_return": 0.15, "sharpe_ratio": 1.2, "max_drawdown": -0.08},
        "equity_curve": [{"date": "2023-01-01", "value": 100000}],
        "trades": [],
        "message": "Backtest completed: 15% return",
    }
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    from src.agents.tools import run_backtest
    result = run_backtest.run(ticker="AAPL")

    assert result["backtest_id"] == "abc123"
    assert result["metrics"]["sharpe_ratio"] == 1.2


# ── Test: get_live_quote ──────────────────────────────────────────────────────

@patch("src.agents.tools.httpx.get")
def test_get_live_quote_success(mock_get):
    """Test that get_live_quote returns price and market status."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "symbol": "AAPL",
        "price": 185.50,
        "volume": 52000000,
        "timestamp": "2024-01-02T20:00:00+00:00",
        "market_open": False,
    }
    mock_response.raise_for_status = MagicMock()
    mock_get.return_value = mock_response

    from src.agents.tools import get_live_quote
    result = get_live_quote.run(ticker="AAPL")

    assert result["symbol"] == "AAPL"
    assert result["price"] == 185.50
    assert "market_open" in result
