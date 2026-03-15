"""
CrewAI tool wrappers for QuantVision FastAPI endpoints.

Each @tool function bridges the LLM reasoning layer with the
existing Python backend by calling API endpoints over HTTP.

Uses crewai.tools.tool decorator for compatibility with CrewAI Agent.
"""

import os
from crewai.tools import tool
import httpx

BASE_URL = os.getenv("QUANTVISION_API_URL", "http://localhost:8000")


@tool("Get Price Prediction")
def get_prediction(ticker: str, model: str = "ensemble") -> dict:
    """Get ML price prediction for a stock ticker.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT').
        model: Model type — 'xgboost', 'random_forest', 'lstm', or 'ensemble'.

    Returns:
        Dict with predicted_price, forecasts, confidence intervals, model_info.
    """
    try:
        r = httpx.post(
            f"{BASE_URL}/api/predict",
            json={"symbol": ticker, "model_type": model, "horizon": 30},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Prediction failed: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": f"Prediction request failed: {str(e)}"}


@tool("Get Technical Signals")
def get_technical_signals(ticker: str) -> dict:
    """Get technical analysis signals, candlestick patterns, and confluence score for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT').

    Returns:
        Dict with candlestick_patterns, chart_patterns, confluence signal
        (RSI, MACD, pattern signals, overall direction, strength).
    """
    try:
        r = httpx.get(
            f"{BASE_URL}/api/patterns/{ticker}",
            params={"interval": "1d", "lookback": 120},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Patterns failed: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": f"Patterns request failed: {str(e)}"}


@tool("Optimize Portfolio")
def optimize_portfolio(tickers: list[str], strategy: str = "max_sharpe") -> dict:
    """Run portfolio optimization and return optimal weights.

    Args:
        tickers: List of stock ticker symbols (e.g. ['AAPL', 'MSFT', 'GOOGL']).
        strategy: Optimization method — 'max_sharpe', 'min_volatility', 'max_return', 'risk_parity'.

    Returns:
        Dict with weights, expected_return, volatility, sharpe_ratio, metrics.
    """
    try:
        r = httpx.post(
            f"{BASE_URL}/api/portfolio/optimize",
            json={"symbols": tickers, "method": strategy},
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Portfolio optimization failed: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": f"Portfolio optimization request failed: {str(e)}"}


@tool("Run Backtest")
def run_backtest(ticker: str, start_date: str = "2023-01-01", end_date: str = "2024-12-31") -> dict:
    """Backtest a trading strategy for a stock ticker historically.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').
        start_date: Backtest start date in YYYY-MM-DD format.
        end_date: Backtest end date in YYYY-MM-DD format.

    Returns:
        Dict with backtest_id, metrics (total_return, sharpe_ratio,
        max_drawdown, win_rate), equity_curve, trades.
    """
    try:
        r = httpx.post(
            f"{BASE_URL}/api/backtest/run",
            json={
                "symbol": ticker,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": 100000,
                "model_type": "xgboost",
                "position_size": 0.1,
            },
            timeout=120.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Backtest failed: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": f"Backtest request failed: {str(e)}"}


@tool("Get Live Quote")
def get_live_quote(ticker: str) -> dict:
    """Get real-time price quote, volume, and market status for a stock.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL').

    Returns:
        Dict with symbol, price, volume, timestamp, market_open status.
    """
    try:
        r = httpx.get(
            f"{BASE_URL}/api/data/quote/{ticker}",
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        return {"error": f"Quote failed: {e.response.status_code}", "detail": e.response.text}
    except Exception as e:
        return {"error": f"Quote request failed: {str(e)}"}
