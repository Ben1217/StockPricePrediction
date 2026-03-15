"""
CrewAI Task definitions for QuantVision agent workflows.

Tasks describe what each agent should accomplish in a workflow.
They are assembled into Crew objects in crew.py.
"""

from crewai import Task
from .agents import (
    market_monitor_agent,
    prediction_agent,
    ta_agent,
    portfolio_agent,
    backtest_agent,
    nl_query_agent,
)


def create_monitor_task(tickers: list[str]) -> Task:
    """Create a market monitoring task for the given tickers."""
    ticker_str = ", ".join(tickers)
    return Task(
        description=(
            f"Monitor live quotes and technical signals for: {ticker_str}. "
            f"Check each ticker for: price changes >2%, volume spikes >3x average, "
            f"RSI extremes (<20 or >80), and any significant candlestick patterns. "
            f"Report all anomalies found with specific numbers."
        ),
        agent=market_monitor_agent,
        expected_output=(
            "A JSON-formatted alert report with: "
            "{ alerts: [{ ticker, trigger_reason, severity, current_price, details }] }"
        ),
    )


def create_prediction_task(tickers: list[str]) -> Task:
    """Create a prediction task for the given tickers."""
    ticker_str = ", ".join(tickers)
    return Task(
        description=(
            f"Generate price predictions for: {ticker_str}. "
            f"For each ticker, use the prediction tool to get a 30-day forecast. "
            f"Report the predicted price, confidence interval, and model used."
        ),
        agent=prediction_agent,
        expected_output=(
            "A JSON-formatted prediction report with: "
            "{ predictions: [{ ticker, current_price, predicted_price, model_used, confidence }] }"
        ),
    )


def create_ta_task(tickers: list[str]) -> Task:
    """Create a technical analysis task for the given tickers."""
    ticker_str = ", ".join(tickers)
    return Task(
        description=(
            f"Analyse technical signals for: {ticker_str}. "
            f"For each ticker, retrieve patterns and confluence data. "
            f"Evaluate RSI, MACD, candlestick patterns, and chart patterns. "
            f"Generate a BUY/SELL/HOLD signal with strength 0-100 and plain-English reasoning."
        ),
        agent=ta_agent,
        expected_output=(
            "A JSON-formatted signal report with: "
            "{ signals: [{ ticker, signal: BUY|SELL|HOLD, strength: 0-100, "
            "reasoning: 'string', indicators_used: [...] }] }"
        ),
    )


def create_portfolio_task(tickers: list[str]) -> Task:
    """Create a portfolio optimization task."""
    ticker_str = ", ".join(tickers)
    return Task(
        description=(
            f"Optimize portfolio allocation for: {ticker_str}. "
            f"Use max_sharpe strategy to determine optimal weights. "
            f"Consider the prediction data and signal context from previous tasks."
        ),
        agent=portfolio_agent,
        expected_output=(
            "A JSON-formatted portfolio report with: "
            "{ weights: {{ ticker: weight }}, expected_return, volatility, sharpe_ratio, "
            "rebalancing_trades: [{{ ticker, action, weight_change }}] }"
        ),
    )


def create_backtest_task(tickers: list[str]) -> Task:
    """Create a backtesting validation task."""
    ticker_str = ", ".join(tickers)
    return Task(
        description=(
            f"Validate signals for: {ticker_str} via historical backtesting. "
            f"Run a 2-year backtest for each ticker. "
            f"Check: Sharpe >0.8, max drawdown <20%, win rate >45%. "
            f"Issue GO or NO-GO for each ticker with supporting metrics."
        ),
        agent=backtest_agent,
        expected_output=(
            "A JSON-formatted validation report with: "
            "{ validations: [{ ticker, decision: GO|NO_GO, sharpe, max_drawdown, "
            "win_rate, rejection_reason: string|null }] }"
        ),
    )


def create_nl_query_task(question: str) -> Task:
    """Create a natural language query task."""
    return Task(
        description=(
            f"Answer the following user question in plain English: '{question}'. "
            f"Use available tools to fetch live data, predictions, technical signals, "
            f"and portfolio metrics as needed. Cite specific data in your answer."
        ),
        agent=nl_query_agent,
        expected_output=(
            "A clear, plain-English answer to the user's question with: "
            "{ answer: 'detailed response string', "
            "supporting_data: { predictions: [...], signals: [...], quotes: [...] }, "
            "citations: ['specific data points referenced'] }"
        ),
    )
