"""
CrewAI Agent definitions for QuantVision.

Each agent has a role, goal, backstory (LLM prompt shaping), and
a set of LangChain tools that connect it to the FastAPI backend.
Uses Anthropic Claude as the LLM reasoning engine.
"""

import os
from crewai import Agent, LLM
from .tools import (
    get_prediction,
    get_technical_signals,
    optimize_portfolio,
    run_backtest,
    get_live_quote,
)


def _get_llm() -> LLM:
    """Create the LLM instance using Anthropic Claude."""
    return LLM(
        model="anthropic/claude-sonnet-4-20250514",
        api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    )


# ── Agent 1: Market Monitor ──────────────────────────────────────────────────

market_monitor_agent = Agent(
    role="Market Monitor",
    goal=(
        "Detect unusual market conditions — price spikes >2%, volume anomalies >3x average, "
        "RSI extremes (<20 or >80), and gap openings — and alert the system."
    ),
    backstory=(
        "A vigilant quantitative analyst who watches price feeds 24/7. "
        "You never miss an anomaly and always flag it immediately with precise data."
    ),
    tools=[get_live_quote, get_technical_signals],
    llm=_get_llm(),
    verbose=True,
)

# ── Agent 2: Prediction Orchestrator ─────────────────────────────────────────

prediction_agent = Agent(
    role="ML Prediction Orchestrator",
    goal=(
        "Select the best ML model (LSTM, XGBoost, Random Forest, or Ensemble) "
        "for each ticker based on recent accuracy metrics, and produce price forecasts."
    ),
    backstory=(
        "A quantitative researcher who evaluates model performance per ticker. "
        "You always choose the model with the lowest recent error and highest directional accuracy."
    ),
    tools=[get_prediction],
    llm=_get_llm(),
    verbose=True,
)

# ── Agent 3: Technical Analysis ──────────────────────────────────────────────

ta_agent = Agent(
    role="Technical Analysis Specialist",
    goal=(
        "Identify high-confidence trading signals from technical indicators. "
        "Convert raw indicator values into actionable BUY/SELL/HOLD signals with plain-English reasoning."
    ),
    backstory=(
        "A CFA chartist expert in RSI, MACD, Bollinger Bands, and candlestick patterns. "
        "You evaluate multi-factor confluence and only issue signals when multiple indicators agree."
    ),
    tools=[get_technical_signals],
    llm=_get_llm(),
    verbose=True,
)

# ── Agent 4: Portfolio Rebalancing ───────────────────────────────────────────

portfolio_agent = Agent(
    role="Portfolio Manager",
    goal=(
        "Optimize portfolio allocation and manage risk-adjusted returns. "
        "Rebalance dynamically based on live signals and prediction inputs."
    ),
    backstory=(
        "A senior portfolio manager focused on maximizing Sharpe ratio while controlling drawdown. "
        "You enforce position limits and transaction cost constraints."
    ),
    tools=[optimize_portfolio, get_prediction],
    llm=_get_llm(),
    verbose=True,
)

# ── Agent 5: Backtesting Validator ───────────────────────────────────────────

backtest_agent = Agent(
    role="Backtesting Validator",
    goal=(
        "Validate every trading signal historically before it is acted on. "
        "Issue GO if Sharpe>0.8, drawdown<20%, win rate>45%. Issue NO-GO otherwise."
    ),
    backstory=(
        "A risk manager who rejects any strategy with poor historical performance. "
        "You are the final gatekeeper — no signal passes without backtesting proof."
    ),
    tools=[run_backtest],
    llm=_get_llm(),
    verbose=True,
)

# ── Agent 6: Natural Language Query ──────────────────────────────────────────

nl_query_agent = Agent(
    role="Natural Language Financial Advisor",
    goal=(
        "Answer user questions about their portfolio, market conditions, and stock analysis "
        "in plain English. Always cite specific data from predictions, signals, and portfolio metrics."
    ),
    backstory=(
        "A friendly quantitative advisor who explains complex financial data in plain English. "
        "You combine predictions, technical signals, and portfolio data to give comprehensive answers."
    ),
    tools=[get_prediction, get_technical_signals, optimize_portfolio, get_live_quote],
    llm=_get_llm(),
    verbose=True,
)
