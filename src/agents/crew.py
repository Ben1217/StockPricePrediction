"""
CrewAI Crew assembly — combines agents + tasks into executable workflows.

Two main entry points:
- run_analysis_crew(tickers)  — full 5-agent pipeline
- run_nl_query(question)      — single-agent NL query
"""

import logging
from crewai import Crew, Process
from .agents import (
    market_monitor_agent,
    prediction_agent,
    ta_agent,
    portfolio_agent,
    backtest_agent,
    nl_query_agent,
)
from .tasks import (
    create_monitor_task,
    create_prediction_task,
    create_ta_task,
    create_portfolio_task,
    create_backtest_task,
    create_nl_query_task,
)

logger = logging.getLogger(__name__)


def run_analysis_crew(tickers: list[str]) -> dict:
    """
    Full analysis pipeline: monitor → predict → TA → portfolio → validate.

    Args:
        tickers: List of stock ticker symbols.

    Returns:
        CrewAI result dict with outputs from all 5 agents.
    """
    logger.info(f"Starting analysis crew for tickers: {tickers}")

    tasks = [
        create_monitor_task(tickers),
        create_prediction_task(tickers),
        create_ta_task(tickers),
        create_portfolio_task(tickers),
        create_backtest_task(tickers),
    ]

    crew = Crew(
        agents=[
            market_monitor_agent,
            prediction_agent,
            ta_agent,
            portfolio_agent,
            backtest_agent,
        ],
        tasks=tasks,
        process=Process.sequential,  # tasks run in order, each sees prior context
        verbose=True,
    )

    result = crew.kickoff()
    logger.info("Analysis crew completed")
    return {"result": str(result), "tickers": tickers, "status": "completed"}


def run_nl_query(question: str) -> dict:
    """
    Answer a plain-English question about stocks or portfolio.

    Args:
        question: Natural language question from the user.

    Returns:
        Dict with the plain-English answer and metadata.
    """
    logger.info(f"NL query: {question}")

    task = create_nl_query_task(question)
    crew = Crew(
        agents=[nl_query_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()
    return {"answer": str(result), "question": question, "status": "completed"}
