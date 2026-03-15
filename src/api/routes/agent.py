"""
Agent API routes — natural language queries and full analysis pipeline.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about stocks or portfolio")


class QueryResponse(BaseModel):
    answer: str
    question: str
    status: str


class AnalysisRequest(BaseModel):
    tickers: list[str] = Field(
        default=["AAPL", "MSFT", "GOOGL"],
        description="List of stock ticker symbols to analyse",
    )


class AnalysisResponse(BaseModel):
    result: str
    tickers: list[str]
    status: str


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def natural_language_query(req: QueryRequest):
    """
    Ask a plain-English question about stocks, portfolio, or market conditions.
    The NL Query Agent will use available tools to fetch data and reason about it.

    Example questions:
    - 'Which stock has the strongest buy signal?'
    - 'Should I rebalance today given current volatility?'
    - 'What is the predicted price for AAPL in 30 days?'
    """
    try:
        from src.agents.crew import run_nl_query
        result = run_nl_query(req.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"NL query failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Agent query failed: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
async def run_full_analysis(req: AnalysisRequest):
    """
    Run the full 5-agent analysis pipeline:
    Monitor → Predict → Technical Analysis → Portfolio Optimize → Backtest Validate.

    Returns results from all agents in the pipeline.
    """
    if not req.tickers:
        raise HTTPException(400, detail="At least one ticker is required")

    try:
        from src.agents.crew import run_analysis_crew
        result = run_analysis_crew(req.tickers)
        return AnalysisResponse(**result)
    except Exception as e:
        logger.error(f"Analysis crew failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Agent analysis failed: {str(e)}")
