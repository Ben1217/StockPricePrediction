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


AGENTS_EXTRA_HINT = (
    "The agent routes require the optional 'agents' extra: "
    'pip install -e ".[agents]"'
)


def _load_agents(attr: str):
    """
    Import a callable from src.agents, translating a missing optional dependency
    into an actionable 503 rather than a bare ImportError in a 500.
    """
    try:
        import src.agents.crew as crew
    except ImportError as exc:
        logger.warning("Agent stack unavailable: %s", exc)
        raise HTTPException(503, detail=f"{AGENTS_EXTRA_HINT} ({exc})") from exc
    return getattr(crew, attr)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
def natural_language_query(req: QueryRequest):
    """
    Ask a plain-English question about stocks, portfolio, or market conditions.
    The NL Query Agent will use available tools to fetch data and reason about it.

    Example questions:
    - 'Which stock has the strongest buy signal?'
    - 'Should I rebalance today given current volatility?'
    - 'What is the predicted price for AAPL in 30 days?'
    """
    run_nl_query = _load_agents("run_nl_query")
    try:
        result = run_nl_query(req.question)
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"NL query failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Agent query failed: {str(e)}")


@router.post("/analyze", response_model=AnalysisResponse)
def run_full_analysis(req: AnalysisRequest):
    """
    Run the full 5-agent analysis pipeline:
    Monitor → Predict → Technical Analysis → Portfolio Optimize → Backtest Validate.

    Returns results from all agents in the pipeline.
    """
    if not req.tickers:
        raise HTTPException(400, detail="At least one ticker is required")

    run_analysis_crew = _load_agents("run_analysis_crew")
    try:
        result = run_analysis_crew(req.tickers)
        return AnalysisResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis crew failed: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Agent analysis failed: {str(e)}")
