"""
FastAPI application entry point.
Stock Price Prediction & Portfolio Optimization API.
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from src.api.routes.data import router as data_router
from src.api.routes.training import router as training_router
from src.api.routes.predict import router as predict_router
from src.api.routes.backtest import router as backtest_router
from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.export import router as export_router
from src.api.routes.patterns import router as patterns_router
from src.api.routes.agent import router as agent_router
from src.api.routes.sentiment import router as sentiment_router
from src.api.security import log_auth_status, parse_origins, security_middleware


app = FastAPI(
    title="QuantVision API",
    description="Stock Price Prediction & Portfolio Optimization",
    version="2.0.0",
)
logger = logging.getLogger(__name__)

# Auth (opt-in via QUANTVISION_API_KEY) + per-client rate limiting.
app.middleware("http")(security_middleware)
log_auth_status()

# CORS. Origins come from QUANTVISION_CORS_ORIGINS (comma-separated) so the same
# build can serve a deployed frontend; the fallback is the local dev servers.
_DEV_ORIGINS = [
    "http://localhost:3000",
    *[f"http://localhost:{port}" for port in range(5173, 5181)],
    *[f"http://127.0.0.1:{port}" for port in range(5173, 5181)],
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_origins(os.getenv("QUANTVISION_CORS_ORIGINS", ""), _DEV_ORIGINS),
    allow_credentials=True,
    # Narrowed from "*": these are the verbs and headers the app actually uses.
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# Register routers
app.include_router(data_router,      prefix="/api/data",      tags=["Data"])
app.include_router(training_router,  prefix="/api/training",  tags=["Training"])
app.include_router(predict_router,   prefix="/api/predict",   tags=["Predictions"])
app.include_router(backtest_router,  prefix="/api/backtest",  tags=["Backtesting"])
app.include_router(portfolio_router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(export_router,    prefix="/api/export",    tags=["Export"])
app.include_router(patterns_router,  prefix="/api/patterns",  tags=["Patterns"])
app.include_router(agent_router,     prefix="/api/agent",     tags=["Agents"])
app.include_router(sentiment_router, prefix="/api/sentiment", tags=["Sentiment"])


@app.get("/")
async def root():
    return {
        "name": "QuantVision API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/data", "/api/training", "/api/predict",
            "/api/backtest", "/api/portfolio", "/api/export",
            "/api/patterns", "/api/agent", "/api/sentiment",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled API exception at %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
