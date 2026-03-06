"""
FastAPI application entry point.
Stock Price Prediction & Portfolio Optimization API.
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes.data import router as data_router
from src.api.routes.training import router as training_router
from src.api.routes.predict import router as predict_router
from src.api.routes.backtest import router as backtest_router
from src.api.routes.portfolio import router as portfolio_router
from src.api.routes.export import router as export_router
from src.api.routes.patterns import router as patterns_router


app = FastAPI(
    title="QuantVision API",
    description="Stock Price Prediction & Portfolio Optimization",
    version="2.0.0",
)

# CORS – allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(data_router,      prefix="/api/data",      tags=["Data"])
app.include_router(training_router,  prefix="/api/training",  tags=["Training"])
app.include_router(predict_router,   prefix="/api/predict",   tags=["Predictions"])
app.include_router(backtest_router,  prefix="/api/backtest",  tags=["Backtesting"])
app.include_router(portfolio_router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(export_router,    prefix="/api/export",    tags=["Export"])
app.include_router(patterns_router,  prefix="/api/patterns",  tags=["Patterns"])


@app.get("/")
async def root():
    return {
        "name": "QuantVision API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": [
            "/api/data", "/api/training", "/api/predict",
            "/api/backtest", "/api/portfolio", "/api/export",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}
