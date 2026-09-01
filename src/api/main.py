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

# Load .env before anything reads os.getenv. It was only ever loaded as a side
# effect of importing src.utils.config_loader further down the import graph, which
# made QUANTVISION_CORS_ORIGINS and QUANTVISION_API_KEY depend on import order.
try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:  # python-dotenv is optional; real env vars still apply
    pass

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
from src.api.routes.direction import router as direction_router
from src.api.routes.models import router as models_router
from src.api.security import (
    cors_headers_for,
    error_middleware,
    log_auth_status,
    parse_origins,
    security_middleware,
)


app = FastAPI(
    title="QuantVision API",
    description="Stock Price Prediction & Portfolio Optimization",
    version="2.0.0",
)
logger = logging.getLogger(__name__)

# Uvicorn configures its own loggers but leaves the root logger bare, so without
# this the tracebacks below are formatted by logging's last-resort handler.
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

# CORS. Origins come from QUANTVISION_CORS_ORIGINS (comma-separated) so the same
# build can serve a deployed frontend; the fallback is the local dev servers.
_DEV_ORIGINS = [
    "http://localhost:3000",
    *[f"http://localhost:{port}" for port in range(5173, 5181)],
    *[f"http://127.0.0.1:{port}" for port in range(5173, 5181)],
]
CORS_ORIGINS = parse_origins(os.getenv("QUANTVISION_CORS_ORIGINS", ""), _DEV_ORIGINS)

# ── Middleware order ─────────────────────────────────────────────────────────
# This does NOT read top-to-bottom. Starlette inserts each registration at the
# front of the list, so the LAST one registered is the OUTERMOST at runtime:
#
#     ServerErrorMiddleware        <- holds the @app.exception_handler(Exception) below
#       CORSMiddleware             <- the only layer that adds CORS headers
#         error_middleware         <- unhandled exceptions become a 500 *under* CORS
#           security_middleware    <- API key + rate limit; its 401/429 get CORS headers
#             ExceptionMiddleware  <- HTTPException / 422 validation
#               router
#
# error_middleware has to stay below CORSMiddleware. A 500 raised above it reaches
# the browser with no Access-Control-Allow-Origin, and Chrome then reports a working
# server as a CORS failure — hiding the traceback that is the real problem.
app.middleware("http")(security_middleware)
app.middleware("http")(error_middleware)
log_auth_status()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
app.include_router(direction_router, prefix="/api/direction", tags=["Direction"])
app.include_router(models_router,    prefix="/api/models",    tags=["Models"])


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
            "/api/direction", "/api/models",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/db")
async def health_db():
    """
    Report the backend's own connection to TimescaleDB.

    Always 200, never raises: this is a diagnostic, and a caller checking whether
    the database is reachable should read the answer rather than handle an error.
    `connected: false` with a `detail` is the useful response when it is down.
    """
    from src.data.timescale_store import (
        ping,
        safe_connection_string,
        timescale_enabled,
    )

    if not timescale_enabled():
        return {
            "connected": False,
            "backend": "sqlite",
            "detail": "DB_TYPE is not set to timescale; the API is not using PostgreSQL.",
        }

    try:
        return {"backend": "timescaledb", **ping()}
    except Exception as exc:
        logger.warning("TimescaleDB health check failed: %s", exc)
        return {
            "connected": False,
            "backend": "timescaledb",
            "connection": safe_connection_string(),
            "detail": str(exc),
        }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Last resort for failures in the CORS/security layers themselves — route errors
    are caught by `error_middleware` below them. Starlette runs this handler in
    ServerErrorMiddleware, which sits *above* CORSMiddleware, so the CORS headers
    have to be attached by hand or the browser reports this 500 as a CORS error.
    """
    logger.exception("Unhandled API exception at %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
        headers=cors_headers_for(request, CORS_ORIGINS),
    )
