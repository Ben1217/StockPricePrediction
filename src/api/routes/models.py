"""
Model readiness and automatic preparation.

The contract this router exists to hold is that **selecting a ticker is enough**.
A client asks what a symbol can serve; if the answer is "not everything yet", the
same call starts the work and hands back a job to watch. No client ever needs to
know that bundles are trained per horizon, that the direction gauge is gated on a
walk-forward report, or that a Python command exists.

Routes:
    GET  /api/models/prepare/{job_id}    poll one preparation job
    POST /api/models/{symbol}/prepare    start (or join) preparation
    GET  /api/models/{symbol}            readiness, plus any job already running

``/prepare/{job_id}`` is declared before ``/{symbol}`` so a job id can never be
read as a ticker.

The readiness route is deliberately cheap — it stats files and parses metadata,
nothing more — because the dashboard calls it on every ticker change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.models.direction_models import MODEL_FACTORIES as DIRECTION_MODEL_FACTORIES
from src.models.model_manager import (
    DEFAULT_DIRECTION_MODEL,
    PREPARATION_HORIZONS,
    assess_symbol,
    auto_prepare_enabled,
)
from src.models.preparation import ensure_prepared, registry
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class PrepareRequest(BaseModel):
    """Options for an explicit preparation request."""

    horizons: Optional[List[int]] = None
    direction_model: str = Field(default=DEFAULT_DIRECTION_MODEL)
    #: Bypass the post-attempt cooldown. This is the Retry button: a user who has
    #: read an error and asked again should not be told to wait fifteen minutes.
    force: bool = Field(default=False)


def _validate_direction_model(model: str) -> str:
    if model not in DIRECTION_MODEL_FACTORIES:
        raise HTTPException(
            422,
            f"Unknown direction model '{model}'. Available: {sorted(DIRECTION_MODEL_FACTORIES)}",
        )
    return model


def _symbol_payload(
    symbol: str,
    direction_model: str,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Readiness plus whatever preparation job is already attached to the symbol."""
    report = assess_symbol(
        symbol,
        horizons=horizons or PREPARATION_HORIZONS,
        direction_model=direction_model,
    )
    job = registry.latest_for(symbol)
    return {
        "symbol": report.symbol,
        "readiness": report.as_dict(),
        "preparation": job.as_dict() if job else None,
        "auto_prepare": auto_prepare_enabled(),
    }


@router.get("/prepare/{job_id}")
def get_preparation_status(job_id: str) -> Dict[str, Any]:
    """Poll one preparation job. 404 once it has aged out of the tracker."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(404, f"Preparation job {job_id} not found or expired")
    return job.as_dict()


@router.post("/{symbol}/prepare")
def prepare_symbol(symbol: str, req: Optional[PrepareRequest] = None) -> Dict[str, Any]:
    """
    Train whatever ``symbol`` is missing, in the background.

    Idempotent by design. Called twice while a job runs, it returns the same job
    both times; called when nothing needs training, it returns the readiness with
    no job attached. A client can therefore fire this on every ticker change
    without reasoning about state.

    Note this ignores ``QUANTVISION_AUTO_PREPARE``: that flag governs whether a
    *prediction* request may start training on its own, and an explicit POST here
    is not that — it is someone asking directly.
    """
    request = req or PrepareRequest()
    direction_model = _validate_direction_model(request.direction_model)
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(422, "A symbol is required")

    job = ensure_prepared(
        symbol,
        horizons=request.horizons or PREPARATION_HORIZONS,
        direction_model=direction_model,
        force=request.force,
        respect_auto_flag=False,
    )

    payload = _symbol_payload(symbol, direction_model, request.horizons)
    payload["preparation"] = job.as_dict() if job else payload["preparation"]
    # "A job is attached", not "a job is still running" — a fast job that already
    # finished did do the work, and reporting False would read as "nothing to do".
    payload["started"] = job is not None
    return payload


@router.get("/{symbol}")
def get_model_readiness(
    symbol: str,
    direction_model: str = Query(DEFAULT_DIRECTION_MODEL, alias="model"),
    prepare: bool = Query(
        False,
        description="Start preparation if the symbol needs it, instead of only reporting",
    ),
) -> Dict[str, Any]:
    """
    What this symbol can serve right now, and what it is still missing.

    Read-only unless ``prepare=true``, which makes it equivalent to the POST
    above — offered because a client that wants "tell me and fix it" in one round
    trip should not have to make two.
    """
    direction_model = _validate_direction_model(direction_model)
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(422, "A symbol is required")

    if prepare:
        ensure_prepared(
            symbol,
            horizons=PREPARATION_HORIZONS,
            direction_model=direction_model,
        )

    return _symbol_payload(symbol, direction_model)
