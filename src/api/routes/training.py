"""
Training API routes — trigger model training, check status, list models.
"""

import logging
import threading
import uuid

from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import (
    BootstrapTrainRequest,
    BootstrapTrainResponse,
    TrainRequest,
    TrainResponse,
    TrainStatus,
)
from src.models.bundle_training import (
    DEFAULT_BOOTSTRAP_MODELS,
    DEFAULT_BOOTSTRAP_SYMBOLS,
    normalize_horizons,
    train_batch_model_bundles,
    train_model_bundles,
)
from src.models.model_bundle import get_model_metadata, list_model_metadata

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job tracker
_jobs = {}  # job_id -> TrainStatus


def _run_training(job_id: str, req: TrainRequest):
    """Background training worker for one symbol/model pair."""
    try:
        job = _jobs[job_id]
        job.status = "running"
        job.progress = 0.1

        horizons = normalize_horizons(req.horizons)
        result = train_model_bundles(
            symbol=req.symbol,
            model_type=req.model_type.value,
            horizons=horizons,
            lookback_days=req.lookback_days,
            test_size=req.test_size,
            params=req.params,
            progress_callback=lambda idx, total, _: setattr(
                job,
                "progress",
                0.2 + 0.75 * (idx / total),
            ),
        )

        job.status = "completed"
        job.progress = 1.0
        job.metrics = result
    except Exception as exc:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(exc)


def _run_bootstrap_training(job_id: str, req: BootstrapTrainRequest):
    """Background training worker for multiple symbols and model types."""
    job = _jobs[job_id]
    job.status = "running"
    job.progress = 0.05

    try:
        model_types = [model_type.value for model_type in (req.model_types or [])] or list(DEFAULT_BOOTSTRAP_MODELS)
        horizons = normalize_horizons(req.horizons)
        result = train_batch_model_bundles(
            symbols=req.symbols or DEFAULT_BOOTSTRAP_SYMBOLS,
            use_sp500=req.use_sp500,
            model_types=model_types,
            horizons=horizons,
            lookback_days=req.lookback_days,
            test_size=req.test_size,
            params=req.params,
            skip_fresh_hours=req.skip_fresh_hours,
            progress_callback=lambda completed, total, *_: setattr(
                job,
                "progress",
                0.05 + 0.9 * (completed / total),
            ),
        )
        job.status = "completed"
        job.progress = 1.0
        job.metrics = result
        if result.get("failure_count"):
            job.error = f"{result['failure_count']} bootstrap training run(s) failed"
    except Exception as exc:
        logger.exception("Bootstrap training job failed")
        job.status = "failed"
        job.error = str(exc)


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    """Trigger exact-bundle training for one symbol/model pair."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = TrainStatus(job_id=job_id, status="pending")

    thread = threading.Thread(target=_run_training, args=(job_id, req), daemon=True)
    thread.start()

    horizons = normalize_horizons(req.horizons)
    return TrainResponse(
        job_id=job_id,
        status="pending",
        model_type=req.model_type.value,
        symbol=req.symbol.upper(),
        message=(
            f"Training {req.model_type.value} next-day direction bundle for "
            f"{req.symbol.upper()} started"
        ),
    )


@router.post("/bootstrap", response_model=BootstrapTrainResponse)
async def bootstrap_training(req: BootstrapTrainRequest):
    """Trigger background training for the supported stock/model grid."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = TrainStatus(job_id=job_id, status="pending")

    thread = threading.Thread(target=_run_bootstrap_training, args=(job_id, req), daemon=True)
    thread.start()

    symbols = [] if req.use_sp500 else [symbol.upper() for symbol in (req.symbols or DEFAULT_BOOTSTRAP_SYMBOLS)]
    model_types = [model_type.value for model_type in (req.model_types or [])] or list(DEFAULT_BOOTSTRAP_MODELS)
    return BootstrapTrainResponse(
        job_id=job_id,
        status="pending",
        symbols=symbols,
        model_types=model_types,
        message=(
            "Bootstrap training started for next-day direction bundles across the S&P 500 universe"
            if req.use_sp500
            else f"Bootstrap training started for {len(symbols)} symbols across {len(model_types)} model types"
        ),
    )


@router.get("/status/{job_id}", response_model=TrainStatus)
async def get_training_status(job_id: str):
    """Poll training job status."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


@router.get("/models")
async def list_models():
    """List saved bundles with metadata."""
    models = list_model_metadata()
    for meta in models:
        meta["model_id"] = meta.get("version_id")
    return {"models": models}


@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get details for a specific bundle version."""
    meta = get_model_metadata(model_id)
    if meta is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return meta
