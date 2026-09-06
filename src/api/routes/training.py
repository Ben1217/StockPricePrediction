"""
Training API routes — trigger model training, check status, list models.
"""

import logging
import math
import threading

from fastapi import APIRouter, HTTPException

from src.api.jobs import JobRegistry
from src.api.schemas.schemas import (
    TRAINABLE_MODEL_TYPES,
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

# Bounded, TTL'd and thread-safe; see src.api.jobs for why the locking is not
# optional and why this per-process scope is what stops the API running with
# --workers > 1.
_jobs: JobRegistry[TrainStatus] = JobRegistry()


def _run_training(job_id: str, req: TrainRequest):
    """Background training worker for one symbol/model pair."""
    job = _jobs.get(job_id)
    if job is None:  # evicted before the worker started
        return
    try:
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
        logger.exception("Training job %s failed", job_id)
        job.status = "failed"
        job.error = str(exc)


def _run_bootstrap_training(job_id: str, req: BootstrapTrainRequest):
    """Background training worker for multiple symbols and model types."""
    job = _jobs.get(job_id)
    if job is None:  # evicted before the worker started
        return
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
def train_model(req: TrainRequest):
    """Trigger exact-bundle training for one symbol/model pair."""
    if req.model_type not in TRAINABLE_MODEL_TYPES:
        # Kronos, TimesFM and Chronos are pre-trained and do no gradient updates
        # here. Accepting the job and reporting success would be a lie.
        raise HTTPException(
            400,
            f"{req.model_type.value} is a zero-shot foundation model and is served without "
            f"training. Trainable model types: "
            f"{', '.join(sorted(m.value for m in TRAINABLE_MODEL_TYPES))}.",
        )

    job_id = _jobs.new_id()
    _jobs.create(job_id, TrainStatus(job_id=job_id, status="pending"))

    thread = threading.Thread(target=_run_training, args=(job_id, req), daemon=True)
    thread.start()

    return TrainResponse(
        job_id=job_id,
        status="pending",
        model_type=req.model_type.value,
        symbol=req.symbol.upper(),
        message=(
            f"Training {req.model_type.value} bundle for {req.symbol.upper()} started"
        ),
    )


@router.post("/bootstrap", response_model=BootstrapTrainResponse)
def bootstrap_training(req: BootstrapTrainRequest):
    """Trigger background training for the supported stock/model grid."""
    job_id = _jobs.new_id()
    _jobs.create(job_id, TrainStatus(job_id=job_id, status="pending"))

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
def get_training_status(job_id: str):
    """Poll training job status."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found or expired")
    return job


def _json_safe(value):
    """
    Replace non-finite floats with None.

    Training metrics legitimately contain NaN (e.g. benchmark_return when the test
    window has no benchmark), and NaN/Infinity are not valid JSON — serialising them
    raised `ValueError: Out of range float values are not JSON compliant` and turned
    this endpoint into a 500.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@router.get("/models")
def list_models():
    """List saved bundles with metadata."""
    models = list_model_metadata()
    for meta in models:
        meta["model_id"] = meta.get("version_id")
    return {"models": _json_safe(models)}


@router.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Get details for a specific bundle version."""
    meta = get_model_metadata(model_id)
    if meta is None:
        raise HTTPException(404, f"Model {model_id} not found")
    return _json_safe(meta)
