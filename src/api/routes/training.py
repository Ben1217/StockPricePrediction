"""
Training API routes — trigger model training, check status, list models.
"""

import uuid
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import TrainRequest, TrainResponse, TrainStatus, ModelInfo

router = APIRouter()

# In-memory job tracker
_jobs = {}  # job_id -> TrainStatus


def _run_training(job_id: str, req: TrainRequest):
    """Background training worker."""
    try:
        _jobs[job_id].status = "running"
        _jobs[job_id].progress = 0.1

        from src.data.data_loader import download_stock_data
        from src.features.technical_indicators import add_all_technical_indicators
        from src.features.feature_engineering import (
            create_features, create_target_variable, prepare_features_for_model, create_sequences
        )
        from src.models.model_trainer import ModelTrainer
        from datetime import timedelta

        # Download data
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=req.lookback_days)).strftime("%Y-%m-%d")
        df = download_stock_data(req.symbol, start_date, end_date)
        if df.empty:
            raise ValueError(f"No data for {req.symbol}")

        _jobs[job_id].progress = 0.3

        # Feature engineering
        df = create_features(df)
        df = create_target_variable(df, target_type="return", horizon=req.horizons[0])
        df = df.dropna()

        _jobs[job_id].progress = 0.5

        # Prepare features
        feature_cols = [c for c in df.columns
                        if c not in ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Target"]]
        X_train, X_test, y_train, y_test, scaler = prepare_features_for_model(
            df, feature_columns=feature_cols, test_size=req.test_size
        )

        _jobs[job_id].progress = 0.6

        # Train
        trainer = ModelTrainer()
        model_type = req.model_type.value

        if model_type == "lstm":
            X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length=60)
            X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length=60)
            trainer.train_model(model_type, X_train_seq, y_train_seq,
                                X_val=X_test_seq, y_val=y_test_seq, params=req.params, save=True)
            results = trainer.evaluate_all_models(X_test_seq, y_test_seq)
        else:
            trainer.train_model(model_type, X_train, y_train,
                                X_val=X_test, y_val=y_test, params=req.params, save=True)
            results = trainer.evaluate_all_models(X_test, y_test)

        _jobs[job_id].progress = 0.9

        # Save metadata
        meta_dir = Path("models/model_metadata")
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": model_type,
            "symbol": req.symbol,
            "trained_at": datetime.now().isoformat(),
            "horizons": req.horizons,
            "lookback_days": req.lookback_days,
            "test_size": req.test_size,
            "samples": len(df),
            "features": len(feature_cols),
            "metrics": results.to_dict() if hasattr(results, "to_dict") else {},
            "params": req.params or {},
        }
        with open(meta_dir / f"{model_type}_{req.symbol}_{job_id[:8]}.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Save feature columns
        with open(meta_dir / "feature_columns.json", "w") as f:
            json.dump(feature_cols, f)

        _jobs[job_id].status = "completed"
        _jobs[job_id].progress = 1.0
        _jobs[job_id].metrics = meta.get("metrics", {})

    except Exception as e:
        _jobs[job_id].status = "failed"
        _jobs[job_id].error = str(e)


@router.post("/train", response_model=TrainResponse)
async def train_model(req: TrainRequest):
    """Trigger model training (runs in background)."""
    job_id = str(uuid.uuid4())
    _jobs[job_id] = TrainStatus(job_id=job_id, status="pending")

    thread = threading.Thread(target=_run_training, args=(job_id, req), daemon=True)
    thread.start()

    return TrainResponse(
        job_id=job_id,
        status="pending",
        model_type=req.model_type.value,
        symbol=req.symbol,
        message=f"Training {req.model_type.value} for {req.symbol} started",
    )


@router.get("/status/{job_id}", response_model=TrainStatus)
async def get_training_status(job_id: str):
    """Poll training job status."""
    if job_id not in _jobs:
        raise HTTPException(404, f"Job {job_id} not found")
    return _jobs[job_id]


@router.get("/models")
async def list_models():
    """List saved models with metadata."""
    meta_dir = Path("models/model_metadata")
    if not meta_dir.exists():
        return {"models": []}

    models = []
    for f in meta_dir.glob("*.json"):
        if f.name == "feature_columns.json":
            continue
        try:
            with open(f) as fh:
                meta = json.load(fh)
                meta["model_id"] = f.stem
                models.append(meta)
        except Exception:
            continue

    models.sort(key=lambda m: m.get("trained_at", ""), reverse=True)
    return {"models": models}


@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get details for a specific model."""
    meta_dir = Path("models/model_metadata")
    path = meta_dir / f"{model_id}.json"
    if not path.exists():
        raise HTTPException(404, f"Model {model_id} not found")
    with open(path) as f:
        return json.load(f)
