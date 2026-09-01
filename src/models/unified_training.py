"""
Training pipeline for the unified bundles.

Produces the artifacts the API serves: a regressor, a classifier, the scaler
they were fitted with, and the exact feature column list, written into the
canonical bundle layout at ``models/bundles/<SYMBOL>/<MODEL_TYPE>/``.

Training and evaluation share one feature builder
-------------------------------------------------
:func:`~src.features.direction_features.build_direction_dataset` produces the
rows here, in ``scripts/unified_benchmark.py``, and behind the serving route.
That is deliberate: a bundle whose serving features were assembled by a second,
subtly different code path is a bundle whose benchmark numbers describe a model
nobody is running.

Horizon
-------
These bundles forecast one timeframe ahead. The horizon is a parameter rather
than a constant so a 15-minute or hourly deployment can retrain against its own
bar size, but a single bundle answers for exactly one horizon and records which.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.data_loader import download_stock_data
from src.features.direction_features import DIRECTION_FEATURE_CONFIG, build_direction_dataset
from src.models.model_bundle import (
    CANONICAL_BUNDLE_LAYOUT,
    LEGACY_METADATA_DIR,
    MODEL_FILE_NAMES,
)
from src.models.unified_models import build_unified_model, UNIFIED_FACTORIES
from src.utils.logger import get_logger

logger = get_logger(__name__)

BUNDLES_DIR = Path("models/bundles")
DEFAULT_LOOKBACK_DAYS = 1825
DEFAULT_HORIZON = 1

# Three years of bars. Below this a fold's training window is too short for the
# 60-bar LSTM windows to leave a usable number of training sequences.
MIN_TRAINING_ROWS = 756

# Held back from the end of the series so the saved metadata carries an honest
# out-of-sample score. Chronological, never sampled.
DEFAULT_TEST_SIZE = 0.15


def _bundle_dir_for(symbol: str, model_type: str, bundles_dir: Path = BUNDLES_DIR) -> Path:
    return bundles_dir / symbol.upper() / model_type


def train_unified_bundle(
    *,
    symbol: str,
    model_type: str,
    horizon: int = DEFAULT_HORIZON,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    test_size: float = DEFAULT_TEST_SIZE,
    params: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
    bundles_dir: Path = BUNDLES_DIR,
) -> Dict[str, Any]:
    """
    Train and persist one unified bundle.

    Returns the metadata written alongside the model, which is also what the
    training API echoes back to the caller.
    """
    symbol = symbol.upper()
    horizon = int(horizon)
    if model_type not in UNIFIED_FACTORIES:
        raise ValueError(f"Unsupported unified model type {model_type!r}; known: {sorted(UNIFIED_FACTORIES)}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    logger.info("Training %s for %s at horizon %d", model_type, symbol, horizon)

    if raw_df is None:
        end = datetime.now()
        start = end - pd.Timedelta(days=lookback_days)
        raw_df = download_stock_data(symbol, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if raw_df is None or raw_df.empty:
        raise ValueError(f"No data available for {symbol}")

    dataset = build_direction_dataset(raw_df, horizon=horizon)
    if len(dataset) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient data for {symbol}: {len(dataset)} usable rows, need {MIN_TRAINING_ROWS}"
        )

    X = dataset.features
    y_return = dataset.forward_return.to_numpy(dtype=np.float64)
    y_direction = dataset.labels.to_numpy(dtype=np.int8)

    # Built through the shared registry so a served bundle carries the same
    # hyperparameters the benchmark reported for that model.
    model = build_unified_model(model_type, params)
    sequence_length = max(1, int(getattr(model, "sequence_length", 1)))

    # Chronological hold-out, with the label horizon purged between the two
    # windows so the last training label cannot resolve inside the test block.
    n_rows = len(X)
    n_test = max(1, int(n_rows * test_size))
    train_end = n_rows - n_test - horizon
    if train_end < MIN_TRAINING_ROWS // 2:
        raise ValueError(f"Training window for {symbol} is too short after the hold-out split")

    train_pos = np.arange(0, train_end)
    test_pos = np.arange(n_rows - n_test, n_rows)

    scaler = StandardScaler().fit(X.to_numpy(dtype=np.float64)[train_pos])
    fold = _fold_from(X, scaler, train_pos, test_pos)

    model.fit(fold, y_return, y_direction)
    metrics = _holdout_metrics(model, fold, dataset, y_return, y_direction)

    # ── Persist ───────────────────────────────────────────────────────────
    bundle_dir = _bundle_dir_for(symbol, model_type, bundles_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_path = bundle_dir / MODEL_FILE_NAMES.get(model_type, "model.joblib")
    scaler_path = bundle_dir / "scaler.joblib"
    feature_columns_path = bundle_dir / "feature_columns.json"

    model.save(str(model_path))
    joblib.dump(scaler, scaler_path)
    feature_columns_path.write_text(json.dumps(dataset.feature_columns, indent=2), encoding="utf-8")

    version_id = f"{model_type}_{symbol}_h{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metadata: Dict[str, Any] = {
        "version_id": version_id,
        "model_type": model_type,
        "symbol": symbol,
        "training_symbol": symbol,
        "horizon": horizon,
        "training_horizon": horizon,
        "horizons": [horizon],
        "objective": "unified_price_and_direction",
        "target_type": "unified",
        "trained_at": datetime.now().isoformat(),
        "lookback_days": lookback_days,
        "feature_columns": list(dataset.feature_columns),
        "feature_count": len(dataset.feature_columns),
        "feature_config": dict(DIRECTION_FEATURE_CONFIG),
        # LoadedModelBundle reads scaler_type and sequence_length from here, so
        # they have to live under "preprocessing" rather than at the top level.
        "preprocessing": {
            "scaler_type": "standard",
            "sequence_length": sequence_length,
        },
        "training_sample_count": int(len(train_pos)),
        "train_base_rate": float(np.mean(y_direction[train_pos])),
        "holdout": metrics,
        "bundle_layout": CANONICAL_BUNDLE_LAYOUT,
        "bundle_dir": str(bundle_dir),
        "artifact_dir": str(bundle_dir),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_columns_path": str(feature_columns_path),
    }

    (bundle_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )
    LEGACY_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (LEGACY_METADATA_DIR / f"{version_id}.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    logger.info(
        "Saved %s for %s to %s (hold-out accuracy %.4f vs base rate %.4f)",
        model_type,
        symbol,
        bundle_dir,
        metrics.get("direction_accuracy", float("nan")),
        metrics.get("base_rate", float("nan")),
    )
    return metadata


def _fold_from(
    X: pd.DataFrame, scaler: StandardScaler, train_pos: np.ndarray, test_pos: np.ndarray
):
    """Package the scaled matrix into the FoldInputs the model layer expects."""
    from src.models.unified_models import FoldInputs

    values = scaler.transform(X.to_numpy(dtype=np.float64))
    return FoldInputs(
        X_scaled=np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        index=X.index,
        feature_columns=list(X.columns),
        train_pos=train_pos,
        test_pos=test_pos,
        scaler=scaler,
    )


def _holdout_metrics(model, fold, dataset, y_return: np.ndarray, y_direction: np.ndarray) -> Dict[str, Any]:
    """
    Score the bundle on its chronological hold-out.

    Recorded in the metadata so the API can tell a caller how the model it is
    serving actually performed, rather than only that it exists. The base rate
    sits next to the accuracy because an accuracy alone says nothing.
    """
    from src.models.unified_evaluation import price_metrics

    if dataset.ohlcv is not None and "Close" in dataset.ohlcv.columns:
        prev_close = dataset.ohlcv["Close"].to_numpy(dtype=np.float64)
    else:
        prev_close = np.full(len(dataset.features), np.nan)

    positions = fold.test_rows(getattr(model, "sequence_length", 1))[1]
    if len(positions) == 0:
        return {}

    predicted_price = np.asarray(model.predict_price(fold, prev_close), dtype=np.float64)
    p_up, _ = model.predict_direction_proba(fold)
    p_up = np.asarray(p_up, dtype=np.float64)

    realised = prev_close[positions] * (1.0 + y_return[positions])
    valid = np.isfinite(predicted_price) & np.isfinite(realised) & np.isfinite(p_up)
    if valid.sum() < 2:
        return {}

    positions = positions[valid]
    truth = y_direction[positions]
    predicted_direction = (p_up[valid] >= 0.5).astype(int)

    scores = price_metrics(realised[valid], predicted_price[valid], prev_close[positions])
    return {
        "n_test": int(valid.sum()),
        "test_start": str(fold.index[positions[0]].date()),
        "test_end": str(fold.index[positions[-1]].date()),
        "base_rate": float(np.mean(truth)),
        "direction_accuracy": float(np.mean(predicted_direction == truth)),
        **{f"price_{key}": round(value, 6) for key, value in scores.items()},
    }


def train_all_unified_bundles(
    symbol: str,
    *,
    model_types: Optional[List[str]] = None,
    horizon: int = DEFAULT_HORIZON,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    raw_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Train every unified bundle for one symbol, downloading the bars once.

    A model that fails to train is reported rather than raised: one broken model
    should not cost the caller the others.
    """
    model_types = list(model_types or UNIFIED_FACTORIES)
    if raw_df is None:
        end = datetime.now()
        start = end - pd.Timedelta(days=lookback_days)
        raw_df = download_stock_data(symbol.upper(), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    trained: Dict[str, Any] = {}
    failed: Dict[str, str] = {}
    for model_type in model_types:
        try:
            trained[model_type] = train_unified_bundle(
                symbol=symbol,
                model_type=model_type,
                horizon=horizon,
                lookback_days=lookback_days,
                raw_df=raw_df,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            logger.error("Training %s for %s failed: %s", model_type, symbol, exc, exc_info=True)
            failed[model_type] = str(exc)

    return {"symbol": symbol.upper(), "horizon": horizon, "trained": trained, "failed": failed}
