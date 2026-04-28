"""
Ensemble Return Regression Training Pipeline.

Trains XGBoost, RandomForest, and LSTM return regressors for a given symbol
and forecast horizon using 5 years of OHLCV data with chronological splitting.

Bundles are saved under the existing bundle tree:
    models/bundles/<SYMBOL>/<MODEL>/<HORIZON>/

Each bundle contains:
    model.{json|joblib|pt}  — trained regressor artifact
    scaler.joblib           — feature scaler fitted on training data only
    feature_columns.json    — ordered list of feature column names
    metadata.json           — metrics, weights, config

Public API:
    train_regression_bundle(symbol, model_type, horizon, ...)
    train_ensemble_for_symbol(symbol, horizons, ...)
    train_ensemble_batch(symbols, horizons, ...)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd

from src.data.data_loader import download_stock_data
from src.features.feature_engineering import (
    build_regression_dataset,
    create_sequences,
    normalize_feature_config,
    split_dataset_chronologically,
)
from src.models.regression_models import REGRESSOR_FACTORIES, REGRESSOR_FILE_NAMES

from src.utils.logger import get_logger

logger = get_logger(__name__)

REGRESSION_BUNDLES_DIR = Path("models/bundles")
REGRESSION_METADATA_DIR = Path("models/model_metadata")
DEFAULT_LOOKBACK_DAYS = 1825  # ~5 years
DEFAULT_HORIZONS = [7, 15, 30, 60]
DEFAULT_MODEL_TYPES = ["xgboost", "random_forest", "lstm"]
SEQUENCE_LENGTH = 60  # LSTM lookback window
MIN_TRAINING_ROWS = 756  # roughly 3 years of daily trading data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _regression_bundle_dir(symbol: str, model_type: str, horizon: int) -> Path:
    return REGRESSION_BUNDLES_DIR / symbol.upper() / model_type / str(int(horizon))


def validate_historical_data(df: pd.DataFrame, symbol: str):
    """
    Validate OHLCV market data for training suitability.
    Abort on empty data, missing columns/values, duplicates, stale data,
    invalid OHLCV values, or abnormal adjusted-close spikes.
    """
    if df.empty:
        raise ValueError(f"Data is empty for {symbol}")

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required yfinance columns for {symbol}: {missing_cols}")

    if df[required].isna().any().any():
        raise ValueError(f"Missing OHLCV data found for {symbol}")

    if df.index.duplicated().any():
        raise ValueError(f"Duplicate dates found in data for {symbol}")

    last_date = pd.Timestamp(df.index.max())
    if last_date.tzinfo is not None:
        last_date = last_date.tz_convert(None)
    if pd.isna(last_date) or (pd.Timestamp.utcnow().tz_localize(None) - last_date).days > 7:
        raise ValueError(f"Market data for {symbol} is outdated (last date: {last_date})")

    if (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError(f"Zero or negative prices detected for {symbol}")
    if (df["Volume"] < 0).any():
        raise ValueError(f"Negative volume detected for {symbol}")
    if ((df["High"] < df["Low"]) | (df["High"] < df["Open"]) | (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) | (df["Low"] > df["Close"])).any():
        raise ValueError(f"Invalid OHLC price relationships detected for {symbol}")

    daily_returns = df["Adj Close"].pct_change().dropna()
    if (daily_returns.abs() > 0.50).any():
        raise ValueError(f"Abnormal adjusted-close daily price spike (>50%) detected for {symbol}")


def _download_data(symbol: str, lookback_days: int) -> pd.DataFrame:
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = download_stock_data(symbol, start, end)
    if df is None or df.empty:
        raise ValueError(f"No data returned for {symbol}")

    validate_historical_data(df, symbol)

    if len(df) < MIN_TRAINING_ROWS:
        raise ValueError(
            f"Insufficient data for {symbol}: {len(df)} rows "
            f"(need at least {MIN_TRAINING_ROWS} daily rows, about 3 years)"
        )
    return df


def _build_lstm_sequences(X_train, y_train, X_val, y_val, X_test, y_test, seq_len: int):
    """Build overlapping LSTM sequences while preserving chronological boundaries."""
    hist = max(0, seq_len - 1)

    X_tr_seq, y_tr_seq = create_sequences(X_train, y_train, sequence_length=seq_len)

    val_X = np.concatenate([X_train[-hist:] if hist else X_train[:0], X_val], axis=0)
    val_y = np.concatenate([y_train[-hist:] if hist else y_train[:0], y_val], axis=0)
    X_va_seq, y_va_seq = create_sequences(val_X, val_y, sequence_length=seq_len)

    test_X = np.concatenate([X_train, X_val], axis=0)
    test_y = np.concatenate([y_train, y_val], axis=0)
    tpre_X = test_X[-hist:] if hist else test_X[:0]
    tpre_y = test_y[-hist:] if hist else test_y[:0]
    X_te_seq, y_te_seq = create_sequences(
        np.concatenate([tpre_X, X_test], axis=0),
        np.concatenate([tpre_y, y_test], axis=0),
        sequence_length=seq_len,
    )

    if len(X_va_seq) == 0 or len(X_te_seq) == 0:
        raise ValueError("Val or test set too short for LSTM sequence length")

    return X_tr_seq, y_tr_seq, X_va_seq, y_va_seq, X_te_seq, y_te_seq


# ---------------------------------------------------------------------------
# Single model × horizon training
# ---------------------------------------------------------------------------

def train_regression_bundle(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    test_size: float = 0.2,
    val_size: float = 0.1,
    params: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Train one return-regression model bundle.

    Returns a dict with keys: bundle_dir, metrics (test + val), model_type,
    symbol, horizon, feature_columns, trained_at.
    """
    symbol = symbol.upper()
    horizon = int(horizon)
    if horizon not in DEFAULT_HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; supported horizons are {DEFAULT_HORIZONS}")
    if model_type not in REGRESSOR_FACTORIES:
        raise ValueError(f"Unsupported model type {model_type}; supported models are {DEFAULT_MODEL_TYPES}")

    logger.info("Training %s return-regression bundle for %s horizon=%dd", model_type, symbol, horizon)

    df = raw_df.copy() if raw_df is not None else _download_data(symbol, lookback_days)
    feature_config = normalize_feature_config()

    dataset, feature_cols, target_col = build_regression_dataset(df, horizon=horizon, feature_config=feature_config)
    if dataset.empty or not feature_cols:
        raise ValueError(f"No usable rows for {symbol} horizon={horizon}")

    split = split_dataset_chronologically(
        dataset,
        feature_columns=feature_cols,
        target_column=target_col,
        scaler_type="standard",
        test_size=test_size,
        val_size=val_size,
    )

    model_params = dict(params or {})
    if model_type == "lstm":
        seq_len = int(model_params.get("sequence_length", SEQUENCE_LENGTH))
        seq_len = min(seq_len, max(10, len(split["X_train"]) // 2))
        model_params["sequence_length"] = seq_len
        X_tr, y_tr, X_va, y_va, X_te, y_te = _build_lstm_sequences(
            split["X_train"], split["y_train"],
            split["X_val"], split["y_val"],
            split["X_test"], split["y_test"],
            seq_len,
        )
    else:
        X_tr, y_tr = split["X_train"], split["y_train"]
        X_va, y_va = split["X_val"], split["y_val"]
        X_te, y_te = split["X_test"], split["y_test"]

    factory = REGRESSOR_FACTORIES[model_type]
    model = factory(model_params if model_params else None)
    model.fit(X_tr, y_tr, X_val=X_va, y_val=y_va)

    # Compute previous close array for directional accuracy
    val_frame = split["val_frame"]
    test_frame = split["test_frame"]

    prev_val = val_frame["Close"].values.astype(np.float32)
    prev_test = test_frame["Close"].values.astype(np.float32)

    # For LSTM the sequence reduction may mismatch frame length — align safely
    if model_type == "lstm":
        n_va = len(X_va)
        n_te = len(X_te)
        prev_val = prev_val[-n_va:] if len(prev_val) >= n_va else np.full(n_va, prev_val[-1] if len(prev_val) else 1.0)
        prev_test = prev_test[-n_te:] if len(prev_test) >= n_te else np.full(n_te, prev_test[-1] if len(prev_test) else 1.0)

    val_metrics = model.evaluate(X_va, y_va, prev_close=prev_val)
    test_metrics = model.evaluate(X_te, y_te, prev_close=prev_test)

    # Persist bundle
    bundle_dir = _regression_bundle_dir(symbol, model_type, horizon)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_filename = REGRESSOR_FILE_NAMES[model_type]
    model_path = bundle_dir / model_filename
    scaler_path = bundle_dir / "scaler.joblib"
    feat_col_path = bundle_dir / "feature_columns.json"
    meta_path = bundle_dir / "metadata.json"

    trained_at = datetime.now().isoformat()
    version_id = f"{model_type}_{symbol}_price_h{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    model.save(str(model_path))
    joblib.dump(split["scaler"], scaler_path)
    feat_col_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    meta = {
        "version_id": version_id,
        "model_type": model_type,
        "symbol": symbol,
        "horizon": horizon,
        "training_horizon": horizon,
        "horizons": [horizon],
        "objective": "future_return_pct",
        "target_type": "return_regression",
        "target_col": target_col,
        "target_columns": [f"target_return_{h}d" for h in DEFAULT_HORIZONS],
        "model_output": "predicted_return",
        "prediction_formula": "predicted_price = current_price * (1 + predicted_return)",
        "ensemble_weights": {"lstm": 0.40, "xgboost": 0.35, "random_forest": 0.25},
        "trained_at": trained_at,
        "lookback_days": lookback_days,
        "data_source": "yfinance",
        "data_range": {
            "start": str(pd.Timestamp(df.index.min()).date()),
            "end": str(pd.Timestamp(df.index.max()).date()),
        },
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "feature_config": feature_config,
        "scaler_type": "standard",
        "sequence_length": int(model_params.get("sequence_length", SEQUENCE_LENGTH)) if model_type == "lstm" else None,
        "validation_policy": {
            "split": "chronological",
            "train": 1.0 - float(test_size) - float(val_size),
            "validation": float(val_size),
            "test": float(test_size),
            "shuffle": False,
        },
        "split_sizes": {
            "train": len(split["train_frame"]),
            "val": len(split["val_frame"]),
            "test": len(split["test_frame"]),
        },
        "training_rows": len(dataset),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metrics": {
            "validation": val_metrics,
            "test": test_metrics,
        },
        "params": model_params,
        "oob_error": getattr(model, "oob_error_", None),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_columns_path": str(feat_col_path),
        "bundle_dir": str(bundle_dir),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    REGRESSION_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (REGRESSION_METADATA_DIR / f"{version_id}.json").write_text(
        json.dumps(meta, indent=2, default=str),
        encoding="utf-8",
    )

    logger.info(
        "%s bundle saved: MAE=%.4f RMSE=%.4f error=%.2fpp DA=%.2f%%",
        model_type, test_metrics["mae"], test_metrics["rmse"],
        test_metrics["mape"], test_metrics["directional_accuracy"] * 100,
    )
    return meta


# ---------------------------------------------------------------------------
# Ensemble training for one symbol (all models × all horizons)
# ---------------------------------------------------------------------------

def train_ensemble_for_symbol(
    *,
    symbol: str,
    horizons: Optional[List[int]] = None,
    model_types: Optional[List[str]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    test_size: float = 0.2,
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """Train all model types × all horizons for one symbol."""
    symbol = symbol.upper()
    horizons = horizons or DEFAULT_HORIZONS
    model_types = model_types or DEFAULT_MODEL_TYPES

    raw_df = _download_data(symbol, lookback_days)
    total = len(horizons) * len(model_types)
    completed = 0
    results: List[Dict] = []
    errors: List[str] = []

    for horizon in horizons:
        for model_type in model_types:
            try:
                meta = train_regression_bundle(
                    symbol=symbol,
                    model_type=model_type,
                    horizon=horizon,
                    lookback_days=lookback_days,
                    test_size=test_size,
                    params=(params or {}).get(model_type),
                    raw_df=raw_df,
                )
                results.append(meta)
            except Exception as exc:
                msg = f"{model_type} h={horizon}: {exc}"
                logger.error("Ensemble training failed — %s", msg)
                errors.append(msg)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)

    return {
        "symbol": symbol,
        "horizons": horizons,
        "model_types": model_types,
        "completed": len(results),
        "errors": errors,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Batch training across multiple symbols
# ---------------------------------------------------------------------------

def train_ensemble_batch(
    *,
    symbols: Iterable[str],
    horizons: Optional[List[int]] = None,
    model_types: Optional[List[str]] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    test_size: float = 0.2,
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
) -> Dict[str, Any]:
    """Train the price-regression ensemble for multiple symbols."""
    resolved = [str(s).upper() for s in symbols if str(s).strip()]
    horizons = horizons or DEFAULT_HORIZONS
    model_types = model_types or DEFAULT_MODEL_TYPES

    runs: List[Dict] = []
    for i, symbol in enumerate(resolved):
        try:
            result = train_ensemble_for_symbol(
                symbol=symbol,
                horizons=horizons,
                model_types=model_types,
                lookback_days=lookback_days,
                test_size=test_size,
                params=params,
            )
            runs.append({"symbol": symbol, "status": "completed", "result": result})
        except Exception as exc:
            logger.error("Ensemble batch training failed for %s: %s", symbol, exc)
            runs.append({"symbol": symbol, "status": "failed", "error": str(exc)})

        if progress_callback:
            progress_callback(i + 1, len(resolved), symbol, "completed")

    success = sum(1 for r in runs if r["status"] == "completed")
    return {
        "symbols": resolved,
        "horizons": horizons,
        "model_types": model_types,
        "success_count": success,
        "failure_count": len(runs) - success,
        "runs": runs,
    }
