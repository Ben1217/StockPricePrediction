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
from src.models.walk_forward import walk_forward_tune

from src.utils.logger import get_logger

logger = get_logger(__name__)

REGRESSION_BUNDLES_DIR = Path("models/bundles")
REGRESSION_METADATA_DIR = Path("models/model_metadata")
DEFAULT_LOOKBACK_DAYS = 1825  # ~5 years
DEFAULT_HORIZONS = [7, 15, 30, 60]
# The 1-day step model is not a product horizon — it exists so the recursive
# forecast mode has a bundle whose target is genuinely the next day's return.
RECURSIVE_STEP_HORIZON = 1
TRAINABLE_HORIZONS = [RECURSIVE_STEP_HORIZON, *DEFAULT_HORIZONS]
DEFAULT_MODEL_TYPES = ["xgboost", "random_forest", "lstm"]
SEQUENCE_LENGTH = 60  # LSTM lookback window
MIN_TRAINING_ROWS = 756  # roughly 3 years of daily trading data

# Chronological 70/15/15: earliest 70% trains, next 15% tunes, most recent 15%
# is scored exactly once. A larger validation slice than the previous 10% makes
# the walk-forward tuning below less hostage to one short window.
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15


# A return regressor earns its place only by beating the constant predictor that
# always answers with the mean of the training targets. Anything at or below that
# has learned the drift of its training window and nothing else, so the bundle is
# stamped as failing and the predictor declines to serve it.
MIN_SKILL_SCORE = 0.0


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


def download_training_data(symbol: str, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Validated OHLCV history for one symbol, ready to train on.

    Public because the preparation service fetches once and hands the same frame
    to every bundle it trains, instead of re-downloading fifteen times. Raises
    with a readable reason — an unknown ticker, a short history, a stale feed —
    which is what a caller shows the user when preparation fails.
    """
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


def _baseline_skill(y_true: np.ndarray, y_pred: np.ndarray, train_mean: float) -> Dict[str, float]:
    """
    Score a model against the constant predictor that always returns `train_mean`.

    skill_score is 1 - MAE(model) / MAE(constant): positive means the model adds
    information, zero means it is indistinguishable from predicting the training
    drift, negative means it is actively worse.
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y_true.size == 0:
        return {"baseline_mae": 0.0, "model_mae": 0.0, "skill_score": 0.0, "prediction_std": 0.0}

    baseline_mae = float(np.mean(np.abs(y_true - train_mean)))
    model_mae = float(np.mean(np.abs(y_true - y_pred)))
    skill = 1.0 - model_mae / baseline_mae if baseline_mae > 0 else 0.0
    return {
        "baseline_mae": round(baseline_mae, 6),
        "model_mae": round(model_mae, 6),
        "skill_score": round(float(skill), 6),
        # A collapsed model emits nearly the same number for every input; the
        # spread of its predictions is the cheapest way to see that from metadata.
        "prediction_std": round(float(np.std(y_pred)), 6),
    }


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


def _tune_hyperparameters(
    *,
    model_type: str,
    factory,
    split: Dict[str, Any],
    base_params: Dict[str, Any],
    horizon: int,
):
    """
    Run walk-forward tuning over the chronological train+validation region.

    The two segments are concatenated in time order so the folds can grow across
    the boundary; the test segment is deliberately excluded so it stays a
    genuine single-use holdout.
    """
    X = np.concatenate([split["X_train"], split["X_val"]], axis=0)
    y = np.concatenate([split["y_train"], split["y_val"]], axis=0)

    sequence_builder = None
    if model_type == "lstm":
        def sequence_builder(X_tr, y_tr, X_sc, y_sc, params):
            seq_len = int(params.get("sequence_length", SEQUENCE_LENGTH))
            seq_len = min(seq_len, max(10, len(X_tr) // 2))
            hist = max(0, seq_len - 1)
            Xtr_s, ytr_s = create_sequences(X_tr, y_tr, sequence_length=seq_len)
            Xsc_s, ysc_s = create_sequences(
                np.concatenate([X_tr[-hist:], X_sc], axis=0) if hist else X_sc,
                np.concatenate([y_tr[-hist:], y_sc], axis=0) if hist else y_sc,
                sequence_length=seq_len,
            )
            if len(Xtr_s) == 0 or len(Xsc_s) == 0:
                return None
            return Xtr_s, ytr_s, Xsc_s, ysc_s

    try:
        return walk_forward_tune(
            model_type=model_type,
            factory=factory,
            X=X,
            y=y,
            base_params=base_params,
            embargo=horizon,
            sequence_builder=sequence_builder,
        )
    except Exception as exc:  # noqa: BLE001 - tuning must never block a retrain
        logger.warning("Walk-forward tuning failed for %s h=%dd: %s", model_type, horizon, exc)
        return None


# ---------------------------------------------------------------------------
# Single model × horizon training
# ---------------------------------------------------------------------------

def train_regression_bundle(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    params: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
    tune: bool = True,
) -> Dict[str, Any]:
    """
    Train one return-regression model bundle.

    Returns a dict with keys: bundle_dir, metrics (test + val), model_type,
    symbol, horizon, feature_columns, trained_at.
    """
    symbol = symbol.upper()
    horizon = int(horizon)
    if horizon not in TRAINABLE_HORIZONS:
        raise ValueError(f"Unsupported horizon {horizon}; supported horizons are {TRAINABLE_HORIZONS}")
    if model_type not in REGRESSOR_FACTORIES:
        raise ValueError(f"Unsupported model type {model_type}; supported models are {DEFAULT_MODEL_TYPES}")

    logger.info("Training %s return-regression bundle for %s horizon=%dd", model_type, symbol, horizon)

    df = raw_df.copy() if raw_df is not None else download_training_data(symbol, lookback_days)
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
        embargo=horizon,
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

    # Hyperparameter selection by walk-forward CV over train+validation. The
    # test segment is not visible here — it is scored exactly once, below.
    tuning = _tune_hyperparameters(
        model_type=model_type,
        factory=factory,
        split=split,
        base_params=model_params,
        horizon=horizon,
    ) if tune else None
    if tuning is not None:
        model_params = dict(tuning.best_params)
        if model_type == "lstm":
            model_params["sequence_length"] = seq_len
        logger.info(
            "Walk-forward tuning chose %s for %s %s h=%dd (mean val %s=%.6f over %d folds)",
            tuning.best_params, symbol, model_type, horizon, tuning.metric,
            tuning.best_score, tuning.n_splits,
        )

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

    # Skill against the constant-train-mean baseline, measured out of sample.
    train_mean = float(np.mean(y_tr)) if len(y_tr) else 0.0
    val_skill = _baseline_skill(y_va, model.predict(X_va), train_mean)
    test_skill = _baseline_skill(y_te, model.predict(X_te), train_mean)
    passes_baseline = bool(test_skill["skill_score"] > MIN_SKILL_SCORE)

    if not passes_baseline:
        logger.warning(
            "%s %s h=%dd fails the baseline gate: skill_score=%.4f (model MAE %.4f vs "
            "constant-baseline MAE %.4f, prediction std %.4f). The bundle is saved for "
            "inspection but will not be served.",
            symbol,
            model_type,
            horizon,
            test_skill["skill_score"],
            test_skill["model_mae"],
            test_skill["baseline_mae"],
            test_skill["prediction_std"],
        )

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
            "embargo": horizon,
            "embargo_reason": (
                "overlapping forward-return targets; purged so no training row resolves "
                "inside the segment it is scored against"
            ),
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
        "train_target_mean": round(train_mean, 6),
        "skill": {
            "validation": val_skill,
            "test": test_skill,
            "min_skill_score": MIN_SKILL_SCORE,
            "baseline": "constant_train_target_mean",
        },
        "passes_baseline": passes_baseline,
        "tuning": tuning.as_metadata() if tuning is not None else {"method": "none", "reason": "tuning disabled or dataset too short"},
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
        "%s bundle saved: MAE=%.4f RMSE=%.4f error=%.2fpp DA=%.2f%% skill=%.4f served=%s",
        model_type, test_metrics["mae"], test_metrics["rmse"],
        test_metrics["mape"], test_metrics["directional_accuracy"] * 100,
        test_skill["skill_score"], passes_baseline,
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
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    params: Optional[Dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    tune: bool = True,
) -> Dict[str, Any]:
    """Train all model types × all horizons for one symbol."""
    symbol = symbol.upper()
    # Includes the 1-day step model: without it the recursive per-step forecast
    # has nothing to roll forward and every horizon falls back to a compounded
    # path drawn from a single model output.
    horizons = horizons or TRAINABLE_HORIZONS
    model_types = model_types or DEFAULT_MODEL_TYPES

    raw_df = download_training_data(symbol, lookback_days)
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
                    val_size=val_size,
                    params=(params or {}).get(model_type),
                    raw_df=raw_df,
                    tune=tune,
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
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
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
                val_size=val_size,
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
