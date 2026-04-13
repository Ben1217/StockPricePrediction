"""
Shared helpers for training symbol-aware model bundles.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data.data_acquisition import get_sp500_tickers
from src.defaults import DEFAULT_INDEX_SYMBOL
from src.data.data_loader import download_stock_data
from src.features.feature_engineering import (
    build_supervised_dataset,
    create_sequences,
    normalize_feature_config,
    split_dataset_chronologically,
)
from src.models.model_bundle import select_model_metadata, save_model_bundle
from src.models.direction_utils import (
    BUY_PROBABILITY_THRESHOLD,
    NEXT_DAY_HORIZON,
    normalize_supported_horizons,
    probability_up,
    simple_long_flat_backtest,
)
from src.models.model_trainer import ModelTrainer

DEFAULT_UI_HORIZONS: List[int] = [NEXT_DAY_HORIZON]
DEFAULT_BUNDLE_HORIZONS: List[int] = [NEXT_DAY_HORIZON]
DEFAULT_BOOTSTRAP_SYMBOLS: List[str] = [
    DEFAULT_INDEX_SYMBOL,
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "TSLA",
    "META",
    "NFLX",
]
DEFAULT_BOOTSTRAP_MODELS: List[str] = ["xgboost", "random_forest", "lstm"]
DEFAULT_TRAINING_HORIZON = NEXT_DAY_HORIZON


def normalize_horizons(horizons: Optional[Iterable[int]] = None) -> List[int]:
    return normalize_supported_horizons(horizons)


def resolve_training_symbols(
    symbols: Optional[Iterable[str]] = None,
    *,
    use_sp500: bool = False,
) -> List[str]:
    if use_sp500:
        sp500_symbols = [symbol.upper() for symbol in get_sp500_tickers() if symbol]
        if not sp500_symbols:
            raise ValueError("Unable to retrieve the S&P 500 ticker universe")
        return sp500_symbols

    resolved = [str(symbol).upper() for symbol in (symbols or DEFAULT_BOOTSTRAP_SYMBOLS) if str(symbol).strip()]
    if not resolved:
        raise ValueError("At least one symbol is required for bundle training")
    return resolved


def download_training_frame(symbol: str, lookback_days: int) -> pd.DataFrame:
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    df = download_stock_data(symbol, start_date, end_date)

    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    if len(df) < 90:
        raise ValueError(
            f"Insufficient data for {symbol}: got {len(df)} rows, need at least 90 "
            f"for a reliable train/val/test split"
        )
    return df


def is_bundle_fresh(
    symbol: str,
    model_type: str,
    *,
    horizons: Optional[Iterable[int]] = None,
    max_age_hours: Optional[int] = None,
) -> bool:
    if max_age_hours is None:
        return False

    meta = select_model_metadata(model_type=model_type, symbol=symbol, horizon=1)
    if not meta:
        return False

    required_horizons = normalize_horizons(horizons) if horizons is not None else []
    available_horizons = {int(h) for h in meta.get("horizons", [])}
    if required_horizons and available_horizons and not set(required_horizons).issubset(available_horizons):
        return False

    trained_at_raw = meta.get("trained_at")
    if not trained_at_raw:
        return False

    try:
        trained_at = datetime.fromisoformat(str(trained_at_raw))
    except ValueError:
        return False

    age = datetime.now() - trained_at
    return age <= timedelta(hours=int(max_age_hours))


def _build_lstm_split_sequences(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    sequence_length: int,
):
    history_len = max(0, sequence_length - 1)

    if len(X_train) < sequence_length:
        raise ValueError(
            f"Need at least {sequence_length} training rows to build LSTM sequences"
        )

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length=sequence_length)

    val_history_X = X_train[-history_len:] if history_len else X_train[:0]
    val_history_y = y_train[-history_len:] if history_len else y_train[:0]
    val_X_full = np.concatenate([val_history_X, X_val], axis=0)
    val_y_full = np.concatenate([val_history_y, y_val], axis=0)
    X_val_seq, y_val_seq = create_sequences(val_X_full, val_y_full, sequence_length=sequence_length)

    test_history_X = np.concatenate([X_train, X_val], axis=0)
    test_history_y = np.concatenate([y_train, y_val], axis=0)
    test_history_prefix_X = test_history_X[-history_len:] if history_len else test_history_X[:0]
    test_history_prefix_y = test_history_y[-history_len:] if history_len else test_history_y[:0]
    test_X_full = np.concatenate([test_history_prefix_X, X_test], axis=0)
    test_y_full = np.concatenate([test_history_prefix_y, y_test], axis=0)
    X_test_seq, y_test_seq = create_sequences(test_X_full, test_y_full, sequence_length=sequence_length)

    if len(X_val_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("Validation or test split is too short for the configured LSTM sequence length")

    return X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq


def _evaluate_direction_split(model, X_eval, y_eval, forward_returns) -> Dict[str, Any]:
    metrics = model.evaluate(X_eval, y_eval)
    probabilities = probability_up(model.predict_proba(X_eval))
    metrics["positive_rate"] = float(np.mean(probabilities >= BUY_PROBABILITY_THRESHOLD))
    metrics["average_probability_up"] = float(np.mean(probabilities))
    return {
        "metrics": metrics,
        "backtest": simple_long_flat_backtest(probabilities, forward_returns),
    }


def train_model_bundles(
    *,
    symbol: str,
    model_type: str,
    horizons: Optional[Iterable[int]] = None,
    lookback_days: int = 756,
    test_size: float = 0.2,
    params: Optional[Dict[str, Any]] = None,
    raw_df: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None,
) -> Dict[str, Any]:
    symbol = symbol.upper()
    supported_horizons = normalize_horizons(horizons)
    df = raw_df.copy() if raw_df is not None else download_training_frame(symbol, lookback_days)

    feature_config = normalize_feature_config()
    scaler_type = "minmax"
    validation_size = min(0.15, max(0.1, test_size / 2))
    training_horizon = DEFAULT_TRAINING_HORIZON

    dataset, feature_cols = build_supervised_dataset(
        df,
        horizon=training_horizon,
        target_type="direction",
        feature_config=feature_config,
    )
    if dataset.empty or not feature_cols:
        raise ValueError("No usable training rows for the bundle")

    split = split_dataset_chronologically(
        dataset,
        feature_columns=feature_cols,
        scaler_type=scaler_type,
        test_size=test_size,
        val_size=validation_size,
    )

    model_params = dict(params or {})
    trainer = ModelTrainer()

    if model_type == "lstm":
        requested_sequence_length = int(model_params.get("sequence_length", 60))
        max_viable_sequence_length = max(10, len(split["X_train"]) // 2)
        sequence_length = min(requested_sequence_length, max_viable_sequence_length)
        model_params["sequence_length"] = sequence_length
        (
            X_train_fit,
            y_train_fit,
            X_val_fit,
            y_val_fit,
            X_test_eval,
            y_test_eval,
        ) = _build_lstm_split_sequences(
            split["X_train"],
            split["y_train"],
            split["X_val"],
            split["y_val"],
            split["X_test"],
            split["y_test"],
            sequence_length=sequence_length,
        )
        serving_mode = "next_day_direction_classifier"
    else:
        X_train_fit = split["X_train"]
        y_train_fit = split["y_train"]
        X_val_fit = split["X_val"]
        y_val_fit = split["y_val"]
        X_test_eval = split["X_test"]
        y_test_eval = split["y_test"]
        serving_mode = "next_day_direction_classifier"

    model = trainer.train_model(
        model_type,
        X_train_fit,
        y_train_fit,
        X_val=X_val_fit,
        y_val=y_val_fit,
        params=model_params,
        save=False,
    )

    validation_summary = _evaluate_direction_split(
        model,
        X_val_fit,
        y_val_fit,
        split["val_frame"]["Forward_Return"].values.astype(np.float32),
    )
    test_summary = _evaluate_direction_split(
        model,
        X_test_eval,
        y_test_eval,
        split["test_frame"]["Forward_Return"].values.astype(np.float32),
    )
    validation_metrics = validation_summary["metrics"]
    test_metrics = test_summary["metrics"]

    trained_at = datetime.now().isoformat()
    bundle_meta = save_model_bundle(
        model=model,
        model_type=model_type,
        symbol=symbol,
        horizon=training_horizon,
        feature_columns=feature_cols,
        scaler=split["scaler"],
        metadata={
            "trained_at": trained_at,
            "objective": "next_day_direction",
            "target_type": "direction",
            "feature_config": feature_config,
            "lookback_days": lookback_days,
            "training_horizon": training_horizon,
            "horizons": supported_horizons,
            "test_size": test_size,
            "validation_size": validation_size,
            "serving_mode": serving_mode,
            "monitoring": {
                "retrain_threshold_hours": 24,
                "stale_after_hours": 24,
            },
            "preprocessing": {
                "scaler_type": scaler_type,
                "sequence_length": int(model_params.get("sequence_length", 60)),
            },
            "split_sizes": {
                "train": len(split["train_frame"]),
                "validation": len(split["val_frame"]),
                "test": len(split["test_frame"]),
            },
            "samples": len(dataset),
            "training_sample_count": len(split["train_frame"]),
            "features": len(feature_cols),
            "feature_count": len(feature_cols),
            "params": model_params,
            "metrics": {
                "validation": validation_metrics,
                "test": test_metrics,
            },
            "backtest": {
                "validation": validation_summary["backtest"],
                "test": test_summary["backtest"],
                "rule": "long_when_probability_up_at_least_0.55_else_flat",
            },
        },
    )

    if progress_callback is not None:
        progress_callback(1, 1, training_horizon)

    shared_bundle_info = {
        "version_id": bundle_meta["version_id"],
        "bundle_dir": bundle_meta["bundle_dir"],
        "training_horizon": training_horizon,
        "objective": "next_day_direction",
        "serving_mode": serving_mode,
        "validation": validation_metrics,
        "test": test_metrics,
        "backtest": test_summary["backtest"],
    }

    return {
        "model_type": model_type,
        "symbol": symbol,
        "training_horizon": training_horizon,
        "horizons": supported_horizons,
        "versions": [bundle_meta["version_id"]],
        "bundle": shared_bundle_info,
        "per_horizon": {str(horizon): dict(shared_bundle_info) for horizon in supported_horizons},
    }


def train_batch_model_bundles(
    *,
    symbols: Optional[Iterable[str]] = None,
    use_sp500: bool = False,
    model_types: Optional[Iterable[str]] = None,
    horizons: Optional[Iterable[int]] = None,
    lookback_days: int = 756,
    test_size: float = 0.2,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
    skip_fresh_hours: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, str, str, str], None]] = None,
) -> Dict[str, Any]:
    resolved_symbols = resolve_training_symbols(symbols, use_sp500=use_sp500)
    resolved_models = [str(model_type) for model_type in (model_types or DEFAULT_BOOTSTRAP_MODELS)]
    supported_horizons = normalize_horizons(horizons)

    total_runs = max(1, len(resolved_symbols) * len(resolved_models))
    completed_runs = 0
    runs: List[Dict[str, Any]] = []

    for symbol in resolved_symbols:
        freshness = {
            model_type: is_bundle_fresh(
                symbol,
                model_type,
                horizons=supported_horizons,
                max_age_hours=skip_fresh_hours,
            )
            for model_type in resolved_models
        }

        if all(freshness.values()):
            for model_type in resolved_models:
                runs.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "status": "skipped",
                        "message": f"Skipping fresh {model_type} bundle for {symbol}",
                    }
                )
                completed_runs += 1
                if progress_callback is not None:
                    progress_callback(completed_runs, total_runs, symbol, model_type, "skipped")
            continue

        raw_df: Optional[pd.DataFrame] = None
        download_error: Optional[str] = None
        try:
            raw_df = download_training_frame(symbol, lookback_days)
        except Exception as exc:
            download_error = str(exc)

        for model_type in resolved_models:
            status = "completed"
            if download_error is not None:
                status = "failed"
                runs.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "status": status,
                        "message": download_error,
                    }
                )
            elif freshness.get(model_type):
                status = "skipped"
                runs.append(
                    {
                        "symbol": symbol,
                        "model_type": model_type,
                        "status": status,
                        "message": f"Skipping fresh {model_type} bundle for {symbol}",
                    }
                )
            else:
                try:
                    result = train_model_bundles(
                        symbol=symbol,
                        model_type=model_type,
                        horizons=supported_horizons,
                        lookback_days=lookback_days,
                        test_size=test_size,
                        params=(params or {}).get(model_type) if params else None,
                        raw_df=raw_df,
                    )
                    runs.append(
                        {
                            "symbol": symbol,
                            "model_type": model_type,
                            "status": status,
                            "result": result,
                        }
                    )
                except Exception as exc:
                    status = "failed"
                    runs.append(
                        {
                            "symbol": symbol,
                            "model_type": model_type,
                            "status": status,
                            "message": str(exc),
                        }
                    )

            completed_runs += 1
            if progress_callback is not None:
                progress_callback(completed_runs, total_runs, symbol, model_type, status)

    success_count = sum(1 for run in runs if run["status"] == "completed")
    skipped_count = sum(1 for run in runs if run["status"] == "skipped")
    failure_count = sum(1 for run in runs if run["status"] == "failed")

    return {
        "symbols": resolved_symbols,
        "model_types": resolved_models,
        "horizons": supported_horizons,
        "success_count": success_count,
        "skipped_count": skipped_count,
        "failure_count": failure_count,
        "runs": runs,
    }
