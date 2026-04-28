"""
Ensemble Price Predictor.

Loads trained return-regression bundles (XGBoost, RandomForest, LSTM),
runs inference, computes a weighted ensemble prediction, and derives a
reliability score from model agreement, prediction spread, and validation error.

Public API:
    EnsemblePricePredictor.predict(symbol, horizon, raw_df) -> EnsembleForecast
    load_ensemble_predictor(symbol, horizon) -> EnsemblePricePredictor | None
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.features.feature_engineering import (
    build_feature_frame,
    normalize_feature_config,
    transform_feature_frame,
    create_sequences,
)
from src.models.regression_models import REGRESSOR_FACTORIES, REGRESSOR_FILE_NAMES
from src.utils.logger import get_logger

logger = get_logger(__name__)

REGRESSION_BUNDLES_DIR = Path("models/bundles")
SEQUENCE_LENGTH = 60
MODEL_TYPES = ["xgboost", "random_forest", "lstm"]
SUPPORTED_HORIZONS = [7, 15, 30, 60]
NEUTRAL_MIN_CHANGE_PCT = 0.5
FIXED_ENSEMBLE_WEIGHTS = {"lstm": 0.40, "xgboost": 0.35, "random_forest": 0.25}
HARD_GAP_LIMITS = {7: 0.08, 15: 0.12, 30: 0.20, 60: 0.30}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelPredictionResult:
    model_type: str
    prediction: float
    weight: float
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float
    predicted_return: float = 0.0
    uncertainty_return: float = 0.0


@dataclass
class EnsembleForecast:
    symbol: str
    horizon: int
    current_price: float
    predicted_price: float
    expected_change_pct: float
    signal: str                    # "Bullish" | "Bearish" | "Neutral"
    reliability: str               # "High" | "Medium" | "Low"
    reason: str
    model_predictions: List[ModelPredictionResult]
    forecast_points: List[Dict]    # [{date, predicted, lower, upper}]
    trained_at: Optional[str] = None
    feature_count: Optional[int] = None
    data_source: str = "yfinance"
    training_rows: Optional[int] = None
    validation_error_pct: Optional[float] = None
    prediction_spread_pct: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Bundle loading
# ---------------------------------------------------------------------------

def _bundle_dir(symbol: str, model_type: str, horizon: int) -> Path:
    return REGRESSION_BUNDLES_DIR / symbol.upper() / model_type / str(int(horizon))


def _legacy_price_regression_bundle_dir(symbol: str, model_type: str, horizon: int) -> Path:
    return REGRESSION_BUNDLES_DIR / symbol.upper() / "price_regression" / str(int(horizon)) / model_type


def _metadata_is_return_regression(meta: Dict) -> bool:
    return (
        meta.get("target_type") == "return_regression"
        or meta.get("model_output") == "predicted_return"
        or str(meta.get("objective", "")).startswith("future_return")
    )


def _load_regression_bundle(symbol: str, model_type: str, horizon: int) -> Optional[Dict]:
    """Load a single regression bundle. Returns dict with model, scaler, meta, feature_cols."""
    bdir = _bundle_dir(symbol, model_type, horizon)
    meta_path = bdir / "metadata.json"
    if not meta_path.exists():
        bdir = _legacy_price_regression_bundle_dir(symbol, model_type, horizon)
        meta_path = bdir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if not _metadata_is_return_regression(meta):
            logger.info(
                "Skipping legacy non-return bundle for %s %s h=%d at %s",
                symbol,
                model_type,
                horizon,
                meta_path,
            )
            return None
        factory = REGRESSOR_FACTORIES[model_type]
        model = factory()
        model_path = Path(meta["model_path"])
        if not model_path.exists():
            model_path = bdir / REGRESSOR_FILE_NAMES[model_type]
        model.load(str(model_path))
        scaler_path = Path(meta["scaler_path"])
        if not scaler_path.exists():
            scaler_path = bdir / "scaler.joblib"
        scaler = joblib.load(scaler_path)
        feature_cols = list(meta["feature_columns"])
        return {"model": model, "scaler": scaler, "meta": meta, "feature_cols": feature_cols}
    except Exception as exc:
        logger.warning("Failed to load %s bundle for %s h=%d: %s", model_type, symbol, horizon, exc)
        return None


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _run_inference(bundle: Dict, feature_frame: pd.DataFrame, model_type: str) -> Optional[Tuple[float, float]]:
    """Run regression inference and return (predicted_return, uncertainty_return)."""
    try:
        feature_cols = bundle["feature_cols"]
        scaler = bundle["scaler"]
        model = bundle["model"]
        meta = bundle["meta"]

        aligned, X = transform_feature_frame(feature_frame, feature_cols, scaler=scaler)
        if aligned.empty or len(X) == 0:
            return None

        if model_type == "lstm":
            seq_len = int(meta.get("sequence_length") or SEQUENCE_LENGTH)
            if len(X) < seq_len:
                return None
            seq = X[-seq_len:][np.newaxis, :, :]
            if hasattr(model, "predict_with_uncertainty"):
                mean_pred, std_pred = model.predict_with_uncertainty(seq, n_samples=30)
                pred = float(np.asarray(mean_pred).reshape(-1)[0])
                uncertainty = float(np.asarray(std_pred).reshape(-1)[0])
            else:
                pred = float(np.asarray(model.predict(seq)).reshape(-1)[0])
                uncertainty = _safe_metric(_validation_metrics(bundle), "rmse", 0.02)
        else:
            pred = float(np.asarray(model.predict(X[-1:])).reshape(-1)[0])
            uncertainty = _safe_metric(_validation_metrics(bundle), "rmse", 0.02)

        return pred, max(float(uncertainty), 1e-6)
    except Exception as exc:
        logger.warning("Inference failed for %s: %s", model_type, exc)
        return None


def _validation_metrics(bundle: Dict) -> Dict:
    """Return the metric set used for weighting and UI display."""
    meta = bundle.get("meta", {})
    return (
        meta.get("val_metrics")
        or meta.get("validation_metrics")
        or meta.get("metrics", {}).get("validation")
        or meta.get("test_metrics")
        or {}
    )


def _test_metrics(bundle: Dict) -> Dict:
    meta = bundle.get("meta", {})
    return meta.get("test_metrics") or meta.get("metrics", {}).get("test") or _validation_metrics(bundle)


def _safe_metric(metrics: Dict, key: str, default: float) -> float:
    try:
        value = float(metrics.get(key, default))
    except (TypeError, ValueError):
        value = default
    if not np.isfinite(value):
        return default
    return value


def _compute_weights(bundles: Dict[str, Dict], current_price: float) -> Dict[str, float]:
    """Return the fixed ensemble weights from the prediction spec."""
    weights = {mtype: FIXED_ENSEMBLE_WEIGHTS[mtype] for mtype in MODEL_TYPES if mtype in bundles}
    total = sum(weights.values())
    if total <= 0:
        return {}

    return {m: weight / total for m, weight in weights.items()}


def _neutral_threshold_pct(recent_volatility: float, horizon: int) -> float:
    horizon_vol_pct = max(float(recent_volatility), 0.0) * np.sqrt(max(int(horizon), 1)) * 100.0
    return float(min(max(NEUTRAL_MIN_CHANGE_PCT, horizon_vol_pct * 0.15), 2.0))


def _direction_vote(prediction: float, current_price: float, threshold_pct: float) -> int:
    change_pct = (float(prediction) - float(current_price)) / max(float(current_price), 1e-6) * 100.0
    if change_pct > threshold_pct:
        return 1
    if change_pct < -threshold_pct:
        return -1
    return 0


def _signal_from_change(change_pct: float, threshold_pct: float) -> str:
    if change_pct > threshold_pct:
        return "Bullish"
    if change_pct < -threshold_pct:
        return "Bearish"
    return "Neutral"


def _reliability_score(
    predictions: Dict[str, float],
    current_price: float,
    bundles: Dict[str, Dict],
    recent_volatility: float,
    horizon: int,
    ensemble_change_pct: float,
    confidence_width_pct: float,
) -> Tuple[str, str, str, float, float]:
    """
    Compute reliability from agreement, spread, validation error, volatility,
    and final confidence interval width.
    """
    if not predictions:
        return "Neutral", "Low", "No models available.", 0.0, 0.0

    threshold_pct = _neutral_threshold_pct(recent_volatility, horizon)
    votes = [
        _direction_vote(prediction, current_price, threshold_pct)
        for prediction in predictions.values()
    ]
    n_models = len(votes)
    bull_votes = sum(1 for vote in votes if vote > 0)
    bear_votes = sum(1 for vote in votes if vote < 0)
    neutral_votes = sum(1 for vote in votes if vote == 0)
    max_agree = max(bull_votes, bear_votes, neutral_votes)

    signal = _signal_from_change(ensemble_change_pct, threshold_pct)

    preds = list(predictions.values())
    spread_pct = (max(preds) - min(preds)) / max(current_price, 1e-6) * 100.0

    mapes = [
        _safe_metric(_validation_metrics(bundle), "mape", 8.0)
        for bundle in bundles.values()
    ]
    avg_mape = float(np.mean(mapes)) if mapes else 10.0
    vol_high = recent_volatility > 0.35

    # Sanity checks for unrealistic gaps
    abs_change = abs(ensemble_change_pct)
    unrealistic_gap = False
    if horizon == 7 and abs_change > 8.0:
        unrealistic_gap = True
    elif horizon == 15 and abs_change > 12.0:
        unrealistic_gap = True
    elif horizon == 30 and abs_change > 20.0:
        unrealistic_gap = True
    elif horizon == 60 and abs_change > 30.0:
        unrealistic_gap = True

    if unrealistic_gap:
        reliability = "Low"
        reason = f"Unrealistic prediction gap ({ensemble_change_pct:.1f}% for {horizon}D). Model predictions are flagged as low reliability."
        return signal, reliability, reason, spread_pct, avg_mape

    if n_models < len(MODEL_TYPES):
        reliability = "Low" if n_models < 2 else "Medium"
        reason = (
            f"Only {n_models} of {len(MODEL_TYPES)} ensemble models are available; "
            "retrain the full ensemble before relying on this forecast."
        )
        return signal, reliability, reason, spread_pct, avg_mape

    high_quality = (
        max_agree == n_models
        and spread_pct < 2.5
        and avg_mape < 4.0
        and confidence_width_pct < 8.0
        and not vol_high
    )
    medium_quality = (
        max_agree >= 2
        and spread_pct < 6.0
        and avg_mape < 8.0
        and confidence_width_pct < 15.0
    )

    direction_str = signal.lower()
    if max_agree == n_models:
        consensus_base = f"All {n_models} models {direction_str} — strong consensus"
    elif max_agree >= 2:
        consensus_base = f"{max_agree} of {n_models} models {direction_str} — moderate consensus"
    else:
        consensus_base = f"Models disagree on direction — weak consensus"

    if high_quality:
        reliability = "High"
        reason = consensus_base
    elif medium_quality:
        reliability = "Medium"
        reason = consensus_base
    else:
        reliability = "Low"
        reason = consensus_base + " (High variance / volatility warning)"

    return signal, reliability, reason, spread_pct, avg_mape


def _spec_reliability_score(
    predictions: Dict[str, float],
    current_price: float,
    bundles: Dict[str, Dict],
    recent_volatility: float,
    horizon: int,
    ensemble_change_pct: float,
    confidence_width_pct: float,
) -> Tuple[str, str, str, float, float]:
    """Reliability logic from the attached prediction spec."""
    if not predictions:
        return "Neutral", "Low", "No models available.", 0.0, 0.0

    threshold_pct = _neutral_threshold_pct(recent_volatility, horizon)
    votes = [_direction_vote(value, current_price, threshold_pct) for value in predictions.values()]
    n_models = len(votes)
    bull_votes = sum(1 for vote in votes if vote > 0)
    bear_votes = sum(1 for vote in votes if vote < 0)
    neutral_votes = sum(1 for vote in votes if vote == 0)
    max_agree = max(bull_votes, bear_votes, neutral_votes)

    signal = _signal_from_change(ensemble_change_pct, threshold_pct)
    ensemble_price = current_price * (1.0 + ensemble_change_pct / 100.0)

    values = list(predictions.values())
    spread_pct = (max(values) - min(values)) / max(current_price, 1e-6) * 100.0
    validation_errors_pct = [
        _safe_metric(_validation_metrics(bundle), "mae", 0.03) * 100.0
        for bundle in bundles.values()
    ]
    avg_error_pct = float(np.mean(validation_errors_pct)) if validation_errors_pct else 3.0

    hard_limit = HARD_GAP_LIMITS.get(int(horizon), 0.30)
    if abs(ensemble_change_pct) / 100.0 > hard_limit:
        return signal, "Low", "Prediction gap is too large for the selected horizon.", spread_pct, avg_error_pct

    vol_range = max(float(recent_volatility), 0.0) * np.sqrt(max(int(horizon), 1))
    lower_bound = current_price * (1.0 - vol_range)
    upper_bound = current_price * (1.0 + vol_range)
    if recent_volatility > 0 and (ensemble_price < lower_bound or ensemble_price > upper_bound):
        return signal, "Low", "Prediction outside volatility-based realistic range.", spread_pct, avg_error_pct

    if n_models < len(MODEL_TYPES):
        reliability = "Low" if n_models < 2 else "Medium"
        reason = (
            f"Only {n_models} of {len(MODEL_TYPES)} ensemble models are available; "
            "retrain the full ensemble before relying on this forecast."
        )
        return signal, reliability, reason, spread_pct, avg_error_pct

    direction_str = signal.lower()
    if max_agree == n_models:
        consensus_base = f"All {n_models} models {direction_str} - strong consensus"
    elif max_agree >= 2:
        consensus_base = f"{max_agree} of {n_models} models {direction_str} - moderate consensus"
    else:
        consensus_base = "Models disagree on direction - weak consensus"

    annualized_vol = recent_volatility * np.sqrt(252.0)
    high_quality = (
        max_agree == n_models
        and spread_pct < 2.5
        and avg_error_pct < 2.0
        and confidence_width_pct < 8.0
        and annualized_vol <= 0.35
    )
    medium_quality = (
        max_agree >= 2
        and spread_pct < 6.0
        and avg_error_pct < 4.0
        and confidence_width_pct < 15.0
    )

    if high_quality:
        return signal, "High", consensus_base, spread_pct, avg_error_pct
    if medium_quality:
        return signal, "Medium", consensus_base, spread_pct, avg_error_pct
    return signal, "Low", consensus_base + " (High variance / volatility warning)", spread_pct, avg_error_pct


def _build_forecast_points(
    predicted_price: float,
    current_price: float,
    horizon: int,
    last_date: pd.Timestamp,
    avg_mape: float,
    weighted_rmse: float,
    spread_pct: float,
    recent_volatility: float,
    raw_predictions: Dict[str, float],
) -> List[Dict]:
    """Generate interpolated daily forecast points with uncertainty bands."""
    future_dates = list(pd.bdate_range(start=last_date, periods=horizon + 1)[1:])
    points = []
    daily_vol = max(float(recent_volatility), 0.0)
    for i, dt in enumerate(future_dates):
        frac = (i + 1) / horizon
        p = current_price + (predicted_price - current_price) * frac
        validation_band = p * (max(avg_mape, 0.1) / 100.0) * np.sqrt(frac)
        rmse_band = max(weighted_rmse, 0.0) * np.sqrt(frac)
        spread_band = current_price * (max(spread_pct, 0.0) / 100.0) * 0.50 * frac
        volatility_band = p * daily_vol * np.sqrt(i + 1) * 0.50
        band = validation_band + 0.50 * rmse_band + spread_band + volatility_band
        point = {
            "date": str(dt.date()),
            "predicted": round(float(p), 2),
            "lower": round(float(max(p - band, 0.0)), 2),
            "upper": round(float(p + band), 2),
        }
        for m, m_pred in raw_predictions.items():
            m_p = current_price + (m_pred - current_price) * frac
            point[m] = round(float(m_p), 2)
        points.append(point)
    return points


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class EnsemblePricePredictor:
    """
    Loads trained return-regression bundles for a symbol/horizon and produces
    a weighted ensemble price forecast with reliability scoring.
    """

    def predict(
        self,
        symbol: str,
        horizon: int,
        raw_df: pd.DataFrame,
    ) -> Optional[EnsembleForecast]:
        symbol = symbol.upper()
        horizon = int(horizon)
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}; supported horizons are {SUPPORTED_HORIZONS}")

        # 1. Load all available bundles
        bundles: Dict[str, Dict] = {}
        for mtype in MODEL_TYPES:
            b = _load_regression_bundle(symbol, mtype, horizon)
            if b is not None:
                bundles[mtype] = b

        if not bundles:
            logger.info("No regression bundles found for %s h=%d", symbol, horizon)
            return None

        current_price = float(raw_df["Close"].iloc[-1])
        last_date = raw_df.index[-1]

        # 2. Build feature frame (shared across all models)
        first_meta = next(iter(bundles.values()))["meta"]
        feature_config = normalize_feature_config(first_meta.get("feature_config"))
        feature_frame = build_feature_frame(raw_df, feature_config=feature_config)

        # 3. Run inference per model. Models output returns, which are
        # converted to prices only after inference.
        model_returns: Dict[str, float] = {}
        model_uncertainty: Dict[str, float] = {}
        raw_predictions: Dict[str, float] = {}
        for mtype, bundle in bundles.items():
            inference = _run_inference(bundle, feature_frame, mtype)
            if inference is not None:
                predicted_return, uncertainty_return = inference
                model_returns[mtype] = predicted_return
                model_uncertainty[mtype] = uncertainty_return
                raw_predictions[mtype] = current_price * (1.0 + predicted_return)

        if not raw_predictions:
            return None

        # 4. Fixed ensemble weights from the spec
        active_bundles = {m: bundles[m] for m in raw_predictions}
        weights = _compute_weights(active_bundles, current_price)

        # 5. Weighted ensemble return, then convert to price
        ensemble_return = sum(model_returns[m] * weights.get(m, 0.0) for m in model_returns)
        weight_total = sum(weights.get(m, 0.0) for m in raw_predictions)
        if weight_total > 0:
            ensemble_return /= weight_total
        ensemble_return = float(ensemble_return)
        ensemble_pred = float(current_price * (1.0 + ensemble_return))

        change_pct = (ensemble_pred - current_price) / max(current_price, 1e-6) * 100

        # 6. Recent daily volatility for adaptive bounds
        vol_source = raw_df["Adj Close"] if "Adj Close" in raw_df.columns else raw_df["Close"]
        daily_ret = vol_source.pct_change().dropna()
        recent_vol = float(daily_ret.tail(20).std()) if len(daily_ret) >= 20 else 0.02

        # 7. Per-model result objects. These metrics are validation metrics,
        # while prediction is the converted price for display.
        model_results: List[ModelPredictionResult] = []
        for mtype in MODEL_TYPES:
            if mtype not in raw_predictions:
                continue
            display_m = _validation_metrics(bundles[mtype])
            model_results.append(ModelPredictionResult(
                model_type=mtype,
                prediction=round(raw_predictions[mtype], 2),
                weight=round(weights.get(mtype, 0.0), 4),
                mae=round(_safe_metric(display_m, "mae", 0.0), 4),
                rmse=round(_safe_metric(display_m, "rmse", 0.0), 4),
                mape=round(_safe_metric(display_m, "mape", 0.0), 4),
                directional_accuracy=round(_safe_metric(display_m, "directional_accuracy", 0.5), 4),
                predicted_return=round(model_returns.get(mtype, 0.0), 6),
                uncertainty_return=round(model_uncertainty.get(mtype, 0.0), 6),
            ))

        # 8. Validation uncertainty ingredients for bands and reliability.
        prediction_values = list(raw_predictions.values())
        spread_pct = (
            (max(prediction_values) - min(prediction_values)) / max(current_price, 1e-6) * 100.0
            if len(prediction_values) > 1
            else 0.0
        )
        avg_mape = sum(
            _safe_metric(_validation_metrics(active_bundles[m]), "mae", 0.03) * 100.0 * weights.get(m, 0.0)
            for m in active_bundles
        )
        weighted_rmse_return = sum(
            max(
                _safe_metric(_validation_metrics(active_bundles[m]), "rmse", 0.03),
                model_uncertainty.get(m, 0.0),
            ) * weights.get(m, 0.0)
            for m in active_bundles
        )
        weighted_rmse = current_price * weighted_rmse_return

        # 9. Forecast timeline
        forecast_points = _build_forecast_points(
            ensemble_pred,
            current_price,
            horizon,
            last_date,
            avg_mape,
            weighted_rmse,
            spread_pct,
            recent_vol,
            raw_predictions,
        )
        final_interval = forecast_points[-1] if forecast_points else None
        confidence_width_pct = (
            (float(final_interval["upper"]) - float(final_interval["lower"])) / max(current_price, 1e-6) * 100.0
            if final_interval
            else 0.0
        )

        # 10. Reliability score
        signal, reliability, reason, spread_pct, avg_mape = _spec_reliability_score(
            raw_predictions,
            current_price,
            active_bundles,
            recent_vol,
            horizon,
            change_pct,
            confidence_width_pct,
        )

        # 11. Training metadata
        trained_ats = [bundles[m]["meta"].get("trained_at") for m in bundles if bundles[m]["meta"].get("trained_at")]
        trained_at = max(trained_ats) if trained_ats else None
        feature_count = list(bundles.values())[0]["meta"].get("feature_count") if bundles else None
        training_rows = [
            int(bundle["meta"].get("training_rows") or bundle["meta"].get("training_sample_count") or 0)
            for bundle in active_bundles.values()
        ]
        data_source = str(first_meta.get("data_source", "yfinance"))
        confidence_interval = None
        if final_interval:
            confidence_interval = {
                "lower": float(final_interval["lower"]),
                "upper": float(final_interval["upper"]),
                "width_pct": round(float(confidence_width_pct), 2),
            }

        return EnsembleForecast(
            symbol=symbol,
            horizon=horizon,
            current_price=round(current_price, 2),
            predicted_price=round(ensemble_pred, 2),
            expected_change_pct=round(change_pct, 2),
            signal=signal,
            reliability=reliability,
            reason=reason,
            model_predictions=model_results,
            forecast_points=forecast_points,
            trained_at=trained_at,
            feature_count=feature_count,
            data_source=data_source,
            training_rows=min([row for row in training_rows if row > 0], default=None),
            validation_error_pct=round(float(avg_mape), 2),
            prediction_spread_pct=round(float(spread_pct), 2),
            confidence_interval=confidence_interval,
        )


def ensemble_bundles_available(symbol: str, horizon: int) -> bool:
    """Return True when the complete three-model regression ensemble exists."""
    for mtype in MODEL_TYPES:
        canonical = _bundle_dir(symbol, mtype, horizon) / "metadata.json"
        legacy = _legacy_price_regression_bundle_dir(symbol, mtype, horizon) / "metadata.json"
        meta_path = canonical if canonical.exists() else legacy if legacy.exists() else None
        if meta_path is None:
            return False
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            return False
        if not _metadata_is_return_regression(meta):
            return False
    return True
