"""
Prediction API routes using saved model bundles.

Forecasts are generated via Monte Carlo simulation over a recursive
one-step model-driven path.  Each scenario samples noise from the
model's calibrated RMSE, producing realistic, curvature-rich forecast
fans rather than a single straight-line extrapolation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.schemas import ForecastPoint, HistoricalSignal, PredictRequest, PredictResponse
from src.features.feature_engineering import build_feature_frame, transform_feature_frame
from src.models.direction_utils import (
    BUY_PROBABILITY_THRESHOLD,
    NEXT_DAY_HORIZON,
    SELL_PROBABILITY_THRESHOLD,
    confidence_from_probability,
    direction_from_probability,
    expected_move_from_probability,
    probability_up,
    signal_from_probability,
)
from src.models.model_bundle import load_model_bundle
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

LOOKBACK_DAYS = 900
N_SCENARIOS = 50       # Monte Carlo paths
N_DISPLAY_PATHS = 12   # scenario lines sent to the frontend


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_prediction_data(symbol: str) -> pd.DataFrame:
    import yfinance as yf

    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)
    df = yf.download(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    return df


def _next_business_date(last_index: pd.Timestamp) -> str:
    date = pd.Timestamp(last_index).tz_localize(None) if pd.Timestamp(last_index).tzinfo else pd.Timestamp(last_index)
    next_date = pd.bdate_range(start=date, periods=2)[1]
    return str(next_date.date())


def _validate_bundle_objective(bundle) -> Optional[str]:
    target_type = getattr(bundle, "target_type", bundle.metadata.get("target_type", "direction"))
    if target_type != "direction":
        return (
            f"{bundle.model_type} bundle for {bundle.symbol} uses legacy "
            f"'{target_type}' targets. Retrain it for next-day direction."
        )
    return None


def _predict_bundle_probabilities(bundle, feature_frame: pd.DataFrame) -> np.ndarray:
    aligned, X = transform_feature_frame(feature_frame, bundle.feature_columns, scaler=bundle.scaler)
    if aligned.empty or len(X) == 0:
        raise ValueError("Not enough aligned feature rows for inference")

    if bundle.model_type == "lstm":
        sequence_length = bundle.sequence_length
        if len(X) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} aligned feature rows for LSTM inference"
            )
        sequence = X[-sequence_length:][np.newaxis, :, :]
        return np.asarray(bundle.model.predict_proba(sequence))

    return np.asarray(bundle.model.predict_proba(X[-1:]))


def _predict_history_probabilities(bundle, feature_frame: pd.DataFrame) -> Tuple[pd.Index, np.ndarray]:
    aligned, X = transform_feature_frame(feature_frame, bundle.feature_columns, scaler=bundle.scaler)
    if aligned.empty or len(X) == 0:
        return aligned.index, np.empty(0, dtype=np.float32)

    if bundle.model_type == "lstm":
        sequence_length = bundle.sequence_length
        if len(X) < sequence_length:
            return aligned.index[:0], np.empty(0, dtype=np.float32)
        sequences = np.array([X[i - sequence_length + 1:i + 1] for i in range(sequence_length - 1, len(X))])
        probabilities = probability_up(bundle.model.predict_proba(sequences))
        return aligned.index[sequence_length - 1:], probabilities

    probabilities = probability_up(bundle.model.predict_proba(X))
    return aligned.index, probabilities


def _future_business_dates(last_index: pd.Timestamp, horizon: int) -> List[pd.Timestamp]:
    start = pd.Timestamp(last_index).tz_localize(None) if pd.Timestamp(last_index).tzinfo else pd.Timestamp(last_index)
    return list(pd.bdate_range(start=start, periods=horizon + 1)[1:])


# ---------------------------------------------------------------------------
# Uncertainty / RMSE helpers
# ---------------------------------------------------------------------------

def _bundle_rmse(bundle) -> float:
    """Extract the per-step return RMSE from the model bundle metadata."""
    metrics = bundle.metadata.get("metrics", {})
    nested = metrics.get("test", metrics)
    rmse = nested.get("rmse")
    if rmse is None:
        return 0.02
    try:
        return max(1e-6, float(rmse))
    except Exception:
        return 0.02


def _forecast_engine_name(bundle) -> str:
    return "recursive_lstm" if bundle.model_type == "lstm" else "recursive_tabular"


def _uncertainty_method_name(bundle) -> str:
    if bundle.model_type == "lstm":
        return "mc_dropout_plus_bundle_test_rmse"
    return "bundle_test_rmse_monte_carlo"


def _artifact_source_name(bundle) -> str:
    layout = getattr(bundle, "bundle_layout", None) or bundle.metadata.get("bundle_layout", "")
    if layout == "legacy_horizon":
        return "legacy_horizon_bundle"
    return "canonical_symbol_model_bundle"


def _build_model_info(
    *,
    symbol: str,
    requested_model: str,
    bundle=None,
    available: bool,
    reason: Optional[str] = None,
    message: Optional[str] = None,
) -> Dict[str, object]:
    metadata = bundle.metadata if bundle is not None else {}
    model_info: Dict[str, object] = {
        "requested_model": requested_model,
        "serving_model": bundle.model_type if bundle is not None else None,
        "requested_horizon": NEXT_DAY_HORIZON,
        "supported_horizons": bundle.supported_horizons if bundle is not None else [NEXT_DAY_HORIZON],
        "objective": metadata.get("objective", "next_day_direction") if bundle is not None else "next_day_direction",
        "target_type": metadata.get("target_type", "direction") if bundle is not None else "direction",
        "serving_mode": metadata.get("serving_mode", "next_day_direction_classifier") if bundle is not None else "next_day_direction_classifier",
        "artifact_source": _artifact_source_name(bundle) if bundle is not None else None,
        "artifact_path": str(bundle.artifact_dir) if bundle is not None else None,
        "bundle_version": bundle.version_id if bundle is not None else None,
        "trained_at": metadata.get("trained_at") if bundle is not None else None,
        "training_symbol": bundle.symbol if bundle is not None else symbol,
        "training_horizon": bundle.horizon if bundle is not None else NEXT_DAY_HORIZON,
        "model_available": bool(available),
        "status": "available" if available else "unavailable",
        "reason": reason,
        "message": message,
        "can_train": True,
        "type": requested_model,
        "source": "trained_bundle" if available else "missing_bundle",
        "feature_count": metadata.get("feature_count") if bundle is not None else None,
        "signal_thresholds": {
            "buy_probability_up": BUY_PROBABILITY_THRESHOLD,
            "sell_probability_up": SELL_PROBABILITY_THRESHOLD,
        },
    }
    return model_info


# ---------------------------------------------------------------------------
# Single-step model inference
# ---------------------------------------------------------------------------

def _predict_return_from_bundle(
    bundle,
    feature_frame: pd.DataFrame,
) -> Tuple[float, float]:
    """
    Predict the next-period return from a model bundle.

    Returns (predicted_return, per_step_rmse).
    """
    aligned, X = transform_feature_frame(feature_frame, bundle.feature_columns, scaler=bundle.scaler)
    if aligned.empty or len(X) == 0:
        raise ValueError("Not enough aligned feature rows for inference")

    uncertainty_return = _bundle_rmse(bundle)

    if bundle.model_type == "lstm":
        sequence_length = bundle.sequence_length
        if len(X) < sequence_length:
            raise ValueError(
                f"Need at least {sequence_length} aligned feature rows for LSTM inference"
            )
        # Build the latest rolling sequence
        sequence = X[-sequence_length:][np.newaxis, :, :]
        if hasattr(bundle.model, "predict_with_uncertainty"):
            mean_pred, lower_pred, upper_pred = bundle.model.predict_with_uncertainty(sequence, n_samples=50)
            pred_return = float(np.asarray(mean_pred).reshape(-1)[0])
            lower_val = float(np.asarray(lower_pred).reshape(-1)[0])
            upper_val = float(np.asarray(upper_pred).reshape(-1)[0])
            uncertainty_return = max(abs(pred_return - lower_val), abs(upper_val - pred_return), 1e-6)
        else:
            pred_return = float(np.asarray(bundle.model.predict(sequence)).reshape(-1)[0])
    else:
        pred_return = float(np.asarray(bundle.model.predict(X[-1:])).reshape(-1)[0])

    return pred_return, max(uncertainty_return, 1e-6)


# ---------------------------------------------------------------------------
# Synthetic future row for recursive feature updates
# ---------------------------------------------------------------------------

def _synthetic_future_row(history: pd.DataFrame, next_date: pd.Timestamp, predicted_close: float) -> pd.DataFrame:
    """Create a plausible OHLCV bar for a future date to feed back into the feature pipeline."""
    last_close = float(history["Close"].iloc[-1])
    recent_range = (
        (history["High"] - history["Low"]) / history["Close"]
    ).replace([np.inf, -np.inf], np.nan).dropna().tail(20)
    intraday_range = float(recent_range.median()) if not recent_range.empty else 0.015
    intraday_range = min(max(intraday_range, 0.002), 0.08)

    open_price = last_close
    high_price = max(open_price, predicted_close) * (1 + intraday_range / 2)
    low_price = min(open_price, predicted_close) * (1 - intraday_range / 2)

    recent_volume = history["Volume"].replace(0, np.nan).dropna().tail(20)
    fallback_volume = float(history["Volume"].iloc[-1]) if len(history) else 0.0
    volume = int(round(float(recent_volume.median()))) if not recent_volume.empty else int(round(fallback_volume))

    return pd.DataFrame(
        {
            "Open": [open_price],
            "High": [high_price],
            "Low": [low_price],
            "Close": [predicted_close],
            "Volume": [max(volume, 0)],
        },
        index=pd.DatetimeIndex([pd.Timestamp(next_date)]),
    )


# ---------------------------------------------------------------------------
# ForecastPoint builder
# ---------------------------------------------------------------------------

def _make_forecast_point(date: pd.Timestamp, predicted_price: float, price_uncertainty: float) -> ForecastPoint:
    return ForecastPoint(
        date=str(pd.Timestamp(date).date()),
        predicted=round(float(predicted_price), 2),
        upper95=round(float(predicted_price + 1.96 * price_uncertainty), 2),
        lower95=round(float(predicted_price - 1.96 * price_uncertainty), 2),
        upper68=round(float(predicted_price + price_uncertainty), 2),
        lower68=round(float(predicted_price - price_uncertainty), 2),
    )


# ---------------------------------------------------------------------------
# Monte Carlo recursive forecast  (NEW — replaces old single-path approach)
# ---------------------------------------------------------------------------

def _monte_carlo_recursive_forecast(
    bundle,
    raw_df: pd.DataFrame,
    horizon: int,
    current_price: float,
    n_scenarios: int = N_SCENARIOS,
    seed: int = 42,
) -> Tuple[List[ForecastPoint], List[List[float]]]:
    """
    Generate multi-step forecast using Monte Carlo simulation.

    For each future step:
      1. Build features from the rolling history (includes predicted rows)
      2. Get the model's predicted return + calibrated RMSE
      3. For each scenario, sample noise ~ N(0, rmse) and accumulate price path
      4. Use the *median* predicted price to build the synthetic future row
         that feeds into the next step's feature engineering

    Returns:
        forecasts: list of ForecastPoint (percentile-based CI bands)
        scenario_paths: list of full price paths for fan chart visualisation
    """
    rng = np.random.RandomState(seed)
    forecast_dates = _future_business_dates(raw_df.index[-1], horizon)

    # Each scenario tracks its own price path
    price_paths = np.full((n_scenarios, horizon), np.nan)  # (scenarios, steps)
    scenario_prices = np.full(n_scenarios, current_price)  # running price per scenario

    rolling_history = raw_df.copy()
    forecasts: List[ForecastPoint] = []

    for step_idx, forecast_date in enumerate(forecast_dates):
        # 1 — Run the model ONCE on the current rolling history
        feature_frame = build_feature_frame(rolling_history, feature_config=bundle.feature_config)
        pred_return, step_rmse = _predict_return_from_bundle(bundle, feature_frame)

        # 2 — Sample per-scenario noise and advance each path
        noise = rng.normal(0, step_rmse, size=n_scenarios)
        scenario_returns = pred_return + noise
        scenario_prices = scenario_prices * (1 + scenario_returns)
        price_paths[:, step_idx] = scenario_prices

        # 3 — Compute percentile-based statistics across all scenarios
        p5 = float(np.percentile(scenario_prices, 5))
        p25 = float(np.percentile(scenario_prices, 25))
        p50 = float(np.percentile(scenario_prices, 50))  # median
        p75 = float(np.percentile(scenario_prices, 75))
        p95 = float(np.percentile(scenario_prices, 95))

        forecasts.append(ForecastPoint(
            date=str(pd.Timestamp(forecast_date).date()),
            predicted=round(p50, 2),
            upper95=round(p95, 2),
            lower95=round(p5, 2),
            upper68=round(p75, 2),
            lower68=round(p25, 2),
        ))

        # 4 — Use the median price to grow the rolling history for the next step
        median_price = p50
        future_row = _synthetic_future_row(rolling_history, forecast_date, median_price)
        rolling_history = pd.concat([rolling_history, future_row])

    # Select a subset of scenario paths for frontend display
    display_indices = np.linspace(0, n_scenarios - 1, min(N_DISPLAY_PATHS, n_scenarios), dtype=int)
    display_paths = [
        [round(float(current_price), 2)] + [round(float(v), 2) for v in price_paths[i]]
        for i in display_indices
    ]

    return forecasts, display_paths


# ---------------------------------------------------------------------------
# Direct multi-horizon forecast (when per-horizon bundles exist)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------

@router.post("", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Run next-day direction inference using trained symbol/model bundles."""
    symbol = req.symbol.upper()
    model_type = req.model_type.value
    raw_df = _download_prediction_data(symbol)
    current_price = float(raw_df["Close"].iloc[-1])

    bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
    if bundle is None:
        message = f"No trained {model_type} bundle found for {symbol}"
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=None,
            available=False,
            reason="missing_bundle",
            message=message,
        )
        return PredictResponse(
            symbol=symbol,
            model_type=model_type,
            horizon=NEXT_DAY_HORIZON,
            current_price=round(current_price, 2),
            direction=None,
            signal=None,
            confidence=None,
            probability_up=None,
            probability_down=None,
            expected_move=None,
            prediction_date=None,
            model_info=model_info,
            status="unavailable",
            model_available=False,
            reason="missing_bundle",
            message=message,
            can_train=True,
        )

    legacy_message = _validate_bundle_objective(bundle)
    if legacy_message:
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=bundle,
            available=False,
            reason="legacy_bundle_requires_retraining",
            message=legacy_message,
        )
        return PredictResponse(
            symbol=symbol,
            model_type=model_type,
            horizon=NEXT_DAY_HORIZON,
            current_price=round(current_price, 2),
            direction=None,
            signal=None,
            confidence=None,
            probability_up=None,
            probability_down=None,
            expected_move=None,
            prediction_date=None,
            model_info=model_info,
            status="unavailable",
            model_available=False,
            reason="legacy_bundle_requires_retraining",
            message=legacy_message,
            can_train=True,
        )

    feature_frame = build_feature_frame(raw_df, feature_config=bundle.feature_config)

    try:
        probabilities = _predict_bundle_probabilities(bundle, feature_frame)
        prob_up = float(probability_up(probabilities)[0])
    except ValueError as exc:
        message = str(exc)
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=bundle,
            available=False,
            reason="insufficient_inference_history",
            message=message,
        )
        return PredictResponse(
            symbol=symbol,
            model_type=model_type,
            horizon=NEXT_DAY_HORIZON,
            current_price=round(current_price, 2),
            direction=None,
            signal=None,
            confidence=None,
            probability_up=None,
            probability_down=None,
            expected_move=None,
            prediction_date=None,
            model_info=model_info,
            status="unavailable",
            model_available=False,
            reason="insufficient_inference_history",
            message=message,
            can_train=True,
        )

    model_info = _build_model_info(
        symbol=symbol,
        requested_model=model_type,
        bundle=bundle,
        available=True,
    )

    prob_down = float(1.0 - prob_up)

    return PredictResponse(
        symbol=symbol,
        model_type=model_type,
        horizon=NEXT_DAY_HORIZON,
        current_price=round(current_price, 2),
        direction=direction_from_probability(prob_up),
        signal=signal_from_probability(prob_up),
        confidence=round(confidence_from_probability(prob_up), 1),
        probability_up=round(prob_up, 4),
        probability_down=round(prob_down, 4),
        expected_move=expected_move_from_probability(prob_up),
        prediction_date=_next_business_date(raw_df.index[-1]),
        model_info=model_info,
        status="ok",
        model_available=True,
        can_train=True,
    )


# ---------------------------------------------------------------------------
# GET /api/predict/historical-signals/{symbol}
# ---------------------------------------------------------------------------

@router.get("/historical-signals/{symbol}", response_model=List[HistoricalSignal])
async def get_historical_signals(
    symbol: str,
    days: int = Query(90, ge=10, le=365),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm"]),
):
    """Return recent next-day direction signals using the saved bundle."""
    symbol = symbol.upper()
    raw_df = _download_prediction_data(symbol)
    bundle = load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
    if bundle is None:
        raise HTTPException(404, f"No trained {model_type} horizon-1 model bundle found for {symbol}")
    if _validate_bundle_objective(bundle):
        raise HTTPException(409, f"{model_type} bundle for {symbol} must be retrained for next-day direction")

    feature_frame = build_feature_frame(raw_df, feature_config=bundle.feature_config)
    pred_index, probabilities = _predict_history_probabilities(bundle, feature_frame)
    if len(probabilities) == 0:
        return []

    prediction_frame = pd.DataFrame({"probability_up": probabilities}, index=pred_index)
    target_frame = prediction_frame.tail(days)
    signals: List[HistoricalSignal] = []
    for date_idx, row in target_frame.iterrows():
        prob_up = float(row["probability_up"])
        signal_type = signal_from_probability(prob_up)
        if signal_type == "HOLD":
            continue
        signals.append(
            HistoricalSignal(
                date=str(date_idx.date()) if hasattr(date_idx, "date") else str(date_idx),
                type=signal_type,
                confidence=round(confidence_from_probability(prob_up), 1),
                predicted_return=None,
                probability_up=round(prob_up, 4),
                direction=direction_from_probability(prob_up),
            )
        )

    return signals
