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
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.schemas import (
    ForecastPoint,
    HistoricalSignal,
    PredictRequest,
    PredictResponse,
    EnsemblePredictRequest,
    EnsemblePredictResponse,
    EnsembleTrainRequest,
    EnsembleSummary,
    EnsembleForecastPoint,
    BestModelForecastResponse,
    ChartForecastPoint,
    DirectionCall,
    ForecastHistoryResponse,
    PriceBar,
    SelectedModel,
    SimpleForecastPoint,
    SimpleForecastResponse,
    SUPPORTED_FORECAST_HORIZONS,
)
from src.data.ohlcv_cache import cached_download
from src.defaults import DEFAULT_INDEX_SYMBOL
from src.models.ensemble_predictor import (
    EnsemblePricePredictor,
    ensemble_availability,
    ensemble_bundle_status,
    regression_bundle_status,
)
from src.api.schemas.schemas import BaseModel
class TrainStatus(BaseModel):
    job_id: str
    status: str
    error: Optional[str] = None
    progress: float = 0.0
    metrics: Optional[Dict] = None

import threading
import uuid
# Bounded: unbounded job dicts retained every job for the process lifetime.
_ensemble_jobs: TTLCache = TTLCache(maxsize=200, ttl=24 * 3600)
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
from src.models.model_bundle import StaleBundleError, load_model_bundle
from src.models.model_selection import (
    DIRECTION_REPORT_MODELS,
    direction_report_candidates,
    model_label,
    select_best_models,
)
from src.models.foundation.forecast_service import (
    FOUNDATION_MEMBERS,
    ForecastUnavailable,
    run_foundation_forecast,
)
from src.models.preparation import preparation_state
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

LOOKBACK_DAYS = 1825
MIN_LONG_WINDOW_HISTORY_ROWS = 260
N_SCENARIOS = 50       # Monte Carlo paths
N_DISPLAY_PATHS = 12   # scenario lines sent to the frontend


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _download_prediction_data(symbol: str) -> pd.DataFrame:
    end = pd.Timestamp.utcnow().tz_localize(None).normalize()
    start = end - pd.Timedelta(days=LOOKBACK_DAYS)

    def _fetch() -> Optional[pd.DataFrame]:
        import yfinance as yf

        raw = yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            prepost=True,
        )
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return raw

    # Every prediction request used to issue its own download; the disk cache
    # collapses those into one call per symbol per TTL window. The "1d-prepost"
    # key keeps these bars separate from the regular-hours frames training uses.
    df = cached_download(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        "1d-prepost",
        _fetch,
    )
    if df is None or df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    df = df.sort_index().ffill().dropna().ffill().dropna()
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    if len(df) < MIN_LONG_WINDOW_HISTORY_ROWS:
        logger.error(
            "Prediction download for %s returned only %s rows; need at least %s rows for long-window indicators such as SMA_200",
            symbol,
            len(df),
            MIN_LONG_WINDOW_HISTORY_ROWS,
        )
        raise HTTPException(
            422,
            f"Need at least {MIN_LONG_WINDOW_HISTORY_ROWS} historical candles for prediction feature engineering.",
        )
    return df


def _valid_price(value) -> Optional[float]:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(price) or price <= 0:
        return None
    return price


def _latest_available_price(symbol: str, raw_df: pd.DataFrame) -> Tuple[float, str]:
    """Resolve the base price, including extended-hours quotes when available."""
    latest_close = _valid_price(raw_df["Close"].iloc[-1])
    if latest_close is None:
        raise HTTPException(422, "Latest close price is unavailable for prediction.")

    info: Dict[str, object] = {}
    fast_info: Dict[str, object] = {}
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        try:
            info = ticker.get_info() or {}
        except Exception as exc:
            logger.info("Could not fetch yfinance info for %s: %s", symbol, exc)
        try:
            raw_fast_info = getattr(ticker, "fast_info", {}) or {}
            fast_info = dict(raw_fast_info) if not isinstance(raw_fast_info, dict) else raw_fast_info
        except Exception as exc:
            logger.info("Could not fetch yfinance fast_info for %s: %s", symbol, exc)
    except Exception as exc:
        logger.info("Could not initialise yfinance ticker for %s: %s", symbol, exc)

    state = str(info.get("marketState") or info.get("market_state") or "").upper()
    regular = _valid_price(info.get("regularMarketPrice") or fast_info.get("last_price"))
    premarket = _valid_price(info.get("preMarketPrice"))
    postmarket = _valid_price(info.get("postMarketPrice"))

    if state == "POST":
        candidates = [("post_market", postmarket), ("regular_market", regular), ("pre_market", premarket)]
    elif state == "REGULAR":
        candidates = [("regular_market", regular), ("post_market", postmarket), ("pre_market", premarket)]
    elif state == "PRE":
        candidates = [("pre_market", premarket), ("post_market", postmarket), ("regular_market", regular)]
    else:
        candidates = [("post_market", postmarket), ("pre_market", premarket), ("regular_market", regular)]

    for source, price in candidates:
        if price is not None:
            return price, source
    return latest_close, "latest_close"


def _next_business_date(last_index: pd.Timestamp) -> str:
    date = pd.Timestamp(last_index).tz_localize(None) if pd.Timestamp(last_index).tzinfo else pd.Timestamp(last_index)
    next_date = pd.bdate_range(start=date, periods=2)[1]
    return str(next_date.date())


def _load_next_day_bundle(model_type: str, symbol: str):
    """Load the next-day bundle, treating an unloadable artifact as no bundle.

    A symbol that only ever went through the legacy regression pipeline has a
    regressor sitting where the direction classifier expects one. Every caller
    below already answers "no trained bundle" with a train-this-symbol
    response; without this guard the load raises first and the request 500s.
    """
    try:
        return load_model_bundle(model_type=model_type, symbol=symbol, horizon=NEXT_DAY_HORIZON)
    except StaleBundleError as exc:
        logger.warning("Ignoring stale %s bundle for %s: %s", model_type, symbol, exc)
        return None


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
# Monte Carlo recursive forecast  (NEW â€” replaces old single-path approach)
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
        # 1 â€” Run the model ONCE on the current rolling history
        feature_frame = build_feature_frame(rolling_history, feature_config=bundle.feature_config)
        pred_return, step_rmse = _predict_return_from_bundle(bundle, feature_frame)

        # 2 â€” Sample per-scenario noise and advance each path
        noise = rng.normal(0, step_rmse, size=n_scenarios)
        scenario_returns = pred_return + noise
        scenario_prices = scenario_prices * (1 + scenario_returns)
        price_paths[:, step_idx] = scenario_prices

        # 3 â€” Compute percentile-based statistics across all scenarios
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

        # 4 â€” Use the median price to grow the rolling history for the next step
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



def _unavailable_message(
    symbol: str,
    horizon: int,
    detail: str,
    preparation: Optional[Dict],
) -> str:
    """
    Say which of four things is true, rather than assuming the common one.

    The distinction that matters is between a model that is *not yet* trained and
    one that *was* trained and did not earn the right to be served. Telling a user
    to retrain the second kind sends them round a loop that ends here again, since
    refitting the same bars reproduces the same verdict. But the reverse mistake
    is just as bad: reporting a missing bundle as a settled measurement would hide
    a training run that failed or never started.
    """
    status = (preparation or {}).get("status")
    unproven = "constant forecast" in detail

    if status in ("queued", "running"):
        return (
            f"Preparing {symbol} models. Training is running in the background; "
            f"this view updates as soon as it finishes."
        )

    if status == "failed":
        return (
            f"Could not prepare {symbol} models: "
            f"{(preparation or {}).get('error') or 'training failed'}."
        )

    if unproven:
        return (
            f"No forecast for {symbol} at horizon {horizon}: {detail}. This is a "
            f"measured out-of-sample result, not a missing model — retraining the "
            f"same history reproduces the same verdict, so none is scheduled."
        )

    if status == "completed":
        # Training ran and the bundle still is not servable. The per-model
        # reasons live in `warnings`; the first one is the actionable one.
        warnings = (preparation or {}).get("warnings") or []
        suffix = f" ({warnings[0]})" if warnings else ""
        return (
            f"Preparation finished for {symbol} but no forecast is available at "
            f"horizon {horizon}: {detail}{suffix}."
        )

    return (
        f"No forecast for {symbol} at horizon {horizon}: {detail}. Automatic "
        f"preparation did not start — request it with POST /api/models/{symbol}/prepare."
    )


def _unavailable_prediction_response(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    current_price: float,
    current_price_source: str,
    reason: str,
    message: str,
    preparation: Optional[Dict] = None,
) -> PredictResponse:
    preparing = bool(preparation and preparation.get("status") in ("queued", "running"))
    status = "preparing" if preparing else "unavailable"
    model_info = {
        "requested_model": model_type,
        "requested_horizon": horizon,
        "model_available": False,
        "status": status,
        "reason": reason,
        "message": message,
        "can_train": True,
        "source": "missing_bundle" if reason == "missing_bundle" else status,
    }
    return PredictResponse(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        prediction_date=None,
        forecasts=[],
        model_info=model_info,
        status=status,
        model_available=False,
        reason=reason,
        message=message,
        can_train=True,
        preparation=preparation,
    )


def _forecast_point_from_ensemble_point(point: Dict) -> ForecastPoint:
    upper95 = point.get("upper_95", point.get("upper"))
    lower95 = point.get("lower_95", point.get("lower"))
    upper68 = point.get("upper_68", point.get("upper"))
    lower68 = point.get("lower_68", point.get("lower"))
    return ForecastPoint(
        date=str(point["date"]),
        predicted=round(float(point["predicted"]), 2),
        upper95=round(float(upper95), 2),
        lower95=round(float(lower95), 2),
        upper68=round(float(upper68), 2),
        lower68=round(float(lower68), 2),
    )


def _predict_regression_model(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> Optional[PredictResponse]:
    predictor = EnsemblePricePredictor()
    forecast = predictor.predict(
        symbol=symbol,
        horizon=horizon,
        raw_df=raw_df,
        model_types=[model_type],
        current_price=current_price,
    )
    if forecast is None:
        return None

    points = [_forecast_point_from_ensemble_point(point) for point in forecast.forecast_points]
    final_point = points[-1] if points else None
    if final_point is None:
        return None

    change_pct = ((final_point.predicted - current_price) / max(current_price, 1e-6)) * 100.0
    model_result = forecast.model_predictions[0] if forecast.model_predictions else None
    model_info = {
        "requested_model": model_type,
        "serving_model": model_type,
        "requested_horizon": horizon,
        "objective": "future_return_pct",
        "target_type": "return_regression",
        "serving_mode": "price_regression",
        "artifact_source": "return_regression_bundle",
        "trained_at": forecast.trained_at,
        "model_available": True,
        "status": "available",
        "reason": None,
        "message": None,
        "can_train": True,
        "source": "trained_bundle",
        "feature_count": forecast.feature_count,
        "current_price_source": current_price_source,
        "metrics": model_result.__dict__ if model_result is not None else {},
        "forecast_engine": forecast.forecast_engine,
        "path_type": getattr(forecast, "path_type", "compounded_interpolation"),
        "per_step_predictions": getattr(forecast, "per_step_predictions", False),
        "model_output_count": getattr(forecast, "model_output_count", 0),
    }
    return PredictResponse(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        predicted_price=final_point.predicted,
        target_price=final_point.predicted,
        expected_change_pct=round(float(change_pct), 2),
        upper95=final_point.upper95,
        lower95=final_point.lower95,
        upper68=final_point.upper68,
        lower68=final_point.lower68,
        direction=forecast.signal,
        signal=forecast.signal,
        confidence=None,
        probability_up=None,
        probability_down=None,
        expected_move=f"{change_pct:+.2f}%",
        prediction_date=points[0].date if points else None,
        forecasts=points,
        model_info=model_info,
        status="ok",
        model_available=True,
        can_train=True,
        scenario_paths=forecast.scenario_paths or None,
    )


def _predict_regression_or_unavailable(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    response = _predict_regression_model(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
        raw_df=raw_df,
        current_price=current_price,
        current_price_source=current_price_source,
    )
    if response is not None:
        return response

    _, status_reason = regression_bundle_status(symbol, model_type, horizon)
    detail = status_reason or f"no usable {model_type} bundle exists"

    # Nothing to serve — so start the training that would fix it, rather than
    # returning a dead end with a command for the user to run. `preparation` is
    # None when training cannot help: the bundle exists and failed its skill
    # gate, and refitting the same bars would fail it again.
    preparation = preparation_state(symbol)
    logger.info(
        "Unusable regression bundle for %s %s h=%s: %s. Preparation %s.",
        symbol,
        model_type,
        horizon,
        detail,
        "started/joined" if preparation else "not applicable",
    )
    return _unavailable_prediction_response(
        symbol=symbol,
        model_type=model_type,
        horizon=horizon,
        current_price=current_price,
        current_price_source=current_price_source,
        reason="unproven_bundle" if "constant forecast" in detail else "missing_bundle",
        message=_unavailable_message(symbol, horizon, detail, preparation),
        preparation=preparation,
    )


# ---------------------------------------------------------------------------
# Unified models: next-timeframe price and direction from one call
# ---------------------------------------------------------------------------

# The unified bundles are trained on the direction feature set, not the
# regression one, so serving has to rebuild features the same way training did.
# Reaching for build_feature_frame here would silently hand the model a
# different, wider matrix and fail on the column check.


def _unified_feature_rows(
    raw_df: pd.DataFrame,
    feature_columns: List[str],
    rows_needed: int,
) -> Tuple[pd.Timestamp, np.ndarray]:
    """
    The most recent ``rows_needed`` complete feature rows, in training column order.

    A tabular model needs one row; an LSTM needs its whole lookback window. Both
    come from :func:`build_direction_dataset`'s own feature builder, so the
    columns a bundle was fitted on are the columns it is served.
    """
    from src.features.chart_patterns import add_chart_pattern_features
    from src.features.direction_features import (
        DIRECTION_FEATURE_CONFIG,
        add_direction_features,
    )
    from src.features.feature_engineering import build_regression_feature_frame

    # Same three steps, same config, same order as build_direction_dataset.
    frame = build_regression_feature_frame(raw_df, feature_config=dict(DIRECTION_FEATURE_CONFIG))
    if frame.empty:
        raise HTTPException(422, "Not enough history to build features")
    frame = add_chart_pattern_features(add_direction_features(frame))

    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise HTTPException(
            422,
            f"Feature pipeline is missing {len(missing)} column(s) the bundle was trained on: "
            f"{missing[:5]}. Retrain the bundle.",
        )

    complete = frame[feature_columns].dropna()
    if len(complete) < rows_needed:
        raise HTTPException(
            422,
            f"Need {rows_needed} complete feature rows for inference, have {len(complete)}",
        )
    window = complete.iloc[-rows_needed:]

    # A sequence model's lookback has to be consecutive bars. Dropping
    # incomplete rows can in principle leave a hole, and a gapped window would
    # be served without complaint and quietly mean something else -- the model
    # would read a jump across the gap as a real price move.
    if rows_needed > 1:
        expected = frame.index[frame.index.get_loc(window.index[0]) : ]
        if not window.index.equals(expected[:rows_needed]):
            raise HTTPException(
                422,
                f"The last {rows_needed} usable feature rows are not consecutive bars; "
                "the feature window has a gap. Refresh the price history and retry.",
            )

    return window.index[-1], window.to_numpy(dtype=np.float64)


def _predict_unified_kronos(
    *,
    symbol: str,
    requested_horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    """
    Zero-shot next-day forecast from the candlestick foundation model.

    Kronos never sees the engineered features: it samples tomorrow's candle from
    the raw bars, and both outputs fall out of the same set of draws -- the
    median sampled close is the price, the share of draws above today's close is
    P(up).
    """
    from src.models.unified_models import foundation_model_availability, get_kronos_singleton

    available, reason = foundation_model_availability("unified_kronos")
    if not available:
        return _unavailable_prediction_response(
            symbol=symbol,
            model_type="unified_kronos",
            horizon=requested_horizon,
            current_price=current_price,
            current_price_source=current_price_source,
            reason="model_not_installed",
            message=(
                f"Kronos is not installed ({reason}). Run scripts/setup_kronos.py and "
                'pip install -e ".[foundation]".'
            ),
        )

    kronos = get_kronos_singleton()
    kronos.set_ohlcv_context(raw_df)
    as_of = pd.Timestamp(raw_df.index[-1])
    # Kronos rebases its forecast onto the last *bar* close, so the band is
    # anchored there even when current_price came from a live quote.
    predicted_price, p_up = kronos.predict_latest(as_of, float(raw_df["Close"].iloc[-1]))

    return _unified_prediction_response(
        symbol=symbol,
        model_type="unified_kronos",
        requested_horizon=requested_horizon,
        raw_df=raw_df,
        current_price=current_price,
        current_price_source=current_price_source,
        predicted_price=predicted_price,
        p_up=p_up,
        source="zero_shot_foundation_model",
        can_train=False,
        extra_info={
            "pretrained": True,
            "gradient_updates": 0,
            "sample_count": kronos.model.sample_count,
            "lookback_bars": kronos.model.lookback,
            # The Monte-Carlo standard error on P(up) at its worst (p = 0.5).
            # Reported so a 52% reading is not mistaken for a real edge.
            "monte_carlo_se_of_p_up": round(float(np.sqrt(0.25 / kronos.model.sample_count)), 4),
        },
    )


def _predict_unified_bundle(
    *,
    symbol: str,
    model_type: str,
    requested_horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    """Next-day forecast from a trained unified bundle (XGBoost / Random Forest / LSTM)."""
    bundle = _load_next_day_bundle(model_type, symbol)
    if bundle is None:
        # Unified bundles are not in the default preparation plan — nothing
        # renders them until a user picks one by name. Asking for one is that
        # signal, so it is trained on demand here rather than for every ticker.
        preparation = preparation_state(symbol, unified_models=[model_type])
        return _unavailable_prediction_response(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            current_price=current_price,
            current_price_source=current_price_source,
            reason="missing_bundle",
            message=_unavailable_message(
                symbol,
                requested_horizon,
                f"no {model_type} bundle is trained for {symbol}",
                preparation,
            ),
            preparation=preparation,
        )

    sequence_length = max(1, int(getattr(bundle.model, "sequence_length", 1)))
    timestamp, rows = _unified_feature_rows(raw_df, bundle.feature_columns, sequence_length)
    if bundle.scaler is not None:
        rows = bundle.scaler.transform(rows)

    predicted_price, p_up = bundle.model.predict_latest(
        np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0), current_price
    )

    return _unified_prediction_response(
        symbol=symbol,
        model_type=model_type,
        requested_horizon=requested_horizon,
        raw_df=raw_df,
        current_price=current_price,
        current_price_source=current_price_source,
        predicted_price=predicted_price,
        p_up=p_up,
        source="trained_bundle",
        can_train=True,
        extra_info={
            "version_id": bundle.version_id,
            "trained_at": bundle.metadata.get("trained_at"),
            "sequence_length": sequence_length,
            "feature_count": len(bundle.feature_columns),
            "features_as_of": str(timestamp.date()),
        },
    )


def _unified_prediction_response(
    *,
    symbol: str,
    model_type: str,
    requested_horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
    predicted_price: float,
    p_up: float,
    source: str,
    can_train: bool,
    extra_info: Optional[Dict] = None,
) -> PredictResponse:
    """
    Package a unified model's two outputs into the dashboard's response shape.

    The horizon is pinned to one bar whatever was requested. These models are
    trained to forecast the next timeframe only; stretching a one-step forecast
    over 30 days would produce a curve with no model behind it.
    """
    expected_change_pct = ((predicted_price / current_price) - 1.0) * 100.0 if current_price else 0.0
    prediction_date = _next_business_date(raw_df.index[-1])

    # The price and the probability come from two heads trained on different
    # losses, so they can point opposite ways on a near-coin-flip day. That is
    # information, not a bug -- but it has to be visible, or the dashboard shows
    # "UP 51%" above a lower target price with nothing to explain it.
    price_says_up = predicted_price >= current_price
    heads_agree = price_says_up == (p_up >= 0.5)

    model_info = {
        "requested_model": model_type,
        "serving_model": model_type,
        "requested_horizon": requested_horizon,
        "served_horizon": NEXT_DAY_HORIZON,
        "objective": "unified_price_and_direction",
        "model_available": True,
        "status": "available",
        "source": source,
        "can_train": can_train,
        "heads_agree": bool(heads_agree),
        "direction_label": "UP" if p_up >= 0.5 else "DOWN",
        **(extra_info or {}),
    }
    if not heads_agree:
        model_info["heads_note"] = (
            "The price head and the direction head disagree on this bar; "
            "P(up) is close to 0.5, so treat the signal as weak."
        )
    if requested_horizon != NEXT_DAY_HORIZON:
        model_info["horizon_note"] = (
            f"{model_type} forecasts one timeframe ahead; the {requested_horizon}-day "
            "request was served as next-day."
        )

    quantiles = (extra_info or {}).get("quantiles", {})
    upper95 = quantiles.get(0.95, quantiles.get(0.975, predicted_price))
    lower95 = quantiles.get(0.05, quantiles.get(0.025, predicted_price))
    upper68 = quantiles.get(0.84, quantiles.get(0.85, predicted_price))
    lower68 = quantiles.get(0.16, quantiles.get(0.15, predicted_price))

    return PredictResponse(
        symbol=symbol,
        model_type=model_type,
        horizon=NEXT_DAY_HORIZON,
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        predicted_price=round(float(predicted_price), 2),
        target_price=round(float(predicted_price), 2),
        expected_change_pct=round(float(expected_change_pct), 2),
        direction=direction_from_probability(p_up),
        signal=signal_from_probability(p_up),
        confidence=round(confidence_from_probability(p_up), 4),
        probability_up=round(float(p_up), 4),
        probability_down=round(float(1.0 - p_up), 4),
        expected_move=f"{expected_change_pct:+.2f}%",
        prediction_date=prediction_date,
        forecasts=[
            ForecastPoint(
                date=prediction_date,
                predicted=round(float(predicted_price), 2),
                upper95=round(float(upper95), 2),
                lower95=round(float(lower95), 2),
                upper68=round(float(upper68), 2),
                lower68=round(float(lower68), 2),
            )
        ],
        model_info=model_info,
        status="ok",
        model_available=True,
        can_train=can_train,
        scenario_paths=None,
    )


def _predict_unified_univariate(
    *,
    symbol: str,
    model_type: str,
    requested_horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    """
    Zero-shot next-bar forecast from TimesFM or Chronos.

    These are optional comparison models: general-purpose time-series
    forecasters rather than financial ones. When the package is absent the route
    says so instead of failing, because the rest of the comparison stands
    without them.
    """
    from src.models.unified_models import build_unified_model, foundation_model_availability

    available, reason = foundation_model_availability(model_type)
    if not available:
        return _unavailable_prediction_response(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            current_price=current_price,
            current_price_source=current_price_source,
            reason="model_not_installed",
            message=(
                f"{model_type} is an optional comparison model and is not installed "
                f'({reason}). Install it with: pip install -e ".[comparison]".'
            ),
        )

    model = build_unified_model(model_type)
    closes = raw_df["Close"].astype(float)
    try:
        predicted_price, p_up = model.predict_latest(closes, float(closes.iloc[-1]))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    return _unified_prediction_response(
        symbol=symbol,
        model_type=model_type,
        requested_horizon=requested_horizon,
        raw_df=raw_df,
        current_price=current_price,
        current_price_source=current_price_source,
        predicted_price=predicted_price,
        p_up=p_up,
        source="zero_shot_foundation_model",
        can_train=False,
        extra_info={"pretrained": True, "gradient_updates": 0, "context_bars": model.lookback},
    )


from src.models.foundation.kronos_pipeline import KronosPipeline
from src.models.foundation.chronos_pipeline import ChronosPipeline
from src.models.foundation.timesfm_pipeline import TimesFMPipeline
from src.models.foundation.baseline_pipeline import BaselinePipeline

_KRONOS_PIPELINE = None
_CHRONOS_PIPELINE = None
_TIMESFM_PIPELINE = None
_BASELINE_PIPELINE = BaselinePipeline()

def _get_foundation_pipeline(model_type: str):
    global _KRONOS_PIPELINE, _CHRONOS_PIPELINE, _TIMESFM_PIPELINE
    if model_type == "unified_kronos":
        if _KRONOS_PIPELINE is None:
            _KRONOS_PIPELINE = KronosPipeline()
        return _KRONOS_PIPELINE
    elif model_type == "unified_chronos":
        if _CHRONOS_PIPELINE is None:
            _CHRONOS_PIPELINE = ChronosPipeline()
        return _CHRONOS_PIPELINE
    elif model_type == "unified_timesfm":
        if _TIMESFM_PIPELINE is None:
            _TIMESFM_PIPELINE = TimesFMPipeline()
        return _TIMESFM_PIPELINE
    elif model_type == "baseline_rw":
        return _BASELINE_PIPELINE
    return None

def _predict_unified_model(
    *,
    symbol: str,
    model_type: str,
    requested_horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    pipeline = _get_foundation_pipeline(model_type)
    if pipeline:
        try:
            result = pipeline.predict(raw_df, horizon=1)
            predicted_price = result["price"]
            p_up = result["p_up"]
            return _unified_prediction_response(
                symbol=symbol,
                model_type=model_type,
                requested_horizon=requested_horizon,
                raw_df=raw_df,
                current_price=current_price,
                current_price_source=current_price_source,
                predicted_price=predicted_price,
                p_up=p_up,
                source="foundation_model",
                can_train=False,
                extra_info={
                    "quantiles": result.get("quantiles", {}),
                    "samples_available": "samples" in result
                }
            )
        except Exception as exc:
            logger.error(f"Pipeline {model_type} failed: {exc}")
            return _unavailable_prediction_response(
                symbol=symbol,
                model_type=model_type,
                horizon=requested_horizon,
                current_price=current_price,
                current_price_source=current_price_source,
                reason="pipeline_error",
                message=f"{model_type} prediction failed: {exc}"
            )
    arguments = {
        "symbol": symbol,
        "requested_horizon": requested_horizon,
        "raw_df": raw_df,
        "current_price": current_price,
        "current_price_source": current_price_source,
    }
    return _predict_unified_bundle(model_type=model_type, **arguments)



# ---------------------------------------------------------------------------
# POST /api/predict
# ---------------------------------------------------------------------------

@router.post("", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Run price-regression forecasts for supported horizons, else direction inference."""
    symbol = req.symbol.upper()
    model_type = req.model_type.value
    requested_horizon = int(req.horizon)
    raw_df = _download_prediction_data(symbol)
    current_price, current_price_source = _latest_available_price(symbol, raw_df)

    # ── Unified models (Kronos, unified_xgboost, …) ──────────────────────
    # These always produce a single next-day price + direction; they never
    # use the legacy per-horizon regression bundles or Monte Carlo paths.
    if model_type.startswith("unified_"):
        return _predict_unified_model(
            symbol=symbol,
            model_type=model_type,
            requested_horizon=requested_horizon,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
        )

    # ── Legacy per-horizon regression bundles ─────────────────────────────
    if requested_horizon in SUPPORTED_FORECAST_HORIZONS:
        return _predict_regression_or_unavailable(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
        )

    # ── Legacy direction-only bundles (next-day) ─────────────────────────
    bundle = _load_next_day_bundle(model_type, symbol)

    if bundle is None:
        preparation = preparation_state(symbol)
        message = _unavailable_message(
            symbol,
            requested_horizon,
            f"no trained {model_type} bundle exists",
            preparation,
        )
        return _unavailable_prediction_response(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            current_price=current_price,
            current_price_source=current_price_source,
            reason="missing_bundle",
            preparation=preparation,
            message=message,
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
            horizon=requested_horizon,
            current_price=round(current_price, 2),
            current_price_source=current_price_source,
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
        # The stored bundle was trained against a different feature set. Retraining is a
        # multi-minute job, so it is never run inline — the client is told to submit it to
        # POST /api/training/train, and the existing bundle is left on disk untouched.
        logger.info(
            "Feature mismatch for %s (%s): %s. Bundle needs retraining via /api/training/train.",
            symbol,
            model_type,
            exc,
        )
        message = (
            f"The stored {model_type} bundle for {symbol} was trained against an older feature "
            f"set and can no longer be used for inference. Retrain it via POST /api/training/train."
        )
        model_info = _build_model_info(
            symbol=symbol,
            requested_model=model_type,
            bundle=bundle,
            available=False,
            reason="stale_bundle_requires_retraining",
            message=message,
        )
        return PredictResponse(
            symbol=symbol,
            model_type=model_type,
            horizon=requested_horizon,
            current_price=round(current_price, 2),
            current_price_source=current_price_source,
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
            reason="stale_bundle_requires_retraining",
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
    forecasts: List[ForecastPoint] = []
    scenario_paths: Optional[List[List[float]]] = None
    try:
        forecasts, scenario_paths = _monte_carlo_recursive_forecast(
            bundle,
            raw_df,
            requested_horizon,
            current_price,
        )
    except Exception as exc:
        logger.warning("Could not build recursive forecast for %s %s: %s", symbol, model_type, exc)

    return PredictResponse(
        symbol=symbol,
        model_type=model_type,
        horizon=requested_horizon,
        current_price=round(current_price, 2),
        current_price_source=current_price_source,
        direction=direction_from_probability(prob_up),
        signal=signal_from_probability(prob_up),
        confidence=round(confidence_from_probability(prob_up), 1),
        probability_up=round(prob_up, 4),
        probability_down=round(prob_down, 4),
        expected_move=expected_move_from_probability(prob_up),
        prediction_date=_next_business_date(raw_df.index[-1]),
        forecasts=forecasts,
        model_info=model_info,
        status="ok",
        model_available=True,
        can_train=True,
        scenario_paths=scenario_paths,
    )


@router.get("", response_model=PredictResponse)
def predict_query(
    symbol: str = Query(DEFAULT_INDEX_SYMBOL),
    model: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm"]),
    model_type: Optional[str] = Query(None, enum=["xgboost", "random_forest", "lstm"]),
    horizon: int = Query(30, ge=1, le=120),
):
    """Query-string friendly prediction endpoint used by browser/debug flows."""
    resolved_model = model_type or model
    return predict(
        PredictRequest(
            symbol=symbol,
            model_type=resolved_model,
            horizon=horizon,
        )
    )


# ---------------------------------------------------------------------------
# GET /api/predict/best/{symbol}
# ---------------------------------------------------------------------------
#
# One request, everything the chart overlay draws.
#
# The dashboard superimposes a forecast on the price chart, so exactly one
# model's numbers end up on screen and something has to choose it. That choice
# is src.models.model_selection, deliberately not this route: the route serves
# the winner, it does not decide who won.
#
# Two winners, because the evaluation suite scores price and direction apart
# and refuses to merge them. The trajectory and its band come from whoever won
# MAE/RMSE/MAPE/R-squared; the up/down call comes from whoever won
# accuracy/F1/AUC. They are usually different models, and the payload names
# both rather than letting the chart imply one model produced everything on it.


#: Prefix of the model names the unified serving path answers for. Anything
#: else is a per-horizon regression bundle or a direction classifier.
_UNIFIED_PREFIX = "unified_"


def _selected_model_payload(selection, metric_family: str) -> Optional[SelectedModel]:
    """Turn a ranking outcome into the response's winner block."""
    winner = selection.winner
    if winner is None:
        return None

    row = next(
        (item for item in selection.ranked if item["model_type"] == winner.model_type),
        None,
    )
    scores = winner.price if metric_family == "price" else winner.direction
    return SelectedModel(
        model_type=winner.model_type,
        label=winner.label,
        evidence=winner.evidence,
        scored_horizon=winner.horizon,
        metrics={key: value for key, value in scores.items() if value is not None},
        metrics_used=list(selection.metrics_used),
        metric_winners=dict(selection.metric_winners),
        mean_rank=(row or {}).get("mean_rank"),
        n_candidates=len(selection.ranked),
        context=dict(winner.context),
    )


def _chart_points(response: PredictResponse, current_price: float) -> List[ChartForecastPoint]:
    """
    The forecast as the chart consumes it: a band per bar, plus the step's sign.

    The first point's direction is measured against today's close and every
    later one against the point before it, so the colour of each segment is the
    move that segment actually represents.
    """
    points: List[ChartForecastPoint] = []
    previous = float(current_price)
    for point in response.forecasts:
        predicted = float(point.predicted)
        change_pct = ((predicted / previous) - 1.0) * 100.0 if previous else 0.0
        # A step of under a basis point is not a direction worth colouring a
        # chart with; calling it flat keeps rounding noise out of the UI.
        if abs(change_pct) < 0.01:
            direction = "flat"
        else:
            direction = "up" if change_pct > 0 else "down"
        points.append(
            ChartForecastPoint(
                date=point.date,
                predicted=round(predicted, 2),
                upper95=round(float(point.upper95), 2),
                lower95=round(float(point.lower95), 2),
                upper68=round(float(point.upper68), 2),
                lower68=round(float(point.lower68), 2),
                direction=direction,
                change_pct=round(change_pct, 4),
            )
        )
        previous = predicted
    return points


def _serve_price_winner(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
) -> PredictResponse:
    """Run whichever serving path the price winner belongs to."""
    arguments = {
        "symbol": symbol,
        "raw_df": raw_df,
        "current_price": current_price,
        "current_price_source": current_price_source,
    }
    if model_type.startswith(_UNIFIED_PREFIX):
        return _predict_unified_model(
            model_type=model_type, requested_horizon=horizon, **arguments
        )
    return _predict_regression_or_unavailable(
        model_type=model_type, horizon=horizon, **arguments
    )


def _direction_from_classifier(
    symbol: str,
    model_type: str,
    raw_df: pd.DataFrame,
) -> DirectionCall:
    """
    The next-bar call from one of the walk-forward direction classifiers.

    These are the estimators GET /api/direction/{symbol} serves, and this is the
    same live call it makes: fit on every labelled row, applied to the one bar
    whose next-day move has not happened yet.
    """
    from src.models.direction_pipeline import predict_next_session

    try:
        prediction = predict_next_session(raw_df, model_name=model_type)
    except Exception as exc:  # noqa: BLE001 - a failed gauge must not 500 the chart
        logger.warning("Direction winner %s failed for %s: %s", model_type, symbol, exc)
        return DirectionCall(
            source="unavailable",
            message=f"The {model_type} direction model could not produce a call ({exc}).",
        )

    if prediction is None:
        return DirectionCall(
            source="unavailable",
            message="No bar has a complete feature vector for the direction model.",
        )

    p_up = float(prediction["probability_up"])
    return DirectionCall(
        direction="UP" if p_up >= 0.5 else "DOWN",
        probability_up=round(p_up, 4),
        probability_down=round(1.0 - p_up, 4),
        confidence=round(confidence_from_probability(p_up), 1),
        signal=signal_from_probability(p_up),
        prediction_date=prediction.get("target_date") or _next_business_date(raw_df.index[-1]),
        source="direction_classifier",
    )


def _direction_call(
    *,
    symbol: str,
    model_type: str,
    horizon: int,
    raw_df: pd.DataFrame,
    current_price: float,
    current_price_source: str,
    price_response: Optional[PredictResponse],
    price_model_type: Optional[str],
) -> DirectionCall:
    """
    The direction winner's call, from whichever path can produce it.

    When the direction winner is also the price winner, the forecast already in
    hand is reused rather than recomputed. Kronos in particular is a transformer
    forward pass, and running it twice for a number already returned would
    double the latency of every chart load.
    """
    if (
        price_response is not None
        and price_model_type == model_type
        and price_response.probability_up is not None
    ):
        p_up = float(price_response.probability_up)
        return DirectionCall(
            direction="UP" if p_up >= 0.5 else "DOWN",
            probability_up=round(p_up, 4),
            probability_down=round(1.0 - p_up, 4),
            confidence=price_response.confidence,
            signal=price_response.signal,
            prediction_date=price_response.prediction_date,
            source="unified_bundle",
        )

    if model_type in DIRECTION_REPORT_MODELS:
        return _direction_from_classifier(symbol, model_type, raw_df)

    if model_type.startswith(_UNIFIED_PREFIX):
        response = _predict_unified_model(
            symbol=symbol,
            model_type=model_type,
            requested_horizon=NEXT_DAY_HORIZON,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
        )
        if response.status != "ok" or response.probability_up is None:
            return DirectionCall(
                source="unavailable",
                message=response.message or f"{model_type} produced no direction probability.",
            )
        p_up = float(response.probability_up)
        return DirectionCall(
            direction="UP" if p_up >= 0.5 else "DOWN",
            probability_up=round(p_up, 4),
            probability_down=round(1.0 - p_up, 4),
            confidence=response.confidence,
            signal=response.signal,
            prediction_date=response.prediction_date,
            source="unified_bundle",
        )

    # A per-horizon regression bundle. It has a directional accuracy but no
    # probability: the sign of its predicted return is the whole of its
    # direction output, so the call is reported without a confidence rather
    # than with a fabricated one.
    response = price_response
    if response is None or price_model_type != model_type:
        response = _predict_regression_or_unavailable(
            symbol=symbol,
            model_type=model_type,
            horizon=horizon,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
        )
    if response.status != "ok" or response.target_price is None:
        return DirectionCall(
            source="unavailable",
            message=response.message or f"{model_type} produced no forecast to take a sign from.",
        )

    rises = float(response.target_price) >= float(current_price)
    return DirectionCall(
        direction="UP" if rises else "DOWN",
        signal=response.signal,
        prediction_date=response.prediction_date,
        source="regression_sign",
        message=(
            f"{model_label(model_type)} is a return-regression bundle: the call is the sign "
            f"of its {horizon}-day forecast, and it carries no calibrated probability."
        ),
    )


def _direction_gate(symbol: str, model_type: str) -> Tuple[bool, Optional[str]]:
    """
    Whether the direction winner's call may be presented as actionable.

    Reuses the verdict the walk-forward run already recorded, so the chart and
    GET /api/direction/{symbol} can never disagree about the same model. A model
    with no stored report is not gated here: the ranking only ever selects from
    servable candidates, and a bundle's own skill gate was applied there.
    """
    for candidate in direction_report_candidates(symbol, (model_type,)):
        if candidate.blocked_reason:
            return False, f"Not tradeable: {candidate.blocked_reason}."
        return True, None
    return True, None


@router.get("/best/{symbol}", response_model=BestModelForecastResponse)
def predict_best_model(
    symbol: str,
    horizon: int = Query(30, ge=1, le=120, description="Prediction window in trading days"),
):
    """
    The best performing model's forecast and direction, ready to draw.

    ``horizon`` is the window the dashboard's 7D/15D/30D/60D selector is on. It
    scopes the comparison as well as the forecast: models are only ranked
    against models scored at the same horizon, because a 30-day error and a
    1-day error are not the same measurement.

    The unified and foundation models answer for the next bar whatever is
    requested. When one of them wins, ``price_model.scored_horizon`` is 1 and
    the forecast is one point long - the response says so rather than stretching
    a one-step number across thirty days.
    """
    symbol = symbol.upper().strip()
    raw_df = _download_prediction_data(symbol)
    current_price, current_price_source = _latest_available_price(symbol, raw_df)
    as_of = str(pd.Timestamp(raw_df.index[-1]).date())

    best = select_best_models(symbol, horizon, reference_price=current_price)
    price_winner = best.price.winner
    direction_winner = best.direction.winner

    if price_winner is None and direction_winner is None:
        preparation = preparation_state(symbol)
        preparing = bool(preparation and preparation.get("status") in ("queued", "running"))
        return BestModelForecastResponse(
            symbol=symbol,
            horizon=horizon,
            as_of=as_of,
            current_price=round(float(current_price), 2),
            current_price_source=current_price_source,
            selection=best.as_dict(),
            status="preparing" if preparing else "unavailable",
            reason="no_servable_model",
            preparation=preparation,
            message=(
                f"Training models for {symbol}; the forecast overlay appears once one of "
                f"them has an out-of-sample record to stand on."
                if preparing
                else best.price.reason
                or f"No model has a scorecard for {symbol} at a {horizon}-day horizon yet."
            ),
        )

    price_response: Optional[PredictResponse] = None
    forecast: List[ChartForecastPoint] = []
    price_message: Optional[str] = None
    if price_winner is not None:
        price_response = _serve_price_winner(
            symbol=symbol,
            model_type=price_winner.model_type,
            horizon=horizon,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
        )
        if price_response.status == "ok":
            forecast = _chart_points(price_response, current_price)
        else:
            price_message = price_response.message

    direction: Optional[DirectionCall] = None
    if direction_winner is not None:
        direction = _direction_call(
            symbol=symbol,
            model_type=direction_winner.model_type,
            horizon=horizon,
            raw_df=raw_df,
            current_price=current_price,
            current_price_source=current_price_source,
            price_response=price_response,
            price_model_type=price_winner.model_type if price_winner else None,
        )
        tradeable, gate_reason = _direction_gate(symbol, direction_winner.model_type)
        direction.tradeable = tradeable and direction.source != "unavailable"
        direction.gate_reason = gate_reason

    # "No model qualified" and "the winner could not be run" are different
    # states, and the second one carries a different remedy, so the message
    # falls back to the ranking's own reason rather than to nothing.
    direction_message = (direction.message if direction else None) or best.direction.reason
    has_direction = direction is not None and direction.source != "unavailable"
    if forecast and has_direction:
        status, reason, message = "ok", None, None
    elif forecast or has_direction:
        status = "partial"
        reason = "direction_model_unavailable" if forecast else "price_model_unavailable"
        message = direction_message if forecast else (price_message or best.price.reason)
    else:
        status = "unavailable"
        reason = "winners_could_not_be_served"
        message = price_message or best.price.reason or direction_message

    model_info = (price_response.model_info or {}) if price_response else {}
    return BestModelForecastResponse(
        symbol=symbol,
        horizon=horizon,
        as_of=as_of,
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        price_model=_selected_model_payload(best.price, "price"),
        direction_model=_selected_model_payload(best.direction, "direction"),
        forecast=forecast,
        direction=direction,
        path_type=model_info.get("path_type"),
        per_step_predictions=bool(model_info.get("per_step_predictions")),
        selection=best.as_dict(),
        status=status,
        reason=reason,
        message=message,
        preparation=(price_response.preparation if price_response else None),
    )


# ---------------------------------------------------------------------------
# GET /api/predict/forecast/{symbol}  — the Predictions tab, in one request
# ---------------------------------------------------------------------------

def _history_bars(raw_df: pd.DataFrame, days: int) -> List[PriceBar]:
    """The tail of the download, as candles the chart can draw."""
    window = raw_df.tail(max(int(days), 1))
    bars: List[PriceBar] = []
    for index, row in window.iterrows():
        values = [row.get(column) for column in ("Open", "High", "Low", "Close")]
        if any(value is None or not np.isfinite(float(value)) for value in values):
            continue
        open_, high, low, close = (float(value) for value in values)
        volume = row.get("Volume")
        bars.append(
            PriceBar(
                date=str(pd.Timestamp(index).date()),
                open=round(open_, 4),
                high=round(high, 4),
                low=round(low, 4),
                close=round(close, 4),
                volume=int(volume) if volume is not None and np.isfinite(float(volume)) else 0,
            )
        )
    return bars


@router.get("/history/{symbol}", response_model=ForecastHistoryResponse)
def forecast_history(
    symbol: str,
    days: int = Query(252, ge=30, le=1260, description="Historical candles to return."),
):
    """
    Just the candles, off the model path entirely.

    The Predictions tab draws its chart from this and asks for the forecast
    separately, because the two cost three orders of magnitude apart: the OHLCV
    download is ~0.08s cold and free once cached, while the forecast is ~7s
    dominated by Kronos sampling 128 transformer paths. Serving both from one
    handler -- which is what this route was split out of -- meant the chart sat
    blank for seven seconds waiting on numbers that go in a box underneath it.

    It reads the SAME `_download_prediction_data` the forecast does, so the bars
    here are the bars the models saw rather than a differently-adjusted series
    from the generic price route. The 6-hour OHLCV cache keeps the two calls on
    one frame; if they ever disagree, the chart drops a forecast point that is
    not strictly after its last candle rather than anchoring it to the wrong bar.

    No live quote is fetched here. `_latest_available_price` costs ~1.25s in
    yfinance `get_info` calls and the candles do not need it.
    """
    symbol = symbol.upper()
    raw_df = _download_prediction_data(symbol)
    return ForecastHistoryResponse(
        symbol=symbol,
        as_of=str(pd.Timestamp(raw_df.index[-1]).date()),
        bars=_history_bars(raw_df, days),
    )


#: Model output by (symbol, last bar date). The forecast depends only on the
#: bars, so it cannot change until a new one prints -- but it costs ~7s to
#: produce, and re-selecting a symbol used to pay that again. Keyed on the bar
#: rather than on a clock, so a new session invalidates it exactly.
#:
#: Only the FoundationForecast is cached, and it holds nothing quote-derived:
#: the price, the direction and the anchor all come off the bars. The current
#: price, the change measured against it and the "quote" split are recomputed
#: per request, because those do move intraday and serving a cached quote would
#: print a stale "Current Price".
_FORECAST_CACHE: TTLCache = TTLCache(maxsize=256, ttl=6 * 3600)
_FORECAST_CACHE_LOCK = threading.Lock()


def _cached_foundation_forecast(symbol: str, raw_df: pd.DataFrame, as_of: str):
    """
    The model output for ``symbol`` on the bar ``as_of``, computed at most once.

    The forecast is a pure function of the bars, so it cannot change until a new
    session prints -- and it costs ~7s to produce, almost all of it Kronos
    sampling 128 transformer paths. Without this, re-selecting a symbol paid
    that again for a number that could not have moved.

    No live quote reaches this function, and none can: the forecast is anchored
    to the last close and nothing else, which is what makes (symbol, bar) a
    complete key. Everything that depends on a quote -- the change measured
    against it, and the "quote" split -- is recomputed by the caller per request.
    """
    key = (symbol, as_of)
    with _FORECAST_CACHE_LOCK:
        hit = _FORECAST_CACHE.get(key)
    if hit is not None:
        logger.debug("%s: forecast served from cache for bar %s", symbol, as_of)
        return hit

    # Deliberately computed outside the lock. This takes seconds, and holding a
    # global lock across it would serialise every symbol behind whichever one
    # arrived first. Two concurrent requests for the same cold symbol may both
    # compute; they produce the same answer, so the duplicate work is wasted but
    # never wrong -- which is the better trade against blocking every other
    # symbol in the watchlist.
    result = run_foundation_forecast(
        raw_df,
        pipeline_factory=_get_foundation_pipeline,
        symbol=symbol,
    )
    with _FORECAST_CACHE_LOCK:
        _FORECAST_CACHE[key] = result
    return result


@router.get("/forecast/{symbol}", response_model=SimpleForecastResponse)
def simple_forecast(symbol: str):
    """
    The forecast alone. The candles come from ``GET /predict/history/{symbol}``.

    The full pipeline runs here — OHLCV, the five technical-analysis feature
    families, Kronos, Chronos-2 and TimesFM 2.5, then inverse-variance
    aggregation — and none of it is in the response. What comes back is the next
    bar the models predict and the direction they call.

    This used to return the candles too, which made the chart wait on the
    models: the bars are a cached download and the forecast is ~7s of transformer
    sampling, so bundling them held a chart that was ready in 0.08s for two
    orders of magnitude longer than it needed. The models still read the full
    download rather than any chart window, because the feature layer needs long
    windows (SMA_200 among them) that a short request would silently truncate.
    """
    symbol = symbol.upper()
    raw_df = _download_prediction_data(symbol)
    as_of = str(pd.Timestamp(raw_df.index[-1]).date())
    current_price, current_price_source = _latest_available_price(symbol, raw_df)

    try:
        result = _cached_foundation_forecast(symbol, raw_df, as_of)
    except ForecastUnavailable as exc:
        logger.info("%s: no foundation forecast — %s (%s)", symbol, exc.message, exc.members_failed)
        # A 200 with status "unavailable": the chart is drawn from its own
        # request and is worth showing even when the box has nothing to say, so
        # an error code here would be reported as a failure of both.
        return SimpleForecastResponse(
            symbol=symbol,
            status="unavailable",
            message=exc.message,
            as_of=as_of,
            # Known without any model: it is just the last bar's close.
            anchor_price=round(float(raw_df["Close"].iloc[-1]), 2),
            current_price=round(float(current_price), 2),
            current_price_source=current_price_source,
        )

    # Two reference prices, and the response has to keep them apart.
    #
    # `anchor` is the close the models read: every member computes p_up as
    # P(next bar > anchor), so the direction call and the expected move both
    # belong to it. The quote is a live price from a session no model saw --
    # `_download_prediction_data` ends its window at today's date and yfinance
    # treats that end as exclusive, so the frame stops at the previous session
    # however fresh the request is.
    #
    # Dividing the forecast by the quote and labelling the result "Expected
    # Change" is what put +3.8% beside a DOWN arrow on PLTR: the stock had
    # fallen 4% since the anchor, so a forecast 0.26% BELOW the bar the models
    # read still landed well above the quote. Both figures are served, each
    # named for what it is measured against.
    anchor = float(result.anchor_price)
    change_pct = ((result.price / anchor) - 1.0) * 100.0 if anchor else 0.0
    quote_change_pct = ((result.price / current_price) - 1.0) * 100.0 if current_price else 0.0

    # `result.split` is the two heads disagreeing on one price, which is the
    # model's to report and travels with the cached run. The quote gap is not a
    # disagreement at all -- it is the box showing a price the forecast was
    # never about -- but it reads as one, so it is detected here, per request,
    # where the quote is. It cannot be cached: the quote moves and the forecast
    # under it does not.
    rises_from_anchor = result.price >= anchor
    if result.split:
        split_reason = "heads"
    elif current_price and rises_from_anchor != (result.price >= float(current_price)):
        split_reason = "quote"
    else:
        split_reason = None

    forecast_date = _next_business_date(raw_df.index[-1])
    point = SimpleForecastPoint(
        date=forecast_date,
        predicted=round(result.price, 2),
        upper_90=round(result.upper_90, 2),
        lower_90=round(result.lower_90, 2),
        upper_68=round(result.upper_68, 2),
        lower_68=round(result.lower_68, 2),
        # The chart hangs this segment off the last candle's close, so it is
        # coloured by the move from that close. Measured against the quote
        # instead, a segment drawn falling was painted green.
        direction="up" if rises_from_anchor else "down",
        change_pct=round(change_pct, 2),
    )

    return SimpleForecastResponse(
        symbol=symbol,
        status="ok",
        as_of=as_of,
        anchor_price=round(anchor, 2),
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        forecast_price=round(result.price, 2),
        expected_change_pct=round(change_pct, 2),
        quote_change_pct=round(quote_change_pct, 2),
        direction=result.direction,
        forecast_date=forecast_date,
        models=result.members_used,
        split=split_reason is not None,
        split_reason=split_reason,
        forecast=[point],
    )


# ---------------------------------------------------------------------------
# GET /api/predict/historical-signals/{symbol}
# ---------------------------------------------------------------------------

@router.get("/historical-signals/{symbol}", response_model=List[HistoricalSignal])
def get_historical_signals(
    symbol: str,
    days: int = Query(90, ge=10, le=365),
    model_type: str = Query("xgboost", enum=["xgboost", "random_forest", "lstm"]),
):
    """Return recent next-day direction signals using the saved bundle."""
    symbol = symbol.upper()
    raw_df = _download_prediction_data(symbol)
    bundle = _load_next_day_bundle(model_type, symbol)
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

# ---------------------------------------------------------------------------
# POST /api/predict/ensemble  â€” weighted ensemble price regression forecast
# ---------------------------------------------------------------------------

@router.post("/ensemble", response_model=EnsemblePredictResponse)
def ensemble_predict(req: EnsemblePredictRequest):
    """
    The foundation ensemble, as an ``EnsemblePredictResponse``.

    Same three models and same aggregation as ``GET /predict/forecast`` --
    both delegate to :func:`run_foundation_forecast`, which is the point: this
    handler used to carry its own copy of the member loop, the quantile
    averaging and the interval assembly, so the two paths could drift into
    disagreeing about the same symbol on the same bar. There is one
    implementation now, and this endpoint is the older response shape over it,
    kept for the Analysis tab's model panel.

    The horizon is accepted and echoed but does not change the forecast: every
    member answers for the next bar (see :func:`run_foundation_forecast`). It is
    part of the request shape this endpoint has always had.
    """
    symbol = req.symbol.upper()
    horizon = req.horizon

    # _download_prediction_data raises HTTPException, which FastAPI turns into the
    # right response on its own, so it is left to propagate.
    raw_df = _download_prediction_data(symbol)
    as_of = str(pd.Timestamp(raw_df.index[-1]).date())
    current_price, current_price_source = _latest_available_price(symbol, raw_df)

    try:
        result = run_foundation_forecast(
            raw_df,
            pipeline_factory=_get_foundation_pipeline,
            symbol=symbol,
        )
    except ForecastUnavailable as exc:
        return EnsemblePredictResponse(
            symbol=symbol,
            as_of=as_of,
            # Known without any model: it is just the last bar's close.
            anchor_price=round(float(raw_df["Close"].iloc[-1]), 2),
            current_price=round(float(current_price), 2),
            current_price_source=current_price_source,
            horizon=horizon,
            status="unavailable",
            model_available=False,
            models_unavailable=exc.members_failed,
            message=exc.message,
        )

    # Measured against the close the models read, exactly as GET
    # /predict/forecast does -- and for a sharper reason here, because the
    # Analysis tab prints this change and `signal` in one string. Divided by a
    # live quote instead, the tile read "+3.82% . DOWN" on a day PLTR had
    # already gapped 4% below the last bar the ensemble saw.
    anchor = float(result.anchor_price)
    change_pct = ((result.price / anchor) - 1.0) * 100.0 if anchor else 0.0
    quote_change_pct = ((result.price / current_price) - 1.0) * 100.0 if current_price else 0.0
    prediction_date = _next_business_date(raw_df.index[-1])

    # "Reliability" describes the width of the aggregated 90% interval relative
    # to the price the band is drawn around -- dispersion, which is the only
    # thing this response actually knows. It is deliberately NOT a calibration
    # claim: no coverage check has been run here (Addendum A Requirement 6.3),
    # so the label speaks about spread rather than confidence.
    #
    # The anchor is the denominator for the same reason it is everywhere else.
    # A quote a session away scales this ratio by however far it has moved, and
    # the buckets below are 2% and 5% wide -- narrow enough for a 4% gap to
    # relabel a band that had not changed.
    relative_width = (result.upper_90 - result.lower_90) / anchor if anchor else float("inf")
    if relative_width <= 0.02:
        reliability = "Narrow spread"
    elif relative_width <= 0.05:
        reliability = "Moderate spread"
    else:
        reliability = "Wide spread"

    bounds = {
        "upper_90": round(result.upper_90, 2),
        "lower_90": round(result.lower_90, 2),
        "upper_68": round(result.upper_68, 2),
        "lower_68": round(result.lower_68, 2),
    }
    point = EnsembleForecastPoint(
        date=prediction_date,
        ensemble=round(result.price, 2),
        prediction=round(result.price, 2),
        **bounds,
    )
    summary = EnsembleSummary(
        target=round(result.price, 2),
        change_pct=round(change_pct, 2),
        quote_change_pct=round(quote_change_pct, 2),
        reliability=reliability,
        consensus="Foundation Aggregation",
        signal=result.direction,
        **bounds,
    )

    # A member that failed is reported rather than quietly dropped: a forecast
    # built from two of three models is still a forecast, but the client has to
    # be able to say so instead of presenting it as the full ensemble.
    unavailable = {
        name: reason
        for name, reason in result.members_failed.items()
        if name in FOUNDATION_MEMBERS
    }
    return EnsemblePredictResponse(
        symbol=symbol,
        as_of=as_of,
        anchor_price=round(anchor, 2),
        current_price=round(float(current_price), 2),
        current_price_source=current_price_source,
        horizon=horizon,
        status="ok",
        model_available=True,
        ensemble=summary,
        forecast_points=[point],
        weights=result.weights,
        model_output_count=len(result.weights),
        models_available=[name for name in FOUNDATION_MEMBERS if name in result.weights],
        models_unavailable=unavailable,
        degraded=bool(unavailable),
        message=(
            "Served without "
            + ", ".join(FOUNDATION_MEMBERS[name] for name in unavailable)
            + "."
        ) if unavailable else None,
    )


# ---------------------------------------------------------------------------
# POST /api/predict/ensemble/train  â€” async ensemble training trigger
# ---------------------------------------------------------------------------

def _run_ensemble_training(job_id: str, req: EnsembleTrainRequest):
    from src.models.ensemble_training import train_ensemble_for_symbol
    job = _ensemble_jobs[job_id]
    job.status = "running"
    job.progress = 0.05
    try:
        def _cb(done, total):
            job.progress = 0.05 + 0.9 * (done / max(total, 1))

        result = train_ensemble_for_symbol(
            symbol=req.symbol,
            horizons=req.horizons,
            model_types=req.model_types,
            lookback_days=req.lookback_days,
            progress_callback=_cb,
        )
        job.status = "completed"
        job.progress = 1.0
        job.metrics = result
    except Exception as exc:
        logger.exception("Ensemble training failed for %s", req.symbol)
        job.status = "failed"
        job.error = str(exc)

@router.post("/ensemble/train")
def train_ensemble(req: EnsembleTrainRequest):
    job_id = str(uuid.uuid4())
    _ensemble_jobs[job_id] = TrainStatus(job_id=job_id, status="pending")
    thread = threading.Thread(target=_run_ensemble_training, args=(job_id, req), daemon=True)
    thread.start()
    return {
        "job_id": job_id,
        "status": "pending",
        "symbol": req.symbol.upper(),
        "horizons": req.horizons,
        "model_types": req.model_types,
    }

@router.get("/ensemble/train/status/{job_id}")
def get_ensemble_training_status(job_id: str):
    if job_id not in _ensemble_jobs:
        raise HTTPException(404, f"Ensemble job {job_id} not found")
    return _ensemble_jobs[job_id]
