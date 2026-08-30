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
    build_regression_feature_frame,
    normalize_feature_config,
    transform_feature_frame,
)
from src.models.regression_models import REGRESSOR_FACTORIES, REGRESSOR_FILE_NAMES
from src.utils.config_loader import get_env_bool
from src.utils.logger import get_logger

logger = get_logger(__name__)

REGRESSION_BUNDLES_DIR = Path("models/bundles")
SEQUENCE_LENGTH = 60
MODEL_TYPES = ["xgboost", "random_forest", "lstm"]
SUPPORTED_HORIZONS = [7, 15, 30, 60]
NEUTRAL_MIN_CHANGE_PCT = 0.5
FIXED_ENSEMBLE_WEIGHTS = {"lstm": 0.40, "xgboost": 0.35, "random_forest": 0.25}
HARD_GAP_LIMITS = {7: 0.08, 15: 0.12, 30: 0.20, 60: 0.30}

N_SCENARIOS = 400      # Monte Carlo paths behind the percentile bands
N_DISPLAY_PATHS = 12   # scenario lines handed to the frontend fan chart

# Bundles must carry proof that they beat the constant-train-mean baseline before
# they are served. Bundles trained before the gate existed have no such record and
# are treated as unproven, because those are precisely the ones that returned a
# fixed number for every input. Set QUANTVISION_ENFORCE_MODEL_SKILL=false to serve
# them anyway while retraining.
ENFORCE_MODEL_SKILL_ENV = "QUANTVISION_ENFORCE_MODEL_SKILL"

# Which mechanism builds the days between today and the horizon.
#
#   auto        (default) — use the recursive path when servable 1-day step
#                bundles exist, otherwise compound. This is the useful default:
#                per-step inference is strictly more informative when it is
#                available, and falling back is not an error worth configuring
#                around.
#   compounded  — each model emits one cumulative horizon-day return and the path
#                compounds toward it. Honest for that target, but every
#                intermediate day is interpolation, which is what draws the
#                forecast as a straight line.
#   recursive   — roll the model forward one step at a time, rebuilding features
#                from a synthetic bar each step, so every day is a real model
#                output. Requires a bundle whose target is the 1-day return;
#                against a horizon-day bundle each step would re-predict the
#                whole horizon, so the mode refuses to run and falls back.
FORECAST_MODE_ENV = "QUANTVISION_FORECAST_MODE"
FORECAST_MODE_COMPOUNDED = "compounded"
FORECAST_MODE_RECURSIVE = "recursive"
FORECAST_MODE_AUTO = "auto"
FORECAST_MODES = (FORECAST_MODE_AUTO, FORECAST_MODE_COMPOUNDED, FORECAST_MODE_RECURSIVE)


def forecast_mode() -> str:
    """Resolve the configured forecast path mode."""
    import os

    mode = str(os.getenv(FORECAST_MODE_ENV, FORECAST_MODE_AUTO)).strip().lower()
    return mode if mode in FORECAST_MODES else FORECAST_MODE_AUTO


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
    scenario_paths: List[List[float]] = field(default_factory=list)
    forecast_engine: str = "compounded_median_with_bootstrap_monte_carlo"
    # How the daily points were produced. The bundles are trained on a single
    # cumulative horizon-day return, so in "compounded" mode each model emits
    # exactly one number and the intermediate days are a compounded path to it,
    # not per-step inference. Stating that here keeps the chart from implying a
    # daily forecast the models never made.
    path_type: str = "compounded_interpolation"
    per_step_predictions: bool = False
    model_output_count: int = 0
    points_per_model_output: Optional[float] = None


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


def skill_enforcement_enabled() -> bool:
    """Whether bundles must prove out-of-sample skill before being served."""
    return get_env_bool(ENFORCE_MODEL_SKILL_ENV, True)


def bundle_skill_failure(meta: Dict) -> Optional[str]:
    """
    Return why a bundle must not be served, or None when it is fit to serve.

    A bundle qualifies by recording `passes_baseline: true`, which training sets
    when the model's out-of-sample MAE beats the constant-train-mean predictor.
    """
    if not skill_enforcement_enabled():
        return None

    if "passes_baseline" not in meta:
        return (
            "was trained before out-of-sample skill was recorded, so there is no "
            "evidence it beats a constant forecast"
        )

    if not meta.get("passes_baseline"):
        test_skill = (meta.get("skill") or {}).get("test") or {}
        score = test_skill.get("skill_score")
        spread = test_skill.get("prediction_std")
        detail = f"skill score {score:+.4f}" if isinstance(score, (int, float)) else "no skill"
        if isinstance(spread, (int, float)):
            detail += f", prediction spread {spread:.4f}"
        return f"does not beat a constant forecast ({detail})"

    return None


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
        skill_failure = bundle_skill_failure(meta)
        if skill_failure:
            logger.warning(
                "Refusing to serve %s %s h=%d: the bundle %s. Retrain via "
                "POST /api/predict/ensemble/train, or set %s=false to serve it anyway.",
                symbol,
                model_type,
                horizon,
                skill_failure,
                ENFORCE_MODEL_SKILL_ENV,
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

        if model_type == "random_forest" and hasattr(model, "model"):
            # Avoid Windows worker-spawn failures during API inference.
            try:
                model.model.n_jobs = 1
            except Exception:
                pass

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


def _compound_path(current_price: float, total_return: float, horizon: int) -> np.ndarray:
    """
    The median price path implied by a terminal return, compounded per step.

    The bundles predict one cumulative `horizon`-day return, so the only honest
    reading of the days in between is constant compounding toward that endpoint:
    price(t) = current * (1 + total_return) ** (t / horizon). This is the median
    of the simulated distribution below, and it lands exactly on the headline
    number at t = horizon. It is smooth, as a conditional expectation should be,
    but it is a curve rather than a straight ramp because growth compounds.
    """
    steps = np.arange(1, horizon + 1, dtype=np.float64)
    total_log_return = np.log1p(max(float(total_return), -0.999999))
    return float(current_price) * np.exp(total_log_return * steps / float(horizon))


def _bundle_step_horizon(bundle: Dict) -> int:
    """The horizon a bundle's target actually spans, in trading days."""
    meta = bundle.get("meta", {})
    target_col = str(meta.get("target_col") or "")
    if target_col.startswith("target_return_") and target_col.endswith("d"):
        try:
            return int(target_col[len("target_return_"):-1])
        except ValueError:
            pass
    try:
        return int(meta.get("horizon") or 0)
    except (TypeError, ValueError):
        return 0


RECURSIVE_STEP_HORIZON = 1


def load_step_bundles(symbol: str, model_types: List[str]) -> Dict[str, Dict]:
    """
    Load the 1-day-return bundles that recursive mode rolls forward.

    Recursive mode cannot use the requested-horizon bundles: a 30-day-return
    model stepped forward 30 times would apply the same 30-day forecast at every
    step. It needs models whose target *is* the next day, which live under
    horizon 1 in the same bundle tree.
    """
    bundles: Dict[str, Dict] = {}
    for mtype in model_types:
        bundle = _load_regression_bundle(symbol, mtype, RECURSIVE_STEP_HORIZON)
        if bundle is not None:
            bundles[mtype] = bundle
    return bundles


def recursive_path_supported(bundles: Dict[str, Dict]) -> Tuple[bool, Optional[str]]:
    """
    Whether per-step recursive inference is meaningful for these bundles.

    Rolling a model forward day by day only produces a daily forecast if the
    model's target *is* the next-day return. A 30-day-return bundle asked for 30
    steps would predict the next 30 days at every step and compound that 30
    times, which is not a per-step forecast — it is the same number applied
    repeatedly. So the mode is gated on the trained target.
    """
    offenders = {
        mtype: _bundle_step_horizon(bundle)
        for mtype, bundle in bundles.items()
        if _bundle_step_horizon(bundle) != 1
    }
    if offenders:
        detail = ", ".join(f"{m} targets a {h}d return" for m, h in offenders.items())
        return False, (
            f"recursive mode needs bundles trained on the 1-day return, but {detail}. "
            "Retrain with horizon=1 to enable per-step inference."
        )
    return True, None


def _bootstrap_shock_pool(raw_df: pd.DataFrame, lookback: int = 504) -> np.ndarray:
    """Demeaned recent daily log returns, used as the sampling pool for path noise."""
    source = raw_df["Adj Close"] if "Adj Close" in raw_df.columns else raw_df["Close"]
    log_returns = np.log(pd.to_numeric(source, errors="coerce")).diff().dropna()
    log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
    pool = log_returns.tail(lookback).to_numpy(dtype=np.float64)
    if pool.size < 20:
        return np.zeros(0, dtype=np.float64)
    # The drift is supplied by the model; the pool contributes shape only.
    return pool - pool.mean()


def _simulate_price_paths(
    *,
    current_price: float,
    total_return: float,
    horizon: int,
    shock_pool: np.ndarray,
    terminal_sigma: float,
    n_scenarios: int,
    seed: int,
    block_size: int = 5,
) -> np.ndarray:
    """
    Simulate `n_scenarios` price paths of length `horizon`.

    Shocks are drawn as contiguous blocks from the asset's own recent log returns
    rather than from a normal distribution, which keeps the fat tails and the
    volatility clustering that make a real price series look the way it does.
    The whole shock matrix is then rescaled so the spread of terminal outcomes
    matches `terminal_sigma`, the uncertainty the model and the market jointly
    justify.

    Returns an array of shape (n_scenarios, horizon).
    """
    rng = np.random.default_rng(seed)
    total_log_return = np.log1p(max(float(total_return), -0.999999))
    drift = np.full(horizon, total_log_return / float(horizon), dtype=np.float64)

    if shock_pool.size >= block_size * 2:
        n_blocks = int(np.ceil(horizon / block_size))
        starts = rng.integers(0, max(shock_pool.size - block_size, 1), size=(n_scenarios, n_blocks))
        offsets = np.arange(block_size)
        # (scenarios, blocks, block_size) -> flatten to (scenarios, blocks*block_size)
        indices = (starts[:, :, None] + offsets[None, None, :]) % shock_pool.size
        shocks = shock_pool[indices].reshape(n_scenarios, -1)[:, :horizon]
    else:
        # No usable history to resample: fall back to Gaussian steps. The bands stay
        # honest and still grow with sqrt(t); only the fat tails and the volatility
        # clustering are lost.
        shocks = rng.standard_normal((n_scenarios, horizon))

    # Match the simulated terminal spread to the target uncertainty.
    realised_sigma = float(np.std(shocks.sum(axis=1)))
    if realised_sigma > 1e-12:
        shocks = shocks * (terminal_sigma / realised_sigma)
    else:
        shocks = np.zeros((n_scenarios, horizon), dtype=np.float64)

    log_paths = np.log(float(current_price)) + np.cumsum(drift + shocks, axis=1)
    return np.exp(log_paths)


def _terminal_sigma(
    *,
    horizon: int,
    model_rmse_return: float,
    recent_volatility: float,
    spread_pct: float,
) -> float:
    """
    Width of the terminal forecast distribution, in log-return units.

    Three quantities have a claim on it and the widest wins: the model's own
    out-of-sample error, the diffusion the market produces on its own over the
    horizon, and the disagreement between the ensemble members. A band narrower
    than any of these would assert precision nobody has.
    """
    sigma_model = float(np.log1p(max(model_rmse_return, 0.0)))
    sigma_market = max(float(recent_volatility), 0.0) * np.sqrt(max(int(horizon), 1))
    sigma_spread = max(float(spread_pct), 0.0) / 100.0 / 2.0
    return max(sigma_model, sigma_market, sigma_spread, 1e-4)


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
    raw_df: Optional[pd.DataFrame] = None,
    seed: int = 0,
    n_scenarios: int = N_SCENARIOS,
    model_paths_override: Optional[Dict[str, np.ndarray]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict], List[List[float]]]:
    """
    Build the daily forecast timeline and the scenario paths behind it.

    The reported `predicted` series is the compounded median path, so it agrees
    exactly with the headline endpoint. The bands come from Monte Carlo
    percentiles, applied as offsets around that median so the interval can never
    invert and so the centre line carries no simulation noise.

    Returns (points, scenario_paths).
    """
    future_dates = list(pd.bdate_range(start=last_date, periods=horizon + 1)[1:])
    total_return = (float(predicted_price) - float(current_price)) / max(float(current_price), 1e-6)

    if model_paths_override:
        # Recursive mode: the centre line is the weighted blend of genuine
        # per-step model paths, so it carries the models' own step-to-step
        # shape instead of a smooth compounded curve.
        stacked = np.vstack([model_paths_override[m] for m in model_paths_override])
        w = np.array([(weights or {}).get(m, 1.0) for m in model_paths_override], dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        median_path = (stacked * w[:, None]).sum(axis=0)
    else:
        median_path = _compound_path(current_price, total_return, horizon)

    model_rmse_return = max(float(weighted_rmse), 0.0) / max(float(current_price), 1e-6)
    terminal_sigma = _terminal_sigma(
        horizon=horizon,
        model_rmse_return=model_rmse_return,
        recent_volatility=recent_volatility,
        spread_pct=spread_pct,
    )
    shock_pool = _bootstrap_shock_pool(raw_df) if raw_df is not None else np.zeros(0)
    paths = _simulate_price_paths(
        current_price=current_price,
        total_return=total_return,
        horizon=horizon,
        shock_pool=shock_pool,
        terminal_sigma=terminal_sigma,
        n_scenarios=n_scenarios,
        seed=seed,
    )

    # Two-sided 95% and 68% intervals.
    p2_5, p16, p50, p84, p97_5 = np.percentile(paths, [2.5, 16.0, 50.0, 84.0, 97.5], axis=0)

    model_paths = model_paths_override or {
        model: _compound_path(
            current_price,
            (float(price) - float(current_price)) / max(float(current_price), 1e-6),
            horizon,
        )
        for model, price in raw_predictions.items()
    }

    points: List[Dict] = []
    for i, dt in enumerate(future_dates):
        centre = float(median_path[i])
        lower_95 = max(centre + float(p2_5[i] - p50[i]), 0.01)
        upper_95 = centre + float(p97_5[i] - p50[i])
        lower_68 = max(centre + float(p16[i] - p50[i]), 0.01)
        upper_68 = centre + float(p84[i] - p50[i])
        point = {
            "date": str(dt.date()),
            "predicted": round(centre, 2),
            "lower": round(lower_95, 2),
            "upper": round(upper_95, 2),
            "lower_95": round(lower_95, 2),
            "upper_95": round(upper_95, 2),
            "lower_68": round(lower_68, 2),
            "upper_68": round(upper_68, 2),
        }
        for model, path in model_paths.items():
            point[model] = round(float(path[i]), 2)
        points.append(point)

    display_indices = np.linspace(0, n_scenarios - 1, min(N_DISPLAY_PATHS, n_scenarios), dtype=int)
    scenario_paths = [
        [round(float(current_price), 2)] + [round(float(v), 2) for v in paths[idx]]
        for idx in display_indices
    ]
    return points, scenario_paths


def _synthetic_next_bar(history: pd.DataFrame, next_date: pd.Timestamp, close: float) -> pd.DataFrame:
    """A plausible OHLCV bar for a predicted close, to feed the next feature build."""
    last_close = float(history["Close"].iloc[-1])
    rng = ((history["High"] - history["Low"]) / history["Close"]).replace(
        [np.inf, -np.inf], np.nan).dropna().tail(20)
    band = float(np.clip(rng.median() if not rng.empty else 0.015, 0.002, 0.08))
    volume = history["Volume"].replace(0, np.nan).dropna().tail(20)
    row = {
        "Open": [last_close],
        "High": [max(last_close, close) * (1 + band / 2)],
        "Low": [min(last_close, close) * (1 - band / 2)],
        "Close": [close],
        "Volume": [int(volume.median()) if not volume.empty else 0],
    }
    if "Adj Close" in history.columns:
        row["Adj Close"] = [close]
    return pd.DataFrame(row, index=pd.DatetimeIndex([pd.Timestamp(next_date)]))


def recursive_model_path(
    bundle: Dict,
    model_type: str,
    raw_df: pd.DataFrame,
    horizon: int,
    current_price: float,
    feature_config: Dict,
) -> Optional[np.ndarray]:
    """
    Roll a 1-day-return bundle forward `horizon` steps, one real inference each.

    Unlike the compounded path this calls the model `horizon` times and returns
    `horizon` genuine outputs. The cost is that step t is conditioned on t-1
    synthetic bars, so the further out it goes the more it is forecasting its own
    output rather than the market. Returns None if any step fails.
    """
    history = raw_df.copy()
    dates = list(pd.bdate_range(start=raw_df.index[-1], periods=horizon + 1)[1:])
    price = float(current_price)
    path: List[float] = []

    for step, next_date in enumerate(dates, start=1):
        try:
            frame = build_regression_feature_frame(history, feature_config=feature_config)
            inference = _run_inference(bundle, frame, model_type)
            if inference is None:
                logger.warning("Recursive step %d/%d returned no inference for %s", step, horizon, model_type)
                return None
            step_return, _ = inference
            price = price * (1.0 + float(step_return))
            path.append(price)
            history = pd.concat([history, _synthetic_next_bar(history, next_date, price)])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recursive step %d/%d failed for %s: %s", step, horizon, model_type, exc)
            return None

    logger.info(
        "RECURSIVE path for %s: %d model calls -> %d genuine per-step outputs "
        "(first=%.4f last=%.4f)", model_type, horizon, len(path), path[0], path[-1],
    )
    return np.asarray(path, dtype=np.float64)


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
        model_types: Optional[List[str]] = None,
        current_price: Optional[float] = None,
    ) -> Optional[EnsembleForecast]:
        symbol = symbol.upper()
        horizon = int(horizon)
        if horizon not in SUPPORTED_HORIZONS:
            raise ValueError(f"Unsupported horizon {horizon}; supported horizons are {SUPPORTED_HORIZONS}")

        # 1. Load all available bundles
        bundles: Dict[str, Dict] = {}
        requested_model_types = model_types or MODEL_TYPES
        requested_model_types = [mtype for mtype in requested_model_types if mtype in MODEL_TYPES]
        for mtype in requested_model_types:
            b = _load_regression_bundle(symbol, mtype, horizon)
            if b is not None:
                bundles[mtype] = b

        if not bundles:
            logger.info("No regression bundles found for %s h=%d", symbol, horizon)
            return None

        current_price = float(current_price if current_price is not None else raw_df["Close"].iloc[-1])
        last_date = raw_df.index[-1]

        # 2. Build feature frame (shared across all models)
        first_meta = next(iter(bundles.values()))["meta"]
        feature_config = normalize_feature_config(first_meta.get("feature_config"))
        feature_frame = build_regression_feature_frame(raw_df, feature_config=feature_config)

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

        # Diagnostic: state plainly how many numbers actually came out of the
        # models versus how many points the chart will draw. One output per
        # model is correct for a cumulative-return target — it is the chart that
        # must not pretend otherwise.
        logger.info(
            "RAW MODEL OUTPUTS for %s h=%dd: %d model call(s), one cumulative "
            "%dd return each -> %s | ensemble path will contain %d points, "
            "%d of which are model outputs and %d interpolated",
            symbol, horizon, len(model_returns), horizon,
            {m: round(r, 6) for m, r in model_returns.items()},
            horizon, 1, horizon - 1,
        )

        # 4. Fixed ensemble weights from the spec
        active_bundles = {m: bundles[m] for m in raw_predictions}
        weights = _compute_weights(active_bundles, current_price)

        # 4b. Resolve the path mode. Recursive is gated on the trained target,
        # so an unsupported request degrades to the compounded path with a
        # warning rather than silently producing a meaningless daily series.
        resolved_path_type = "compounded_interpolation"
        recursive_paths: Dict[str, np.ndarray] = {}
        mode = forecast_mode()
        # In auto mode a missing step bundle is the expected path, not a
        # misconfiguration, so it is logged at info. An explicit recursive request
        # that cannot be honoured stays a warning.
        fallback_log = logger.warning if mode == FORECAST_MODE_RECURSIVE else logger.info
        if mode in (FORECAST_MODE_RECURSIVE, FORECAST_MODE_AUTO):
            step_bundles = load_step_bundles(symbol, list(active_bundles))
            if not step_bundles:
                fallback_log(
                    "%s=%s for %s h=%dd but no servable horizon-%d step "
                    "bundles exist. Train them first. Falling back to the compounded path.",
                    FORECAST_MODE_ENV, mode, symbol, horizon,
                    RECURSIVE_STEP_HORIZON,
                )
            else:
                supported, why_not = recursive_path_supported(step_bundles)
                if not supported:
                    fallback_log(
                        "%s=%s for %s h=%dd but %s Falling back to the compounded path.",
                        FORECAST_MODE_ENV, mode, symbol, horizon, why_not,
                    )
                else:
                    step_config = normalize_feature_config(
                        next(iter(step_bundles.values()))["meta"].get("feature_config")
                    )
                    for mtype, bundle in step_bundles.items():
                        path = recursive_model_path(
                            bundle, mtype, raw_df, horizon, current_price, step_config
                        )
                        if path is not None:
                            recursive_paths[mtype] = path
                    if recursive_paths:
                        resolved_path_type = "recursive_per_step"
                        # The endpoint now comes from the rolled-forward path itself,
                        # so the headline agrees with the line the chart draws.
                        for mtype, path in recursive_paths.items():
                            raw_predictions[mtype] = float(path[-1])
                            model_returns[mtype] = float(path[-1]) / max(current_price, 1e-6) - 1.0
                        active_bundles = {m: step_bundles[m] for m in recursive_paths}
                        weights = _compute_weights(active_bundles, current_price)
                    else:
                        fallback_log(
                            "Recursive mode produced no usable path for %s h=%dd; "
                            "falling back to the compounded path.", symbol, horizon,
                        )

        # Final composition of the series the client will receive.
        if resolved_path_type == "recursive_per_step":
            logger.info(
                "FORECAST PATH for %s h=%dd: recursive_per_step — %d points, all %d "
                "from model inference (%d calls per model across %d model(s))",
                symbol, horizon, horizon, horizon, horizon, len(recursive_paths),
            )
        else:
            logger.info(
                "FORECAST PATH for %s h=%dd: compounded_interpolation — %d points from "
                "1 model output per model; %d points are interpolated, not predicted",
                symbol, horizon, horizon, horizon - 1,
            )

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

        # 9. Forecast timeline. The seed is derived from the symbol and horizon so a
        # chart is reproducible across refreshes but different names differ.
        forecast_points, scenario_paths = _build_forecast_points(
            ensemble_pred,
            current_price,
            horizon,
            last_date,
            avg_mape,
            weighted_rmse,
            spread_pct,
            recent_vol,
            raw_predictions,
            raw_df=raw_df,
            seed=abs(hash((symbol, horizon))) % (2**32),
            model_paths_override=recursive_paths or None,
            weights=weights,
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
            scenario_paths=scenario_paths,
            path_type=resolved_path_type,
            per_step_predictions=resolved_path_type == "recursive_per_step",
            model_output_count=(
                len(model_returns) * horizon
                if resolved_path_type == "recursive_per_step"
                else len(model_returns)
            ),
            points_per_model_output=(
                1.0 if resolved_path_type == "recursive_per_step" else float(horizon)
            ),
        )


def regression_bundle_status(symbol: str, model_type: str, horizon: int) -> Tuple[bool, Optional[str]]:
    """
    Report whether one bundle is servable, and why not if it isn't.

    The reason is worded to be shown to a user, since it tells them what to do.
    """
    canonical = _bundle_dir(symbol, model_type, horizon) / "metadata.json"
    legacy = _legacy_price_regression_bundle_dir(symbol, model_type, horizon) / "metadata.json"
    meta_path = canonical if canonical.exists() else legacy if legacy.exists() else None
    if meta_path is None:
        return False, f"no {model_type} bundle is trained for {symbol.upper()} at horizon {horizon}"
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as exc:
        return False, f"the {model_type} bundle metadata could not be read ({exc})"
    if not _metadata_is_return_regression(meta):
        return False, f"the {model_type} bundle predicts prices rather than returns and must be retrained"
    skill_failure = bundle_skill_failure(meta)
    if skill_failure:
        return False, f"the {model_type} bundle {skill_failure}"
    return True, None


def ensemble_availability(symbol: str, horizon: int) -> Tuple[List[str], Dict[str, str]]:
    """
    Split the ensemble members into those that can be served and those that cannot.

    Returns (servable_model_types, {model_type: reason_it_cannot_be_served}).
    """
    servable: List[str] = []
    blocked: Dict[str, str] = {}
    for mtype in MODEL_TYPES:
        available, reason = regression_bundle_status(symbol, mtype, horizon)
        if available:
            servable.append(mtype)
        else:
            blocked[mtype] = reason or "unavailable"
    return servable, blocked


def ensemble_bundle_status(symbol: str, horizon: int) -> Tuple[bool, Optional[str]]:
    """
    Report whether an ensemble forecast can be served, and why not if it cannot.

    A three-member ensemble that refuses to answer because one member is
    unservable is the wrong trade: the remaining models still carry a forecast,
    and the honest response is that forecast plus a note about what is missing.
    Requiring all three made every horizon for a symbol go dark whenever a single
    bundle failed the skill gate, which is what put "Prediction model unavailable"
    on a tab whose other two models were ready to serve.

    So availability means "at least one member is servable". The reason string is
    still populated when members are missing, so callers can degrade reliability
    and tell the user which models are absent.
    """
    servable, blocked = ensemble_availability(symbol, horizon)
    if not servable:
        # Nothing to serve — report the first reason, which is the actionable one.
        first = next(iter(blocked.values()), None) if blocked else None
        return False, first or f"no bundles are trained for {symbol.upper()} at horizon {horizon}"
    if blocked:
        detail = "; ".join(f"{m} excluded because {why}" for m, why in blocked.items())
        return True, f"partial ensemble ({len(servable)} of {len(MODEL_TYPES)} models): {detail}"
    return True, None


def ensemble_bundles_available(symbol: str, horizon: int) -> bool:
    """Return True when the complete three-model regression ensemble is servable."""
    available, _ = ensemble_bundle_status(symbol, horizon)
    return available
