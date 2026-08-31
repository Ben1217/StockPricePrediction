"""
Next-day direction API.

Serves the artifacts ``scripts/direction_backtest.py`` writes, plus one live
number: P(up) for the session after the last printed close.

The design rule this router exists to enforce is that **the gauge is never
served alone**. A probability of 0.58 is a number anyone can render as a dial;
whether it is worth acting on is decided by the walk-forward verdict sitting
next to it. So every response carries the evaluation, and ``next_session.
tradeable`` is false whenever the model failed its ship criteria, with the
reason spelled out. This mirrors how the regression bundles are gated on proven
skill (see ``bundle_skill_failure`` in :mod:`src.models.ensemble_predictor`):
a model that cannot beat a constant classifier does not get to look confident.

Evaluation results are read from disk rather than computed per request. A
walk-forward run is dozens of model fits over a decade of bars; it belongs in a
scheduled job or a terminal, not inside an HTTP handler. When no report exists
the response says so and names the command that produces one.

Routes:
    GET /api/direction/            list available reports
    GET /api/direction/{symbol}    evaluation, gated gauge, curves
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.data.direction_data import load_daily_bars
from src.features.direction_features import build_direction_dataset
from src.models.direction_models import MODEL_FACTORIES
from src.models.direction_pipeline import predict_next_session
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

REPORT_DIR = Path("data/direction_backtests")

# Window for the rolling hit-rate strip. 60 sessions is about a quarter: long
# enough that the number is not pure noise, short enough to show a model
# decaying rather than averaging that away over the whole test period.
ROLLING_HIT_RATE_WINDOW = 60

# History pulled for the live gauge. Shorter than an evaluation run needs -
# this only has to cover the feature warm-up plus enough rows to fit on.
GAUGE_LOOKBACK_DAYS = 365 * 6


def _stem(symbol: str, model: str) -> str:
    return f"{symbol.upper().replace('^', '')}_{model}"


def _report_path(symbol: str, model: str) -> Path:
    return REPORT_DIR / f"{_stem(symbol, model)}_report.json"


def _load_report(symbol: str, model: str) -> Optional[Dict[str, Any]]:
    path = _report_path(symbol, model)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read direction report %s: %s", path, exc)
        return None


def _load_predictions(symbol: str, model: str) -> Optional[pd.DataFrame]:
    path = REPORT_DIR / f"{_stem(symbol, model)}_predictions.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read direction predictions %s: %s", path, exc)
        return None


def _load_equity_curve(symbol: str, model: str) -> Optional[pd.DataFrame]:
    path = REPORT_DIR / f"{_stem(symbol, model)}_equity_curve.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except (OSError, ValueError) as exc:
        logger.warning("Could not read direction equity curve %s: %s", path, exc)
        return None


def _rolling_hit_rate(predictions: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Rolling accuracy of the hard label over ``ROLLING_HIT_RATE_WINDOW`` sessions.

    Emitted alongside the rolling base rate, because a hit-rate strip without
    the base rate beside it is unreadable: 54% looks like skill until you see
    the market went up 55% of the time in that window.
    """
    if predictions is None or predictions.empty:
        return []
    if not {"prediction", "label"}.issubset(predictions.columns):
        return []

    correct = (predictions["prediction"] == predictions["label"]).astype(float)
    window = min(ROLLING_HIT_RATE_WINDOW, len(predictions))
    hit_rate = correct.rolling(window, min_periods=window).mean()
    base_rate = predictions["label"].astype(float).rolling(window, min_periods=window).mean()

    points: List[Dict[str, Any]] = []
    for timestamp, value in hit_rate.dropna().items():
        points.append({
            "date": str(pd.Timestamp(timestamp).date()),
            "hit_rate": round(float(value), 6),
            "base_rate": round(float(base_rate.loc[timestamp]), 6),
            "window": int(window),
        })
    return points


def _equity_points(curve: pd.DataFrame) -> List[Dict[str, Any]]:
    if curve is None or curve.empty:
        return []
    columns = {"equity", "benchmark_equity"}
    if not columns.issubset(curve.columns):
        return []
    points: List[Dict[str, Any]] = []
    for timestamp, row in curve.iterrows():
        points.append({
            "date": str(pd.Timestamp(timestamp).date()),
            "strategy": round(float(row["equity"]), 6),
            "benchmark": round(float(row["benchmark_equity"]), 6),
            "position": int(row["position"]) if "position" in curve.columns else None,
            "drawdown": round(float(row["drawdown"]), 6) if "drawdown" in curve.columns else None,
        })
    return points


def _summarise_evaluation(report: Dict[str, Any]) -> Dict[str, Any]:
    pooled = report["pooled"]
    model_metrics = pooled["model"]
    leakage = report.get("leakage_check") or {}
    return {
        "generated_at": report.get("generated_at"),
        "n_test_days": pooled["n_test_rows"],
        "test_range": pooled["test_range"],
        "n_folds": report["config"]["n_folds_run"],
        "accuracy": model_metrics["accuracy"],
        "accuracy_ci": [model_metrics["accuracy_ci_low"], model_metrics["accuracy_ci_high"]],
        "balanced_accuracy": model_metrics["balanced_accuracy"],
        "test_base_rate": model_metrics["base_rate"],
        "roc_auc": model_metrics["roc_auc"],
        "brier_score": model_metrics["brier_score"],
        "log_loss": model_metrics["log_loss"],
        "mcc": model_metrics["mcc"],
        "brier_skill_score": model_metrics["skill"]["brier_skill_score"],
        "calibration": model_metrics["calibration"],
        "baselines": {
            name: {
                "accuracy": metrics["accuracy"],
                "accuracy_ci": [metrics["accuracy_ci_low"], metrics["accuracy_ci_high"]],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "mcc": metrics["mcc"],
            }
            for name, metrics in pooled["baselines"].items()
        },
        "best_baseline": pooled["best_baseline"],
        "edge_vs_best_baseline": pooled["edge_vs_best_baseline"],
        "leakage_check_passed": leakage.get("passed"),
        "leakage_check_note": leakage.get("note"),
        # The out-of-sample calibration of the price range. A band is only worth
        # rendering next to its coverage: one claiming 90% that covers 60% is
        # decoration, and the client needs the number to say so.
        "price_band": pooled.get("price_band"),
        "n_features": report["config"].get("n_features"),
        "model_options": report["config"].get("model_kwargs") or {},
    }


def _gate_reason(report: Dict[str, Any]) -> Optional[str]:
    """
    Why the live probability must not be presented as actionable, or None.

    Reads the verdict the walk-forward run already computed rather than
    re-deriving a rule here, so the gauge and the report can never disagree.
    """
    verdict = report.get("verdict") or {}
    if verdict.get("ship"):
        return None

    failed = verdict.get("failed_criteria") or []
    readable = {
        "beats_best_baseline_accuracy": "it does not beat the best naive baseline",
        "accuracy_edge_is_significant": "its accuracy edge is inside the noise",
        "positive_probability_skill": "its probabilities score worse than a constant base rate",
        "beats_buy_and_hold_after_costs": "it loses to buy and hold after costs",
        "survives_the_charged_cost": "its edge dies below the cost charged",
        "passes_leakage_check": "it failed the shuffled-label leakage check",
    }
    reasons = [readable.get(name, name) for name in failed]
    if not reasons:
        return "the walk-forward run did not clear its ship criteria"
    return "Not tradeable: " + "; ".join(reasons) + "."


@router.get("/")
def list_direction_reports() -> Dict[str, Any]:
    """Every symbol/model pair that has a stored walk-forward report."""
    if not REPORT_DIR.exists():
        return {"reports": [], "report_dir": str(REPORT_DIR)}

    reports: List[Dict[str, Any]] = []
    for path in sorted(REPORT_DIR.glob("*_report.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        reports.append({
            "symbol": payload.get("data", {}).get("ticker"),
            "model": payload.get("config", {}).get("model"),
            "generated_at": payload.get("generated_at"),
            "accuracy": payload.get("pooled", {}).get("model", {}).get("accuracy"),
            "ship": payload.get("verdict", {}).get("ship"),
        })
    return {"reports": reports, "report_dir": str(REPORT_DIR)}


@router.get("/{symbol}")
def get_direction(
    symbol: str,
    model: str = Query("logistic", enum=sorted(MODEL_FACTORIES)),
    include_gauge: bool = Query(True, description="Fit on all history and predict the next session"),
) -> Dict[str, Any]:
    """
    Evaluation, gated live probability, rolling hit rate, and equity curve.

    A 404 here means no walk-forward report has been produced for this
    symbol/model. That is deliberate: the alternative is serving a probability
    with nothing to say whether it is worth anything, which is the failure mode
    the whole pipeline was built to remove.
    """
    # `enum=` on a plain str Query is an OpenAPI hint, not a validator, so an
    # unknown model would otherwise fall through to the 404 below and tell the
    # user to run a command with a model name that does not exist.
    if model not in MODEL_FACTORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Available: {sorted(MODEL_FACTORIES)}",
        )

    symbol = symbol.upper().strip()
    report = _load_report(symbol, model)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No direction report for {symbol} ({model}). Generate one with: "
                f"python scripts/direction_backtest.py --ticker {symbol} --model {model}"
            ),
        )

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "model": model,
        "horizon_days": report["config"]["horizon"],
        "data": {
            key: report.get("data", {}).get(key)
            for key in ("first_bar", "last_bar", "clean_rows", "price_basis", "content_sha256")
        },
        "evaluation": _summarise_evaluation(report),
        "verdict": report.get("verdict"),
        "backtest": report["pooled"]["backtest"],
        "rolling_hit_rate": _rolling_hit_rate(_load_predictions(symbol, model)),
        "equity_curve": _equity_points(_load_equity_curve(symbol, model)),
        "next_session": None,
    }

    if not include_gauge:
        return payload

    gate = _gate_reason(report)
    try:
        bars = load_daily_bars(
            symbol,
            start=(pd.Timestamp.today().normalize() - pd.Timedelta(days=GAUGE_LOOKBACK_DAYS)),
            require_min_rows=False,
        )
        dataset = build_direction_dataset(bars.frame)
        prediction = predict_next_session(bars.frame, model_name=model, dataset=dataset)
    except Exception as exc:  # noqa: BLE001 - a stale gauge must not 500 the report
        logger.warning("Live direction gauge failed for %s: %s", symbol, exc)
        payload["next_session"] = {"available": False, "error": str(exc)}
        return payload

    if prediction is None:
        payload["next_session"] = {
            "available": False,
            "error": "No bar has a complete feature vector",
        }
        return payload

    prediction.update({
        "available": True,
        "tradeable": gate is None,
        "gate_reason": gate,
        # Restated next to the number so a client cannot render the gauge
        # without the caveat travelling with it.
        "caveat": (
            "Out-of-sample accuracy "
            f"{payload['evaluation']['accuracy']:.1%} "
            f"(95% CI {payload['evaluation']['accuracy_ci'][0]:.1%}"
            f"-{payload['evaluation']['accuracy_ci'][1]:.1%}) over "
            f"{payload['evaluation']['n_test_days']} test days."
        ),
    })
    payload["next_session"] = prediction
    return payload
