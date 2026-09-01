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
walk-forward run is dozens of model fits over a decade of bars; it belongs on a
background worker, not inside an HTTP handler. But "not in this request" is not
the same as "not our problem": when no report exists, the route starts one
through :mod:`src.models.preparation` and answers ``status: "preparing"`` with
the job to poll. The gauge still never ships without its evaluation — it is just
that the evaluation now gets produced rather than demanded of the user.

``/{symbol}/analysis`` answers a different question from ``/{symbol}``. The
latter serves one model's gated probability and its track record. The former
serves the *reasoned* direction call: seven named categories of evidence — trend,
momentum, volume, price action, support/resistance, volatility regime and what
followed the most similar setups in this symbol's own past — each with the
percentage points it contributed, blended with the classifier by measured skill.
It deliberately does not require a stored walk-forward report: the evidence stack
evaluates itself, so a symbol nobody has trained yet still gets an honest answer,
and the classifier joins the blend once its report exists.

Routes:
    GET /api/direction/                     list available reports
    GET /api/direction/{symbol}             evaluation, gated gauge, curves
    GET /api/direction/{symbol}/analysis    direction, probability, confidence, evidence
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.data.direction_data import load_daily_bars
from src.features.direction_features import build_direction_dataset
from src.models.direction_evidence import analyse_direction
from src.models.direction_models import MODEL_FACTORIES
from src.models.direction_pipeline import DEFAULT_REPORT_DIR, predict_next_session, report_stem
from src.models.preparation import preparation_state
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()

REPORT_DIR = DEFAULT_REPORT_DIR

# Window for the rolling hit-rate strip. 60 sessions is about a quarter: long
# enough that the number is not pure noise, short enough to show a model
# decaying rather than averaging that away over the whole test period.
ROLLING_HIT_RATE_WINDOW = 60

# History pulled for the live gauge. Shorter than an evaluation run needs -
# this only has to cover the feature warm-up plus enough rows to fit on.
GAUGE_LOOKBACK_DAYS = 365 * 6

# History for the evidence analysis. Longer than the gauge needs, because the
# nearest-neighbour read is only as good as the number of past setups it has to
# choose from, and the stack's own walk-forward wants test folds on top of a
# training window.
ANALYSIS_LOOKBACK_DAYS = 365 * 10

# The analysis is roughly a second of feature building and model fitting on a
# decade of bars. Daily bars change once a day, so the result is cached against
# the last printed bar: a dashboard switching between tickers re-reads rather
# than recomputes, and a new session's close invalidates the entry by changing
# the key. Bounded so a long-lived process cannot accumulate every symbol a user
# has ever typed.
ANALYSIS_CACHE_TTL_SECONDS = 15 * 60
ANALYSIS_CACHE_MAX_ENTRIES = 64
_analysis_cache: "Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]]" = {}
_analysis_cache_lock = threading.Lock()


# The naming is the writer's, imported rather than restated. A reader with its
# own copy of the rule is a run that silently never happened the day one of them
# changes — and ^GSPC, whose caret is stripped, is exactly where they would.
def _stem(symbol: str, model: str) -> str:
    return report_stem(symbol, model)


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


def _no_evaluation_message(symbol: str, model: str, preparation: Optional[Dict[str, Any]]) -> str:
    """
    Why there is still no evaluation, when one is not being produced right now.

    Three different situations, and reporting the wrong one is how a broken
    pipeline gets mistaken for a disabled setting: the last run failed, a run
    finished without leaving a report, or nothing ever started.
    """
    status = (preparation or {}).get("status")

    if status == "failed":
        return (
            f"The walk-forward evaluation for {symbol} ({model}) failed: "
            f"{preparation.get('error') or 'unknown error'}."
        )

    if status == "completed":
        warnings = preparation.get("warnings") or []
        suffix = f" ({warnings[0]})" if warnings else ""
        return (
            f"Preparation for {symbol} finished without producing a walk-forward "
            f"evaluation for {model}{suffix}."
        )

    return (
        f"No walk-forward evaluation exists for {symbol} ({model}) and automatic "
        f"preparation did not start. Request one with POST /api/models/{symbol}/prepare."
    )


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

    When no report exists the response is a 200 with ``status: "preparing"`` and
    no ``evaluation`` — the walk-forward run has been started, and the client
    polls ``preparation.job_id`` until it lands. What has *not* changed is the
    rule underneath: no gauge is served here without the evaluation beside it.
    Only the remedy has, from telling a user to run a command to running it.
    """
    # `enum=` on a plain str Query is an OpenAPI hint, not a validator, so an
    # unknown model would otherwise fall through and start a preparation run for
    # an estimator that does not exist.
    if model not in MODEL_FACTORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Available: {sorted(MODEL_FACTORIES)}",
        )

    symbol = symbol.upper().strip()
    report = _load_report(symbol, model)
    if report is None:
        preparation = preparation_state(symbol, direction_model=model)
        preparing = bool(preparation and preparation.get("status") in ("queued", "running"))
        return {
            "symbol": symbol,
            "model": model,
            "status": "preparing" if preparing else "unavailable",
            "evaluation": None,
            "verdict": None,
            "backtest": None,
            "rolling_hit_rate": [],
            "equity_curve": [],
            "next_session": None,
            "preparation": preparation,
            "message": (
                f"Running the walk-forward evaluation for {symbol} ({model}). "
                f"The gauge appears once it has an out-of-sample record to stand on."
                if preparing
                else _no_evaluation_message(symbol, model, preparation)
            ),
        }

    payload: Dict[str, Any] = {
        "symbol": symbol,
        "model": model,
        "status": "ok",
        "preparation": None,
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


def _cached_analysis(key: "Tuple[str, str]") -> Optional[Dict[str, Any]]:
    with _analysis_cache_lock:
        entry = _analysis_cache.get(key)
        if entry is None:
            return None
        expires_at, payload = entry
        if expires_at < time.monotonic():
            _analysis_cache.pop(key, None)
            return None
        return payload


def _store_analysis(key: "Tuple[str, str]", payload: Dict[str, Any]) -> None:
    with _analysis_cache_lock:
        if len(_analysis_cache) >= ANALYSIS_CACHE_MAX_ENTRIES:
            # Drop whatever expires soonest rather than an arbitrary entry, so
            # the symbols a user is actively switching between survive.
            oldest = min(_analysis_cache, key=lambda name: _analysis_cache[name][0])
            _analysis_cache.pop(oldest, None)
        _analysis_cache[key] = (time.monotonic() + ANALYSIS_CACHE_TTL_SECONDS, payload)


def clear_analysis_cache() -> None:
    """Drop every cached analysis. Called by tests; harmless in production."""
    with _analysis_cache_lock:
        _analysis_cache.clear()


def _classifier_contribution(
    symbol: str,
    model: str,
    bars: pd.DataFrame,
) -> Dict[str, Any]:
    """
    The stored classifier's live probability and its *measured* skill, or why not.

    The skill number is the whole point of fetching this. A probability with no
    out-of-sample record beside it cannot be given a blend weight, so a symbol
    with no walk-forward report contributes nothing here rather than
    contributing an unmeasured opinion — and the analysis says so instead of
    quietly leaving the classifier out.
    """
    report = _load_report(symbol, model)
    if report is None:
        return {
            "included": False,
            "reason": (
                f"no walk-forward report for {symbol} ({model}), so the classifier has no "
                "measured skill to be weighted by"
            ),
        }

    skill = (
        report.get("pooled", {})
        .get("model", {})
        .get("skill", {})
        .get("brier_skill_score")
    )
    if skill is None:
        return {"included": False, "reason": "the stored report carries no Brier skill score"}

    gate = _gate_reason(report)
    try:
        dataset = build_direction_dataset(bars)
        prediction = predict_next_session(bars, model_name=model, dataset=dataset)
    except Exception as exc:  # noqa: BLE001 - the evidence answer must survive a dead classifier
        logger.warning("Classifier leg of the direction analysis failed for %s: %s", symbol, exc)
        return {"included": False, "reason": f"the classifier could not be fitted: {exc}"}

    if prediction is None:
        return {"included": False, "reason": "no bar has a complete feature vector for the classifier"}

    return {
        "included": True,
        "probability_up": float(prediction["probability_up"]),
        "skill": float(skill),
        "tradeable": gate is None,
        "gate_reason": gate,
        "accuracy": report.get("pooled", {}).get("model", {}).get("accuracy"),
        "price_forecast": prediction.get("price_forecast"),
    }


@router.get("/{symbol}/analysis")
def get_direction_analysis(
    symbol: str,
    model: str = Query("logistic", enum=sorted(MODEL_FACTORIES)),
    refresh: bool = Query(False, description="Recompute instead of serving the cached analysis"),
) -> Dict[str, Any]:
    """
    Direction, probability, confidence and the evidence that produced them.

    Combines seven named categories of evidence read off this symbol's own bars
    with the stored classifier, weighting the two by their measured
    out-of-sample skill. Returns NEUTRAL — with the reason — whenever no source
    has demonstrated an edge over the base rate, which is the answer a daily
    direction model owes far more often than a dashboard usually admits.

    Unlike ``/{symbol}``, this route does not require a stored walk-forward
    report. The evidence stack runs its own walk-forward, so the answer stands
    on a measured record either way; the classifier simply joins the blend once
    its report exists.
    """
    if model not in MODEL_FACTORIES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Available: {sorted(MODEL_FACTORIES)}",
        )

    symbol = symbol.upper().strip()
    try:
        bars = load_daily_bars(
            symbol,
            start=(pd.Timestamp.today().normalize() - pd.Timedelta(days=ANALYSIS_LOOKBACK_DAYS)),
            require_min_rows=False,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    last_bar = str(pd.Timestamp(bars.frame.index[-1]).date()) if len(bars.frame) else "none"
    cache_key = (f"{symbol}:{model}", last_bar)
    if not refresh:
        cached = _cached_analysis(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

    classifier = _classifier_contribution(symbol, model, bars.frame)
    try:
        analysis = analyse_direction(
            bars.frame,
            symbol=symbol,
            model_name=model if classifier.get("included") else None,
            model_probability=classifier.get("probability_up"),
            model_skill=classifier.get("skill"),
            model_tradeable=classifier.get("tradeable"),
            model_gate_reason=classifier.get("gate_reason"),
        )
    except ValueError as exc:
        # Too little history for a structure read, a nearest-neighbour sample or
        # a walk-forward. That is a real answer, not a failure to render.
        return {
            "symbol": symbol,
            "model": model,
            "status": "unavailable",
            "message": str(exc),
            "data": {key: bars.meta.get(key) for key in ("first_bar", "last_bar", "clean_rows", "price_basis")},
            # Read-only. Starting a training run for a symbol that does not have
            # enough history to analyse would queue work already known to fail.
            "preparation": preparation_state(symbol, direction_model=model, auto_start=False),
        }

    payload: Dict[str, Any] = {
        **analysis,
        "model": model,
        "status": "ok",
        "cached": False,
        "classifier_note": classifier.get("reason"),
        "data": {
            key: bars.meta.get(key)
            for key in ("first_bar", "last_bar", "clean_rows", "price_basis", "content_sha256")
        },
        # The classifier's own calibrated band, when it has one. It is reported
        # beside the analog range rather than instead of it: they are answers to
        # different questions, and a client that shows both can see when they
        # disagree.
        "model_price_forecast": classifier.get("price_forecast"),
        # Preparation is started only when it would actually change this answer -
        # that is, when the classifier is missing its walk-forward report and so
        # contributes nothing to the blend. On the path where the classifier is
        # already in, this is a read of whatever job happens to exist.
        "preparation": preparation_state(
            symbol,
            direction_model=model,
            auto_start=not classifier.get("included", False),
        ),
    }
    _store_analysis(cache_key, payload)
    return payload
