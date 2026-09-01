"""
Automatic model preparation: training a symbol into a servable state.

Selecting a ticker is the whole user action. Everything between that and a
rendered forecast — fetching bars, training the bundles that are missing,
running the walk-forward evaluation, confirming the result serves — belongs
here, behind one call.

**Preparation never runs inside a request.** A walk-forward evaluation is dozens
of model fits over a decade of bars and the price bundles are fifteen more fits;
doing that inline would hold an HTTP connection open for minutes and hand the
browser a timeout instead of a forecast. So a serving route calls
:func:`ensure_prepared`, which either says "nothing to do" or hands back a job
the client can poll while the work happens on a background thread.

Four guards keep an auto-triggered pipeline from becoming a denial of service
against its own server:

* **Deduplication.** One job per symbol. Every extra request while it runs joins
  the running job rather than starting a second one, so tab switches and
  re-renders are free.
* **A bounded pool.** ``QUANTVISION_PREPARE_WORKERS`` threads (default 1) run
  jobs; the rest queue. Training is CPU-bound, and a pool wide enough to starve
  the inference path would make every *other* symbol on the dashboard slow.
* **A cooldown.** After a job finishes, the symbol is not re-prepared for
  ``QUANTVISION_PREPARE_COOLDOWN_SECONDS``. This is what stops the loop when
  preparation cannot fix what it found — a model type that errors every run, or
  bundles that train fine and then fail their skill gate. Without it, every page
  load would restart the same doomed training run.
* **A kill switch.** ``QUANTVISION_AUTO_PREPARE=false`` disables automatic
  starts entirely; explicit requests still work.

What preparation will *not* do is retrain a bundle that trained successfully and
then failed its out-of-sample skill gate. That is a measurement, not a gap, and
refitting the same bars reproduces it. See :mod:`src.models.model_manager`.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from cachetools import TTLCache

from src.models.model_manager import (
    COMPONENT_DIRECTION,
    COMPONENT_PRICE,
    COMPONENT_UNIFIED,
    DEFAULT_DIRECTION_MODEL,
    PREPARATION_HORIZONS,
    ReadinessReport,
    assess_symbol,
    auto_prepare_enabled,
)
from src.utils.config_loader import get_env_int
from src.utils.logger import get_logger

logger = get_logger(__name__)

PREPARE_WORKERS_ENV = "QUANTVISION_PREPARE_WORKERS"
PREPARE_COOLDOWN_ENV = "QUANTVISION_PREPARE_COOLDOWN_SECONDS"
DEFAULT_PREPARE_WORKERS = 1
DEFAULT_PREPARE_COOLDOWN_SECONDS = 900  # 15 minutes

MAX_TRACKED_JOBS = 200
JOB_TTL_SECONDS = 24 * 3600

# Job states
QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"

# Stage states
PENDING = "pending"
DONE = "done"
SKIPPED = "skipped"

#: The pipeline, in execution order, with each stage's share of overall progress.
#:
#: Two things about this order are deliberate and were chosen from measurements,
#: not symmetry. The walk-forward evaluation runs **before** the price bundles
#: even though the forecast is the bigger feature: on a real symbol it completes
#: in about seven seconds against tens of minutes for the bundles, and it feeds
#: the panel at the top of the tab. Running it last would leave the most valuable
#: and cheapest artifact hostage to the slowest one.
#:
#: The weights follow the same measurements. A bar that spent most of its travel
#: on the four cheap stages would be a lie about the wait.
STAGES: Sequence[Dict[str, Any]] = (
    {"name": "market_data", "label": "Fetching market data", "weight": 0.03},
    {"name": "direction_model", "label": "Evaluating next-day direction", "weight": 0.07},
    {"name": "price_models", "label": "Training prediction models", "weight": 0.85},
    {"name": "predictions", "label": "Generating predictions", "weight": 0.03},
    {"name": "analysis", "label": "Preparing analysis", "weight": 0.02},
)

#: Cheapest first. Gradient-boosted trees and forests fit in seconds; the LSTM
#: takes minutes per bundle because it is tuned over a walk-forward grid. Since
#: the ensemble serves any subset of its members, training in this order means a
#: forecast appears after the trees rather than after everything.
MODEL_TRAINING_ORDER = ("xgboost", "random_forest", "lstm")


def worker_count() -> int:
    return max(1, get_env_int(PREPARE_WORKERS_ENV, DEFAULT_PREPARE_WORKERS))


def cooldown_seconds() -> int:
    return max(0, get_env_int(PREPARE_COOLDOWN_ENV, DEFAULT_PREPARE_COOLDOWN_SECONDS))


@dataclass
class Stage:
    name: str
    label: str
    weight: float
    state: str = PENDING
    detail: Optional[str] = None
    progress: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "state": self.state,
            "detail": self.detail,
            "progress": round(self.progress, 4),
        }


@dataclass
class PreparationJob:
    """One symbol's trip from "nothing on disk" to "servable", observable while it runs."""

    job_id: str
    symbol: str
    horizons: List[int]
    direction_model: str
    unified_models: List[str] = field(default_factory=list)
    status: str = QUEUED
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    stages: List[Stage] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    trained: List[str] = field(default_factory=list)
    readiness: Optional[Dict[str, Any]] = None

    def stage(self, name: str) -> Stage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    @property
    def progress(self) -> float:
        total = sum(stage.weight for stage in self.stages) or 1.0
        done = sum(
            stage.weight * (1.0 if stage.state in (DONE, SKIPPED) else stage.progress)
            for stage in self.stages
        )
        return min(1.0, done / total)

    @property
    def active(self) -> bool:
        return self.status in (QUEUED, RUNNING)

    def as_dict(self) -> Dict[str, Any]:
        """
        A snapshot for the wire, read from a request thread while a worker writes.

        Deliberately unlocked. Every field here is a single attribute read, which
        the GIL makes atomic, so the worst outcome is a snapshot whose stages come
        from two instants a millisecond apart. Taking a lock to make a progress
        bar strictly consistent would put the training thread and every polling
        request in contention for no benefit anyone can see.
        """
        return {
            "job_id": self.job_id,
            "symbol": self.symbol,
            "status": self.status,
            "progress": round(self.progress, 4),
            "stages": [stage.as_dict() for stage in self.stages],
            "error": self.error,
            "warnings": list(self.warnings),
            "trained": list(self.trained),
            "horizons": list(self.horizons),
            "direction_model": self.direction_model,
            "unified_models": list(self.unified_models),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "readiness": self.readiness,
        }


def _new_stages() -> List[Stage]:
    return [Stage(name=s["name"], label=s["label"], weight=s["weight"]) for s in STAGES]


# ---------------------------------------------------------------------------
# The work itself
# ---------------------------------------------------------------------------

def _price_training_order(model_type: str) -> int:
    try:
        return MODEL_TRAINING_ORDER.index(model_type)
    except ValueError:
        return len(MODEL_TRAINING_ORDER)


def _plan(report: ReadinessReport) -> Dict[str, Any]:
    """Split what needs training by the pipeline each artifact is trained through."""
    # Sorted cheapest-model-first, then by horizon. The readiness scan yields
    # horizon-major order, which interleaves one slow LSTM fit between every pair
    # of fast ones and delays the first servable ensemble by the full run.
    price = sorted(
        (
            (component.model_type, int(component.horizon))
            for component in report.trainable
            if component.component == COMPONENT_PRICE
        ),
        key=lambda pair: (_price_training_order(pair[0]), pair[1]),
    )
    unified = [
        component.model_type
        for component in report.trainable
        if component.component == COMPONENT_UNIFIED
    ]
    direction = any(
        component.component == COMPONENT_DIRECTION for component in report.trainable
    )
    return {"price": price, "unified": unified, "direction": direction}


def _run_price_training(job: PreparationJob, pairs: List[tuple], raw_df) -> None:
    """
    Train exactly the bundles the plan named, reusing the existing pipeline.

    Each pair is trained independently and a failure is recorded rather than
    raised: one model type that cannot fit must not cost the user the two that
    can, and the ensemble is explicitly built to serve a subset.
    """
    from src.models.ensemble_training import train_regression_bundle

    stage = job.stage("price_models")
    total = len(pairs)
    for index, (model_type, horizon) in enumerate(pairs):
        stage.detail = f"{model_type} · {horizon}d ({index + 1} of {total})"
        try:
            train_regression_bundle(
                symbol=job.symbol,
                model_type=model_type,
                horizon=horizon,
                raw_df=raw_df,
            )
            job.trained.append(f"{model_type}@{horizon}")
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            logger.exception("Auto-training failed for %s %s h=%s", job.symbol, model_type, horizon)
            job.warnings.append(f"{model_type} at {horizon}d could not be trained: {exc}")
        stage.progress = (index + 1) / max(total, 1)


def _run_unified_training(job: PreparationJob, model_types: List[str], raw_df) -> None:
    """Train the next-bar bundles a request explicitly asked for."""
    from src.models.unified_training import train_unified_bundle

    stage = job.stage("price_models")
    for index, model_type in enumerate(model_types):
        stage.detail = f"{model_type} ({index + 1} of {len(model_types)})"
        try:
            train_unified_bundle(symbol=job.symbol, model_type=model_type, raw_df=raw_df)
            job.trained.append(model_type)
        except Exception as exc:  # noqa: BLE001 - recorded, not fatal
            logger.exception("Auto-training failed for %s %s", job.symbol, model_type)
            job.warnings.append(f"{model_type} could not be trained: {exc}")


def _run_direction_training(job: PreparationJob, bars) -> None:
    """
    Run the walk-forward evaluation and write the same three files the CLI writes.

    Identical settings to ``scripts/direction_backtest.py`` — folds, embargo,
    cost, and the shuffled-label leakage check — because the route that reads the
    result presents it as a measured out-of-sample record, and a browser-started
    run that quietly skipped the leakage check would not be one.
    """
    from src.backtesting.direction_backtest import DEFAULT_THRESHOLD_GRID
    from src.features.direction_features import build_direction_dataset
    from src.models.direction_pipeline import run_walk_forward, write_direction_outputs

    stage = job.stage("direction_model")
    stage.detail = "Building the labelled dataset"
    dataset = build_direction_dataset(bars.frame)

    stage.progress = 0.15
    stage.detail = f"Walk-forward folds for {job.direction_model}"
    result = run_walk_forward(
        dataset,
        model_name=job.direction_model,
        threshold_grid=DEFAULT_THRESHOLD_GRID,
        data_meta=bars.meta,
    )

    stage.progress = 0.9
    stage.detail = "Writing the evaluation report"
    write_direction_outputs(result, job.symbol, job.direction_model)
    job.trained.append(f"direction:{job.direction_model}")
    stage.detail = (
        "Ship" if result.ship else "Evaluated — the verdict is do not ship, which the gauge reports"
    )


def _execute(job: PreparationJob, plan: Dict[str, Any]) -> None:
    """The pipeline, in the order the user was told it runs."""
    from src.data.direction_data import load_daily_bars
    from src.models.ensemble_training import download_training_data

    job.status = RUNNING
    job.started_at = datetime.now().isoformat()

    # ── 1. Market data ───────────────────────────────────────────────────
    # Both downloads happen here so an unknown or delisted ticker fails once,
    # in the stage named for it, instead of halfway through a training run.
    stage = job.stage("market_data")
    stage.state = RUNNING
    raw_df = None
    bars = None
    if plan["price"] or plan["unified"]:
        stage.detail = "Daily OHLCV for the price models"
        raw_df = download_training_data(job.symbol)
        stage.progress = 0.5
    if plan["direction"]:
        stage.detail = "Adjusted bars for the direction evaluation"
        bars = load_daily_bars(job.symbol)
    stage.state = DONE
    stage.detail = None

    # ── 2. Direction evaluation ──────────────────────────────────────────
    # Before the bundles, not after: seconds of work feeding the panel at the
    # top of the tab, versus tens of minutes for the forecast below it.
    stage = job.stage("direction_model")
    if plan["direction"]:
        stage.state = RUNNING
        _run_direction_training(job, bars)
        stage.state = DONE
    else:
        stage.state = SKIPPED
        stage.detail = "Evaluation already on disk"

    # ── 3. Price and next-bar bundles ────────────────────────────────────
    stage = job.stage("price_models")
    if plan["price"] or plan["unified"]:
        stage.state = RUNNING
        if plan["price"]:
            _run_price_training(job, plan["price"], raw_df)
        if plan["unified"]:
            _run_unified_training(job, plan["unified"], raw_df)
        stage.state = DONE
        stage.progress = 1.0
        stage.detail = f"{len(job.trained)} bundle(s) trained"
    else:
        stage.state = SKIPPED
        stage.detail = "Already trained"

    # ── 4/5. Confirm the result actually serves ──────────────────────────
    # Re-assessing is the only honest way to finish: training that "succeeded"
    # while leaving nothing servable is a distinction the user has to be told
    # about, and it is what the tabs read to decide what to render.
    stage = job.stage("predictions")
    stage.state = RUNNING
    report = assess_symbol(
        job.symbol,
        horizons=job.horizons,
        direction_model=job.direction_model,
        unified_models=job.unified_models,
    )
    job.readiness = report.as_dict()
    stage.state = DONE
    stage.detail = (
        "Forecasts available" if report.price_ready
        else "No price model cleared its out-of-sample baseline"
    )

    stage = job.stage("analysis")
    stage.state = DONE
    stage.detail = (
        "Direction evaluation available" if report.direction_ready
        else "Direction evaluation unavailable"
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class PreparationRegistry:
    """Job bookkeeping: dedupe by symbol, bound the pool, honour the cooldown."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: TTLCache = TTLCache(maxsize=MAX_TRACKED_JOBS, ttl=JOB_TTL_SECONDS)
        self._by_symbol: Dict[str, str] = {}
        # symbol -> (monotonic time the attempt finished, the job that made it)
        self._last_attempt: Dict[str, tuple] = {}
        self._executor: Optional[ThreadPoolExecutor] = None

    # ── internals ────────────────────────────────────────────────────────
    def _pool(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=worker_count(),
                thread_name_prefix="model-prepare",
            )
        return self._executor

    def _active_job(self, symbol: str) -> Optional[PreparationJob]:
        job_id = self._by_symbol.get(symbol)
        return self._jobs.get(job_id) if job_id else None

    def _cooling_down(self, symbol: str) -> bool:
        """
        Whether the last attempt for ``symbol`` is recent enough to hold off a repeat.

        The record is (when, which job), and the job half is load-bearing: the
        cooldown exists to avoid repeating an attempt whose outcome the caller can
        still be shown. Once that job has aged out of the tracker there is nothing
        left to show, so the entry is dropped and a fresh run is allowed. Without
        this, a symbol could be held in a cooldown whose evidence no longer exists
        and `ensure` would answer None — indistinguishable, to a caller, from
        "there is nothing to train".
        """
        cooldown = cooldown_seconds()
        if not cooldown:
            return False
        record = self._last_attempt.get(symbol)
        if record is None:
            return False
        attempted_at, job_id = record
        if job_id not in self._jobs:
            self._last_attempt.pop(symbol, None)
            return False
        return (time.monotonic() - attempted_at) < cooldown

    def _worker(self, job: PreparationJob, plan: Dict[str, Any]) -> None:
        try:
            _execute(job, plan)
            job.status = COMPLETED
        except Exception as exc:  # noqa: BLE001 - surfaced to the client verbatim
            logger.exception("Model preparation failed for %s", job.symbol)
            job.status = FAILED
            job.error = str(exc) or exc.__class__.__name__
            for stage in job.stages:
                if stage.state == RUNNING:
                    stage.state = FAILED
                    stage.detail = job.error
        finally:
            job.finished_at = datetime.now().isoformat()
            with self._lock:
                self._last_attempt[job.symbol] = (time.monotonic(), job.job_id)
                if self._by_symbol.get(job.symbol) == job.job_id:
                    self._by_symbol.pop(job.symbol, None)

    # ── public API ───────────────────────────────────────────────────────
    def get(self, job_id: str) -> Optional[PreparationJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def active_for(self, symbol: str) -> Optional[PreparationJob]:
        with self._lock:
            job = self._active_job(symbol.upper().strip())
        return job if job and job.active else None

    def latest_for(self, symbol: str) -> Optional[PreparationJob]:
        """The running job if there is one, else the most recent finished job."""
        symbol = symbol.upper().strip()
        with self._lock:
            running = self._active_job(symbol)
            if running:
                return running
            finished = [job for job in self._jobs.values() if job.symbol == symbol]
        return max(finished, key=lambda job: job.created_at) if finished else None

    def ensure(
        self,
        symbol: str,
        *,
        horizons: Optional[Sequence[int]] = None,
        direction_model: str = DEFAULT_DIRECTION_MODEL,
        unified_models: Optional[Sequence[str]] = None,
        force: bool = False,
        readiness: Optional[ReadinessReport] = None,
    ) -> Optional[PreparationJob]:
        """
        Start (or join) preparation for ``symbol``.

        Returns the job to poll, or ``None`` when there is nothing training can
        add — either everything is already trained, or what is left is
        ``unproven`` and would come out the same way again.
        """
        symbol = symbol.upper().strip()
        resolved_horizons = list(horizons or PREPARATION_HORIZONS)
        resolved_unified = list(unified_models or [])

        with self._lock:
            running = self._active_job(symbol)
            if running and running.active:
                return running

            report = readiness or assess_symbol(
                symbol,
                horizons=resolved_horizons,
                direction_model=direction_model,
                unified_models=resolved_unified,
            )
            if not report.needs_training:
                return None

            if not force and self._cooling_down(symbol):
                # The last attempt did not resolve everything. Hand back what it
                # said rather than restarting a run that just finished failing.
                return self.latest_for(symbol)

            plan = _plan(report)
            job = PreparationJob(
                job_id=str(uuid.uuid4()),
                symbol=symbol,
                horizons=resolved_horizons,
                direction_model=direction_model,
                unified_models=resolved_unified,
                stages=_new_stages(),
            )
            self._jobs[job.job_id] = job
            self._by_symbol[symbol] = job.job_id
            self._pool().submit(self._worker, job, plan)

        logger.info(
            "Preparing %s: %d price bundle(s), direction evaluation %s",
            symbol,
            len(plan["price"]),
            "queued" if plan["direction"] else "already present",
        )
        return job

    def reset(self) -> None:
        """Drop all job state. Tests only."""
        with self._lock:
            self._jobs.clear()
            self._by_symbol.clear()
            self._last_attempt.clear()


registry = PreparationRegistry()


def ensure_prepared(
    symbol: str,
    *,
    horizons: Optional[Sequence[int]] = None,
    direction_model: str = DEFAULT_DIRECTION_MODEL,
    unified_models: Optional[Sequence[str]] = None,
    force: bool = False,
    readiness: Optional[ReadinessReport] = None,
    respect_auto_flag: bool = True,
) -> Optional[PreparationJob]:
    """
    Module-level entry point used by the serving routes.

    ``respect_auto_flag`` is what separates the two callers. A prediction request
    that finds nothing to serve prepares the symbol *implicitly*, and an operator
    must be able to turn that off with ``QUANTVISION_AUTO_PREPARE=false``. An
    explicit ``POST /api/models/{symbol}/prepare`` is the operator asking, so it
    passes ``respect_auto_flag=False`` and runs regardless.
    """
    if respect_auto_flag and not auto_prepare_enabled():
        return None
    return registry.ensure(
        symbol,
        horizons=horizons,
        direction_model=direction_model,
        unified_models=unified_models,
        force=force,
        readiness=readiness,
    )


def preparation_state(
    symbol: str,
    *,
    horizons: Optional[Sequence[int]] = None,
    direction_model: str = DEFAULT_DIRECTION_MODEL,
    unified_models: Optional[Sequence[str]] = None,
    auto_start: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    The ``preparation`` block a serving response carries when it cannot answer yet.

    Callers embed this instead of a "go train it yourself" message, so a client
    that only knows how to read one prediction endpoint still gets told that work
    is underway and which job to watch.

    Serving routes deliberately leave ``horizons`` unset. Scoping a job to the one
    horizon a request happened to ask for would let that narrow run claim the
    symbol's cooldown, and every other horizon would then sit untrained until it
    expired. One symbol, one plan, whichever route noticed first.
    """
    job = None
    if auto_start:
        job = ensure_prepared(
            symbol,
            horizons=horizons,
            direction_model=direction_model,
            unified_models=unified_models,
        )
    if job is None:
        job = registry.latest_for(symbol)
    if job is None:
        return None
    return job.as_dict()
