"""
Model manager: decides what a symbol still needs before it can be served.

Every serving path in this project — the multi-horizon price ensemble, the
next-day direction gauge, and the analysis view that reads both — asks the same
question first: *is there a usable artifact on disk for this symbol?* Before this
module each path answered it privately, with its own notion of "usable", and the
only cure for "no" was a human running a training command. That is what put
"Train ensemble for NVDA" in front of a user who had done nothing but type a
ticker.

The answer is a classification, not a boolean, because the ways an artifact can
be unusable call for different responses:

    missing / invalid / incompatible   training fixes it — prepare the symbol
    stale                              usable now, refresh in the background
    unproven                           training does NOT fix it

``unproven`` is the one that matters most here. A bundle that trained cleanly and
then failed its out-of-sample skill gate is not a gap in the artifact tree; it is
a measurement, and the measurement comes out the same on the next fit of the same
bars. Treating it as "missing" would put the app in a retrain loop that never
converges — and on this repository's own bundles that is the common case, not the
corner case: most trained bundles do not beat a constant forecast. So the manager
reports it as a finished result with a reason, and the caller renders the reason
instead of a training button.

Nothing here trains anything. :mod:`src.models.preparation` owns that, and calls
this module to decide what to run and to confirm afterwards that it worked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.models.ensemble_predictor import (
    MODEL_TYPES as PRICE_MODEL_TYPES,
    bundle_skill_failure,
    metadata_is_return_regression,
    regression_bundle_dir,
)
from src.models.ensemble_training import TRAINABLE_HORIZONS
from src.models.direction_pipeline import DEFAULT_REPORT_DIR, report_stem
# The root `load_model_bundle` serves unified bundles from — the same constant, so
# the manager cannot look somewhere the predict route does not.
from src.models.model_bundle import BUNDLES_DIR as UNIFIED_BUNDLES_DIR
from src.models.regression_models import REGRESSOR_FILE_NAMES
from src.utils.config_loader import get_env_bool, get_env_int
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Where scripts/direction_backtest.py and the preparation service both write
# walk-forward artifacts, and where src.api.routes.direction reads them.
DIRECTION_REPORT_DIR = DEFAULT_REPORT_DIR

DEFAULT_DIRECTION_MODEL = "logistic"

# Horizons the Predictions tab offers. The 1-day step bundle is trained too (it
# is what the recursive per-step forecast rolls forward) but it is not a horizon
# a user can select, so its absence never blocks serving.
SERVING_HORIZONS: List[int] = [7, 15, 30, 60]
RECURSIVE_STEP_HORIZON = 1

# Everything preparation trains: the four display horizons plus the step bundle.
PREPARATION_HORIZONS: List[int] = [RECURSIVE_STEP_HORIZON, *SERVING_HORIZONS]

# ── Retraining policy ────────────────────────────────────────────────────────
AUTO_PREPARE_ENV = "QUANTVISION_AUTO_PREPARE"
MODEL_MAX_AGE_ENV = "QUANTVISION_MODEL_MAX_AGE_DAYS"
DEFAULT_MODEL_MAX_AGE_DAYS = 30

# Unified bundles answer for the next bar with one price and one P(up). They are
# never part of the default plan: the Predictions tab fetches one only when a
# user picks it by name, and training all three for every ticker anyone glances
# at would triple the wait for models most sessions never open. They are prepared
# on demand instead, when a request for one finds nothing to serve.
UNIFIED_MODEL_TYPES: List[str] = ["unified_xgboost", "unified_random_forest", "unified_lstm"]

# ── Component states ─────────────────────────────────────────────────────────
COMPONENT_PRICE = "price_model"
COMPONENT_DIRECTION = "direction_model"
COMPONENT_UNIFIED = "unified_model"

STATE_READY = "ready"
STATE_STALE = "stale"
STATE_MISSING = "missing"
STATE_INVALID = "invalid"
STATE_INCOMPATIBLE = "incompatible"
STATE_UNPROVEN = "unproven"

#: States whose artifact can be served right now. ``stale`` is included on
#: purpose: an artifact two months old still answers, and blanking the page while
#: a refresh runs would be a worse lie than showing the age beside the number.
SERVING_STATES = frozenset({STATE_READY, STATE_STALE})

#: States that a training run resolves. ``unproven`` is deliberately absent.
TRAINABLE_STATES = frozenset({STATE_MISSING, STATE_INVALID, STATE_INCOMPATIBLE, STATE_STALE})


def auto_prepare_enabled() -> bool:
    """Whether a serving request may start training for a symbol it cannot serve."""
    return get_env_bool(AUTO_PREPARE_ENV, True)


def model_max_age_days() -> int:
    """Age past which an artifact is refreshed in the background. 0 disables it."""
    return max(0, get_env_int(MODEL_MAX_AGE_ENV, DEFAULT_MODEL_MAX_AGE_DAYS))


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed


def _age_days(trained_at: Any) -> Optional[float]:
    stamp = _parse_timestamp(trained_at)
    if stamp is None:
        return None
    return (datetime.now() - stamp).total_seconds() / 86400.0


@dataclass
class ComponentStatus:
    """One artifact's verdict: what it is, whether it serves, whether to train it."""

    component: str
    key: str
    state: str
    detail: Optional[str] = None
    trained_at: Optional[str] = None
    age_days: Optional[float] = None
    model_type: Optional[str] = None
    horizon: Optional[int] = None

    @property
    def servable(self) -> bool:
        return self.state in SERVING_STATES

    @property
    def needs_training(self) -> bool:
        return self.state in TRAINABLE_STATES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "key": self.key,
            "state": self.state,
            "detail": self.detail,
            "trained_at": self.trained_at,
            "age_days": round(self.age_days, 1) if self.age_days is not None else None,
            "model_type": self.model_type,
            "horizon": self.horizon,
            "servable": self.servable,
            "needs_training": self.needs_training,
        }


@dataclass
class ReadinessReport:
    """What a symbol can serve now, what needs training, and what never will."""

    symbol: str
    horizons: List[int]
    direction_model: str
    components: List[ComponentStatus] = field(default_factory=list)

    # ── Views over `components` ──────────────────────────────────────────
    @property
    def price_components(self) -> List[ComponentStatus]:
        return [c for c in self.components if c.component == COMPONENT_PRICE]

    @property
    def direction_component(self) -> Optional[ComponentStatus]:
        for component in self.components:
            if component.component == COMPONENT_DIRECTION:
                return component
        return None

    @property
    def trainable(self) -> List[ComponentStatus]:
        return [c for c in self.components if c.needs_training]

    @property
    def gated_horizons(self) -> List[int]:
        """
        Horizons whose absence actually blocks the UI.

        The 1-day step bundle is trained but never displayed: without it the
        ensemble falls back to a compounded path from one model output, which is
        a degradation the response already declares. Letting it gate the whole
        tab would black out four working horizons over a fifth nobody asked for.
        """
        displayed = [h for h in self.horizons if h != RECURSIVE_STEP_HORIZON]
        return displayed or list(self.horizons)

    @property
    def price_ready(self) -> bool:
        """At least one ensemble member serves every displayed horizon."""
        horizons = self.gated_horizons
        if not horizons:
            return False
        return all(
            any(c.servable and c.horizon == horizon for c in self.price_components)
            for horizon in horizons
        )

    @property
    def direction_ready(self) -> bool:
        component = self.direction_component
        return bool(component and component.servable)

    @property
    def ready(self) -> bool:
        return self.price_ready and self.direction_ready

    @property
    def needs_training(self) -> bool:
        return bool(self.trainable)

    @property
    def blocked(self) -> List[ComponentStatus]:
        """Artifacts that exist, cannot serve, and will not be fixed by training."""
        return [c for c in self.components if c.state == STATE_UNPROVEN]

    def summary(self) -> str:
        """One sentence a user can read, chosen by what is actually true."""
        if self.needs_training:
            count = len(self.trainable)
            noun = "artifact" if count == 1 else "artifacts"
            return f"{self.symbol} needs {count} model {noun} trained."
        if not self.ready and self.blocked:
            return (
                f"{self.symbol} models are trained, but {len(self.blocked)} of them did not "
                f"beat their out-of-sample baseline, so those forecasts are withheld "
                f"rather than presented as skill."
            )
        return f"{self.symbol} models are trained and ready."

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizons": self.horizons,
            "direction_model": self.direction_model,
            "ready": self.ready,
            "price_ready": self.price_ready,
            "direction_ready": self.direction_ready,
            "needs_training": self.needs_training,
            "components": [c.as_dict() for c in self.components],
            "trainable": [c.key for c in self.trainable],
            "blocked": [{"key": c.key, "detail": c.detail} for c in self.blocked],
            "summary": self.summary(),
            "max_age_days": model_max_age_days(),
            "auto_prepare": auto_prepare_enabled(),
        }


# ---------------------------------------------------------------------------
# Price regression bundles
# ---------------------------------------------------------------------------

def assess_price_bundle(symbol: str, model_type: str, horizon: int) -> ComponentStatus:
    """
    Classify one ``models/bundles/<SYMBOL>/<MODEL>/<HORIZON>/`` bundle.

    The serving rules are the predictor's own — a bundle rejected here would have
    been rejected by :func:`src.models.ensemble_predictor.regression_bundle_status`
    too. What this adds is the *reason*, kept separate so the caller can tell a
    gap that training fills from a verdict that training only repeats.
    """
    common: Dict[str, Any] = {
        "component": COMPONENT_PRICE,
        "key": f"{model_type}@{horizon}",
        "model_type": model_type,
        "horizon": int(horizon),
    }

    bundle_dir = regression_bundle_dir(symbol, model_type, horizon)
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return ComponentStatus(
            state=STATE_MISSING,
            detail=f"no {model_type} bundle is trained for {symbol.upper()} at horizon {horizon}",
            **common,
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ComponentStatus(
            state=STATE_INVALID,
            detail=f"the {model_type} bundle metadata could not be read ({exc})",
            **common,
        )

    trained_at = meta.get("trained_at")
    age = _age_days(trained_at)
    dated: Dict[str, Any] = {
        "trained_at": str(trained_at) if trained_at else None,
        "age_days": age,
    }

    if not metadata_is_return_regression(meta):
        return ComponentStatus(
            state=STATE_INCOMPATIBLE,
            detail=(
                f"the {model_type} bundle predicts prices rather than returns and "
                f"must be retrained against the current objective"
            ),
            **common,
            **dated,
        )

    # An artifact the metadata points at but that is not on disk is a broken
    # bundle, not a missing one — the distinction shows up only in the message.
    model_path = Path(str(meta.get("model_path", "")))
    if not model_path.exists():
        model_path = bundle_dir / REGRESSOR_FILE_NAMES.get(model_type, "model.joblib")
    if not model_path.exists():
        return ComponentStatus(
            state=STATE_INVALID,
            detail=f"the {model_type} bundle metadata exists but its model artifact is missing",
            **common,
            **dated,
        )

    # A bundle predating the skill record has no evidence either way, so it is
    # retrained to produce one. A bundle that recorded a failure has its
    # evidence, and refitting the same bars would only reproduce it.
    if "passes_baseline" not in meta:
        return ComponentStatus(
            state=STATE_INCOMPATIBLE,
            detail=(
                f"the {model_type} bundle was trained before out-of-sample skill was "
                f"recorded, so there is no evidence it beats a constant forecast"
            ),
            **common,
            **dated,
        )

    skill_failure = bundle_skill_failure(meta)
    if skill_failure:
        return ComponentStatus(
            state=STATE_UNPROVEN,
            detail=f"the {model_type} bundle {skill_failure}",
            **common,
            **dated,
        )

    max_age = model_max_age_days()
    if max_age and age is not None and age > max_age:
        return ComponentStatus(
            state=STATE_STALE,
            detail=f"the {model_type} bundle is {age:.0f} days old (refresh policy: {max_age} days)",
            **common,
            **dated,
        )

    return ComponentStatus(state=STATE_READY, detail=None, **common, **dated)


# ---------------------------------------------------------------------------
# Unified next-bar bundles
# ---------------------------------------------------------------------------

def assess_unified_bundle(
    symbol: str,
    model_type: str,
    bundles_dir: Path = UNIFIED_BUNDLES_DIR,
) -> ComponentStatus:
    """
    Classify one ``models/bundles/<SYMBOL>/<UNIFIED_MODEL>/`` bundle.

    There is no skill gate on these — the serving path loads whatever is on disk
    — so the states reduce to present, broken, or old.
    """
    common: Dict[str, Any] = {
        "component": COMPONENT_UNIFIED,
        "key": model_type,
        "model_type": model_type,
        "horizon": RECURSIVE_STEP_HORIZON,
    }

    bundle_dir = bundles_dir / symbol.upper() / model_type
    meta_path = bundle_dir / "metadata.json"
    if not meta_path.exists():
        return ComponentStatus(
            state=STATE_MISSING,
            detail=f"no {model_type} bundle is trained for {symbol.upper()}",
            **common,
        )

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ComponentStatus(
            state=STATE_INVALID,
            detail=f"the {model_type} bundle metadata could not be read ({exc})",
            **common,
        )

    trained_at = meta.get("trained_at")
    age = _age_days(trained_at)
    dated: Dict[str, Any] = {
        "trained_at": str(trained_at) if trained_at else None,
        "age_days": age,
    }

    if not (bundle_dir / "model.joblib").exists():
        return ComponentStatus(
            state=STATE_INVALID,
            detail=f"the {model_type} bundle metadata exists but its model artifact is missing",
            **common,
            **dated,
        )

    max_age = model_max_age_days()
    if max_age and age is not None and age > max_age:
        return ComponentStatus(
            state=STATE_STALE,
            detail=f"the {model_type} bundle is {age:.0f} days old (refresh policy: {max_age} days)",
            **common,
            **dated,
        )

    return ComponentStatus(state=STATE_READY, detail=None, **common, **dated)


# ---------------------------------------------------------------------------
# Direction walk-forward reports
# ---------------------------------------------------------------------------

def direction_report_path(
    symbol: str,
    model: str = DEFAULT_DIRECTION_MODEL,
    report_dir: Path = DIRECTION_REPORT_DIR,
) -> Path:
    return report_dir / f"{report_stem(symbol, model)}_report.json"


def assess_direction_report(
    symbol: str,
    model: str = DEFAULT_DIRECTION_MODEL,
    report_dir: Path = DIRECTION_REPORT_DIR,
) -> ComponentStatus:
    """
    Classify the stored walk-forward evaluation for one symbol/model.

    A report whose verdict is "do not ship" is ``ready``, not ``unproven``: the
    evaluation *is* the artifact being asked for, and the route gates the gauge on
    the verdict inside it. Re-running the walk-forward would reproduce the same
    verdict and destroy the only thing that makes the gauge readable.
    """
    common: Dict[str, Any] = {
        "component": COMPONENT_DIRECTION,
        "key": f"direction:{model}",
        "model_type": model,
        "horizon": 1,
    }

    path = direction_report_path(symbol, model, report_dir)
    if not path.exists():
        return ComponentStatus(
            state=STATE_MISSING,
            detail=f"no walk-forward evaluation has been run for {symbol.upper()} ({model})",
            **common,
        )

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ComponentStatus(
            state=STATE_INVALID,
            detail=f"the stored walk-forward report could not be read ({exc})",
            **common,
        )

    if not isinstance(report, dict) or "pooled" not in report or "verdict" not in report:
        return ComponentStatus(
            state=STATE_INVALID,
            detail="the stored walk-forward report is missing its pooled results",
            **common,
        )

    generated_at = report.get("generated_at")
    age = _age_days(generated_at)
    dated: Dict[str, Any] = {
        "trained_at": str(generated_at) if generated_at else None,
        "age_days": age,
    }

    max_age = model_max_age_days()
    if max_age and age is not None and age > max_age:
        return ComponentStatus(
            state=STATE_STALE,
            detail=f"the walk-forward evaluation is {age:.0f} days old (refresh policy: {max_age} days)",
            **common,
            **dated,
        )

    return ComponentStatus(state=STATE_READY, detail=None, **common, **dated)


# ---------------------------------------------------------------------------
# Whole-symbol readiness
# ---------------------------------------------------------------------------

def normalise_horizons(horizons: Optional[Sequence[int]]) -> List[int]:
    """Requested horizons, restricted to those the training pipeline supports."""
    if not horizons:
        return list(SERVING_HORIZONS)
    resolved = sorted({int(h) for h in horizons if int(h) in TRAINABLE_HORIZONS})
    return resolved or list(SERVING_HORIZONS)


def assess_symbol(
    symbol: str,
    *,
    horizons: Optional[Sequence[int]] = None,
    model_types: Optional[Sequence[str]] = None,
    direction_model: str = DEFAULT_DIRECTION_MODEL,
    unified_models: Optional[Sequence[str]] = None,
    report_dir: Path = DIRECTION_REPORT_DIR,
) -> ReadinessReport:
    """
    Classify every artifact the dashboard needs for one symbol.

    ``unified_models`` is opt-in. Leaving it empty is what keeps the default plan
    to the artifacts the tabs actually render on load.
    """
    symbol = symbol.upper().strip()
    resolved_horizons = normalise_horizons(horizons)
    resolved_models = list(model_types or PRICE_MODEL_TYPES)

    components: List[ComponentStatus] = [
        assess_price_bundle(symbol, model_type, horizon)
        for horizon in resolved_horizons
        for model_type in resolved_models
    ]
    components.extend(
        assess_unified_bundle(symbol, model_type)
        for model_type in (unified_models or [])
        if model_type in UNIFIED_MODEL_TYPES
    )
    components.append(assess_direction_report(symbol, direction_model, report_dir))

    return ReadinessReport(
        symbol=symbol,
        horizons=resolved_horizons,
        direction_model=direction_model,
        components=components,
    )
