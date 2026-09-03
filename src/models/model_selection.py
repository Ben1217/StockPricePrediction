"""
Picking the best performing model, per symbol and per horizon.

The dashboard draws one forecast on the chart, so something has to decide whose
forecast it is. This module is that decision, and it is kept out of the route so
the rule can be read, tested and argued with in one place instead of being
implied by whichever branch a request happens to take.

Two winners, not one
--------------------
A model that tracks the *level* well is not the model that calls the *sign*
well, and the evaluation suite already refuses to merge the two into a single
score (see :mod:`src.models.unified_evaluation`). So this module returns two
winners:

    price      -- lowest MAE / RMSE / MAPE and highest R-squared. Draws the
                  trajectory and its band.
    direction  -- highest accuracy / F1 / ROC-AUC. Draws the up/down call.

They are frequently different models, and the payload says so rather than
quietly serving one model's number under the other's name.

What makes two numbers comparable
---------------------------------
Three things have to match before two models can be ranked against each other,
and each one is a way this could silently produce a wrong winner:

*Horizon.* A 30-day MAPE is several times a 1-day MAPE for the same model.
Candidates are therefore grouped by the horizon they were scored at, and a
model evaluated at another horizon is never ranked against them.

*Units.* The per-horizon regression bundles are trained on the forward return,
so their stored MAE and RMSE are in return units (0.06 = six percent), while the
unified bundles store theirs in dollars. Return-space errors are converted with
a reference close before anything is compared -- ``mape`` needs no conversion,
because a relative error is the same number in both spaces.

*Metric coverage.* Not every source records every metric: the legacy bundles
have no R-squared. Ranking on a metric that only some candidates have would
hand the win to whoever happened to be measured, so the ranking runs on the
metrics *all* candidates in the group share, and reports which those were.

How the ranking works
---------------------
Borda count over the shared metrics: each candidate is ranked on each metric,
and the mean of those ranks decides the winner. One combined ordering is more
stable than picking a single metric to be the arbiter, and it makes the common
case -- a model that wins three of four metrics -- come out the way a reader
would expect. The per-metric winners are reported alongside, so a split
decision is visible instead of averaged away.

Only servable models are ranked
-------------------------------
A winner nobody can serve is not a winner. Candidates are filtered through the
same skill gates the serving routes apply (``bundle_skill_failure`` for the
regression bundles, the walk-forward ``verdict`` for the direction reports), so
the model this module names is one :mod:`src.api.routes.predict` can actually
run. A model that scored well and failed its gate is still reported, with the
reason, under ``excluded``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.models.direction_pipeline import DEFAULT_REPORT_DIR, report_stem
from src.models.ensemble_predictor import bundle_skill_failure, skill_enforcement_enabled
from src.models.model_bundle import BUNDLES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

#: Head-to-head walk-forward artifacts written by ``scripts/unified_benchmark.py``.
BENCHMARK_ARTIFACT = Path("artifacts") / "benchmark_results.json"

#: Horizons the per-horizon regression bundles are trained for.
REGRESSION_HORIZONS: Tuple[int, ...] = (7, 15, 30, 60)

#: The next-bar horizon the unified and direction models answer at.
NEXT_BAR_HORIZON = 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Metric:
    """One scored quantity, and which end of it is good."""

    key: str
    label: str
    #: True when a smaller number is a better model (errors), False for scores.
    lower_is_better: bool


#: The price scorecard from the brief, in the order it is reported.
PRICE_METRICS: Tuple[Metric, ...] = (
    Metric("mae", "MAE", True),
    Metric("rmse", "RMSE", True),
    Metric("mape", "MAPE", True),
    # R-squared on the forward *return*, not the price level. Against the price
    # level any forecast of "roughly today's close" scores near 1.0, so the
    # level R-squared cannot separate models and the return one can. Both are
    # carried; only this one is ranked on.
    Metric("r2", "R2", False),
)

#: The direction scorecard, likewise.
DIRECTION_METRICS: Tuple[Metric, ...] = (
    Metric("accuracy", "Accuracy", False),
    Metric("f1", "F1", False),
    Metric("roc_auc", "AUC", False),
)


def _finite(value: Any) -> Optional[float]:
    """A float, or None for anything that cannot be ranked (None, NaN, inf, text)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """
    One model's scorecard for one horizon, with the provenance of the numbers.

    ``model_type`` is the name the serving routes accept, so a winner can be
    handed straight to ``POST /api/predict`` without a lookup table in between.
    """

    model_type: str
    label: str
    #: "walk_forward_benchmark" | "bundle_holdout" | "direction_walk_forward"
    evidence: str
    horizon: int
    price: Dict[str, Optional[float]] = field(default_factory=dict)
    direction: Dict[str, Optional[float]] = field(default_factory=dict)
    #: Extra context a reader needs to weigh the numbers: sample size, base rate.
    context: Dict[str, Any] = field(default_factory=dict)
    #: Why this model must not be served, or None when it is servable.
    blocked_reason: Optional[str] = None

    @property
    def servable(self) -> bool:
        return self.blocked_reason is None

    def scores(self, family: str) -> Dict[str, Optional[float]]:
        return self.price if family == "price" else self.direction

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "label": self.label,
            "evidence": self.evidence,
            "horizon": self.horizon,
            "price": dict(self.price),
            "direction": dict(self.direction),
            "context": dict(self.context),
            "servable": self.servable,
            "blocked_reason": self.blocked_reason,
        }


MODEL_LABELS: Dict[str, str] = {
    "lstm": "LSTM",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "ensemble": "Ensemble",
    "unified_lstm": "Unified LSTM",
    "unified_xgboost": "Unified XGBoost",
    "unified_random_forest": "Unified Random Forest",
    "unified_ensemble": "Unified Ensemble",
    "unified_kronos": "Kronos",
    "unified_timesfm": "TimesFM",
    "unified_chronos": "Chronos",
    "logistic": "Logistic",
    "gradient_boosting": "Gradient Boosting",
    "tabpfn": "TabPFN",
    "kronos": "Kronos",
    "foundation_ensemble": "Foundation Ensemble",
}


def model_label(model_type: str) -> str:
    return MODEL_LABELS.get(model_type, model_type.replace("_", " ").title())


def _read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Could not read %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Source 1 - the head-to-head walk-forward benchmark
# ---------------------------------------------------------------------------

def _benchmark_candidates(
    symbol: str,
    horizon: int,
    artifact: Path = BENCHMARK_ARTIFACT,
) -> List[Candidate]:
    """
    Candidates from ``artifacts/benchmark_results.json``.

    This is the strongest evidence available: every model in it was scored over
    the same expanding-window folds, on the same features and the same targets,
    which is exactly the comparability the ranking needs. It only covers symbols
    somebody has actually run the benchmark for, so it is usually a subset.
    """
    payload = _read_json(artifact)
    if not isinstance(payload, list):
        return []

    candidates: List[Candidate] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("symbol", "")).upper() != symbol.upper():
            continue
        # The benchmark defaults to daily bars at horizon 1; rows written by an
        # intraday or multi-bar run carry their own interval and horizon and
        # must not be pooled with the daily ones.
        if str(row.get("interval", "1d")) != "1d":
            continue
        if int(row.get("horizon", NEXT_BAR_HORIZON)) != horizon:
            continue

        model_type = str(row.get("model_name") or "")
        if not model_type:
            continue

        candidates.append(
            Candidate(
                model_type=model_type,
                label=model_label(model_type),
                evidence="walk_forward_benchmark",
                horizon=horizon,
                price={
                    "mae": _finite(row.get("price_mae")),
                    "rmse": _finite(row.get("price_rmse")),
                    "mape": _finite(row.get("price_mape")),
                    "r2": _finite(row.get("price_r2_return")),
                    "r2_level": _finite(row.get("price_r2")),
                },
                direction={
                    "accuracy": _finite(row.get("direction_accuracy")),
                    "f1": _finite(row.get("direction_f1")),
                    "roc_auc": _finite(row.get("direction_roc_auc")),
                    "brier_score": _finite(row.get("direction_brier_score")),
                },
                context={
                    "n_test_rows": row.get("total_test_rows"),
                    "n_folds": row.get("n_splits"),
                    "base_rate": _finite(row.get("base_rate")),
                    "accuracy_std": _finite(row.get("direction_accuracy_std")),
                },
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Source 2 - the saved bundles' own holdout metrics
# ---------------------------------------------------------------------------

def _bundle_metadata_paths(symbol: str, bundles_dir: Path) -> List[Path]:
    """Every ``metadata.json`` under ``models/bundles/<SYMBOL>/``, at either depth."""
    symbol_dir = bundles_dir / symbol.upper()
    if not symbol_dir.is_dir():
        return []
    return sorted(symbol_dir.glob("*/metadata.json")) + sorted(symbol_dir.glob("*/*/metadata.json"))


def _regression_candidate(
    meta: Dict[str, Any], horizon: int, reference_price: Optional[float]
) -> Candidate:
    """
    A per-horizon regression bundle, converted into price-space scores.

    Its stored errors are on the forward return, so MAE and RMSE are multiplied
    by ``reference_price`` to reach dollars. Without a reference price the two
    are dropped rather than reported in the wrong units, and the ranking falls
    back to whatever the group still shares.
    """
    test = meta.get("test_metrics") or (meta.get("metrics") or {}).get("test") or {}
    mae_return = _finite(test.get("mae"))
    rmse_return = _finite(test.get("rmse"))
    scale = reference_price if reference_price and reference_price > 0 else None

    blocked = bundle_skill_failure(meta)
    model_type = str(meta.get("model_type") or "")
    return Candidate(
        model_type=model_type,
        label=model_label(model_type),
        evidence="bundle_holdout",
        horizon=horizon,
        price={
            "mae": mae_return * scale if (mae_return is not None and scale) else None,
            "rmse": rmse_return * scale if (rmse_return is not None and scale) else None,
            # Already relative: |price error| / price is the same number as the
            # absolute return error, so no conversion is needed or wanted.
            "mape": _finite(test.get("mape")),
            # These bundles record a skill score against a constant forecast
            # rather than an R-squared. Reporting the skill score in the R2
            # column would compare two different statistics.
            "r2": None,
            "mae_return": mae_return,
            "rmse_return": rmse_return,
            "skill_score": _finite(((meta.get("skill") or {}).get("test") or {}).get("skill_score")),
        },
        direction={
            # The regression bundles record how often the sign of their
            # predicted return was right. That is an accuracy, but it comes with
            # no F1 and no AUC, so it never wins the direction ranking on its
            # own - it is reported for context.
            "accuracy": _finite(test.get("directional_accuracy")),
            "f1": None,
            "roc_auc": None,
        },
        context={
            "n_test_rows": (meta.get("split_sizes") or {}).get("test"),
            "trained_at": meta.get("trained_at"),
            "target": meta.get("target_type"),
            "reference_price": scale,
        },
        blocked_reason=f"the bundle {blocked}" if blocked else None,
    )


def _unified_candidate(meta: Dict[str, Any]) -> Optional[Candidate]:
    """A unified bundle, whose holdout block is already in price space."""
    holdout = meta.get("holdout") or {}
    if not holdout:
        return None
    model_type = str(meta.get("model_type") or "")
    return Candidate(
        model_type=model_type,
        label=model_label(model_type),
        evidence="bundle_holdout",
        horizon=int(meta.get("horizon") or NEXT_BAR_HORIZON),
        price={
            "mae": _finite(holdout.get("price_mae")),
            "rmse": _finite(holdout.get("price_rmse")),
            "mape": _finite(holdout.get("price_mape")),
            "r2": _finite(holdout.get("price_r2_return")),
            "r2_level": _finite(holdout.get("price_r2")),
        },
        direction={
            "accuracy": _finite(holdout.get("direction_accuracy")),
            # A single chronological holdout records the accuracy only. F1 and
            # AUC come from the walk-forward suite, which is why a symbol with
            # no benchmark run has a thinner direction table.
            "f1": None,
            "roc_auc": None,
        },
        context={
            "n_test_rows": holdout.get("n_test"),
            "base_rate": _finite(holdout.get("base_rate")),
            "test_range": [holdout.get("test_start"), holdout.get("test_end")],
            "trained_at": meta.get("trained_at"),
        },
    )


def _bundle_candidates(
    symbol: str,
    horizon: int,
    reference_price: Optional[float],
    bundles_dir: Path = BUNDLES_DIR,
) -> List[Candidate]:
    """Candidates read from the saved bundles under ``models/bundles/<SYMBOL>/``."""
    candidates: List[Candidate] = []
    for path in _bundle_metadata_paths(symbol, bundles_dir):
        meta = _read_json(path)
        if not isinstance(meta, dict):
            continue

        if meta.get("objective") == "unified_price_and_direction":
            unified = _unified_candidate(meta)
            if unified is not None and unified.horizon == horizon and unified.model_type:
                candidates.append(unified)
            continue

        if meta.get("target_type") != "return_regression":
            continue
        if int(meta.get("horizon") or 0) != horizon:
            continue
        regression = _regression_candidate(meta, horizon, reference_price)
        if regression.model_type:
            candidates.append(regression)

    # One entry per model: a symbol can hold both a canonical and a legacy
    # bundle for the same pair, and the newer training run is the live one.
    latest: Dict[str, Candidate] = {}
    for candidate in candidates:
        existing = latest.get(candidate.model_type)
        if existing is None or str(candidate.context.get("trained_at") or "") >= str(
            existing.context.get("trained_at") or ""
        ):
            latest[candidate.model_type] = candidate
    return list(latest.values())


# ---------------------------------------------------------------------------
# Source 3 - the direction walk-forward reports
# ---------------------------------------------------------------------------

def direction_report_candidates(
    symbol: str,
    models: Iterable[str],
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> List[Candidate]:
    """
    Candidates from ``data/direction_backtests/*_report.json``.

    These are the classifiers the direction API serves, scored over pooled
    walk-forward test folds with accuracy, F1 and ROC-AUC - the full direction
    scorecard, which is why they are worth reading even though they answer only
    for the next bar. A report whose ``verdict.ship`` is false is carried with
    the reason, and never wins.
    """
    candidates: List[Candidate] = []
    for model in models:
        path = report_dir / f"{report_stem(symbol, model)}_report.json"
        if not path.exists():
            continue
        report = _read_json(path)
        if not isinstance(report, dict):
            continue

        pooled_root = report.get("pooled") or {}
        pooled = pooled_root.get("model") or {}
        if not pooled:
            continue
        horizon = int((report.get("config") or {}).get("horizon") or NEXT_BAR_HORIZON)
        verdict = report.get("verdict") or {}
        failed = verdict.get("failed_criteria") or []

        candidates.append(
            Candidate(
                model_type=model,
                label=model_label(model),
                evidence="direction_walk_forward",
                horizon=horizon,
                price={},
                direction={
                    "accuracy": _finite(pooled.get("accuracy")),
                    "f1": _finite((pooled.get("class_up") or {}).get("f1")),
                    "roc_auc": _finite(pooled.get("roc_auc")),
                    "brier_score": _finite(pooled.get("brier_score")),
                    "mcc": _finite(pooled.get("mcc")),
                },
                context={
                    "n_test_rows": pooled.get("n"),
                    "base_rate": _finite(pooled.get("base_rate")),
                    "n_folds": (report.get("config") or {}).get("n_folds_run"),
                    "generated_at": report.get("generated_at"),
                    "edge_pp": _finite(
                        (pooled_root.get("edge_vs_best_baseline") or {}).get("edge_pp")
                    ),
                },
                blocked_reason=(
                    None
                    if verdict.get("ship")
                    else "it did not clear its walk-forward ship criteria"
                    + (f" ({', '.join(failed)})" if failed else "")
                ),
            )
        )
    return candidates


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def _shared_metrics(candidates: Sequence[Candidate], family: str) -> List[Metric]:
    """
    The metrics every candidate in the group carries.

    Ranking on a metric only some candidates were measured on would decide the
    winner by who happened to be measured, so the intersection is the rankable
    set. It can legitimately be empty, and the caller reports that rather than
    inventing an ordering.
    """
    metrics = PRICE_METRICS if family == "price" else DIRECTION_METRICS
    return [
        metric
        for metric in metrics
        if all(
            _finite(candidate.scores(family).get(metric.key)) is not None
            for candidate in candidates
        )
    ]


def _rank_positions(candidates: Sequence[Candidate], family: str, metric: Metric) -> Dict[str, float]:
    """
    Competition ranks (1 = best) for one metric, ties sharing the average rank.

    Sharing the rank matters: two models with an identical MAPE must not have
    the tie broken by dictionary order, because that order is not evidence.
    """
    values = [
        (candidate.model_type, float(candidate.scores(family)[metric.key]))
        for candidate in candidates
    ]
    values.sort(key=lambda item: item[1], reverse=not metric.lower_is_better)

    ranks: Dict[str, float] = {}
    index = 0
    while index < len(values):
        stop = index
        while stop + 1 < len(values) and values[stop + 1][1] == values[index][1]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[values[position][0]] = shared
        index = stop + 1
    return ranks


@dataclass
class Selection:
    """The outcome of one ranking: who won, on what, and against whom."""

    family: str
    winner: Optional[Candidate]
    metrics_used: List[str]
    ranked: List[Dict[str, Any]]
    excluded: List[Dict[str, Any]]
    metric_winners: Dict[str, str]
    reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "winner": self.winner.as_dict() if self.winner else None,
            "metrics_used": list(self.metrics_used),
            "ranked": list(self.ranked),
            "excluded": list(self.excluded),
            "metric_winners": dict(self.metric_winners),
            "reason": self.reason,
        }


def rank_candidates(candidates: Sequence[Candidate], family: str) -> Selection:
    """
    Rank one family of metrics and name a winner.

    Borda count over the shared metrics; the mean rank decides, and a tie on the
    mean is broken by the larger test sample, because the same score over more
    out-of-sample rows is the better-evidenced one.
    """
    servable = [candidate for candidate in candidates if candidate.servable]
    excluded = [
        {**candidate.as_dict(), "excluded_because": candidate.blocked_reason}
        for candidate in candidates
        if not candidate.servable
    ]

    scorable = [
        candidate
        for candidate in servable
        if any(_finite(value) is not None for value in candidate.scores(family).values())
    ]
    if not scorable:
        return Selection(
            family=family,
            winner=None,
            metrics_used=[],
            ranked=[],
            excluded=excluded,
            metric_winners={},
            reason=f"No servable model carries a {family} scorecard for this symbol and horizon.",
        )

    metrics = _shared_metrics(scorable, family)
    if not metrics:
        # Every candidate has *some* score but no metric is common to all of
        # them, so there is no ordering that is not an artifact of who was
        # measured. Fall back to the largest genuinely comparable subset rather
        # than ranking apples against oranges.
        comparable_metrics, comparable = _largest_comparable_group(scorable, family)
        # The ones that fell out are recorded, not dropped. A candidate that
        # silently vanished from the table would read as never having existed,
        # when what happened is that nothing it was measured on could be
        # compared with the rest.
        kept = {candidate.model_type for candidate in comparable}
        excluded.extend(
            {
                **candidate.as_dict(),
                "excluded_because": (
                    f"its {family} scorecard shares no metric with the models it would "
                    f"be ranked against"
                ),
            }
            for candidate in scorable
            if candidate.model_type not in kept
        )
        metrics, scorable = comparable_metrics, comparable

    if not metrics:
        return Selection(
            family=family,
            winner=None,
            metrics_used=[],
            ranked=[{**candidate.as_dict(), "mean_rank": None} for candidate in scorable],
            excluded=excluded,
            metric_winners={},
            reason=(
                f"The {family} scorecards do not share a metric, so no two of them "
                f"can be ranked against each other."
            ),
        )

    per_metric = {metric.key: _rank_positions(scorable, family, metric) for metric in metrics}
    metric_winners = {
        metric.key: min(per_metric[metric.key].items(), key=lambda item: item[1])[0]
        for metric in metrics
    }

    ranked_rows: List[Dict[str, Any]] = []
    for candidate in scorable:
        positions = [per_metric[metric.key][candidate.model_type] for metric in metrics]
        ranked_rows.append(
            {
                **candidate.as_dict(),
                "mean_rank": round(sum(positions) / len(positions), 3),
                "metric_ranks": {
                    metric.key: per_metric[metric.key][candidate.model_type] for metric in metrics
                },
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            row["mean_rank"],
            -float(row["context"].get("n_test_rows") or 0),
            row["model_type"],
        )
    )
    winner = next(
        candidate for candidate in scorable if candidate.model_type == ranked_rows[0]["model_type"]
    )

    return Selection(
        family=family,
        winner=winner,
        metrics_used=[metric.key for metric in metrics],
        ranked=ranked_rows,
        excluded=excluded,
        metric_winners=metric_winners,
    )


def _largest_comparable_group(
    candidates: Sequence[Candidate], family: str
) -> Tuple[List[Metric], List[Candidate]]:
    """
    The biggest subset of candidates that do share a metric, and that metric set.

    Used only when the full group has no metric in common. Preferring the larger
    subset keeps the comparison as wide as the evidence allows; ties go to the
    subset scored on more metrics.
    """
    metrics = PRICE_METRICS if family == "price" else DIRECTION_METRICS
    best: Tuple[List[Metric], List[Candidate]] = ([], [])
    for metric in metrics:
        group = [
            candidate
            for candidate in candidates
            if _finite(candidate.scores(family).get(metric.key)) is not None
        ]
        if len(group) < 2:
            continue
        shared = _shared_metrics(group, family)
        if (len(group), len(shared)) > (len(best[1]), len(best[0])):
            best = (shared, group)

    if best[1]:
        return best

    # Nothing is comparable with anything else. A single scored candidate is
    # still a usable answer - it is the only model with evidence - so it is
    # returned alone, on whichever metric it has.
    for metric in metrics:
        group = [
            candidate
            for candidate in candidates
            if _finite(candidate.scores(family).get(metric.key)) is not None
        ]
        if group:
            return ([metric], group[:1])
    return ([], [])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

#: Direction classifiers with stored walk-forward reports worth reading.
DIRECTION_REPORT_MODELS: Tuple[str, ...] = (
    "logistic",
    "gradient_boosting",
    "tabpfn",
    "kronos",
    "foundation_ensemble",
)


def collect_candidates(
    symbol: str,
    horizon: int,
    *,
    reference_price: Optional[float] = None,
    bundles_dir: Path = BUNDLES_DIR,
    benchmark_artifact: Path = BENCHMARK_ARTIFACT,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> List[Candidate]:
    """
    Every model with a scorecard for ``symbol`` at ``horizon``, deduplicated.

    Where two sources describe the same model, the walk-forward benchmark wins:
    it scored the model over several folds against the same folds every other
    model saw, which a single chronological holdout did not.
    """
    ordered: List[Candidate] = []
    ordered.extend(_benchmark_candidates(symbol, horizon, benchmark_artifact))
    ordered.extend(_bundle_candidates(symbol, horizon, reference_price, bundles_dir))
    if horizon == NEXT_BAR_HORIZON:
        ordered.extend(direction_report_candidates(symbol, DIRECTION_REPORT_MODELS, report_dir))

    seen: Dict[str, Candidate] = {}
    for candidate in ordered:
        if candidate.model_type not in seen:
            seen[candidate.model_type] = candidate
    return list(seen.values())


@dataclass
class BestModels:
    """Both winners for one symbol and horizon, with the tables behind them."""

    symbol: str
    horizon: int
    price: Selection
    direction: Selection
    candidates: List[Candidate]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "price": self.price.as_dict(),
            "direction": self.direction.as_dict(),
            "skill_enforcement": skill_enforcement_enabled(),
        }


def select_best_models(
    symbol: str,
    horizon: int,
    *,
    reference_price: Optional[float] = None,
    bundles_dir: Path = BUNDLES_DIR,
    benchmark_artifact: Path = BENCHMARK_ARTIFACT,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> BestModels:
    """
    The best price model and the best direction model for one symbol/horizon.

    ``reference_price`` is the close the return-space bundle errors are scaled
    by. Pass the current price: it is the level those forecasts are made from,
    so it is the level their dollar error would be incurred at.
    """
    symbol = symbol.upper().strip()
    candidates = collect_candidates(
        symbol,
        horizon,
        reference_price=reference_price,
        bundles_dir=bundles_dir,
        benchmark_artifact=benchmark_artifact,
        report_dir=report_dir,
    )
    return BestModels(
        symbol=symbol,
        horizon=horizon,
        price=rank_candidates(candidates, "price"),
        direction=rank_candidates(candidates, "direction"),
        candidates=candidates,
    )
