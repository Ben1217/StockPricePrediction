"""
The A9 leakage checklist, executed rather than asserted.

Addendum A is blunt about this: "Each of the following must be individually
verified and the verification recorded. Any one of them silently violated
invalidates the whole evaluation." A checklist that always returns "pass" is
worse than no checklist -- it converts an unexamined assumption into a claim of
having examined it. So this module *runs* the checks it can run, and where no
code can settle the question it records an explicit **attestation** instead of a
pass, so the distinction survives into the report.

The central instrument: as-of invariance
----------------------------------------
Most leakage reduces to one question -- does the value computed for bar ``t``
change when you delete every bar after ``t``? If it does, the value used data it
could not have had. :func:`as_of_invariance` answers that directly: it rebuilds
the quantity on a truncated prefix and compares against the full-series build at
the same timestamp. A centred rolling window, a full-sample z-score, a reversed
series, a global ``fit`` -- all of them show up as a difference, and none of them
can hide behind a plausible-looking output.

That test is what makes checks 1, 2, 3 and 8 real rather than decorative.

Status vocabulary
-----------------
``pass``            code ran and the property held
``fail``            code ran and the property was violated
``attested``        no code can decide it; a human asserted it, and who, and how
``not_applicable``  the inputs for this check were not supplied, with a reason

``all_passed`` is True only when nothing failed. An ``attested`` result is not a
pass and is counted separately, because a report whose ten green ticks are all
attestations has verified nothing.

Public API:
    as_of_invariance(builder, frame, probe_positions, ...) -> dict
    check_01_rolling_features_only(...) -> CheckResult
    ... check_10_frozen_snapshot(...) -> CheckResult
    run_leakage_checklist(**inputs) -> LeakageReport
    record_leakage_report(report, path) -> Path
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

PASS = "pass"
FAIL = "fail"
NOT_APPLICABLE = "not_applicable"
ATTESTED = "attested"

#: The ten items, in the order Addendum A lists them. The report always carries
#: all ten so a missing check cannot be mistaken for a passing one.
CHECK_NAMES: Tuple[str, ...] = (
    "rolling features only",
    "no global scaling",
    "support/resistance as-of",
    "corporate-action adjustment",
    "purge and embargo applied",
    "HAC or non-overlapping windows",
    "calibration and weights fitted on validation folds only",
    "timestamp alignment",
    "universe selection independent of outcomes",
    "frozen data snapshot",
)

#: Fitting on any of these is leakage under A9 item 7 -- ensemble weights fitted
#: on the test window are called out by name in the Addendum.
_FORBIDDEN_FIT_SCOPES = frozenset({"test", "full_sample", "full", "all"})

#: Scopes a fit record may legitimately declare. Anything else cannot be
#: verified automatically and is reported as attested rather than passed.
_RECOGNISED_SCOPES = _FORBIDDEN_FIT_SCOPES | {"validation", "train", "inner_train"}

#: A9 item 4, verbatim enough to be quoted in the report.
_BACK_ADJUSTMENT_LIMITATION = (
    "Back-adjusted prices are a look-ahead trap: a split- and dividend-adjusted "
    "close series as downloaded today embeds adjustment factors derived from "
    "corporate actions that had not yet occurred at the historical decision "
    "point. Unadjusted prices with as-of adjustment factors were not available "
    "from this source, so the limitation is stated rather than removed."
)


@dataclass(frozen=True)
class CheckResult:
    """One checklist item and the evidence behind its verdict."""

    check_id: int
    name: str
    status: str
    detail: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LeakageReport:
    """All ten results, plus the summary the report chapter quotes."""

    results: Tuple[CheckResult, ...]

    @property
    def all_passed(self) -> bool:
        """True only when nothing failed. Attestations do not make this True on their own."""
        return not any(result.status == FAIL for result in self.results)

    @property
    def n_failed(self) -> int:
        return sum(1 for result in self.results if result.status == FAIL)

    @property
    def n_attested(self) -> int:
        return sum(1 for result in self.results if result.status == ATTESTED)

    @property
    def n_verified(self) -> int:
        return sum(1 for result in self.results if result.status == PASS)

    @property
    def n_not_applicable(self) -> int:
        return sum(1 for result in self.results if result.status == NOT_APPLICABLE)

    def failures(self) -> List[CheckResult]:
        return [result for result in self.results if result.status == FAIL]

    def summary_table(self) -> List[Dict[str, Any]]:
        return [
            {"check": result.check_id, "name": result.name,
             "status": result.status, "detail": result.detail}
            for result in self.results
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "all_passed": self.all_passed,
            "n_verified_by_code": self.n_verified,
            "n_failed": self.n_failed,
            "n_attested": self.n_attested,
            "n_not_applicable": self.n_not_applicable,
            "caveat": (
                "an 'attested' status is a human assertion, not a code-verified "
                "property; only 'pass' was proven by execution"
            ),
            "results": [result.to_dict() for result in self.results],
        }


# ---------------------------------------------------------------------------
# The central instrument
# ---------------------------------------------------------------------------


def _coerce_frame(value: Any) -> Optional[pd.DataFrame]:
    """Accept a DataFrame, a Series, or a dataset object exposing ``.features``."""
    if isinstance(value, pd.DataFrame):
        return value
    if isinstance(value, pd.Series):
        return value.to_frame()
    features = getattr(value, "features", None)
    if isinstance(features, pd.DataFrame):
        return features
    return None


def as_of_invariance(
    builder: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    *,
    probe_positions: Sequence[int],
    columns: Optional[Sequence[str]] = None,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> Dict[str, Any]:
    """
    Does the value at bar ``t`` change when the bars after ``t`` are deleted?

    For each probe position ``t`` the builder is re-run on ``frame.iloc[:t+1]``
    and its **last row** is compared against the full-series build at the same
    timestamp. Any construct that reads forward -- a centred window, a
    full-sample standardisation, a reversed series, a global fit -- produces a
    difference here and is localised to the offending column and bar.

    A column that is entirely NaN in the truncated build is reported as
    ``insufficient_history`` rather than pass or fail: a 200-bar moving average
    at bar 50 is undefined, not leaky, and conflating the two would either hide
    real failures or manufacture false ones.
    """
    full = _coerce_frame(builder(frame))
    if full is None:
        raise TypeError(
            "builder must return a DataFrame, a Series, or an object with a "
            "'.features' DataFrame"
        )

    candidates = list(columns) if columns is not None else list(full.columns)
    missing = [column for column in candidates if column not in full.columns]
    if missing:
        raise ValueError(f"columns not produced by the builder: {missing}")

    worst: Dict[str, Dict[str, Any]] = {
        column: {"max_abs_diff": 0.0, "max_rel_diff": 0.0, "worst_probe": None,
                 "compared": 0, "insufficient_history": 0,
                 "forward_filled_by_full_build": 0}
        for column in candidates
    }
    probes_run, probes_skipped = 0, []
    rows_trimmed: List[Dict[str, Any]] = []

    for position in probe_positions:
        position = int(position)
        if position < 0 or position >= len(frame):
            probes_skipped.append({"position": position, "reason": "out of range"})
            continue

        timestamp = frame.index[position]
        try:
            truncated = _coerce_frame(builder(frame.iloc[: position + 1]))
        except Exception as exc:  # noqa: BLE001 - one short prefix must not kill the sweep
            probes_skipped.append({"position": position, "reason": f"builder raised: {exc}"})
            continue

        if truncated is None or truncated.empty:
            probes_skipped.append({"position": position, "reason": "empty truncated build"})
            continue

        # The newest row the truncated build produced is the as-of value. It is
        # NOT necessarily the row at ``timestamp``: a builder whose label reads
        # h bars forward trims its own tail, so from a prefix ending at bar t it
        # emits rows only up to t-h. Demanding an exact match there silently
        # skipped every probe and left the check reporting "not compared" for
        # every column -- a checklist that ran nothing while looking like it had.
        # What must hold is weaker and still sufficient: whatever the truncated
        # build produced for some timestamp T <= t, computed from bars <= t
        # alone, must equal what the full build produced for that same T.
        as_of_timestamp = truncated.index[-1]
        if as_of_timestamp > timestamp:
            probes_skipped.append(
                {"position": position,
                 "reason": f"truncated build emitted a row at {as_of_timestamp}, "
                           f"after the prefix end {timestamp}"}
            )
            continue
        if as_of_timestamp not in full.index:
            probes_skipped.append(
                {"position": position,
                 "reason": f"as-of row {as_of_timestamp} is absent from the full build"}
            )
            continue
        if as_of_timestamp != timestamp:
            rows_trimmed.append(
                {"position": position, "prefix_end": str(timestamp),
                 "as_of_row": str(as_of_timestamp)}
            )

        probes_run += 1
        as_of_row = truncated.iloc[-1]
        full_row = full.loc[as_of_timestamp]
        if isinstance(full_row, pd.DataFrame):  # duplicate timestamps
            full_row = full_row.iloc[-1]

        for column in candidates:
            if column not in truncated.columns:
                continue
            a, b = as_of_row.get(column), full_row.get(column)
            if a is None or b is None:
                continue
            a, b = float(a) if pd.notna(a) else np.nan, float(b) if pd.notna(b) else np.nan
            record = worst[column]
            if np.isnan(a) and np.isnan(b):
                # Neither build could compute it -- genuinely too little history.
                continue
            if np.isnan(a):
                # The full-series build produced a value the as-of build could
                # NOT compute from the same prefix. That asymmetry is only
                # possible if the full build read bars after this timestamp, so
                # it is the signature of a forward-looking window -- exactly
                # what a centred rolling window looks like, whose trailing rows
                # are NaN as-of but filled once future bars exist. Counting it
                # as "insufficient history" let centred windows pass silently,
                # which is the single most important leak this tool must catch.
                record["compared"] += 1
                record["forward_filled_by_full_build"] += 1
                record["max_abs_diff"] = float("inf")
                record["max_rel_diff"] = float("inf")
                record["worst_probe"] = str(as_of_timestamp)
                continue
            if np.isnan(b):
                # The as-of build has a value the full build lost (usually a row
                # dropped by a later dropna). Not leakage; not comparable.
                continue

            record["compared"] += 1
            absolute = abs(a - b)
            relative = absolute / max(abs(b), 1e-30)
            if absolute > record["max_abs_diff"]:
                record["max_abs_diff"] = absolute
                record["max_rel_diff"] = relative
                record["worst_probe"] = str(as_of_timestamp)

    per_column: Dict[str, Dict[str, Any]] = {}
    leaky: List[str] = []
    for column, record in worst.items():
        if record["forward_filled_by_full_build"]:
            verdict = FAIL
            leaky.append(column)
            per_column[column] = {**record, "verdict": verdict}
            continue
        if record["compared"] == 0:
            verdict = "insufficient_history" if record["insufficient_history"] else "not_compared"
        elif record["max_abs_diff"] <= atol or record["max_rel_diff"] <= rtol:
            verdict = PASS
        else:
            verdict = FAIL
            leaky.append(column)
        per_column[column] = {**record, "verdict": verdict}

    return {
        "probes_requested": len(list(probe_positions)),
        "probes_run": probes_run,
        "probes_skipped": probes_skipped,
        "rows_trimmed_by_builder": rows_trimmed,
        "n_columns": len(candidates),
        "leaky_columns": leaky,
        "invariant": not leaky and probes_run > 0,
        "per_column": per_column,
        "rtol": rtol,
        "atol": atol,
    }


def _numeric_signature(value: Any) -> List[float]:
    """Flatten a nested dict/list/scalar into a comparable numeric vector."""
    out: List[float] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node, key=str):
                walk(node[key])
        elif isinstance(node, (list, tuple, np.ndarray)):
            for item in node:
                walk(item)
        elif isinstance(node, (int, float, np.integer, np.floating)) and not isinstance(node, bool):
            out.append(float(node))

    walk(value)
    return out


# ---------------------------------------------------------------------------
# The ten checks
# ---------------------------------------------------------------------------


def check_01_rolling_features_only(
    builder: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    probe_positions: Sequence[int],
    *,
    columns: Optional[Sequence[str]] = None,
) -> CheckResult:
    """A9.1 -- no feature may use data at or after its own timestamp."""
    report = as_of_invariance(builder, frame, probe_positions=probe_positions, columns=columns)
    if report["probes_run"] == 0:
        return CheckResult(1, CHECK_NAMES[0], NOT_APPLICABLE,
                           "no probe produced a comparable row", report)
    if report["leaky_columns"]:
        return CheckResult(
            1, CHECK_NAMES[0], FAIL,
            f"{len(report['leaky_columns'])} feature(s) changed when future bars were "
            f"removed: {', '.join(report['leaky_columns'][:8])}",
            report,
        )
    return CheckResult(
        1, CHECK_NAMES[0], PASS,
        f"{report['n_columns']} features were identical across {report['probes_run']} "
        f"as-of rebuilds",
        report,
    )


def check_02_no_global_scaling(
    builder: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    probe_positions: Sequence[int],
    *,
    columns: Optional[Sequence[str]] = None,
    tolerance: float = 1e-9,
) -> CheckResult:
    """
    A9.2 -- no z-score, min-max or standardisation over the full sample.

    Two independent signals. The as-of rebuild catches a scaler whose statistics
    move when the sample is truncated. The direct scan catches the tell-tale
    fingerprint: a column standardised over the whole sample has mean exactly 0
    and standard deviation exactly 1 over that sample, which a correctly
    *rolling* z-score essentially never does.
    """
    report = as_of_invariance(builder, frame, probe_positions=probe_positions, columns=columns)
    full = _coerce_frame(builder(frame))
    suspects: List[str] = []
    if full is not None:
        for column in (columns if columns is not None else full.columns):
            series = pd.to_numeric(full[column], errors="coerce").dropna()
            if len(series) < 30:
                continue
            centred = abs(float(series.mean())) < tolerance
            unit_scale = abs(float(series.std(ddof=0)) - 1.0) < tolerance
            if centred and unit_scale:
                suspects.append(str(column))

    evidence = {**report, "globally_standardised_suspects": suspects}
    if report["leaky_columns"] or suspects:
        parts = []
        if report["leaky_columns"]:
            parts.append(f"{len(report['leaky_columns'])} column(s) not as-of invariant")
        if suspects:
            parts.append(
                f"{len(suspects)} column(s) are exactly standardised over the full "
                f"sample ({', '.join(suspects[:6])})"
            )
        return CheckResult(2, CHECK_NAMES[1], FAIL, "; ".join(parts), evidence)
    if report["probes_run"] == 0:
        return CheckResult(2, CHECK_NAMES[1], NOT_APPLICABLE,
                           "no probe produced a comparable row", evidence)
    return CheckResult(
        2, CHECK_NAMES[1], PASS,
        "no full-sample standardisation detected and all scaling was as-of invariant",
        evidence,
    )


def check_03_support_resistance_as_of(
    level_fn: Callable[[pd.DataFrame], Any],
    frame: pd.DataFrame,
    probe_positions: Sequence[int],
    *,
    rtol: float = 1e-9,
) -> CheckResult:
    """
    A9.3 -- levels must be recomputed from a backward window at each bar.

    ``level_fn`` returns a nested structure of levels, so the comparison is on
    the flattened numeric signature rather than on a frame. The question is the
    same one as everywhere else: does the answer for bar ``t`` depend on bars
    after ``t``?

    A failure here does not necessarily mean the level function is broken -- it
    means the function is **not safe to call once on a full history and reuse**.
    ``src/features/support_resistance.py`` is the live example: ``_find_pivots``
    uses a *centred* window, so a pivot at bar ``i`` is only confirmable at
    ``i + window``, and ``detect_support_resistance`` slices ``df.iloc[-100:]``
    off whatever frame it is handed. Called once on the full series and applied
    to an earlier decision date, both constructs read the future. The detail
    below names which of the two produced the difference.
    """
    try:
        full_signature = _numeric_signature(level_fn(frame))
    except Exception as exc:  # noqa: BLE001 - report rather than abort the checklist
        return CheckResult(3, CHECK_NAMES[2], NOT_APPLICABLE,
                           f"level function raised on the full frame: {exc}", {})

    differences: List[Dict[str, Any]] = []
    probes_run = 0
    for position in probe_positions:
        position = int(position)
        if position < 0 or position >= len(frame):
            continue
        try:
            as_of_signature = _numeric_signature(level_fn(frame.iloc[: position + 1]))
        except Exception as exc:  # noqa: BLE001 - a short prefix may legitimately fail
            differences.append({"position": position, "reason": f"raised: {exc}"})
            continue
        probes_run += 1
        if len(as_of_signature) != len(full_signature) or not np.allclose(
            as_of_signature, full_signature, rtol=rtol, atol=1e-12
        ):
            differences.append({
                "position": position,
                "timestamp": str(frame.index[position]),
                "as_of_values": as_of_signature[:8],
                "full_series_values": full_signature[:8],
            })

    evidence = {
        "probes_run": probes_run,
        "n_differing": len(differences),
        "differences": differences[:10],
        "known_constructs": [
            "_find_pivots uses a centred window (highs[i-w : i+w+1]), so a pivot at "
            "bar i is only confirmable at bar i+w",
            "detect_support_resistance slices df.iloc[-100:] off whatever frame it is "
            "given, so the window follows the end of the frame",
        ],
    }
    if probes_run == 0:
        return CheckResult(3, CHECK_NAMES[2], NOT_APPLICABLE,
                           "no probe position could be evaluated", evidence)
    if differences:
        return CheckResult(
            3, CHECK_NAMES[2], FAIL,
            f"levels differed from the full-series build at {len(differences)}/"
            f"{probes_run} probes: this function must be re-called per bar on a "
            f"backward window, never computed once over the whole history",
            evidence,
        )
    return CheckResult(3, CHECK_NAMES[2], PASS,
                       f"levels were as-of invariant across {probes_run} probes", evidence)


def check_04_corporate_action_adjustment(loader_meta: Mapping[str, Any]) -> CheckResult:
    """
    A9.4 -- back-adjusted series embed factors from corporate actions that had
    not yet happened at the decision point.

    ``pass`` requires positive evidence of unadjusted bars with as-of factors.
    Anything else is ``attested`` with the limitation text attached; it is never
    a silent pass, because a silent pass here is precisely the trap.
    """
    if not loader_meta:
        return CheckResult(4, CHECK_NAMES[3], NOT_APPLICABLE,
                           "no loader metadata supplied", {})

    point_in_time = bool(loader_meta.get("point_in_time_adjustment", False))
    back_adjusted = bool(
        loader_meta.get("adjusted", False)
        or loader_meta.get("auto_adjust", False)
        or loader_meta.get("dividend_adjusted", False)
    )
    evidence = {
        "loader_meta": dict(loader_meta),
        "point_in_time_adjustment": point_in_time,
        "back_adjusted": back_adjusted,
    }

    if point_in_time:
        return CheckResult(4, CHECK_NAMES[3], PASS,
                           "unadjusted bars with as-of adjustment factors were used", evidence)
    evidence["limitation"] = _BACK_ADJUSTMENT_LIMITATION
    return CheckResult(
        4, CHECK_NAMES[3], ATTESTED,
        "a back-adjusted series was used; the look-ahead limitation is recorded "
        "and must appear in the results-table caption",
        evidence,
    )


def check_05_purge_and_embargo(
    folds: Sequence[Any], *, horizon: int, embargo: Optional[int] = None
) -> CheckResult:
    """
    A9.5 / A4.2 -- verify the gap on **every** fold, not the configuration.

    The required separation is ``horizon + embargo`` bars between the last
    training row and the first test row: the purge removes the training rows
    whose labels resolve inside the test window, and the embargo adds the
    serial-correlation buffer on top.
    """
    if not folds:
        return CheckResult(5, CHECK_NAMES[4], NOT_APPLICABLE, "no folds supplied", {})

    horizon = int(horizon)
    embargo = horizon if embargo is None else int(embargo)
    required = horizon + embargo

    observed: List[Dict[str, Any]] = []
    worst_gap, worst_fold = None, None
    for index, fold in enumerate(folds, start=1):
        train = np.asarray(getattr(fold, "train_pos", fold[0] if isinstance(fold, tuple) else []))
        test = np.asarray(getattr(fold, "test_pos", fold[1] if isinstance(fold, tuple) else []))
        if train.size == 0 or test.size == 0:
            continue
        gap = int(test.min() - train.max() - 1)
        identifier = int(getattr(fold, "fold", index))
        observed.append({"fold": identifier, "gap_bars": gap, "required": required})
        if worst_gap is None or gap < worst_gap:
            worst_gap, worst_fold = gap, identifier

    evidence = {"horizon": horizon, "embargo": embargo, "required_gap_bars": required,
                "min_gap_bars": worst_gap, "worst_fold": worst_fold, "per_fold": observed}

    if not observed:
        return CheckResult(5, CHECK_NAMES[4], NOT_APPLICABLE,
                           "no fold carried both train and test positions", evidence)
    if worst_gap is None or worst_gap < required:
        return CheckResult(
            5, CHECK_NAMES[4], FAIL,
            f"fold {worst_fold} leaves only {worst_gap} bars between train and test; "
            f"{required} are required (purge {horizon} + embargo {embargo})",
            evidence,
        )
    return CheckResult(
        5, CHECK_NAMES[4], PASS,
        f"all {len(observed)} folds leave at least {worst_gap} bars "
        f"(>= {required} required)",
        evidence,
    )


def check_06_hac_or_non_overlapping(
    *, horizon: int, hac_lag: Optional[int] = None, non_overlapping: bool = False
) -> CheckResult:
    """
    A9.6 / A4.3 -- overlapping h-period returns need a HAC variance of at least
    lag ``h - 1``, or non-overlapping evaluation windows.
    """
    horizon = int(horizon)
    evidence = {"horizon": horizon, "hac_lag": hac_lag, "non_overlapping": non_overlapping,
                "required_lag": max(0, horizon - 1)}

    if horizon == 1:
        return CheckResult(6, CHECK_NAMES[5], PASS,
                           "horizon is 1, so consecutive returns do not overlap", evidence)
    if non_overlapping:
        return CheckResult(6, CHECK_NAMES[5], PASS,
                           "evaluation used non-overlapping windows", evidence)
    if hac_lag is None:
        return CheckResult(
            6, CHECK_NAMES[5], FAIL,
            f"horizon {horizon} overlaps but no HAC lag was applied and windows overlap",
            evidence,
        )
    if int(hac_lag) < horizon - 1:
        return CheckResult(
            6, CHECK_NAMES[5], FAIL,
            f"HAC lag {hac_lag} is below the required {horizon - 1} for horizon {horizon}",
            evidence,
        )
    return CheckResult(6, CHECK_NAMES[5], PASS,
                       f"HAC lag {hac_lag} >= required {horizon - 1}", evidence)


def check_07_fitted_on_validation_only(fit_records: Sequence[Mapping[str, Any]]) -> CheckResult:
    """
    A9.7 -- calibration mappings, ensemble weights and hyperparameters may be
    fitted on validation folds only, and refit within each fold.

    ``fit_records`` entries are ``{name, fitted_on, fold}``. Anything fitted on
    ``test`` or ``full_sample`` is leakage; ensemble weights fitted on the test
    window are called out by name in A9 item 7.
    """
    if not fit_records:
        return CheckResult(7, CHECK_NAMES[6], NOT_APPLICABLE, "no fit records supplied", {})

    forbidden = _FORBIDDEN_FIT_SCOPES
    offenders = [
        dict(record) for record in fit_records
        if str(record.get("fitted_on", "")).strip().lower() in forbidden
    ]
    unknown = [
        dict(record) for record in fit_records
        if str(record.get("fitted_on", "")).strip().lower() not in _RECOGNISED_SCOPES
    ]
    evidence = {"n_records": len(fit_records), "offenders": offenders,
                "unrecognised_scopes": unknown,
                "records": [dict(record) for record in fit_records]}

    if offenders:
        names = ", ".join(str(record.get("name", "?")) for record in offenders)
        return CheckResult(
            7, CHECK_NAMES[6], FAIL,
            f"{len(offenders)} component(s) were fitted on test or full-sample data: {names}",
            evidence,
        )
    if unknown:
        return CheckResult(
            7, CHECK_NAMES[6], ATTESTED,
            f"{len(unknown)} record(s) declared an unrecognised fitting scope; "
            f"they could not be verified automatically",
            evidence,
        )
    return CheckResult(7, CHECK_NAMES[6], PASS,
                       f"all {len(fit_records)} fitted components declared a "
                       f"train or validation scope", evidence)


def check_08_timestamp_alignment(
    feature_index: pd.Index,
    label_index: pd.Index,
    *,
    horizon: int,
    bar_index: Optional[pd.Index] = None,
    same_bar_columns: Optional[Sequence[str]] = None,
) -> CheckResult:
    """
    A9.8 -- a feature from bar ``t``'s close may only inform a decision at or
    after that close, and the label must resolve strictly later.

    When ``bar_index`` (the full trading calendar) is supplied, the positional
    gap is checked to be exactly ``horizon``; without it, only strict ordering
    can be verified, and the check says so rather than implying more.
    """
    feature_index = pd.Index(feature_index)
    label_index = pd.Index(label_index)
    if len(feature_index) != len(label_index):
        return CheckResult(
            8, CHECK_NAMES[7], FAIL,
            f"feature index has {len(feature_index)} rows but the label index has "
            f"{len(label_index)}; they cannot be aligned",
            {"n_features": len(feature_index), "n_labels": len(label_index)},
        )
    if len(feature_index) == 0:
        return CheckResult(8, CHECK_NAMES[7], NOT_APPLICABLE, "empty index", {})

    not_after = int(np.sum(label_index.to_numpy() <= feature_index.to_numpy()))
    same_bar = list(same_bar_columns or [])
    evidence: Dict[str, Any] = {
        "n_rows": len(feature_index),
        "labels_not_strictly_after_decision": not_after,
        "same_bar_columns": same_bar,
        "horizon": int(horizon),
        "positional_gap_checked": bar_index is not None,
    }

    wrong_gap = None
    if bar_index is not None:
        calendar = pd.Index(bar_index)
        positions = calendar.get_indexer(feature_index)
        label_positions = calendar.get_indexer(label_index)
        valid = (positions >= 0) & (label_positions >= 0)
        gaps = label_positions[valid] - positions[valid]
        wrong_gap = int(np.sum(gaps != int(horizon)))
        evidence["rows_with_wrong_positional_gap"] = wrong_gap
        evidence["n_rows_locatable_in_calendar"] = int(valid.sum())

    problems = []
    if not_after:
        problems.append(f"{not_after} label(s) resolve at or before their decision bar")
    if same_bar:
        problems.append(f"same-bar columns declared: {', '.join(same_bar[:6])}")
    if wrong_gap:
        problems.append(f"{wrong_gap} row(s) have a positional gap other than {horizon}")

    if problems:
        return CheckResult(8, CHECK_NAMES[7], FAIL, "; ".join(problems), evidence)
    detail = f"all {len(feature_index)} labels resolve strictly after their decision bar"
    if bar_index is not None:
        detail += f", exactly {horizon} bar(s) later"
    else:
        detail += " (no calendar supplied, so the exact gap was not verified)"
    return CheckResult(8, CHECK_NAMES[7], PASS, detail, evidence)


def check_09_universe_independent_of_outcomes(
    selection_rule: str, attested_by: str = ""
) -> CheckResult:
    """
    A9.9 -- no ticker included or excluded because of how the model performed.

    There is no code that can prove this: the evidence would be the *absence* of
    a decision, which leaves no trace in the data. So this check is an
    attestation by construction, and it records the rule and its author rather
    than pretending to have verified them.
    """
    rule = (selection_rule or "").strip()
    evidence = {"selection_rule": rule, "attested_by": attested_by,
                "verifiable_by_code": False}
    if not rule:
        return CheckResult(
            9, CHECK_NAMES[8], FAIL,
            "no universe selection rule was stated; an unstated rule cannot be "
            "shown to be outcome-independent",
            evidence,
        )
    return CheckResult(
        9, CHECK_NAMES[8], ATTESTED,
        f"selection rule attested{' by ' + attested_by if attested_by else ''}: {rule}",
        evidence,
    )


def check_10_frozen_snapshot(verify_report: Optional[Mapping[str, Any]]) -> CheckResult:
    """
    A9.10 -- evaluation runs against a versioned frozen snapshot, never a live
    API call, because vendors silently revise history.

    Consumes the dict from ``src.evaluation.snapshot.verify_snapshot``.
    """
    if not verify_report:
        return CheckResult(
            10, CHECK_NAMES[9], NOT_APPLICABLE,
            "no snapshot verification report supplied; the run cannot be shown to "
            "have used a frozen dataset",
            {},
        )
    ok = bool(verify_report.get("ok", False))
    evidence = dict(verify_report)
    if ok:
        return CheckResult(10, CHECK_NAMES[9], PASS,
                           "every frame re-hashed to its recorded content hash", evidence)
    return CheckResult(
        10, CHECK_NAMES[9], FAIL,
        "the snapshot did not verify: at least one frame's content hash moved, so "
        "the run is not reproducible from the recorded manifest",
        evidence,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_leakage_checklist(
    *,
    feature_builder: Optional[Callable[[pd.DataFrame], Any]] = None,
    frame: Optional[pd.DataFrame] = None,
    probe_positions: Optional[Sequence[int]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    level_fn: Optional[Callable[[pd.DataFrame], Any]] = None,
    loader_meta: Optional[Mapping[str, Any]] = None,
    folds: Optional[Sequence[Any]] = None,
    horizon: int = 1,
    embargo: Optional[int] = None,
    hac_lag: Optional[int] = None,
    non_overlapping: bool = False,
    fit_records: Optional[Sequence[Mapping[str, Any]]] = None,
    feature_index: Optional[pd.Index] = None,
    label_index: Optional[pd.Index] = None,
    bar_index: Optional[pd.Index] = None,
    same_bar_columns: Optional[Sequence[str]] = None,
    selection_rule: str = "",
    attested_by: str = "",
    snapshot_verify_report: Optional[Mapping[str, Any]] = None,
) -> LeakageReport:
    """
    Run every check for which inputs were supplied and record the rest.

    The returned report **always** carries exactly ten results in order. A check
    with no inputs is ``not_applicable`` with the reason stated -- it is never
    omitted, because a ten-row table with nine rows in it invites the reader to
    assume the tenth passed.
    """
    results: List[CheckResult] = []
    have_builder = feature_builder is not None and frame is not None and probe_positions is not None

    if have_builder:
        try:
            results.append(check_01_rolling_features_only(
                feature_builder, frame, probe_positions, columns=feature_columns))
        except Exception as exc:  # noqa: BLE001 - a failed check is a result, not a crash
            logger.error("check 1 raised: %s", exc, exc_info=True)
            results.append(CheckResult(1, CHECK_NAMES[0], FAIL, f"check raised: {exc}", {}))
        try:
            results.append(check_02_no_global_scaling(
                feature_builder, frame, probe_positions, columns=feature_columns))
        except Exception as exc:  # noqa: BLE001
            logger.error("check 2 raised: %s", exc, exc_info=True)
            results.append(CheckResult(2, CHECK_NAMES[1], FAIL, f"check raised: {exc}", {}))
    else:
        reason = "no feature builder, frame and probe positions supplied"
        results.append(CheckResult(1, CHECK_NAMES[0], NOT_APPLICABLE, reason, {}))
        results.append(CheckResult(2, CHECK_NAMES[1], NOT_APPLICABLE, reason, {}))

    if level_fn is not None and frame is not None and probe_positions is not None:
        try:
            results.append(check_03_support_resistance_as_of(level_fn, frame, probe_positions))
        except Exception as exc:  # noqa: BLE001
            logger.error("check 3 raised: %s", exc, exc_info=True)
            results.append(CheckResult(3, CHECK_NAMES[2], FAIL, f"check raised: {exc}", {}))
    else:
        results.append(CheckResult(3, CHECK_NAMES[2], NOT_APPLICABLE,
                                   "no support/resistance level function supplied", {}))

    results.append(check_04_corporate_action_adjustment(loader_meta or {}))
    results.append(check_05_purge_and_embargo(folds or [], horizon=horizon, embargo=embargo))
    results.append(check_06_hac_or_non_overlapping(
        horizon=horizon, hac_lag=hac_lag, non_overlapping=non_overlapping))
    results.append(check_07_fitted_on_validation_only(fit_records or []))

    if feature_index is not None and label_index is not None:
        results.append(check_08_timestamp_alignment(
            feature_index, label_index, horizon=horizon,
            bar_index=bar_index, same_bar_columns=same_bar_columns))
    else:
        results.append(CheckResult(8, CHECK_NAMES[7], NOT_APPLICABLE,
                                   "no feature and label index supplied", {}))

    results.append(check_09_universe_independent_of_outcomes(selection_rule, attested_by))
    results.append(check_10_frozen_snapshot(snapshot_verify_report))

    results.sort(key=lambda result: result.check_id)
    report = LeakageReport(tuple(results))
    logger.info(
        "leakage checklist: %d verified, %d failed, %d attested, %d not applicable",
        report.n_verified, report.n_failed, report.n_attested, report.n_not_applicable,
    )
    for failure in report.failures():
        logger.error("LEAKAGE CHECK %d FAILED (%s): %s",
                     failure.check_id, failure.name, failure.detail)
    return report


def record_leakage_report(report: LeakageReport, path: Path | str) -> Path:
    """
    Write the JSON record A9 demands -- "the verification recorded".

    The evaluation is only defensible if this artifact exists alongside the
    results table, so the harness writes it on every run, pass or fail.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2, default=str)
    logger.info("Recorded the leakage checklist to %s", destination)
    return destination
