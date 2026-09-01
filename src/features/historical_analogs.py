"""
Historical analogs: "has this chart looked like this before, and what happened?"

The direction model asks whether a *feature vector* implies an up move. This
module asks a narrower and more legible question: among the bars in this
symbol's own past, which ones looked most like today, and how often did the next
session rise after them? It is a k-nearest-neighbour read over a deliberately
small setup descriptor, and its answer is a frequency with a sample size and a
confidence interval attached - not a probability dressed up as one.

Why a separate, smaller feature set
-----------------------------------
Nearest-neighbour distance degrades as dimension grows: over the 46 columns the
direction dataset carries, every pair of rows is roughly equidistant and "most
similar" stops meaning anything. ``ANALOG_FEATURE_COLUMNS`` is ten columns
spanning the six things a person means by "a similar setup" - trend position,
momentum, volatility regime, participation, position in the channel, and swing
structure - which is low enough for distance to still rank rows the way a reader
would.

Causality
---------
Two separate rules, both enforced here rather than left to the caller:

* **Standardisation is trailing.** Columns are z-scored against an expanding
  window ending at the row itself, so a 2019 row is never scaled by a mean that
  includes 2024. A single whole-series ``fit`` would leak the future into every
  earlier row's distance.
* **Neighbours are strictly older, and their labels are resolved.** A neighbour
  of row ``t`` must sit at ``j <= t - 1 - horizon``, so the forward return being
  averaged had already printed by the time the query bar closed.

Public API:
    analog_matches(features, forward_return, ...) -> dict
    analog_up_rate_series(features, forward_return, ...) -> Series
    ANALOG_FEATURE_COLUMNS
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..models.direction_metrics import wilson_interval
from ..utils.logger import get_logger

logger = get_logger(__name__)

# The setup descriptor. Ten columns, one or two per thing a reader means by
# "similar": where price sits against its means, how it is moving, how volatile
# and how heavily traded it is, where it sits in its range, and what the swing
# structure is doing.
ANALOG_FEATURE_COLUMNS: List[str] = [
    "Close_SMA20_Ratio",
    "Close_SMA50_Ratio",
    "RSI_14",
    "MACD_Norm",
    "Return_20d",
    "Volatility_Ratio",
    "Volume_Zscore",
    "Donchian_Position_20",
    "Trend_R2_20",
    "PA_Structure_20",
]

# Neighbours per query. Large enough that the up-rate is not one coin flip,
# small enough that the matches are still recognisably the same setup. The
# binomial interval reported beside it is what says whether 40 was enough.
DEFAULT_K = 40

# Rows of history required before an analog read is emitted at all. Below this
# the "nearest" neighbours are simply the only neighbours.
MIN_ANALOG_HISTORY = 250

# Query rows per distance-matrix chunk. Keeps peak memory at roughly
# chunk x n_reference floats rather than n x n.
_CHUNK_ROWS = 256


def _expanding_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column against an expanding trailing window.

    Row ``t`` is scaled by the mean and standard deviation of rows ``0..t``,
    which are all knowable at ``t``. A zero or missing standard deviation maps
    the row to NaN rather than to a fabricated 0.
    """
    mean = frame.expanding(min_periods=2).mean()
    std = frame.expanding(min_periods=2).std()
    return (frame - mean) / std.where(std > 0)


def _prepare(
    features: pd.DataFrame,
    columns: Optional[Sequence[str]],
) -> tuple:
    """Resolve the descriptor columns and return their causally z-scored matrix."""
    requested = list(columns or ANALOG_FEATURE_COLUMNS)
    resolved = [col for col in requested if col in features.columns]
    if not resolved:
        raise ValueError(
            "No analog descriptor column is present. Expected some of "
            f"{requested}; the frame has {list(features.columns)[:12]}..."
        )
    if len(resolved) < len(requested):
        logger.warning(
            "Analog descriptor missing %d column(s): %s",
            len(requested) - len(resolved),
            [col for col in requested if col not in resolved],
        )
    scaled = _expanding_zscore(features[resolved].astype(float))
    return resolved, scaled


def _neighbour_indices(
    query: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> tuple:
    """
    Indices and distances of the ``k`` nearest reference rows for one query row.

    Reference rows containing NaN are excluded rather than imputed - a row whose
    descriptor is half missing is not a near miss, it is not a comparable at all.
    """
    usable = np.isfinite(reference).all(axis=1)
    if not usable.any():
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    candidates = np.flatnonzero(usable)
    deltas = reference[candidates] - query
    distances = np.sqrt(np.einsum("ij,ij->i", deltas, deltas))
    take = min(k, distances.size)
    nearest = np.argpartition(distances, take - 1)[:take]
    nearest = nearest[np.argsort(distances[nearest])]
    return candidates[nearest], distances[nearest]


def analog_matches(
    features: pd.DataFrame,
    forward_return: pd.Series,
    *,
    query: Optional[pd.Series] = None,
    columns: Optional[Sequence[str]] = None,
    k: int = DEFAULT_K,
    horizon: int = 1,
    min_history: int = MIN_ANALOG_HISTORY,
    max_examples: int = 8,
) -> Dict[str, Any]:
    """
    The ``k`` historical setups most similar to ``query``, and what followed them.

    Parameters
    ----------
    features : DataFrame
        Descriptor columns for every labelled bar, indexed by decision date.
    forward_return : Series
        The realised return that resolved after each of those bars, on the same
        index as ``features``.
    query : Series, optional
        The row to match. Defaults to the last row of ``features`` - but when
        the caller has an *unlabelled* latest bar (the one anyone actually wants
        a read for), it should pass that row here, appended to ``features`` so
        the trailing z-score scales it consistently.
    k : int
        Neighbours to retrieve.
    horizon : int
        Bars the label resolves over; neighbours are held back that many bars so
        their outcome had printed before the query bar closed.

    Returns
    -------
    dict
        ``available`` false with a ``reason`` when the history is too short,
        otherwise the up-rate with its Wilson interval, the forward-return
        distribution of the matches, and a handful of dated examples.
    """
    if len(features) < min_history:
        return {
            "available": False,
            "reason": f"{len(features)} rows of history, below the {min_history}-row floor "
                      "for a nearest-neighbour read",
            "n_matches": 0,
        }

    resolved, scaled = _prepare(features, columns)

    if query is None:
        query_position = len(scaled) - 1
        query_vector = scaled.iloc[query_position].to_numpy(dtype=float)
        query_label = scaled.index[query_position]
        # The last labelled row can only look back at rows before it.
        cutoff = query_position - horizon
    else:
        # An unlabelled query row: scale it against the same expanding
        # statistics by appending it before standardising.
        appended = pd.concat([features[resolved].astype(float), query[resolved].to_frame().T])
        scaled_with_query = _expanding_zscore(appended)
        query_vector = scaled_with_query.iloc[-1].to_numpy(dtype=float)
        query_label = query.name
        # Every labelled row is older than the unlabelled query, and every one
        # of their outcomes has printed.
        cutoff = len(scaled)

    if not np.isfinite(query_vector).all():
        return {
            "available": False,
            "reason": "the current bar has an incomplete setup descriptor",
            "n_matches": 0,
        }

    reference = scaled.iloc[:cutoff].to_numpy(dtype=float) if cutoff > 0 else np.empty((0, len(resolved)))
    if reference.shape[0] < k:
        return {
            "available": False,
            "reason": f"only {reference.shape[0]} comparable historical bars, fewer than k={k}",
            "n_matches": int(reference.shape[0]),
        }

    positions, distances = _neighbour_indices(query_vector, reference, k)
    if positions.size == 0:
        return {"available": False, "reason": "no historical bar has a complete descriptor", "n_matches": 0}

    dates = features.index[positions]
    outcomes = pd.to_numeric(forward_return.reindex(dates), errors="coerce")
    valid = outcomes.notna().to_numpy()
    outcomes = outcomes[valid]
    dates = dates[valid]
    distances = distances[valid]
    if outcomes.empty:
        return {"available": False, "reason": "matched bars have no resolved outcome", "n_matches": 0}

    ups = int((outcomes > 0).sum())
    trials = int(outcomes.size)
    up_rate = ups / trials

    examples = [
        {
            "date": str(pd.Timestamp(date).date()),
            "distance": round(float(distance), 4),
            "forward_return": round(float(value), 6),
        }
        for date, distance, value in list(zip(dates, distances, outcomes.to_numpy()))[:max_examples]
    ]

    return {
        "available": True,
        "as_of": str(pd.Timestamp(query_label).date()) if query_label is not None else None,
        "n_matches": trials,
        "k": int(k),
        "up_rate": round(up_rate, 6),
        "up_rate_ci": [round(bound, 6) for bound in wilson_interval(ups, trials)],
        # The unconditional rate over the same reference window. An analog
        # up-rate of 58% is only interesting against the 53% the symbol posts
        # anyway, so the comparison ships with the number.
        "reference_up_rate": round(
            float((pd.to_numeric(forward_return.iloc[:cutoff], errors="coerce") > 0).mean()), 6
        ) if cutoff > 0 else None,
        "mean_forward_return": round(float(outcomes.mean()), 6),
        "median_forward_return": round(float(outcomes.median()), 6),
        "forward_return_p10": round(float(outcomes.quantile(0.10)), 6),
        "forward_return_p90": round(float(outcomes.quantile(0.90)), 6),
        "mean_distance": round(float(np.mean(distances)), 4),
        "descriptor_columns": resolved,
        "examples": examples,
    }


def analog_up_rate_series(
    features: pd.DataFrame,
    forward_return: pd.Series,
    *,
    columns: Optional[Sequence[str]] = None,
    k: int = DEFAULT_K,
    horizon: int = 1,
    min_history: int = MIN_ANALOG_HISTORY,
) -> pd.Series:
    """
    The analog up-rate at every bar, computed causally, for use as a feature.

    Row ``t`` holds the share of the ``k`` most similar *earlier* bars that were
    followed by an up move. Rows before ``min_history`` are NaN: an analog read
    over 80 bars of history is a description of the only bars available, not of
    the similar ones.

    Cost is one distance matrix per chunk of query rows, so the whole series
    costs ``O(n^2 d)`` - a few hundred milliseconds at the ~2500 bars a decade
    of daily data holds.
    """
    resolved, scaled = _prepare(features, columns)
    matrix = scaled.to_numpy(dtype=float)
    outcomes = pd.to_numeric(forward_return.reindex(features.index), errors="coerce").to_numpy(dtype=float)
    n = matrix.shape[0]
    result = np.full(n, np.nan)

    complete = np.isfinite(matrix).all(axis=1) & np.isfinite(outcomes)

    for start in range(min_history, n, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, n)
        block = matrix[start:stop]
        block_ok = np.isfinite(block).all(axis=1)
        if not block_ok.any():
            continue
        # Squared Euclidean distance from every query row in the block to every
        # reference row, masked afterwards to the rows each query may see.
        distances = np.sqrt(
            np.maximum(
                (block ** 2).sum(axis=1, keepdims=True)
                - 2.0 * block @ matrix.T
                + (matrix ** 2).sum(axis=1)[None, :],
                0.0,
            )
        )
        for offset in range(stop - start):
            row = start + offset
            if not block_ok[offset]:
                continue
            cutoff = row - horizon
            if cutoff < k:
                continue
            eligible = complete[:cutoff]
            if eligible.sum() < k:
                continue
            candidate_positions = np.flatnonzero(eligible)
            candidate_distances = distances[offset, candidate_positions]
            nearest = np.argpartition(candidate_distances, k - 1)[:k]
            matched = candidate_positions[nearest]
            result[row] = float((outcomes[matched] > 0).mean())

    return pd.Series(result, index=features.index, name="analog_up_rate")
