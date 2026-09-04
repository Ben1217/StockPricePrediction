"""
Splitting protocol for the Addendum A evaluation package (Requirement A4).

A walk-forward split that only moves forward in time is necessary but nowhere
near sufficient once the target is an h-step-ahead return. Two leaks survive a
naive chronological split, and both of them inflate out-of-sample scores:

1. **Label overlap (the purge).** The label attached to training row ``t`` is
   ``ln(P(t+h) / P(t))``. It is not knowable at ``t``; it resolves at ``t+h``.
   So the last ``h`` rows of any training block carry labels built from bars
   that live inside the test window. Training on them is training on the test
   set. Those ``h`` rows are dropped -- purged -- not merely gapped over.

2. **Serial-correlation bleed (the embargo).** Even after purging, the bar
   immediately before the test window and the bar immediately after it share
   volatility state, a stale news shock, a not-yet-mean-reverted spread. A
   model fit right up to the boundary can memorise that state and score on its
   own persistence. An additional ``embargo`` bars are therefore vacated after
   the purge. The default embargo is ``horizon``, and passing an embargo below
   the horizon is an error: it would leave the purge doing double duty and the
   protocol would silently be weaker than it claims.

   The layout, per fold, is::

       [------------- train -------------][ purge h ][ embargo e ][--- test ---]
                                          ^                       ^
                                     train_end               test_start

   so ``train_end = test_start - embargo - horizon``, and the purged and
   embargoed positions are reported *separately* so a reviewer can audit the
   gap instead of trusting a single number.

The third trap this module handles is arithmetic rather than temporal.
Overlapping h-period returns are autocorrelated **by construction** -- 252
overlapping 20-day returns are not 252 independent observations, they are
closer to 20 -- so any t-statistic, confidence interval or power calculation
computed on ``n`` of them using ``sqrt(n)`` is overstated.
:func:`effective_sample_size` gives the honest denominator and
:func:`non_overlapping_positions` gives the alternative: throw away ``h - 1``
of every ``h`` observations and keep a genuinely independent sample.

Finally, a model selected, tuned and thresholded on the same tickers it is
reported on has been fit to those tickers even if no single fold leaked.
:func:`held_out_ticker_split` reserves a sector-stratified slice of the
universe that the research loop never sees, so there is one number left at the
end that nobody optimised against.

Nothing here touches data. Every function deals in integer row *positions* and
plain Python containers, which is what makes the protocol testable in
microseconds and reusable across price, direction and quantile evaluation.

Public API:
    Fold(fold, train_pos, test_pos, purged_pos, embargo_pos)
    purged_walk_forward_splits(n_rows, *, horizon, test_size=63, n_splits=4,
                               min_train=252, embargo=None) -> List[Fold]
    effective_sample_size(n, horizon) -> float
    non_overlapping_positions(positions, horizon) -> np.ndarray
    non_overlapping_folds(folds, horizon) -> List[Fold]
    HeldOutSplit(in_universe, held_out, by_sector, holdout_fraction_actual)
    held_out_ticker_split(tickers, sectors, *, holdout_fraction=0.2, seed=42) -> HeldOutSplit
    describe_split(folds, index) -> List[Dict[str, Any]]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Bucket for tickers whose sector is missing, None or blank. They are still
#: stratified -- "unknown" is a sector like any other for splitting purposes,
#: and silently dropping them would shrink the universe without saying so.
UNKNOWN_SECTOR = "UNKNOWN"


def _as_positions(values: Any, name: str) -> np.ndarray:
    """Coerce to a flat int64 position array, rejecting negatives."""
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size and int(array.min()) < 0:
        raise ValueError(f"{name} contains negative row positions: min={int(array.min())}")
    return array


@dataclass(frozen=True, eq=False)
class Fold:
    """
    One walk-forward fold, expressed purely as integer row positions.

    ``train_pos``, ``purged_pos``, ``embargo_pos`` and ``test_pos`` partition a
    contiguous stretch of the series, in that chronological order. The purge and
    embargo blocks are kept rather than discarded so that :func:`describe_split`
    can put the exact gap into the results artifact -- a reader must be able to
    check the protocol without rerunning it.

    ``eq=False`` on purpose: the fields are numpy arrays, and the ``__eq__`` /
    ``__hash__`` a frozen dataclass would synthesise raise "truth value of an
    array is ambiguous" the moment anyone compares two folds. Compare the arrays
    with ``np.array_equal`` instead.
    """

    fold: int
    train_pos: np.ndarray
    test_pos: np.ndarray
    purged_pos: np.ndarray
    embargo_pos: np.ndarray

    def __post_init__(self) -> None:
        if int(self.fold) < 1:
            raise ValueError(f"fold numbers are 1-based, got {self.fold}")
        object.__setattr__(self, "fold", int(self.fold))
        for name in ("train_pos", "test_pos", "purged_pos", "embargo_pos"):
            object.__setattr__(self, name, _as_positions(getattr(self, name), name))

    @property
    def n_train(self) -> int:
        """Rows the model is fit on, after the purge has been taken out."""
        return int(self.train_pos.size)

    @property
    def n_test(self) -> int:
        """Rows the model is scored on."""
        return int(self.test_pos.size)


def purged_walk_forward_splits(
    n_rows: int,
    *,
    horizon: int,
    test_size: int = 63,
    n_splits: int = 4,
    min_train: int = 252,
    embargo: Optional[int] = None,
) -> List[Fold]:
    """
    Expanding-window folds with an explicit purge and an explicit embargo.

    The test blocks are contiguous, equal-length, non-overlapping and cover the
    *tail* of the series, so the most recent data is always scored. Fold 1 is
    the OLDEST test block; the training window grows to meet each subsequent
    one::

        fold 1:  [--- train ---][purge][embargo][test]
        fold 2:  [------ train ------][purge][embargo][test]
        fold 3:  [--------- train --------][purge][embargo][test]

    Parameters
    ----------
    n_rows : int
        Length of the (chronologically ordered) series.
    horizon : int
        Forecast horizon ``h`` in bars. The last ``h`` rows of each candidate
        training region are purged, because their labels resolve at or after
        the test window opens.
    test_size : int, default 63
        Rows per test block. 63 is one trading quarter. Fixed rather than
        derived from ``n_rows`` so every fold's confidence interval has the same
        width and the folds stay comparable to each other.
    n_splits : int, default 4
        Number of test blocks to attempt. Folds that cannot be honoured are
        dropped, so the returned list may be shorter.
    min_train : int, default 252
        Minimum training rows. A fold whose training window would fall below
        this is DROPPED, not shrunk -- a fold scored off a model fit on 40 rows
        is noise dressed as evidence, and shrinking it instead would silently
        make the folds incomparable.
    embargo : int, optional
        Extra bars vacated between the purged training block and the test
        block. Defaults to ``horizon``. Must be ``>= horizon``; a smaller value
        raises ``ValueError``, because the embargo is defined as protection *on
        top of* the purge and not as a substitute for it.

    Returns
    -------
    list of Fold
        Chronologically ordered, renumbered 1..k over the folds that survived.
        Empty when the series is too short for even one fold, in which case a
        warning names the exact row shortfall.

    Raises
    ------
    ValueError
        On a negative ``n_rows``, a non-positive ``horizon`` / ``test_size`` /
        ``n_splits`` / ``min_train``, or ``embargo < horizon``.
    """
    n_rows = int(n_rows)
    horizon = int(horizon)
    test_size = int(test_size)
    n_splits = int(n_splits)
    min_train = int(min_train)

    if n_rows < 0:
        raise ValueError(f"n_rows must be non-negative, got {n_rows}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 bar, got {horizon}")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1 row, got {test_size}")
    if n_splits < 1:
        raise ValueError(f"n_splits must be >= 1, got {n_splits}")
    if min_train < 1:
        raise ValueError(f"min_train must be >= 1 row, got {min_train}")

    embargo = horizon if embargo is None else int(embargo)
    if embargo < horizon:
        raise ValueError(
            f"embargo={embargo} is below horizon={horizon}: the embargo is an additional "
            f"gap on top of the {horizon}-bar purge, so it can never be smaller than the "
            f"horizon without reopening the label-overlap leak"
        )

    folds: List[Fold] = []
    for i in range(n_splits):
        # Fold i tests the block (n_splits - i) blocks back from the end, which
        # is what puts the oldest block first.
        test_start = n_rows - (n_splits - i) * test_size
        test_end = min(test_start + test_size, n_rows)
        train_end = test_start - embargo - horizon
        if test_start < 0 or train_end < min_train:
            logger.debug(
                "Dropping fold %d: test_start=%d, train_end=%d (min_train=%d)",
                i + 1,
                test_start,
                train_end,
                min_train,
            )
            continue
        folds.append(
            Fold(
                fold=len(folds) + 1,
                train_pos=np.arange(0, train_end, dtype=np.int64),
                test_pos=np.arange(test_start, test_end, dtype=np.int64),
                purged_pos=np.arange(train_end, train_end + horizon, dtype=np.int64),
                embargo_pos=np.arange(train_end + horizon, test_start, dtype=np.int64),
            )
        )

    if not folds:
        # The most recent fold is the cheapest one to satisfy, so if it does not
        # fit then nothing does: it needs min_train + horizon + embargo + test_size.
        needed = min_train + horizon + embargo + test_size
        logger.warning(
            "No purged walk-forward folds fit: n_rows=%s, horizon=%s, test_size=%s, "
            "n_splits=%s, embargo=%s, min_train=%s -- one fold needs %s rows, short by %s",
            n_rows,
            horizon,
            test_size,
            n_splits,
            embargo,
            min_train,
            needed,
            max(0, needed - n_rows),
        )
    return folds


def effective_sample_size(n: int, horizon: int) -> float:
    """
    Independent-observation equivalent of ``n`` OVERLAPPING h-period returns.

    Derivation
    ----------
    Let the one-step log increments ``x_t`` be i.i.d. with variance ``sigma^2``
    -- the martingale null this package tests against. The overlapping h-period
    return is the running sum

        R_t = x_{t+1} + ... + x_{t+h}

    Two such sums starting ``k`` bars apart share ``h - k`` of their increments
    when ``k < h`` and none at all when ``k >= h``, so

        Cov(R_t, R_{t+k}) = (h - k) * sigma^2      for 0 <= k < h
        rho_k             = 1 - k/h                for 0 <= k < h,  0 beyond.

    That autocorrelation is not an empirical finding about markets; it is
    arithmetic, present even under perfect independence of the underlying bars.
    The variance of the sample mean of ``n`` consecutive ``R_t`` is the standard
    long-run-variance expression

        Var(mean) = (gamma_0 / n) * [ 1 + 2 * sum_{k=1}^{n-1} (1 - k/n) * rho_k ]

    where ``(1 - k/n)`` is the number of lag-k pairs divided by ``n``.
    Substituting ``rho_k`` and truncating at ``min(h - 1, n - 1)``, beyond which
    it vanishes, gives the variance inflation factor

        f = 1 + 2 * sum_{k=1}^{min(h-1, n-1)} (1 - k/n) * (1 - k/h)

    and the effective sample size ``n / f``: the number of independent draws
    whose mean would be exactly this noisy. Every ``sqrt(n)`` in a t-statistic,
    a confidence interval or a power calculation built on overlapping returns
    should be a ``sqrt(n_eff)``. For n=252, h=20 the factor is about 12.6, so
    252 overlapping monthly returns are worth about 20 independent ones, and a
    t-statistic computed the naive way is overstated by roughly 3.5x.

    ``horizon=1`` has no overlap, the sum is empty, and the answer is exactly
    ``float(n)``. ``f >= 1`` always, because every term of the sum is
    non-negative, so the effective size never exceeds ``n``.

    Raises
    ------
    ValueError
        If ``n < 1`` or ``horizon < 1``. There is no defensible effective size
        for an empty sample or a zero-bar horizon, so nothing is invented here.
    """
    n = int(n)
    horizon = int(horizon)
    if n < 1:
        raise ValueError(f"n must be >= 1 observation, got {n}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 bar, got {horizon}")

    k_max = min(horizon - 1, n - 1)
    if k_max < 1:
        return float(n)

    k = np.arange(1, k_max + 1, dtype=np.float64)
    inflation = 1.0 + 2.0 * float(np.sum((1.0 - k / n) * (1.0 - k / horizon)))
    return float(n) / inflation


def non_overlapping_positions(
    positions: Union[Sequence[int], np.ndarray], horizon: int
) -> np.ndarray:
    """
    Thin ``positions`` to every ``horizon``-th element, starting at the first.

    The h-period return anchored at ``t`` consumes bars ``t+1 .. t+h``, so the
    next anchor sharing no bar with it is ``t + h``. Taking ``[::horizon]`` of a
    contiguous, chronologically ordered position array therefore leaves a sample
    of h-period returns that are independent under the null instead of
    mechanically autocorrelated. It costs ``(h-1)/h`` of the rows -- which is
    the honest price, and is roughly what :func:`effective_sample_size` says
    those rows were worth in the first place.

    An empty input returns an empty ``int64`` array; ``horizon=1`` returns the
    positions unchanged (as a fresh array).

    Raises
    ------
    ValueError
        If ``horizon < 1``.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1 bar, got {horizon}")
    array = _as_positions(positions, "positions")
    return array[::horizon].copy()


def non_overlapping_folds(folds: Sequence[Fold], horizon: int) -> List[Fold]:
    """
    Rebuild ``folds`` with their test blocks thinned by :func:`non_overlapping_positions`.

    Only ``test_pos`` is thinned. Training rows are left alone on purpose: the
    overlap problem is a *scoring* problem -- it corrupts the standard errors of
    the test statistics -- while a model is perfectly entitled to learn from
    every overlapping training row it can get.

    The purge and embargo blocks are carried through unchanged, so the audit
    trail from :func:`describe_split` still describes the real protocol.
    """
    return [
        Fold(
            fold=fold.fold,
            train_pos=fold.train_pos,
            test_pos=non_overlapping_positions(fold.test_pos, horizon),
            purged_pos=fold.purged_pos,
            embargo_pos=fold.embargo_pos,
        )
        for fold in folds
    ]


@dataclass(frozen=True)
class HeldOutSplit:
    """
    A sector-stratified partition of the ticker universe.

    ``in_universe`` is what research is allowed to touch: feature selection,
    hyperparameters, thresholds, the lot. ``held_out`` is scored exactly once,
    at the end. Stratifying by sector matters because sectors are the dominant
    source of cross-sectional correlation -- a random holdout can easily come
    out all-utilities, and "the model generalises" would then mean "the model
    generalises to utilities".

    ``by_sector`` maps each sector to ``{"in_universe": [...], "held_out": [...]}``
    so the balance is inspectable. ``holdout_fraction_actual`` is the realised
    fraction, which differs from the requested one because of integer rounding
    and the guarantee that no sector is held out entirely.
    """

    in_universe: List[str]
    held_out: List[str]
    by_sector: Dict[str, Dict[str, List[str]]]
    holdout_fraction_actual: float


def held_out_ticker_split(
    tickers: Sequence[str],
    sectors: Mapping[str, str],
    *,
    holdout_fraction: float = 0.2,
    seed: int = 42,
) -> HeldOutSplit:
    """
    Reserve a stratified slice of the universe that the research loop never sees.

    Per sector, ``round(holdout_fraction * n_members)`` tickers are held out,
    with two guards:

    * at least 1 when the sector has >= 2 members and ``holdout_fraction > 0``,
      so a small sector is not silently unrepresented in the holdout;
    * never all of a sector, so every sector still has in-universe members to
      train on. A singleton sector therefore contributes nothing to the holdout.

    Determinism: tickers are sorted before being shuffled and sectors are
    visited in sorted order, so the answer depends on ``seed`` alone and not on
    the insertion order of ``sectors`` or of the input sequence.

    Tickers missing from ``sectors`` (or mapped to ``None`` / blank) go into the
    ``"UNKNOWN"`` bucket and are stratified like any other sector.

    Raises
    ------
    ValueError
        On duplicate tickers, an empty ticker list (the realised holdout
        fraction would be 0/0, undefined rather than zero), or a
        ``holdout_fraction`` outside ``[0, 1)``. 1.0 is excluded because holding
        out the whole universe leaves nothing to train on.
    """
    holdout_fraction = float(holdout_fraction)
    if not 0.0 <= holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be in [0, 1), got {holdout_fraction}")

    ticker_list = [str(t) for t in tickers]
    if not ticker_list:
        raise ValueError("tickers is empty: the realised holdout fraction would be undefined")
    unique = set(ticker_list)
    if len(unique) != len(ticker_list):
        duplicates = sorted({t for t in ticker_list if ticker_list.count(t) > 1})
        raise ValueError(
            f"duplicate tickers in the universe: {duplicates} "
            f"({len(ticker_list)} entries, {len(unique)} unique)"
        )

    grouped: Dict[str, List[str]] = {}
    for ticker in ticker_list:
        raw = sectors.get(ticker) if sectors is not None else None
        name = str(raw).strip() if raw is not None and str(raw).strip() else UNKNOWN_SECTOR
        grouped.setdefault(name, []).append(ticker)

    rng = np.random.default_rng(seed)
    in_universe: List[str] = []
    held_out: List[str] = []
    by_sector: Dict[str, Dict[str, List[str]]] = {}

    for sector in sorted(grouped):
        members = sorted(grouped[sector])
        n_members = len(members)

        n_hold = int(round(holdout_fraction * n_members))
        if holdout_fraction > 0.0 and n_members >= 2:
            n_hold = max(1, n_hold)
        # Never strand a sector with no in-universe members. This also forces a
        # singleton sector to n_hold = 0.
        n_hold = max(0, min(n_hold, n_members - 1))

        order = rng.permutation(n_members)
        shuffled = [members[int(j)] for j in order]
        sector_held = sorted(shuffled[:n_hold])
        sector_in = sorted(shuffled[n_hold:])

        held_out.extend(sector_held)
        in_universe.extend(sector_in)
        by_sector[sector] = {"in_universe": sector_in, "held_out": sector_held}

    held_out.sort()
    in_universe.sort()
    return HeldOutSplit(
        in_universe=in_universe,
        held_out=held_out,
        by_sector=by_sector,
        holdout_fraction_actual=round(len(held_out) / len(ticker_list), 6),
    )


def _iso_date(index: pd.DatetimeIndex, position: Optional[int]) -> Optional[str]:
    """ISO calendar date at a row position, or None when there is no such row."""
    if position is None:
        return None
    return pd.Timestamp(index[position]).date().isoformat()


def describe_split(folds: Sequence[Fold], index: pd.DatetimeIndex) -> List[Dict[str, Any]]:
    """
    Render the protocol as rows fit for the results artifact.

    This is the audit trail. A reader who was not present at the run has to be
    able to see, per fold, where training stopped, where scoring started, how
    many bars were purged, how many embargoed, and how much wall-clock time the
    gap actually covered. ``gap_calendar_days`` is deliberately not
    ``gap_bars``: a gap straddling a weekend or a holiday is longer in days than
    in bars, and the bleed an embargo protects against lives in calendar time.

    Per fold: ``fold``, ``train_start``, ``train_end``, ``test_start``,
    ``test_end`` (ISO dates), ``n_train``, ``n_test``, ``purge_bars``,
    ``embargo_bars``, ``gap_bars`` (their sum) and ``gap_calendar_days``.

    Endpoints of an empty block come back as ``None`` rather than a fabricated
    date, and ``gap_calendar_days`` is ``None`` when either side of the gap has
    no bar to measure from.

    Raises
    ------
    ValueError
        If any fold references a row beyond the end of ``index``, with both
        lengths in the message.
    """
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)

    rows: List[Dict[str, Any]] = []
    for fold in folds:
        blocks = (fold.train_pos, fold.test_pos, fold.purged_pos, fold.embargo_pos)
        highest = max((int(block.max()) for block in blocks if block.size), default=-1)
        if highest >= len(index):
            raise ValueError(
                f"fold {fold.fold} references row position {highest} but the index has "
                f"{len(index)} rows"
            )

        train_last = int(fold.train_pos[-1]) if fold.n_train else None
        test_first = int(fold.test_pos[0]) if fold.n_test else None
        gap_days: Optional[int] = None
        if train_last is not None and test_first is not None:
            gap_days = int(
                (pd.Timestamp(index[test_first]) - pd.Timestamp(index[train_last])).days
            )

        rows.append(
            {
                "fold": fold.fold,
                "train_start": _iso_date(index, int(fold.train_pos[0]) if fold.n_train else None),
                "train_end": _iso_date(index, train_last),
                "test_start": _iso_date(index, test_first),
                "test_end": _iso_date(index, int(fold.test_pos[-1]) if fold.n_test else None),
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "purge_bars": int(fold.purged_pos.size),
                "embargo_bars": int(fold.embargo_pos.size),
                "gap_bars": int(fold.purged_pos.size + fold.embargo_pos.size),
                "gap_calendar_days": gap_days,
            }
        )
    return rows
