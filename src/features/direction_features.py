"""
Features and labels for the next-day direction classifier.

The target here is not a forward return. It is the sign of one:

    fwd = Close.shift(-1) / Close - 1        # Close is already adjusted
    y   = (fwd > 0).astype(int)

That change is the whole point of this pipeline. A 30-day forward-return
regression emits a single scalar per bundle and the "path" drawn from it is
``P_0 * (1 + r) ** (t/30)`` — a monotone curve that cannot, as a matter of
arithmetic, show a direction change. Predicting the sign of tomorrow's move
asks a question the data can actually answer or refuse to answer, and the
refusal is legible: accuracy below the base rate.

Feature set
-----------
``STATIONARY_REGRESSION_FEATURE_COLUMNS`` is reused unchanged — those 13 columns
are already scale-free, which is what a classifier wants — plus six
short-horizon directional columns the stationary set has no analogue for:

    Overnight_Gap             Open / Prev_Close - 1
    Intraday_Return           Close / Open - 1
    Close_Position_In_Range   (Close - Low) / (High - Low)
    Return_Sign_Lag1/2/3      sign of Daily_Return one, two, three bars back

The lagged signs carry the *shape* of the recent path (up-up-down vs
down-down-up), which short-term reversal is defined on. The sign of *today's*
return is deliberately not a separate column: ``Daily_Return`` is already in the
stationary set as a signed value, so both a linear model and a tree recover it.

On top of those, :mod:`src.features.chart_patterns` contributes 27 columns that
describe the *shape* of the recent candles rather than the level of price —
candle geometry, breakout and channel position, trend cleanliness, and volume
confirmation. That is the "read the chart" half of the feature set, and it is
what a sequence model like Kronos gets for free from raw candles.

The legacy 95-column set and the wide inference superset stay out. Columns are
added when they describe something the existing set cannot express, not to make
the matrix wider.

Leakage rules enforced here
---------------------------
Every column at row ``t`` is a function of bars at or before ``t``. There are no
centred rolling windows, no reversed series, no full-series ``fit``. Rows whose
features or target are incomplete are dropped, never filled — an imputed feature
row at the start of a series is a fabricated observation, and an imputed target
is a fabricated answer. Scaling belongs to the model layer, fitted per training
fold (see :mod:`src.models.direction_models`).

Public API:
    add_direction_features(df) -> DataFrame
    build_direction_dataset(df, ...) -> DirectionDataset
    latest_feature_row(df, ...) -> (Timestamp, DataFrame) | None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..utils.logger import get_logger
from .chart_patterns import (
    CHART_PATTERN_FEATURE_COLUMNS,
    add_chart_pattern_features,
)
from .feature_engineering import (
    STATIONARY_REGRESSION_FEATURE_COLUMNS,
    build_regression_feature_frame,
)

logger = get_logger(__name__)

# Lags for the sign-of-return columns. 1..3 covers the window over which
# short-term reversal is documented; beyond that the signs are noise.
RETURN_SIGN_LAGS: tuple[int, ...] = (1, 2, 3)

DIRECTION_EXTRA_FEATURE_COLUMNS: List[str] = [
    "Overnight_Gap",
    "Intraday_Return",
    "Close_Position_In_Range",
] + [f"Return_Sign_Lag{lag}" for lag in RETURN_SIGN_LAGS]

# The full default feature set: 13 stationary + 6 directional + 27 chart-pattern
# = 46 columns.
#
# On the size question: 46 features over ~2 500 rows is a ratio of about 54
# observations per feature, which regularised models handle. The earlier warning
# against the 95-column legacy set was about 95 features over ~1 000 rows — and
# about *what* those columns were, mostly price levels and near-duplicate
# oscillators. These 27 are chart geometry with little overlap with the 19, and
# the walk-forward folds plus the shuffled-label check remain the arbiter: if
# the extra columns only bought in-sample fit, the pooled accuracy says so.
DIRECTION_FEATURE_COLUMNS: List[str] = (
    list(STATIONARY_REGRESSION_FEATURE_COLUMNS)
    + DIRECTION_EXTRA_FEATURE_COLUMNS
    + list(CHART_PATTERN_FEATURE_COLUMNS)
)

# The 19-column set the pipeline shipped with, kept so a run can be reproduced
# against it and the chart-pattern columns can be shown to earn their place.
DIRECTION_BASE_FEATURE_COLUMNS: List[str] = (
    list(STATIONARY_REGRESSION_FEATURE_COLUMNS) + DIRECTION_EXTRA_FEATURE_COLUMNS
)

# Feature config for direction work. Regime detection is off because
# MarketRegimeDetector fits an HMM on the whole series, which would let a
# 2019 row carry information from 2024. Candlesticks are off to hold the
# column count down.
DIRECTION_FEATURE_CONFIG: Dict[str, object] = {
    "include_technical": True,
    "include_lags": False,
    "include_regime": False,
    "include_candlesticks": False,
    "lag_periods": [],
}

# A feature column must have at least this many non-null observations in the
# raw frame to be usable; below it the column is a warm-up artefact.
MIN_FEATURE_OBSERVATIONS = 100


@dataclass
class DirectionDataset:
    """
    An aligned, leakage-checked supervised dataset for next-day direction.

    Every series shares one index: the *decision dates*. Row ``t`` holds
    features known at the close of ``t`` and the label for the move that
    resolves ``horizon`` bars later.

    ``entry_open`` and ``exit_close`` are the prices a backtest is allowed to
    transact at given a signal formed at the close of ``t``: the open of ``t+1``
    and the close of ``t+horizon``. They are carried here rather than re-derived
    downstream so the execution lag cannot drift out of step with the label.

    ``ohlcv`` is the raw Open/High/Low/Close/Volume frame on the same index as
    ``features``. It is optional: tabular models ignore it, but sequence-based
    models (Kronos) need the original candlestick geometry. When present it
    covers the same date range as ``features`` and is sliced in lockstep.
    """

    features: pd.DataFrame
    labels: pd.Series
    forward_return: pd.Series
    entry_open: pd.Series
    exit_close: pd.Series
    feature_columns: List[str]
    meta: Dict[str, Any] = field(default_factory=dict)
    ohlcv: Optional[pd.DataFrame] = None

    def __len__(self) -> int:
        return len(self.features)

    @property
    def base_rate(self) -> float:
        """Share of labels that are 1 (an up move). The number to beat."""
        if len(self.labels) == 0:
            return float("nan")
        return float(self.labels.mean())

    @property
    def index(self) -> pd.Index:
        return self.features.index

    def slice(self, positions: Sequence[int]) -> "DirectionDataset":
        """Positional subset, preserving alignment across every series."""
        idx = np.asarray(positions, dtype=int)
        return DirectionDataset(
            features=self.features.iloc[idx],
            labels=self.labels.iloc[idx],
            forward_return=self.forward_return.iloc[idx],
            entry_open=self.entry_open.iloc[idx],
            exit_close=self.exit_close.iloc[idx],
            feature_columns=list(self.feature_columns),
            meta=dict(self.meta),
            ohlcv=self.ohlcv.iloc[idx] if self.ohlcv is not None else None,
        )


def add_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the six short-horizon directional columns.

    Requires ``Open``/``High``/``Low``/``Close`` and uses ``Daily_Return`` when
    present, deriving it from ``Close`` otherwise. Every column reads only the
    current bar or earlier ones.
    """
    data = df.copy()
    required = {"Open", "High", "Low", "Close"}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"add_direction_features requires {sorted(missing)}")

    open_ = pd.to_numeric(data["Open"], errors="coerce")
    high = pd.to_numeric(data["High"], errors="coerce")
    low = pd.to_numeric(data["Low"], errors="coerce")
    close = pd.to_numeric(data["Close"], errors="coerce")

    # Where the session opened relative to yesterday's close: the part of the
    # daily move that cannot be traded intraday.
    prev_close = close.shift(1).replace(0, np.nan)
    data["Overnight_Gap"] = open_ / prev_close - 1.0

    # The part that can be: open to close on the same bar.
    data["Intraday_Return"] = close / open_.replace(0, np.nan) - 1.0

    # Where the close sits inside the day's range. 1.0 is a close on the high,
    # 0.0 on the low. A zero-width range (High == Low, a limit-locked or
    # untraded bar) has the close trivially at both ends, so the midpoint 0.5 is
    # the defined value rather than an imputed one.
    span = high - low
    position = (close - low) / span.where(span > 0)
    data["Close_Position_In_Range"] = position.where(span > 0, 0.5).where(span.notna())

    daily_return = data["Daily_Return"] if "Daily_Return" in data.columns else close.pct_change()
    daily_return = pd.to_numeric(daily_return, errors="coerce")
    for lag in RETURN_SIGN_LAGS:
        lagged = daily_return.shift(lag)
        # np.sign propagates NaN, so warm-up rows stay missing and get dropped
        # rather than silently becoming a "flat day" signal.
        data[f"Return_Sign_Lag{lag}"] = np.sign(lagged)

    return data


def _deadband_thresholds(
    daily_return: pd.Series,
    volatility: Optional[pd.Series],
    sigma_multiple: float,
    window: int = 20,
) -> pd.Series:
    """
    Per-row deadband tau = sigma_multiple * sigma_20d, using only past returns.

    ``volatility`` is the frame's existing 20-day return standard deviation when
    available; it is recomputed from ``daily_return`` otherwise. Both are
    trailing windows ending at ``t``, so tau at ``t`` is knowable at ``t``.
    """
    sigma = volatility if volatility is not None else daily_return.rolling(window).std()
    sigma = pd.to_numeric(sigma, errors="coerce")
    return sigma * float(sigma_multiple)


def build_direction_dataset(
    df: pd.DataFrame,
    *,
    horizon: int = 1,
    deadband_sigma_multiple: float = 0.0,
    feature_columns: Optional[Sequence[str]] = None,
    feature_config: Optional[Dict[str, object]] = None,
) -> DirectionDataset:
    """
    Turn adjusted OHLCV bars into an aligned (X, y) direction dataset.

    Parameters
    ----------
    df : DataFrame
        Adjusted daily OHLCV, ideally from
        :func:`src.data.direction_data.load_daily_bars` so that every column
        sits on one price basis.
    horizon : int
        Bars ahead the label resolves over. 1 is the supported and intended
        case; larger values need an embargo of ``horizon`` bars in the splitter
        because consecutive labels then overlap.
    deadband_sigma_multiple : float
        When > 0, rows whose forward return falls inside
        ``+/- multiple * sigma_20d`` are dropped rather than labelled. This cuts
        the noisiest labels but changes the base rate, so the resulting base
        rate is recorded in ``meta`` and must be reported next to any accuracy
        figure. 0.0 (the default) keeps every row.
    feature_columns : sequence of str, optional
        Overrides ``DIRECTION_FEATURE_COLUMNS``.
    feature_config : dict, optional
        Overrides ``DIRECTION_FEATURE_CONFIG``. Passing
        ``include_regime=True`` reintroduces a whole-series HMM fit and is
        rejected.

    Returns
    -------
    DirectionDataset

    Raises
    ------
    ValueError
        On a non-positive horizon, a leakage-prone feature config, or a frame
        that yields no usable rows.
    """
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    config = dict(DIRECTION_FEATURE_CONFIG)
    if feature_config:
        config.update(feature_config)
    if config.get("include_regime"):
        raise ValueError(
            "include_regime=True fits a regime model on the whole series, which "
            "leaks future information into every earlier row. Direction datasets "
            "must be built with include_regime=False."
        )

    requested = list(feature_columns or DIRECTION_FEATURE_COLUMNS)

    frame = build_regression_feature_frame(df, feature_config=config)
    if frame.empty:
        raise ValueError("Feature frame is empty; check the input bars")
    frame = add_direction_features(frame)
    frame = add_chart_pattern_features(frame)

    close = pd.to_numeric(frame["Close"], errors="coerce")
    open_ = pd.to_numeric(frame["Open"], errors="coerce")

    # The label. shift(-horizon) reads forward by design and is the ONLY
    # forward-looking operation in this module.
    forward_return = close.shift(-horizon) / close.replace(0, np.nan) - 1.0

    # Prices a signal formed at the close of t may transact at: it cannot touch
    # bar t, so entry is the next open and exit is the close it resolves on.
    entry_open = open_.shift(-1)
    exit_close = close.shift(-horizon)

    resolved = [
        col for col in requested
        if col in frame.columns and int(frame[col].notna().sum()) >= MIN_FEATURE_OBSERVATIONS
    ]
    dropped = [col for col in requested if col not in resolved]
    if dropped:
        logger.warning(
            "Dropping %d feature(s) with insufficient history: %s", len(dropped), dropped
        )
    if not resolved:
        raise ValueError("No requested feature column has enough history to be usable")

    aligned = frame[resolved].copy()
    aligned["__forward_return"] = forward_return
    aligned["__entry_open"] = entry_open
    aligned["__exit_close"] = exit_close

    rows_before = len(aligned)
    aligned = aligned.dropna()
    rows_after_dropna = len(aligned)

    deadband_dropped = 0
    if deadband_sigma_multiple > 0:
        tau = _deadband_thresholds(
            pd.to_numeric(frame.get("Daily_Return", close.pct_change()), errors="coerce"),
            pd.to_numeric(frame["Volatility"], errors="coerce") if "Volatility" in frame else None,
            deadband_sigma_multiple,
        ).reindex(aligned.index)
        keep = (aligned["__forward_return"].abs() > tau) & tau.notna()
        deadband_dropped = int((~keep).sum())
        aligned = aligned[keep]

    if aligned.empty:
        raise ValueError("No usable rows remain after dropping incomplete and deadband rows")

    labels = (aligned["__forward_return"] > 0).astype(np.int8)
    labels.name = "direction_up"

    # Raw OHLCV on the aligned index, for sequence-based models (Kronos).
    ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in frame.columns]
    ohlcv_frame = frame[ohlcv_cols].reindex(aligned.index) if ohlcv_cols else None

    dataset = DirectionDataset(
        features=aligned[resolved],
        labels=labels,
        forward_return=aligned["__forward_return"].rename("forward_return"),
        entry_open=aligned["__entry_open"].rename("entry_open"),
        exit_close=aligned["__exit_close"].rename("exit_close"),
        feature_columns=resolved,
        meta={
            "horizon": horizon,
            "n_rows": int(len(aligned)),
            "n_features": len(resolved),
            "rows_before_dropna": int(rows_before),
            "rows_dropped_incomplete": int(rows_before - rows_after_dropna),
            "rows_dropped_deadband": deadband_dropped,
            "deadband_sigma_multiple": float(deadband_sigma_multiple),
            "base_rate": float(labels.mean()),
            "first_date": str(aligned.index[0].date()),
            "last_date": str(aligned.index[-1].date()),
            "dropped_features": dropped,
        },
        ohlcv=ohlcv_frame,
    )

    logger.info(
        "Direction dataset: %d rows x %d features, horizon=%d, base rate %.4f "
        "(dropped %d incomplete, %d deadband)",
        len(dataset), len(resolved), horizon, dataset.base_rate,
        dataset.meta["rows_dropped_incomplete"], deadband_dropped,
    )
    return dataset


def latest_feature_row(
    df: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    feature_config: Optional[Dict[str, object]] = None,
) -> Optional[tuple[pd.Timestamp, pd.DataFrame]]:
    """
    The most recent bar with a complete feature vector and no resolved label.

    :func:`build_direction_dataset` deliberately drops this row: its forward
    return does not exist yet, so it cannot be trained or scored on. It is also
    the only row anyone actually wants a prediction for — the close has printed,
    tomorrow has not. This returns it separately so a live P(up tomorrow) is
    served from exactly the feature pipeline the evaluation used, rather than a
    second, subtly different one.

    Returns ``(timestamp, one_row_frame)``, or None when no row has a complete
    feature vector.
    """
    config = dict(DIRECTION_FEATURE_CONFIG)
    if feature_config:
        config.update(feature_config)
    if config.get("include_regime"):
        raise ValueError("include_regime=True leaks future information; refused")

    requested = list(feature_columns or DIRECTION_FEATURE_COLUMNS)
    frame = build_regression_feature_frame(df, feature_config=config)
    if frame.empty:
        return None
    frame = add_direction_features(frame)
    frame = add_chart_pattern_features(frame)

    available = [col for col in requested if col in frame.columns]
    if len(available) != len(requested):
        missing = [col for col in requested if col not in available]
        raise ValueError(f"Feature frame is missing required columns: {missing}")

    complete = frame[available].dropna()
    if complete.empty:
        return None
    timestamp = complete.index[-1]
    return timestamp, complete.iloc[[-1]]
