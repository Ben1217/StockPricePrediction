"""
Cross-sectional evaluation: IC, RankIC, IC-IR and the quintile spread (A7.5).

This arm exists for **fairness**, not completeness. Kronos's published headline
metric is RankIC -- a cross-sectional *ranking* statistic asking whether the
model orders assets correctly relative to each other -- not the per-asset
point-accuracy statistic the rest of this module measures. Judging it only on
per-asset accuracy would be a straw man. So the evaluation the paper was
actually validated on is reproduced here, and the two are reported side by side.

Skipped dates are skipped, not zeroed
-------------------------------------
A date with too few names to rank cannot produce an IC or a quintile spread.
Recording a 0.0 for it would be a fabricated observation: it drags the mean
toward zero and, worse, adds a zero-variance point that inflates the Sharpe
ratio of the spread portfolio. Such dates are dropped and **counted**, and the
count is returned so no silent truncation reaches the report.

On the significance of a mean IC
--------------------------------
``ic_ir * sqrt(n_dates)`` assumes independent dates. Overlapping h-period
returns violate that by construction, so the naive t-statistic here is
optimistic. The per-date IC series is returned precisely so the caller can pass
it to :func:`src.evaluation.testing.newey_west_variance` for a HAC-corrected
standard error.

Public API:
    compute_ic(predictions, realized_returns, min_names) -> dict
    quintile_spread_portfolio(predictions, realized_returns, ...) -> dict
    cross_sectional_scorecard(predictions, realized_returns, ...) -> dict
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Fewer names than this and a cross-sectional correlation is noise, not a rank.
_MIN_NAMES = 5


def _align(predictions: pd.DataFrame, realized_returns: pd.DataFrame):
    """Common dates and tickers, in a stable order, so results are reproducible."""
    dates = predictions.index.intersection(realized_returns.index)
    tickers = predictions.columns.intersection(realized_returns.columns)
    if len(dates) == 0 or len(tickers) == 0:
        raise ValueError(
            f"no overlap between predictions and realised returns: "
            f"{len(dates)} shared dates, {len(tickers)} shared tickers"
        )
    return (
        predictions.loc[dates, tickers].sort_index(),
        realized_returns.loc[dates, tickers].sort_index(),
    )


def _per_date_correlations(
    predictions: pd.DataFrame,
    realized: pd.DataFrame,
    min_names: int,
) -> pd.DataFrame:
    """
    Per-date Pearson (IC) and Spearman (RankIC) correlation across the cross-section.

    Computed explicitly rather than via ``DataFrame.corrwith`` so that the usable
    name count per date, and the reason a date was skipped, are both observable.
    """
    rows: List[Dict[str, Any]] = []
    for date in predictions.index:
        pair = pd.concat(
            [predictions.loc[date].rename("pred"), realized.loc[date].rename("real")], axis=1
        ).dropna()
        n_names = int(len(pair))
        if n_names < min_names:
            rows.append({"date": date, "n_names": n_names, "ic": np.nan,
                         "rank_ic": np.nan, "skipped": True,
                         "reason": f"only {n_names} usable names, need {min_names}"})
            continue
        pred, real = pair["pred"].to_numpy(), pair["real"].to_numpy()
        # A constant cross-section has no ordering, so correlation is undefined.
        if np.ptp(pred) == 0 or np.ptp(real) == 0:
            rows.append({"date": date, "n_names": n_names, "ic": np.nan,
                         "rank_ic": np.nan, "skipped": True,
                         "reason": "predictions or realised returns are constant across names"})
            continue
        rows.append({
            "date": date,
            "n_names": n_names,
            "ic": float(stats.pearsonr(pred, real).statistic),
            "rank_ic": float(stats.spearmanr(pred, real).statistic),
            "skipped": False,
            "reason": None,
        })
    return pd.DataFrame(rows).set_index("date")


def _summarise(series: pd.Series, label: str) -> Dict[str, Any]:
    values = series.dropna()
    n = int(len(values))
    if n == 0:
        return {f"{label}_mean": None, f"{label}_std": None, f"{label}_ir": None,
                f"{label}_hit_rate": None, f"{label}_t_stat_naive": None,
                f"{label}_p_value_naive": None, f"{label}_n_dates": 0}

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    ir = mean / std if std > 0 else None
    t_stat = ir * np.sqrt(n) if ir is not None else None
    return {
        f"{label}_mean": round(mean, 6),
        f"{label}_std": round(std, 6),
        f"{label}_ir": round(ir, 6) if ir is not None else None,
        f"{label}_hit_rate": round(float((values > 0).mean()), 6),
        f"{label}_t_stat_naive": round(float(t_stat), 6) if t_stat is not None else None,
        f"{label}_p_value_naive": (
            round(float(2.0 * stats.t.sf(abs(t_stat), df=n - 1)), 8)
            if t_stat is not None and n > 1 else None
        ),
        f"{label}_n_dates": n,
    }


def compute_ic(
    predictions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    min_names: int = _MIN_NAMES,
) -> Dict[str, Any]:
    """
    Information Coefficient (Pearson) and RankIC (Spearman) per date, then averaged.

    ``predictions`` and ``realized_returns`` are date x ticker frames. IC-IR is
    the mean divided by the standard deviation of the per-date series.

    The naive t-statistic assumes independent dates and is labelled as such; see
    the module docstring.
    """
    predictions, realized = _align(predictions, realized_returns)
    per_date = _per_date_correlations(predictions, realized, int(min_names))

    n_skipped = int(per_date["skipped"].sum())
    if n_skipped:
        logger.info(
            "cross-sectional IC: skipped %d/%d dates with too few or constant names",
            n_skipped, len(per_date),
        )

    summary: Dict[str, Any] = {
        "n_dates_total": int(len(per_date)),
        "n_dates_skipped": n_skipped,
        "n_dates_used": int(len(per_date) - n_skipped),
        "min_names": int(min_names),
    }
    summary.update(_summarise(per_date["ic"], "ic"))
    summary.update(_summarise(per_date["rank_ic"], "rank_ic"))
    summary["per_date"] = [
        {"date": str(pd.Timestamp(idx).date()), "ic": None if pd.isna(row.ic) else round(row.ic, 6),
         "rank_ic": None if pd.isna(row.rank_ic) else round(row.rank_ic, 6),
         "n_names": int(row.n_names), "skipped": bool(row.skipped)}
        for idx, row in per_date.iterrows()
    ]
    return summary


def quintile_spread_portfolio(
    predictions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    periods_per_year: int = 252,
    n_quantiles: int = 5,
    min_names: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Long the top predicted quantile, short the bottom, equal-weighted, per date.

    ``min_names`` defaults to ``n_quantiles`` -- you cannot form five buckets from
    four names. Dates below it are skipped and counted, never recorded as a zero
    return: a fabricated zero both drags the mean and, because it has no
    variance, inflates the Sharpe ratio of everything around it.

    Ties are broken by ``rank(method="first")`` before bucketing so the buckets
    are always well defined; where the names do not divide evenly, ``pd.qcut``
    gives the remainder to the lower buckets.
    """
    predictions, realized = _align(predictions, realized_returns)
    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    floor = n_quantiles if min_names is None else int(min_names)

    records: List[Dict[str, Any]] = []
    n_skipped = 0
    for date in predictions.index:
        pair = pd.concat(
            [predictions.loc[date].rename("pred"), realized.loc[date].rename("real")], axis=1
        ).dropna()
        if len(pair) < floor:
            n_skipped += 1
            continue
        ranks = pair["pred"].rank(method="first", ascending=True)
        try:
            buckets = pd.qcut(ranks, q=n_quantiles, labels=False)
        except ValueError as exc:  # noqa: BLE001 - degenerate cross-section, skip and count
            logger.debug("qcut failed on %s: %s", date, exc)
            n_skipped += 1
            continue
        top = pair["real"][buckets == n_quantiles - 1]
        bottom = pair["real"][buckets == 0]
        if top.empty or bottom.empty:
            n_skipped += 1
            continue
        records.append({
            "date": date,
            "n_names": int(len(pair)),
            "top_return": float(top.mean()),
            "bottom_return": float(bottom.mean()),
            "spread": float(top.mean() - bottom.mean()),
        })

    if n_skipped:
        logger.info("quintile spread: skipped %d/%d dates", n_skipped, len(predictions.index))

    result: Dict[str, Any] = {
        "n_quantiles": n_quantiles,
        "min_names": floor,
        "n_dates_total": int(len(predictions.index)),
        "n_dates_skipped": n_skipped,
        "n_dates_used": len(records),
        "risk_free_rate_note": (
            "a long/short spread is self-funding, so the Sharpe ratio here uses a "
            "zero risk-free rate"
        ),
    }

    if not records:
        result.update({
            "quintile_spread_annualised_return": None,
            "quintile_spread_annualised_volatility": None,
            "quintile_spread_sharpe_ratio": None,
            "quintile_spread_hit_rate": None,
            "reason": "no date had enough names to form the buckets",
            "per_date": [],
        })
        return result

    frame = pd.DataFrame(records).set_index("date")
    spread = frame["spread"]
    n = int(len(spread))
    ppy = int(periods_per_year)

    growth = float(np.prod(1.0 + spread.to_numpy()))
    annualised_return = growth ** (ppy / n) - 1.0 if growth > 0 else -1.0
    volatility = float(spread.std(ddof=1)) if n > 1 else 0.0
    annualised_volatility = volatility * np.sqrt(ppy)

    result.update({
        "quintile_spread_mean_per_period": round(float(spread.mean()), 6),
        "quintile_spread_annualised_return": round(annualised_return, 6),
        "quintile_spread_annualised_volatility": round(annualised_volatility, 6),
        "quintile_spread_sharpe_ratio": (
            round(float(spread.mean() / volatility * np.sqrt(ppy)), 6) if volatility > 0 else None
        ),
        "quintile_spread_hit_rate": round(float((spread > 0).mean()), 6),
        "per_date": [
            {"date": str(pd.Timestamp(idx).date()), "spread": round(row.spread, 6),
             "top_return": round(row.top_return, 6), "bottom_return": round(row.bottom_return, 6),
             "n_names": int(row.n_names)}
            for idx, row in frame.iterrows()
        ],
    })
    return result


def cross_sectional_scorecard(
    predictions: pd.DataFrame,
    realized_returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    n_quantiles: int = 5,
    min_names: int = _MIN_NAMES,
) -> Dict[str, Any]:
    """The single call the harness makes for the A7.5 arm."""
    return {
        "ic": compute_ic(predictions, realized_returns, min_names=min_names),
        "quintile_spread": quintile_spread_portfolio(
            predictions,
            realized_returns,
            periods_per_year=periods_per_year,
            n_quantiles=n_quantiles,
        ),
        "note": (
            "RankIC is the metric Kronos was published on. The naive IC t-statistic "
            "assumes independent dates; pass the per-date IC series to "
            "src.evaluation.testing.newey_west_variance for a HAC-corrected standard error."
        ),
    }
