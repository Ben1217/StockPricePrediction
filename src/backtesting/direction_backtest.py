"""
Long/flat backtest for a next-day direction probability.

Execution, stated once and enforced everywhere below:

    signal formed at the CLOSE of t  ->  enter at the OPEN of t+1
                                    ->  exit at the CLOSE of t+1

so the return the strategy can actually capture is ``Close(t+1) / Open(t+1) - 1``.

That is deliberately **not** the classifier's target, which is
``Close(t+1) / Close(t) - 1``. The overnight gap between the close that produced
the signal and the open that fills the order is not available to a strategy that
decides at the close, and on a next-day horizon that gap is a large share of the
whole move. A backtest that pays itself the close-to-close return is claiming a
fill it never had; the difference is the single most common way a direction
study reports an edge that does not exist. The prices used here come from
``DirectionDataset.entry_open`` / ``exit_close``, which are built alongside the
label so the execution lag cannot drift out of step with it.

Costs are charged as a **round trip per active day**, because this rule opens
and closes a position every day it trades. 5-10 bps is the floor for a liquid US
large cap; anything lower is an assertion about the user's broker, not about the
market.

The number worth reading in the output is ``breakeven_cost_bps``. "The edge dies
above X bps" survives a change of broker, a change of size, and a change of year
in a way that a single Sharpe figure does not.

Public API:
    run_long_flat_backtest(...) -> BacktestResult
    select_threshold(...) -> ThresholdChoice
    DEFAULT_THRESHOLD_GRID
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..portfolio.performance_metrics import calculate_max_drawdown, calculate_sharpe_ratio
from ..utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252

# Thresholds on P(up). Below ~0.50 the rule is "always long" and the backtest
# just re-derives buy & hold; above ~0.62 a model this weak trades a handful of
# days a year and the result is noise.
DEFAULT_THRESHOLD_GRID: tuple[float, ...] = (
    0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62,
)

# A threshold that produces fewer trades than this on the validation window is
# rejected: its statistics are not estimates, they are anecdotes.
MIN_VALIDATION_TRADES = 15

# Bisection bounds for the breakeven-cost search, in basis points.
_MAX_BREAKEVEN_BPS = 1000.0
_BREAKEVEN_TOLERANCE_BPS = 1e-4


@dataclass
class BacktestResult:
    """Equity curve plus the strategy and benchmark scorecards."""

    equity_curve: pd.DataFrame
    metrics: Dict[str, Any] = field(default_factory=dict)
    benchmark_metrics: Dict[str, Any] = field(default_factory=dict)
    breakeven: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.equity_curve)


@dataclass
class ThresholdChoice:
    """The threshold picked on validation, and what every candidate scored."""

    threshold: float
    objective: str
    score: float
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    fell_back: bool = False


def _tradeable_returns(entry_open: Sequence, exit_close: Sequence) -> np.ndarray:
    """The open-to-close return a signal formed at the previous close can capture."""
    opens = np.asarray(entry_open, dtype=np.float64).reshape(-1)
    closes = np.asarray(exit_close, dtype=np.float64).reshape(-1)
    if opens.shape != closes.shape:
        raise ValueError(f"entry_open ({opens.shape}) and exit_close ({closes.shape}) must align")
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = closes / np.where(opens > 0, opens, np.nan) - 1.0
    return returns


def _net_returns(
    positions: np.ndarray,
    gross_returns: np.ndarray,
    cost_bps: float,
) -> np.ndarray:
    """
    Position-weighted returns after a round-trip cost on every active day.

    A day the strategy sits flat earns and costs nothing, so it contributes a
    factor of exactly 1 to the equity curve while still counting as a period in
    the Sharpe denominator — which is correct: idle capital is part of the risk
    the strategy takes, and excluding those days would inflate the ratio.
    """
    cost = float(cost_bps) / 1e4
    return positions * gross_returns - positions * cost


def _drawdown_series(equity: np.ndarray) -> np.ndarray:
    running_max = np.maximum.accumulate(equity)
    return equity / running_max - 1.0


def _performance(
    returns: np.ndarray,
    positions: Optional[np.ndarray],
    *,
    risk_free_rate: float,
    periods_per_year: int,
    label: str,
) -> Dict[str, Any]:
    """Standard scorecard for one return stream. ``positions`` is None for buy & hold."""
    returns = np.asarray(returns, dtype=np.float64).reshape(-1)
    n = int(returns.size)
    if n == 0:
        return {"label": label, "n_periods": 0}

    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    years = n / float(periods_per_year)
    # A negative terminal equity is impossible here (daily returns > -100%), but
    # a total loss makes the CAGR root undefined rather than -100%.
    cagr = float((equity[-1]) ** (1.0 / years) - 1.0) if years > 0 and equity[-1] > 0 else None

    active = positions.astype(bool) if positions is not None else np.ones(n, dtype=bool)
    active_returns = returns[active]
    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]
    n_active = int(active.sum())

    metrics: Dict[str, Any] = {
        "label": label,
        "n_periods": n,
        "years": round(years, 4),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6) if cagr is not None else None,
        "sharpe": round(float(calculate_sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        )), 4),
        "annualized_volatility": round(float(np.std(returns) * np.sqrt(periods_per_year)), 6),
        "max_drawdown": round(float(calculate_max_drawdown(returns)), 6),
        "hit_rate": round(float(np.mean(active_returns > 0)), 6) if n_active else None,
        "avg_win": round(float(np.mean(wins)), 6) if wins.size else None,
        "avg_loss": round(float(np.mean(losses)), 6) if losses.size else None,
        "win_loss_ratio": (
            round(float(abs(np.mean(wins) / np.mean(losses))), 4)
            if wins.size and losses.size and np.mean(losses) != 0 else None
        ),
        "time_in_market_pct": round(100.0 * n_active / n, 4),
        "n_active_periods": n_active,
    }
    if positions is not None:
        # Each active day is one full round trip: in at the open, out at the close.
        metrics["round_trips"] = n_active
        metrics["round_trips_per_year"] = round(n_active / years, 2) if years > 0 else None
        # Turnover counts both legs, so a strategy long every day turns over 2x/day.
        metrics["turnover_per_year"] = round(2.0 * n_active / years, 2) if years > 0 else None
    return metrics


def _benchmark_returns(entry_open: np.ndarray, exit_close: np.ndarray, cost_bps: float) -> np.ndarray:
    """
    Buy & hold on the same bars, charged one round trip for the whole period.

    Bought at the first entry open and marked to each exit close afterwards, so
    it spans exactly the window the strategy is scored over. The single round
    trip is taken on the first period, which is when the position is opened.
    """
    n = len(exit_close)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    returns = np.empty(n, dtype=np.float64)
    returns[0] = exit_close[0] / entry_open[0] - 1.0
    if n > 1:
        returns[1:] = exit_close[1:] / exit_close[:-1] - 1.0
    returns[0] -= float(cost_bps) / 1e4
    return returns


def _terminal_wealth(positions: np.ndarray, gross_returns: np.ndarray, cost_bps: float) -> float:
    return float(np.prod(1.0 + _net_returns(positions, gross_returns, cost_bps)))


def _bisect_breakeven(objective, lo: float = 0.0, hi: float = _MAX_BREAKEVEN_BPS) -> Optional[float]:
    """
    Smallest cost in [lo, hi] bps at which ``objective(cost)`` crosses zero.

    ``objective`` must be non-increasing in cost, which both callers are:
    raising the cost can only reduce the strategy's terminal wealth, and reduces
    it far faster than it reduces a one-round-trip benchmark.
    """
    f_lo = objective(lo)
    if f_lo <= 0:
        # The edge is already gone at zero cost. There is nothing to erode.
        return 0.0
    if objective(hi) > 0:
        # Still positive at an absurd cost; report it as unbounded rather than
        # pinning it to the search ceiling and implying precision.
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if objective(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < _BREAKEVEN_TOLERANCE_BPS:
            break
    return 0.5 * (lo + hi)


def _describe_thresholds(thresholds: np.ndarray) -> str:
    """One threshold prints as a number; pooled per-fold thresholds print as a range."""
    if thresholds.size == 0:
        return "n/a"
    if np.all(thresholds == thresholds[0]):
        return f"{float(thresholds[0]):.3f}"
    return f"{float(thresholds.min()):.3f}-{float(thresholds.max()):.3f} (per fold)"


def run_long_flat_backtest(
    probabilities: Sequence,
    entry_open: Sequence,
    exit_close: Sequence,
    *,
    threshold: float | Sequence[float] = 0.55,
    cost_bps: float = 10.0,
    index: Optional[pd.Index] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> BacktestResult:
    """
    Run the long/flat rule and score it against buy & hold on the same bars.

    Parameters
    ----------
    probabilities : sequence of float
        P(up) for the move that resolves after ``entry_open``/``exit_close``.
    entry_open, exit_close : sequence of float
        The open the order fills at and the close it exits at, aligned to
        ``probabilities`` — normally ``DirectionDataset.entry_open`` and
        ``.exit_close`` restricted to the test window.
    threshold : float or sequence of float
        Go long when ``P(up) > threshold``; stay flat otherwise. Chosen on
        validation by :func:`select_threshold`, never on the window being
        scored. A sequence gives one threshold per row, which is what pooling
        walk-forward folds needs: each fold carries the threshold its own
        validation window earned, and a single pooled curve can still be drawn
        across all of them.
    cost_bps : float
        Round-trip cost in basis points, charged on every active day.
    risk_free_rate : float
        Annual rate subtracted before the Sharpe ratio. Defaults to 0 so the
        strategy and the benchmark are compared on identical terms.

    Returns
    -------
    BacktestResult
        ``equity_curve`` has one row per decision date, carrying the position,
        the gross and net return, the cost paid, and both equity curves.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    opens = np.asarray(entry_open, dtype=np.float64).reshape(-1)
    closes = np.asarray(exit_close, dtype=np.float64).reshape(-1)
    if not (len(probabilities) == len(opens) == len(closes)):
        raise ValueError(
            f"length mismatch: probabilities={len(probabilities)}, "
            f"entry_open={len(opens)}, exit_close={len(closes)}"
        )

    if index is None:
        index = pd.RangeIndex(len(probabilities))
    elif len(index) != len(probabilities):
        raise ValueError(f"index has {len(index)} entries but {len(probabilities)} rows were given")

    gross = _tradeable_returns(opens, closes)
    if not np.all(np.isfinite(gross)):
        raise ValueError("entry_open/exit_close produced a non-finite return; check for zero prices")

    thresholds = np.asarray(threshold, dtype=np.float64).reshape(-1)
    if thresholds.size == 1:
        thresholds = np.repeat(thresholds, len(probabilities))
    elif thresholds.size != len(probabilities):
        raise ValueError(
            f"threshold has {thresholds.size} entries but {len(probabilities)} rows were given"
        )

    # Strictly greater than: at threshold 0.50 a probability of exactly 0.50 is
    # a coin, and a coin is not a reason to pay a spread.
    positions = (probabilities > thresholds).astype(np.float64)
    costs = positions * (float(cost_bps) / 1e4)
    net = _net_returns(positions, gross, cost_bps)
    benchmark = _benchmark_returns(opens, closes, cost_bps)

    equity = np.cumprod(1.0 + net)
    benchmark_equity = np.cumprod(1.0 + benchmark)

    curve = pd.DataFrame({
        "probability_up": probabilities,
        "threshold": thresholds,
        "position": positions,
        "entry_open": opens,
        "exit_close": closes,
        "gross_return": gross,
        "cost": costs,
        "net_return": net,
        "equity": equity,
        "drawdown": _drawdown_series(equity),
        "benchmark_return": benchmark,
        "benchmark_equity": benchmark_equity,
    }, index=index)
    curve.index.name = "decision_date"

    metrics = _performance(
        net, positions, risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year, label="long_flat",
    )
    benchmark_metrics = _performance(
        benchmark, None, risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year, label="buy_and_hold",
    )

    active = positions.astype(bool)
    gross_active = gross[active]
    breakeven = {
        "cost_charged_bps": float(cost_bps),
        # The average gross move captured per trade. Rounded to bps, this is the
        # raw edge the costs have to come out of.
        "mean_gross_return_per_trade_bps": (
            round(float(np.mean(gross_active) * 1e4), 4) if gross_active.size else None
        ),
        "breakeven_cost_bps_positive": _bisect_breakeven(
            lambda c: _terminal_wealth(positions, gross, c) - 1.0
        ),
        "breakeven_cost_bps_vs_buy_and_hold": _bisect_breakeven(
            lambda c: _terminal_wealth(positions, gross, c)
            - float(np.prod(1.0 + _benchmark_returns(opens, closes, c)))
        ),
    }

    logger.info(
        "Backtest: threshold=%s cost=%.1fbps -> %d/%d days long, net %.2f%% vs "
        "buy & hold %.2f%%, breakeven %s bps",
        _describe_thresholds(thresholds), cost_bps, int(active.sum()), len(positions),
        100 * metrics.get("total_return", 0.0), 100 * benchmark_metrics.get("total_return", 0.0),
        breakeven["breakeven_cost_bps_positive"],
    )

    return BacktestResult(
        equity_curve=curve,
        metrics=metrics,
        benchmark_metrics=benchmark_metrics,
        breakeven=breakeven,
        config={
            "threshold": (
                float(thresholds[0]) if np.all(thresholds == thresholds[0])
                else [round(float(t), 4) for t in np.unique(thresholds)]
            ),
            "threshold_is_per_row": bool(not np.all(thresholds == thresholds[0])),
            "cost_bps": float(cost_bps),
            "risk_free_rate": float(risk_free_rate),
            "periods_per_year": int(periods_per_year),
            "execution": "signal at close(t); enter open(t+1); exit close(t+1)",
        },
    )


def select_threshold(
    probabilities: Sequence,
    entry_open: Sequence,
    exit_close: Sequence,
    *,
    cost_bps: float = 10.0,
    grid: Sequence[float] = DEFAULT_THRESHOLD_GRID,
    objective: str = "sharpe",
    min_trades: int = MIN_VALIDATION_TRADES,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> ThresholdChoice:
    """
    Pick the long/flat threshold on a validation window.

    Call this with validation data only. The threshold is a fitted parameter
    like any other, and choosing it on the window you then report is how a
    backtest reports the best of eight coin flips as a strategy.

    Candidates that trade fewer than ``min_trades`` times are rejected before
    scoring: the highest Sharpe in a grid search is otherwise reliably the
    threshold that took three trades and won all three. If every candidate is
    rejected, the most permissive threshold in the grid is returned with
    ``fell_back=True`` so the caller can see the choice was not earned.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    opens = np.asarray(entry_open, dtype=np.float64).reshape(-1)
    closes = np.asarray(exit_close, dtype=np.float64).reshape(-1)
    gross = _tradeable_returns(opens, closes)

    if objective not in {"sharpe", "total_return"}:
        raise ValueError(f"objective must be 'sharpe' or 'total_return', got {objective!r}")

    candidates: List[Dict[str, Any]] = []
    best_threshold, best_score = None, -np.inf

    for threshold in grid:
        positions = (probabilities > float(threshold)).astype(np.float64)
        n_trades = int(positions.sum())
        net = _net_returns(positions, gross, cost_bps)
        if objective == "sharpe":
            score = float(calculate_sharpe_ratio(net, risk_free_rate=0.0, periods_per_year=periods_per_year))
        else:
            score = float(np.prod(1.0 + net) - 1.0)

        eligible = n_trades >= min_trades
        candidates.append({
            "threshold": float(threshold),
            "n_trades": n_trades,
            "eligible": eligible,
            "score": round(score, 6),
        })
        if eligible and score > best_score:
            best_threshold, best_score = float(threshold), score

    fell_back = best_threshold is None
    if fell_back:
        best_threshold = float(min(grid))
        best_score = float("nan")
        logger.warning(
            "No threshold in %s traded at least %d times on the validation window; "
            "falling back to the most permissive (%.2f)",
            list(grid), min_trades, best_threshold,
        )

    return ThresholdChoice(
        threshold=best_threshold,
        objective=objective,
        score=best_score,
        candidates=candidates,
        fell_back=fell_back,
    )
