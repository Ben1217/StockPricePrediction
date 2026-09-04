"""
Economic evaluation and the paper-trading overlay (Requirement A7).

A statistically significant edge that dies at six basis points of cost is not an
edge. This module is what separates a finance project from a generic
time-series project: it converts a directional forecast into a fully specified
rule, charges it, and reports the cost at which the edge disappears.

Alignment, stated once because getting it wrong silently destroys the result
-----------------------------------------------------------------------------
``period_log_returns[t]`` is the log return realised over the forecast horizon
*starting* at decision bar ``t`` -- the same quantity the evaluator scores, and
the same alignment ``DirectionDataset.forward_return`` carries. The position
formed from ``p_up[t]`` is therefore applied to ``period_log_returns[t]``
directly, with **no shift**. Shifting here would pair yesterday's signal with
today's forward return, double-lagging the strategy and quietly turning a real
edge into noise (or noise into an edge).

Why geometric annualisation
---------------------------
``mean(r) * periods_per_year`` overstates what an investor actually compounds,
and it is inconsistent with a maximum drawdown measured on a compounded equity
curve. Both numbers appear in the same table, so both are computed on the same
compounding convention.

Why the break-even cost is the headline
---------------------------------------
Annualised return depends on the cost assumption, and every reader has a
different one. The break-even round-trip cost does not: "the edge dies above
X bps" survives a change of broker, of size, and of year. It is also where most
published "predictive" equity models quietly fail.

Public API:
    CostModel(commission_bps_per_side, spread_bps_per_side)
    long_flat_positions(p_up, threshold) -> ndarray
    long_short_positions(p_up, threshold) -> ndarray
    strategy_returns(positions, period_log_returns, cost) -> dict
    performance_summary(net_returns, *, periods_per_year, ...) -> dict
    max_drawdown(equity_or_returns) -> dict
    breakeven_round_trip_bps(positions, period_log_returns, ...) -> dict
    buy_and_hold(period_log_returns, cost, *, periods_per_year, ...) -> dict
    paper_trading_overlay(p_up, period_log_returns, *, cost, ...) -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Upper bound for the break-even bisection, in basis points. A round trip
#: costing more than this is not a market anyone trades.
_MAX_BREAKEVEN_BPS = 10_000.0


def _as_float_array(values: Sequence, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} is empty")
    return array


@dataclass(frozen=True)
class CostModel:
    """
    The cost assumption, stated explicitly so a reader can disagree with it.

    Costs are charged per side, on the notional traded. A move from flat to long
    trades one unit; a flip from short to long trades two, and is charged twice.

    Market impact is **excluded**. That is a real limitation, not an oversight:
    modelling it needs a liquidity model and an order-size assumption this
    project does not have. :meth:`to_dict` always carries the exclusion so it
    cannot fall out of the results table.
    """

    commission_bps_per_side: float = 1.0
    spread_bps_per_side: float = 2.5

    @property
    def per_side_bps(self) -> float:
        return float(self.commission_bps_per_side) + float(self.spread_bps_per_side)

    @property
    def per_side_fraction(self) -> float:
        return self.per_side_bps / 1e4

    @property
    def round_trip_bps(self) -> float:
        return 2.0 * self.per_side_bps

    def to_dict(self) -> Dict[str, Any]:
        return {
            "commission_bps_per_side": round(float(self.commission_bps_per_side), 6),
            "spread_bps_per_side": round(float(self.spread_bps_per_side), 6),
            "per_side_bps": round(self.per_side_bps, 6),
            "round_trip_bps": round(self.round_trip_bps, 6),
            "market_impact_included": False,
            "limitation": (
                "market impact is excluded; costs are modelled as a fixed per-side "
                "charge on traded notional"
            ),
        }


# ---------------------------------------------------------------------------
# Signal rules (A7.1) -- fully specified, never fitted on test data
# ---------------------------------------------------------------------------


def long_flat_positions(p_up: Sequence, threshold: float = 0.5) -> np.ndarray:
    """+1 when P(up) > threshold, else flat. The threshold is fixed at 0.5 by A3.3."""
    probabilities = _as_float_array(p_up, "p_up")
    return np.where(probabilities > float(threshold), 1.0, 0.0)


def long_short_positions(p_up: Sequence, threshold: float = 0.5) -> np.ndarray:
    """+1 when P(up) > threshold, else -1. Always invested, so always paying."""
    probabilities = _as_float_array(p_up, "p_up")
    return np.where(probabilities > float(threshold), 1.0, -1.0)


# ---------------------------------------------------------------------------
# Costed strategy returns
# ---------------------------------------------------------------------------


def strategy_returns(
    positions: Sequence,
    period_log_returns: Sequence,
    cost: CostModel,
) -> Dict[str, Any]:
    """
    Gross, cost and net simple returns per holding period.

    The forecast horizon is the rebalance period, so each row is one holding
    period. Positions start flat, which is why the first entry is charged.

    ``traded[t] = |position[t] - position[t-1]|`` is notional turned over, so
    ``cost[t] = traded[t] * per_side_fraction``: entering a long from flat pays
    one side, and flipping short-to-long pays two.
    """
    weights = _as_float_array(positions, "positions")
    log_returns = _as_float_array(period_log_returns, "period_log_returns")
    if weights.size != log_returns.size:
        raise ValueError(
            f"length mismatch: positions={weights.size}, "
            f"period_log_returns={log_returns.size}"
        )

    # Simple returns are what a position earns; log returns are what we score.
    simple = np.expm1(log_returns)
    previous = np.concatenate(([0.0], weights[:-1]))
    traded = np.abs(weights - previous)

    gross = weights * simple
    costs = traded * cost.per_side_fraction
    net = gross - costs

    return {
        "gross": gross,
        "cost": costs,
        "net": net,
        "traded": traded,
        "positions": weights,
        "total_gross": float(np.nansum(gross)),
        "total_cost": float(np.nansum(costs)),
        "total_net": float(np.nansum(net)),
        "total_traded": float(np.nansum(traded)),
        "n_periods": int(weights.size),
    }


# ---------------------------------------------------------------------------
# Performance (A7.3)
# ---------------------------------------------------------------------------


def max_drawdown(returns: Sequence) -> Dict[str, Any]:
    """
    Maximum drawdown on the compounded equity curve, and whether it recovered.

    ``time_to_recovery_periods`` is **None** when the curve never regains its
    prior peak inside the sample. Filling in the sample end there would report a
    recovery that did not happen.
    """
    simple = _as_float_array(returns, "returns")
    equity = np.cumprod(1.0 + np.nan_to_num(simple, nan=0.0))
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0

    trough = int(np.argmin(drawdowns))
    worst = float(drawdowns[trough])
    peak = int(np.argmax(equity[: trough + 1])) if trough > 0 else 0

    recovery: Optional[int] = None
    after = np.nonzero(equity[trough:] >= equity[peak])[0]
    if after.size:
        recovery = int(trough + after[0])

    return {
        "max_drawdown": round(worst, 6),
        "peak_index": peak,
        "trough_index": trough,
        "recovery_index": recovery,
        "time_to_recovery_periods": None if recovery is None else int(recovery - peak),
        "recovered": recovery is not None,
    }


def performance_summary(
    net_returns: Sequence,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    traded: Optional[Sequence] = None,
) -> Dict[str, Any]:
    """
    The A7.3 table for one return stream.

    ``risk_free_rate`` is an **annual** rate and is echoed in the output, because
    a Sharpe ratio without its risk-free rate is not interpretable.

    Sortino uses the downside deviation ``sqrt(mean(min(r - MAR, 0)^2))`` taken
    over *every* observation, not the standard deviation of the negative subset.
    The latter is a common and wrong shortcut: it drops the zero terms and
    de-means the losses, which flatters a strategy that is flat most of the time.
    """
    simple = _as_float_array(net_returns, "net_returns")
    usable = simple[np.isfinite(simple)]
    n = int(usable.size)
    ppy = int(periods_per_year)
    if n == 0 or ppy <= 0:
        raise ValueError(f"need finite returns and periods_per_year > 0; got n={n}, ppy={ppy}")

    growth = float(np.prod(1.0 + usable))
    total_return = growth - 1.0
    # Geometric, so it agrees with the compounded equity curve the drawdown uses.
    annualised_return = growth ** (ppy / n) - 1.0 if growth > 0 else -1.0
    volatility = float(np.std(usable, ddof=1)) if n > 1 else 0.0
    annualised_volatility = volatility * np.sqrt(ppy)

    periodic_rf = (1.0 + float(risk_free_rate)) ** (1.0 / ppy) - 1.0
    excess = usable - periodic_rf
    excess_mean = float(np.mean(excess))

    sharpe = (
        float(excess_mean / np.std(excess, ddof=1) * np.sqrt(ppy))
        if n > 1 and np.std(excess, ddof=1) > 0
        else None
    )
    shortfall = np.minimum(excess, 0.0)
    downside_deviation = float(np.sqrt(np.mean(shortfall**2)))
    sortino = (
        float(excess_mean / downside_deviation * np.sqrt(ppy))
        if downside_deviation > 0
        else None
    )

    wins = usable[usable > 0]
    losses = usable[usable < 0]
    decided = wins.size + losses.size
    average_win = float(np.mean(wins)) if wins.size else None
    average_loss = float(np.mean(losses)) if losses.size else None

    summary: Dict[str, Any] = {
        "n_periods": n,
        "total_return": round(total_return, 6),
        "annualised_return": round(annualised_return, 6),
        "annualised_volatility": round(annualised_volatility, 6),
        "sharpe_ratio": round(sharpe, 6) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 6) if sortino is not None else None,
        "risk_free_rate_annual": round(float(risk_free_rate), 6),
        "hit_rate": round(wins.size / decided, 6) if decided else None,
        "average_win": round(average_win, 6) if average_win is not None else None,
        "average_loss": round(average_loss, 6) if average_loss is not None else None,
        "win_loss_ratio": (
            round(abs(average_win / average_loss), 6)
            if average_win is not None and average_loss not in (None, 0.0)
            else None
        ),
        "periods_per_year": ppy,
    }
    summary.update(max_drawdown(usable))

    if traded is not None:
        turned = _as_float_array(traded, "traded")
        summary["turnover_annualised"] = round(float(np.nansum(turned)) / n * ppy, 6)
    else:
        summary["turnover_annualised"] = None
    return summary


# ---------------------------------------------------------------------------
# Break-even cost (A7.4) -- the headline economic number
# ---------------------------------------------------------------------------


def breakeven_round_trip_bps(
    positions: Sequence,
    period_log_returns: Sequence,
    *,
    benchmark_log_returns: Optional[Sequence] = None,
) -> Dict[str, Any]:
    """
    The per-side cost, in bps, at which the strategy's gross alpha is consumed.

    Two solutions are reported because they answer slightly different questions:

    ``arithmetic_bps``
        Solves ``sum(gross) = cost_fraction * sum(traded)`` exactly. This is the
        number to quote: it is closed-form and independent of path.
    ``compounded_bps``
        Bisects terminal wealth, which is strictly decreasing in cost, so the
        root is unique. It differs from the arithmetic answer only when returns
        are large enough for compounding to matter.

    Both are **None** when gross alpha is already negative at zero cost -- there
    is nothing for costs to erode, and reporting a number there would imply the
    strategy had an edge to lose. ``unbounded`` is True when the edge survives
    even at :data:`_MAX_BREAKEVEN_BPS`.
    """
    weights = _as_float_array(positions, "positions")
    log_returns = _as_float_array(period_log_returns, "period_log_returns")
    if weights.size != log_returns.size:
        raise ValueError(
            f"length mismatch: positions={weights.size}, "
            f"period_log_returns={log_returns.size}"
        )

    simple = np.expm1(log_returns)
    previous = np.concatenate(([0.0], weights[:-1]))
    traded = np.abs(weights - previous)
    gross = weights * simple

    total_gross = float(np.nansum(gross))
    total_traded = float(np.nansum(traded))

    result: Dict[str, Any] = {
        "total_gross_return": round(total_gross, 8),
        "total_traded_notional": round(total_traded, 8),
        "arithmetic_bps": None,
        "compounded_bps": None,
        "unbounded": False,
        "reason": None,
    }

    if total_traded <= 0:
        result["reason"] = "the rule never trades, so no cost is ever charged"
        return result
    if total_gross <= 0:
        result["reason"] = "gross alpha is not positive at zero cost; there is no edge to erode"
        return result

    result["arithmetic_bps"] = round(1e4 * total_gross / total_traded, 6)

    def terminal_wealth(per_side_bps: float) -> float:
        net = gross - traded * (per_side_bps / 1e4)
        return float(np.prod(1.0 + np.nan_to_num(net, nan=0.0)))

    baseline = 1.0
    if benchmark_log_returns is not None:
        benchmark = _as_float_array(benchmark_log_returns, "benchmark_log_returns")
        if benchmark.size != log_returns.size:
            raise ValueError(
                f"length mismatch: benchmark={benchmark.size}, returns={log_returns.size}"
            )
        baseline = float(np.prod(1.0 + np.expm1(benchmark)))
        result["benchmark_terminal_wealth"] = round(baseline, 8)

    if terminal_wealth(0.0) <= baseline:
        result["reason"] = "the strategy does not beat the comparison even at zero cost"
        return result
    if terminal_wealth(_MAX_BREAKEVEN_BPS) > baseline:
        result["unbounded"] = True
        result["reason"] = f"still ahead at {_MAX_BREAKEVEN_BPS:.0f} bps per side"
        return result

    low, high = 0.0, _MAX_BREAKEVEN_BPS
    for _ in range(200):
        middle = 0.5 * (low + high)
        if terminal_wealth(middle) > baseline:
            low = middle
        else:
            high = middle
        if high - low < 1e-6:
            break
    result["compounded_bps"] = round(0.5 * (low + high), 6)
    return result


# ---------------------------------------------------------------------------
# Comparisons and the harness entry point
# ---------------------------------------------------------------------------


def buy_and_hold(
    period_log_returns: Sequence,
    cost: CostModel,
    *,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """Always long, charged one entry side. The A7.3 comparison row."""
    log_returns = _as_float_array(period_log_returns, "period_log_returns")
    weights = np.ones_like(log_returns)
    # Enter once and hold: only the first bar turns over notional.
    traded = np.zeros_like(log_returns)
    traded[0] = 1.0
    net = weights * np.expm1(log_returns) - traded * cost.per_side_fraction
    summary = performance_summary(
        net, periods_per_year=periods_per_year, risk_free_rate=risk_free_rate, traded=traded
    )
    summary["rule"] = "buy_and_hold"
    return summary


def paper_trading_overlay(
    p_up: Sequence,
    period_log_returns: Sequence,
    *,
    cost: CostModel,
    periods_per_year: int,
    risk_free_rate: float = 0.0,
    threshold: float = 0.5,
    rules: Sequence[str] = ("long_flat", "long_short"),
) -> Dict[str, Any]:
    """
    Run every rule plus buy-and-hold on one forecast stream (A7.1-A7.4).

    No rule parameter is tuned here. The threshold is fixed, the notional is
    fixed, and the rebalance period is the forecast horizon.
    """
    builders = {"long_flat": long_flat_positions, "long_short": long_short_positions}
    unknown = [rule for rule in rules if rule not in builders]
    if unknown:
        raise ValueError(f"unknown rules {unknown}; known: {sorted(builders)}")

    report: Dict[str, Any] = {
        "cost_model": cost.to_dict(),
        "threshold": float(threshold),
        "rules": {},
        "buy_and_hold": buy_and_hold(
            period_log_returns,
            cost,
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
        ),
    }

    for rule in rules:
        weights = builders[rule](p_up, threshold)
        traded_returns = strategy_returns(weights, period_log_returns, cost)
        summary = performance_summary(
            traded_returns["net"],
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_rate,
            traded=traded_returns["traded"],
        )
        summary["gross_total_return"] = round(traded_returns["total_gross"], 6)
        summary["cost_total"] = round(traded_returns["total_cost"], 6)
        summary["breakeven"] = breakeven_round_trip_bps(
            weights, period_log_returns, benchmark_log_returns=period_log_returns
        )
        summary["rule"] = rule
        report["rules"][rule] = summary

    return report
