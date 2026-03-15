"""
Portfolio API routes — optimization, efficient frontier, metrics,
rebalancing, correlation, Monte Carlo simulation, sectors, alerts, drift.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas.schemas import (
    PortfolioOptimizeRequest, PortfolioOptimizeResponse, EfficientFrontierResponse
)
from src.portfolio.optimization import (
    optimize_portfolio, calculate_efficient_frontier, calculate_rebalancing_trades
)
from src.portfolio.performance_metrics import (
    calculate_portfolio_metrics, calculate_contribution,
    calculate_correlation_matrix, run_monte_carlo
)
from src.portfolio.weight_tracker import save_weights, get_last_weights, calculate_drift
from src.portfolio.sector import get_sector_allocation
from src.portfolio.risk_controls import check_risk_limits

router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_returns(symbols, lookback_days):
    """Fetch historical returns for multiple symbols."""
    import yfinance as yf
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")
    frames = {}
    for sym in symbols:
        df = yf.download(sym, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if not df.empty:
            frames[sym] = df["Close"]
    if not frames:
        raise HTTPException(404, "No data for any of the given symbols")
    prices = pd.DataFrame(frames).dropna()
    returns = prices.pct_change().dropna().tail(lookback_days)
    return returns, prices


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/optimize", response_model=PortfolioOptimizeResponse)
async def optimize(req: PortfolioOptimizeRequest):
    """Run portfolio optimization."""
    returns, prices = _fetch_returns(req.symbols, req.lookback_days)
    if returns.empty or len(returns) < 30:
        raise HTTPException(400, "Insufficient data for optimization")

    constraints = req.constraints or {"max_position": 0.4, "min_position": 0.02}
    weights = optimize_portfolio(
        returns, objective=req.method.value,
        risk_free_rate=req.risk_free_rate, constraints=constraints
    )

    # Compute portfolio metrics
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252
    w = np.array([weights.get(s, 0) for s in returns.columns])
    exp_ret = float(mean_ret.values @ w)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (exp_ret - req.risk_free_rate) / vol if vol > 0 else 0

    port_daily_ret = (returns * pd.Series(weights)).sum(axis=1)
    perf = calculate_portfolio_metrics(port_daily_ret)

    # Save weights snapshot for drift tracking
    portfolio_id = getattr(req, "portfolio_id", "default")
    save_weights(portfolio_id, req.method.value, weights)

    return PortfolioOptimizeResponse(
        method=req.method.value,
        weights={k: round(v, 4) for k, v in weights.items()},
        expected_return=round(exp_ret, 4),
        volatility=round(vol, 4),
        sharpe_ratio=round(sharpe, 4),
        metrics={k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
                 for k, v in perf.items()},
    )


@router.post("/frontier", response_model=EfficientFrontierResponse)
async def efficient_frontier(req: PortfolioOptimizeRequest):
    """Calculate and return efficient frontier points."""
    returns, _ = _fetch_returns(req.symbols, req.lookback_days)
    if returns.empty or len(returns) < 30:
        raise HTTPException(400, "Insufficient data")

    vols, rets, weights_list = calculate_efficient_frontier(returns, n_points=50)

    points = []
    for v, r, w in zip(vols, rets, weights_list):
        points.append({
            "volatility": round(float(v), 4),
            "return": round(float(r), 4),
            "sharpe": round((float(r) - 0.04) / float(v), 4) if float(v) > 0 else 0,
            "weights": {k: round(float(wv), 4) for k, wv in w.items()},
        })

    optimal = max(points, key=lambda p: p["sharpe"]) if points else {}
    return EfficientFrontierResponse(points=points, optimal_portfolio=optimal)


@router.get("/metrics")
async def portfolio_metrics(
    symbols: str = "AAPL,MSFT,GOOGL",
    lookback: int = 252,
    include_attribution: bool = False,
    weights: Optional[str] = None,
):
    """
    Get portfolio performance metrics.

    Set include_attribution=true and optionally pass weights as a JSON string
    to include per-stock return attribution breakdown.
    """
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    returns_df, _ = _fetch_returns(sym_list, lookback)
    if returns_df.empty:
        raise HTTPException(404, "No data")

    # Compute portfolio returns
    if weights:
        w = json.loads(weights)
    else:
        w = {s: 1 / len(sym_list) for s in sym_list}

    port_returns = (returns_df * pd.Series(w)).sum(axis=1)
    metrics = calculate_portfolio_metrics(port_returns)
    clean = {
        k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
        for k, v in metrics.items()
    }

    result = {"symbols": sym_list, "metrics": clean}

    # Optional: per-stock attribution
    if include_attribution:
        result["attribution"] = calculate_contribution(returns_df, w)

    return result


# ══════════════════════════════════════════════════════════════════════════════
# NEW ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

# ── §4.2 Rebalancing Engine ──────────────────────────────────────────────────

class RebalanceRequest(BaseModel):
    current_holdings: dict = Field(
        ..., description="Ticker → current dollar value, e.g. {'AAPL': 5000.0}"
    )
    target_weights: dict = Field(
        ..., description="Ticker → target weight (must sum to 1)"
    )
    total_portfolio_value: float = Field(..., description="Total portfolio $ value")
    transaction_cost_bps: float = Field(
        default=10.0, description="Transaction cost in basis points (10 = 0.10%)"
    )


@router.post("/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """
    Compute exact BUY/SELL trades to reach target weights,
    with dollar amounts, drift, and transaction cost estimates.
    """
    result = calculate_rebalancing_trades(
        current_holdings=request.current_holdings,
        target_weights=request.target_weights,
        total_portfolio_value=request.total_portfolio_value,
        transaction_cost_bps=request.transaction_cost_bps,
    )
    return result


# ── §4.4 Weight Drift Tracking ───────────────────────────────────────────────

@router.get("/drift")
async def get_weight_drift(
    portfolio_id: str = "default",
    current_values: str = "{}",
    total_value: float = 100000.0,
):
    """
    Compare current market values to the last saved weight snapshot.
    Returns drift per ticker and flags anything beyond 5% threshold.

    current_values: JSON string, e.g. '{"AAPL": 5200.0, "MSFT": 3100.0}'
    """
    snapshot = get_last_weights(portfolio_id)
    if not snapshot:
        raise HTTPException(404, "No saved weights found. Run /optimize first.")

    cv = json.loads(current_values) if isinstance(current_values, str) else current_values
    drift = calculate_drift(
        target_weights=snapshot["weights"],
        current_values=cv,
        total_value=total_value,
    )
    return {
        "drift": drift,
        "last_rebalanced": snapshot["saved_at"],
        "strategy": snapshot["strategy"],
        "needs_rebalance_count": sum(1 for v in drift.values() if v["needs_rebalance"]),
    }


# ── §5.1 Correlation Heatmap ─────────────────────────────────────────────────

@router.get("/correlation")
async def get_correlation(
    symbols: str = "AAPL,MSFT,GOOGL",
    lookback_days: int = 90,
    high_corr_threshold: float = 0.80,
):
    """
    Compute pairwise correlation matrix for the given symbols.
    Flags pairs with correlation ≥ threshold as poorly diversified.
    """
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    returns_df, _ = _fetch_returns(sym_list, lookback_days)
    if returns_df.empty:
        raise HTTPException(404, "No data")
    return calculate_correlation_matrix(returns_df, high_corr_threshold)


# ── §5.2 Monte Carlo Simulation ──────────────────────────────────────────────

class SimulateRequest(BaseModel):
    symbols: list = Field(default=["AAPL", "MSFT", "GOOGL"])
    weights: dict = Field(
        default={"AAPL": 0.33, "MSFT": 0.34, "GOOGL": 0.33},
        description="Ticker → weight (must sum to 1)",
    )
    n_simulations: int = Field(default=1000, ge=100, le=10000)
    n_days: int = Field(default=252, ge=20, le=1260)
    initial_value: float = Field(default=100000.0, gt=0)
    lookback_days: int = Field(default=252, ge=30)


@router.post("/simulate")
async def simulate_portfolio(request: SimulateRequest):
    """
    Run Monte Carlo simulation for the portfolio.
    Returns 5 percentile paths (p10–p90 fan chart) and probability statistics.
    """
    returns_df, _ = _fetch_returns(request.symbols, request.lookback_days)
    if returns_df.empty:
        raise HTTPException(404, "No data")
    return run_monte_carlo(
        returns_df=returns_df,
        weights=request.weights,
        n_simulations=request.n_simulations,
        n_days=request.n_days,
        initial_value=request.initial_value,
    )


# ── §5.3 Sector Allocation ───────────────────────────────────────────────────

@router.get("/sectors")
async def get_sectors(
    symbols: str = "AAPL,MSFT,GOOGL",
    weights: Optional[str] = None,
):
    """
    Get sector breakdown for the portfolio.
    Returns sector-level weights, tickers per sector, and concentration warnings.

    weights: optional JSON string, e.g. '{"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}'
    """
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    w = json.loads(weights) if weights else {s: 1 / len(sym_list) for s in sym_list}
    return get_sector_allocation(w)


# ── §5.4 Risk Controls & Alerts ──────────────────────────────────────────────

@router.get("/alerts")
async def get_risk_alerts(
    symbols: str = "AAPL,MSFT,GOOGL",
    weights: str = "{}",
    lookback_days: int = 90,
):
    """
    Run all risk checks and return prioritised alerts.
    Checks: position concentration, sector limits, stop-loss,
    Sharpe ratio, drawdown, correlation diversification.
    """
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    w = json.loads(weights) if weights != "{}" else {s: 1 / len(sym_list) for s in sym_list}

    returns_df, _ = _fetch_returns(sym_list, lookback_days)
    if returns_df.empty:
        raise HTTPException(404, "No data")

    # Portfolio metrics
    w_arr = np.array([w.get(t, 0) for t in returns_df.columns])
    port_returns = returns_df.values @ w_arr
    metrics = calculate_portfolio_metrics(pd.Series(port_returns))

    # Sector weights
    sectors = get_sector_allocation(w)

    # Correlation
    correlation = calculate_correlation_matrix(returns_df)

    alerts = check_risk_limits(
        current_weights=w,
        sector_weights=sectors["sector_weights"],
        portfolio_metrics=metrics,
        correlation_result=correlation,
    )
    return {
        "alerts": alerts,
        "alert_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
    }
