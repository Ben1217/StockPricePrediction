"""
Portfolio API routes — optimization, efficient frontier, metrics.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException

from src.api.schemas.schemas import (
    PortfolioOptimizeRequest, PortfolioOptimizeResponse, EfficientFrontierResponse
)
from src.portfolio.optimization import optimize_portfolio, calculate_efficient_frontier
from src.portfolio.performance_metrics import calculate_portfolio_metrics

router = APIRouter()


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

    # Find optimal point (max Sharpe)
    optimal = max(points, key=lambda p: p["sharpe"]) if points else {}

    return EfficientFrontierResponse(points=points, optimal_portfolio=optimal)


@router.get("/metrics")
async def portfolio_metrics(symbols: str = "AAPL,MSFT,GOOGL", lookback: int = 252):
    """Get portfolio performance metrics for equal-weighted portfolio."""
    sym_list = [s.strip() for s in symbols.split(",")]
    returns, _ = _fetch_returns(sym_list, lookback)
    if returns.empty:
        raise HTTPException(404, "No data")
    eq_returns = returns.mean(axis=1)
    metrics = calculate_portfolio_metrics(eq_returns)
    clean = {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
             for k, v in metrics.items()}
    return {"symbols": sym_list, "metrics": clean}
