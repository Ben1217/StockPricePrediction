"""
Portfolio API routes — optimization, efficient frontier, metrics,
rebalancing, correlation, Monte Carlo simulation, sectors, alerts, drift.
"""

import json
import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.api.schemas.schemas import (
    PortfolioOptimizeRequest, PortfolioOptimizeResponse, EfficientFrontierResponse
)
from src.data.ohlcv_cache import cached_download, normalize_ohlcv_frame, safe_yf_download
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
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _finite(value, default: float = 0.0) -> float:
    """
    Coerce `value` to a float that JSON can actually represent.

    NaN and ±Inf are legitimate outcomes of the maths here — a zero-variance
    window makes a Sharpe ratio infinite, an empty CVaR tail makes it NaN — but
    they cannot be rendered, so they collapse to `default` for the typed fields
    that must hold a number.
    """
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _json_safe(obj):
    """
    Recursively replace non-finite floats with None and numpy scalars with builtins.

    FastAPI renders responses with ``json.dumps(..., allow_nan=False)``, so one NaN
    anywhere in the payload raises ValueError *after* the handler has returned and
    the browser sees a bare 500. Reporting the unmeasurable cell as null keeps the
    rest of the answer intact, which is what the caller actually needs.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _parse_json_param(raw: Optional[str], name: str) -> Optional[dict]:
    """
    Parse a JSON-encoded query parameter into a dict.

    A malformed value used to reach `json.loads` unguarded and escape as a 500,
    which the browser then reported as a CORS error rather than as bad input.
    """
    if raw is None or raw == "":
        return None
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(400, f"Query parameter '{name}' is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise HTTPException(400, f"Query parameter '{name}' must be a JSON object")
    return value


def _fetch_returns(symbols, lookback_days):
    """
    Fetch aligned daily returns for `symbols`.

    Downloads go through the shared OHLCV cache rather than calling yfinance
    directly. Optimize, frontier, correlation and alerts all ask for the same
    tickers over the same window, and a burst of identical uncached requests is
    what earns a 429 from Yahoo. `cached_download` also retries with backoff and
    falls back to an expired entry, so a provider hiccup degrades instead of
    raising.

    Everything after the download is defensive because this function assembles a
    frame from several independent sources, and any one of them being malformed
    used to take down the whole request:

      * ``safe_yf_download`` holds the global download lock and returns exactly
        one ticker's columns. Without it, two overlapping requests merged into a
        single frame with duplicate 'Close' labels, ``df["Close"]`` came back
        two-dimensional, and ``pd.DataFrame({...})`` raised "Data must be
        1-dimensional" — a 500 for every request in flight.
      * Duplicate timestamps are collapsed upstream, since duplicate index labels
        make the alignment below raise instead of aligning.
      * Non-positive prices are dropped, because ``pct_change`` turns them into
        ±Inf and FastAPI renders with ``allow_nan=False``.
    """
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

    seen, ordered = set(), []
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if sym and sym not in seen:
            seen.add(sym)
            ordered.append(sym)
    if not ordered:
        raise HTTPException(400, "No symbols given.")

    frames, failed = {}, []
    for sym in ordered:
        def _download(symbol=sym):
            return safe_yf_download(symbol, start=start, end=end)

        try:
            # "1d-adj" namespaces these adjusted closes away from the raw bars
            # training caches under the same ticker and window. The key covers
            # ticker/range/interval only, so sharing a tag across callers with
            # different auto_adjust settings serves one caller the other's prices.
            df = cached_download(sym, start, end, "1d-adj", _download)
            # Cache entries predate the normalizer, so clean on the way out too.
            df = normalize_ohlcv_frame(df, sym)
        except Exception:  # noqa: BLE001 - provider raises many shapes
            logger.exception("Price download failed for %s", sym)
            df = None

        if df is None or df.empty:
            failed.append(sym)
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        # A non-positive price makes pct_change return inf, and FastAPI renders with
        # allow_nan=False, so a single bad tick would 500 the whole response.
        close = close[close > 0]
        if close.empty:
            failed.append(sym)
            continue
        frames[sym] = close

    if not frames:
        raise HTTPException(
            404,
            f"No price data for {', '.join(failed) or 'the given symbols'}. "
            "Check the tickers, or retry shortly if the data provider is rate-limiting.",
        )
    if failed:
        logger.warning("Dropping symbols with no price data: %s", ", ".join(failed))

    prices = pd.concat(frames, axis=1, join="outer").sort_index().dropna()
    prices.columns = list(frames)
    returns = (
        prices.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .tail(lookback_days)
    )
    return returns, prices


# ══════════════════════════════════════════════════════════════════════════════
# EXISTING ENDPOINTS (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/optimize", response_model=PortfolioOptimizeResponse)
def optimize(req: PortfolioOptimizeRequest):
    """Run portfolio optimization."""
    returns, prices = _fetch_returns(req.symbols, req.lookback_days)
    if returns.empty or len(returns) < 30:
        raise HTTPException(400, "Insufficient data for optimization")

    constraints = req.constraints or {"max_position": 0.4, "min_position": 0.02}
    weights = optimize_portfolio(
        returns, objective=req.method.value,
        risk_free_rate=req.risk_free_rate, constraints=constraints
    )
    # A solver that half-converges can hand back NaN weights; those would poison
    # every metric below and then fail to render as a `float` response field.
    weights = {str(k): _finite(v) for k, v in weights.items()}

    # Compute portfolio metrics
    mean_ret = returns.mean() * 252
    cov = returns.cov() * 252
    w = np.array([weights.get(s, 0) for s in returns.columns])
    exp_ret = _finite(mean_ret.values @ w)
    # A covariance quadratic form can land a hair below zero on a near-singular
    # window, and sqrt of that is NaN rather than an error.
    variance = _finite(w @ cov.values @ w)
    vol = math.sqrt(variance) if variance > 0 else 0.0
    sharpe = (exp_ret - req.risk_free_rate) / vol if vol > 0 else 0.0

    port_daily_ret = (returns * pd.Series(weights)).sum(axis=1)
    perf = calculate_portfolio_metrics(port_daily_ret)

    # Save weights snapshot for drift tracking
    portfolio_id = getattr(req, "portfolio_id", "default")
    save_weights(portfolio_id, req.method.value, weights)

    return PortfolioOptimizeResponse(
        method=req.method.value,
        weights={k: round(v, 4) for k, v in weights.items()},  # already finite
        expected_return=round(exp_ret, 4),
        volatility=round(vol, 4),
        sharpe_ratio=round(sharpe, 4),
        metrics=_json_safe({
            k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
            for k, v in perf.items()
        }),
    )


@router.post("/frontier", response_model=EfficientFrontierResponse)
def efficient_frontier(req: PortfolioOptimizeRequest):
    """Calculate and return efficient frontier points."""
    returns, _ = _fetch_returns(req.symbols, req.lookback_days)
    if returns.empty or len(returns) < 30:
        raise HTTPException(400, "Insufficient data")

    vols, rets, weights_list = calculate_efficient_frontier(returns, n_points=50)

    points = []
    for v, r, w in zip(vols, rets, weights_list):
        # Degenerate solves produce a NaN volatility; such a point cannot be
        # plotted or compared, so drop it rather than render it as a hole.
        vol, ret = float(v), float(r)
        if not (math.isfinite(vol) and math.isfinite(ret)) or vol <= 0:
            continue
        points.append({
            "volatility": round(vol, 4),
            "return": round(ret, 4),
            "sharpe": round((ret - 0.04) / vol, 4),
            "weights": {k: round(_finite(wv), 4) for k, wv in w.items()},
        })

    if not points:
        raise HTTPException(
            422,
            "Could not build an efficient frontier from this data — the covariance "
            "matrix is degenerate. Try a longer lookback or fewer correlated symbols.",
        )

    optimal = max(points, key=lambda p: p["sharpe"])
    return EfficientFrontierResponse(points=points, optimal_portfolio=optimal)


@router.get("/metrics")
def portfolio_metrics(
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
    w = _parse_json_param(weights, "weights") or {s: 1 / len(sym_list) for s in sym_list}

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

    return _json_safe(result)


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
def rebalance_portfolio(request: RebalanceRequest):
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
    return _json_safe(result)


# ── §4.4 Weight Drift Tracking ───────────────────────────────────────────────

@router.get("/drift")
def get_weight_drift(
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

    cv = _parse_json_param(current_values, "current_values") or {}
    drift = calculate_drift(
        target_weights=snapshot["weights"],
        current_values=cv,
        total_value=total_value,
    )
    return _json_safe({
        "drift": drift,
        "last_rebalanced": snapshot["saved_at"],
        "strategy": snapshot["strategy"],
        "needs_rebalance_count": sum(1 for v in drift.values() if v["needs_rebalance"]),
    })


# ── §5.1 Correlation Heatmap ─────────────────────────────────────────────────

@router.get("/correlation")
def get_correlation(
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
    # A symbol that never moved in the window has zero variance, so its whole row
    # of correlations is NaN; those cells serialise as null instead of 500ing.
    return _json_safe(calculate_correlation_matrix(returns_df, high_corr_threshold))


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
def simulate_portfolio(request: SimulateRequest):
    """
    Run Monte Carlo simulation for the portfolio.
    Returns 5 percentile paths (p10–p90 fan chart) and probability statistics.
    """
    returns_df, _ = _fetch_returns(request.symbols, request.lookback_days)
    if returns_df.empty:
        raise HTTPException(404, "No data")
    return _json_safe(run_monte_carlo(
        returns_df=returns_df,
        weights=request.weights,
        n_simulations=request.n_simulations,
        n_days=request.n_days,
        initial_value=request.initial_value,
    ))


# ── §5.3 Sector Allocation ───────────────────────────────────────────────────

@router.get("/sectors")
def get_sectors(
    symbols: str = "AAPL,MSFT,GOOGL",
    weights: Optional[str] = None,
):
    """
    Get sector breakdown for the portfolio.
    Returns sector-level weights, tickers per sector, and concentration warnings.

    weights: optional JSON string, e.g. '{"AAPL": 0.4, "MSFT": 0.3, "GOOGL": 0.3}'
    """
    sym_list = [s.strip().upper() for s in symbols.split(",")]
    w = _parse_json_param(weights, "weights") or {s: 1 / len(sym_list) for s in sym_list}
    return _json_safe(get_sector_allocation(w))


# ── §5.4 Risk Controls & Alerts ──────────────────────────────────────────────

@router.get("/alerts")
def get_risk_alerts(
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
    w = _parse_json_param(weights, "weights") or {s: 1 / len(sym_list) for s in sym_list}

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
    return _json_safe({
        "alerts": alerts,
        "alert_count": len(alerts),
        "critical_count": sum(1 for a in alerts if a["severity"] == "CRITICAL"),
    })
