"""
Portfolio Performance Metrics
Calculate risk-adjusted returns and performance statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


def calculate_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio

    Parameters
    ----------
    returns : pandas.Series or np.ndarray
        Daily returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Trading periods per year

    Returns
    -------
    float
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio (using downside deviation)

    Parameters
    ----------
    returns : pandas.Series or np.ndarray
        Daily returns
    risk_free_rate : float
        Annual risk-free rate
    periods_per_year : int
        Trading periods per year

    Returns
    -------
    float
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    target_return = risk_free_rate / periods_per_year
    excess_returns = returns - target_return

    # Downside returns only
    downside_returns = np.where(excess_returns < 0, excess_returns, 0)
    downside_std = np.std(downside_returns)

    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std
    return sortino * np.sqrt(periods_per_year)


def calculate_max_drawdown(
    returns: Union[pd.Series, np.ndarray]
) -> float:
    """
    Calculate maximum drawdown

    Parameters
    ----------
    returns : pandas.Series or np.ndarray
        Daily returns

    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    return drawdown.min()


def calculate_calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)

    Parameters
    ----------
    returns : pandas.Series or np.ndarray
        Daily returns
    periods_per_year : int
        Trading periods per year

    Returns
    -------
    float
        Calmar ratio
    """
    annual_return = np.mean(returns) * periods_per_year
    max_dd = abs(calculate_max_drawdown(returns))

    if max_dd == 0:
        return 0.0

    return annual_return / max_dd


def calculate_portfolio_metrics(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.04
) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio metrics

    Parameters
    ----------
    returns : pandas.Series or np.ndarray
        Daily returns
    risk_free_rate : float
        Annual risk-free rate

    Returns
    -------
    dict
        Dictionary of metrics
    """
    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return {}

    # Basic statistics
    total_return = (1 + returns).prod() - 1
    annual_return = np.mean(returns) * 252
    annual_volatility = np.std(returns) * np.sqrt(252)

    # Risk metrics
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()

    # Performance ratios
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns)

    # Win rate
    win_rate = (returns > 0).sum() / len(returns)

    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'calmar_ratio': calmar,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'win_rate': win_rate,
        'num_periods': len(returns),
    }

    return metrics


def calculate_contribution(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
) -> Dict:
    """
    Per-stock contribution to total portfolio return.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame with columns = tickers, index = dates, values = daily returns
    weights : dict
        {ticker: weight_float}

    Returns
    -------
    dict
        by_stock breakdown, top_contributor, bottom_contributor
    """
    contributions = {}
    for ticker, w in weights.items():
        if ticker not in returns_df.columns:
            continue
        stock_returns = returns_df[ticker]
        annual_return = float(stock_returns.mean() * 252)
        annual_vol = float(stock_returns.std() * np.sqrt(252))
        contribution = w * annual_return
        positive_days = int((stock_returns > 0).sum())
        negative_days = int((stock_returns < 0).sum())

        contributions[ticker] = {
            "weight": round(w, 4),
            "stock_annual_return": round(annual_return, 4),
            "stock_annual_volatility": round(annual_vol, 4),
            "contribution_to_portfolio": round(contribution, 4),
            "positive_days": positive_days,
            "negative_days": negative_days,
            "stock_sharpe": round(
                annual_return / annual_vol if annual_vol > 0 else 0, 3
            ),
        }

    ranked = sorted(
        contributions.items(),
        key=lambda x: x[1]["contribution_to_portfolio"],
        reverse=True,
    )
    return {
        "by_stock": dict(ranked),
        "top_contributor": ranked[0][0] if ranked else None,
        "bottom_contributor": ranked[-1][0] if ranked else None,
    }


def calculate_correlation_matrix(
    returns_df: pd.DataFrame,
    high_corr_threshold: float = 0.80,
) -> Dict:
    """
    Compute pairwise Pearson correlation between all holdings.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame columns=tickers, values=daily returns
    high_corr_threshold : float
        Threshold for flagging high correlation pairs

    Returns
    -------
    dict
        matrix (nested dict), tickers, high_corr_pairs, avg_correlation
    """
    corr = returns_df.corr().round(3)
    tickers = list(corr.columns)

    high_corr_pairs = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            val = corr.loc[t1, t2]
            if abs(val) >= high_corr_threshold:
                high_corr_pairs.append({
                    "ticker_a": t1,
                    "ticker_b": t2,
                    "correlation": float(val),
                    "warning": "HIGH CORRELATION — poor diversification",
                })

    # Average correlation excluding diagonal
    mask = ~np.eye(len(tickers), dtype=bool)
    avg_corr = float(corr.values[mask].mean()) if len(tickers) > 1 else 0.0

    return {
        "matrix": corr.to_dict(),
        "tickers": tickers,
        "high_corr_pairs": high_corr_pairs,
        "avg_correlation": round(avg_corr, 3),
    }


def run_monte_carlo(
    returns_df: pd.DataFrame,
    weights: Dict[str, float],
    n_simulations: int = 1000,
    n_days: int = 252,
    initial_value: float = 100000.0,
) -> Dict:
    """
    Simulate n_simulations possible future paths for the portfolio
    using historical return distribution (mean + std dev).

    Parameters
    ----------
    returns_df : pandas.DataFrame
        Historical daily returns per ticker
    weights : dict
        {ticker: weight}
    n_simulations : int
        Number of simulation paths
    n_days : int
        Forecast horizon in trading days
    initial_value : float
        Starting portfolio value in $

    Returns
    -------
    dict
        Percentile paths for fan chart, summary statistics, probabilities
    """
    w_arr = np.array([weights.get(t, 0) for t in returns_df.columns])
    port_daily = returns_df.values @ w_arr

    mean_r = port_daily.mean()
    std_r = port_daily.std()

    np.random.seed(42)
    sim_matrix = np.zeros((n_simulations, n_days))
    for i in range(n_simulations):
        daily_r = np.random.normal(mean_r, std_r, n_days)
        sim_matrix[i] = initial_value * np.cumprod(1 + daily_r)

    final_values = sim_matrix[:, -1]

    return {
        "path_p10": np.percentile(sim_matrix, 10, axis=0).tolist(),
        "path_p25": np.percentile(sim_matrix, 25, axis=0).tolist(),
        "path_p50": np.percentile(sim_matrix, 50, axis=0).tolist(),
        "path_p75": np.percentile(sim_matrix, 75, axis=0).tolist(),
        "path_p90": np.percentile(sim_matrix, 90, axis=0).tolist(),
        "final_median": round(float(np.median(final_values)), 2),
        "final_p10": round(float(np.percentile(final_values, 10)), 2),
        "final_p90": round(float(np.percentile(final_values, 90)), 2),
        "worst_case": round(float(final_values.min()), 2),
        "best_case": round(float(final_values.max()), 2),
        "prob_profit": round(float((final_values > initial_value).mean()), 4),
        "prob_gain_20pct": round(float((final_values > initial_value * 1.20).mean()), 4),
        "prob_loss_20pct": round(float((final_values < initial_value * 0.80).mean()), 4),
        "n_simulations": n_simulations,
        "n_days": n_days,
        "initial_value": initial_value,
    }
