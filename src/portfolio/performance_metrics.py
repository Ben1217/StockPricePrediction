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
