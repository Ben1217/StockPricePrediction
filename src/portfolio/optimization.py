"""
Portfolio Optimization Module
Mean-variance optimization and efficient frontier
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import cvxpy as cp

from ..utils.logger import get_logger
from ..utils.config_loader import get_config_value

logger = get_logger(__name__)


def optimize_portfolio(
    returns: pd.DataFrame,
    objective: str = 'max_sharpe',
    risk_free_rate: float = 0.04,
    constraints: Optional[Dict] = None
) -> Dict[str, float]:
    """
    Optimize portfolio allocation

    Parameters
    ----------
    returns : pandas.DataFrame
        Historical returns for each asset
    objective : str
        Optimization objective: 'max_sharpe', 'min_volatility', 'max_return'
    risk_free_rate : float
        Annual risk-free rate
    constraints : dict, optional
        Constraints like max_position, min_position

    Returns
    -------
    dict
        Optimal weights for each asset
    """
    if constraints is None:
        constraints = get_config_value('portfolio.optimization.constraints', {
            'max_position': 0.15,
            'min_position': 0.05,
        })

    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252  # Annualize
    cov_matrix = returns.cov() * 252

    # Define optimization variable
    weights = cp.Variable(n_assets)

    # Calculate portfolio return and volatility
    portfolio_return = mean_returns.values @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix.values)

    # Constraints
    cons = [
        cp.sum(weights) == 1,  # Weights sum to 1
        weights >= constraints.get('min_position', 0),  # Min weight
        weights <= constraints.get('max_position', 1),  # Max weight
    ]

    # Objective
    if objective == 'max_sharpe':
        # Maximize Sharpe ratio (approximation)
        obj = cp.Maximize(portfolio_return - risk_free_rate)
        cons.append(portfolio_variance <= 0.04)  # Target volatility constraint
    elif objective == 'min_volatility':
        obj = cp.Minimize(portfolio_variance)
    elif objective == 'max_return':
        obj = cp.Maximize(portfolio_return)
    else:
        raise ValueError(f"Unknown objective: {objective}")

    # Solve
    problem = cp.Problem(obj, cons)
    try:
        problem.solve()
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return {col: 1/n_assets for col in returns.columns}  # Equal weight fallback

    if weights.value is None:
        logger.warning("Optimization did not converge, using equal weights")
        return {col: 1/n_assets for col in returns.columns}

    # Create result dictionary
    result = {col: w for col, w in zip(returns.columns, weights.value)}

    logger.info(f"Portfolio optimized with {objective} objective")
    return result


def calculate_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 100
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Calculate efficient frontier points

    Parameters
    ----------
    returns : pandas.DataFrame
        Historical returns
    n_points : int
        Number of points on the frontier

    Returns
    -------
    tuple
        (volatilities, expected_returns, weights_list)
    """
    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), n_points)

    volatilities = []
    expected_returns = []
    weights_list = []

    for target in target_returns:
        weights = cp.Variable(n_assets)
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)

        constraints = [
            cp.sum(weights) == 1,
            weights >= 0,
            mean_returns.values @ weights >= target
        ]

        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        try:
            problem.solve()
            if weights.value is not None:
                vol = np.sqrt(portfolio_variance.value)
                ret = mean_returns.values @ weights.value

                volatilities.append(vol)
                expected_returns.append(ret)
                weights_list.append({col: w for col, w in zip(returns.columns, weights.value)})
        except:
            continue

    return np.array(volatilities), np.array(expected_returns), weights_list


def get_optimal_weights(
    returns: pd.DataFrame,
    predictions: Dict[str, float],
    method: str = 'combined'
) -> Dict[str, float]:
    """
    Get optimal weights combining historical optimization and predictions

    Parameters
    ----------
    returns : pandas.DataFrame
        Historical returns
    predictions : dict
        Predicted returns for each asset
    method : str
        Method: 'historical', 'prediction', 'combined'

    Returns
    -------
    dict
        Optimal weights
    """
    if method == 'historical':
        return optimize_portfolio(returns)
    elif method == 'prediction':
        # Simple prediction-based allocation
        total = sum(max(0, p) for p in predictions.values())
        if total == 0:
            n = len(predictions)
            return {k: 1/n for k in predictions}
        return {k: max(0, v) / total for k, v in predictions.items()}
    else:
        # Combined approach
        hist_weights = optimize_portfolio(returns)
        pred_weights = get_optimal_weights(returns, predictions, 'prediction')

        return {
            k: 0.5 * hist_weights.get(k, 0) + 0.5 * pred_weights.get(k, 0)
            for k in set(hist_weights) | set(pred_weights)
        }
