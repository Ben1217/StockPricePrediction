"""
Tests for the parametrised efficient frontier.

The frontier used to rebuild and re-canonicalise its QP once per point. It now
builds one problem with a `cp.Parameter` target and re-solves it, which is ~4x
faster on a typical basket. These pin the property that matters: the answers are
the same ones, and every returned point is actually on the feasible set.
"""

import numpy as np
import pandas as pd
import pytest

import cvxpy as cp

from src.portfolio.optimization import calculate_efficient_frontier


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.normal(0.0005, 0.012, (500, 8)),
        columns=[f"S{i}" for i in range(8)],
        index=pd.date_range("2023-01-01", periods=500, freq="B"),
    )


def _rebuild_per_point(returns, n_points):
    """The original implementation, kept here as the reference answer."""
    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    vols, rets, weights_list = [], [], []
    for target in np.linspace(mean_returns.min(), mean_returns.max(), n_points):
        weights = cp.Variable(n_assets)
        variance = cp.quad_form(weights, cov_matrix.values)
        problem = cp.Problem(
            cp.Minimize(variance),
            [cp.sum(weights) == 1, weights >= 0, mean_returns.values @ weights >= target],
        )
        problem.solve()
        if weights.value is not None:
            vols.append(np.sqrt(variance.value))
            rets.append(mean_returns.values @ weights.value)
            weights_list.append(dict(zip(returns.columns, weights.value)))
    return np.array(vols), np.array(rets), weights_list


def test_matches_the_per_point_rebuild_it_replaced(returns):
    old_vols, old_rets, old_weights = _rebuild_per_point(returns, 30)
    new_vols, new_rets, new_weights = calculate_efficient_frontier(returns, n_points=30)

    assert len(new_vols) == len(old_vols)
    # Solver tolerance, not algorithmic difference: these agree far below any
    # precision a weight or a volatility is ever displayed at.
    assert np.allclose(new_vols, old_vols, rtol=1e-7, atol=1e-10)
    assert np.allclose(new_rets, old_rets, rtol=1e-7, atol=1e-10)
    for old, new in zip(old_weights, new_weights):
        assert np.allclose([old[k] for k in old], [new[k] for k in old], atol=1e-7)


def test_every_point_satisfies_the_constraints(returns):
    _, _, weights_list = calculate_efficient_frontier(returns, n_points=25)
    assert weights_list
    for weights in weights_list:
        values = np.array(list(weights.values()))
        assert values.sum() == pytest.approx(1.0, abs=1e-8)
        # Long-only. A tiny negative residual is the interior-point solver's, and
        # the tightened tolerances keep it at machine-noise scale rather than the
        # 1e-5 the default settings returned.
        assert values.min() > -1e-9


def test_the_frontier_is_monotone_in_return(returns):
    vols, rets, _ = calculate_efficient_frontier(returns, n_points=25)
    # Higher target return costs at least as much volatility, which is what makes
    # the curve a frontier rather than a scatter.
    assert np.all(np.diff(rets) >= -1e-9)
    assert np.all(np.diff(vols) >= -1e-6)


def test_the_problem_is_built_once_not_once_per_point(returns, monkeypatch):
    """The point of the change: one canonicalisation, many solves."""
    built = []
    original = cp.Problem

    class CountingProblem(original):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cp, "Problem", CountingProblem)
    calculate_efficient_frontier(returns, n_points=40)
    assert sum(built) == 1
