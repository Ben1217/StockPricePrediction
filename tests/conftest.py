"""
pytest configuration and fixtures
"""

import os

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.model_manager import AUTO_PREPARE_ENV
from src.models.preparation import registry as preparation_registry


# Models whose "unit" test performs real foundation-model inference. Kronos is a
# ~102M-parameter transformer sampled Monte-Carlo style: the repo's own benchmark
# notes budget roughly ten seconds PER ROW on CPU, and
# TestExpandingWindowSplits::test_interface_and_probability_range scores 200 test
# rows at sample_count=128 for each of these names. That is half an hour per
# parameter, so `pytest tests/unit` never reached its summary line -- it stalled
# at 117 of 583 tests and looked like a hang.
#
# The same reasoning as _no_automatic_training below: a unit run must not kick
# off multi-minute real work. These are deselected by default and reported as
# skips (never silently dropped), and set QV_RUN_SLOW_MODELS=1 to run them.
SLOW_MODEL_IDS = ("kronos", "tabpfn", "foundation_ensemble")
SLOW_MODELS_ENV = "QV_RUN_SLOW_MODELS"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow_model: performs real foundation-model inference; deselected unless "
        f"{SLOW_MODELS_ENV}=1",
    )


def pytest_collection_modifyitems(config, items):
    """Skip the heavyweight foundation-model parametrisations by default."""
    if os.environ.get(SLOW_MODELS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    skip = pytest.mark.skip(
        reason=(
            "runs real foundation-model inference (minutes to hours on CPU); "
            f"set {SLOW_MODELS_ENV}=1 to include it"
        )
    )
    for item in items:
        # Match the parametrised id, e.g. "...[foundation_ensemble]", rather than
        # the test name, so only the heavy parameters are affected.
        _, bracket, remainder = item.name.partition("[")
        parameter = remainder[:-1] if bracket else ""
        if parameter in SLOW_MODEL_IDS or "slow_model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_automatic_training(monkeypatch):
    """
    Keep the suite offline.

    Serving routes now start model preparation when they find nothing to serve,
    and preparation downloads years of bars and fits models on a background
    thread. A test that hits one of those routes would otherwise kick off a real
    multi-minute training run whose output lands in the working tree — so
    automatic starts are off by default and a test that wants one turns it on
    explicitly with `monkeypatch.setenv(AUTO_PREPARE_ENV, "true")`.
    """
    monkeypatch.setenv(AUTO_PREPARE_ENV, "false")
    preparation_registry.reset()
    yield
    preparation_registry.reset()


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    data = pd.DataFrame({
        'Open': close + np.random.randn(100) * 0.2,
        'High': close + np.abs(np.random.randn(100)) * 0.5,
        'Low': close - np.abs(np.random.randn(100)) * 0.5,
        'Close': close,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_returns():
    """Generate sample returns for testing"""
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)
    return returns
