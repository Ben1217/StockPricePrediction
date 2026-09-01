"""
pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.models.model_manager import AUTO_PREPARE_ENV
from src.models.preparation import registry as preparation_registry


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
