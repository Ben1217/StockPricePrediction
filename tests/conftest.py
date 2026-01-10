"""
pytest configuration and fixtures
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


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
