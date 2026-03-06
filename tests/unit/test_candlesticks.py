"""
Unit test: Candlestick Pattern Detection (Item 5)
Validates that detect_candlestick_patterns correctly identifies known setups.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.features.candlestick_patterns import detect_candlestick_patterns


@pytest.fixture
def sample_ohlc():
    """Create a mock OHLC DataFrame with specific patterns."""
    data = {
        'Open':  [100, 105, 102, 100, 100],
        'High':  [105, 110, 108, 105, 101],
        'Low':   [95,  100, 101, 80,  99],
        'Close': [104, 102, 103, 104, 100],
    }
    # Index 0: Bullish wide bar
    # Index 1: Bearish normal bar
    # Index 2: Doji (Open=102, Close=103, High=108, Low=101) -> body=1, range=7
    # Index 3: Hammer (Open=100, Close=104, High=105, Low=80) 
    #          -> body=4, lower=20, upper=1
    # Index 4: Doji (Open=100, Close=100, High=101, Low=99) -> body=0
    return pd.DataFrame(data)

def test_detect_doji(sample_ohlc):
    """Should detect Doji at index 4 (body == 0)."""
    patterns = detect_candlestick_patterns(sample_ohlc)
    assert patterns['cdl_doji'].iloc[4] == 1, "Doji not detected on zero body"

def test_detect_hammer(sample_ohlc):
    """Should detect Hammer at index 3 (long lower wick)."""
    patterns = detect_candlestick_patterns(sample_ohlc)
    assert patterns['cdl_hammer'].iloc[3] == 1, "Hammer not detected"

def test_candlestick_output_shape(sample_ohlc):
    """Output should have same length and 9 pattern columns."""
    patterns = detect_candlestick_patterns(sample_ohlc)
    assert len(patterns) == len(sample_ohlc)
    assert len(patterns.columns) == 9
    assert all(col.startswith('cdl_') for col in patterns.columns)
