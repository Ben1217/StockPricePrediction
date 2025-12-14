"""
Feature Engineering Module
Calculates technical indicators for stock data
"""

import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df):
    """
    Add technical indicators to stock data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Stock data with OHLCV columns
    
    Returns:
    --------
    pandas.DataFrame
        Data with added technical indicators
    """
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Moving Averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Low'] = bollinger.bollinger_lband()
    
    # ATR (Average True Range)
    data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
    
    print(f"✅ Added {len(data.columns) - len(df.columns)} technical indicators")
    
    return data

# Future functions to add:
# - calculate_momentum_indicators()
# - calculate_volatility_indicators()
# - calculate_volume_indicators()
