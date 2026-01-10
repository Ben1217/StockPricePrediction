"""
Technical Indicators Module
Calculate all technical indicators for stock data
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)


def add_all_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to stock data

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data with OHLCV columns

    Returns
    -------
    pandas.DataFrame
        Data with all technical indicators added
    """
    data = df.copy()
    
    initial_cols = len(data.columns)
    
    # Add returns first (needed for predictions)
    data['Returns'] = data['Close'].pct_change()
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Add all indicator categories
    data = add_trend_indicators(data)
    data = add_momentum_indicators(data)
    data = add_volatility_indicators(data)
    data = add_volume_indicators(data)
    
    new_cols = len(data.columns) - initial_cols
    logger.info(f"Added {new_cols} technical indicators")
    
    return data


def add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend-following indicators"""
    data = df.copy()
    
    close = data['Close']
    
    # Simple Moving Averages
    data['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    data['SMA_50'] = ta.trend.sma_indicator(close, window=50)
    data['SMA_200'] = ta.trend.sma_indicator(close, window=200)
    
    # Exponential Moving Averages
    data['EMA_12'] = ta.trend.ema_indicator(close, window=12)
    data['EMA_26'] = ta.trend.ema_indicator(close, window=26)
    data['EMA_50'] = ta.trend.ema_indicator(close, window=50)
    
    # MACD
    macd = ta.trend.MACD(close)
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['MACD_Histogram'] = macd.macd_diff()
    
    # ADX (Average Directional Index)
    data['ADX'] = ta.trend.adx(data['High'], data['Low'], close)
    
    # Ichimoku Cloud
    ichimoku = ta.trend.IchimokuIndicator(data['High'], data['Low'])
    data['Ichimoku_A'] = ichimoku.ichimoku_a()
    data['Ichimoku_B'] = ichimoku.ichimoku_b()
    
    return data


def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators"""
    data = df.copy()
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # RSI
    data['RSI'] = ta.momentum.rsi(close, window=14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    
    # Williams %R
    data['Williams_R'] = ta.momentum.williams_r(high, low, close)
    
    # ROC (Rate of Change)
    data['ROC'] = ta.momentum.roc(close, window=10)
    
    # Ultimate Oscillator
    data['Ultimate_Osc'] = ta.momentum.ultimate_oscillator(high, low, close)
    
    # TSI (True Strength Index)
    data['TSI'] = ta.momentum.tsi(close)
    
    return data


def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators"""
    data = df.copy()
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    
    # ATR (Average True Range)
    data['ATR'] = ta.volatility.average_true_range(high, low, close)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close)
    data['BB_High'] = bollinger.bollinger_hband()
    data['BB_Mid'] = bollinger.bollinger_mavg()
    data['BB_Low'] = bollinger.bollinger_lband()
    data['BB_Width'] = bollinger.bollinger_wband()
    data['BB_Pband'] = bollinger.bollinger_pband()
    
    # Keltner Channel
    keltner = ta.volatility.KeltnerChannel(high, low, close)
    data['KC_High'] = keltner.keltner_channel_hband()
    data['KC_Low'] = keltner.keltner_channel_lband()
    
    # Donchian Channel
    donchian = ta.volatility.DonchianChannel(high, low, close)
    data['DC_High'] = donchian.donchian_channel_hband()
    data['DC_Low'] = donchian.donchian_channel_lband()
    
    return data


def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators"""
    data = df.copy()
    
    close = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']
    
    # OBV (On-Balance Volume)
    data['OBV'] = ta.volume.on_balance_volume(close, volume)
    
    # CMF (Chaikin Money Flow)
    data['CMF'] = ta.volume.chaikin_money_flow(high, low, close, volume)
    
    # MFI (Money Flow Index)
    data['MFI'] = ta.volume.money_flow_index(high, low, close, volume)
    
    # VWAP (Volume Weighted Average Price) - simplified
    data['VWAP'] = (volume * (high + low + close) / 3).cumsum() / volume.cumsum()
    
    # Volume SMA
    data['Volume_SMA_20'] = volume.rolling(window=20).mean()
    data['Volume_Ratio'] = volume / data['Volume_SMA_20']
    
    return data
