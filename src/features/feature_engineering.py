"""
Feature Engineering Module
Create features and prepare data for machine learning models
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .technical_indicators import add_all_technical_indicators
from ..utils.logger import get_logger

logger = get_logger(__name__)


def create_features(
    df: pd.DataFrame,
    include_technical: bool = True,
    include_lags: bool = True,
    lag_periods: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Create all features for model training

    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock data with OHLCV columns
    include_technical : bool
        Whether to add technical indicators
    include_lags : bool
        Whether to add lagged features
    lag_periods : list
        Periods for lag features

    Returns
    -------
    pandas.DataFrame
        Data with all features
    """
    data = df.copy()
    
    # Add technical indicators
    if include_technical:
        data = add_all_technical_indicators(data)
    
    # Calculate returns
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    # Price-based features
    data['High_Low_Range'] = (data['High'] - data['Low']) / data['Close']
    data['Open_Close_Range'] = (data['Close'] - data['Open']) / data['Open']
    
    # Add lagged features
    if include_lags:
        for lag in lag_periods:
            data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
            data[f'Return_Lag_{lag}'] = data['Daily_Return'].shift(lag)
            data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
    
    # Day of week (if datetime index)
    if isinstance(data.index, pd.DatetimeIndex):
        data['DayOfWeek'] = data.index.dayofweek
        data['Month'] = data.index.month
        data['Quarter'] = data.index.quarter
    
    logger.info(f"Created features: {len(data.columns)} total columns")
    
    return data


def create_target_variable(
    df: pd.DataFrame,
    target_type: str = 'return',
    horizon: int = 1,
    classification_threshold: float = 0.0
) -> pd.DataFrame:
    """
    Create target variable for prediction

    Parameters
    ----------
    df : pandas.DataFrame
        Data with Close prices
    target_type : str
        Type of target: 'return', 'price', 'direction'
    horizon : int
        Prediction horizon (days ahead)
    classification_threshold : float
        Threshold for direction classification

    Returns
    -------
    pandas.DataFrame
        Data with target column added
    """
    data = df.copy()
    
    if target_type == 'return':
        data['Target'] = data['Close'].pct_change(horizon).shift(-horizon)
    elif target_type == 'price':
        data['Target'] = data['Close'].shift(-horizon)
    elif target_type == 'direction':
        future_return = data['Close'].pct_change(horizon).shift(-horizon)
        data['Target'] = (future_return > classification_threshold).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")
    
    logger.info(f"Created target variable: {target_type}, horizon={horizon}")
    
    return data


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    target_column: str = 'Target',
    scaler_type: str = 'minmax',
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Prepare features for model training

    Parameters
    ----------
    df : pandas.DataFrame
        Data with features and target
    feature_columns : list, optional
        Columns to use as features (None = all numeric except target)
    target_column : str
        Name of target column
    scaler_type : str
        Type of scaler: 'minmax', 'standard', or None
    test_size : float
        Proportion for test set

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    data = df.copy()
    
    # Remove rows with NaN in target
    data = data.dropna(subset=[target_column])
    
    # Select features
    if feature_columns is None:
        exclude = [target_column, 'Target', 'Close', 'Open', 'High', 'Low']
        feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col not in exclude]
    
    # Drop rows with NaN in features
    data = data.dropna(subset=feature_columns)
    
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Time series split (no shuffling)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = None
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif scaler_type == 'standard':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    logger.info(f"Prepared data: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test, scaler


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM model

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    sequence_length : int
        Length of each sequence

    Returns
    -------
    tuple
        X_sequences, y_sequences
    """
    X_seq, y_seq = [], []
    
    for i in range(sequence_length, len(X)):
        X_seq.append(X[i-sequence_length:i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)
