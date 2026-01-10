"""
Data Preprocessing Module
Functions for cleaning and preparing data
"""

import pandas as pd
import numpy as np
from typing import Optional, List

from ..utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by removing duplicates and sorting

    Parameters
    ----------
    df : pandas.DataFrame
        Raw stock data

    Returns
    -------
    pandas.DataFrame
        Cleaned data
    """
    data = df.copy()
    
    # Remove duplicate indices
    data = data[~data.index.duplicated(keep='first')]
    
    # Sort by date
    data = data.sort_index()
    
    # Remove rows with all NaN values
    data = data.dropna(how='all')
    
    logger.info(f"Cleaned data: {len(data)} rows remaining")
    return data


def handle_missing_values(
    df: pd.DataFrame,
    method: str = 'ffill',
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Handle missing values in stock data

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data with potential missing values
    method : str
        Method for handling missing values ('ffill', 'bfill', 'interpolate', 'drop')
    columns : list, optional
        Specific columns to process

    Returns
    -------
    pandas.DataFrame
        Data with missing values handled
    """
    data = df.copy()
    
    if columns is None:
        columns = data.columns.tolist()
    
    missing_before = data[columns].isnull().sum().sum()
    
    if method == 'ffill':
        data[columns] = data[columns].ffill()
    elif method == 'bfill':
        data[columns] = data[columns].bfill()
    elif method == 'interpolate':
        data[columns] = data[columns].interpolate(method='linear')
    elif method == 'drop':
        data = data.dropna(subset=columns)
    else:
        logger.warning(f"Unknown method: {method}. Using forward fill.")
        data[columns] = data[columns].ffill()
    
    missing_after = data[columns].isnull().sum().sum()
    logger.info(f"Missing values: {missing_before} -> {missing_after} ({method})")
    
    return data


def calculate_returns(
    df: pd.DataFrame,
    price_column: str = 'Close',
    periods: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Calculate returns for various periods

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data with price column
    price_column : str
        Name of the price column
    periods : list
        List of periods for return calculation

    Returns
    -------
    pandas.DataFrame
        Data with added return columns
    """
    data = df.copy()
    
    for period in periods:
        col_name = f'Return_{period}d' if period > 1 else 'Daily_Return'
        data[col_name] = data[price_column].pct_change(period)
    
    # Cumulative return
    data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod() - 1
    
    return data


def normalize_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize volume data using z-score

    Parameters
    ----------
    df : pandas.DataFrame
        Data with Volume column

    Returns
    -------
    pandas.DataFrame
        Data with normalized volume
    """
    data = df.copy()
    
    if 'Volume' in data.columns:
        mean_vol = data['Volume'].rolling(window=20).mean()
        std_vol = data['Volume'].rolling(window=20).std()
        data['Volume_Normalized'] = (data['Volume'] - mean_vol) / std_vol
    
    return data
