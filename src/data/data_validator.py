"""
Data Validator Module
Functions for validating data quality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)


def validate_data_quality(df: pd.DataFrame, ticker: str = "") -> Dict[str, any]:
    """
    Validate stock data quality and return a report

    Parameters
    ----------
    df : pandas.DataFrame
        Stock data to validate
    ticker : str
        Ticker symbol for logging

    Returns
    -------
    dict
        Validation report with issues found
    """
    report = {
        "ticker": ticker,
        "total_rows": len(df),
        "date_range": None,
        "missing_values": {},
        "duplicate_dates": 0,
        "negative_prices": 0,
        "zero_volume_days": 0,
        "outliers": {},
        "is_valid": True,
        "issues": []
    }
    
    if df.empty:
        report["is_valid"] = False
        report["issues"].append("DataFrame is empty")
        return report
    
    # Date range
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        report["date_range"] = (str(df.index.min()), str(df.index.max()))
    
    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    if any(missing > len(df) * 0.05):  # More than 5% missing
        report["issues"].append("Significant missing values detected")
    
    # Duplicate dates
    report["duplicate_dates"] = df.index.duplicated().sum()
    if report["duplicate_dates"] > 0:
        report["issues"].append(f"{report['duplicate_dates']} duplicate dates found")
    
    # Negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                report["negative_prices"] += neg_count
                report["issues"].append(f"Negative {col} prices found")
    
    # Zero volume
    if 'Volume' in df.columns:
        report["zero_volume_days"] = (df['Volume'] == 0).sum()
        if report["zero_volume_days"] > len(df) * 0.1:
            report["issues"].append("More than 10% zero volume days")
    
    # Outliers (using IQR method)
    for col in price_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 3*IQR) | (df[col] > Q3 + 3*IQR)).sum()
            if outliers > 0:
                report["outliers"][col] = outliers
    
    # Set validity
    report["is_valid"] = len(report["issues"]) == 0
    
    if report["is_valid"]:
        logger.info(f"{ticker}: Data validation passed")
    else:
        logger.warning(f"{ticker}: Data validation issues: {report['issues']}")
    
    return report


def check_data_completeness(
    df: pd.DataFrame,
    expected_columns: List[str] = None
) -> Tuple[bool, List[str]]:
    """
    Check if data has all expected columns

    Parameters
    ----------
    df : pandas.DataFrame
        Data to check
    expected_columns : list
        List of expected column names

    Returns
    -------
    tuple
        (is_complete, missing_columns)
    """
    if expected_columns is None:
        expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    missing = [col for col in expected_columns if col not in df.columns]
    is_complete = len(missing) == 0
    
    return is_complete, missing


def detect_gaps(df: pd.DataFrame, freq: str = 'B') -> List[pd.Timestamp]:
    """
    Detect gaps in time series data

    Parameters
    ----------
    df : pandas.DataFrame
        Time series data with DatetimeIndex
    freq : str
        Expected frequency ('B' for business days, 'D' for daily)

    Returns
    -------
    list
        List of missing dates
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return []
    
    expected_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    missing_dates = expected_range.difference(df.index)
    
    return missing_dates.tolist()
