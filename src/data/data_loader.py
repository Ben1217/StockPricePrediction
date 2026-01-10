"""
Data Loader Module
Downloads and loads stock market data from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from ..utils.logger import get_logger

logger = get_logger(__name__)


def download_stock_data(
    ticker: str,
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """
    Download historical stock data from Yahoo Finance

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'AAPL')
    start_date : str or datetime
        Start date for data download
    end_date : str or datetime
        End date for data download
    interval : str
        Data interval ('1d', '1h', '15m', '5m', '1m')

    Returns
    -------
    pandas.DataFrame or None
        Historical OHLCV data, or None if download fails
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, 
                          interval=interval, progress=False)
        
        # Fix MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if not data.empty:
            logger.info(f"Downloaded {len(data)} records for {ticker}")
            return data
        else:
            logger.warning(f"No data returned for {ticker}")
            return None
            
    except Exception as e:
        logger.error(f"Error downloading {ticker}: {e}")
        return None


def load_multiple_stocks(
    tickers: List[str],
    start_date: Union[str, datetime],
    end_date: Union[str, datetime],
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Download data for multiple stocks

    Parameters
    ----------
    tickers : list
        List of ticker symbols
    start_date : str or datetime
        Start date
    end_date : str or datetime
        End date
    interval : str
        Data interval

    Returns
    -------
    dict
        Dictionary with ticker as key and DataFrame as value
    """
    data_dict = {}
    
    for ticker in tickers:
        data = download_stock_data(ticker, start_date, end_date, interval)
        if data is not None:
            data_dict[ticker] = data
            
    logger.info(f"Successfully loaded {len(data_dict)}/{len(tickers)} stocks")
    return data_dict


def load_from_csv(filepath: str, parse_dates: bool = True) -> Optional[pd.DataFrame]:
    """
    Load stock data from a CSV file

    Parameters
    ----------
    filepath : str
        Path to the CSV file
    parse_dates : bool
        Whether to parse the index as dates

    Returns
    -------
    pandas.DataFrame or None
        Loaded data
    """
    try:
        if parse_dates:
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        else:
            data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} rows from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def save_to_csv(data: pd.DataFrame, filepath: str) -> bool:
    """
    Save stock data to a CSV file

    Parameters
    ----------
    data : pandas.DataFrame
        Data to save
    filepath : str
        Destination path

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        data.to_csv(filepath)
        logger.info(f"Saved {len(data)} rows to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")
        return False
