"""
Data Loader Module
Downloads and loads stock market data from Yahoo Finance
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def download_stock_data(ticker, start_date, end_date):
    """
    Download historical stock data from Yahoo Finance
    
    Parameters:
    -----------
    ticker : str
        Stock ticker symbol (e.g., 'SPY', 'AAPL')
    start_date : str or datetime
        Start date for data download
    end_date : str or datetime
        End date for data download
    
    Returns:
    --------
    pandas.DataFrame
        Historical OHLCV data
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✅ Downloaded {len(data)} days of data for {ticker}")
        return data
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return None

def load_multiple_stocks(tickers, start_date, end_date):
    """
    Download data for multiple stocks
    
    Parameters:
    -----------
    tickers : list
        List of ticker symbols
    start_date : str or datetime
        Start date
    end_date : str or datetime
        End date
    
    Returns:
    --------
    dict
        Dictionary with ticker as key and DataFrame as value
    """
    data_dict = {}
    for ticker in tickers:
        data = download_stock_data(ticker, start_date, end_date)
        if data is not None:
            data_dict[ticker] = data
    return data_dict

# Future functions to add:
# - save_data()
# - load_data()
# - update_data()
