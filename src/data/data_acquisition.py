"""
Data Acquisition Module
Functions for fetching stock constituents and bulk downloading
"""

import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from typing import List, Dict, Optional
from pathlib import Path
from tqdm import tqdm

from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_sp500_tickers() -> List[str]:
    """
    Get current S&P 500 constituent tickers from Wikipedia

    Returns
    -------
    list
        List of S&P 500 ticker symbols
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
        logger.info(f"Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        logger.error(f"Error fetching S&P 500 tickers: {e}")
        return []


def get_russell2000_sample(n: int = 100) -> List[str]:
    """
    Get a sample of Russell 2000 tickers

    Parameters
    ----------
    n : int
        Number of tickers to return

    Returns
    -------
    list
        List of ticker symbols
    """
    # Common Russell 2000 ETF and sample small-cap stocks
    sample_tickers = [
        'IWM',  # Russell 2000 ETF
        'AEHR', 'AGCO', 'AIRC', 'AMED', 'ANGO',
        'APPN', 'ARWR', 'AZTA', 'BCPC', 'BELFB',
    ]
    return sample_tickers[:n]


def download_index_data(
    index_name: str,
    start_date: str,
    end_date: str,
    output_dir: str = "data/raw/daily"
) -> Dict[str, pd.DataFrame]:
    """
    Download data for all stocks in an index

    Parameters
    ----------
    index_name : str
        Name of the index ('sp500', 'russell2000')
    start_date : str
        Start date in YYYY-MM-DD format
    end_date : str
        End date in YYYY-MM-DD format
    output_dir : str
        Directory to save downloaded data

    Returns
    -------
    dict
        Dictionary of ticker -> DataFrame
    """
    # Get tickers based on index
    if index_name.lower() == 'sp500':
        tickers = get_sp500_tickers()
    elif index_name.lower() == 'russell2000':
        tickers = get_russell2000_sample()
    else:
        logger.error(f"Unknown index: {index_name}")
        return {}

    # Create output directory
    output_path = Path(output_dir) / index_name
    output_path.mkdir(parents=True, exist_ok=True)

    data_dict = {}
    failed = []

    logger.info(f"Downloading {len(tickers)} stocks from {index_name}...")

    for ticker in tqdm(tickers, desc=f"Downloading {index_name}"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if not data.empty:
                # Save to CSV
                data.to_csv(output_path / f"{ticker}.csv")
                data_dict[ticker] = data
            else:
                failed.append(ticker)
                
        except Exception as e:
            logger.warning(f"Failed to download {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"Downloaded {len(data_dict)}/{len(tickers)} stocks")
    if failed:
        logger.warning(f"Failed tickers: {failed[:10]}{'...' if len(failed) > 10 else ''}")

    return data_dict
