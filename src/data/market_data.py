"""
Market Data Module
Fetch market data for heatmap visualization
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import streamlit as st

from ..utils.logger import get_logger

logger = get_logger(__name__)


@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_sp500_constituents() -> pd.DataFrame:
    """
    Fetch S&P 500 constituents with sector information from Wikipedia.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Symbol, Company, Sector, Industry
    """
    logger.info("Fetching S&P 500 constituents from Wikipedia...")
    
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        
        # Standardize column names
        df = df.rename(columns={
            'Symbol': 'Symbol',
            'Security': 'Company',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry'
        })
        
        # Clean up symbols (remove dots, replace with dashes for yfinance)
        df['Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        
        # Keep only needed columns
        df = df[['Symbol', 'Company', 'Sector', 'Industry']].copy()
        
        logger.info(f"Successfully fetched {len(df)} S&P 500 constituents")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching S&P 500 constituents: {e}")
        return pd.DataFrame(columns=['Symbol', 'Company', 'Sector', 'Industry'])


@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_market_quotes(symbols: List[str]) -> pd.DataFrame:
    """
    Fetch current market quotes for a list of symbols.
    
    Parameters
    ----------
    symbols : List[str]
        List of ticker symbols
    
    Returns
    -------
    pd.DataFrame
        DataFrame with price, change, volume data
    """
    logger.info(f"Fetching quotes for {len(symbols)} symbols...")
    
    try:
        # Download data for all symbols at once
        tickers = yf.Tickers(' '.join(symbols))
        
        data = []
        for symbol in symbols:
            try:
                ticker = tickers.tickers.get(symbol)
                if ticker is None:
                    continue
                    
                info = ticker.fast_info
                hist = ticker.history(period='5d')
                
                if hist.empty:
                    continue
                
                current_price = hist['Close'].iloc[-1]
                prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                
                daily_return = ((current_price - prev_close) / prev_close) * 100
                
                data.append({
                    'Symbol': symbol,
                    'Price': current_price,
                    'PrevClose': prev_close,
                    'DailyReturn': daily_return,
                    'Volume': hist['Volume'].iloc[-1] if 'Volume' in hist else 0,
                    'MarketCap': getattr(info, 'market_cap', 0) or 0
                })
                
            except Exception as e:
                logger.debug(f"Error fetching {symbol}: {e}")
                continue
        
        df = pd.DataFrame(data)
        logger.info(f"Successfully fetched quotes for {len(df)} symbols")
        return df
        
    except Exception as e:
        logger.error(f"Error fetching market quotes: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_batch_quotes(symbols: List[str], period: str = '5d') -> pd.DataFrame:
    """
    Fetch quotes in batch using yfinance download (faster).
    
    Parameters
    ----------
    symbols : List[str]
        List of ticker symbols
    period : str
        Period for historical data
    
    Returns
    -------
    pd.DataFrame
        DataFrame with price and return data
    """
    logger.info(f"Batch fetching {len(symbols)} symbols...")
    
    try:
        # Download all at once - much faster
        df_prices = yf.download(
            symbols, 
            period=period, 
            progress=False,
            threads=True
        )
        
        if df_prices.empty:
            return pd.DataFrame()
        
        # Get latest prices and calculate returns
        data = []
        
        # Handle multi-level columns from batch download
        if isinstance(df_prices.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    if ('Close', symbol) not in df_prices.columns:
                        continue
                    
                    closes = df_prices[('Close', symbol)].dropna()
                    volumes = df_prices[('Volume', symbol)].dropna() if ('Volume', symbol) in df_prices.columns else pd.Series([0])
                    
                    if len(closes) < 2:
                        continue
                    
                    current_price = closes.iloc[-1]
                    prev_close = closes.iloc[-2]
                    daily_return = ((current_price - prev_close) / prev_close) * 100
                    
                    data.append({
                        'Symbol': symbol,
                        'Price': current_price,
                        'PrevClose': prev_close,
                        'DailyReturn': daily_return,
                        'Volume': volumes.iloc[-1] if len(volumes) > 0 else 0
                    })
                except Exception:
                    continue
        else:
            # Single symbol case
            closes = df_prices['Close'].dropna()
            if len(closes) >= 2:
                data.append({
                    'Symbol': symbols[0],
                    'Price': closes.iloc[-1],
                    'PrevClose': closes.iloc[-2],
                    'DailyReturn': ((closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]) * 100,
                    'Volume': df_prices['Volume'].iloc[-1] if 'Volume' in df_prices else 0
                })
        
        result = pd.DataFrame(data)
        logger.info(f"Batch fetch complete: {len(result)} symbols")
        return result
        
    except Exception as e:
        logger.error(f"Error in batch fetch: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_market_caps(symbols: List[str]) -> Dict[str, float]:
    """
    Fetch market caps for symbols (slower, cache longer).
    
    Parameters
    ----------
    symbols : List[str]
        List of ticker symbols
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping symbol to market cap
    """
    logger.info(f"Fetching market caps for {len(symbols)} symbols...")
    
    market_caps = {}
    
    # Process in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        
        try:
            tickers = yf.Tickers(' '.join(batch))
            
            for symbol in batch:
                try:
                    ticker = tickers.tickers.get(symbol)
                    if ticker:
                        info = ticker.fast_info
                        market_caps[symbol] = getattr(info, 'market_cap', 0) or 1e9  # Default 1B
                except Exception:
                    market_caps[symbol] = 1e9  # Default
                    
        except Exception as e:
            logger.debug(f"Error fetching batch market caps: {e}")
            for symbol in batch:
                market_caps[symbol] = 1e9
    
    logger.info(f"Fetched market caps for {len(market_caps)} symbols")
    return market_caps


def get_market_heatmap_data(market: str = "S&P 500") -> pd.DataFrame:
    """
    Get complete market data for heatmap visualization.
    
    Parameters
    ----------
    market : str
        Market to fetch ("S&P 500")
    
    Returns
    -------
    pd.DataFrame
        Complete DataFrame with all heatmap data
    """
    logger.info(f"Building heatmap data for {market}...")
    
    # Get constituents with sectors
    constituents = get_sp500_constituents()
    
    if constituents.empty:
        logger.error("No constituents data available")
        return pd.DataFrame()
    
    symbols = constituents['Symbol'].tolist()
    
    # Fetch quotes in batch (faster)
    quotes = fetch_batch_quotes(symbols)
    
    if quotes.empty:
        logger.error("No quotes data available")
        return pd.DataFrame()
    
    # Fetch market caps
    market_caps = fetch_market_caps(symbols)
    
    # Merge all data
    df = constituents.merge(quotes, on='Symbol', how='inner')
    
    # Add market caps
    df['MarketCap'] = df['Symbol'].map(market_caps).fillna(1e9)
    
    # Calculate display values
    df['MarketCapB'] = df['MarketCap'] / 1e9  # In billions
    df['VolumeM'] = df['Volume'] / 1e6  # In millions
    
    # Create display text for tiles
    df['DisplayText'] = df.apply(
        lambda x: f"{x['Symbol']}<br>{x['DailyReturn']:+.1f}%", 
        axis=1
    )
    
    # Sort by market cap within sectors
    df = df.sort_values(['Sector', 'MarketCap'], ascending=[True, False])
    
    logger.info(f"Heatmap data ready: {len(df)} stocks")
    return df


def calculate_period_returns(symbols: List[str], period: str) -> pd.DataFrame:
    """
    Calculate returns for a specific time period.
    
    Parameters
    ----------
    symbols : List[str]
        List of ticker symbols
    period : str
        Period: '1D', '5D', '1M', '3M', 'YTD', '1Y'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Symbol and Return columns
    """
    # Map period to yfinance format
    period_map = {
        '1D': '5d',   # Need 5d to get previous close
        '5D': '10d',
        '1M': '1mo',
        '3M': '3mo',
        'YTD': 'ytd',
        '1Y': '1y'
    }
    
    yf_period = period_map.get(period, '5d')
    lookback = {'1D': 1, '5D': 5, '1M': 21, '3M': 63, 'YTD': None, '1Y': 252}.get(period, 1)
    
    try:
        df_prices = yf.download(symbols, period=yf_period, progress=False, threads=True)
        
        if df_prices.empty:
            return pd.DataFrame()
        
        data = []
        
        if isinstance(df_prices.columns, pd.MultiIndex):
            for symbol in symbols:
                try:
                    if ('Close', symbol) not in df_prices.columns:
                        continue
                    
                    closes = df_prices[('Close', symbol)].dropna()
                    
                    if len(closes) < 2:
                        continue
                    
                    current = closes.iloc[-1]
                    
                    if lookback and len(closes) > lookback:
                        start = closes.iloc[-lookback - 1]
                    else:
                        start = closes.iloc[0]
                    
                    period_return = ((current - start) / start) * 100
                    
                    data.append({
                        'Symbol': symbol,
                        'PeriodReturn': period_return
                    })
                except Exception:
                    continue
        
        return pd.DataFrame(data)
        
    except Exception as e:
        logger.error(f"Error calculating period returns: {e}")
        return pd.DataFrame()
