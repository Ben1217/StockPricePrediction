"""
Data Storage Module
Functions for saving and loading data from database
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List

from ..utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = "database/stock_data.db"


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Get database connection"""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_database(db_path: str = DEFAULT_DB_PATH, schema_path: str = "database/schema.sql"):
    """
    Initialize database with schema

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    schema_path : str
        Path to SQL schema file
    """
    try:
        conn = get_connection(db_path)
        
        with open(schema_path, 'r') as f:
            schema = f.read()
        
        conn.executescript(schema)
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {db_path}")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def save_to_database(
    df: pd.DataFrame,
    table_name: str,
    symbol: str,
    db_path: str = DEFAULT_DB_PATH,
    if_exists: str = 'append'
) -> bool:
    """
    Save DataFrame to database

    Parameters
    ----------
    df : pandas.DataFrame
        Data to save
    table_name : str
        Target table name
    symbol : str
        Stock symbol
    db_path : str
        Database path
    if_exists : str
        How to handle existing data ('append', 'replace', 'fail')

    Returns
    -------
    bool
        True if successful
    """
    try:
        conn = get_connection(db_path)
        
        data = df.copy()
        data['symbol'] = symbol
        
        if 'Date' not in data.columns and data.index.name:
            data = data.reset_index()
            data = data.rename(columns={data.columns[0]: 'date'})
        
        data.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()
        
        logger.info(f"Saved {len(data)} rows to {table_name} for {symbol}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save to database: {e}")
        return False


def load_from_database(
    table_name: str,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> Optional[pd.DataFrame]:
    """
    Load data from database

    Parameters
    ----------
    table_name : str
        Source table name
    symbol : str, optional
        Filter by symbol
    start_date : str, optional
        Filter by start date
    end_date : str, optional
        Filter by end date
    db_path : str
        Database path

    Returns
    -------
    pandas.DataFrame or None
        Loaded data
    """
    try:
        conn = get_connection(db_path)
        
        query = f"SELECT * FROM {table_name}"
        conditions = []
        
        if symbol:
            conditions.append(f"symbol = '{symbol}'")
        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, parse_dates=['date'])
        conn.close()
        
        if not df.empty:
            df = df.set_index('date')
        
        logger.info(f"Loaded {len(df)} rows from {table_name}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load from database: {e}")
        return None
