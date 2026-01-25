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


def get_postgres_connection_string(
    user: str = "postgres",
    password: str = "password",
    host: str = "localhost",
    port: str = "5432",
    dbname: str = "stock_data"
) -> str:
    """Get PostgreSQL connection string"""
    import os
    user = os.getenv("POSTGRES_USER", user)
    password = os.getenv("POSTGRES_PASSWORD", password)
    host = os.getenv("POSTGRES_HOST", host)
    port = os.getenv("POSTGRES_PORT", port)
    dbname = os.getenv("POSTGRES_DB", dbname)
    
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_db_engine():
    """Get SQLAlchemy engine for PostgreSQL"""
    from sqlalchemy import create_engine
    return create_engine(get_postgres_connection_string())


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
    if_exists: str = 'append',
    use_postgres: bool = False
) -> bool:
    """
    Save DataFrame to database (SQLite or PostgreSQL)

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
        if use_postgres:
            engine = get_db_engine()
            # PostgreSQL requires slightly different handling
            data = df.copy()
            data['symbol'] = symbol
             
            if 'Date' not in data.columns and data.index.name:
                data = data.reset_index()
                data = data.rename(columns={data.columns[0]: 'date'})
                
            # Rename 'date' to 'timestamp' for intraday tables if needed, 
            # but schema uses 'date' for daily and 'timestamp' for intraday.
            # We need to handle this mapping carefully.
            if table_name == 'intraday_prices' and 'date' in data.columns:
                 data = data.rename(columns={'date': 'timestamp'})
            
            data.to_sql(table_name, engine, if_exists=if_exists, index=False, method='multi')
            logger.info(f"Saved {len(data)} rows to PG table {table_name} for {symbol}")
            return True
            
        else:
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
    db_path: str = DEFAULT_DB_PATH,
    use_postgres: bool = False
) -> Optional[pd.DataFrame]:
    """
    Load data from database (SQLite or PostgreSQL)

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
        if use_postgres:
            engine = get_db_engine()
            query = f"SELECT * FROM {table_name}"
            conditions = []
            
            # Determine date column name based on table (simple heuristic)
            date_col = 'timestamp' if 'intraday' in table_name else 'date'
            
            if symbol:
                conditions.append(f"symbol = '{symbol}'")
            if start_date:
                conditions.append(f"{date_col} >= '{start_date}'")
            if end_date:
                conditions.append(f"{date_col} <= '{end_date}'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += f" ORDER BY {date_col}"
            
            df = pd.read_sql_query(query, engine, parse_dates=[date_col])
            
            if not df.empty:
                df = df.set_index(date_col)
                df.index.name = 'date' # Standardize index name for app compatibility
                
            logger.info(f"Loaded {len(df)} rows from PG table {table_name}")
            return df

        else:
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

def init_timescale_db(schema_path: str = "database/schema_timescale.sql"):
    """Initialize TimescaleDB schema"""
    try:
        engine = get_db_engine()
        with open(schema_path, 'r') as f:
            schema = f.read()
            
        with engine.connect() as conn:
            from sqlalchemy import text
            # Split schema by semicolon to execute one by one
            # (sqlalchemy doesn't support executescript like sqlite)
            # This is a naive split, might need more robustness for complex SQL
            statements = schema.split(';')
            for statement in statements:
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()
            
        logger.info("TimescaleDB initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize TimescaleDB: {e}")
        raise
