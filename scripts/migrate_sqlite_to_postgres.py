
import os
import sys
import sqlite3
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_storage import get_db_engine, init_timescale_db, DEFAULT_DB_PATH

def migrate_data():
    """Migrate data from SQLite to PostgreSQL"""
    print("Starting migration...")
    
    # 1. Initialize Postgres Schema
    try:
        print("Initializing TimescaleDB schema...")
        init_timescale_db()
    except Exception as e:
        print(f"Error initializing schema: {e}")
        return

    # 2. Connect to SQLite
    sqlite_path = Path(DEFAULT_DB_PATH)
    if not sqlite_path.exists():
        print(f"No SQLite database found at {sqlite_path}")
        return
        
    conn_lite = sqlite3.connect(sqlite_path)
    pg_engine = get_db_engine()
    
    # 3. Get list of tables
    cursor = conn_lite.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    # Skip internal tables
    skip_tables = ['sqlite_sequence', 'schema_migrations']
    
    for table in tables:
        if table in skip_tables:
            continue
            
        print(f"Migrating table: {table}...")
        
        try:
            # Read in chunks to avoid memory issues
            chunk_size = 10000
            offset = 0
            total_rows = 0
            
            while True:
                query = f"SELECT * FROM {table} LIMIT {chunk_size} OFFSET {offset}"
                df = pd.read_sql_query(query, conn_lite)
                
                if df.empty:
                    break
                    
                # Transform data if necessary
                # Handle boolean columns (sqlite uses 0/1, pg uses true/false)
                if 'is_active' in df.columns:
                    df['is_active'] = df['is_active'].astype(bool)
                
                # Handle date/timestamp column naming differences
                # Schema mismatch handling:
                # SQLite 'daily_prices' -> 'date'
                # PG 'daily_prices' -> 'date' (Match)
                # SQLite 'intraday_prices' -> 'timestamp'
                # PG 'intraday_prices' -> 'timestamp' (Match)
                
                # Write to Postgres
                df.to_sql(table, pg_engine, if_exists='append', index=False, method='multi')
                
                rows_migrated = len(df)
                total_rows += rows_migrated
                offset += chunk_size
                print(f"  Migrated {rows_migrated} rows (Total: {total_rows})")
                
            print(f"Completed {table}: {total_rows} rows.")
            
        except Exception as e:
            print(f"Failed to migrate table {table}: {e}")
            
    conn_lite.close()
    print("Migration complete!")

if __name__ == "__main__":
    migrate_data()
