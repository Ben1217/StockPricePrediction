"""
Data module - Data acquisition, loading, preprocessing, and storage
"""

from .data_loader import download_stock_data, load_multiple_stocks
from .data_acquisition import get_sp500_tickers, download_index_data
from .data_preprocessing import clean_data, handle_missing_values
from .data_storage import save_to_database, load_from_database
from .data_validator import validate_data_quality

__all__ = [
    "download_stock_data",
    "load_multiple_stocks",
    "get_sp500_tickers",
    "download_index_data",
    "clean_data",
    "handle_missing_values",
    "save_to_database",
    "load_from_database",
    "validate_data_quality",
]
