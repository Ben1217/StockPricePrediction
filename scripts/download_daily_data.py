"""
Download daily price data for all stocks
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_acquisition import download_index_data, get_sp500_tickers
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config, get_env

logger = setup_logger(__name__)


def main():
    """Main function to download daily data"""
    config = load_config()

    start_date = get_env('DATA_START_DATE', '2019-01-01')
    end_date = get_env('DATA_END_DATE', '2024-12-31')

    logger.info("=" * 60)
    logger.info("DOWNLOADING DAILY STOCK DATA")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")

    # Download S&P 500 data
    if get_env('ENABLE_SP500', 'True').lower() == 'true':
        logger.info("Downloading S&P 500 stocks...")
        data_dict = download_index_data('sp500', start_date, end_date)
        logger.info(f"Downloaded {len(data_dict)} S&P 500 stocks")

    # Download Russell 2000 data (optional)
    if get_env('ENABLE_RUSSELL2000', 'False').lower() == 'true':
        logger.info("Downloading Russell 2000 stocks...")
        data_dict = download_index_data('russell2000', start_date, end_date)
        logger.info(f"Downloaded {len(data_dict)} Russell 2000 stocks")

    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
