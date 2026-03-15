"""
Sector Allocation — yFinance sector lookup with file caching.

Maps stock tickers to GICS sectors and computes sector-level
portfolio weights with concentration warnings.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

from ..utils.logger import get_logger

logger = get_logger(__name__)

CACHE_FILE = Path("data/sector_cache.json")
CACHE_TTL_DAYS = 30  # sector classifications rarely change


def _load_cache() -> dict:
    """Load sector cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Persist sector cache to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def get_ticker_sector(ticker: str) -> str:
    """
    Return the sector for a ticker, using a 30-day file cache.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol

    Returns
    -------
    str
        Sector name (e.g. 'Technology', 'Healthcare') or 'Unknown'
    """
    cache = _load_cache()
    entry = cache.get(ticker)

    if entry:
        cached_at = datetime.fromisoformat(entry["cached_at"])
        if datetime.utcnow() - cached_at < timedelta(days=CACHE_TTL_DAYS):
            return entry["sector"]

    # Cache miss — fetch from yFinance
    try:
        info = yf.Ticker(ticker).info
        sector = info.get("sector", "Unknown")
    except Exception as e:
        logger.warning(f"Failed to fetch sector for {ticker}: {e}")
        sector = "Unknown"

    cache[ticker] = {"sector": sector, "cached_at": datetime.utcnow().isoformat()}
    _save_cache(cache)
    return sector


def get_sector_allocation(weights: dict) -> dict:
    """
    Compute sector-level weight totals and concentration warnings.

    Parameters
    ----------
    weights : dict
        {ticker: weight_float}

    Returns
    -------
    dict
        sector_weights, sector_tickers, concentration_warnings,
        most_concentrated_sector
    """
    sector_weights: dict[str, float] = {}
    sector_tickers: dict[str, list] = {}

    for ticker, w in weights.items():
        sector = get_ticker_sector(ticker)
        sector_weights[sector] = sector_weights.get(sector, 0.0) + w
        sector_tickers.setdefault(sector, []).append(ticker)

    warnings = [
        {
            "sector": s,
            "weight": round(w, 4),
            "message": f"Sector concentration > 35% ({w:.1%})",
        }
        for s, w in sector_weights.items()
        if w > 0.35
    ]

    return {
        "sector_weights": {s: round(w, 4) for s, w in sector_weights.items()},
        "sector_tickers": sector_tickers,
        "concentration_warnings": warnings,
        "most_concentrated_sector": max(sector_weights, key=sector_weights.get)
        if sector_weights
        else None,
    }
