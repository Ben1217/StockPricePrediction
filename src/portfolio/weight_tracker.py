"""
Weight Tracker — SQLite persistence for portfolio weight snapshots.

Stores target weights after each optimization run and computes
drift between saved targets and current market values.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path("data/portfolio.db")


def _get_conn() -> sqlite3.Connection:
    """Get SQLite connection, creating the table if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS weight_snapshots (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            portfolio_id TEXT    NOT NULL,
            strategy     TEXT    NOT NULL,
            weights_json TEXT    NOT NULL,
            saved_at     TEXT    NOT NULL
        )
    """)
    conn.commit()
    return conn


def save_weights(portfolio_id: str, strategy: str, weights: dict) -> None:
    """
    Persist target weights after a successful optimization.

    Parameters
    ----------
    portfolio_id : str
        Identifier for the portfolio (e.g. user ID or "default")
    strategy : str
        Optimization strategy used (e.g. "max_sharpe")
    weights : dict
        {ticker: weight_float}
    """
    conn = _get_conn()
    conn.execute(
        "INSERT INTO weight_snapshots (portfolio_id, strategy, weights_json, saved_at)"
        " VALUES (?, ?, ?, ?)",
        (portfolio_id, strategy, json.dumps(weights), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved weight snapshot for portfolio={portfolio_id}, strategy={strategy}")


def get_last_weights(portfolio_id: str) -> dict | None:
    """
    Retrieve the most recently saved weights for a portfolio.

    Returns
    -------
    dict or None
        {weights: {...}, saved_at: str, strategy: str} or None if no snapshot exists
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT weights_json, saved_at, strategy FROM weight_snapshots"
        " WHERE portfolio_id=? ORDER BY id DESC LIMIT 1",
        (portfolio_id,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    return {"weights": json.loads(row[0]), "saved_at": row[1], "strategy": row[2]}


def calculate_drift(
    target_weights: dict,
    current_values: dict,
    total_value: float,
    drift_threshold: float = 0.05,
) -> dict:
    """
    Compute weight drift between target weights and current holdings.

    Parameters
    ----------
    target_weights : dict
        {ticker: target_weight} from the last saved snapshot
    current_values : dict
        {ticker: current_dollar_value} current market value per holding
    total_value : float
        Total portfolio market value
    drift_threshold : float
        Flag tickers with drift beyond this threshold

    Returns
    -------
    dict
        Drift per ticker with needs_rebalance flag
    """
    drift = {}
    for ticker, target_w in target_weights.items():
        current_w = current_values.get(ticker, 0) / total_value if total_value > 0 else 0
        d = current_w - target_w
        drift[ticker] = {
            "target_weight": round(target_w, 4),
            "current_weight": round(current_w, 4),
            "drift": round(d, 4),
            "needs_rebalance": abs(d) > drift_threshold,
        }
    return drift
