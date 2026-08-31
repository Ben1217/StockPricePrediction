"""
TimescaleDB persistence for daily OHLCV bars.

The generic helpers in `data_storage` treat PostgreSQL as "SQLite with a different
connection string" — `DataFrame.to_sql(if_exists='append')`, which violates
`daily_prices_pk` the moment the same bar is fetched twice, and which ignores the
`stocks` foreign key that `daily_prices` declares. This module talks to the
hypertables in `database/schema_timescale.sql` on their own terms: upserts keyed
on (symbol, date), and the parent `stocks` row the foreign key requires.

Off by default. `DB_TYPE=timescale` in .env turns it on; with anything else
`timescale_enabled()` is False and the API keeps its provider-only behaviour, so
a developer with no database running is unaffected.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from .data_storage import get_db_engine, get_postgres_connection_string
from ..utils.logger import get_logger

logger = get_logger(__name__)

DAILY_TABLE = "daily_prices"

# DB_TYPE is free text in .env; accept the spellings someone would plausibly write.
_ENABLED_VALUES = frozenset({"timescale", "timescaledb", "postgres", "postgresql"})

# ON CONFLICT needs the column list rather than `ON CONSTRAINT daily_prices_pk`:
# on a hypertable the constraint is re-created per chunk, so only the inference
# form resolves reliably. (symbol, date) is the primary key and includes `date`,
# the partitioning column — which is what makes the upsert legal here at all.
_UPSERT_DAILY = """
INSERT INTO daily_prices
    (symbol, date, open, high, low, close, adjusted_close, volume)
VALUES
    (:symbol, :date, :open, :high, :low, :close, :adjusted_close, :volume)
ON CONFLICT (symbol, date) DO UPDATE SET
    open           = EXCLUDED.open,
    high           = EXCLUDED.high,
    low            = EXCLUDED.low,
    close          = EXCLUDED.close,
    adjusted_close = EXCLUDED.adjusted_close,
    volume         = EXCLUDED.volume
"""

_ENSURE_SYMBOL = """
INSERT INTO stocks (symbol, last_updated) VALUES (:symbol, NOW())
ON CONFLICT (symbol) DO NOTHING
"""


def timescale_enabled() -> bool:
    """
    True when DB_TYPE selects the PostgreSQL/TimescaleDB backend.

    DB_TYPE has sat in .env since the schema was written but was read by nothing.
    This is the one place that gives it an effect.
    """
    return os.getenv("DB_TYPE", "sqlite").strip().lower() in _ENABLED_VALUES


def safe_connection_string() -> str:
    """The configured DSN with the password masked, for logs and /health/db."""
    credentials, separator, rest = get_postgres_connection_string().partition("://")[2].partition("@")
    if not separator:
        return "postgresql://<unparsed>"
    return f"postgresql://{credentials.split(':', 1)[0]}:***@{rest}"


def _utc(value) -> Optional[pd.Timestamp]:
    """Normalise a date-ish value to a UTC timestamp; the column is TIMESTAMPTZ."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _num(value) -> Optional[float]:
    """float(), but NaN and inf become NULL rather than a value the schema rejects."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number) or number in (float("inf"), float("-inf")):
        return None
    return number


def ping() -> dict:
    """
    Open a real connection and report what answered.

    Deliberately more than `SELECT 1`: the point is to prove the TimescaleDB
    extension is loaded and the hypertables exist, not merely that something is
    listening on 5432.
    """
    from sqlalchemy import text

    engine = get_db_engine()
    with engine.connect() as conn:
        server_version = conn.execute(text("SHOW server_version")).scalar()
        extension_version = conn.execute(
            text("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
        ).scalar()
        hypertables = [
            {"name": name, "dimensions": dimensions, "chunks": chunks}
            for name, dimensions, chunks in conn.execute(
                text(
                    "SELECT hypertable_name, num_dimensions, num_chunks "
                    "FROM timescaledb_information.hypertables "
                    "ORDER BY hypertable_name"
                )
            )
        ]
        daily_rows = conn.execute(text(f"SELECT COUNT(*) FROM {DAILY_TABLE}")).scalar()

    return {
        "connected": True,
        "connection": safe_connection_string(),
        "server_version": server_version,
        "timescaledb_version": extension_version,
        "hypertables": hypertables,
        "daily_prices_rows": daily_rows,
    }


def save_daily_prices(symbol: str, df: pd.DataFrame) -> int:
    """
    Upsert daily bars into the `daily_prices` hypertable. Returns rows written.

    Accepts a provider frame (DatetimeIndex, yfinance-cased columns) and maps it
    onto the schema's snake_case columns. Re-running over overlapping data is safe
    and is the normal case — the most recent provider fetch wins.
    """
    from sqlalchemy import text

    if df is None or df.empty:
        return 0

    frame = df[~df.index.duplicated(keep="last")].sort_index()
    ticker = symbol.upper()

    rows = []
    for index_value, row in frame.iterrows():
        stamp = _utc(index_value)
        close = _num(row.get("Close"))
        # `date` is half the primary key and `close` is the column every downstream
        # consumer reads, so a row missing either one is not a usable bar.
        if stamp is None or close is None:
            continue
        volume = _num(row.get("Volume"))
        rows.append({
            "symbol": ticker,
            "date": stamp.to_pydatetime(),
            "open": _num(row.get("Open")),
            "high": _num(row.get("High")),
            "low": _num(row.get("Low")),
            "close": close,
            "adjusted_close": _num(row.get("Adj Close")) or close,
            "volume": None if volume is None else int(volume),
        })

    if not rows:
        return 0

    engine = get_db_engine()
    with engine.begin() as conn:
        # The foreign key means the parent row has to exist first, in the same
        # transaction — otherwise a fresh database rejects every bar.
        conn.execute(text(_ENSURE_SYMBOL), {"symbol": ticker})
        conn.execute(text(_UPSERT_DAILY), rows)

    logger.info("Upserted %d daily bars for %s into TimescaleDB", len(rows), ticker)
    return len(rows)


def load_daily_prices(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read daily bars back in provider shape: a DatetimeIndex plus Open/High/Low/
    Close/Adj Close/Volume, so this can stand in for a provider fetch.

    Bounds are bound parameters rather than interpolated text, and `end` is widened
    to the end of that day so the caller's inclusive range stays inclusive.
    """
    from sqlalchemy import text

    sql = (
        "SELECT date, open, high, low, close, adjusted_close, volume "
        f"FROM {DAILY_TABLE} WHERE symbol = :symbol"
    )
    params: dict = {"symbol": symbol.upper()}

    start_stamp = _utc(start)
    if start_stamp is not None:
        sql += " AND date >= :start"
        params["start"] = start_stamp.to_pydatetime()

    end_stamp = _utc(end)
    if end_stamp is not None:
        sql += " AND date < :end"
        params["end"] = (end_stamp + pd.Timedelta(days=1)).to_pydatetime()

    sql += " ORDER BY date"

    engine = get_db_engine()
    with engine.connect() as conn:
        frame = pd.read_sql_query(text(sql), conn, params=params, parse_dates=["date"])

    if frame.empty:
        return pd.DataFrame()

    frame = frame.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "adjusted_close": "Adj Close",
        "volume": "Volume",
    })
    frame = frame.set_index("date")
    # Daily bars are stored at UTC midnight; handing back a naive index keeps the
    # response formatter emitting "YYYY-MM-DD" exactly as the provider path does.
    if getattr(frame.index, "tz", None) is not None:
        frame.index = frame.index.tz_convert("UTC").tz_localize(None)
    frame.index.name = "Date"
    # NUMERIC comes back as Decimal, which the arithmetic downstream cannot mix
    # with floats; cast once here rather than at every call site.
    return frame.apply(pd.to_numeric, errors="coerce")
