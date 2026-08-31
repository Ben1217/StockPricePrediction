"""
Unit tests for the TimescaleDB storage layer.

Covers the parts that decide *whether* and *how* the database is used — the
DB_TYPE switch, DSN masking, value coercion, and the read-through coverage rule.
No server required: the one test that touches SQL asserts on a fake engine, so
the suite still runs with nothing listening on 5432.
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.api.routes.data import _daily_store_covers
from src.data.timescale_store import (
    _num,
    _utc,
    safe_connection_string,
    save_daily_prices,
    timescale_enabled,
)


# ── The DB_TYPE switch ────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["timescale", "timescaledb", "postgres", "postgresql", "TimeScale", "  postgres  "])
def test_timescale_enabled_for_postgres_spellings(monkeypatch, value):
    monkeypatch.setenv("DB_TYPE", value)
    assert timescale_enabled() is True


@pytest.mark.parametrize("value", ["sqlite", "", "mysql", "none"])
def test_timescale_disabled_for_everything_else(monkeypatch, value):
    monkeypatch.setenv("DB_TYPE", value)
    assert timescale_enabled() is False


def test_timescale_defaults_to_disabled(monkeypatch):
    """An unset DB_TYPE must not silently start requiring a database."""
    monkeypatch.delenv("DB_TYPE", raising=False)
    assert timescale_enabled() is False


# ── DSN masking ───────────────────────────────────────────────────────────────

def test_connection_string_masks_the_password(monkeypatch):
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "sup3rs3cret")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_DB", "stock_data")

    masked = safe_connection_string()

    assert "sup3rs3cret" not in masked
    assert masked == "postgresql://postgres:***@localhost:5432/stock_data"


# ── Value coercion ────────────────────────────────────────────────────────────

def test_num_rejects_values_the_schema_cannot_store():
    assert _num(None) is None
    assert _num(float("nan")) is None
    assert _num(float("inf")) is None
    assert _num("not a number") is None
    assert _num("12.5") == 12.5
    assert _num(3) == 3.0


def test_utc_normalises_naive_and_aware_stamps():
    naive = _utc("2026-01-05")
    assert str(naive.tz) == "UTC"
    assert naive.hour == 0

    aware = _utc(pd.Timestamp("2026-01-05 14:30", tz="US/Eastern"))
    assert str(aware.tz) == "UTC"
    assert aware.hour == 19


# ── Read-through coverage rule ────────────────────────────────────────────────

def _frame(dates):
    return pd.DataFrame({"Close": range(len(dates))}, index=pd.DatetimeIndex(dates))


def test_empty_store_never_covers():
    assert _daily_store_covers(pd.DataFrame(), "2026-01-01", "2026-01-31") is False
    assert _daily_store_covers(None, "2026-01-01", "2026-01-31") is False


def test_full_window_covers():
    stored = _frame(pd.date_range("2026-01-01", "2026-01-31", freq="D"))
    assert _daily_store_covers(stored, "2026-01-01", "2026-01-31") is True


def test_weekend_edges_still_cover():
    """The window ends on a Saturday and Sunday; the bars stop on the Friday."""
    stored = _frame(pd.date_range("2026-01-05", "2026-01-30", freq="D"))
    assert _daily_store_covers(stored, "2026-01-03", "2026-01-31") is True


def test_stale_store_does_not_cover():
    """Bars ending three weeks early must fall through to the provider."""
    stored = _frame(pd.date_range("2026-01-01", "2026-01-10", freq="D"))
    assert _daily_store_covers(stored, "2026-01-01", "2026-01-31") is False


def test_store_starting_late_does_not_cover():
    stored = _frame(pd.date_range("2026-01-20", "2026-01-31", freq="D"))
    assert _daily_store_covers(stored, "2026-01-01", "2026-01-31") is False


# ── Writes ────────────────────────────────────────────────────────────────────

def test_save_skips_empty_frames_without_touching_the_database():
    with patch("src.data.timescale_store.get_db_engine") as engine:
        assert save_daily_prices("AAPL", pd.DataFrame()) == 0
        assert save_daily_prices("AAPL", None) == 0
    engine.assert_not_called()


def test_save_upserts_the_parent_row_before_the_bars():
    """The daily_prices foreign key means `stocks` has to be written first."""
    connection = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection

    frame = pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [12.0, 13.0],
            "Low": [9.0, 10.0],
            "Close": [11.5, 12.5],
            "Volume": [1000, 2000],
        },
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06"]),
    )

    with patch("src.data.timescale_store.get_db_engine", return_value=engine):
        written = save_daily_prices("aapl", frame)

    assert written == 2
    statements = [str(call.args[0]) for call in connection.execute.call_args_list]
    assert "INSERT INTO stocks" in statements[0]
    assert "INSERT INTO daily_prices" in statements[1]

    rows = connection.execute.call_args_list[1].args[1]
    assert [row["symbol"] for row in rows] == ["AAPL", "AAPL"]
    # No "Adj Close" column in the frame, so it falls back to the close.
    assert rows[0]["adjusted_close"] == 11.5
    assert rows[0]["volume"] == 1000


def test_save_drops_bars_with_no_close():
    """`date` is half the primary key and a NaN close is not a usable bar."""
    connection = MagicMock()
    engine = MagicMock()
    engine.begin.return_value.__enter__.return_value = connection

    frame = pd.DataFrame(
        {"Open": [10.0, 11.0], "Close": [11.5, float("nan")], "Volume": [1000, 2000]},
        index=pd.DatetimeIndex(["2026-01-05", "2026-01-06"]),
    )

    with patch("src.data.timescale_store.get_db_engine", return_value=engine):
        assert save_daily_prices("AAPL", frame) == 1
