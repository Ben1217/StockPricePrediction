"""
Tests for the consolidated per-interval request windows.

These numbers lived in three copies - two server-side tables and one in the
frontend - and the frontend's comment claimed to mirror the server's. It did, for
three of the four buckets; `sentiment` had no server counterpart at all. The
table now lives in one module and is served, so the client clamps to the server's
real numbers instead of a copy nothing reconciles.

The last test here reads the frontend's bundled fallback and compares it against
the served table. It is the only thing standing between the two copies and a
silent drift, so it deliberately fails on any difference rather than warning.
"""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.limits import (
    BUCKET_UNITS,
    DEFAULT_INTERVAL,
    INTERVAL_LIMITS,
    INTERVALS,
    as_payload,
    clamp,
    limits_for,
)


@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def test_every_interval_defines_every_bucket():
    for interval in INTERVALS:
        assert interval in INTERVAL_LIMITS, interval
        for bucket in BUCKET_UNITS:
            assert bucket in INTERVAL_LIMITS[interval], f"{interval}.{bucket}"


def test_every_window_is_ordered_and_positive():
    for interval, buckets in INTERVAL_LIMITS.items():
        for bucket, (lower, upper) in buckets.items():
            assert lower > 0, f"{interval}.{bucket}"
            assert lower <= upper, f"{interval}.{bucket}"


def test_an_unknown_interval_falls_back_to_the_daily_row():
    """
    These endpoints validate the interval through Query(enum=...), so reaching
    the table with something else means an internal caller passed it. A sane
    window beats a 500.
    """
    assert limits_for("nonsense", "prices") == limits_for(DEFAULT_INTERVAL, "prices")


def test_an_unknown_bucket_is_a_programming_error():
    # Unlike an interval, a bucket name is never client-supplied, so a typo here
    # should surface rather than silently clamp to something plausible.
    with pytest.raises(KeyError):
        limits_for("1d", "nope")


@pytest.mark.parametrize(
    "interval, bucket, value, expected",
    [
        ("1d", "prices", 1, 30),          # below the floor
        ("1d", "prices", 120, 120),       # inside the window
        ("1d", "prices", 99999, 420),     # above the ceiling
        ("1mo", "lookback", 10, 120),
        ("1mo", "lookback", 9999, 240),
        ("1m", "prices", 500, 7),         # a degenerate window still clamps
    ],
)
def test_clamping(interval, bucket, value, expected):
    assert clamp(interval, value, bucket) == expected


def test_support_resistance_floors_fill_the_detectors_window():
    """
    The detector reads the last 100 bars whatever it is sent, so a floor below
    that would hand it a thinner read than it was designed for - which is what a
    clamped request lands on.

    Scoped to daily and longer, which is where `lookback` actually governs the
    download. The intraday rows floor at 60, deliberately: those intervals fetch
    by a fixed `period` ("7d" on 1m, "730d" on 1h) rather than by this number, so
    the frame arrives with far more than 100 bars regardless of what is asked for.
    """
    from src.api.routes.patterns import SR_ANALYSIS_BARS

    for interval in ("1d", "1wk", "1mo"):
        lower, _ = limits_for(interval, "lookback")
        assert lower >= SR_ANALYSIS_BARS, interval


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------

def test_the_limits_endpoint_serves_the_table(client):
    response = client.get("/api/data/limits")
    assert response.status_code == 200

    body = response.json()
    assert body["intervals"] == list(INTERVALS)
    assert body["default_interval"] == DEFAULT_INTERVAL
    assert body["units"] == dict(BUCKET_UNITS)

    for interval, buckets in INTERVAL_LIMITS.items():
        for bucket, (lower, upper) in buckets.items():
            assert body["limits"][interval][bucket] == [lower, upper]


def test_the_payload_declares_which_buckets_are_bars(client):
    """
    Two units share this table. `lookback` counts bars of the interval; the rest
    are calendar days. Spending one as the other is what left the monthly
    support/resistance panel empty, so the payload says which is which.
    """
    units = client.get("/api/data/limits").json()["units"]
    assert units["lookback"] == "bars"
    assert units["prices"] == "calendar_days"
    assert units["indicators"] == "calendar_days"
    assert units["sentiment"] == "calendar_days"


def test_the_payload_is_json_serialisable():
    # Tuples would round-trip as lists anyway; this pins the shape the client
    # parses rather than leaving it to FastAPI's encoder.
    assert json.loads(json.dumps(as_payload())) == as_payload()


# ---------------------------------------------------------------------------
# Client/server agreement
# ---------------------------------------------------------------------------

_FRONTEND_API = Path(__file__).resolve().parents[2] / "quantvision" / "src" / "utils" / "api.js"

#: The frontend's key for each server bucket, mirroring LIMIT_KEY_BY_BUCKET there.
_CLIENT_KEY = {
    "prices": "priceDays",
    "indicators": "indicatorDays",
    "sentiment": "sentimentDays",
    "lookback": "lookback",
}


def _parse_frontend_fallback():
    """Read FALLBACK_INTERVAL_LIMITS out of utils/api.js."""
    source = _FRONTEND_API.read_text(encoding="utf-8")
    block = re.search(
        r"const FALLBACK_INTERVAL_LIMITS = \{(.*?)\n\};", source, re.DOTALL
    )
    assert block, "FALLBACK_INTERVAL_LIMITS not found in utils/api.js"

    table = {}
    for line in block.group(1).splitlines():
        row = re.match(r'\s*"([^"]+)":\s*\{(.*)\},\s*$', line)
        if not row:
            continue
        interval, body = row.group(1), row.group(2)
        table[interval] = {
            key: (int(low), int(high))
            for key, low, high in re.findall(r"(\w+):\s*\[(\d+),\s*(\d+)\]", body)
        }
    return table


@pytest.mark.skipif(not _FRONTEND_API.exists(), reason="frontend sources not present")
def test_the_frontend_fallback_matches_the_served_table():
    """
    The client adopts the served table at startup, so this fallback only applies
    before that lands or when the API is unreachable. It should still agree: a
    fallback that clamps to different numbers than the server is a difference
    nobody would notice until a request came back 422.
    """
    frontend = _parse_frontend_fallback()
    assert set(frontend) == set(INTERVALS)

    mismatches = []
    for interval in INTERVALS:
        for bucket, (lower, upper) in INTERVAL_LIMITS[interval].items():
            client_key = _CLIENT_KEY[bucket]
            got = frontend[interval].get(client_key)
            if got != (lower, upper):
                mismatches.append(
                    f"{interval}.{bucket}: server {(lower, upper)} vs frontend {got}"
                )

    assert not mismatches, "\n".join(
        ["frontend fallback has drifted from src/api/limits.py:", *mismatches]
    )
