"""
Per-interval request windows, in one place.

These numbers existed in three copies that had drifted apart: a `prices` and
`indicators` table in `routes/data.py`, a support/resistance `lookback` table in
`routes/patterns.py`, and a fourth table in the frontend's `utils/api.js` whose
comment claimed to mirror "the server's own clamps". Three of the four buckets
did have a server clamp. The fourth, `sentimentDays`, never did - the sentiment
route accepts `days` from 30 to 7000 on every interval, so the client table was
the only thing enforcing it and nothing said so.

Two different units live here, which is why each bucket names its own
------------------------------------------------------------------
`prices`, `indicators` and `sentiment` are **calendar days**. `lookback` is
**bars of the requested interval** - 240 on `1mo` means twenty years of monthly
candles, not eight months of calendar. Mixing the two is what left the monthly
support/resistance panel empty for every symbol: a bar count spent as days
fetched fifteen candles for an algorithm that reads a hundred.

Served, not just shared
-----------------------
`GET /api/data/limits` returns this table so the frontend clamps to the server's
real numbers instead of a hardcoded copy that goes stale silently. The client
keeps a bundled fallback for when the API is unreachable, but the served table
wins whenever it is available.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

#: The intervals every windowed endpoint accepts.
INTERVALS: Tuple[str, ...] = ("1m", "5m", "15m", "1h", "4h", "1d", "1wk", "1mo")

#: The interval used when a request names one this table does not cover.
DEFAULT_INTERVAL = "1d"

#: What each bucket's numbers measure. Served alongside the table so a client
#: cannot repeat the bars-as-days mistake by inspection alone.
BUCKET_UNITS: Mapping[str, str] = {
    "prices": "calendar_days",
    "indicators": "calendar_days",
    "sentiment": "calendar_days",
    "lookback": "bars",
}

#: (minimum, maximum) per interval per bucket.
#:
#: The floors are not politeness - they are what the downstream computation needs
#: to produce anything. `indicators` at 120 days covers a 100-bar warm-up; the
#: `lookback` floors sit above the detector's own 100-bar analysis slice so a
#: clamped request still fills the window it reads.
INTERVAL_LIMITS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "1m":  {"prices": (7, 7),        "indicators": (60, 120),  "sentiment": (120, 120),   "lookback": (60, 90)},
    "5m":  {"prices": (30, 60),      "indicators": (60, 120),  "sentiment": (120, 180),   "lookback": (60, 120)},
    "15m": {"prices": (30, 60),      "indicators": (60, 120),  "sentiment": (120, 180),   "lookback": (60, 120)},
    "1h":  {"prices": (180, 730),    "indicators": (120, 240), "sentiment": (240, 730),   "lookback": (120, 365)},
    "4h":  {"prices": (180, 730),    "indicators": (120, 240), "sentiment": (240, 730),   "lookback": (120, 365)},
    "1d":  {"prices": (30, 420),     "indicators": (120, 320), "sentiment": (240, 420),   "lookback": (120, 420)},
    "1wk": {"prices": (730, 3650),   "indicators": (120, 300), "sentiment": (800, 2600),  "lookback": (120, 300)},
    "1mo": {"prices": (1825, 3650),  "indicators": (120, 180), "sentiment": (1200, 3650), "lookback": (120, 240)},
}


def limits_for(interval: str, bucket: str) -> Tuple[int, int]:
    """
    The ``(minimum, maximum)`` for one interval and bucket.

    An unknown interval falls back to the daily row rather than raising: these
    endpoints validate the interval through their own ``Query(enum=...)``, so
    reaching here with something else means a caller inside the process passed a
    value, and a sane window beats a 500.
    """
    row = INTERVAL_LIMITS.get(interval, INTERVAL_LIMITS[DEFAULT_INTERVAL])
    if bucket not in row:
        raise KeyError(f"Unknown request-window bucket {bucket!r}; expected one of {sorted(BUCKET_UNITS)}")
    return row[bucket]


def clamp(interval: str, value: int, bucket: str) -> int:
    """``value`` brought inside the window for ``interval``/``bucket``."""
    lower, upper = limits_for(interval, bucket)
    return min(max(int(value), lower), upper)


def as_payload() -> Dict[str, object]:
    """
    The table in the shape ``GET /api/data/limits`` serves.

    Tuples become two-element lists so the JSON reads as a range rather than as
    an object with min/max keys the client would have to unpack differently from
    its own bundled fallback.
    """
    return {
        "intervals": list(INTERVALS),
        "default_interval": DEFAULT_INTERVAL,
        "units": dict(BUCKET_UNITS),
        "limits": {
            interval: {bucket: [lower, upper] for bucket, (lower, upper) in buckets.items()}
            for interval, buckets in INTERVAL_LIMITS.items()
        },
    }
