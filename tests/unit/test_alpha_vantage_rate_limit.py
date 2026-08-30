"""
The shared Alpha Vantage limiter.

The free tier enforces two independent limits: 25 requests/day (resets at UTC
midnight) and roughly 1 request/second. Both are answered with an HTTP 200
"Information" block instead of data, so both are handled before the wire.
"""

from datetime import date, datetime, timezone

import pytest

import src.data.alpha_vantage_provider as av
from src.data.provider_errors import QuotaExceededError


@pytest.fixture
def limiter(monkeypatch):
    """Reset limiter state and record sleeps instead of performing them."""
    monkeypatch.setattr(av, "_calls_today", 0)
    monkeypatch.setattr(av, "_quota_date", None)
    monkeypatch.setattr(av, "_last_call_time", 0.0)

    slept = []
    monkeypatch.setattr(av.time, "sleep", lambda s: slept.append(s))
    return slept


def test_first_call_does_not_sleep(limiter, monkeypatch):
    monkeypatch.setattr(av.time, "monotonic", lambda: 1000.0)
    av.consume_request_slot()
    assert limiter == []
    assert av._calls_today == 1


def test_back_to_back_calls_are_spaced_to_the_burst_limit(limiter, monkeypatch):
    """Two immediate calls must be separated by at least _MIN_INTERVAL."""
    monkeypatch.setattr(av.time, "monotonic", lambda: 1000.0)
    av.consume_request_slot()
    av.consume_request_slot()

    assert len(limiter) == 1
    assert limiter[0] == pytest.approx(av._MIN_INTERVAL)


def test_no_sleep_when_enough_time_already_passed(limiter, monkeypatch):
    clock = iter([1000.0, 1000.0, 1000.0 + av._MIN_INTERVAL + 5, 1010.0])
    monkeypatch.setattr(av.time, "monotonic", lambda: next(clock))
    av.consume_request_slot()
    av.consume_request_slot()
    assert limiter == []


def test_daily_quota_raises_at_the_limit(limiter, monkeypatch):
    monkeypatch.setattr(av.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(av, "_calls_today", av._DAILY_LIMIT)
    monkeypatch.setattr(av, "_quota_date", datetime.now(timezone.utc).date())

    with pytest.raises(QuotaExceededError) as excinfo:
        av.consume_request_slot()
    assert str(av._DAILY_LIMIT) in str(excinfo.value)


def test_quota_counter_resets_on_a_new_utc_day(limiter, monkeypatch):
    monkeypatch.setattr(av.time, "monotonic", lambda: 1000.0)
    monkeypatch.setattr(av, "_calls_today", av._DAILY_LIMIT)
    monkeypatch.setattr(av, "_quota_date", date(2000, 1, 1))  # a stale day

    av.consume_request_slot()  # must not raise: the day rolled over
    assert av._calls_today == 1
    assert av._quota_date == datetime.now(timezone.utc).date()


def test_every_alpha_vantage_call_site_uses_the_shared_limiter():
    """
    The counter is only meaningful if all three raw call sites go through it.
    Import-level check so a new bypassing call site is caught here.
    """
    import src.api.routes.data as data_route
    import src.data.live_data as live_data

    assert data_route.consume_request_slot is av.consume_request_slot
    assert live_data.consume_request_slot is av.consume_request_slot
