"""
Status-code mapping for Alpha Vantage failures on the data routes.

A free key cannot serve extended-hours quotes at all, and can be throttled at any
time. Both used to surface as 502 Bad Gateway, which misdescribes them and marks
them retryable to clients that treat 5xx as transient.

No network: the provider layer is patched, so these assert the route contract.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.provider_errors import (
    PremiumEndpointError,
    ProviderError,
    QuotaExceededError,
)

client = TestClient(app)


def test_extended_quote_via_alpha_vantage_is_501_not_502():
    """The reported bug: GET /extended-quote?source=alpha_vantage returned 502."""
    resp = client.get("/api/data/extended-quote/META", params={"source": "alpha_vantage"})
    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert "TIME_SERIES_INTRADAY" in detail
    assert "premium" in detail.lower()


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (PremiumEndpointError("premium only", endpoint="X"), 501),
        (QuotaExceededError("out of quota"), 429),
        (ProviderError("unrecognised upstream reply"), 502),
        (RuntimeError("connection reset"), 502),
    ],
)
def test_live_quote_maps_provider_errors_to_honest_statuses(
    monkeypatch, error, expected_status
):
    def boom(symbol, source="yfinance"):
        raise error

    monkeypatch.setattr("src.api.routes.data.fetch_live_quote", boom)
    resp = client.get("/api/data/quote/AAPL", params={"source": "alpha_vantage"})
    assert resp.status_code == expected_status


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (PremiumEndpointError("premium only", endpoint="X"), 501),
        (QuotaExceededError("throttled"), 429),
        (ProviderError("unrecognised upstream reply"), 502),
    ],
)
def test_prices_maps_provider_errors_instead_of_500(monkeypatch, error, expected_status):
    """A throttled /prices call used to land on 404 'no data' or a blanket 500."""

    def boom(symbol, start, end):
        raise error

    monkeypatch.setattr("src.api.routes.data._fetch_alpha_vantage_history", boom)
    resp = client.get("/api/data/prices/AAPL", params={"source": "alpha_vantage", "days": 30})
    assert resp.status_code == expected_status
