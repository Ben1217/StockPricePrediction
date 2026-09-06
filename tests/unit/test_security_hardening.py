"""
Tests for API-key comparison and rate-limit client identity.
"""

import ipaddress

import pytest
from fastapi.testclient import TestClient

from src.api import security


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, "s3cret-key-value")
    security.limiter.reset()
    from src.api.main import app

    yield TestClient(app)
    security.limiter.reset()


def test_a_wrong_key_is_refused(client):
    assert client.get("/api/data/sources", headers={"X-API-Key": "wrong"}).status_code == 401


def test_a_prefix_of_the_key_is_refused(client):
    """
    compare_digest, not ==. Equality short-circuits on the first differing byte,
    so the time a rejection takes leaks how much of the key a guess got right.
    """
    assert client.get("/api/data/sources", headers={"X-API-Key": "s3cret"}).status_code == 401


def test_the_correct_key_is_accepted(client):
    assert client.get("/api/data/sources", headers={"X-API-Key": "s3cret-key-value"}).status_code == 200


def test_a_missing_key_is_refused(client):
    assert client.get("/api/data/sources").status_code == 401


# ---------------------------------------------------------------------------
# Rate-limit identity
# ---------------------------------------------------------------------------

class _Request:
    """Minimal stand-in for the two attributes _client_id reads."""

    def __init__(self, headers, peer="10.0.0.1"):
        self.headers = headers
        self.client = type("C", (), {"host": peer})()


def test_forwarded_for_is_ignored_by_default(monkeypatch):
    """
    The header is client-supplied. Honouring it unconditionally means a caller can
    mint a fresh rate-limit budget per request by varying the value, which is the
    same as having no limiter at all.
    """
    monkeypatch.delenv(security.TRUST_FORWARDED_FOR_ENV, raising=False)
    request = _Request({"x-forwarded-for": "1.2.3.4"}, peer="10.0.0.1")
    assert security._client_id(request) == "ip:10.0.0.1"


def test_forwarded_for_is_used_when_a_proxy_is_trusted(monkeypatch):
    monkeypatch.setenv(security.TRUST_FORWARDED_FOR_ENV, "true")
    request = _Request({"x-forwarded-for": "1.2.3.4, 10.0.0.9"}, peer="10.0.0.1")
    assert security._client_id(request) == "ip:1.2.3.4"


def test_a_malformed_forwarded_for_falls_back_to_the_peer(monkeypatch):
    """An unparseable value would otherwise become an arbitrary bucket key."""
    monkeypatch.setenv(security.TRUST_FORWARDED_FOR_ENV, "true")
    request = _Request({"x-forwarded-for": "not-an-ip"}, peer="10.0.0.1")
    assert security._client_id(request) == "ip:10.0.0.1"


def test_an_api_key_identifies_the_client_ahead_of_any_address(monkeypatch):
    monkeypatch.setenv(security.TRUST_FORWARDED_FOR_ENV, "true")
    request = _Request({"X-API-Key": "abcdefghijklmnop", "x-forwarded-for": "1.2.3.4"})
    assert security._client_id(request) == "key:abcdefghijkl"


@pytest.mark.parametrize("value", ["1.2.3.4", "::1", "2001:db8::1"])
def test_valid_addresses_survive_the_round_trip(monkeypatch, value):
    monkeypatch.setenv(security.TRUST_FORWARDED_FOR_ENV, "true")
    request = _Request({"x-forwarded-for": value})
    assert security._client_id(request) == f"ip:{ipaddress.ip_address(value)}"
