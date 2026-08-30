"""
CORS headers must survive a server error.

The regression this guards: Starlette builds the stack as ``ServerErrorMiddleware
-> user middleware -> ExceptionMiddleware -> router``, so a 500 produced by an
``@app.exception_handler(Exception)`` is created *above* CORSMiddleware and reaches
the browser with no Access-Control-Allow-Origin. Chrome then reports it as

    Access to fetch at '.../api/portfolio/optimize' from origin
    'http://localhost:5173' has been blocked by CORS policy: No
    'Access-Control-Allow-Origin' header is present on the requested resource.

which sends the investigation to the CORS config while the actual cause — the
traceback behind the 500 — stays invisible. The ordering in main.py is the fix, and
ordering is exactly the kind of property a later edit breaks without noticing.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import CORS_ORIGINS, app
from src.api.security import cors_headers_for, limiter

ORIGIN = "http://localhost:5173"


@pytest.fixture
def client():
    limiter.reset()
    # raise_server_exceptions=False makes TestClient return the 500 the browser
    # would see instead of re-raising it into the test.
    yield TestClient(app, raise_server_exceptions=False)
    limiter.reset()


@pytest.fixture
def boom(client):
    """Register a route that raises, so the failure path is exercised for real."""
    path = "/api/portfolio/__cors_regression_boom"

    @app.get(path)
    def _boom():
        raise RuntimeError("simulated failure inside a route")

    yield path
    app.router.routes = [r for r in app.router.routes if getattr(r, "path", None) != path]


def test_unhandled_exception_still_carries_cors_headers(client, boom):
    response = client.get(boom, headers={"Origin": ORIGIN})

    assert response.status_code == 500
    assert response.headers.get("access-control-allow-origin") == ORIGIN, (
        "A 500 without this header reaches the browser as a CORS error, hiding the "
        "real exception. Check that error_middleware is registered below CORSMiddleware."
    )


def test_successful_response_carries_cors_headers(client):
    response = client.get("/health", headers={"Origin": ORIGIN})

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_preflight_allows_the_dev_origin(client):
    response = client.options(
        "/api/portfolio/optimize",
        headers={
            "Origin": ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == ORIGIN
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_unknown_origin_is_not_echoed_on_an_error(client, boom):
    """The error path must not become a wildcard that the success path never was."""
    response = client.get(boom, headers={"Origin": "https://evil.example.com"})

    assert response.status_code == 500
    assert "access-control-allow-origin" not in response.headers


def test_error_middleware_is_registered_below_cors():
    """
    Two independent mechanisms keep the header on a 500: `error_middleware` handles
    the exception below CORSMiddleware, and the last-resort handler attaches the
    header by hand. Either alone satisfies the tests above, which means a reorder
    could silently disable one and go unnoticed until the other is touched too.
    This pins the ordering directly.

    Starlette inserts each registration at the front of `user_middleware`, so a
    LOWER index is an OUTER layer at runtime.
    """
    from starlette.middleware.cors import CORSMiddleware

    names = [m.cls.__name__ for m in app.user_middleware]
    dispatches = [
        getattr(m.kwargs.get("dispatch", None), "__name__", None) for m in app.user_middleware
    ]

    cors_index = names.index(CORSMiddleware.__name__)
    error_index = dispatches.index("error_middleware")
    security_index = dispatches.index("security_middleware")

    assert cors_index < error_index, (
        "CORSMiddleware must be outside error_middleware, or the 500 it builds "
        "never passes through the layer that adds Access-Control-Allow-Origin."
    )
    assert error_index < security_index, (
        "error_middleware must be outside security_middleware so an exception in "
        "the auth/rate-limit path is caught below CORS too."
    )


def test_cors_headers_for_matches_the_allow_list():
    """Unit-level cover for the last-resort handler's manual header construction."""
    class _Request:
        def __init__(self, origin=None):
            self.headers = {"origin": origin} if origin else {}

    assert cors_headers_for(_Request(), CORS_ORIGINS) == {}
    assert cors_headers_for(_Request("https://nope.example"), CORS_ORIGINS) == {}

    allowed = cors_headers_for(_Request(ORIGIN), CORS_ORIGINS)
    assert allowed["Access-Control-Allow-Origin"] == ORIGIN
    assert allowed["Vary"] == "Origin"

    wildcard = cors_headers_for(_Request("https://anything.example"), ["*"])
    assert wildcard["Access-Control-Allow-Origin"] == "*"
