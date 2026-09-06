"""
API key authentication and rate limiting.

Both are **opt-in** so the default local-development experience is unchanged:

* Auth activates only when ``QUANTVISION_API_KEY`` is set. Until then every request
  is allowed and a warning is logged once at startup.
* Rate limiting is always on but with generous defaults, and is stricter on the
  requests that start expensive work (training, backtests, agent calls). The strict
  budgets apply to state-changing methods only, so polling a job's status does not
  consume the budget for starting jobs.

The limiter is per-process and in-memory. That is the right scope for a single
uvicorn worker; running multiple workers needs a shared store (Redis) for both the
limiter and the job/backtest state (see the notes in src/api/jobs.py).

Clients are identified by API key when one is presented and by peer address
otherwise. ``X-Forwarded-For`` is honoured only when
``QUANTVISION_TRUST_PROXY_HEADERS`` is set, because a forgeable header used as a
limiter key is not a limiter — see :data:`TRUST_FORWARDED_FOR_ENV`.
"""

from __future__ import annotations

import ipaddress
import logging
import os
import secrets
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterable, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY_ENV = "QUANTVISION_API_KEY"
API_KEY_HEADER = "X-API-Key"

# Paths that never require a key: liveness, service metadata, and the docs.
# /health/db reports reachability and a password-masked DSN, nothing queryable.
PUBLIC_PATHS = frozenset({"/", "/health", "/health/db", "/docs", "/redoc", "/openapi.json"})

# (requests, window_seconds) per client, by path prefix. First match wins.
#
# These apply to state-changing methods only. Several of these endpoints have a
# status sub-resource — GET /api/predict/ensemble/train/status/{job_id} sits under
# the training prefix — and a job that runs for minutes is polled far more often
# than it is started. Charging those polls against the hourly training budget
# exhausts it in seconds and then 429s the poll and the next training request
# alike, which is exactly what a client sees as "I clicked train once and got 429".
EXPENSIVE_LIMITS = (
    ("/api/training/bootstrap", (2, 3600)),
    ("/api/training/train", (10, 3600)),
    ("/api/agent/", (20, 3600)),
    ("/api/backtest/run", (20, 3600)),
    ("/api/predict/ensemble/train", (10, 3600)),
    # Preparation is fired automatically when a user picks a ticker, so the
    # budget has to cover ordinary browsing rather than deliberate training
    # runs. It can be this loose because the route is idempotent: repeat calls
    # for a symbol join the running job, and a bounded worker pool with a
    # post-attempt cooldown — not this limiter — is what caps the actual work.
    ("/api/models/", (120, 3600)),
)
# Methods that can start expensive work. Reads fall through to DEFAULT_LIMIT.
EXPENSIVE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DEFAULT_LIMIT = (600, 60)  # 10 requests/second sustained, per client


def get_api_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


def auth_enabled() -> bool:
    return bool(get_api_key())


def log_auth_status() -> None:
    if auth_enabled():
        logger.info("API key authentication is ENABLED (%s header required).", API_KEY_HEADER)
    else:
        logger.warning(
            "API key authentication is DISABLED — every endpoint is open, including "
            "training and agent routes. Set %s before exposing this server beyond "
            "localhost.",
            API_KEY_ENV,
        )


def _limit_for(path: str, method: str) -> tuple[int, int]:
    if method.upper() in EXPENSIVE_METHODS:
        for prefix, limit in EXPENSIVE_LIMITS:
            if path.startswith(prefix):
                return limit
    return DEFAULT_LIMIT


class SlidingWindowLimiter:
    """Per-(client, bucket) sliding window counter."""

    def __init__(self) -> None:
        self._hits: Dict[tuple[str, str], Deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, client: str, bucket: str, limit: int, window: int) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - window
        key = (client, bucket)
        with self._lock:
            hits = self._hits.get(key)
            if hits is None:
                hits = deque()
                self._hits[key] = hits
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(1, int(hits[0] + window - now) + 1)
            hits.append(now)
            # Opportunistic cleanup so idle clients do not accumulate forever.
            if len(self._hits) > 5000:
                for stale_key in [k for k, v in self._hits.items() if not v or v[-1] < cutoff]:
                    self._hits.pop(stale_key, None)
            return True, 0

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


limiter = SlidingWindowLimiter()


#: Whether to believe ``X-Forwarded-For``.
#:
#: The header is client-supplied and trivially forged, so honouring it
#: unconditionally turns the rate limiter off: a caller that varies the header per
#: request gets a fresh budget every time, and one that pins it to somebody else's
#: address spends *their* budget. It is only meaningful when a proxy this service
#: trusts is guaranteed to overwrite it, so it is opt-in and off by default —
#: which is also the safe default for the local-development case, where there is
#: no proxy and ``request.client.host`` is already the truth.
TRUST_FORWARDED_FOR_ENV = "QUANTVISION_TRUST_PROXY_HEADERS"


def trust_forwarded_for() -> bool:
    return os.getenv(TRUST_FORWARDED_FOR_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _peer_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _forwarded_client_ip(request: Request) -> Optional[str]:
    """The left-most valid address in ``X-Forwarded-For``, or None."""
    forwarded = request.headers.get("x-forwarded-for", "")
    candidate = forwarded.split(",")[0].strip()
    if not candidate:
        return None
    try:
        # Validated rather than echoed: an unparseable value is an attempt to make
        # the bucket key arbitrary, and an arbitrary key is an unlimited budget.
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        logger.debug("Ignoring malformed X-Forwarded-For value %r", candidate[:64])
        return None


def _client_id(request: Request) -> str:
    # When the key is present it is the better identity; otherwise fall back to IP.
    key = request.headers.get(API_KEY_HEADER)
    if key:
        return f"key:{key[:12]}"
    if trust_forwarded_for():
        forwarded = _forwarded_client_ip(request)
        if forwarded:
            return f"ip:{forwarded}"
    return f"ip:{_peer_ip(request)}"


def _bucket(path: str, method: str) -> str:
    """
    Counter key for this request. Must agree with `_limit_for`: a read charged to
    the training bucket would still consume the hourly training budget even though
    it is measured against the default limit.
    """
    if method.upper() in EXPENSIVE_METHODS:
        for prefix, _ in EXPENSIVE_LIMITS:
            if path.startswith(prefix):
                return prefix
    return "default"


async def error_middleware(request: Request, call_next):
    """
    Turn any unhandled exception into a 500 *below* the CORS layer.

    Starlette builds the stack as ``ServerErrorMiddleware -> user middleware ->
    ExceptionMiddleware -> router``, so a 500 produced by an ``@app.exception_handler
    (Exception)`` is created above CORSMiddleware and carries none of its headers.
    The browser then reports a plain server error as "No 'Access-Control-Allow-Origin'
    header is present", which points the investigation at CORS instead of at the
    traceback that actually caused it. Catching here — inside CORSMiddleware — means
    the 500 travels back out through it and arrives labelled as what it is.

    HTTPException and request-validation errors never reach this: ExceptionMiddleware
    sits below and converts those to responses first.
    """
    try:
        return await call_next(request)
    except Exception:  # noqa: BLE001 - deliberately the catch-all for the CORS layer
        logger.exception("Unhandled exception handling %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})


async def security_middleware(request: Request, call_next):
    """Enforce the API key (when configured) and the per-client rate limit."""
    path = request.url.path

    # CORS preflight carries no custom headers by design.
    if request.method == "OPTIONS" or path in PUBLIC_PATHS:
        return await call_next(request)

    if auth_enabled():
        provided = request.headers.get(API_KEY_HEADER, "")
        # compare_digest, not ==. Python's string equality returns as soon as two
        # bytes differ, so the time it takes leaks how long a common prefix the
        # guess shares with the key — enough, over many requests, to recover the
        # key one character at a time.
        if not provided or not secrets.compare_digest(provided, get_api_key()):
            return JSONResponse(
                status_code=401,
                content={"detail": f"Missing or invalid {API_KEY_HEADER} header"},
            )

    limit, window = _limit_for(path, request.method)
    allowed, retry_after = limiter.check(
        _client_id(request), _bucket(path, request.method), limit, window
    )
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded ({limit} requests per {window}s)"},
            headers={"Retry-After": str(retry_after)},
        )

    return await call_next(request)


def parse_origins(raw: str, fallback: Iterable[str]) -> list[str]:
    """Parse a comma-separated CORS origin list, falling back to local dev origins."""
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or list(fallback)


def cors_headers_for(request: Request, allowed_origins: Iterable[str]) -> dict[str, str]:
    """
    CORS headers for a response built above CORSMiddleware, which cannot add its own.

    Only ``ServerErrorMiddleware`` is up there — it is the outermost layer and holds
    the ``Exception`` handler — so this covers the narrow case of a failure inside the
    CORS or security middleware itself. Everything below is handled by
    `error_middleware`. The origin is echoed only when it is on the allow list, so
    this stays as strict as CORSMiddleware rather than becoming a wildcard back door.
    """
    origin = request.headers.get("origin")
    if not origin:
        return {}
    allowed = list(allowed_origins)
    if "*" not in allowed and origin not in allowed:
        return {}
    return {
        "Access-Control-Allow-Origin": "*" if allowed == ["*"] else origin,
        "Access-Control-Allow-Credentials": "true",
        "Vary": "Origin",
    }
