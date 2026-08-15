"""
API key authentication and rate limiting.

Both are **opt-in** so the default local-development experience is unchanged:

* Auth activates only when ``QUANTVISION_API_KEY`` is set. Until then every request
  is allowed and a warning is logged once at startup.
* Rate limiting is always on but with generous defaults, and is stricter on the
  endpoints that start expensive work (training, backtests, agent calls).

The limiter is per-process and in-memory. That is the right scope for a single
uvicorn worker; running multiple workers needs a shared store (Redis) for both the
limiter and the job/backtest state (see the notes in routes/training.py).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Deque, Dict, Iterable

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

API_KEY_ENV = "QUANTVISION_API_KEY"
API_KEY_HEADER = "X-API-Key"

# Paths that never require a key: liveness, service metadata, and the docs.
PUBLIC_PATHS = frozenset({"/", "/health", "/docs", "/redoc", "/openapi.json"})

# (requests, window_seconds) per client, by path prefix. First match wins.
EXPENSIVE_LIMITS = (
    ("/api/training/bootstrap", (2, 3600)),
    ("/api/training/train", (10, 3600)),
    ("/api/agent/", (20, 3600)),
    ("/api/backtest/run", (20, 3600)),
    ("/api/predict/ensemble/train", (10, 3600)),
)
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


def _limit_for(path: str) -> tuple[int, int]:
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


def _client_id(request: Request) -> str:
    # When the key is present it is the better identity; otherwise fall back to IP.
    key = request.headers.get(API_KEY_HEADER)
    if key:
        return f"key:{key[:12]}"
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


def _bucket(path: str) -> str:
    for prefix, _ in EXPENSIVE_LIMITS:
        if path.startswith(prefix):
            return prefix
    return "default"


async def security_middleware(request: Request, call_next):
    """Enforce the API key (when configured) and the per-client rate limit."""
    path = request.url.path

    # CORS preflight carries no custom headers by design.
    if request.method == "OPTIONS" or path in PUBLIC_PATHS:
        return await call_next(request)

    if auth_enabled():
        provided = request.headers.get(API_KEY_HEADER, "")
        if not provided or provided != get_api_key():
            return JSONResponse(
                status_code=401,
                content={"detail": f"Missing or invalid {API_KEY_HEADER} header"},
            )

    limit, window = _limit_for(path)
    allowed, retry_after = limiter.check(_client_id(request), _bucket(path), limit, window)
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
