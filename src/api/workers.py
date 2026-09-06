"""
Refuse to start with more than one worker while server state lives in-process.

Six stores in this API are plain per-process objects: the training and ensemble
job registries, the preparation registry, uploaded datasets, stored backtest
results, the confluence store, the direction-analysis cache and the rate limiter.
Every one of them is documented as single-worker-only in the module that owns it.

The problem is that running more workers anyway does not fail. It half-works,
which is worse:

  * ``POST /api/training/train`` returns a job id from worker A; the client's
    status poll is balanced to worker B and gets 404 for a job that is running
    perfectly well.
  * ``POST /api/data/upload`` succeeds, and the follow-up read of that dataset
    404s from a different worker.
  * ``POST /api/backtest/run`` stores its result on one worker;
    ``GET /api/backtest/results/{id}`` 404s from any other.
  * The rate limiter enforces its budget per worker, so N workers permit N times
    the configured limit -- silently, and in the direction that costs money at
    the data provider.

None of that appears in a log as an error. It appears as intermittent 404s that
look like a client bug, which is why this check exists rather than another
comment.

Lift the restriction by moving that state to a shared store (Redis, or the
database that is already a dependency), not by setting the escape hatch. The
escape hatch is for when that work is done and this check is the only thing left
pointing at the old constraint.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

#: Set to a truthy value once server state is shared between processes.
ALLOW_MULTIPLE_WORKERS_ENV = "QUANTVISION_ALLOW_MULTIPLE_WORKERS"

#: Environment variables that set a worker count. ``WEB_CONCURRENCY`` is the one
#: gunicorn reads and the one most PaaS providers set for you, which makes it the
#: likeliest way this happens by accident rather than by decision.
WORKER_COUNT_ENV_VARS = ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_WORKERS")

#: CLI spellings for the same thing, in both ``--workers 4`` and ``--workers=4``
#: forms. ``-w`` is gunicorn's short option.
_WORKER_FLAGS = ("--workers", "-w")

_TRUTHY = {"1", "true", "yes", "on"}


class MultipleWorkersUnsupported(RuntimeError):
    """Raised at startup when more than one worker is requested."""


def multiple_workers_allowed() -> bool:
    return os.getenv(ALLOW_MULTIPLE_WORKERS_ENV, "").strip().lower() in _TRUTHY


def _positive_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def worker_count_from_env(environ: Optional[dict] = None) -> Optional[int]:
    """The worker count named by an environment variable, if any."""
    source = os.environ if environ is None else environ
    for name in WORKER_COUNT_ENV_VARS:
        count = _positive_int(source.get(name))
        if count is not None:
            return count
    return None


def worker_count_from_argv(argv: Optional[Sequence[str]] = None) -> Optional[int]:
    """The worker count named on the command line, if any."""
    args: List[str] = list(sys.argv if argv is None else argv)
    for index, arg in enumerate(args):
        if arg in _WORKER_FLAGS:
            # `--workers 4`: the count is the next argument.
            if index + 1 < len(args):
                count = _positive_int(args[index + 1])
                if count is not None:
                    return count
        elif arg.startswith("--workers="):
            count = _positive_int(arg.split("=", 1)[1])
            if count is not None:
                return count
    return None


def requested_worker_count(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[dict] = None,
) -> Optional[int]:
    """
    How many workers this process was asked to run, or None when nothing said.

    The command line wins over the environment, matching how uvicorn and gunicorn
    both resolve it: an explicit flag is a decision, ``WEB_CONCURRENCY`` is a
    default someone else set.
    """
    from_argv = worker_count_from_argv(argv)
    if from_argv is not None:
        return from_argv
    return worker_count_from_env(environ)


def enforce_single_worker(
    argv: Optional[Sequence[str]] = None,
    environ: Optional[dict] = None,
) -> None:
    """
    Raise :class:`MultipleWorkersUnsupported` when several workers are requested.

    Called at import time from ``src.api.main`` so the process dies at startup
    with an explanation, rather than serving intermittent 404s that look like a
    client fault.
    """
    count = requested_worker_count(argv, environ)
    if count is None or count <= 1:
        return

    if multiple_workers_allowed():
        logger.warning(
            "Running %d workers with %s set. Job status, uploaded datasets and "
            "stored backtest results are per-process and will only be visible to "
            "the worker that created them unless they have been moved to a shared "
            "store.",
            count,
            ALLOW_MULTIPLE_WORKERS_ENV,
        )
        return

    raise MultipleWorkersUnsupported(
        f"QuantVision was started with {count} workers, but its job registries, "
        "uploaded datasets, backtest results and rate limiter are all per-process. "
        "With more than one worker a job started on one worker is invisible to a "
        "status poll routed to another, uploads and backtest results 404 from the "
        "wrong worker, and the rate limiter allows N times its configured budget. "
        "These failures are intermittent 404s rather than errors, so the server "
        "refuses to start instead.\n"
        "Run a single worker (scale out with separate containers behind a load "
        "balancer if you need throughput), or move that state to a shared store "
        f"and set {ALLOW_MULTIPLE_WORKERS_ENV}=true."
    )
