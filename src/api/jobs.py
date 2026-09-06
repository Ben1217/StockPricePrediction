"""
Background-job tracking for the routes that start work on a thread.

Two routers grew the same tracker independently — ``/api/training`` and
``/api/predict/ensemble`` — and they did not grow the same one. The training
copy guards its ``TTLCache`` with a lock and tolerates a job evicted before its
worker starts; the ensemble copy does neither, which left three ways for a
perfectly ordinary request to fail:

* ``TTLCache`` is not thread-safe. Its expiry bookkeeping is a linked list
  mutated on *read* as well as write, so a status poll arriving on the FastAPI
  threadpool while a worker thread writes can corrupt it. One lock covers both.
* ``job = _jobs[job_id]`` at the top of a worker raises ``KeyError`` when the
  entry aged out between the POST and the thread being scheduled. That escapes
  into the thread, where nothing is watching, and the job silently never runs.
* ``if job_id not in cache: 404`` followed by ``return cache[job_id]`` is a
  time-of-check/time-of-use race against the TTL: an entry that expires between
  the two lines raises ``KeyError`` and the poll answers 500 instead of 404.

The registry is per-process and in-memory, which is the correct scope for a
single uvicorn worker and the reason this API cannot yet run ``--workers > 1``:
a job created on one worker is invisible to the others. Moving to Redis or the
SQLite database means reimplementing this class, not its callers.

Status objects are stored by reference and mutated in place by their worker —
``job.progress = 0.4`` from a training callback is seen by the next poll. That
is the existing contract and is preserved deliberately; the lock here protects
the *container*, which is the part that was corrupting.
"""

from __future__ import annotations

import threading
import uuid
from typing import Dict, Generic, Optional, TypeVar

from cachetools import TTLCache

#: Jobs outlive a long training run but are not kept for the process lifetime.
#: An unbounded dict retained every job ever submitted.
MAX_TRACKED_JOBS = 200
JOB_TTL_SECONDS = 24 * 3600

StatusT = TypeVar("StatusT")


class JobRegistry(Generic[StatusT]):
    """Bounded, TTL'd, thread-safe map of job id to its mutable status object."""

    def __init__(
        self,
        maxsize: int = MAX_TRACKED_JOBS,
        ttl: int = JOB_TTL_SECONDS,
    ) -> None:
        self._jobs: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._lock = threading.Lock()

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    def create(self, job_id: str, status: StatusT) -> StatusT:
        """Register ``status`` under ``job_id`` and return it."""
        with self._lock:
            self._jobs[job_id] = status
        return status

    def get(self, job_id: str) -> Optional[StatusT]:
        """
        The status object, or None when the job is unknown or has expired.

        None rather than ``KeyError`` is the whole point: both callers of this —
        a worker starting up and a status poll — have a correct answer for a job
        that is no longer tracked, and neither has one for an exception.
        """
        with self._lock:
            return self._jobs.get(job_id)

    def snapshot(self) -> Dict[str, StatusT]:
        """A shallow copy of the live entries, for listing endpoints and tests."""
        with self._lock:
            return dict(self._jobs)

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def __contains__(self, job_id: object) -> bool:
        with self._lock:
            return job_id in self._jobs

    def __len__(self) -> int:
        with self._lock:
            return len(self._jobs)
