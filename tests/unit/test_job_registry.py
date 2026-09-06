"""
Tests for the shared background-job registry.

These cover the three failures the /api/predict/ensemble copy of this tracker had
before it shared an implementation with /api/training: an unlocked TTLCache, a
worker that raised KeyError on an evicted job, and a status poll that checked
membership and then read, racing the TTL between the two.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from src.api.jobs import JobRegistry


class Status:
    """Stand-in for TrainStatus: mutated in place by its worker."""

    def __init__(self, job_id: str, status: str = "pending"):
        self.job_id = job_id
        self.status = status
        self.progress = 0.0


def test_create_then_get_returns_the_same_object():
    registry: JobRegistry[Status] = JobRegistry()
    job_id = registry.new_id()
    created = registry.create(job_id, Status(job_id))

    # Identity, not equality: workers mutate the stored object and pollers must
    # observe those mutations rather than a copy taken at insert time.
    assert registry.get(job_id) is created
    created.progress = 0.5
    assert registry.get(job_id).progress == 0.5


def test_get_returns_none_for_an_unknown_job():
    registry: JobRegistry[Status] = JobRegistry()
    assert registry.get("never-existed") is None


def test_an_evicted_job_reads_as_none_rather_than_raising():
    """
    The failure that lost jobs silently.

    A worker thread began with ``job = _jobs[job_id]``. When the entry had been
    evicted between the POST and the thread being scheduled, that raised KeyError
    inside a bare thread — nothing caught it, nothing logged it, and the caller
    polled a job that would never move off "pending".
    """
    registry: JobRegistry[Status] = JobRegistry(maxsize=2)
    first = registry.new_id()
    registry.create(first, Status(first))
    for _ in range(3):  # push the first entry out on size
        other = registry.new_id()
        registry.create(other, Status(other))

    assert registry.get(first) is None  # not KeyError


def test_expired_entries_read_as_none():
    registry: JobRegistry[Status] = JobRegistry(ttl=0)
    job_id = registry.new_id()
    registry.create(job_id, Status(job_id))
    assert registry.get(job_id) is None


def test_ids_are_unique():
    registry: JobRegistry[Status] = JobRegistry()
    assert len({registry.new_id() for _ in range(500)}) == 500


def test_concurrent_writes_and_reads_keep_the_container_consistent():
    """
    The reason the lock is not optional.

    cachetools maintains its expiry ordering as a linked list that is mutated on
    read as well as on write, so an unsynchronised reader racing a writer can
    corrupt it. Every job written here must be readable afterwards.
    """
    registry: JobRegistry[Status] = JobRegistry(maxsize=4096)
    ids = [registry.new_id() for _ in range(400)]
    errors = []

    def writer(chunk):
        try:
            for job_id in chunk:
                registry.create(job_id, Status(job_id))
        except Exception as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    def reader():
        try:
            for _ in range(400):
                registry.get(ids[0])
                len(registry)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(ids[i::4],)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(registry) == 400
    assert all(registry.get(job_id) is not None for job_id in ids)


def test_snapshot_is_a_copy():
    registry: JobRegistry[Status] = JobRegistry()
    job_id = registry.new_id()
    registry.create(job_id, Status(job_id))

    snapshot = registry.snapshot()
    snapshot.clear()
    assert registry.get(job_id) is not None


# ---------------------------------------------------------------------------
# The ensemble route, which is what actually regressed
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    from src.api.main import app

    return TestClient(app)


def test_ensemble_status_answers_404_for_an_expired_job(client, monkeypatch):
    """
    Previously: `if job_id not in cache` then `return cache[job_id]`. An entry that
    expired between those two lines raised KeyError, and the poll answered 500
    where the contract says 404.
    """
    from src.api.routes import predict as predict_route

    monkeypatch.setattr(predict_route, "_ensemble_jobs", JobRegistry(ttl=0))
    predict_route._ensemble_jobs.create("gone", object())

    response = client.get("/api/predict/ensemble/train/status/gone")
    assert response.status_code == 404


def test_ensemble_worker_returns_quietly_when_its_job_was_evicted(monkeypatch):
    from src.api.routes import predict as predict_route
    from src.api.schemas.schemas import EnsembleTrainRequest

    monkeypatch.setattr(predict_route, "_ensemble_jobs", JobRegistry(ttl=0))
    request = EnsembleTrainRequest(symbol="AAPL", horizons=[7], model_types=["xgboost"])

    # No exception, and no attempt to train: the job is gone, so there is nothing
    # to report progress against.
    predict_route._run_ensemble_training("evicted", request)
