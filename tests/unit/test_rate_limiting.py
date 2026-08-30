import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.security import (
    DEFAULT_LIMIT,
    EXPENSIVE_LIMITS,
    _bucket,
    _limit_for,
    limiter,
    security_middleware,
)

ENSEMBLE_TRAIN = "/api/predict/ensemble/train"
ENSEMBLE_STATUS = "/api/predict/ensemble/train/status/job-123"
TRAIN_LIMIT = dict(EXPENSIVE_LIMITS)[ENSEMBLE_TRAIN][0]


@pytest.fixture
def client():
    limiter.reset()
    app = FastAPI()
    app.middleware("http")(security_middleware)

    @app.post(ENSEMBLE_TRAIN)
    def start_training():
        return {"job_id": "job-123", "status": "pending"}

    @app.get(ENSEMBLE_TRAIN + "/status/{job_id}")
    def training_status(job_id: str):
        return {"job_id": job_id, "status": "running"}

    yield TestClient(app)
    limiter.reset()


def test_status_polling_does_not_consume_the_training_budget(client):
    """
    The regression: GET .../ensemble/train/status/{id} starts with the training
    prefix, so it used to be charged against the 10-per-hour training budget. A job
    polled every three seconds burned the whole hour's allowance in half a minute,
    after which the poll and the next training request both returned 429 — read by
    the user as "I clicked train once and got 429".
    """
    for _ in range(TRAIN_LIMIT * 3):
        assert client.get(ENSEMBLE_STATUS).status_code == 200

    assert client.post(ENSEMBLE_TRAIN).status_code == 200


def test_training_posts_are_still_capped(client):
    for _ in range(TRAIN_LIMIT):
        assert client.post(ENSEMBLE_TRAIN).status_code == 200

    response = client.post(ENSEMBLE_TRAIN)
    assert response.status_code == 429
    assert "Rate limit exceeded" in response.json()["detail"]
    assert int(response.headers["Retry-After"]) > 0


def test_reads_are_still_rate_limited_by_the_default_budget(client):
    """Exempting reads from the expensive budget must not leave them unlimited."""
    assert _limit_for(ENSEMBLE_STATUS, "GET") == DEFAULT_LIMIT

    for _ in range(DEFAULT_LIMIT[0]):
        assert client.get(ENSEMBLE_STATUS).status_code == 200
    assert client.get(ENSEMBLE_STATUS).status_code == 429


def test_limit_and_bucket_agree_on_which_requests_are_expensive():
    """
    A mismatch is silent and costly: charging a request to the training counter
    while measuring it against the default limit would drain the training budget
    without ever rejecting the request that drained it.
    """
    for path, method, expensive in (
        (ENSEMBLE_TRAIN, "POST", True),
        (ENSEMBLE_TRAIN, "GET", False),
        (ENSEMBLE_STATUS, "GET", False),
        ("/api/training/bootstrap", "POST", True),
        ("/api/training/status/job-1", "GET", False),
        ("/api/backtest/run", "POST", True),
        ("/api/backtest/results", "GET", False),
        ("/api/agent/query", "POST", True),
    ):
        limit = _limit_for(path, method)
        bucket = _bucket(path, method)
        assert (limit != DEFAULT_LIMIT) == expensive, f"{method} {path} limit"
        assert (bucket != "default") == expensive, f"{method} {path} bucket"
