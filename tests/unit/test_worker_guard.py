"""
Tests for the single-worker startup guard.

The guard exists because a multi-worker start does not fail - it half-works. A
job created on worker A is invisible to a status poll routed to worker B, and the
client sees an intermittent 404 that looks like its own bug. These pin the
detection (both CLI and environment spellings), the escape hatch, and the cases
that must NOT trip it, since a false positive here refuses to boot a server that
was configured correctly.
"""

import pytest

from src.api.workers import (
    ALLOW_MULTIPLE_WORKERS_ENV,
    MultipleWorkersUnsupported,
    enforce_single_worker,
    requested_worker_count,
    worker_count_from_argv,
    worker_count_from_env,
)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "argv, expected",
    [
        (["uvicorn", "src.api.main:app"], None),
        (["uvicorn", "src.api.main:app", "--workers", "4"], 4),
        (["uvicorn", "src.api.main:app", "--workers=4"], 4),
        (["gunicorn", "-w", "8", "src.api.main:app"], 8),
        (["uvicorn", "src.api.main:app", "--workers", "1"], 1),
        # A flag with no value, or a non-numeric one, names no worker count.
        (["uvicorn", "src.api.main:app", "--workers"], None),
        (["uvicorn", "src.api.main:app", "--workers", "auto"], None),
        # --reload is the single-worker development default.
        (["uvicorn", "src.api.main:app", "--reload", "--port", "8000"], None),
    ],
)
def test_worker_count_is_read_from_the_command_line(argv, expected):
    assert worker_count_from_argv(argv) == expected


@pytest.mark.parametrize(
    "environ, expected",
    [
        ({}, None),
        ({"WEB_CONCURRENCY": "4"}, 4),
        ({"UVICORN_WORKERS": "2"}, 2),
        ({"GUNICORN_WORKERS": "16"}, 16),
        ({"WEB_CONCURRENCY": "1"}, 1),
        ({"WEB_CONCURRENCY": ""}, None),
        ({"WEB_CONCURRENCY": "not-a-number"}, None),
        ({"WEB_CONCURRENCY": "0"}, None),
        ({"WEB_CONCURRENCY": "-2"}, None),
    ],
)
def test_worker_count_is_read_from_the_environment(environ, expected):
    assert worker_count_from_env(environ) == expected


def test_the_command_line_beats_the_environment():
    """An explicit flag is a decision; WEB_CONCURRENCY is a default someone set."""
    assert requested_worker_count(
        ["uvicorn", "app", "--workers", "2"], {"WEB_CONCURRENCY": "9"}
    ) == 2


def test_the_environment_is_used_when_the_command_line_is_silent():
    assert requested_worker_count(["uvicorn", "app"], {"WEB_CONCURRENCY": "9"}) == 9


# ---------------------------------------------------------------------------
# Enforcement
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _no_escape_hatch(monkeypatch):
    monkeypatch.delenv(ALLOW_MULTIPLE_WORKERS_ENV, raising=False)


@pytest.mark.parametrize(
    "argv, environ",
    [
        (["uvicorn", "src.api.main:app"], {}),
        (["uvicorn", "src.api.main:app", "--reload"], {}),
        (["uvicorn", "src.api.main:app", "--workers", "1"], {}),
        (["uvicorn", "src.api.main:app"], {"WEB_CONCURRENCY": "1"}),
        # pytest's own argv must never look like a worker request.
        (["pytest", "tests/", "-q"], {}),
    ],
)
def test_a_single_worker_start_is_allowed(argv, environ):
    enforce_single_worker(argv, environ)


@pytest.mark.parametrize(
    "argv, environ",
    [
        (["uvicorn", "src.api.main:app", "--workers", "2"], {}),
        (["uvicorn", "src.api.main:app", "--workers=4"], {}),
        (["gunicorn", "-w", "8", "src.api.main:app"], {}),
        (["uvicorn", "src.api.main:app"], {"WEB_CONCURRENCY": "3"}),
    ],
)
def test_multiple_workers_are_refused(argv, environ):
    with pytest.raises(MultipleWorkersUnsupported):
        enforce_single_worker(argv, environ)


def test_the_refusal_explains_what_breaks_and_how_to_proceed():
    """
    A guard that only says "not supported" gets worked around with the escape
    hatch. This one has to name the failure and the real fix.
    """
    with pytest.raises(MultipleWorkersUnsupported) as excinfo:
        enforce_single_worker(["uvicorn", "app", "--workers", "4"], {})

    message = str(excinfo.value)
    assert "4 workers" in message
    assert "status poll" in message
    assert "shared store" in message
    assert ALLOW_MULTIPLE_WORKERS_ENV in message


def test_the_escape_hatch_permits_multiple_workers(monkeypatch, caplog):
    monkeypatch.setenv(ALLOW_MULTIPLE_WORKERS_ENV, "true")
    with caplog.at_level("WARNING"):
        enforce_single_worker(["uvicorn", "app", "--workers", "4"], {})
    # Permitted, but never silently: the state is still per-process.
    assert any("per-process" in record.message for record in caplog.records)


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_truthy_spellings_of_the_escape_hatch(monkeypatch, value):
    monkeypatch.setenv(ALLOW_MULTIPLE_WORKERS_ENV, value)
    enforce_single_worker(["uvicorn", "app", "--workers", "4"], {})


@pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
def test_falsey_spellings_do_not_open_the_escape_hatch(monkeypatch, value):
    monkeypatch.setenv(ALLOW_MULTIPLE_WORKERS_ENV, value)
    with pytest.raises(MultipleWorkersUnsupported):
        enforce_single_worker(["uvicorn", "app", "--workers", "4"], {})
