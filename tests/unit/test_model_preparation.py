"""
Automatic model preparation: classification, guards, and the API contract.

The behaviour worth pinning here is the distinction the whole feature rests on.
A symbol with no bundles is a job to run. A symbol whose bundles trained and then
failed their out-of-sample skill gate is a *result*, and re-running training on it
would refit the same bars for the same verdict. Confusing the two turns "select a
ticker and we prepare it for you" into a training loop that never converges, on
most of this repository's own symbols.

Nothing here trains anything: the executor is stubbed out. What is being tested is
the decision to train, not the training.
"""

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import app  # noqa: E402
from src.api.security import API_KEY_ENV, limiter  # noqa: E402
from src.models import preparation  # noqa: E402
from src.models.model_manager import (  # noqa: E402
    AUTO_PREPARE_ENV,
    MODEL_MAX_AGE_ENV,
    STATE_INCOMPATIBLE,
    STATE_INVALID,
    STATE_MISSING,
    STATE_READY,
    STATE_STALE,
    STATE_UNPROVEN,
    assess_direction_report,
    assess_price_bundle,
    assess_symbol,
)


def _drain_preparation_jobs(timeout=10.0):
    """
    Wait for any worker still in the pool to finish.

    `registry.reset()` clears the job dictionaries but does not touch the thread
    pool, so a job still running at the end of a test keeps going -- and calls
    whatever `preparation._execute` is bound to *by then*, which is the next
    test's monkeypatched stub. That test would see a call it never made, from a
    symbol it never asked for. Draining first is what keeps each test's executor
    list its own.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with preparation.registry._lock:
            in_flight = [job for job in preparation.registry._jobs.values() if job.active]
        if not in_flight:
            return
        time.sleep(0.01)
    raise AssertionError(
        "preparation jobs still running at teardown: "
        + ", ".join(f"{job.symbol}/{job.job_id}" for job in in_flight)
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    limiter.reset()
    preparation.registry.reset()
    yield
    _drain_preparation_jobs()
    limiter.reset()
    preparation.registry.reset()


@pytest.fixture
def bundles_dir(tmp_path, monkeypatch):
    """Point the regression-bundle lookup at an empty tree."""
    root = tmp_path / "bundles"
    root.mkdir()
    monkeypatch.setattr("src.models.ensemble_predictor.REGRESSION_BUNDLES_DIR", root)
    return root


@pytest.fixture
def report_dir(tmp_path):
    path = tmp_path / "direction_backtests"
    path.mkdir()
    return path


def _write_bundle(bundles_dir, symbol, model_type, horizon, **overrides):
    """A metadata + artifact pair the manager will accept unless told otherwise."""
    directory = bundles_dir / symbol.upper() / model_type / str(horizon)
    directory.mkdir(parents=True, exist_ok=True)
    meta = {
        "symbol": symbol.upper(),
        "model_type": model_type,
        "horizon": horizon,
        "target_type": "return_regression",
        "model_output": "predicted_return",
        "trained_at": "2026-08-30T12:00:00",
        "passes_baseline": True,
        "skill": {"test": {"skill_score": 0.12, "prediction_std": 0.02}},
        "model_path": str(directory / "model.json"),
    }
    meta.update(overrides)
    (directory / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    filename = {"xgboost": "model.json", "random_forest": "model.joblib", "lstm": "model.pt"}[model_type]
    (directory / filename).write_text("artifact", encoding="utf-8")
    return directory


def _write_report(report_dir, symbol, model="logistic", generated_at="2026-08-30T12:00:00"):
    payload = {
        "generated_at": generated_at,
        "pooled": {"model": {"accuracy": 0.54}},
        "verdict": {"ship": False},
    }
    path = report_dir / f"{symbol.upper()}_{model}_report.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestPriceBundleClassification:
    def test_absent_bundle_is_missing(self, bundles_dir):
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_MISSING
        assert status.needs_training is True
        assert status.servable is False

    def test_trained_bundle_with_skill_is_ready(self, bundles_dir):
        _write_bundle(bundles_dir, "NVDA", "xgboost", 30)
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_READY
        assert status.servable is True
        assert status.needs_training is False

    def test_failed_skill_gate_is_unproven_and_never_retrained(self, bundles_dir):
        """
        The load-bearing case.

        A bundle that trained and lost to a constant forecast is a measurement.
        Marking it "missing" would have every page load restart a run whose
        outcome is already known — and on this repository most bundles are in
        exactly this state.
        """
        _write_bundle(
            bundles_dir, "NVDA", "xgboost", 30,
            passes_baseline=False,
            skill={"test": {"skill_score": -0.31, "prediction_std": 0.001}},
        )
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_UNPROVEN
        assert status.servable is False
        assert status.needs_training is False
        assert "constant forecast" in status.detail

    def test_bundle_without_a_skill_record_is_retrained(self, bundles_dir):
        """No evidence either way is a gap. Training produces the evidence."""
        directory = bundles_dir / "NVDA" / "xgboost" / "30"
        directory.mkdir(parents=True)
        (directory / "metadata.json").write_text(json.dumps({
            "target_type": "return_regression",
            "trained_at": "2026-08-30T12:00:00",
            "model_path": str(directory / "model.json"),
        }), encoding="utf-8")
        (directory / "model.json").write_text("artifact", encoding="utf-8")

        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_INCOMPATIBLE
        assert status.needs_training is True

    def test_price_objective_bundle_is_incompatible(self, bundles_dir):
        _write_bundle(
            bundles_dir, "NVDA", "xgboost", 30,
            target_type="price_regression",
            model_output="predicted_price",
            objective="absolute_price",
        )
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_INCOMPATIBLE
        assert status.needs_training is True

    def test_missing_artifact_is_invalid(self, bundles_dir):
        directory = _write_bundle(bundles_dir, "NVDA", "xgboost", 30)
        (directory / "model.json").unlink()
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_INVALID
        assert status.needs_training is True

    def test_old_bundle_is_stale_but_still_serves(self, bundles_dir, monkeypatch):
        """
        Staleness schedules a refresh; it does not black out the page.

        A two-month-old forecast is worse than a fresh one and far better than
        an empty panel, so `stale` is both servable and trainable.
        """
        monkeypatch.setenv(MODEL_MAX_AGE_ENV, "7")
        _write_bundle(bundles_dir, "NVDA", "xgboost", 30, trained_at="2020-01-01T00:00:00")
        status = assess_price_bundle("NVDA", "xgboost", 30)
        assert status.state == STATE_STALE
        assert status.servable is True
        assert status.needs_training is True

    def test_age_policy_can_be_switched_off(self, bundles_dir, monkeypatch):
        monkeypatch.setenv(MODEL_MAX_AGE_ENV, "0")
        _write_bundle(bundles_dir, "NVDA", "xgboost", 30, trained_at="2019-01-01T00:00:00")
        assert assess_price_bundle("NVDA", "xgboost", 30).state == STATE_READY


class TestDirectionReportClassification:
    def test_absent_report_is_missing(self, report_dir):
        status = assess_direction_report("NVDA", "logistic", report_dir)
        assert status.state == STATE_MISSING
        assert status.needs_training is True

    def test_do_not_ship_report_is_ready(self, report_dir):
        """
        A "do not ship" verdict is the artifact, not a failure to produce one.

        Re-running the walk-forward because the model lost would delete the only
        thing that makes the gauge readable and replace it with the same verdict.
        """
        _write_report(report_dir, "NVDA")
        status = assess_direction_report("NVDA", "logistic", report_dir)
        assert status.state == STATE_READY
        assert status.needs_training is False

    def test_truncated_report_is_invalid(self, report_dir):
        (report_dir / "NVDA_logistic_report.json").write_text("{not json", encoding="utf-8")
        assert assess_direction_report("NVDA", "logistic", report_dir).state == STATE_INVALID

    def test_report_without_pooled_results_is_invalid(self, report_dir):
        (report_dir / "NVDA_logistic_report.json").write_text(
            json.dumps({"generated_at": "2026-08-30T12:00:00"}), encoding="utf-8"
        )
        assert assess_direction_report("NVDA", "logistic", report_dir).state == STATE_INVALID

    def test_caret_symbols_use_the_same_stem_the_writer_does(self, report_dir):
        _write_report(report_dir, "GSPC")
        assert assess_direction_report("^GSPC", "logistic", report_dir).state == STATE_READY


class TestSymbolReadiness:
    def test_step_horizon_never_gates_the_ui(self, bundles_dir, report_dir):
        """
        The 1-day bundle is trained but not displayed.

        It exists so the recursive forecast has something to roll forward; the
        ensemble degrades to a compounded path without it. Letting it block the
        tab would hide four working horizons over a fifth nobody selected.
        """
        for horizon in (7, 15, 30, 60):
            _write_bundle(bundles_dir, "NVDA", "xgboost", horizon)
        _write_report(report_dir, "NVDA")

        report = assess_symbol("NVDA", horizons=[1, 7, 15, 30, 60], report_dir=report_dir)
        assert report.price_ready is True
        assert report.direction_ready is True
        assert report.ready is True
        # It is still queued for training — just not gating.
        assert any(c.key == "xgboost@1" for c in report.trainable)

    def test_one_servable_member_is_enough_per_horizon(self, bundles_dir, report_dir):
        """A partial ensemble still answers, so it counts as ready."""
        _write_bundle(bundles_dir, "NVDA", "xgboost", 30)
        _write_bundle(bundles_dir, "NVDA", "lstm", 30, passes_baseline=False)
        _write_report(report_dir, "NVDA")

        report = assess_symbol("NVDA", horizons=[30], report_dir=report_dir)
        assert report.price_ready is True
        assert [c.key for c in report.blocked] == ["lstm@30"]

    def test_all_unproven_needs_no_training(self, bundles_dir, report_dir):
        """The terminal state: nothing serves, and nothing training can do fixes it."""
        for model_type in ("xgboost", "random_forest", "lstm"):
            _write_bundle(bundles_dir, "NVDA", model_type, 30, passes_baseline=False)
        _write_report(report_dir, "NVDA")

        report = assess_symbol("NVDA", horizons=[30], report_dir=report_dir)
        assert report.ready is False
        assert report.needs_training is False
        assert len(report.blocked) == 3
        assert "did not beat" in report.summary()


# ---------------------------------------------------------------------------
# The registry's guards
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_executor(monkeypatch):
    """Replace the training run with a recorded no-op that never finishes on its own."""
    calls = []

    def _fake_execute(job, plan):
        calls.append((job.symbol, plan))

    monkeypatch.setattr(preparation, "_execute", _fake_execute)
    return calls


def _needs_training(monkeypatch, needed=True):
    """Force the readiness answer so these tests exercise the registry, not the disk."""
    class _Report:
        needs_training = needed
        trainable = []

        def as_dict(self):
            return {"needs_training": needed}

    monkeypatch.setattr(preparation, "assess_symbol", lambda *a, **k: _Report())
    return _Report


class TestPreparationRegistry:
    def test_nothing_to_train_returns_no_job(self, monkeypatch, stub_executor):
        _needs_training(monkeypatch, needed=False)
        assert preparation.ensure_prepared("NVDA", respect_auto_flag=False) is None
        assert stub_executor == []

    def test_repeat_requests_join_the_running_job(self, monkeypatch):
        """Tab switches and re-renders must not each start a training run."""
        started = []
        gate = {"release": False}

        def _slow_execute(job, plan):
            started.append(job.job_id)
            while not gate["release"]:
                time.sleep(0.01)

        monkeypatch.setattr(preparation, "_execute", _slow_execute)
        _needs_training(monkeypatch)

        first = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        # Give the pool a moment to pick the job up before asking again.
        for _ in range(200):
            if started:
                break
            time.sleep(0.01)
        second = preparation.ensure_prepared("NVDA", respect_auto_flag=False)

        assert second.job_id == first.job_id
        gate["release"] = True
        assert len(started) == 1

    def test_cooldown_blocks_an_immediate_second_attempt(self, monkeypatch, stub_executor):
        """
        The guard that stops a doomed run repeating on every page load.

        A finished attempt that left the symbol still needing training is not
        retried until the cooldown expires — otherwise a model type that errors
        every run would be restarted by every request that notices.
        """
        monkeypatch.setenv(preparation.PREPARE_COOLDOWN_ENV, "3600")
        _needs_training(monkeypatch)

        first = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        _wait_until_finished(first)

        second = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        assert second.job_id == first.job_id
        assert len(stub_executor) == 1

    def test_cooldown_expires_with_the_job_that_justified_it(self, monkeypatch, stub_executor):
        """
        A cooldown must not outlive the job it is holding back a repeat of.

        The point of the cooldown is to hand back the previous attempt instead of
        repeating it. Once that job has aged out of the tracker there is nothing
        to hand back, and continuing to block would make `ensure` answer None —
        which every caller reads as "there is nothing to train".
        """
        monkeypatch.setenv(preparation.PREPARE_COOLDOWN_ENV, "3600")
        _needs_training(monkeypatch)

        first = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        _wait_until_finished(first)

        # Simulate the job ageing out while the cooldown record survives.
        preparation.registry._jobs.pop(first.job_id, None)

        second = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        assert second is not None
        assert second.job_id != first.job_id

    def test_force_bypasses_the_cooldown(self, monkeypatch, stub_executor):
        """The Retry button: a user who read the error and asked again."""
        monkeypatch.setenv(preparation.PREPARE_COOLDOWN_ENV, "3600")
        _needs_training(monkeypatch)

        first = preparation.ensure_prepared("NVDA", respect_auto_flag=False)
        _wait_until_finished(first)

        second = preparation.ensure_prepared("NVDA", respect_auto_flag=False, force=True)
        assert second.job_id != first.job_id
        _wait_for_executor(stub_executor, 2)

    def test_auto_flag_gates_implicit_starts_only(self, monkeypatch, stub_executor):
        monkeypatch.setenv(AUTO_PREPARE_ENV, "false")
        _needs_training(monkeypatch)

        assert preparation.ensure_prepared("NVDA") is None
        assert stub_executor == []
        assert preparation.ensure_prepared("NVDA", respect_auto_flag=False) is not None

    def test_a_failing_run_reports_its_reason(self, monkeypatch):
        def _boom(job, plan):
            raise ValueError("No data returned for ZZZZ")

        monkeypatch.setattr(preparation, "_execute", _boom)
        _needs_training(monkeypatch)

        job = preparation.ensure_prepared("ZZZZ", respect_auto_flag=False)
        _wait_until_finished(job)

        assert job.status == "failed"
        assert job.error == "No data returned for ZZZZ"
        assert job.as_dict()["error"] == "No data returned for ZZZZ"


def _wait_until_finished(job, timeout=10.0):
    deadline = time.monotonic() + timeout
    while job.active and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not job.active, f"job {job.job_id} did not finish within {timeout}s"


def _wait_for_executor(calls, expected, timeout=10.0):
    """
    Wait until the stub executor has recorded ``expected`` runs.

    `ensure()` submits to a thread pool and returns; the worker that calls
    `_execute` runs later. Asserting the call count straight after `ensure`
    returns therefore tests how fast the pool was scheduled, not whether the job
    was started -- and fails whenever the machine is busy enough to delay it.
    """
    deadline = time.monotonic() + timeout
    while len(calls) < expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(calls) == expected, (
        f"expected {expected} preparation run(s) within {timeout}s, saw {len(calls)}: {calls}"
    )


class TestJobProgress:
    def test_progress_is_weighted_by_stage(self):
        job = preparation.PreparationJob(
            job_id="j", symbol="NVDA", horizons=[30], direction_model="logistic",
            stages=preparation._new_stages(),
        )
        assert job.progress == 0.0
        job.stage("market_data").state = preparation.DONE
        assert 0.0 < job.progress < 0.1
        for stage in job.stages:
            stage.state = preparation.DONE
        assert job.progress == 1.0

    def test_skipped_stages_count_as_complete(self):
        """A symbol whose bundles are already trained must not sit at 15% forever."""
        job = preparation.PreparationJob(
            job_id="j", symbol="NVDA", horizons=[30], direction_model="logistic",
            stages=preparation._new_stages(),
        )
        job.stage("price_models").state = preparation.SKIPPED
        assert job.progress == pytest.approx(0.85)

    def test_bundle_training_dominates_the_bar(self):
        """
        The weights are measurements, not a tidy split.

        A walk-forward run takes seconds and the bundles take tens of minutes, so
        a bar that gave them comparable travel would stall for the entire wait.
        """
        weights = {stage["name"]: stage["weight"] for stage in preparation.STAGES}
        assert weights["price_models"] > 0.5
        assert weights["price_models"] > 5 * weights["direction_model"]
        assert sum(weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HTTP contract
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return TestClient(app)


class TestModelsRoute:
    def test_readiness_is_read_only(self, client, monkeypatch, stub_executor):
        """The call made on every ticker change must never start training."""
        _needs_training(monkeypatch)
        response = client.get("/api/models/NVDA")
        assert response.status_code == 200
        assert stub_executor == []

    def test_readiness_can_prepare_on_request(self, client, monkeypatch, stub_executor):
        _needs_training(monkeypatch)
        monkeypatch.setenv(AUTO_PREPARE_ENV, "true")
        response = client.get("/api/models/NVDA?prepare=true")
        assert response.status_code == 200
        _wait_for_executor(stub_executor, 1)
        assert [symbol for symbol, _ in stub_executor] == ["NVDA"]

    def test_prepare_ignores_the_auto_flag(self, client, monkeypatch, stub_executor):
        """
        The kill switch governs implicit starts, not explicit asks.

        QUANTVISION_AUTO_PREPARE=false means "a prediction request may not train
        on its own"; an operator POSTing to this route is not that.
        """
        monkeypatch.setenv(AUTO_PREPARE_ENV, "false")
        _needs_training(monkeypatch)

        response = client.post("/api/models/NVDA/prepare")
        assert response.status_code == 200
        assert response.json()["started"] is True
        _wait_for_executor(stub_executor, 1)
        assert [symbol for symbol, _ in stub_executor] == ["NVDA"]

    def test_prepare_is_idempotent_when_nothing_is_missing(self, client, monkeypatch, stub_executor):
        _needs_training(monkeypatch, needed=False)
        payload = client.post("/api/models/NVDA/prepare").json()
        assert payload["started"] is False
        assert stub_executor == []

    def test_unknown_direction_model_is_rejected(self, client):
        response = client.post(
            "/api/models/NVDA/prepare",
            json={"direction_model": "crystal_ball"},
        )
        assert response.status_code == 422
        assert "Unknown direction model" in response.json()["detail"]

    def test_job_id_is_not_read_as_a_ticker(self, client):
        """`/prepare/{job_id}` is declared first so this cannot resolve to a symbol."""
        response = client.get("/api/models/prepare/does-not-exist")
        assert response.status_code == 404
        assert "not found or expired" in response.json()["detail"]

    def test_a_started_job_can_be_polled(self, client, monkeypatch):
        gate = {"release": False}
        monkeypatch.setattr(
            preparation, "_execute",
            lambda job, plan: [time.sleep(0.01) for _ in iter(lambda: not gate["release"], False)],
        )
        _needs_training(monkeypatch)

        job_id = client.post("/api/models/NVDA/prepare").json()["preparation"]["job_id"]
        status = client.get(f"/api/models/prepare/{job_id}").json()
        gate["release"] = True

        assert status["job_id"] == job_id
        assert status["symbol"] == "NVDA"
        # Execution order, and it is not the order of importance: the cheap
        # walk-forward evaluation runs before the long bundle training so the
        # panel at the top of the tab is not held behind the one below it.
        assert [stage["name"] for stage in status["stages"]] == [
            "market_data", "direction_model", "price_models", "predictions", "analysis",
        ]
