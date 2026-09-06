"""
Concurrency tests for LSTMModel inference.

The bundle cache shares one loaded model across threads. That is safe for the
tree models, whose predict methods only read fitted state, but it was not safe
for the LSTM: `predict_proba` needs the torch module in eval mode and
`predict_with_uncertainty` needs it in train mode, because train mode is what
keeps dropout live for the Monte-Carlo draws. On a shared instance the two
overlap, whichever set the mode last wins for both, and the loser silently
returns the wrong kind of answer.

Measured on an unlocked build: 149 of 12,000 concurrent `predict_proba` calls
came back perturbed by dropout, and *every one* of 164 uncertainty bands
collapsed below half its true width, the narrowest reaching exactly zero.

These tests force the damaging interleaving with an event rather than hunting for
it with brute-force contention. Hammering works, but only at a scale that makes
the suite minutes slower, and it tests the scheduler as much as the code.
"""

import threading

import numpy as np
import pytest

from src.models.lstm_model import LSTMModel


SEQ_LEN = 6
N_FEATURES = 3
#: Long enough for the other thread to be scheduled and do its work, short enough
#: that a blocked thread does not slow the suite. Only ever a lower bound on how
#: long the window is open, so a slow machine makes the test more reliable rather
#: than flakier.
WINDOW_SECONDS = 0.25
#: How long a paused predict_proba waits for the other thread to flip the mode.
#: Under the lock this always expires, so it is also the floor on that test's
#: runtime -- short enough not to matter, long enough that a loaded machine still
#: gives the sampler its chance on an unlocked build.
HOLD_SECONDS = 1.0


@pytest.fixture
def fitted_lstm():
    """A small real LSTM, fitted on separable data so predictions are stable."""
    rng = np.random.default_rng(11)
    n = 64
    X = rng.normal(0, 1, (n, SEQ_LEN, N_FEATURES)).astype(np.float32)
    # A learnable signal: the class is decided by the last step of feature 0.
    y = (X[:, -1, 0] > 0).astype(np.float32)

    model = LSTMModel(params={
        "units": 8,
        "layers": 1,
        "dropout": 0.5,   # high, so a lost dropout mask is unmistakable
        "epochs": 3,
        "batch_size": 16,
        "patience": 3,
        "sequence_length": SEQ_LEN,
    })
    model.build(input_shape=(SEQ_LEN, N_FEATURES))
    model.fit(X, y)
    return model


def _sample(rng):
    return rng.normal(0, 1, (4, SEQ_LEN, N_FEATURES)).astype(np.float32)


def _open_a_window_mid_forward(model, opened, draws_before_open=1):
    """
    Wrap the torch module so one forward pass pauses, holding the call open.

    Returns a callable that restores the original module. The pause is what gives
    a second thread a deterministic chance to interfere; with the inference lock
    in place that thread simply blocks until the pause ends.
    """
    inner = model.model
    state = {"calls": 0}

    def forward(x):
        state["calls"] += 1
        result = original_forward(x)
        if state["calls"] == draws_before_open:
            opened.set()
            # Not a synchronisation point, just a window: the other thread either
            # gets in here (no lock) or blocks until the caller finishes (lock).
            threading.Event().wait(WINDOW_SECONDS)
        return result

    original_forward = inner.forward
    inner.forward = forward
    return lambda: setattr(inner, "forward", original_forward)


def test_predict_proba_is_deterministic(fitted_lstm):
    """Baseline: with dropout off, the same input gives the same answer."""
    X = _sample(np.random.default_rng(3))
    first = fitted_lstm.predict_proba(X)
    for _ in range(5):
        assert np.allclose(fitted_lstm.predict_proba(X), first)


def _hook_mode_switches(model, eval_entered, train_entered, hold_for):
    """
    Instrument the module's ``train``/``eval`` so the two threads can rendezvous
    on the exact interleaving that breaks ``predict_proba``.

    predict_proba sets eval mode and *then* forwards, so the damaging order is:
    predictor sets eval -> sampler sets train -> predictor forwards, now sampling
    dropout. A fixed sleep does not reproduce it, because a short MC run finishes
    and restores eval mode inside the sleep. Waiting on the mode switch itself
    does, and it needs no timing assumptions.

    Returns a restore callable.
    """
    inner = model.model
    original_train = inner.train
    state = {"eval_hooked": False}

    def train(mode: bool = True):
        result = original_train(mode)
        if mode:
            train_entered.set()
        elif not state["eval_hooked"]:
            # This is predict_proba's eval(): announce it, then hold the call open
            # until the sampler has flipped the module (or it is clear it cannot,
            # because the lock is holding it out).
            state["eval_hooked"] = True
            eval_entered.set()
            train_entered.wait(hold_for)
        return result

    inner.train = train
    return lambda: setattr(inner, "train", original_train)


def test_predict_proba_is_not_perturbed_by_a_concurrent_mc_dropout_run(fitted_lstm):
    """
    predict_proba sets eval mode, then forwards. A concurrent MC-dropout run that
    sets train mode between those two steps makes that forward pass sample
    dropout, so a deterministic prediction comes back randomly perturbed.

    It never raises. It just answers wrongly, which is why the assertion is on the
    value rather than on an exception.

    Under the lock the sampler cannot get in at all: it blocks on acquisition, the
    predictor's wait expires, and the forward runs in the eval mode it set.
    """
    X = _sample(np.random.default_rng(5))
    expected = fitted_lstm.predict_proba(X)

    eval_entered = threading.Event()
    train_entered = threading.Event()
    restore = _hook_mode_switches(fitted_lstm, eval_entered, train_entered, HOLD_SECONDS)
    result = {}
    errors = []

    def predictor():
        try:
            result["probs"] = fitted_lstm.predict_proba(X)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    def sampler():
        try:
            assert eval_entered.wait(5), "predictor never reached eval()"
            # Enough draws that it is unambiguously still in train mode when the
            # predictor resumes, rather than finished and tidied up.
            fitted_lstm.predict_with_uncertainty(X, n_samples=200)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    try:
        threads = [threading.Thread(target=predictor), threading.Thread(target=sampler)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "inference deadlocked"
    finally:
        restore()

    assert errors == []
    assert np.allclose(result["probs"], expected), (
        "predict_proba ran with dropout active during a concurrent MC-dropout run "
        f"(max deviation {np.abs(result['probs'] - expected).max():.4f})"
    )


def test_the_uncertainty_band_is_not_collapsed_by_a_concurrent_predict(fitted_lstm):
    """
    The mirror-image failure, and the more damaging one.

    A concurrent predict_proba calls eval() partway through the sample loop, so
    the remaining draws run with dropout off and the band narrows toward zero -
    an uncertainty estimate reporting no uncertainty, which is wrong in the
    direction that reads as confidence.
    """
    X = _sample(np.random.default_rng(7))

    # The band this model actually produces with nothing else running.
    _, lower, upper = fitted_lstm.predict_with_uncertainty(X, n_samples=32)
    reference_width = float(np.mean(upper - lower))
    assert reference_width > 0, "fixture must have live dropout for this test to mean anything"

    opened = threading.Event()
    restore = _open_a_window_mid_forward(fitted_lstm, opened)
    result = {}

    def sampler():
        _, low, high = fitted_lstm.predict_with_uncertainty(X, n_samples=32)
        result["width"] = float(np.mean(high - low))

    def predictor():
        assert opened.wait(5), "sampler never reached its forward pass"
        fitted_lstm.predict_proba(X)

    try:
        threads = [threading.Thread(target=sampler), threading.Thread(target=predictor)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
            assert not thread.is_alive(), "inference deadlocked"
    finally:
        restore()

    # Sampling noise moves the width around; a lost dropout mask on 31 of 32 draws
    # collapses it. Half the reference separates the two without being flaky.
    assert result["width"] > reference_width * 0.5, (
        f"band collapsed to {result['width']:.5f} against a reference width of "
        f"{reference_width:.5f}"
    )


def test_eval_mode_is_restored_when_a_forward_pass_fails(fitted_lstm):
    """
    predict_with_uncertainty switches the module to train mode. If a draw raises
    and the mode is not restored, every later predict_proba silently runs with
    dropout on - a failure that outlives the request that caused it.
    """
    X = _sample(np.random.default_rng(9))
    expected = fitted_lstm.predict_proba(X)

    bad = np.zeros((4, SEQ_LEN, N_FEATURES + 1), dtype=np.float32)
    with pytest.raises(Exception):
        fitted_lstm.predict_with_uncertainty(bad, n_samples=4)

    assert not fitted_lstm.model.training
    assert np.allclose(fitted_lstm.predict_proba(X), expected)


def test_concurrent_predicts_agree_with_the_single_threaded_answer(fitted_lstm):
    """Many threads on one shared model, which is what the bundle cache creates."""
    X = _sample(np.random.default_rng(15))
    expected = fitted_lstm.predict_proba(X)

    mismatches = []
    errors = []

    def worker():
        try:
            for _ in range(50):
                if not np.allclose(fitted_lstm.predict_proba(X), expected):
                    mismatches.append(1)
        except Exception as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert mismatches == []


def test_the_model_survives_a_pickle_round_trip(fitted_lstm):
    """
    The lock is not picklable, so it is dropped on the way out and rebuilt on the
    way in. joblib persists these models, so a broken round trip would surface as
    a load failure rather than as anything about threads.
    """
    import pickle

    X = _sample(np.random.default_rng(13))
    expected = fitted_lstm.predict_proba(X)

    restored = pickle.loads(pickle.dumps(fitted_lstm))
    assert np.allclose(restored.predict_proba(X), expected)
    # Usable concurrently straight away, with its own fresh lock.
    assert restored._lock() is not fitted_lstm._lock()
