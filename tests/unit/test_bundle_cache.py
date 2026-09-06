"""
Tests for the model-bundle cache.

The cache exists because `load_model_bundle` deserialised the model and scaler on
every call. Its correctness rests entirely on invalidation: it must serve the same
object while the artifacts are untouched, and must stop serving it the moment
training rewrites them. These exercise both against real files on disk rather than
mocks, because the property under test *is* the filesystem interaction.
"""

import threading
import time

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.models.bundle_cache import BundleCache, bundle_cache
from src.models.model_bundle import StaleBundleError, load_model_bundle, save_model_bundle
from src.models.random_forest_model import RandomForestModel


@pytest.fixture
def legacy_dir(tmp_path):
    """
    An empty stand-in for models/model_metadata.

    `select_model_metadata` falls back to that directory when the canonical
    lookup misses, and in this repo it holds real trained bundles - so a test
    that passes only `bundles_dir` is *not* isolated: it will happily load
    models/bundles/AAPL/random_forest/1 instead of its own fixture, and pass or
    fail based on whatever a developer last trained.
    """
    directory = tmp_path / "legacy_metadata"
    directory.mkdir()
    return directory


@pytest.fixture
def bundles_dir(tmp_path):
    """A real saved bundle: model.joblib, scaler.joblib, metadata.json."""
    X = np.array([[0.1, 1.0], [0.2, 1.1], [0.3, 1.2], [0.4, 1.3], [0.5, 1.4]], dtype=np.float32)
    y = np.array([0, 1, 1, 0, 1], dtype=np.float32)
    scaler = MinMaxScaler().fit(X)
    model = RandomForestModel(params={"n_estimators": 4, "random_state": 42, "n_jobs": 1})
    model.fit(scaler.transform(X).astype(np.float32), y)

    directory = tmp_path / "bundles"
    save_model_bundle(
        model=model,
        model_type="random_forest",
        symbol="AAPL",
        horizon=1,
        feature_columns=["feature_a", "feature_b"],
        scaler=scaler,
        metadata={
            "horizons": [1],
            "target_type": "direction",
            "objective": "next_day_direction",
        },
        models_dir=directory,
    )
    return directory


def _load(bundles_dir, legacy_dir, **kwargs):
    return load_model_bundle(
        model_type="random_forest",
        symbol="AAPL",
        horizon=1,
        bundles_dir=bundles_dir,
        metadata_dir=legacy_dir,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Behaviour through load_model_bundle
# ---------------------------------------------------------------------------

def test_uncached_loads_return_distinct_objects(bundles_dir, legacy_dir):
    """The default is unchanged: every caller still gets its own instance."""
    first = _load(bundles_dir, legacy_dir)
    second = _load(bundles_dir, legacy_dir)
    assert first is not None
    assert first is not second
    assert first.model is not second.model


def test_cached_loads_return_the_same_object(bundles_dir, legacy_dir):
    bundle_cache.clear()
    first = _load(bundles_dir, legacy_dir, use_cache=True)
    second = _load(bundles_dir, legacy_dir, use_cache=True)
    assert first is second
    assert bundle_cache.stats()["hits"] == 1


def test_the_cached_bundle_is_equivalent_to_a_fresh_one(bundles_dir, legacy_dir):
    bundle_cache.clear()
    fresh = _load(bundles_dir, legacy_dir)
    cached = _load(bundles_dir, legacy_dir, use_cache=True)

    assert cached.version_id == fresh.version_id
    assert cached.model_type == fresh.model_type
    assert cached.symbol == fresh.symbol
    assert cached.feature_columns == fresh.feature_columns
    assert cached.metadata == fresh.metadata

    rows = np.array([[0.25, 1.15]], dtype=np.float32)
    assert np.allclose(
        cached.model.predict_proba(cached.scaler.transform(rows)),
        fresh.model.predict_proba(fresh.scaler.transform(rows)),
    )


def test_rewriting_the_model_artifact_invalidates_the_entry(bundles_dir, legacy_dir):
    """
    The invalidation that matters: retraining must be visible without a restart.
    A TTL would not cover this - a bundle is stale because it was rewritten, not
    because time passed.
    """
    bundle_cache.clear()
    first = _load(bundles_dir, legacy_dir, use_cache=True)

    model_path = bundles_dir / "AAPL" / "random_forest" / "model.joblib"
    # st_mtime_ns is fine-grained, but sleep briefly so the change is unambiguous
    # on filesystems that round timestamps.
    time.sleep(0.01)
    model_path.write_bytes(model_path.read_bytes())

    second = _load(bundles_dir, legacy_dir, use_cache=True)
    assert second is not first
    assert bundle_cache.stats()["invalidations"] == 1


def test_rewriting_the_metadata_invalidates_the_entry(bundles_dir, legacy_dir):
    bundle_cache.clear()
    first = _load(bundles_dir, legacy_dir, use_cache=True)

    metadata_path = bundles_dir / "AAPL" / "random_forest" / "metadata.json"
    time.sleep(0.01)
    metadata_path.write_text(metadata_path.read_text(encoding="utf-8"), encoding="utf-8")

    assert _load(bundles_dir, legacy_dir, use_cache=True) is not first


def test_deleting_the_artifact_sends_the_next_load_back_to_disk(bundles_dir, legacy_dir):
    """
    The cache must re-read the filesystem once the file it cached is gone, rather
    than keep serving an object whose artifact no longer exists.
    """
    bundle_cache.clear()
    assert _load(bundles_dir, legacy_dir, use_cache=True) is not None

    (bundles_dir / "AAPL" / "random_forest" / "model.joblib").unlink()

    # metadata.json still names the model, so the reload attempts it and reports
    # the artifact as unloadable. Raising is the proof it went back to disk; a
    # still-cached bundle would have returned silently.
    with pytest.raises(StaleBundleError):
        _load(bundles_dir, legacy_dir, use_cache=True)


def test_a_missing_bundle_is_not_cached(tmp_path, legacy_dir):
    """
    Caching "no bundle" would keep answering None after training wrote one, which
    is precisely the state the preparation flow exists to move a symbol out of.
    """
    bundle_cache.clear()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _load(empty, legacy_dir, use_cache=True) is None
    assert len(bundle_cache) == 0


def test_explicit_metadata_bypasses_the_cache(bundles_dir, legacy_dir):
    bundle_cache.clear()
    reference = _load(bundles_dir, legacy_dir)
    again = load_model_bundle(metadata=reference.metadata, use_cache=True)
    assert again is not None
    assert again is not reference
    assert len(bundle_cache) == 0


# ---------------------------------------------------------------------------
# BundleCache itself
# ---------------------------------------------------------------------------

def test_eviction_is_least_recently_used(tmp_path):
    cache = BundleCache(maxsize=2)
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths.append(path)

    def put(key, path):
        return cache.get_or_load(key, lambda: {"key": key}, lambda _bundle: [path])

    put("a", paths[0])
    put("b", paths[1])
    put("a", paths[0])          # refreshes a's recency
    put("c", paths[2])          # evicts b, the least recently used

    assert cache.get_or_load("a", lambda: None, lambda _bundle: []) is not None
    assert cache.get_or_load("b", lambda: None, lambda _bundle: []) is None


def test_concurrent_access_is_consistent(tmp_path):
    """Entries are shared by reference across threads, so the container must hold up."""
    cache = BundleCache(maxsize=256)
    files = {}
    for i in range(50):
        path = tmp_path / f"f{i}"
        path.write_text(str(i), encoding="utf-8")
        files[i] = path

    errors = []

    def worker():
        try:
            for i in range(50):
                cache.get_or_load(i, lambda i=i: {"i": i}, lambda _bundle, i=i: [files[i]])
        except Exception as exc:  # pragma: no cover - the failure being guarded
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert len(cache) == 50
    for i in range(50):
        assert cache.get_or_load(i, lambda: None, lambda _bundle: [])["i"] == i


def test_clear_resets_entries_and_counters(tmp_path):
    cache = BundleCache()
    path = tmp_path / "x"
    path.write_text("x", encoding="utf-8")
    cache.get_or_load("k", lambda: {"v": 1}, lambda _bundle: [path])
    cache.get_or_load("k", lambda: {"v": 1}, lambda _bundle: [path])

    assert cache.stats()["hits"] == 1
    cache.clear()
    assert cache.stats() == {
        "entries": 0,
        "maxsize": cache.maxsize,
        "hits": 0,
        "misses": 0,
        "invalidations": 0,
    }
