"""
Process-local cache of loaded model bundles.

``load_model_bundle`` reads metadata, deserialises the model artifact and then
the scaler on every call. For XGBoost that is a JSON parse; for an LSTM it is a
full ``torch.load`` and state-dict restore. Nothing cached any of it, so a single
request that ranks candidate models paid the cost once per candidate, and the
Predictions tab re-paid it on every poll for bars that had not changed.

Invalidation is by file identity, never by clock
-----------------------------------------------
A TTL would be wrong here. A bundle is not stale because time passed; it is stale
because training rewrote it, and that can happen a second after it was cached or
never. So an entry records the ``(mtime_ns, size)`` of every file it was built
from -- metadata, model, scaler -- and is re-validated against the filesystem on
each hit. Retraining changes those stats, the entry is dropped, and the next
caller loads the new artifact. Deleting a bundle drops it too.

This is a `stat` per file per hit (three syscalls, microseconds) against a model
deserialisation, which is the trade this module exists to make.

Sharing
-------
Cached bundles are shared by reference across threads, so a cached model must be
safe to call concurrently. The three in use are:

  * XGBoost and scikit-learn estimators -- ``predict``/``predict_proba`` read
    fitted state and hold no per-call state on the estimator.
  * :class:`~src.models.lstm_model.LSTMModel` -- was *not* safe, because
    ``predict_proba`` and ``predict_with_uncertainty`` fight over the module's
    train/eval mode. It now serialises its own inference behind a per-instance
    lock, which is where that fix belongs.

Nothing in the codebase mutates a bundle returned by ``load_model_bundle``; every
consumer calls a predict method and reads attributes. If that ever stops being
true, the mutating caller wants its own copy, not a weaker cache.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..utils.logger import get_logger

logger = get_logger(__name__)

#: Distinct bundles held at once. Each is a fitted model plus a scaler; a few
#: dozen is small next to the frames the same process caches, and the working set
#: is "models for the symbols currently on screen", not the whole index.
DEFAULT_MAX_ENTRIES = 32

#: Identity of one file on disk: (mtime_ns, size). None when it does not exist,
#: which is itself a meaningful state -- an entry built when the scaler was
#: absent must not survive one appearing.
FileStat = Optional[Tuple[int, int]]


def _stat(path: Optional[Path]) -> FileStat:
    if path is None:
        return None
    try:
        info = path.stat()
    except OSError:
        return None
    return (info.st_mtime_ns, info.st_size)


def _fingerprint(paths: List[Optional[Path]]) -> Tuple[FileStat, ...]:
    return tuple(_stat(path) for path in paths)


class BundleCache:
    """LRU cache of loaded bundles, validated against artifact mtimes."""

    def __init__(self, maxsize: int = DEFAULT_MAX_ENTRIES) -> None:
        self.maxsize = int(maxsize)
        # key -> (fingerprint, paths, bundle)
        self._entries: "OrderedDict[Any, Tuple[Tuple[FileStat, ...], List[Optional[Path]], Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def get_or_load(
        self,
        key: Any,
        loader: Callable[[], Any],
        paths_of: Callable[[Any], List[Optional[Path]]],
    ) -> Any:
        """
        The cached bundle for ``key``, loading it through ``loader`` when needed.

        ``paths_of`` maps a freshly loaded bundle to the files it was built from,
        so the cache learns what to watch from the bundle rather than having to
        predict those paths before the load resolves them.

        ``loader`` runs outside the lock. Two callers racing on a cold key both
        load, and the second overwrites the first -- which costs one redundant
        load and keeps a slow deserialisation from blocking every other symbol.
        Holding the lock across the load would serialise the exact work this
        cache exists to avoid.
        """
        cached = self._lookup(key)
        if cached is not None:
            return cached

        bundle = loader()
        if bundle is None:
            # "No bundle" is a real answer but a cheap one to recompute, and
            # caching it would keep answering "none" after training wrote one.
            return None

        paths = [Path(p) for p in paths_of(bundle) if p is not None]
        self._store(key, paths, bundle)
        return bundle

    def _lookup(self, key: Any) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            fingerprint, paths, bundle = entry
            if _fingerprint(paths) != fingerprint:
                # Retrained, replaced or deleted since this was cached.
                del self._entries[key]
                self.invalidations += 1
                self.misses += 1
                logger.info("Bundle cache: artifacts changed for %s; reloading", key)
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return bundle

    def _store(self, key: Any, paths: List[Optional[Path]], bundle: Any) -> None:
        with self._lock:
            self._entries[key] = (_fingerprint(paths), paths, bundle)
            self._entries.move_to_end(key)
            while len(self._entries) > self.maxsize:
                self._entries.popitem(last=False)

    def invalidate(self, key: Any) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self.hits = self.misses = self.invalidations = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "invalidations": self.invalidations,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


#: The cache the API serves from.
bundle_cache = BundleCache()
