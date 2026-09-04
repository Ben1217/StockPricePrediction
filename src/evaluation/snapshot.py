"""
Frozen, content-addressed dataset snapshots (Addendum A, A10.1 and A9 item 10).

Price vendors revise history quietly. Yahoo restates an adjustment factor after
a late dividend declaration, back-fills a session it missed, and occasionally
moves a close by a cent months after the fact. An evaluation that calls the API
at run time is therefore scoring a *different dataset* on every run, and the
p-value it reports cannot be reproduced -- not by a reviewer, and not by its own
author six weeks later. Re-running a backtest until the data cooperates is the
same error as re-running it until the seed does; it is just harder to see.

So evaluation reads a snapshot, never a live API. :func:`write_snapshot` freezes
a set of frames to disk with a SHA-256 per ticker plus a manifest hash over the
whole set; :func:`read_snapshot` re-hashes every frame it loads and refuses to
hand back a dataset that has drifted from its manifest. A *silent* mismatch is
the single failure this module exists to prevent, so the default is to raise,
and the exception names every ticker whose hash moved.

Two hashes, two jobs
--------------------
``SnapshotEntry.content_sha256`` is :func:`src.data.direction_data.frame_content_hash`
-- the same hasher the bar loader already stamps into every ``BarLoad.meta``, so
a snapshot entry can be compared directly against a fresh download's metadata
without a second, subtly different hashing convention existing in the codebase.
That hasher rounds values to 10 decimals before digesting them, and this module
is the reason: a frame must survive a Parquet (or CSV) round-trip with an
unchanged digest. Parquet stores float64 bit-exactly and CSV writes the shortest
repr that round-trips, so in practice nothing changes at all; the rounding is
the belt to that braces. Prices are O(1e3), where float64 resolves ~1e-13, so
discarding everything below 1e-10 throws away storage noise and no data.

``SnapshotManifest.manifest_sha256`` is a SHA-256 over the *sorted* per-ticker
``(ticker, content_sha256)`` pairs, tab-joined and newline-separated. Sorting
makes it independent of the order the frames were handed in; using only the
pairs makes it independent of the file layout, so the same bars written as
Parquet here and CSV there give the same manifest hash. It identifies the
dataset, not the directory.

Survivorship (A4.5)
-------------------
A universe taken from today's index membership has already dropped everything
that failed. The up base rate and the long-horizon drift are both biased upward
by exactly the names that are missing, and no amount of walk-forward discipline
*inside* the snapshot repairs a universe chosen after the fact. That cannot be
left implicit, so a manifest is not writable without a survivorship record:
``write_snapshot`` raises ``ValueError`` rather than let the question go
unanswered, and :func:`snapshot_caption` prints the limitation under every table
whenever the universe is not point-in-time.

Public API:
    survivorship_record(point_in_time, source, as_of, note="") -> dict
    write_snapshot(frames, path, *, snapshot_id, source, notes="", sectors=None,
                   survivorship=None, storage_format="auto") -> SnapshotManifest
    read_snapshot(path, *, verify=True) -> (dict[str, DataFrame], SnapshotManifest)
    verify_snapshot(path) -> dict
    snapshot_caption(manifest) -> str
    manifest_content_hash(entries) -> str
    SnapshotEntry, SnapshotManifest, SnapshotIntegrityError
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.direction_data import frame_content_hash
from ..utils.logger import get_logger

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"
BARS_DIRNAME = "bars"
STORAGE_FORMATS = ("parquet", "csv")

# A4.5. The exact sentence that must appear in the manifest, and under every
# results table, when the universe was not reconstructed point-in-time.
SURVIVORSHIP_LIMITATION = (
    "Universe built from a current index-membership snapshot. Delisted and failed "
    "companies are excluded, which inflates the up base rate and long-horizon drift "
    "and makes these results non-transferable."
)


class SnapshotIntegrityError(RuntimeError):
    """
    A snapshot on disk no longer matches the manifest that describes it.

    Carries the machine-readable detail alongside the message: ``mismatched`` is
    a list of ``(ticker, expected_sha256, actual_sha256)``, ``missing`` a list of
    tickers whose bar file is gone, and ``manifest_hash_mismatch`` says whether
    the manifest's own hash disagrees with its entries -- which is what an edited
    manifest looks like.
    """

    def __init__(
        self,
        message: str,
        *,
        mismatched: Sequence[Tuple[str, str, str]] = (),
        missing: Sequence[str] = (),
        manifest_hash_mismatch: bool = False,
    ) -> None:
        super().__init__(message)
        self.mismatched: List[Tuple[str, str, str]] = [
            (str(t), str(expected), str(actual)) for t, expected, actual in mismatched
        ]
        self.missing: List[str] = [str(t) for t in missing]
        self.manifest_hash_mismatch = bool(manifest_hash_mismatch)

    @property
    def tickers(self) -> List[str]:
        """Every ticker implicated, mismatched or missing, in sorted order."""
        return sorted({item[0] for item in self.mismatched} | set(self.missing))


@dataclass(frozen=True)
class SnapshotEntry:
    """One frozen ticker: what was stored, over what range, and its content hash."""

    ticker: str
    rows: int
    start: str
    end: str
    content_sha256: str
    columns: List[str]
    sector: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "rows": int(self.rows),
            "start": self.start,
            "end": self.end,
            "content_sha256": self.content_sha256,
            "columns": list(self.columns),
            "sector": self.sector,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotEntry":
        required = ("ticker", "rows", "start", "end", "content_sha256")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"snapshot entry is missing required fields: {missing}")
        return cls(
            ticker=str(payload["ticker"]),
            rows=int(payload["rows"]),
            start=str(payload["start"]),
            end=str(payload["end"]),
            content_sha256=str(payload["content_sha256"]),
            columns=[str(column) for column in (payload.get("columns") or [])],
            sector=payload.get("sector"),
        )


@dataclass(frozen=True)
class SnapshotManifest:
    """
    The provenance record for a frozen dataset.

    ``manifest_sha256`` is a hash of the *dataset*, computed by
    :func:`manifest_content_hash` from the sorted per-ticker content hashes, so
    it does not move when the frames are handed over in a different order or
    stored in a different file format. ``storage_format`` records how the frames
    were actually written, so :func:`read_snapshot` can round-trip them.
    """

    snapshot_id: str
    created_at: str
    entries: List[SnapshotEntry]
    manifest_sha256: str
    library_versions: Dict[str, str]
    source: str
    notes: str = ""
    survivorship: Dict[str, Any] = field(default_factory=dict)
    storage_format: str = "parquet"

    @property
    def tickers(self) -> List[str]:
        return [entry.ticker for entry in self.entries]

    @property
    def start(self) -> Optional[str]:
        """Earliest first-bar date across entries; None for an empty manifest."""
        return min((entry.start for entry in self.entries), default=None)

    @property
    def end(self) -> Optional[str]:
        """Latest last-bar date across entries; None for an empty manifest."""
        return max((entry.end for entry in self.entries), default=None)

    @property
    def total_rows(self) -> int:
        return int(sum(entry.rows for entry in self.entries))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "entries": [entry.to_dict() for entry in self.entries],
            "manifest_sha256": self.manifest_sha256,
            "library_versions": dict(self.library_versions),
            "source": self.source,
            "notes": self.notes,
            "survivorship": dict(self.survivorship),
            "storage_format": self.storage_format,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SnapshotManifest":
        required = ("snapshot_id", "created_at", "manifest_sha256")
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"snapshot manifest is missing required fields: {missing}")
        storage_format = str(payload.get("storage_format", "parquet"))
        if storage_format not in STORAGE_FORMATS:
            raise ValueError(
                f"manifest records storage_format={storage_format!r}, "
                f"which is not one of {STORAGE_FORMATS}"
            )
        versions = payload.get("library_versions") or {}
        return cls(
            snapshot_id=str(payload["snapshot_id"]),
            created_at=str(payload["created_at"]),
            entries=[SnapshotEntry.from_dict(item) for item in (payload.get("entries") or [])],
            manifest_sha256=str(payload["manifest_sha256"]),
            library_versions={str(k): str(v) for k, v in versions.items()},
            source=str(payload.get("source", "")),
            notes=str(payload.get("notes", "")),
            survivorship=dict(payload.get("survivorship") or {}),
            storage_format=storage_format,
        )


def manifest_content_hash(entries: Iterable[Any]) -> str:
    """
    SHA-256 over the sorted ``(ticker, content_sha256)`` pairs of a snapshot.

    Accepts :class:`SnapshotEntry` objects, mappings carrying those two keys, or
    bare 2-tuples. The payload is one ``ticker<TAB>hash`` line per pair, sorted
    and joined with ``\\n``, so the digest is independent both of the order the
    entries arrive in and of how the frames were stored on disk. It answers "is
    this the same dataset?", not "is this the same directory?".
    """
    pairs: List[Tuple[str, str]] = []
    for entry in entries:
        if isinstance(entry, SnapshotEntry):
            pairs.append((str(entry.ticker), str(entry.content_sha256)))
        elif isinstance(entry, Mapping):
            pairs.append((str(entry["ticker"]), str(entry["content_sha256"])))
        else:
            ticker, digest = entry
            pairs.append((str(ticker), str(digest)))
    payload = "\n".join(f"{ticker}\t{digest}" for ticker, digest in sorted(pairs))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def survivorship_record(
    point_in_time: bool,
    source: str,
    as_of: Optional[str],
    note: str = "",
) -> Dict[str, Any]:
    """
    The A4.5 survivorship record every snapshot manifest must carry.

    Parameters
    ----------
    point_in_time : bool
        True only if the universe was reconstructed as it stood on ``as_of`` --
        including the names that have since been delisted, acquired, or gone to
        zero. Taking today's index constituents and back-filling their prices is
        *not* point-in-time, however far back the prices go.
    source : str
        Where the membership list came from. Required: an unattributed universe
        cannot be re-derived.
    as_of : str, optional
        The date the membership is asserted for. Mandatory when
        ``point_in_time`` is True, because a point-in-time claim carrying no date
        is not a claim about anything.
    note : str
        Free text -- the reconstitution rule, the vendor's file version, and so on.

    Returns
    -------
    dict
        ``point_in_time``, ``source``, ``as_of``, ``note`` and ``limitation``.
        ``limitation`` holds :data:`SURVIVORSHIP_LIMITATION` when the universe is
        not point-in-time and is ``None`` when it is -- ``None`` because there is
        no survivorship limitation to state, not because one is unknown.

    Raises
    ------
    ValueError
        ``source`` empty, ``as_of`` unparseable, or ``point_in_time`` asserted
        without an ``as_of``.
    """
    if not isinstance(point_in_time, bool):
        raise ValueError(f"point_in_time must be a bool, got {point_in_time!r}")

    cleaned_source = str(source or "").strip()
    if not cleaned_source:
        raise ValueError(
            "survivorship_record requires a source naming where the universe came from"
        )

    as_of_string: Optional[str] = None
    if as_of is not None and str(as_of).strip():
        try:
            as_of_string = str(pd.Timestamp(as_of).date())
        except (ValueError, TypeError) as exc:
            raise ValueError(f"as_of is not a parseable date: {as_of!r} ({exc})") from exc

    if point_in_time and as_of_string is None:
        raise ValueError(
            "a point-in-time universe must carry the as_of date it is point-in-time as of; "
            "pass point_in_time=False if the membership list is simply today's"
        )

    return {
        "point_in_time": point_in_time,
        "source": cleaned_source,
        "as_of": as_of_string,
        "note": str(note or ""),
        "limitation": None if point_in_time else SURVIVORSHIP_LIMITATION,
    }


def _validate_survivorship(record: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    Normalise a survivorship record, refusing to proceed without one (A4.5).

    A hand-built record is accepted, but the mandatory limitation sentence is
    forced into it when the universe is not point-in-time: the point of the rule
    is that the caveat travels with the data, and a caller who omitted the
    sentence did not thereby make the bias go away. Any extra text the caller
    supplied is kept after it.
    """
    if record is None:
        raise ValueError(
            "write_snapshot requires a survivorship record (Addendum A4.5). Pass "
            "survivorship=survivorship_record(point_in_time=..., source=..., as_of=...). "
            "A universe that silently drops delisted names inflates every up-rate in the "
            "report, so the question is answered in the manifest or the snapshot is not "
            "written."
        )
    if not isinstance(record, Mapping):
        raise ValueError(f"survivorship must be a mapping, got {type(record).__name__}")
    if "point_in_time" not in record:
        raise ValueError("survivorship record must state point_in_time (True or False)")

    point_in_time = record["point_in_time"]
    if not isinstance(point_in_time, bool):
        raise ValueError(f"survivorship['point_in_time'] must be a bool, got {point_in_time!r}")

    source = str(record.get("source") or "").strip()
    if not source:
        raise ValueError("survivorship record must name the universe source")

    normalised: Dict[str, Any] = dict(record)
    normalised["point_in_time"] = point_in_time
    normalised["source"] = source
    normalised["as_of"] = record.get("as_of")
    normalised["note"] = str(record.get("note") or "")

    if point_in_time:
        normalised["limitation"] = record.get("limitation")
    else:
        existing = str(record.get("limitation") or "").strip()
        if SURVIVORSHIP_LIMITATION in existing:
            normalised["limitation"] = existing
        else:
            normalised["limitation"] = " ".join(
                part for part in (SURVIVORSHIP_LIMITATION, existing) if part
            )
    return normalised


def _iso_utc_now() -> str:
    """ISO 8601 in UTC with an explicit Z. Local timestamps are not comparable."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _library_versions() -> Dict[str, str]:
    """
    The versions that produced the snapshot.

    ``python``/``numpy``/``pandas`` always; ``yfinance`` and ``pyarrow`` when
    importable, since the first decides what the bars looked like on download
    and the second decides whether the frames could be stored as Parquet.
    """
    versions: Dict[str, str] = {
        "python": platform.python_version(),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
    }
    for name in ("yfinance", "pyarrow"):
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - both are optional at evaluation time
            logger.debug("Version probe for %s failed: %s", name, exc)
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def _parquet_available() -> bool:
    """True when pandas has an engine that can write Parquet."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            importlib.import_module(engine)
            return True
        except Exception:  # noqa: BLE001 - absence is the whole question being asked
            continue
    return False


def _resolve_storage_format(requested: str) -> str:
    choice = str(requested or "auto").strip().lower()
    if choice not in ("auto",) + STORAGE_FORMATS:
        raise ValueError(f"storage_format must be 'auto', 'parquet' or 'csv', got {requested!r}")
    if choice == "csv":
        return "csv"

    available = _parquet_available()
    if choice == "parquet":
        if not available:
            raise ValueError(
                "storage_format='parquet' was requested but neither pyarrow nor fastparquet "
                "is importable; pass storage_format='csv' to choose the fallback explicitly"
            )
        return "parquet"
    if available:
        return "parquet"

    logger.warning(
        "No Parquet engine importable (pyarrow/fastparquet); writing the snapshot as CSV. "
        "The manifest records the format so read_snapshot still round-trips, and "
        "frame_content_hash rounds to 10 dp so the content hashes are unaffected."
    )
    return "csv"


def _safe_stem(ticker: str) -> str:
    """Filesystem-safe stem for a ticker. Collisions are caught by the caller."""
    stem = "".join(c for c in str(ticker) if c.isalnum() or c in "-_.")
    return stem or "TICKER"


def _bar_filename(ticker: str, storage_format: str) -> str:
    return f"{_safe_stem(ticker)}.{storage_format}"


def _normalise_ticker(ticker: Any) -> str:
    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("snapshot frames cannot be keyed by an empty ticker")
    return symbol


def _validate_frame(ticker: str, frame: Any) -> pd.DataFrame:
    """
    Reject anything :func:`frame_content_hash` could not hash reproducibly.

    The hasher casts the whole frame to float64 and reads the index as int64
    nanoseconds, so a non-numeric column or a non-datetime index would either
    raise deep inside hashing or hash something other than what was stored. An
    out-of-order or duplicated index is rejected too: the digest covers the rows
    in the order given, so two frames holding the same bars in a different order
    are -- correctly -- different snapshots, and one of them is a bug.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(
            f"snapshot entry for {ticker} must be a DataFrame, got {type(frame).__name__}"
        )
    if frame.empty:
        raise ValueError(
            f"snapshot frame for {ticker} is empty; an entry needs a real date range and a "
            f"content hash over real rows"
        )
    if frame.shape[1] == 0:
        raise ValueError(f"snapshot frame for {ticker} has no columns")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(
            f"snapshot frame for {ticker} must have a DatetimeIndex, "
            f"got {type(frame.index).__name__}"
        )
    if frame.index.tz is not None:
        raise ValueError(
            f"snapshot frame for {ticker} has a tz-aware index ({frame.index.tz}); store "
            f"tz-naive bars as src.data.direction_data.clean_daily_bars produces them, so "
            f"the CSV fallback and Parquet agree on the index"
        )
    if not frame.index.is_unique:
        duplicates = int(frame.index.duplicated().sum())
        raise ValueError(f"snapshot frame for {ticker} has {duplicates} duplicate index entries")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"snapshot frame for {ticker} is not sorted ascending by date")

    non_numeric = [
        str(column)
        for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if non_numeric:
        raise ValueError(
            f"snapshot frame for {ticker} has non-numeric columns {non_numeric}; "
            f"frame_content_hash casts the frame to float64 and cannot hash them"
        )
    return frame


def _write_frame(frame: pd.DataFrame, path: Path, storage_format: str) -> None:
    if storage_format == "parquet":
        frame.to_parquet(path)
        return
    # The CSV fallback preserves values -- pandas writes the shortest repr that
    # round-trips a float64 -- and therefore preserves the content hash. It does
    # not promise dtype fidelity the way Parquet does, and it labels an unnamed
    # index "Date" so the header has a cell to put it in.
    frame.to_csv(path, index_label=frame.index.name or "Date")


def _read_frame(path: Path, storage_format: str) -> pd.DataFrame:
    if storage_format == "parquet":
        frame = pd.read_parquet(path)
    elif storage_format == "csv":
        frame = pd.read_csv(path, index_col=0)
    else:
        raise ValueError(f"manifest records unknown storage_format {storage_format!r}")
    frame.index = pd.to_datetime(frame.index)
    return frame


def write_snapshot(
    frames: Mapping[str, pd.DataFrame],
    path: Path | str,
    *,
    snapshot_id: str,
    source: str,
    notes: str = "",
    sectors: Optional[Mapping[str, Optional[str]]] = None,
    survivorship: Optional[Mapping[str, Any]] = None,
    storage_format: str = "auto",
) -> SnapshotManifest:
    """
    Freeze ``frames`` to ``path`` and return the manifest describing them.

    Layout::

        <path>/manifest.json
        <path>/bars/<TICKER>.parquet      (or .csv when no Parquet engine exists)

    Every frame is hashed with :func:`src.data.direction_data.frame_content_hash`
    *before* it is written, and the digest goes into the manifest. Parquet stores
    float64 bit-exactly, so the digest survives the round-trip unchanged; the
    hasher's 10-decimal rounding exists for exactly this reason and covers the
    CSV fallback too, where values pass through a decimal repr.

    The manifest hash is :func:`manifest_content_hash` over the sorted per-ticker
    pairs, so writing the same data twice -- in either key order, in either
    format -- produces the same ``manifest_sha256``. That is the number a report
    quotes to assert two runs saw the same dataset.

    Parameters
    ----------
    frames : mapping of ticker -> DataFrame
        Numeric columns, tz-naive DatetimeIndex, unique and sorted ascending.
    path : path-like
        Snapshot root. Created if absent; an existing snapshot at the same path
        is overwritten entry by entry.
    snapshot_id : str
        The name the report refers to this dataset by, e.g. ``sp100_2015_2024_v1``.
    source : str
        Where the bars came from, e.g. ``yfinance auto_adjust=False + dividend
        adjustment (src.data.direction_data.load_daily_bars)``.
    notes : str
        Free text carried into the manifest.
    sectors : mapping, optional
        ticker -> sector, for the cross-sectional slices. Absent tickers get None.
    survivorship : mapping
        Required. Build it with :func:`survivorship_record`. ``None`` raises
        ``ValueError``: A4.5 says the bias is disclosed, not silently ignored.
    storage_format : {'auto', 'parquet', 'csv'}
        ``'auto'`` uses Parquet when an engine is importable and falls back to
        CSV with a logged warning when it is not. ``'parquet'`` raises rather
        than quietly downgrade a format the caller asked for by name.

    Raises
    ------
    ValueError
        No survivorship record; empty ``frames``; a frame that cannot be hashed
        reproducibly; two tickers whose filenames would collide.
    """
    root = Path(path)
    identifier = str(snapshot_id or "").strip()
    if not identifier:
        raise ValueError("snapshot_id is required: a snapshot the report cannot name is useless")
    source_string = str(source or "").strip()
    if not source_string:
        raise ValueError("source is required: it is the only record of where these bars came from")

    survivorship_normalised = _validate_survivorship(survivorship)

    if not frames:
        raise ValueError("write_snapshot was given no frames; there is nothing to freeze")

    resolved_format = _resolve_storage_format(storage_format)
    sector_map = {_normalise_ticker(k): v for k, v in dict(sectors or {}).items()}

    validated: Dict[str, pd.DataFrame] = {}
    for raw_ticker, frame in frames.items():
        ticker = _normalise_ticker(raw_ticker)
        if ticker in validated:
            raise ValueError(f"duplicate ticker {ticker} after normalisation of {raw_ticker!r}")
        validated[ticker] = _validate_frame(ticker, frame)

    stems: Dict[str, str] = {}
    for ticker in validated:
        stem = _safe_stem(ticker)
        if stem in stems:
            raise ValueError(
                f"tickers {stems[stem]} and {ticker} both sanitise to the filename stem "
                f"{stem!r}; rename one before snapshotting"
            )
        stems[stem] = ticker

    bars_dir = root / BARS_DIRNAME
    bars_dir.mkdir(parents=True, exist_ok=True)

    entries: List[SnapshotEntry] = []
    for ticker in sorted(validated):
        frame = validated[ticker]
        digest = frame_content_hash(frame)
        _write_frame(frame, bars_dir / _bar_filename(ticker, resolved_format), resolved_format)
        entries.append(
            SnapshotEntry(
                ticker=ticker,
                rows=int(len(frame)),
                start=str(pd.Timestamp(frame.index[0]).date()),
                end=str(pd.Timestamp(frame.index[-1]).date()),
                content_sha256=digest,
                columns=[str(column) for column in frame.columns],
                sector=sector_map.get(ticker),
            )
        )

    expected_files = {_bar_filename(entry.ticker, resolved_format) for entry in entries}
    stale = sorted(
        item.name for item in bars_dir.iterdir() if item.is_file() and item.name not in expected_files
    )
    if stale:
        logger.warning(
            "Snapshot %s left %d file(s) in %s that its manifest does not list: %s. "
            "verify_snapshot will report them as extra.",
            identifier, len(stale), bars_dir, stale,
        )

    manifest = SnapshotManifest(
        snapshot_id=identifier,
        created_at=_iso_utc_now(),
        entries=entries,
        manifest_sha256=manifest_content_hash(entries),
        library_versions=_library_versions(),
        source=source_string,
        notes=str(notes or ""),
        survivorship=survivorship_normalised,
        storage_format=resolved_format,
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    logger.info(
        "Froze snapshot %s: %d tickers, %d rows, [%s..%s], format=%s, manifest sha256=%s",
        identifier, len(entries), manifest.total_rows, manifest.start, manifest.end,
        resolved_format, manifest.manifest_sha256[:12],
    )
    return manifest


def _load_manifest(root: Path) -> SnapshotManifest:
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No snapshot manifest at {manifest_path}")
    return SnapshotManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def _rehash_entries(
    root: Path, manifest: SnapshotManifest
) -> Tuple[Dict[str, pd.DataFrame], List[Tuple[str, str, str]], List[str]]:
    """Load and re-hash every entry. Returns ``(frames, mismatched, missing)``."""
    frames: Dict[str, pd.DataFrame] = {}
    mismatched: List[Tuple[str, str, str]] = []
    missing: List[str] = []
    bars_dir = root / BARS_DIRNAME

    for entry in manifest.entries:
        file_path = bars_dir / _bar_filename(entry.ticker, manifest.storage_format)
        if not file_path.exists():
            missing.append(entry.ticker)
            continue
        try:
            frame = _read_frame(file_path, manifest.storage_format)
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not hide the rest
            logger.error("Snapshot file for %s could not be read: %s", entry.ticker, exc)
            mismatched.append((entry.ticker, entry.content_sha256, f"unreadable: {exc}"))
            continue
        actual = frame_content_hash(frame)
        if actual != entry.content_sha256:
            mismatched.append((entry.ticker, entry.content_sha256, actual))
        frames[entry.ticker] = frame
    return frames, mismatched, missing


def _integrity_message(
    root: Path,
    manifest: SnapshotManifest,
    mismatched: Sequence[Tuple[str, str, str]],
    missing: Sequence[str],
    recomputed: str,
) -> str:
    parts: List[str] = []
    if mismatched:
        detail = ", ".join(
            f"{ticker} (manifest {expected[:12]}, on disk {str(actual)[:12]})"
            for ticker, expected, actual in mismatched
        )
        parts.append(f"content hash moved for {len(mismatched)} ticker(s): {detail}")
    if missing:
        parts.append(f"bar file missing for {len(missing)} ticker(s): {', '.join(missing)}")
    if recomputed != manifest.manifest_sha256:
        parts.append(
            f"manifest records sha256 {manifest.manifest_sha256[:12]} but its own entries "
            f"hash to {recomputed[:12]}"
        )
    return (
        f"Snapshot {manifest.snapshot_id} at {root} does not match its manifest: "
        + "; ".join(parts)
    )


def read_snapshot(
    path: Path | str, *, verify: bool = True
) -> Tuple[Dict[str, pd.DataFrame], SnapshotManifest]:
    """
    Load a frozen snapshot, re-hashing every frame against the manifest.

    Parameters
    ----------
    path : path-like
        Snapshot root, the directory holding ``manifest.json``.
    verify : bool
        When True (the default), a content hash that has moved, a missing bar
        file, or a manifest whose own hash disagrees with its entries raises
        :class:`SnapshotIntegrityError`, and the message names every ticker
        involved. When False the same problems are logged at ERROR and the
        frames are returned anyway -- for inspecting a damaged snapshot, never
        for scoring a model with one.

    Returns
    -------
    (frames, manifest)
        ``frames`` is keyed by the manifest's tickers. A ticker whose file is
        missing or unreadable is absent from it.

    Raises
    ------
    FileNotFoundError
        No ``manifest.json`` under ``path``.
    SnapshotIntegrityError
        ``verify=True`` and the data on disk is not the data the manifest
        describes. Verification is on by default because the failure this module
        exists to prevent is the one nobody notices.
    """
    root = Path(path)
    manifest = _load_manifest(root)
    frames, mismatched, missing = _rehash_entries(root, manifest)
    recomputed = manifest_content_hash(manifest.entries)
    manifest_moved = recomputed != manifest.manifest_sha256

    if mismatched or missing or manifest_moved:
        message = _integrity_message(root, manifest, mismatched, missing, recomputed)
        if verify:
            raise SnapshotIntegrityError(
                message,
                mismatched=mismatched,
                missing=missing,
                manifest_hash_mismatch=manifest_moved,
            )
        logger.error("%s (verify=False, returning it anyway)", message)

    _log_version_drift(manifest)
    logger.info(
        "Loaded snapshot %s: %d tickers, %d rows, [%s..%s], manifest sha256=%s%s",
        manifest.snapshot_id, len(frames), manifest.total_rows, manifest.start, manifest.end,
        manifest.manifest_sha256[:12], "" if verify else " (unverified)",
    )
    return frames, manifest


def _log_version_drift(manifest: SnapshotManifest) -> None:
    """Note when the reading environment is not the writing environment."""
    current = _library_versions()
    drift = {
        name: f"{recorded} -> {current[name]}"
        for name, recorded in manifest.library_versions.items()
        if name in current and current[name] != recorded
    }
    if drift:
        logger.info(
            "Snapshot %s was written under different library versions: %s. The content "
            "hashes still match, so the bars themselves are identical.",
            manifest.snapshot_id, drift,
        )


def verify_snapshot(path: Path | str) -> Dict[str, Any]:
    """
    Non-raising integrity report for a snapshot directory.

    The same checks ``read_snapshot(verify=True)`` performs, but the findings
    are returned rather than raised, so a report generator can print the state
    of every ticker instead of dying on the first bad one.

    Returns
    -------
    dict
        ``ok`` (bool: nothing missing, nothing extra, every hash matching, and
        the manifest hash agreeing with its own entries), ``snapshot_id``,
        ``storage_format``, ``manifest_sha256`` and ``manifest_sha256_recomputed``,
        ``manifest_hash_ok``, ``tickers`` (one row per entry with ``expected``,
        ``actual`` and ``ok``; ``actual`` is None when the file is missing, since
        an absent file has no hash rather than a zero one), ``missing_files``,
        ``extra_files``, and ``error`` -- non-None only when the manifest itself
        could not be read, in which case nothing else could be checked.
    """
    root = Path(path)
    report: Dict[str, Any] = {
        "path": str(root),
        "ok": False,
        "snapshot_id": None,
        "storage_format": None,
        "manifest_sha256": None,
        "manifest_sha256_recomputed": None,
        "manifest_hash_ok": False,
        "tickers": [],
        "missing_files": [],
        "extra_files": [],
        "error": None,
    }

    try:
        manifest = _load_manifest(root)
    except Exception as exc:  # noqa: BLE001 - the report must survive an unreadable manifest
        logger.error("Snapshot manifest at %s could not be read: %s", root, exc)
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report

    frames, mismatched, missing = _rehash_entries(root, manifest)
    del frames  # this is a report about hashes; do not hold the data in it

    recomputed = manifest_content_hash(manifest.entries)
    mismatch_map = {ticker: actual for ticker, _, actual in mismatched}
    missing_set = set(missing)

    rows: List[Dict[str, Any]] = []
    for entry in manifest.entries:
        if entry.ticker in missing_set:
            actual: Optional[str] = None
        else:
            actual = mismatch_map.get(entry.ticker, entry.content_sha256)
        rows.append(
            {
                "ticker": entry.ticker,
                "expected": entry.content_sha256,
                "actual": actual,
                "ok": entry.ticker not in missing_set and entry.ticker not in mismatch_map,
            }
        )

    bars_dir = root / BARS_DIRNAME
    expected_files = {
        _bar_filename(entry.ticker, manifest.storage_format) for entry in manifest.entries
    }
    extra_files = (
        sorted(
            item.name
            for item in bars_dir.iterdir()
            if item.is_file() and item.name not in expected_files
        )
        if bars_dir.is_dir()
        else []
    )

    report.update(
        {
            "snapshot_id": manifest.snapshot_id,
            "storage_format": manifest.storage_format,
            "manifest_sha256": manifest.manifest_sha256,
            "manifest_sha256_recomputed": recomputed,
            "manifest_hash_ok": recomputed == manifest.manifest_sha256,
            "tickers": rows,
            "missing_files": sorted(missing_set),
            "extra_files": extra_files,
        }
    )
    report["ok"] = bool(
        report["manifest_hash_ok"]
        and manifest.entries
        and not missing_set
        and not mismatch_map
        and not extra_files
    )
    return report


def snapshot_caption(manifest: SnapshotManifest | Mapping[str, Any]) -> str:
    """
    The one-line provenance caption A10.2 requires under every results table.

    Carries the snapshot id, the date range it covers, the ticker count, the
    manifest hash prefix, and -- when the universe was not point-in-time -- the
    A4.5 survivorship limitation in full. The limitation is omitted only when
    ``point_in_time`` is True, i.e. when there is genuinely none to state; a
    manifest carrying no survivorship record at all is captioned "not recorded"
    rather than left looking clean.
    """
    if isinstance(manifest, Mapping):
        manifest = SnapshotManifest.from_dict(manifest)

    count = len(manifest.entries)
    span = f"{manifest.start}..{manifest.end}" if count else "no data"
    parts = [
        f"Snapshot {manifest.snapshot_id}",
        span,
        f"{count} ticker{'' if count == 1 else 's'}",
        f"manifest sha256 {manifest.manifest_sha256[:12]}",
        f"frozen {manifest.created_at}",
        f"source {manifest.source}" if manifest.source else "",
    ]
    caption = " | ".join(part for part in parts if part)

    survivorship = manifest.survivorship or {}
    point_in_time = survivorship.get("point_in_time")
    if point_in_time is False:
        limitation = str(survivorship.get("limitation") or "").strip() or SURVIVORSHIP_LIMITATION
        caption += f" | SURVIVORSHIP: {limitation}"
    elif point_in_time is None:
        caption += " | SURVIVORSHIP: not recorded"
    return caption
