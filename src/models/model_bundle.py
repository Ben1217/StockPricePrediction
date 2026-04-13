"""
Model bundle helpers.

Canonical bundles live under ``models/bundles/<SYMBOL>/<MODEL_TYPE>/`` and
contain the full inference contract for a single symbol/model pair. Legacy
horizon-scoped bundles remain readable for backward compatibility.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib

from ..utils.logger import get_logger
from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel

logger = get_logger(__name__)

BUNDLES_DIR = Path("models/bundles")
LEGACY_MODELS_DIR = Path("models/saved_models")
LEGACY_METADATA_DIR = Path("models/model_metadata")

MODEL_FILE_NAMES = {
    "xgboost": "model.json",
    "random_forest": "model.joblib",
    "lstm": "model.pt",
}

MODEL_FACTORIES = {
    "xgboost": XGBoostModel,
    "random_forest": RandomForestModel,
    "lstm": LSTMModel,
}

CANONICAL_BUNDLE_LAYOUT = "canonical_symbol_model"
LEGACY_BUNDLE_LAYOUT = "legacy_horizon"


def _normalize_horizons(values: Optional[Iterable[int]]) -> List[int]:
    if values is None:
        return []
    return sorted({int(value) for value in values if int(value) > 0})


def _normalise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    normalised = dict(meta)
    normalised["feature_columns"] = list(normalised.get("feature_columns", []))

    if "symbol" in normalised and normalised["symbol"] is not None:
        normalised["symbol"] = str(normalised["symbol"]).upper()

    training_horizon = normalised.get("training_horizon", normalised.get("horizon", 1))
    normalised["training_horizon"] = int(training_horizon)
    normalised["horizon"] = int(training_horizon)

    horizons = normalised.get("horizons", normalised.get("supported_horizons"))
    normalized_horizons = _normalize_horizons(horizons)
    if not normalized_horizons:
        normalized_horizons = [int(training_horizon)]
    if int(training_horizon) not in normalized_horizons:
        normalized_horizons.append(int(training_horizon))
        normalized_horizons = sorted(set(normalized_horizons))
    normalised["horizons"] = normalized_horizons

    feature_count = normalised.get("feature_count", normalised.get("features"))
    if feature_count is None:
        feature_count = len(normalised["feature_columns"])
    normalised["feature_count"] = int(feature_count)

    sample_count = normalised.get("training_sample_count", normalised.get("samples"))
    if sample_count is not None:
        normalised["training_sample_count"] = int(sample_count)

    if "training_symbol" not in normalised and normalised.get("symbol"):
        normalised["training_symbol"] = normalised["symbol"]

    bundle_layout = normalised.get("bundle_layout")
    if not bundle_layout:
        bundle_dir = str(normalised.get("bundle_dir", "")).replace("/", "\\").strip("\\")
        tail = bundle_dir.split("\\")[-1] if bundle_dir else ""
        bundle_layout = LEGACY_BUNDLE_LAYOUT if tail.isdigit() else CANONICAL_BUNDLE_LAYOUT
    normalised["bundle_layout"] = bundle_layout

    return normalised


def _canonical_bundle_dir_for(symbol: str, model_type: str, bundles_dir: Path = BUNDLES_DIR) -> Path:
    return bundles_dir / symbol.upper() / model_type


def _legacy_bundle_dir_for(
    symbol: str,
    model_type: str,
    horizon: int,
    bundles_dir: Path = BUNDLES_DIR,
) -> Path:
    return bundles_dir / symbol.upper() / model_type / str(int(horizon))


def _read_metadata(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            meta = _normalise_metadata(json.load(handle))
    except Exception as exc:  # pragma: no cover - best effort metadata loading
        logger.warning("Failed to read model metadata %s: %s", path, exc)
        return None

    if "bundle_dir" not in meta:
        if path.name == "metadata.json" and path.parent.name.isdigit():
            meta["bundle_dir"] = str(path.parent)
            meta["bundle_layout"] = LEGACY_BUNDLE_LAYOUT
        elif path.name == "metadata.json":
            meta["bundle_dir"] = str(path.parent)
            meta["bundle_layout"] = CANONICAL_BUNDLE_LAYOUT
        elif "artifact_dir" in meta:
            meta["bundle_dir"] = meta["artifact_dir"]

    if "artifact_dir" not in meta and "bundle_dir" in meta:
        meta["artifact_dir"] = meta["bundle_dir"]

    return meta


def _iter_canonical_metadata_paths(bundles_dir: Path) -> List[Path]:
    if not bundles_dir.exists():
        return []
    return sorted(
        path
        for path in bundles_dir.glob("*/*/metadata.json")
        if not path.parent.name.isdigit()
    )


def _iter_legacy_bundle_metadata_paths(bundles_dir: Path) -> List[Path]:
    if not bundles_dir.exists():
        return []
    return sorted(bundles_dir.glob("*/*/*/metadata.json"))


def _iter_legacy_metadata_paths(metadata_dir: Path) -> List[Path]:
    if not metadata_dir.exists():
        return []
    return sorted(
        path
        for path in metadata_dir.glob("*.json")
        if path.name != "feature_columns.json"
    )


def _meta_matches_filters(
    meta: Dict[str, Any],
    model_type: Optional[str],
    symbol: Optional[str],
) -> bool:
    if model_type and meta.get("model_type") != model_type:
        return False
    if symbol and meta.get("symbol") != symbol.upper():
        return False
    return True


def _meta_supports_horizon(meta: Dict[str, Any], horizon: Optional[int]) -> bool:
    if horizon is None:
        return True

    requested = int(horizon)
    supported = _normalize_horizons(meta.get("horizons"))
    if supported:
        return requested in supported or int(meta.get("training_horizon", 1)) == 1
    training_horizon = int(meta.get("training_horizon", meta.get("horizon", 1)))
    return training_horizon == 1 or training_horizon == requested


def _metadata_rank(meta: Dict[str, Any], requested_horizon: Optional[int]) -> tuple:
    training_horizon = int(meta.get("training_horizon", meta.get("horizon", 1)))
    exact_match = 1 if requested_horizon is not None and training_horizon == int(requested_horizon) else 0
    recursive_one_step = 1 if training_horizon == 1 else 0
    trained_at = str(meta.get("trained_at", ""))
    return exact_match, recursive_one_step, trained_at


@dataclass
class LoadedModelBundle:
    """Resolved model artifact bundle used for inference."""

    version_id: str
    model_type: str
    symbol: str
    horizon: int
    model: Any
    feature_columns: List[str]
    scaler: Any
    metadata: Dict[str, Any]
    artifact_dir: Path
    model_path: Path
    scaler_path: Optional[Path]

    @property
    def feature_config(self) -> Dict[str, Any]:
        return dict(self.metadata.get("feature_config", {}))

    @property
    def target_type(self) -> str:
        return str(self.metadata.get("target_type", "direction"))

    @property
    def objective(self) -> str:
        return str(self.metadata.get("objective", "next_day_direction"))

    @property
    def scaler_type(self) -> Optional[str]:
        preprocessing = self.metadata.get("preprocessing", {})
        return preprocessing.get("scaler_type")

    @property
    def sequence_length(self) -> int:
        preprocessing = self.metadata.get("preprocessing", {})
        return int(preprocessing.get("sequence_length", 60))

    @property
    def supported_horizons(self) -> List[int]:
        horizons = _normalize_horizons(self.metadata.get("horizons"))
        return horizons or [self.horizon]

    @property
    def bundle_layout(self) -> str:
        return str(self.metadata.get("bundle_layout", CANONICAL_BUNDLE_LAYOUT))


def list_model_metadata(
    model_type: Optional[str] = None,
    symbol: Optional[str] = None,
    horizon: Optional[int] = None,
    bundles_dir: Path = BUNDLES_DIR,
    metadata_dir: Path = LEGACY_METADATA_DIR,
) -> List[Dict[str, Any]]:
    """List saved model bundles, newest first."""
    canonical_items: Dict[tuple, Dict[str, Any]] = {}
    legacy_items: Dict[tuple, Dict[str, Any]] = {}

    for path in _iter_canonical_metadata_paths(bundles_dir):
        meta = _read_metadata(path)
        if not meta:
            continue
        if not _meta_matches_filters(meta, model_type, symbol):
            continue
        if not _meta_supports_horizon(meta, horizon):
            continue
        key = (meta.get("symbol"), meta.get("model_type"))
        current = canonical_items.get(key)
        if current is None or str(meta.get("trained_at", "")) >= str(current.get("trained_at", "")):
            canonical_items[key] = meta

    for path in _iter_legacy_bundle_metadata_paths(bundles_dir) + _iter_legacy_metadata_paths(metadata_dir):
        meta = _read_metadata(path)
        if not meta:
            continue
        if not _meta_matches_filters(meta, model_type, symbol):
            continue
        if not _meta_supports_horizon(meta, horizon):
            continue
        key = (meta.get("symbol"), meta.get("model_type"))
        if key in canonical_items:
            continue
        current = legacy_items.get(key)
        if current is None or _metadata_rank(meta, horizon) > _metadata_rank(current, horizon):
            legacy_items[key] = meta

    items = list(canonical_items.values()) + list(legacy_items.values())
    items.sort(key=lambda item: item.get("trained_at", ""), reverse=True)
    return items


def select_model_metadata(
    model_type: str,
    symbol: Optional[str] = None,
    horizon: Optional[int] = None,
    bundles_dir: Path = BUNDLES_DIR,
    metadata_dir: Path = LEGACY_METADATA_DIR,
) -> Optional[Dict[str, Any]]:
    """Select the newest matching model bundle metadata entry."""
    if symbol:
        canonical_path = _canonical_bundle_dir_for(symbol, model_type, bundles_dir) / "metadata.json"
        if canonical_path.exists():
            canonical_meta = _read_metadata(canonical_path)
            if canonical_meta and _meta_supports_horizon(canonical_meta, horizon):
                return canonical_meta

        if horizon is not None:
            legacy_exact_path = _legacy_bundle_dir_for(symbol, model_type, horizon, bundles_dir) / "metadata.json"
            if legacy_exact_path.exists():
                legacy_exact_meta = _read_metadata(legacy_exact_path)
                if legacy_exact_meta and _meta_supports_horizon(legacy_exact_meta, horizon):
                    return legacy_exact_meta

        legacy_one_step_path = _legacy_bundle_dir_for(symbol, model_type, 1, bundles_dir) / "metadata.json"
        if legacy_one_step_path.exists():
            legacy_one_step_meta = _read_metadata(legacy_one_step_path)
            if legacy_one_step_meta and _meta_supports_horizon(legacy_one_step_meta, horizon):
                return legacy_one_step_meta

    items = list_model_metadata(
        model_type=model_type,
        symbol=symbol,
        horizon=horizon,
        bundles_dir=bundles_dir,
        metadata_dir=metadata_dir,
    )
    return items[0] if items else None


def get_model_metadata(
    version_id: str,
    bundles_dir: Path = BUNDLES_DIR,
    metadata_dir: Path = LEGACY_METADATA_DIR,
) -> Optional[Dict[str, Any]]:
    for path in _iter_canonical_metadata_paths(bundles_dir) + _iter_legacy_bundle_metadata_paths(bundles_dir):
        meta = _read_metadata(path)
        if meta and meta.get("version_id") == version_id:
            return meta

    legacy_path = metadata_dir / f"{version_id}.json"
    if legacy_path.exists():
        return _read_metadata(legacy_path)
    return None


def save_model_bundle(
    *,
    model,
    model_type: str,
    symbol: str,
    horizon: int,
    feature_columns: List[str],
    scaler,
    metadata: Dict[str, Any],
    models_dir: Path = BUNDLES_DIR,
    metadata_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Persist a trained model together with its preprocessing artifacts."""
    symbol = symbol.upper()
    training_horizon = int(metadata.get("training_horizon", horizon or 1))
    supported_horizons = _normalize_horizons(
        metadata.get("horizons", metadata.get("supported_horizons", [training_horizon]))
    )
    if training_horizon not in supported_horizons:
        supported_horizons.append(training_horizon)
        supported_horizons = sorted(set(supported_horizons))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = metadata.get("version_id") or f"{model_type}_{symbol}_h{training_horizon}_{timestamp}"

    bundle_dir = _canonical_bundle_dir_for(symbol, model_type, bundles_dir=models_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    model_filename = MODEL_FILE_NAMES.get(model_type, "model.joblib")
    model_path = bundle_dir / model_filename
    scaler_path = bundle_dir / "scaler.joblib"
    feature_columns_path = bundle_dir / "feature_columns.json"
    metadata_path = bundle_dir / "metadata.json"

    model.save(str(model_path))
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    else:
        scaler_path = None

    with open(feature_columns_path, "w", encoding="utf-8") as handle:
        json.dump(list(feature_columns), handle, indent=2)

    trained_at = metadata.get("trained_at") or datetime.now().isoformat()
    bundle_meta = _normalise_metadata(
        {
            **metadata,
            "version_id": version_id,
            "model_type": model_type,
            "symbol": symbol,
            "training_symbol": symbol,
            "training_horizon": training_horizon,
            "horizon": training_horizon,
            "horizons": supported_horizons,
            "trained_at": trained_at,
            "feature_columns": list(feature_columns),
            "feature_count": metadata.get("feature_count", len(feature_columns)),
            "training_sample_count": metadata.get("training_sample_count", metadata.get("samples")),
            "bundle_layout": CANONICAL_BUNDLE_LAYOUT,
            "bundle_dir": str(bundle_dir),
            "artifact_dir": str(bundle_dir),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path) if scaler_path else None,
            "feature_columns_path": str(feature_columns_path),
        }
    )

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(bundle_meta, handle, indent=2, default=str)

    if metadata_dir is not None:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_dir / f"{version_id}.json", "w", encoding="utf-8") as handle:
            json.dump(bundle_meta, handle, indent=2, default=str)

    logger.info(
        "Saved %s model bundle for %s at %s",
        model_type,
        symbol,
        bundle_dir,
    )
    return bundle_meta


def load_model_bundle(
    *,
    metadata: Optional[Dict[str, Any]] = None,
    model_type: Optional[str] = None,
    symbol: Optional[str] = None,
    horizon: Optional[int] = None,
    bundles_dir: Path = BUNDLES_DIR,
    metadata_dir: Path = LEGACY_METADATA_DIR,
) -> Optional[LoadedModelBundle]:
    """Load a saved model bundle."""
    meta = metadata or select_model_metadata(
        model_type=str(model_type),
        symbol=symbol,
        horizon=horizon,
        bundles_dir=bundles_dir,
        metadata_dir=metadata_dir,
    )
    if not meta:
        return None

    resolved_model_type = str(meta["model_type"])
    factory = MODEL_FACTORIES.get(resolved_model_type)
    if factory is None:
        raise ValueError(f"Unsupported model type in metadata: {resolved_model_type}")

    artifact_dir = Path(meta.get("bundle_dir") or meta.get("artifact_dir") or "")
    model_path = Path(meta["model_path"])
    if not model_path.exists() and artifact_dir:
        fallback_model_path = artifact_dir / MODEL_FILE_NAMES.get(resolved_model_type, "model.joblib")
        if fallback_model_path.exists():
            model_path = fallback_model_path

    scaler_path_value = meta.get("scaler_path")
    scaler_path = Path(scaler_path_value) if scaler_path_value else None
    if scaler_path and not scaler_path.exists() and artifact_dir:
        fallback_scaler_path = artifact_dir / "scaler.joblib"
        if fallback_scaler_path.exists():
            scaler_path = fallback_scaler_path

    feature_columns = list(meta.get("feature_columns", []))
    feature_columns_path_value = meta.get("feature_columns_path")
    if not feature_columns and feature_columns_path_value:
        feature_columns_path = Path(feature_columns_path_value)
        if not feature_columns_path.exists() and artifact_dir:
            feature_columns_path = artifact_dir / "feature_columns.json"
        if feature_columns_path.exists():
            with open(feature_columns_path, encoding="utf-8") as handle:
                feature_columns = list(json.load(handle))

    model = factory()
    model.load(str(model_path))
    scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    return LoadedModelBundle(
        version_id=str(meta["version_id"]),
        model_type=resolved_model_type,
        symbol=str(meta["symbol"]),
        horizon=int(meta.get("training_horizon", meta.get("horizon", 1))),
        model=model,
        feature_columns=feature_columns,
        scaler=scaler,
        metadata=meta,
        artifact_dir=artifact_dir,
        model_path=model_path,
        scaler_path=scaler_path,
    )
