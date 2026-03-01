"""
Model Version Registry.
Tracks model versions with metadata, supports listing, comparing, and loading.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from ..utils.logger import get_logger

logger = get_logger(__name__)

REGISTRY_DIR = Path("models/model_metadata")
MODELS_DIR = Path("models/saved_models")


class ModelRegistry:
    """Manages model versions and metadata."""

    def __init__(self):
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def register(
        self,
        model_type: str,
        symbol: str,
        metrics: Dict,
        params: Dict,
        horizons: List[int],
        dataset_hash: Optional[str] = None,
        feature_columns: Optional[List[str]] = None,
    ) -> str:
        """
        Register a trained model version.

        Returns
        -------
        str
            Version ID
        """
        version_id = f"{model_type}_{symbol}_{datetime.now():%Y%m%d_%H%M%S}"
        meta = {
            "version_id": version_id,
            "model_type": model_type,
            "symbol": symbol,
            "trained_at": datetime.now().isoformat(),
            "horizons": horizons,
            "metrics": metrics,
            "params": params,
            "dataset_hash": dataset_hash or "",
            "feature_columns": feature_columns or [],
            "model_path": str(MODELS_DIR / model_type),
        }
        path = REGISTRY_DIR / f"{version_id}.json"
        with open(path, "w") as f:
            json.dump(meta, f, indent=2, default=str)
        logger.info(f"Registered model version: {version_id}")
        return version_id

    def list_versions(
        self, model_type: Optional[str] = None, symbol: Optional[str] = None
    ) -> List[Dict]:
        """List registered model versions, newest first."""
        versions = []
        for f in REGISTRY_DIR.glob("*.json"):
            if f.name == "feature_columns.json":
                continue
            try:
                with open(f) as fh:
                    meta = json.load(fh)
                if model_type and meta.get("model_type") != model_type:
                    continue
                if symbol and meta.get("symbol") != symbol:
                    continue
                versions.append(meta)
            except Exception:
                continue
        versions.sort(key=lambda v: v.get("trained_at", ""), reverse=True)
        return versions

    def get_version(self, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific version."""
        path = REGISTRY_DIR / f"{version_id}.json"
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def get_best_version(
        self, model_type: str, symbol: str, metric: str = "rmse"
    ) -> Optional[Dict]:
        """Get the best version by a given metric (lower is better)."""
        versions = self.list_versions(model_type=model_type, symbol=symbol)
        if not versions:
            return None
        return min(
            versions,
            key=lambda v: v.get("metrics", {}).get(metric, float("inf")),
        )

    def compare_versions(self, version_ids: List[str]) -> List[Dict]:
        """Compare multiple versions side by side."""
        results = []
        for vid in version_ids:
            meta = self.get_version(vid)
            if meta:
                results.append({
                    "version_id": vid,
                    "model_type": meta["model_type"],
                    "symbol": meta["symbol"],
                    "trained_at": meta["trained_at"],
                    "metrics": meta.get("metrics", {}),
                })
        return results

    @staticmethod
    def compute_dataset_hash(data) -> str:
        """Compute hash of a dataset for reproducibility."""
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            content = data.to_csv().encode()
        elif isinstance(data, bytes):
            content = data
        else:
            content = str(data).encode()
        return hashlib.sha256(content).hexdigest()[:16]
