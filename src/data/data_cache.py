"""
Caching layer for external data API calls.
Supports in-memory (TTLCache) and disk-based (JSON) caching.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
from cachetools import TTLCache

from ..utils.logger import get_logger

logger = get_logger(__name__)


class DataCache:
    """
    Two-tier cache: fast in-memory TTL cache backed by disk persistence.
    """

    def __init__(
        self,
        memory_maxsize: int = 500,
        memory_ttl: int = 300,
        disk_dir: str = "data/cache",
    ):
        self._mem = TTLCache(maxsize=memory_maxsize, ttl=memory_ttl)
        self._disk = Path(disk_dir)
        self._disk.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key_hash(key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        if key in self._mem:
            return self._mem[key]
        disk_file = self._disk / f"{self._key_hash(key)}.json"
        if disk_file.exists():
            try:
                with open(disk_file) as f:
                    entry = json.load(f)
                # Check TTL on disk (24h default)
                ts = datetime.fromisoformat(entry["timestamp"])
                if (datetime.now() - ts).total_seconds() < entry.get("ttl", 86400):
                    self._mem[key] = entry["data"]
                    return entry["data"]
                else:
                    disk_file.unlink(missing_ok=True)
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any, disk_ttl: int = 86400):
        """Set value in both memory and disk cache."""
        self._mem[key] = value
        disk_file = self._disk / f"{self._key_hash(key)}.json"
        try:
            with open(disk_file, "w") as f:
                json.dump({
                    "key": key,
                    "data": value,
                    "timestamp": datetime.now().isoformat(),
                    "ttl": disk_ttl,
                }, f, default=str)
        except Exception as e:
            logger.warning(f"Disk cache write failed: {e}")

    def invalidate(self, key: str):
        """Remove entry from both caches."""
        self._mem.pop(key, None)
        disk_file = self._disk / f"{self._key_hash(key)}.json"
        disk_file.unlink(missing_ok=True)

    def clear(self):
        """Clear all caches."""
        self._mem.clear()
        for f in self._disk.glob("*.json"):
            f.unlink(missing_ok=True)
        logger.info("Data cache cleared")


# Global singleton
_cache_instance: Optional[DataCache] = None


def get_data_cache() -> DataCache:
    """Get the global DataCache singleton."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = DataCache()
    return _cache_instance
