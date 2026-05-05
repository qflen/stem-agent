"""JSON file storage adapter; concrete implementation of the Storage port."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonStorageAdapter:
    """Persists data as JSON files in a directory.

    Satisfies StoragePort via structural subtyping.
    """

    def __init__(self, base_dir: str) -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        """Convert a key to a file path, sanitizing for filesystem safety."""
        safe_key = key.replace("/", "_").replace("\\", "_")
        return self._base_dir / f"{safe_key}.json"

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist data as a JSON file."""
        path = self._key_to_path(key)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def load(self, key: str) -> dict[str, Any] | None:
        """Load data from a JSON file, or return None if it doesn't exist."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys (filenames without .json extension) matching a prefix."""
        keys = []
        for path in sorted(self._base_dir.glob("*.json")):
            key = path.stem
            if key.startswith(prefix):
                keys.append(key)
        return keys
