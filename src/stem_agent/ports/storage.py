"""Storage port — protocol for persisting agent state and journal data."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StoragePort(Protocol):
    """Structural interface for state persistence.

    Implementations may use JSON files, databases, cloud storage, etc.
    The core never knows or cares which.
    """

    def save(self, key: str, data: dict[str, Any]) -> None:
        """Persist data under a given key.

        Args:
            key: Unique identifier for this data blob (e.g., "journal_2024-01-15").
            data: JSON-serializable dictionary to persist.
        """
        ...

    def load(self, key: str) -> dict[str, Any] | None:
        """Load previously persisted data.

        Args:
            key: The key used when saving.

        Returns:
            The data dictionary, or None if the key does not exist.
        """
        ...

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all persisted keys matching an optional prefix.

        Args:
            prefix: Filter keys starting with this string.

        Returns:
            Sorted list of matching keys.
        """
        ...
