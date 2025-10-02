"""Thread-safe component registry utilities."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, Iterator, MutableMapping, Optional, TypeVar

__all__ = ["Registry", "DuplicateComponentError", "UnknownComponentError"]

T = TypeVar("T")


class DuplicateComponentError(ValueError):
    """Raised when attempting to register a component that already exists."""


class UnknownComponentError(KeyError):
    """Raised when a requested component does not exist."""


@dataclass(slots=True)
class _Entry(Generic[T]):
    name: str
    factory: T
    description: Optional[str]


class Registry(Generic[T]):
    """Concurrent-safe registry for named components.

    Parameters
    ----------
    kind:
        Human-friendly label describing the component type, used for error messages.
    allow_replace:
        If ``True`` existing entries may be replaced via :meth:`register`.
    """

    def __init__(self, kind: str, *, allow_replace: bool = False) -> None:
        self._kind = kind
        self._allow_replace = allow_replace
        self._entries: MutableMapping[str, _Entry[T]] = {}
        self._lock = threading.RLock()

    def register(self, name: str, factory: T, *, description: str | None = None) -> None:
        """Register a factory under ``name``.

        Raises
        ------
        DuplicateComponentError
            If ``name`` already exists and replacements are disallowed.
        """

        key = name.lower().strip()
        if not key:
            msg = f"{self._kind} name must not be empty"
            raise ValueError(msg)

        with self._lock:
            if key in self._entries and not self._allow_replace:
                msg = f"{self._kind.capitalize()} '{name}' already registered"
                raise DuplicateComponentError(msg)
            self._entries[key] = _Entry(name=key, factory=factory, description=description)

    def get(self, name: str) -> T:
        """Return the factory registered under ``name``.

        Raises
        ------
        UnknownComponentError
            If ``name`` has not been registered.
        """

        key = name.lower().strip()
        with self._lock:
            try:
                return self._entries[key].factory
            except KeyError as exc:  # pragma: no cover - defensive branch
                msg = f"Unknown {self._kind} '{name}'"
                raise UnknownComponentError(msg) from exc

    def require(self, name: str) -> T:
        """Alias for :meth:`get` for fluency."""

        return self.get(name)

    def items(self) -> Iterable[tuple[str, T]]:
        """Iterate over registered ``(name, factory)`` tuples."""

        with self._lock:
            snapshot = list(self._entries.values())
        return ((entry.name, entry.factory) for entry in snapshot)

    def __contains__(self, name: object) -> bool:  # pragma: no cover - trivial
        if not isinstance(name, str):  # pragma: no cover - safety
            return False
        key = name.lower().strip()
        with self._lock:
            return key in self._entries

    def __iter__(self) -> Iterator[str]:  # pragma: no cover - trivial
        with self._lock:
            return iter(list(self._entries.keys()))

    def describe(self) -> Dict[str, Optional[str]]:
        """Return mapping of component names to optional descriptions."""

        with self._lock:
            return {entry.name: entry.description for entry in self._entries.values()}

    def copy(self) -> "Registry[T]":
        """Return a shallow copy preserving configuration."""

        clone = Registry(self._kind, allow_replace=self._allow_replace)
        with self._lock:
            clone._entries = dict(self._entries)
        return clone


Factory = TypeVar("Factory", bound=Callable[..., object])

__all__.append("Factory")
