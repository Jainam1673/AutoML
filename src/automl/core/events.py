"""Lightweight synchronous event bus for experiment instrumentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import DefaultDict, Dict, Iterable, List, Protocol, Sequence, Type

__all__ = [
    "Event",
    "EventListener",
    "EventBus",
    "RunStarted",
    "RunCompleted",
    "CandidateEvaluated",
]


@dataclass(frozen=True, slots=True)
class Event:
    """Base event carrying a UTC timestamp."""

    created_at: datetime = field(init=False)

    def __post_init__(self) -> None:  # pragma: no cover - trivial
        object.__setattr__(self, "created_at", datetime.now(timezone.utc))


class EventListener(Protocol):
    """Callable protocol for event listeners."""

    def __call__(self, event: "Event") -> None:  # pragma: no cover - protocol
        ...


class EventBus:
    """In-process pub/sub hub for :class:`Event` instances."""

    def __init__(self) -> None:
        self._listeners: DefaultDict[Type[Event], List[EventListener]] = DefaultDict(list)

    def subscribe(self, event_type: Type[Event], listener: EventListener) -> None:
        """Register ``listener`` to receive ``event_type`` notifications."""

        self._listeners[event_type].append(listener)

    def unsubscribe(self, event_type: Type[Event], listener: EventListener) -> None:
        """Remove ``listener`` if previously subscribed."""

        listeners = self._listeners.get(event_type)
        if not listeners:  # pragma: no cover - defensive branch
            return
        try:
            listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive branch
            pass

    def publish(self, event: Event) -> None:
        """Send ``event`` to listeners registered for its concrete type."""

        for listener in list(self._listeners.get(type(event), ())):
            listener(event)

    def listeners(self, event_type: Type[Event]) -> Sequence[EventListener]:
        """Return subscribed listeners for inspection/testing."""

        return tuple(self._listeners.get(event_type, ()))

    def drain(self) -> Dict[Type[Event], Iterable[EventListener]]:  # pragma: no cover - debugging helper
        return {etype: tuple(listeners) for etype, listeners in self._listeners.items()}


@dataclass(frozen=True, slots=True)
class RunStarted(Event):
    run_id: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class RunCompleted(Event):
    run_id: str
    best_score: float
    best_pipeline: Dict[str, object]
    candidate_count: int


@dataclass(frozen=True, slots=True)
class CandidateEvaluated(Event):
    run_id: str
    candidate_index: int
    params: Dict[str, object]
    score: float