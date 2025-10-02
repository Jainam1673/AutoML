"""Dataset primitives for AutoML workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

__all__ = ["DatasetBundle", "DatasetProvider"]


@dataclass(slots=True)
class DatasetBundle:
    """Container for feature matrix, target vector, and metadata."""

    features: np.ndarray
    target: np.ndarray
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.target.shape[0]:
            msg = (
                "Features and target must share the same number of rows. "
                f"Got {self.features.shape[0]} and {self.target.shape[0]}."
            )
            raise ValueError(msg)


class DatasetProvider(Protocol):
    """Callable responsible for producing a :class:`DatasetBundle`."""

    def __call__(self) -> DatasetBundle:
        ...
