"""Dataset loaders, feature stores, and synthetic data generators."""

from __future__ import annotations

from .base import DatasetBundle, DatasetProvider
from .builtin import iris_dataset

# Convenience aliases
class BuiltinDatasets:
    """Built-in dataset collection."""
    iris = staticmethod(iris_dataset)

class DatasetLoader:
    """Dataset loader utility."""
    @staticmethod
    def load(name: str) -> DatasetBundle:
        """Load a built-in dataset by name."""
        if name == "iris":
            return iris_dataset()
        raise ValueError(f"Unknown dataset: {name}")

__all__ = [
    "DatasetBundle",
    "DatasetProvider", 
    "BuiltinDatasets",
    "DatasetLoader",
    "iris_dataset",
]
