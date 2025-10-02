"""Built-in dataset providers."""

from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Any, Mapping, cast

import numpy as np

from .base import DatasetBundle

__all__ = ["iris_dataset"]


@lru_cache(maxsize=None)
def iris_dataset() -> DatasetBundle:
    """Return the classic Iris dataset as a :class:`DatasetBundle`."""

    datasets = importlib.import_module("sklearn.datasets")
    iris = cast(Any, datasets.load_iris())
    features = np.asarray(iris.data, dtype=float)
    target = np.asarray(iris.target, dtype=int)
    metadata: Mapping[str, Any] = {
        "feature_names": tuple(getattr(iris, "feature_names", ())),
        "target_names": tuple(getattr(iris, "target_names", ())),
        "description": getattr(iris, "DESCR", None),
    }
    return DatasetBundle(features=features, target=target, metadata=metadata)
