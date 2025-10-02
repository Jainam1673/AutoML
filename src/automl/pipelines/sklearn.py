"""Convenience factories for scikit-learn preprocessing components."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

__all__ = ["standard_scaler", "pca"]


def standard_scaler(params: Mapping[str, object] | None = None) -> StandardScaler:
    configuration: dict[str, Any] = {
        "with_mean": True,
        "with_std": True,
        "copy": True,
    }
    if params:
        configuration.update(params)
    return StandardScaler(**configuration)


def pca(params: Mapping[str, object] | None = None) -> PCA:
    configuration: dict[str, Any] = {
        "n_components": None,
        "whiten": False,
        "svd_solver": "auto",
        "random_state": 42,
    }
    if params:
        configuration.update(params)
    return PCA(**configuration)
