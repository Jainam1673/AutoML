"""Caching module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .redis_cache import (
        RedisCache,
        FeatureCache,
        PredictionCache,
    )
    __all__.extend([
        "RedisCache",
        "FeatureCache",
        "PredictionCache",
    ])
except ImportError:
    pass
