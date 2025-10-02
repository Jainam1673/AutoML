"""Monitoring module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .metrics import (
        MetricsCollector,
        DriftDetector,
        PerformanceMonitor,
    )
    __all__.extend([
        "MetricsCollector",
        "DriftDetector",
        "PerformanceMonitor",
    ])
except ImportError:
    pass
