"""Monitoring module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .metrics import (
        MetricsCollector,
        DriftDetector,
        PerformanceMonitor,
    )
    # Alias for convenience
    ModelMonitor = PerformanceMonitor
    
    __all__.extend([
        "MetricsCollector",
        "DriftDetector",
        "PerformanceMonitor",
        "ModelMonitor",
    ])
except ImportError:
    pass
