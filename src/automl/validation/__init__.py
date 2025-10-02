"""Validation module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .data_quality import (
        DataValidator,
        DataQualityReport,
        AutoFixer,
        AnomalyDetector,
    )
    __all__.extend([
        "DataValidator",
        "DataQualityReport",
        "AutoFixer",
        "AnomalyDetector",
    ])
except ImportError:
    pass
