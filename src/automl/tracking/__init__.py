"""Tracking module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .mlflow_integration import (
        MLflowTracker,
        ExperimentTracker,
        ModelRegistry,
        MLflowConfig,
    )
    # Alias for convenience
    MLflowIntegration = MLflowTracker
    
    __all__.extend([
        "MLflowTracker",
        "MLflowIntegration",
        "ExperimentTracker",
        "ModelRegistry",
        "MLflowConfig",
    ])
except ImportError:
    pass
