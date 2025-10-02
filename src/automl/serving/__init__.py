"""Serving module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .api import (
        ModelServer,
        PredictionRequest,
        PredictionResponse,
        BatchPredictionRequest,
        create_app,
        run_server,
    )
    __all__.extend([
        "ModelServer",
        "PredictionRequest",
        "PredictionResponse",
        "BatchPredictionRequest",
        "create_app",
        "run_server",
    ])
except ImportError:
    pass
