"""Dashboard module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .app import AutoMLDashboard, run_dashboard
    __all__.extend(["AutoMLDashboard", "run_dashboard"])
except ImportError:
    pass
