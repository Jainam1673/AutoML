"""Neural Architecture Search module initialization."""

from __future__ import annotations

__all__: list[str] = []

try:
    from .auto_keras import AutoKerasClassifier, AutoKerasRegressor
    __all__.extend(["AutoKerasClassifier", "AutoKerasRegressor"])
except ImportError:
    pass

try:
    from .keras_tuner import KerasTunerNAS
    __all__.extend(["KerasTunerNAS"])
except ImportError:
    pass

try:
    from .custom_nas import CustomNAS, SearchSpace, PerformancePredictor
    __all__.extend(["CustomNAS", "SearchSpace", "PerformancePredictor"])
except ImportError:
    pass
