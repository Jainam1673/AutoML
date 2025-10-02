"""Pipeline graph abstractions and execution plans.

Provides pipeline builders and preprocessing:
- sklearn: Standard scikit-learn preprocessing
- advanced: Advanced feature engineering and transformations
"""

from __future__ import annotations

from . import advanced, sklearn
from .base import PipelineFactory, PipelineArtifact
from .advanced import AutoFeatureEngineer, TimeSeriesFeatureEngineer

# Convenience aliases
SklearnPipeline = PipelineFactory
AdvancedPipeline = AutoFeatureEngineer

__all__ = [
    "sklearn",
    "advanced",
    "PipelineFactory",
    "PipelineArtifact",
    "SklearnPipeline",
    "AdvancedPipeline",
    "AutoFeatureEngineer",
    "TimeSeriesFeatureEngineer",
]
