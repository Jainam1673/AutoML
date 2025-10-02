"""Optimization strategy protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from ..datasets.base import DatasetBundle

__all__ = [
    "OptimizationCandidate",
    "OptimizationResult",
    "OptimizationContext",
    "Optimizer",
]


@dataclass(slots=True)
class OptimizationCandidate:
    """Representation of a single hyper-parameter configuration."""

    params: Mapping[str, object]
    score: float


@dataclass(slots=True)
class OptimizationResult:
    """Outcome produced by an optimizer run."""

    best_params: Mapping[str, object]
    best_score: float
    candidates: Sequence[OptimizationCandidate]


@dataclass(slots=True)
class OptimizationContext:
    """Lightweight container passed to optimizers."""

    run_id: str
    dataset: DatasetBundle
    pipeline_builder: "PipelineBuilder"
    model_search_space: Iterable[Mapping[str, object]]
    scoring: str
    cv_folds: int


class PipelineBuilder(Protocol):
    """Factory responsible for constructing ready-to-fit pipelines."""

    def __call__(self, params: Mapping[str, object]) -> Any:
        ...


class Optimizer(Protocol):
    """Interface implemented by optimization strategies."""

    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        ...