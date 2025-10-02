"""Stochastic hyper-parameter search strategies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from ..core.events import CandidateEvaluated, EventBus
from .base import OptimizationCandidate, OptimizationContext, OptimizationResult, Optimizer

__all__ = ["RandomSearchOptimizer", "RandomSearchSettings"]


@dataclass(slots=True)
class RandomSearchSettings:
    """Configuration for :class:`RandomSearchOptimizer`."""

    max_trials: int = 20
    random_seed: int | None = 42
    shuffle_space: bool = True


class RandomSearchOptimizer(Optimizer):
    """Evaluate random samples from a finite search space."""

    def __init__(self, *, event_bus: EventBus | None = None, settings: RandomSearchSettings | None = None) -> None:
        self._bus = event_bus or EventBus()
        self._settings = settings or RandomSearchSettings()

    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        search_space = list(context.model_search_space)
        if not search_space:
            search_space = [{}]

        if self._settings.shuffle_space:
            rng = random.Random(self._settings.random_seed)
            rng.shuffle(search_space)

        max_trials = min(self._settings.max_trials, len(search_space))
        candidates: list[OptimizationCandidate] = []
        pipeline_builder = context.pipeline_builder

        splitter = StratifiedKFold(n_splits=context.cv_folds, shuffle=True, random_state=self._settings.random_seed)

        for idx, params in enumerate(search_space[:max_trials]):
            pipeline = pipeline_builder(params)
            scores = cross_val_score(
                pipeline,
                context.dataset.features,
                context.dataset.target,
                scoring=context.scoring,
                cv=splitter,
            )
            score = float(np.mean(scores))
            candidate = OptimizationCandidate(params=params, score=score)
            candidates.append(candidate)
            self._bus.publish(
                CandidateEvaluated(
                    run_id=context.run_id,
                    candidate_index=idx,
                    params=dict(params),
                    score=score,
                )
            )

        best = max(candidates, key=lambda c: c.score)
        return OptimizationResult(best_params=best.params, best_score=best.score, candidates=candidates)

