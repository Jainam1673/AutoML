"""State-of-the-art Optuna-based hyperparameter optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, PercentilePruner, SuccessiveHalvingPruner
from optuna.samplers import (
    CmaEsSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)
from sklearn.model_selection import StratifiedKFold

from ..core.events import CandidateEvaluated, EventBus
from .base import OptimizationCandidate, OptimizationContext, OptimizationResult, Optimizer

__all__ = [
    "OptunaOptimizer",
    "OptunaSettings",
    "MultiObjectiveOptunaOptimizer",
    "SamplerType",
    "PrunerType",
]

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class SamplerType:
    """Enumeration of available Optuna samplers."""

    TPE = "tpe"
    CMAES = "cmaes"
    RANDOM = "random"
    GRID = "grid"
    QMCS = "qmcs"  # Quasi-Monte Carlo Sampling
    NSGAII = "nsgaii"  # For multi-objective


@dataclass(slots=True)
class PrunerType:
    """Enumeration of available Optuna pruners."""

    MEDIAN = "median"
    PERCENTILE = "percentile"
    HYPERBAND = "hyperband"
    SUCCESSIVE_HALVING = "successive_halving"
    NONE = "none"


@dataclass(slots=True)
class OptunaSettings:
    """Configuration for :class:`OptunaOptimizer`."""

    n_trials: int = 100
    timeout: int | None = None  # seconds
    sampler: str = SamplerType.TPE
    pruner: str = PrunerType.HYPERBAND
    n_jobs: int = -1
    show_progress_bar: bool = True
    random_seed: int | None = 42
    
    # Advanced settings
    use_early_stopping: bool = True
    min_resource: int = 1
    max_resource: int = 10
    reduction_factor: int = 3
    
    # TPE-specific settings
    n_startup_trials: int = 10
    n_ei_candidates: int = 24
    
    # Multi-objective settings
    population_size: int | None = None
    mutation_prob: float | None = None
    crossover_prob: float | None = None


class OptunaOptimizer(Optimizer):
    """Advanced hyperparameter optimization using Optuna framework.
    
    Supports:
    - Tree-structured Parzen Estimator (TPE)
    - CMA-ES for continuous optimization
    - Quasi-Monte Carlo sampling
    - Hyperband pruning for early stopping
    - Parallel optimization with multiple workers
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        settings: OptunaSettings | None = None,
    ) -> None:
        self._bus = event_bus or EventBus()
        self._settings = settings or OptunaSettings()
        self._study: optuna.Study | None = None

    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        """Execute optimization using Optuna's advanced algorithms."""
        
        # Create sampler
        sampler = self._create_sampler()
        
        # Create pruner
        pruner = self._create_pruner()
        
        # Create study
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
            study_name=f"automl_{context.run_id}",
        )
        
        # Define objective function
        def objective(trial: optuna.Trial) -> float:
            # Sample from search space
            params = self._sample_params(trial, list(context.model_search_space))
            
            # Build pipeline
            pipeline = context.pipeline_builder(params)
            
            # Evaluate with cross-validation
            splitter = StratifiedKFold(
                n_splits=context.cv_folds,
                shuffle=True,
                random_state=self._settings.random_seed,
            )
            
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(
                splitter.split(context.dataset.features, context.dataset.target)
            ):
                X_train = context.dataset.features[train_idx]
                y_train = context.dataset.target[train_idx]
                X_val = context.dataset.features[val_idx]
                y_val = context.dataset.target[val_idx]
                
                # Fit pipeline
                pipeline.fit(X_train, y_train)
                
                # Score on validation set
                score = pipeline.score(X_val, y_val)
                scores.append(score)
                
                # Report intermediate value for pruning
                if self._settings.use_early_stopping:
                    trial.report(score, fold_idx)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            mean_score = float(np.mean(scores))
            
            # Publish event
            self._bus.publish(
                CandidateEvaluated(
                    run_id=context.run_id,
                    candidate_index=trial.number,
                    params=dict(params),
                    score=mean_score,
                )
            )
            
            return mean_score
        
        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self._settings.n_trials,
            timeout=self._settings.timeout,
            n_jobs=self._settings.n_jobs,
            show_progress_bar=self._settings.show_progress_bar,
            catch=(Exception,),
        )
        
        # Collect results
        candidates = [
            OptimizationCandidate(
                params=trial.params,
                score=trial.value if trial.value is not None else float("-inf"),
            )
            for trial in self._study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        best_trial = self._study.best_trial
        
        return OptimizationResult(
            best_params=best_trial.params,
            best_score=best_trial.value if best_trial.value is not None else 0.0,
            candidates=candidates,
        )

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on settings."""
        seed = self._settings.random_seed
        
        if self._settings.sampler == SamplerType.TPE:
            return TPESampler(
                n_startup_trials=self._settings.n_startup_trials,
                n_ei_candidates=self._settings.n_ei_candidates,
                seed=seed,
            )
        elif self._settings.sampler == SamplerType.CMAES:
            return CmaEsSampler(seed=seed)
        elif self._settings.sampler == SamplerType.RANDOM:
            return RandomSampler(seed=seed)
        elif self._settings.sampler == SamplerType.QMCS:
            return QMCSampler(seed=seed)
        else:
            return TPESampler(seed=seed)

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on settings."""
        if not self._settings.use_early_stopping or self._settings.pruner == PrunerType.NONE:
            return optuna.pruners.NopPruner()
        
        if self._settings.pruner == PrunerType.MEDIAN:
            return MedianPruner(
                n_startup_trials=self._settings.n_startup_trials,
                n_warmup_steps=2,
            )
        elif self._settings.pruner == PrunerType.PERCENTILE:
            return PercentilePruner(
                percentile=25.0,
                n_startup_trials=self._settings.n_startup_trials,
            )
        elif self._settings.pruner == PrunerType.HYPERBAND:
            return HyperbandPruner(
                min_resource=self._settings.min_resource,
                max_resource=self._settings.max_resource,
                reduction_factor=self._settings.reduction_factor,
            )
        elif self._settings.pruner == PrunerType.SUCCESSIVE_HALVING:
            return SuccessiveHalvingPruner(
                min_resource=self._settings.min_resource,
                reduction_factor=self._settings.reduction_factor,
            )
        else:
            return HyperbandPruner()

    def _sample_params(
        self,
        trial: optuna.Trial,
        search_space: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Sample parameters from search space using trial."""
        if not search_space or len(search_space) == 1:
            return dict(search_space[0]) if search_space else {}
        
        # For discrete space, use categorical
        idx = trial.suggest_categorical("config_idx", list(range(len(search_space))))
        return dict(search_space[idx])

    def get_study(self) -> optuna.Study | None:
        """Return the underlying Optuna study for analysis."""
        return self._study


@dataclass(slots=True)
class MultiObjectiveSettings(OptunaSettings):
    """Settings for multi-objective optimization."""
    
    objectives: list[str] | None = None  # e.g., ["accuracy", "f1_score"]
    reference_point: list[float] | None = None


class MultiObjectiveOptunaOptimizer(Optimizer):
    """Multi-objective hyperparameter optimization using NSGA-II.
    
    Optimizes multiple metrics simultaneously using Pareto optimization.
    """

    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        settings: MultiObjectiveSettings | None = None,
    ) -> None:
        self._bus = event_bus or EventBus()
        self._settings = settings or MultiObjectiveSettings()
        self._study: optuna.Study | None = None

    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        """Execute multi-objective optimization."""
        
        # Create NSGA-II sampler for multi-objective optimization
        sampler_kwargs = {
            "population_size": self._settings.population_size or 50,
            "seed": self._settings.random_seed,
        }
        if self._settings.mutation_prob is not None:
            sampler_kwargs["mutation_prob"] = self._settings.mutation_prob
        if self._settings.crossover_prob is not None:
            sampler_kwargs["crossover_prob"] = self._settings.crossover_prob
        
        sampler = NSGAIISampler(**sampler_kwargs)
        
        # Create study with multiple objectives
        directions = ["maximize"] * len(self._settings.objectives or ["accuracy"])
        
        self._study = optuna.create_study(
            directions=directions,
            sampler=sampler,
            study_name=f"automl_mo_{context.run_id}",
        )
        
        def objective(trial: optuna.Trial) -> tuple[float, ...]:
            # Sample params and evaluate
            params = self._sample_params(trial, list(context.model_search_space))
            pipeline = context.pipeline_builder(params)
            
            # Cross-validation
            from sklearn.model_selection import cross_validate
            
            splitter = StratifiedKFold(
                n_splits=context.cv_folds,
                shuffle=True,
                random_state=self._settings.random_seed,
            )
            
            scoring = self._settings.objectives or [context.scoring]
            cv_results = cross_validate(
                pipeline,
                context.dataset.features,
                context.dataset.target,
                cv=splitter,
                scoring=scoring,
                n_jobs=1,
            )
            
            # Return mean scores for each objective
            return tuple(
                float(np.mean(cv_results[f"test_{metric}"]))
                for metric in scoring
            )
        
        # Run optimization
        self._study.optimize(
            objective,
            n_trials=self._settings.n_trials,
            timeout=self._settings.timeout,
            n_jobs=self._settings.n_jobs,
            show_progress_bar=self._settings.show_progress_bar,
        )
        
        # Get best trial (first Pareto front member)
        best_trials = self._study.best_trials
        best_trial = best_trials[0] if best_trials else None
        
        if best_trial is None:
            raise RuntimeError("No successful trials completed")
        
        candidates = [
            OptimizationCandidate(
                params=trial.params,
                score=trial.values[0] if trial.values else float("-inf"),
            )
            for trial in self._study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        
        return OptimizationResult(
            best_params=best_trial.params,
            best_score=best_trial.values[0] if best_trial.values else 0.0,
            candidates=candidates,
        )

    def _sample_params(
        self,
        trial: optuna.Trial,
        search_space: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Sample parameters from search space."""
        if not search_space or len(search_space) == 1:
            return dict(search_space[0]) if search_space else {}
        
        idx = trial.suggest_categorical("config_idx", list(range(len(search_space))))
        return dict(search_space[idx])

    def get_pareto_front(self) -> list[optuna.trial.FrozenTrial]:
        """Return trials on the Pareto front."""
        if self._study is None:
            return []
        return self._study.best_trials
