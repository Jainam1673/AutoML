"""Distributed computing integration with Ray for exabyte-scale processing.

This module provides Ray-based distributed hyperparameter optimization,
distributed data processing, and scalable model training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

try:
    import ray
    from ray import tune
    from ray.tune import Trainable
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.search.bayesopt import BayesOptSearch
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from ..core.events import CandidateEvaluated, EventBus
from .base import OptimizationCandidate, OptimizationContext, OptimizationResult, Optimizer

__all__ = [
    "RayTuneOptimizer",
    "RayTuneSettings",
    "DistributedOptimizer",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RayTuneSettings:
    """Configuration for Ray Tune distributed optimization."""
    
    # Ray settings
    num_cpus: int = -1  # -1 = all available
    num_gpus: int = 0
    ray_address: str | None = None  # None = local, or "ray://host:port" for cluster
    
    # Optimization settings
    num_samples: int = 100  # Number of hyperparameter combinations to try
    max_concurrent_trials: int = 4  # Parallel trials
    grace_period: int = 10  # Min iterations before early stopping
    
    # Scheduler
    scheduler: str = "asha"  # "asha", "pbt", or "none"
    reduction_factor: int = 4  # For ASHA scheduler
    
    # Search algorithm
    search_algorithm: str = "optuna"  # "optuna", "bayesopt", "random"
    
    # Storage
    storage_path: str | None = None  # For distributed storage
    resume: bool = False  # Resume from checkpoint


class RayTuneOptimizer(Optimizer):
    """Distributed hyperparameter optimization using Ray Tune.
    
    Scales to hundreds of nodes and thousands of parallel trials.
    Supports advanced scheduling and search algorithms.
    
    Features:
    - Distributed parallel trials across cluster
    - Early stopping with ASHA or PBT schedulers
    - Advanced search with Optuna or BayesOpt
    - Fault tolerance and checkpointing
    - Resource-aware scheduling (CPU/GPU)
    """
    
    def __init__(
        self,
        *,
        event_bus: EventBus | None = None,
        settings: RayTuneSettings | None = None,
    ) -> None:
        if not RAY_AVAILABLE:
            raise ImportError(
                "Ray is not installed. Install with: pip install 'ray[tune]' or "
                "uv pip install 'ray[tune]'"
            )
        
        self._bus = event_bus or EventBus()
        self._settings = settings or RayTuneSettings()
        self._context: OptimizationContext | None = None
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(
                address=self._settings.ray_address,
                num_cpus=self._settings.num_cpus if self._settings.num_cpus > 0 else None,
                num_gpus=self._settings.num_gpus,
                ignore_reinit_error=True,
            )
            logger.info(f"Ray initialized with {ray.cluster_resources()}")
    
    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        """Execute distributed optimization using Ray Tune."""
        self._context = context
        
        # Define search space for Ray Tune
        search_space = self._convert_search_space(context.model_search_space)
        
        # Create scheduler
        scheduler = self._create_scheduler()
        
        # Create search algorithm
        search_alg = self._create_search_algorithm()
        
        # Define training function
        trainable = self._create_trainable(context)
        
        # Run distributed optimization
        logger.info(
            f"Starting Ray Tune optimization with {self._settings.num_samples} samples "
            f"and {self._settings.max_concurrent_trials} concurrent trials"
        )
        
        analysis = tune.run(
            trainable,
            config=search_space,
            num_samples=self._settings.num_samples,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                "cpu": max(1, ray.cluster_resources().get("CPU", 1) // self._settings.max_concurrent_trials),
                "gpu": self._settings.num_gpus / self._settings.max_concurrent_trials if self._settings.num_gpus > 0 else 0,
            },
            local_dir=self._settings.storage_path,
            resume=self._settings.resume,
            verbose=1,
        )
        
        # Get best configuration
        best_trial = analysis.best_trial
        best_config = best_trial.config
        best_score = best_trial.last_result["score"]
        
        # Convert results
        candidates = []
        for trial in analysis.trials:
            if trial.last_result:
                candidates.append(
                    OptimizationCandidate(
                        params=trial.config,
                        score=trial.last_result.get("score", 0.0),
                    )
                )
        
        logger.info(
            f"Ray Tune optimization completed. Best score: {best_score:.6f} "
            f"from {len(candidates)} trials"
        )
        
        return OptimizationResult(
            best_params=best_config,
            best_score=best_score,
            candidates=candidates,
        )
    
    def _convert_search_space(
        self,
        search_space: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Convert AutoML search space to Ray Tune format."""
        # For simplicity, we'll use the first space item
        # In production, this should be more sophisticated
        if not search_space or len(search_space) == 0:
            return {}
        
        base_space = search_space[0] if search_space else {}
        tune_space = {}
        
        for key, value in base_space.items():
            if isinstance(value, list):
                # Categorical choice
                tune_space[key] = tune.choice(value)
            elif isinstance(value, tuple) and len(value) == 2:
                # Range (min, max)
                if isinstance(value[0], int):
                    tune_space[key] = tune.randint(value[0], value[1])
                else:
                    tune_space[key] = tune.uniform(value[0], value[1])
            else:
                # Fixed value
                tune_space[key] = value
        
        return tune_space
    
    def _create_scheduler(self):
        """Create Ray Tune scheduler based on settings."""
        if self._settings.scheduler == "asha":
            return ASHAScheduler(
                metric="score",
                mode="max",
                grace_period=self._settings.grace_period,
                reduction_factor=self._settings.reduction_factor,
            )
        elif self._settings.scheduler == "pbt":
            return PopulationBasedTraining(
                metric="score",
                mode="max",
                perturbation_interval=4,
            )
        else:
            return None
    
    def _create_search_algorithm(self):
        """Create search algorithm based on settings."""
        if self._settings.search_algorithm == "optuna":
            return OptunaSearch(metric="score", mode="max")
        elif self._settings.search_algorithm == "bayesopt":
            return BayesOptSearch(metric="score", mode="max")
        else:
            return None
    
    def _create_trainable(self, context: OptimizationContext) -> Callable:
        """Create Ray Tune trainable function."""
        
        def train_fn(config: dict[str, Any]) -> dict[str, Any]:
            """Training function executed on each worker."""
            from sklearn.model_selection import cross_val_score
            
            # Build pipeline with current config
            pipeline = context.pipeline_builder(config)
            
            # Evaluate with cross-validation
            scores = cross_val_score(
                pipeline,
                context.dataset.X_train,
                context.dataset.y_train,
                cv=context.cv_folds,
                scoring=context.scoring,
                n_jobs=1,  # Already parallel at Ray level
            )
            
            score = float(np.mean(scores))
            
            # Publish event
            self._bus.publish(
                CandidateEvaluated(
                    run_id=context.run_id,
                    params=config,
                    score=score,
                )
            )
            
            # Return result for Ray Tune
            return {"score": score}
        
        return train_fn


class DistributedOptimizer(Optimizer):
    """Wrapper that distributes any optimizer across Ray cluster.
    
    Takes any existing optimizer and runs it in parallel across
    multiple nodes using Ray's distributed computing.
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        num_workers: int = 4,
        ray_address: str | None = None,
    ) -> None:
        """Initialize distributed optimizer.
        
        Args:
            base_optimizer: The optimizer to distribute
            num_workers: Number of parallel workers
            ray_address: Ray cluster address (None for local)
        """
        if not RAY_AVAILABLE:
            raise ImportError("Ray is not installed. Install with: pip install ray")
        
        self.base_optimizer = base_optimizer
        self.num_workers = num_workers
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(address=ray_address, ignore_reinit_error=True)
    
    def optimize(self, context: OptimizationContext) -> OptimizationResult:
        """Distribute optimization across Ray workers."""
        
        @ray.remote
        def optimize_remote(ctx: OptimizationContext) -> OptimizationResult:
            """Run optimization on remote worker."""
            return self.base_optimizer.optimize(ctx)
        
        # Submit multiple optimization jobs
        futures = [
            optimize_remote.remote(context)
            for _ in range(self.num_workers)
        ]
        
        # Gather results
        results = ray.get(futures)
        
        # Combine results (take best)
        best_result = max(results, key=lambda r: r.best_score)
        
        # Merge all candidates
        all_candidates = []
        for result in results:
            all_candidates.extend(result.candidates)
        
        best_result.candidates = all_candidates
        
        return best_result
