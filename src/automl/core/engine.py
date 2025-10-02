"""End-to-end AutoML orchestration engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..datasets.base import DatasetProvider
from ..models.base import ModelFactory
from ..optimizers.base import OptimizationContext, Optimizer
from ..optimizers.random_search import RandomSearchOptimizer, RandomSearchSettings
from ..pipelines.base import PipelineFactory, PreprocessorFactory
from .config import AutoMLConfig
from .events import EventBus, RunCompleted, RunStarted
from .registry import Registry

__all__ = ["AutoMLEngine", "EngineInstrumentation", "default_engine"]


@dataclass(slots=True)
class EngineInstrumentation:
    """Holds instrumentation utilities used by the engine."""

    events: EventBus


class AutoMLEngine:
    """Central coordinator responsible for executing AutoML runs."""

    def __init__(self, *, instrumentation: EngineInstrumentation | None = None) -> None:
        self.datasets: Registry[DatasetProvider] = Registry("dataset")
        self.preprocessors: Registry[PreprocessorFactory] = Registry("preprocessor")
        self.models: Registry[ModelFactory] = Registry("model")
        self.optimizers: Registry[Callable[[Mapping[str, Any] | None], Optimizer]] = Registry("optimizer")
        self.instrumentation = instrumentation or EngineInstrumentation(events=EventBus())
        self._register_defaults()

    def _register_defaults(self) -> None:
        if "random_search" not in self.optimizers:
            self.optimizers.register(
                "random_search",
                lambda params=None: RandomSearchOptimizer(
                    event_bus=self.instrumentation.events,
                    settings=RandomSearchSettings(**(params or {})),
                ),
                description="Stochastic hyper-parameter search",
            )

    def register_dataset(self, name: str, provider: DatasetProvider, *, description: str | None = None) -> None:
        self.datasets.register(name, provider, description=description)

    def register_preprocessor(
        self, name: str, factory: PreprocessorFactory, *, description: str | None = None
    ) -> None:
        self.preprocessors.register(name, factory, description=description)

    def register_model(self, name: str, factory: ModelFactory, *, description: str | None = None) -> None:
        self.models.register(name, factory, description=description)

    def register_optimizer(
        self,
        name: str,
        factory: Callable[[Mapping[str, Any] | None], Optimizer],
        *,
        description: str | None = None,
    ) -> None:
        self.optimizers.register(name, factory, description=description)

    def run(self, config: AutoMLConfig) -> dict[str, Any]:
        run_id = f"run-{uuid.uuid4()}"
        hashed = config.hash()
        self.instrumentation.events.publish(RunStarted(run_id=run_id, config_hash=hashed))

        dataset_provider = self.datasets.require(config.dataset.name)
        dataset = dataset_provider()

        pipeline_factory = self._build_pipeline_factory(config)
        optimizer = self._resolve_optimizer(config)

        search_space = config.pipeline.model.search_space or [{}]
        merged_space = [self._merge_params(config.pipeline.model.base_params, candidate) for candidate in search_space]

        preprocessor_overrides = [item.params for item in config.pipeline.preprocessors]

        context = OptimizationContext(
            run_id=run_id,
            dataset=dataset,
            pipeline_builder=lambda params: pipeline_factory.build(
                model_params=params,
                preprocessor_overrides=preprocessor_overrides,
            ),
            model_search_space=merged_space,
            scoring=config.optimizer.scoring,
            cv_folds=config.optimizer.cv_folds,
        )

        result = optimizer.optimize(context)

        report = {
            "run_id": run_id,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "candidates": [
                {"params": candidate.params, "score": candidate.score} for candidate in result.candidates
            ],
        }

        self.instrumentation.events.publish(
            RunCompleted(
                run_id=run_id,
                best_score=result.best_score,
                best_pipeline={"model": config.pipeline.model.name, "params": result.best_params},
                candidate_count=len(result.candidates),
            )
        )
        return report

    def _build_pipeline_factory(self, config: AutoMLConfig) -> PipelineFactory:
        preprocessor_factories: list[PreprocessorFactory] = []
        for preprocessor in config.pipeline.preprocessors:
            factory = self.preprocessors.require(preprocessor.name)
            preprocessor_factories.append(factory)

        model_factory = self.models.require(config.pipeline.model.name)
        return PipelineFactory(preprocessor_factories, model_factory)

    @staticmethod
    def _merge_params(
        base: Mapping[str, Any] | None,
        candidate: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        merged = dict(base or {})
        merged.update(candidate)
        return merged

    def _resolve_optimizer(self, config: AutoMLConfig) -> Optimizer:
        factory = self.optimizers.require(config.optimizer.name)
        return factory(config.optimizer.params)


def default_engine() -> AutoMLEngine:
    """Factory returning an engine with built-in defaults loaded."""

    from ..datasets.builtin import iris_dataset
    from ..pipelines import sklearn as sklearn_pipelines
    from ..pipelines import advanced as advanced_pipelines
    from ..models import sklearn as sklearn_models
    from ..models import boosting as boosting_models
    from ..models import ensemble as ensemble_models
    from ..optimizers import optuna_optimizer as optuna_opt

    engine = AutoMLEngine()
    
    # Register datasets
    engine.register_dataset("iris", iris_dataset, description="Classic Iris dataset")

    # Register basic preprocessors
    engine.register_preprocessor(
        "standard_scaler",
        sklearn_pipelines.standard_scaler,
        description="Normalize features to zero mean and unit variance",
    )
    engine.register_preprocessor(
        "pca",
        sklearn_pipelines.pca,
        description="Dimensionality reduction using Principal Component Analysis",
    )
    
    # Register advanced preprocessors
    engine.register_preprocessor(
        "robust_scaler",
        advanced_pipelines.robust_scaler,
        description="Robust scaling resistant to outliers",
    )
    engine.register_preprocessor(
        "power_transformer",
        advanced_pipelines.power_transformer,
        description="Power transformation for non-normal distributions",
    )
    engine.register_preprocessor(
        "quantile_transformer",
        advanced_pipelines.quantile_transformer,
        description="Quantile-based uniform or normal distribution transformation",
    )
    engine.register_preprocessor(
        "polynomial_features",
        advanced_pipelines.polynomial_features,
        description="Generate polynomial and interaction features",
    )
    
    # Register basic sklearn models
    engine.register_model(
        "logistic_regression",
        sklearn_models.logistic_regression,
        description="Multinomial logistic regression classifier",
    )
    engine.register_model(
        "random_forest_classifier",
        sklearn_models.random_forest_classifier,
        description="Tree-based ensemble classifier",
    )
    
    # Register advanced gradient boosting models
    engine.register_model(
        "xgboost_classifier",
        boosting_models.xgboost_classifier,
        description="State-of-the-art XGBoost classifier with GPU support",
    )
    engine.register_model(
        "xgboost_regressor",
        boosting_models.xgboost_regressor,
        description="State-of-the-art XGBoost regressor with GPU support",
    )
    engine.register_model(
        "lightgbm_classifier",
        boosting_models.lightgbm_classifier,
        description="LightGBM classifier with GPU support",
    )
    engine.register_model(
        "lightgbm_regressor",
        boosting_models.lightgbm_regressor,
        description="LightGBM regressor with GPU support",
    )
    engine.register_model(
        "catboost_classifier",
        boosting_models.catboost_classifier,
        description="CatBoost classifier with automatic categorical handling",
    )
    engine.register_model(
        "catboost_regressor",
        boosting_models.catboost_regressor,
        description="CatBoost regressor with automatic categorical handling",
    )
    
    # Register ensemble models
    engine.register_model(
        "auto_ensemble_classifier",
        lambda params=None: ensemble_models.AutoEnsembleClassifier(**(params or {})),
        description="Automatic ensemble classifier with intelligent model selection",
    )
    engine.register_model(
        "auto_ensemble_regressor",
        lambda params=None: ensemble_models.AutoEnsembleRegressor(**(params or {})),
        description="Automatic ensemble regressor with intelligent model selection",
    )
    
    # Register advanced optimizers
    engine.register_optimizer(
        "optuna",
        lambda params: optuna_opt.OptunaOptimizer(
            event_bus=engine.instrumentation.events,
            settings=optuna_opt.OptunaSettings(**(params or {})),
        ),
        description="State-of-the-art Optuna optimizer with TPE/CMA-ES/NSGA-II",
    )
    engine.register_optimizer(
        "optuna_multiobjective",
        lambda params: optuna_opt.MultiObjectiveOptunaOptimizer(
            event_bus=engine.instrumentation.events,
            settings=optuna_opt.MultiObjectiveSettings(**(params or {})),
        ),
        description="Multi-objective optimization using NSGA-II",
    )
    
    # Register distributed optimizer (Ray Tune) - optional
    try:
        from ..optimizers import ray_optimizer as ray_opt
        engine.register_optimizer(
            "ray_tune",
            lambda params: ray_opt.RayTuneOptimizer(
                event_bus=engine.instrumentation.events,
                settings=ray_opt.RayTuneSettings(**(params or {})),
            ),
            description="Distributed hyperparameter optimization with Ray Tune - exabyte scale",
        )
    except ImportError:
        # Ray is optional dependency
        pass
    
    return engine
