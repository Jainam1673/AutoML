"""MLflow experiment tracking and model registry for production ML.

Tracks experiments, logs metrics, stores artifacts, and manages model lifecycle
at exabyte scale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlflow.models import infer_signature
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

__all__ = [
    "MLflowTracker",
    "ExperimentTracker",
    "ModelRegistry",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MLflowConfig:
    """Configuration for MLflow tracking."""
    
    tracking_uri: str = "sqlite:///mlflow.db"  # or "http://mlflow-server:5000"
    experiment_name: str = "automl-experiment"
    artifact_location: str | None = None  # S3/GCS/Azure path
    registry_uri: str | None = None  # Model registry URI


class MLflowTracker:
    """Track AutoML experiments with MLflow.
    
    Features:
    - Automatic experiment tracking
    - Hyperparameter logging
    - Metrics and loss curves
    - Model artifacts and metadata
    - Distributed tracking support
    - Integration with cloud storage (S3, GCS, Azure)
    """
    
    def __init__(self, config: MLflowConfig | None = None) -> None:
        """Initialize MLflow tracker.
        
        Args:
            config: MLflow configuration
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "MLflow is not installed. Install with: pip install mlflow or "
                "uv pip install mlflow"
            )
        
        self.config = config or MLflowConfig()
        self.client: MlflowClient | None = None
        self.experiment_id: str | None = None
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        # Set tracking URI
        mlflow.set_tracking_uri(self.config.tracking_uri)
        logger.info(f"MLflow tracking URI: {self.config.tracking_uri}")
        
        # Set registry URI if provided
        if self.config.registry_uri:
            mlflow.set_registry_uri(self.config.registry_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                self.config.experiment_name,
                artifact_location=self.config.artifact_location,
            )
            logger.info(f"Created MLflow experiment: {self.config.experiment_name}")
        except Exception:
            # Experiment already exists
            experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
            if experiment:
                self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {self.config.experiment_name}")
        
        # Initialize client
        self.client = MlflowClient()
    
    def start_run(
        self,
        run_name: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> str:
        """Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Tags to attach to run
        
        Returns:
            Run ID
        """
        run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags,
        )
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run.info.run_id
    
    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
    
    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number for time-series metrics
        """
        mlflow.log_metrics(metrics, step=step)
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """Log single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        mlflow.log_metric(key, value, step=step)
    
    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        signature: Any = None,
        input_example: Any = None,
        registered_model_name: str | None = None,
    ) -> None:
        """Log trained model.
        
        Args:
            model: Trained model
            artifact_path: Path within run to store model
            signature: Model signature (input/output schema)
            input_example: Example input for inference
            registered_model_name: Register model with this name
        """
        mlflow.sklearn.log_model(
            model,
            artifact_path=artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name,
        )
        logger.info(f"Logged model to {artifact_path}")
    
    def log_artifact(self, local_path: str | Path) -> None:
        """Log artifact file.
        
        Args:
            local_path: Path to local file
        """
        mlflow.log_artifact(str(local_path))
    
    def log_artifacts(self, local_dir: str | Path) -> None:
        """Log directory of artifacts.
        
        Args:
            local_dir: Path to local directory
        """
        mlflow.log_artifacts(str(local_dir))
    
    def end_run(self) -> None:
        """End current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")
    
    def get_best_run(
        self,
        metric: str = "test_score",
        ascending: bool = False,
    ) -> Any:
        """Get best run from experiment.
        
        Args:
            metric: Metric to optimize
            ascending: True for minimization, False for maximization
        
        Returns:
            Best run
        """
        if not self.experiment_id:
            raise ValueError("No experiment ID set")
        
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        
        if len(runs) == 0:
            raise ValueError("No runs found")
        
        return runs.iloc[0]


class ExperimentTracker:
    """High-level experiment tracking interface.
    
    Automatically tracks AutoML runs with MLflow.
    """
    
    def __init__(self, mlflow_tracker: MLflowTracker) -> None:
        """Initialize experiment tracker.
        
        Args:
            mlflow_tracker: MLflow tracker instance
        """
        self.tracker = mlflow_tracker
        self.active_run: str | None = None
    
    def track_optimization(
        self,
        run_name: str,
        params: Mapping[str, Any],
        metrics: Mapping[str, float],
        model: Any = None,
        artifacts: list[Path] | None = None,
    ) -> str:
        """Track a complete optimization run.
        
        Args:
            run_name: Name for this run
            params: Hyperparameters
            metrics: Evaluation metrics
            model: Trained model (optional)
            artifacts: Additional artifacts (optional)
        
        Returns:
            Run ID
        """
        # Start run
        run_id = self.tracker.start_run(
            run_name=run_name,
            tags={"framework": "automl", "version": "0.1.0"},
        )
        
        try:
            # Log parameters
            self.tracker.log_params(params)
            
            # Log metrics
            self.tracker.log_metrics(metrics)
            
            # Log model if provided
            if model is not None:
                self.tracker.log_model(model)
            
            # Log artifacts if provided
            if artifacts:
                for artifact in artifacts:
                    self.tracker.log_artifact(artifact)
            
            logger.info(f"Tracked optimization run: {run_id}")
            
        finally:
            # Always end run
            self.tracker.end_run()
        
        return run_id


class ModelRegistry:
    """Manage model lifecycle with MLflow Model Registry.
    
    Features:
    - Version control for models
    - Stage transitions (Staging → Production)
    - Model metadata and lineage
    - A/B testing support
    """
    
    def __init__(self, config: MLflowConfig | None = None) -> None:
        """Initialize model registry.
        
        Args:
            config: MLflow configuration
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is not installed")
        
        self.config = config or MLflowConfig()
        mlflow.set_tracking_uri(self.config.tracking_uri)
        
        if self.config.registry_uri:
            mlflow.set_registry_uri(self.config.registry_uri)
        
        self.client = MlflowClient()
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Mapping[str, str] | None = None,
    ) -> Any:
        """Register a model.
        
        Args:
            model_uri: URI of model (e.g., "runs:/run-id/model")
            name: Model name
            tags: Optional tags
        
        Returns:
            Registered model version
        """
        result = mlflow.register_model(model_uri, name)
        
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(name, result.version, key, value)
        
        logger.info(f"Registered model {name} version {result.version}")
        return result
    
    def transition_model_stage(
        self,
        name: str,
        version: int,
        stage: str,
    ) -> None:
        """Transition model to new stage.
        
        Args:
            name: Model name
            version: Model version
            stage: Target stage ("Staging", "Production", "Archived")
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
        )
        logger.info(f"Transitioned {name} v{version} to {stage}")
    
    def get_latest_model(
        self,
        name: str,
        stage: str = "Production",
    ) -> str:
        """Get URI of latest model in stage.
        
        Args:
            name: Model name
            stage: Stage to get model from
        
        Returns:
            Model URI
        """
        model_uri = f"models:/{name}/{stage}"
        logger.info(f"Loading model from {model_uri}")
        return model_uri
    
    def load_production_model(self, name: str) -> Any:
        """Load production model.
        
        Args:
            name: Model name
        
        Returns:
            Loaded model
        """
        model_uri = self.get_latest_model(name, "Production")
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded production model: {name}")
        return model
