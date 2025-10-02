"""Comprehensive benchmarking suite for AutoML.

Includes OpenML integration, standard benchmark datasets,
and leaderboard tracking.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkSuite",
    "LeaderboardManager",
]

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    
    dataset_name: str
    model_name: str
    task_type: str
    metric_name: str
    score: float
    train_time: float
    predict_time: float
    n_samples: int
    n_features: int
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create from dictionary."""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


class BenchmarkSuite:
    """Comprehensive benchmark suite.
    
    Tests models on standard datasets with consistent evaluation.
    """
    
    def __init__(self, results_dir: Path | None = None) -> None:
        """Initialize benchmark suite.
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir or Path("./benchmark_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []
    
    def load_openml_dataset(self, dataset_id: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Load dataset from OpenML.
        
        Args:
            dataset_id: OpenML dataset ID
        
        Returns:
            Tuple of (X, y, metadata)
        """
        try:
            import openml
        except ImportError:
            raise ImportError("openml required: pip install openml")
        
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, _, _ = dataset.get_data()
        
        metadata = {
            "name": dataset.name,
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "task_type": "classification" if dataset.qualities["NumberOfClasses"] else "regression",
        }
        
        return X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y, metadata
    
    def benchmark_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        task_type: str = "classification",
        cv: int = 5,
    ) -> BenchmarkResult:
        """Benchmark a single model.
        
        Args:
            model: Model to benchmark
            X: Features
            y: Target
            dataset_name: Dataset name
            task_type: "classification" or "regression"
            cv: Cross-validation folds
        
        Returns:
            Benchmark result
        """
        model_name = model.__class__.__name__
        
        # Training time
        start_time = time.time()
        model.fit(X, y)
        train_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X)
        predict_time = time.time() - start_time
        
        # Metrics
        if task_type == "classification":
            metric_name = "accuracy"
            score = accuracy_score(y, y_pred)
            additional_metrics = {
                "f1": f1_score(y, y_pred, average="weighted"),
                "precision": precision_score(y, y_pred, average="weighted"),
                "recall": recall_score(y, y_pred, average="weighted"),
            }
        else:
            metric_name = "r2"
            score = r2_score(y, y_pred)
            additional_metrics = {
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "mae": np.abs(y - y_pred).mean(),
            }
        
        result = BenchmarkResult(
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            metric_name=metric_name,
            score=score,
            train_time=train_time,
            predict_time=predict_time,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            additional_metrics=additional_metrics,
        )
        
        self.results.append(result)
        
        logger.info(
            f"Benchmarked {model_name} on {dataset_name}: "
            f"{metric_name}={score:.4f} (train={train_time:.2f}s)"
        )
        
        return result
    
    def run_classification_suite(
        self,
        models: list[Any],
        datasets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run classification benchmark suite.
        
        Args:
            models: Models to benchmark
            datasets: Dataset names (None = use built-in)
        
        Returns:
            Results DataFrame
        """
        from sklearn.datasets import (
            load_iris,
            load_wine,
            load_breast_cancer,
            load_digits,
        )
        
        # Built-in datasets
        builtin_datasets = {
            "iris": load_iris,
            "wine": load_wine,
            "breast_cancer": load_breast_cancer,
            "digits": load_digits,
        }
        
        if datasets is None:
            datasets = list(builtin_datasets.keys())
        
        for dataset_name in datasets:
            if dataset_name in builtin_datasets:
                data = builtin_datasets[dataset_name]()
                X, y = data.data, data.target
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            logger.info(f"Benchmarking on {dataset_name}")
            
            for model in models:
                try:
                    self.benchmark_model(
                        model,
                        X,
                        y,
                        dataset_name=dataset_name,
                        task_type="classification",
                    )
                except Exception as e:
                    logger.error(f"Failed to benchmark {model.__class__.__name__}: {e}")
        
        return self.get_results_dataframe()
    
    def run_regression_suite(
        self,
        models: list[Any],
        datasets: list[str] | None = None,
    ) -> pd.DataFrame:
        """Run regression benchmark suite.
        
        Args:
            models: Models to benchmark
            datasets: Dataset names (None = use built-in)
        
        Returns:
            Results DataFrame
        """
        from sklearn.datasets import (
            load_diabetes,
            fetch_california_housing,
        )
        
        builtin_datasets = {
            "diabetes": load_diabetes,
            "california_housing": fetch_california_housing,
        }
        
        if datasets is None:
            datasets = list(builtin_datasets.keys())
        
        for dataset_name in datasets:
            if dataset_name in builtin_datasets:
                data = builtin_datasets[dataset_name]()
                X, y = data.data, data.target
            else:
                logger.warning(f"Unknown dataset: {dataset_name}")
                continue
            
            logger.info(f"Benchmarking on {dataset_name}")
            
            for model in models:
                try:
                    self.benchmark_model(
                        model,
                        X,
                        y,
                        dataset_name=dataset_name,
                        task_type="regression",
                    )
                except Exception as e:
                    logger.error(f"Failed to benchmark {model.__class__.__name__}: {e}")
        
        return self.get_results_dataframe()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame.
        
        Returns:
            Results DataFrame
        """
        if not self.results:
            return pd.DataFrame()
        
        data = [result.to_dict() for result in self.results]
        return pd.DataFrame(data)
    
    def save_results(self, path: Path | None = None) -> None:
        """Save benchmark results.
        
        Args:
            path: Output path (None = use timestamp)
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.results_dir / f"benchmark_{timestamp}.json"
        
        data = [result.to_dict() for result in self.results]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {path}")
    
    def load_results(self, path: Path) -> None:
        """Load benchmark results.
        
        Args:
            path: Results file path
        """
        with open(path) as f:
            data = json.load(f)
        
        self.results = [BenchmarkResult.from_dict(r) for r in data]
        
        logger.info(f"Loaded {len(self.results)} results from {path}")


class LeaderboardManager:
    """Manage model leaderboards.
    
    Tracks best models across datasets and tasks.
    """
    
    def __init__(self, leaderboard_dir: Path | None = None) -> None:
        """Initialize leaderboard manager.
        
        Args:
            leaderboard_dir: Directory for leaderboards
        """
        self.leaderboard_dir = leaderboard_dir or Path("./leaderboards")
        self.leaderboard_dir.mkdir(parents=True, exist_ok=True)
    
    def update_leaderboard(
        self,
        result: BenchmarkResult,
        leaderboard_name: str = "global",
    ) -> None:
        """Update leaderboard with new result.
        
        Args:
            result: Benchmark result
            leaderboard_name: Leaderboard name
        """
        leaderboard_path = self.leaderboard_dir / f"{leaderboard_name}.json"
        
        # Load existing leaderboard
        if leaderboard_path.exists():
            with open(leaderboard_path) as f:
                leaderboard = json.load(f)
        else:
            leaderboard = []
        
        # Add new result
        leaderboard.append(result.to_dict())
        
        # Sort by score (descending)
        leaderboard.sort(key=lambda x: x["score"], reverse=True)
        
        # Keep top 100
        leaderboard = leaderboard[:100]
        
        # Save
        with open(leaderboard_path, "w") as f:
            json.dump(leaderboard, f, indent=2)
    
    def get_leaderboard(
        self,
        leaderboard_name: str = "global",
        dataset_name: str | None = None,
        task_type: str | None = None,
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Get leaderboard.
        
        Args:
            leaderboard_name: Leaderboard name
            dataset_name: Filter by dataset
            task_type: Filter by task type
            top_k: Number of top entries
        
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_path = self.leaderboard_dir / f"{leaderboard_name}.json"
        
        if not leaderboard_path.exists():
            return pd.DataFrame()
        
        with open(leaderboard_path) as f:
            leaderboard = json.load(f)
        
        # Filter
        if dataset_name:
            leaderboard = [r for r in leaderboard if r["dataset_name"] == dataset_name]
        
        if task_type:
            leaderboard = [r for r in leaderboard if r["task_type"] == task_type]
        
        # Convert to DataFrame
        df = pd.DataFrame(leaderboard[:top_k])
        
        return df
    
    def compare_models(
        self,
        model_names: list[str],
        leaderboard_name: str = "global",
    ) -> pd.DataFrame:
        """Compare models across datasets.
        
        Args:
            model_names: Models to compare
            leaderboard_name: Leaderboard name
        
        Returns:
            Comparison DataFrame
        """
        leaderboard = self.get_leaderboard(leaderboard_name, top_k=1000)
        
        if leaderboard.empty:
            return pd.DataFrame()
        
        # Filter by models
        comparison = leaderboard[leaderboard["model_name"].isin(model_names)]
        
        # Pivot for easy comparison
        pivot = comparison.pivot_table(
            index="dataset_name",
            columns="model_name",
            values="score",
            aggfunc="max",
        )
        
        return pivot
    
    def get_best_model(
        self,
        dataset_name: str,
        task_type: str,
        leaderboard_name: str = "global",
    ) -> dict[str, Any] | None:
        """Get best model for a dataset.
        
        Args:
            dataset_name: Dataset name
            task_type: Task type
            leaderboard_name: Leaderboard name
        
        Returns:
            Best model info or None
        """
        leaderboard = self.get_leaderboard(
            leaderboard_name,
            dataset_name=dataset_name,
            task_type=task_type,
            top_k=1,
        )
        
        if leaderboard.empty:
            return None
        
        return leaderboard.iloc[0].to_dict()
    
    def export_leaderboard(
        self,
        leaderboard_name: str = "global",
        format: str = "markdown",
    ) -> str:
        """Export leaderboard as formatted string.
        
        Args:
            leaderboard_name: Leaderboard name
            format: Output format ("markdown", "html", "latex")
        
        Returns:
            Formatted leaderboard
        """
        df = self.get_leaderboard(leaderboard_name, top_k=20)
        
        if df.empty:
            return "No leaderboard data available."
        
        # Select columns
        columns = [
            "model_name",
            "dataset_name",
            "score",
            "train_time",
            "timestamp",
        ]
        df = df[columns]
        
        if format == "markdown":
            return df.to_markdown(index=False)
        elif format == "html":
            return df.to_html(index=False)
        elif format == "latex":
            return df.to_latex(index=False)
        else:
            return str(df)
