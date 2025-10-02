"""Meta-learning for AutoML - Learn from previous experiments.

Warm-start optimization using meta-features and algorithm recommendation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

__all__ = [
    "MetaLearner",
    "MetaFeatureExtractor",
    "AlgorithmRecommender",
    "WarmStarter",
]

logger = logging.getLogger(__name__)


@dataclass
class MetaFeatures:
    """Meta-features describing a dataset."""
    
    # Basic statistics
    n_samples: int
    n_features: int
    n_classes: int | None
    task_type: str  # "classification" or "regression"
    
    # Statistical features
    mean_std: float
    mean_skewness: float
    mean_kurtosis: float
    
    # Information theory
    class_entropy: float | None
    feature_entropy: float
    
    # Complexity
    dimensionality: float  # n_features / n_samples
    class_imbalance: float | None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetaFeatures:
        """Create from dictionary."""
        return cls(**data)


class MetaFeatureExtractor:
    """Extract meta-features from datasets."""
    
    def extract(self, X: np.ndarray, y: np.ndarray | None = None) -> MetaFeatures:
        """Extract meta-features.
        
        Args:
            X: Features
            y: Labels (optional, for classification)
        
        Returns:
            Meta-features
        """
        from scipy import stats
        
        n_samples, n_features = X.shape
        
        # Determine task type
        if y is None:
            task_type = "unsupervised"
            n_classes = None
        elif len(np.unique(y)) < 20:  # Arbitrary threshold
            task_type = "classification"
            n_classes = len(np.unique(y))
        else:
            task_type = "regression"
            n_classes = None
        
        # Statistical features
        mean_std = float(np.mean(np.std(X, axis=0)))
        mean_skewness = float(np.mean(stats.skew(X, axis=0)))
        mean_kurtosis = float(np.mean(stats.kurtosis(X, axis=0)))
        
        # Information theory
        feature_entropy = float(np.mean([
            stats.entropy(np.histogram(X[:, i], bins=10)[0] + 1e-10)
            for i in range(min(n_features, 50))  # Sample to avoid slowdown
        ]))
        
        class_entropy = None
        class_imbalance = None
        if task_type == "classification" and y is not None:
            class_counts = np.bincount(y.astype(int))
            class_probs = class_counts / class_counts.sum()
            class_entropy = float(stats.entropy(class_probs + 1e-10))
            class_imbalance = float(class_counts.max() / class_counts.min())
        
        # Complexity
        dimensionality = n_features / n_samples
        
        return MetaFeatures(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            task_type=task_type,
            mean_std=mean_std,
            mean_skewness=mean_skewness,
            mean_kurtosis=mean_kurtosis,
            class_entropy=class_entropy,
            feature_entropy=feature_entropy,
            dimensionality=dimensionality,
            class_imbalance=class_imbalance,
        )


@dataclass
class ExperimentRecord:
    """Record of a past experiment."""
    
    meta_features: MetaFeatures
    algorithm: str
    hyperparameters: dict[str, Any]
    performance: float
    runtime: float


class AlgorithmRecommender:
    """Recommend algorithms based on meta-features.
    
    Uses meta-learning to suggest which algorithms are likely
    to perform well on a new dataset.
    """
    
    def __init__(self) -> None:
        """Initialize algorithm recommender."""
        self.recommender_model: RandomForestClassifier | None = None
        self.algorithm_encoder: dict[str, int] = {}
        self.experiments: list[ExperimentRecord] = []
    
    def add_experiment(self, record: ExperimentRecord) -> None:
        """Add experiment to knowledge base.
        
        Args:
            record: Experiment record
        """
        self.experiments.append(record)
    
    def fit(self) -> None:
        """Train recommendation model from experiments."""
        if len(self.experiments) < 10:
            logger.warning("Not enough experiments for meta-learning")
            return
        
        # Extract features
        X = []
        y = []
        
        for exp in self.experiments:
            # Convert meta-features to vector
            features = [
                exp.meta_features.n_samples,
                exp.meta_features.n_features,
                exp.meta_features.n_classes or 0,
                exp.meta_features.mean_std,
                exp.meta_features.mean_skewness,
                exp.meta_features.mean_kurtosis,
                exp.meta_features.feature_entropy,
                exp.meta_features.dimensionality,
                exp.meta_features.class_imbalance or 0,
            ]
            X.append(features)
            
            # Encode algorithm
            if exp.algorithm not in self.algorithm_encoder:
                self.algorithm_encoder[exp.algorithm] = len(self.algorithm_encoder)
            y.append(self.algorithm_encoder[exp.algorithm])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train recommender
        self.recommender_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        self.recommender_model.fit(X, y)
        
        logger.info(f"Trained recommender on {len(self.experiments)} experiments")
    
    def recommend(
        self,
        meta_features: MetaFeatures,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Recommend top algorithms for dataset.
        
        Args:
            meta_features: Dataset meta-features
            top_k: Number of algorithms to recommend
        
        Returns:
            List of (algorithm, confidence) tuples
        """
        if self.recommender_model is None:
            logger.warning("Recommender not trained, returning default")
            return [("xgboost_classifier", 1.0)]
        
        # Convert meta-features to vector
        features = np.array([[
            meta_features.n_samples,
            meta_features.n_features,
            meta_features.n_classes or 0,
            meta_features.mean_std,
            meta_features.mean_skewness,
            meta_features.mean_kurtosis,
            meta_features.feature_entropy,
            meta_features.dimensionality,
            meta_features.class_imbalance or 0,
        ]])
        
        # Get probabilities
        probs = self.recommender_model.predict_proba(features)[0]
        
        # Get top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        # Decode algorithm names
        algorithm_decoder = {v: k for k, v in self.algorithm_encoder.items()}
        recommendations = [
            (algorithm_decoder[idx], float(probs[idx]))
            for idx in top_indices
        ]
        
        logger.info(f"Recommended algorithms: {recommendations}")
        
        return recommendations


class WarmStarter:
    """Warm-start hyperparameter optimization from previous runs.
    
    Uses meta-learning to suggest good starting hyperparameters
    based on similar datasets.
    """
    
    def __init__(self, knowledge_base_path: Path | None = None) -> None:
        """Initialize warm starter.
        
        Args:
            knowledge_base_path: Path to save/load knowledge base
        """
        self.knowledge_base_path = knowledge_base_path
        self.experiments: list[ExperimentRecord] = []
        
        if knowledge_base_path and knowledge_base_path.exists():
            self.load_knowledge_base()
    
    def add_experiment(
        self,
        meta_features: MetaFeatures,
        algorithm: str,
        hyperparameters: dict[str, Any],
        performance: float,
        runtime: float,
    ) -> None:
        """Add experiment result.
        
        Args:
            meta_features: Dataset meta-features
            algorithm: Algorithm name
            hyperparameters: Hyperparameters used
            performance: Achieved performance
            runtime: Training time in seconds
        """
        record = ExperimentRecord(
            meta_features=meta_features,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            performance=performance,
            runtime=runtime,
        )
        self.experiments.append(record)
        
        # Auto-save
        if self.knowledge_base_path:
            self.save_knowledge_base()
    
    def get_warm_start_configs(
        self,
        meta_features: MetaFeatures,
        algorithm: str,
        n_configs: int = 10,
    ) -> list[dict[str, Any]]:
        """Get warm-start configurations.
        
        Args:
            meta_features: Dataset meta-features
            algorithm: Algorithm to configure
            n_configs: Number of configurations to return
        
        Returns:
            List of hyperparameter configurations
        """
        # Find similar experiments
        similar_experiments = [
            exp for exp in self.experiments
            if exp.algorithm == algorithm
        ]
        
        if not similar_experiments:
            logger.info(f"No previous experiments for {algorithm}")
            return []
        
        # Compute similarity (simple Euclidean distance)
        similarities = []
        for exp in similar_experiments:
            distance = self._compute_meta_feature_distance(
                meta_features,
                exp.meta_features,
            )
            # Weight by performance
            similarity = exp.performance / (1 + distance)
            similarities.append(similarity)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-n_configs:][::-1]
        
        configs = [
            similar_experiments[idx].hyperparameters
            for idx in top_indices
        ]
        
        logger.info(f"Found {len(configs)} warm-start configurations")
        
        return configs
    
    def _compute_meta_feature_distance(
        self,
        mf1: MetaFeatures,
        mf2: MetaFeatures,
    ) -> float:
        """Compute distance between meta-features."""
        # Normalize and compute Euclidean distance
        features1 = np.array([
            np.log1p(mf1.n_samples),
            np.log1p(mf1.n_features),
            mf1.mean_std,
            mf1.dimensionality,
        ])
        
        features2 = np.array([
            np.log1p(mf2.n_samples),
            np.log1p(mf2.n_features),
            mf2.mean_std,
            mf2.dimensionality,
        ])
        
        return float(np.linalg.norm(features1 - features2))
    
    def save_knowledge_base(self) -> None:
        """Save knowledge base to disk."""
        if not self.knowledge_base_path:
            return
        
        data = {
            "experiments": [
                {
                    "meta_features": exp.meta_features.to_dict(),
                    "algorithm": exp.algorithm,
                    "hyperparameters": exp.hyperparameters,
                    "performance": exp.performance,
                    "runtime": exp.runtime,
                }
                for exp in self.experiments
            ]
        }
        
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.knowledge_base_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved knowledge base with {len(self.experiments)} experiments")
    
    def load_knowledge_base(self) -> None:
        """Load knowledge base from disk."""
        if not self.knowledge_base_path or not self.knowledge_base_path.exists():
            return
        
        with open(self.knowledge_base_path) as f:
            data = json.load(f)
        
        self.experiments = [
            ExperimentRecord(
                meta_features=MetaFeatures.from_dict(exp["meta_features"]),
                algorithm=exp["algorithm"],
                hyperparameters=exp["hyperparameters"],
                performance=exp["performance"],
                runtime=exp["runtime"],
            )
            for exp in data["experiments"]
        ]
        
        logger.info(f"Loaded knowledge base with {len(self.experiments)} experiments")


class MetaLearner:
    """High-level meta-learning coordinator."""
    
    def __init__(self, knowledge_base_path: Path | None = None) -> None:
        """Initialize meta-learner.
        
        Args:
            knowledge_base_path: Path to knowledge base
        """
        self.extractor = MetaFeatureExtractor()
        self.recommender = AlgorithmRecommender()
        self.warm_starter = WarmStarter(knowledge_base_path)
    
    def analyze_dataset(self, X: np.ndarray, y: np.ndarray | None = None) -> MetaFeatures:
        """Analyze dataset and extract meta-features.
        
        Args:
            X: Features
            y: Labels (optional)
        
        Returns:
            Meta-features
        """
        return self.extractor.extract(X, y)
    
    def get_recommendations(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        top_k: int = 3,
    ) -> list[tuple[str, float]]:
        """Get algorithm recommendations for dataset.
        
        Args:
            X: Features
            y: Labels
            top_k: Number of algorithms to recommend
        
        Returns:
            Algorithm recommendations with confidence scores
        """
        meta_features = self.analyze_dataset(X, y)
        return self.recommender.recommend(meta_features, top_k)
    
    def get_warm_start_configs(
        self,
        X: np.ndarray,
        y: np.ndarray | None,
        algorithm: str,
        n_configs: int = 10,
    ) -> list[dict[str, Any]]:
        """Get warm-start configurations for algorithm.
        
        Args:
            X: Features
            y: Labels
            algorithm: Algorithm name
            n_configs: Number of configurations
        
        Returns:
            Hyperparameter configurations
        """
        meta_features = self.analyze_dataset(X, y)
        return self.warm_starter.get_warm_start_configs(
            meta_features,
            algorithm,
            n_configs,
        )
