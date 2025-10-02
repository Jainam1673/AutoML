"""Production monitoring with Prometheus metrics and model drift detection.

Comprehensive observability for AutoML systems at exabyte scale.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

__all__ = [
    "MetricsCollector",
    "DriftDetector",
    "PerformanceMonitor",
]

logger = logging.getLogger(__name__)


@dataclass
class MetricsCollector:
    """Collect Prometheus metrics for model serving.
    
    Tracks:
    - Request counts
    - Latency percentiles
    - Error rates
    - Prediction distributions
    - Cache hit rates
    """
    
    registry: Any = field(default_factory=lambda: CollectorRegistry() if PROMETHEUS_AVAILABLE else None)
    
    def __post_init__(self) -> None:
        """Initialize Prometheus metrics."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not installed, metrics disabled")
            return
        
        # Request metrics
        self.request_count = Counter(
            "automl_requests_total",
            "Total number of prediction requests",
            ["model_id", "status"],
            registry=self.registry,
        )
        
        self.prediction_count = Counter(
            "automl_predictions_total",
            "Total number of predictions made",
            ["model_id"],
            registry=self.registry,
        )
        
        # Latency metrics
        self.request_latency = Histogram(
            "automl_request_latency_seconds",
            "Request latency in seconds",
            ["model_id"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )
        
        # Error metrics
        self.error_count = Counter(
            "automl_errors_total",
            "Total number of errors",
            ["model_id", "error_type"],
            registry=self.registry,
        )
        
        # Prediction metrics
        self.prediction_value = Summary(
            "automl_prediction_value",
            "Distribution of prediction values",
            ["model_id"],
            registry=self.registry,
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            "automl_cache_hits_total",
            "Total cache hits",
            ["model_id"],
            registry=self.registry,
        )
        
        self.cache_misses = Counter(
            "automl_cache_misses_total",
            "Total cache misses",
            ["model_id"],
            registry=self.registry,
        )
        
        # Model performance
        self.model_score = Gauge(
            "automl_model_score",
            "Current model performance score",
            ["model_id", "metric"],
            registry=self.registry,
        )
    
    def record_request(
        self,
        model_id: str,
        latency: float,
        status: str = "success",
        num_predictions: int = 1,
    ) -> None:
        """Record a prediction request.
        
        Args:
            model_id: Model identifier
            latency: Request latency in seconds
            status: Request status (success/error)
            num_predictions: Number of predictions in request
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.request_count.labels(model_id=model_id, status=status).inc()
        self.prediction_count.labels(model_id=model_id).inc(num_predictions)
        self.request_latency.labels(model_id=model_id).observe(latency)
    
    def record_error(
        self,
        model_id: str,
        error_type: str,
    ) -> None:
        """Record an error.
        
        Args:
            model_id: Model identifier
            error_type: Type of error
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.error_count.labels(model_id=model_id, error_type=error_type).inc()
    
    def record_predictions(
        self,
        model_id: str,
        predictions: np.ndarray,
    ) -> None:
        """Record prediction values.
        
        Args:
            model_id: Model identifier
            predictions: Prediction values
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        for pred in predictions:
            self.prediction_value.labels(model_id=model_id).observe(float(pred))
    
    def record_cache_hit(self, model_id: str) -> None:
        """Record cache hit."""
        if PROMETHEUS_AVAILABLE:
            self.cache_hits.labels(model_id=model_id).inc()
    
    def record_cache_miss(self, model_id: str) -> None:
        """Record cache miss."""
        if PROMETHEUS_AVAILABLE:
            self.cache_misses.labels(model_id=model_id).inc()
    
    def update_model_score(
        self,
        model_id: str,
        metric: str,
        score: float,
    ) -> None:
        """Update model performance score.
        
        Args:
            model_id: Model identifier
            metric: Metric name (e.g., "accuracy", "f1")
            score: Score value
        """
        if PROMETHEUS_AVAILABLE:
            self.model_score.labels(model_id=model_id, metric=metric).set(score)
    
    def export_metrics(self) -> bytes:
        """Export metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        if not PROMETHEUS_AVAILABLE:
            return b""
        
        return generate_latest(self.registry)


@dataclass
class DriftDetector:
    """Detect data drift and model performance degradation.
    
    Critical for production ML systems to detect when models
    need retraining due to changing data distributions.
    """
    
    window_size: int = 1000  # Number of recent predictions to track
    threshold: float = 0.1  # Drift detection threshold
    
    def __post_init__(self) -> None:
        """Initialize drift detection."""
        self.reference_distribution: np.ndarray | None = None
        self.recent_predictions: deque = deque(maxlen=self.window_size)
        self.recent_features: deque = deque(maxlen=self.window_size)
    
    def set_reference_distribution(
        self,
        predictions: np.ndarray,
        features: np.ndarray | None = None,
    ) -> None:
        """Set reference distribution for drift detection.
        
        Call this with predictions/features from training or validation set.
        
        Args:
            predictions: Reference predictions
            features: Reference features (optional)
        """
        self.reference_distribution = predictions
        logger.info(f"Set reference distribution with {len(predictions)} samples")
    
    def add_prediction(
        self,
        prediction: float,
        features: np.ndarray | None = None,
    ) -> None:
        """Add new prediction to tracking.
        
        Args:
            prediction: Model prediction
            features: Input features (optional)
        """
        self.recent_predictions.append(prediction)
        if features is not None:
            self.recent_features.append(features)
    
    def detect_prediction_drift(self) -> tuple[bool, float]:
        """Detect drift in prediction distribution.
        
        Uses Kolmogorov-Smirnov test to compare recent predictions
        with reference distribution.
        
        Returns:
            (drift_detected, drift_score) tuple
        """
        if self.reference_distribution is None:
            logger.warning("No reference distribution set")
            return False, 0.0
        
        if len(self.recent_predictions) < 100:
            # Need minimum samples
            return False, 0.0
        
        # Convert to array
        recent = np.array(list(self.recent_predictions))
        
        # Compute KS statistic
        from scipy import stats
        statistic, pvalue = stats.ks_2samp(
            self.reference_distribution,
            recent,
        )
        
        drift_detected = statistic > self.threshold
        
        if drift_detected:
            logger.warning(
                f"Prediction drift detected! KS statistic: {statistic:.4f}, "
                f"p-value: {pvalue:.4f}"
            )
        
        return drift_detected, float(statistic)
    
    def detect_feature_drift(self) -> dict[str, Any]:
        """Detect drift in feature distributions.
        
        Returns:
            Dictionary with drift information per feature
        """
        if len(self.recent_features) < 100:
            return {}
        
        # This is a simplified version
        # In production, you'd track reference feature distributions too
        recent_features = np.vstack(list(self.recent_features))
        
        drift_info = {}
        for i in range(recent_features.shape[1]):
            feature_values = recent_features[:, i]
            
            # Simple drift detection: check if mean/std changed significantly
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            
            drift_info[f"feature_{i}"] = {
                "mean": float(mean),
                "std": float(std),
            }
        
        return drift_info


@dataclass
class PerformanceMonitor:
    """Monitor model performance in production.
    
    Tracks real-time metrics and alerts on degradation.
    """
    
    window_size: int = 1000
    alert_threshold: float = 0.05  # Alert if performance drops > 5%
    
    def __post_init__(self) -> None:
        """Initialize performance monitoring."""
        self.baseline_score: float | None = None
        self.recent_scores: deque = deque(maxlen=self.window_size)
        self.alert_count = 0
    
    def set_baseline(self, score: float) -> None:
        """Set baseline performance score.
        
        Args:
            score: Baseline score from validation
        """
        self.baseline_score = score
        logger.info(f"Set baseline performance: {score:.4f}")
    
    def add_score(self, score: float) -> None:
        """Add new performance score.
        
        Args:
            score: Performance score
        """
        self.recent_scores.append(score)
    
    def check_degradation(self) -> tuple[bool, float]:
        """Check for performance degradation.
        
        Returns:
            (degradation_detected, current_score) tuple
        """
        if self.baseline_score is None or len(self.recent_scores) < 10:
            return False, 0.0
        
        current_score = float(np.mean(list(self.recent_scores)))
        degradation = self.baseline_score - current_score
        
        if degradation > self.alert_threshold:
            self.alert_count += 1
            logger.warning(
                f"Performance degradation detected! "
                f"Baseline: {self.baseline_score:.4f}, "
                f"Current: {current_score:.4f}, "
                f"Drop: {degradation:.4f}"
            )
            return True, current_score
        
        return False, current_score
    
    def get_statistics(self) -> dict[str, float]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance stats
        """
        if not self.recent_scores:
            return {}
        
        scores = np.array(list(self.recent_scores))
        
        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "baseline": self.baseline_score or 0.0,
            "alert_count": self.alert_count,
        }
