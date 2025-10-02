"""Advanced data validation with automatic fixing.

Comprehensive data quality checks, anomaly detection, and automatic corrections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "DataValidator",
    "DataQualityReport",
    "AutoFixer",
    "AnomalyDetector",
]

logger = logging.getLogger(__name__)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    
    # Basic stats
    n_samples: int
    n_features: int
    
    # Missing values
    missing_values: dict[str, float]  # feature -> percentage
    total_missing_percentage: float
    
    # Duplicates
    n_duplicate_rows: int
    duplicate_percentage: float
    
    # Outliers
    outlier_features: dict[str, int]  # feature -> count
    total_outliers: int
    
    # Data types
    feature_types: dict[str, str]
    
    # Quality score
    quality_score: float  # 0-100
    
    # Issues found
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    @property
    def n_missing(self) -> int:
        """Total number of missing values across all features."""
        return sum(int(pct * self.n_samples / 100) for pct in self.missing_values.values())
    
    @property
    def missing_percentage(self) -> float:
        """Percentage of missing values (alias for total_missing_percentage)."""
        return self.total_missing_percentage
    
    @property
    def is_clean(self) -> bool:
        """Whether the data has no critical issues."""
        return len(self.issues) == 0 and self.quality_score >= 80.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "missing_values": self.missing_values,
            "total_missing_percentage": self.total_missing_percentage,
            "n_missing": self.n_missing,
            "missing_percentage": self.missing_percentage,
            "n_duplicate_rows": self.n_duplicate_rows,
            "duplicate_percentage": self.duplicate_percentage,
            "outlier_features": self.outlier_features,
            "total_outliers": self.total_outliers,
            "feature_types": self.feature_types,
            "quality_score": self.quality_score,
            "is_clean": self.is_clean,
            "issues": self.issues,
            "warnings": self.warnings,
        }
    
    def __str__(self) -> str:
        """String representation."""
        lines = [
            "=" * 80,
            "DATA QUALITY REPORT",
            "=" * 80,
            f"Dataset: {self.n_samples} samples × {self.n_features} features",
            f"Quality Score: {self.quality_score:.1f}/100",
            "",
            f"Missing Values: {self.total_missing_percentage:.2f}%",
            f"Duplicate Rows: {self.n_duplicate_rows} ({self.duplicate_percentage:.2f}%)",
            f"Outliers Detected: {self.total_outliers}",
        ]
        
        if self.issues:
            lines.extend([
                "",
                "🚨 CRITICAL ISSUES:",
                *[f"  - {issue}" for issue in self.issues],
            ])
        
        if self.warnings:
            lines.extend([
                "",
                "⚠️  WARNINGS:",
                *[f"  - {warning}" for warning in self.warnings],
            ])
        
        lines.append("=" * 80)
        return "\n".join(lines)


class DataValidator:
    """Comprehensive data validation and quality assessment."""
    
    def __init__(
        self,
        missing_threshold: float = 0.5,
        outlier_threshold: float = 3.0,
        duplicate_threshold: float = 0.1,
    ) -> None:
        """Initialize data validator.
        
        Args:
            missing_threshold: Max allowed missing value percentage
            outlier_threshold: Z-score threshold for outliers
            duplicate_threshold: Max allowed duplicate percentage
        """
        self.missing_threshold = missing_threshold
        self.outlier_threshold = outlier_threshold
        self.duplicate_threshold = duplicate_threshold
    
    def validate(self, data: pd.DataFrame) -> DataQualityReport:
        """Validate data and generate quality report.
        
        Args:
            data: Input dataframe
        
        Returns:
            Data quality report
        """
        logger.info(f"Validating dataset: {data.shape}")
        
        # Basic stats
        n_samples, n_features = data.shape
        
        # Missing values
        missing_values = {}
        for col in data.columns:
            missing_pct = (data[col].isna().sum() / len(data)) * 100
            if missing_pct > 0:
                missing_values[col] = missing_pct
        
        total_missing = sum(missing_values.values()) / len(data.columns) if missing_values else 0
        
        # Duplicates
        n_duplicates = data.duplicated().sum()
        duplicate_pct = (n_duplicates / n_samples) * 100
        
        # Outliers (for numeric columns)
        outlier_features = {}
        total_outliers = 0
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outliers = (z_scores > self.outlier_threshold).sum()
            if outliers > 0:
                outlier_features[col] = outliers
                total_outliers += outliers
        
        # Feature types
        feature_types = {str(col): str(dtype) for col, dtype in data.dtypes.items()}
        
        # Quality score
        quality_score = self._compute_quality_score(
            total_missing,
            duplicate_pct,
            total_outliers / (n_samples * n_features) * 100,
        )
        
        # Issues and warnings
        issues = []
        warnings = []
        
        if total_missing > self.missing_threshold * 100:
            issues.append(f"High missing value rate: {total_missing:.1f}%")
        
        if duplicate_pct > self.duplicate_threshold * 100:
            issues.append(f"High duplicate rate: {duplicate_pct:.1f}%")
        
        if total_outliers > n_samples * 0.1:
            warnings.append(f"Many outliers detected: {total_outliers}")
        
        # Check for constant features
        for col in data.columns:
            if data[col].nunique() == 1:
                warnings.append(f"Constant feature: {col}")
        
        report = DataQualityReport(
            n_samples=n_samples,
            n_features=n_features,
            missing_values=missing_values,
            total_missing_percentage=total_missing,
            n_duplicate_rows=n_duplicates,
            duplicate_percentage=duplicate_pct,
            outlier_features=outlier_features,
            total_outliers=total_outliers,
            feature_types=feature_types,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings,
        )
        
        logger.info(f"Validation complete. Quality score: {quality_score:.1f}/100")
        
        return report
    
    def _compute_quality_score(
        self,
        missing_pct: float,
        duplicate_pct: float,
        outlier_pct: float,
    ) -> float:
        """Compute overall quality score (0-100)."""
        score = 100.0
        
        # Penalize missing values
        score -= missing_pct * 0.5
        
        # Penalize duplicates
        score -= duplicate_pct * 0.8
        
        # Penalize outliers (less severe)
        score -= outlier_pct * 0.2
        
        return max(0.0, min(100.0, score))


class AutoFixer:
    """Automatically fix common data issues."""
    
    def __init__(self, strategy: str = "moderate") -> None:
        """Initialize auto-fixer.
        
        Args:
            strategy: "conservative", "moderate", or "aggressive"
        """
        self.strategy = strategy
    
    def fix(self, data: pd.DataFrame, report: DataQualityReport) -> pd.DataFrame:
        """Automatically fix data issues.
        
        Args:
            data: Input dataframe
            report: Data quality report
        
        Returns:
            Fixed dataframe
        """
        fixed_data = data.copy()
        
        logger.info(f"Auto-fixing data with {self.strategy} strategy")
        
        # Remove duplicates
        if report.duplicate_percentage > 1.0:
            fixed_data = fixed_data.drop_duplicates()
            logger.info(f"Removed {report.n_duplicate_rows} duplicate rows")
        
        # Handle missing values
        for col, missing_pct in report.missing_values.items():
            if missing_pct > 80:
                # Drop column if too many missing
                fixed_data = fixed_data.drop(columns=[col])
                logger.info(f"Dropped column {col} (missing: {missing_pct:.1f}%)")
            
            elif missing_pct > 10:
                # Impute missing values
                if fixed_data[col].dtype in [np.float64, np.int64]:
                    # Numeric: use median
                    fixed_data[col].fillna(fixed_data[col].median(), inplace=True)
                else:
                    # Categorical: use mode
                    fixed_data[col].fillna(fixed_data[col].mode()[0], inplace=True)
                
                logger.info(f"Imputed missing values in {col}")
        
        # Handle outliers (if aggressive)
        if self.strategy == "aggressive":
            for col in report.outlier_features:
                if fixed_data[col].dtype in [np.float64, np.int64]:
                    # Clip outliers to ±3 std
                    mean = fixed_data[col].mean()
                    std = fixed_data[col].std()
                    fixed_data[col] = fixed_data[col].clip(mean - 3 * std, mean + 3 * std)
            
            logger.info(f"Clipped outliers in {len(report.outlier_features)} columns")
        
        # Remove constant features
        for col in fixed_data.columns:
            if fixed_data[col].nunique() == 1:
                fixed_data = fixed_data.drop(columns=[col])
                logger.info(f"Dropped constant feature: {col}")
        
        logger.info(f"Auto-fixing complete. Shape: {fixed_data.shape}")
        
        return fixed_data


class AnomalyDetector:
    """Detect anomalies in data."""
    
    def __init__(self, contamination: float = 0.1) -> None:
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies
        """
        self.contamination = contamination
        self.detector_: Any = None
    
    def fit_detect(self, X: np.ndarray) -> np.ndarray:
        """Fit detector and find anomalies.
        
        Args:
            X: Feature matrix
        
        Returns:
            Binary mask (True = anomaly)
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            self.detector_ = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1,
            )
            
            predictions = self.detector_.fit_predict(X)
            anomalies = predictions == -1
            
            n_anomalies = anomalies.sum()
            logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)")
            
            return anomalies
        
        except ImportError:
            logger.warning("sklearn not available for anomaly detection")
            return np.zeros(len(X), dtype=bool)
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for samples.
        
        Args:
            X: Feature matrix
        
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if self.detector_ is None:
            self.fit_detect(X)
        
        scores = -self.detector_.score_samples(X)
        return scores
