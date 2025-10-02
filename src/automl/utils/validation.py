"""Data validation and quality check utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "validate_features_target",
    "check_missing_values",
    "check_data_types",
    "check_target_distribution",
    "validate_config",
    "DataValidator",
]


def validate_features_target(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
) -> tuple[bool, list[str]]:
    """Validate feature matrix and target vector.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check shapes
    if len(X) != len(y):
        issues.append(
            f"Shape mismatch: X has {len(X)} samples, y has {len(y)} samples"
        )
    
    # Check for empty data
    if len(X) == 0:
        issues.append("Feature matrix is empty")
    
    if len(y) == 0:
        issues.append("Target vector is empty")
    
    # Check for all NaN features
    if isinstance(X, pd.DataFrame):
        all_nan_cols = X.columns[X.isna().all()].tolist()
        if all_nan_cols:
            issues.append(f"Features with all NaN values: {all_nan_cols}")
    elif isinstance(X, np.ndarray):
        all_nan_cols = [i for i in range(X.shape[1]) if np.isnan(X[:, i]).all()]
        if all_nan_cols:
            issues.append(f"Feature indices with all NaN values: {all_nan_cols}")
    
    # Check for infinite values
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number])
        has_inf = np.isinf(numeric_cols).any(axis=0)
        inf_cols = [col for col, has in zip(numeric_cols.columns, has_inf) if has]
        if inf_cols:
            issues.append(f"Features with infinite values: {inf_cols}")
    elif isinstance(X, np.ndarray):
        if np.isinf(X).any():
            issues.append("Feature matrix contains infinite values")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def check_missing_values(
    X: np.ndarray | pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Check for missing values in data.
    
    Args:
        X: Data to check
        threshold: Threshold for high missing rate (0-1)
        
    Returns:
        Dictionary with missing value statistics
    """
    if isinstance(X, pd.DataFrame):
        missing_counts = X.isna().sum()
        missing_rates = missing_counts / len(X)
        high_missing = missing_rates[missing_rates > threshold].to_dict()
        
        return {
            "total_missing": int(missing_counts.sum()),
            "missing_rate_overall": float(missing_counts.sum() / X.size),
            "features_with_high_missing": high_missing,
            "features_with_any_missing": int((missing_counts > 0).sum()),
        }
    else:
        missing_mask = np.isnan(X)
        missing_counts = missing_mask.sum(axis=0)
        missing_rates = missing_counts / X.shape[0]
        high_missing_indices = np.where(missing_rates > threshold)[0].tolist()
        
        return {
            "total_missing": int(missing_mask.sum()),
            "missing_rate_overall": float(missing_mask.sum() / X.size),
            "feature_indices_with_high_missing": high_missing_indices,
            "features_with_any_missing": int((missing_counts > 0).sum()),
        }


def check_data_types(
    X: pd.DataFrame,
) -> dict[str, Any]:
    """Analyze data types in DataFrame.
    
    Args:
        X: DataFrame to analyze
        
    Returns:
        Dictionary with data type statistics
    """
    if not isinstance(X, pd.DataFrame):
        return {"error": "Input must be a pandas DataFrame"}
    
    dtypes = X.dtypes.value_counts().to_dict()
    
    return {
        "data_types": {str(k): int(v) for k, v in dtypes.items()},
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": X.select_dtypes(include=["object", "category"]).columns.tolist(),
        "datetime_features": X.select_dtypes(include=["datetime64"]).columns.tolist(),
    }


def check_target_distribution(
    y: np.ndarray | pd.Series,
    task: str = "classification",
) -> dict[str, Any]:
    """Analyze target variable distribution.
    
    Args:
        y: Target variable
        task: Task type ('classification' or 'regression')
        
    Returns:
        Dictionary with distribution statistics
    """
    if task == "classification":
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique.tolist(), counts.tolist()))
        
        # Check for class imbalance
        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")
        
        return {
            "n_classes": len(unique),
            "class_distribution": class_distribution,
            "imbalance_ratio": float(imbalance_ratio),
            "is_balanced": imbalance_ratio < 2.0,
        }
    else:
        return {
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "median": float(np.median(y)),
            "q25": float(np.percentile(y, 25)),
            "q75": float(np.percentile(y, 75)),
        }


def validate_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate AutoML configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    required_keys = ["dataset", "pipeline", "optimizer"]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            issues.append(f"Missing required key: {key}")
    
    # Validate dataset config
    if "dataset" in config:
        if "name" not in config["dataset"]:
            issues.append("Dataset config missing 'name' field")
    
    # Validate pipeline config
    if "pipeline" in config:
        if "model" not in config["pipeline"]:
            issues.append("Pipeline config missing 'model' field")
        else:
            if "name" not in config["pipeline"]["model"]:
                issues.append("Model config missing 'name' field")
    
    # Validate optimizer config
    if "optimizer" in config:
        if "name" not in config["optimizer"]:
            issues.append("Optimizer config missing 'name' field")
        
        cv_folds = config["optimizer"].get("cv_folds", 5)
        if not isinstance(cv_folds, int) or cv_folds < 2:
            issues.append(f"cv_folds must be an integer >= 2, got {cv_folds}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


class DataValidator:
    """Comprehensive data validation for AutoML pipelines."""
    
    def __init__(
        self,
        check_missing: bool = True,
        check_types: bool = True,
        check_distribution: bool = True,
        missing_threshold: float = 0.5,
    ):
        """Initialize data validator.
        
        Args:
            check_missing: Whether to check for missing values
            check_types: Whether to check data types
            check_distribution: Whether to check target distribution
            missing_threshold: Threshold for high missing rate
        """
        self.check_missing = check_missing
        self.check_types = check_types
        self.check_distribution = check_distribution
        self.missing_threshold = missing_threshold
    
    def validate(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series | None = None,
        task: str = "classification",
    ) -> dict[str, Any]:
        """Perform comprehensive validation.
        
        Args:
            X: Feature matrix
            y: Target vector (optional)
            task: Task type
            
        Returns:
            Validation report dictionary
        """
        report: dict[str, Any] = {"is_valid": True, "issues": [], "checks": {}}
        
        # Basic validation
        if y is not None:
            is_valid, issues = validate_features_target(X, y)
            report["is_valid"] = report["is_valid"] and is_valid
            report["issues"].extend(issues)
        
        # Missing values check
        if self.check_missing:
            missing_info = check_missing_values(X, self.missing_threshold)
            report["checks"]["missing_values"] = missing_info
            
            if missing_info["missing_rate_overall"] > self.missing_threshold:
                report["issues"].append(
                    f"High overall missing rate: {missing_info['missing_rate_overall']:.2%}"
                )
                report["is_valid"] = False
        
        # Data types check
        if self.check_types and isinstance(X, pd.DataFrame):
            types_info = check_data_types(X)
            report["checks"]["data_types"] = types_info
        
        # Target distribution check
        if self.check_distribution and y is not None:
            dist_info = check_target_distribution(y, task)
            report["checks"]["target_distribution"] = dist_info
            
            if task == "classification" and not dist_info.get("is_balanced", True):
                report["issues"].append(
                    f"Class imbalance detected: ratio = {dist_info['imbalance_ratio']:.2f}"
                )
        
        return report
