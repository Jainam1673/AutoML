"""Tests for data validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from automl.validation import DataValidator, AutoFixer, AnomalyDetector


def test_data_validator_missing_values() -> None:
    """Test missing value detection."""
    # Create data with missing values
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [5, np.nan, 7, 8],
    })
    
    validator = DataValidator(missing_threshold=0.3)
    report = validator.validate(df)
    
    assert report.n_missing > 0
    assert report.missing_percentage > 0
    assert not report.is_clean


def test_data_validator_duplicates() -> None:
    """Test duplicate detection."""
    df = pd.DataFrame({
        "a": [1, 2, 1, 4],
        "b": [5, 6, 5, 8],
    })
    
    validator = DataValidator()
    report = validator.validate(df)
    
    assert report.n_duplicates > 0
    assert report.duplicate_percentage > 0


def test_auto_fixer_missing_values() -> None:
    """Test automatic missing value fixing."""
    df = pd.DataFrame({
        "numeric": [1.0, 2.0, np.nan, 4.0],
        "category": ["a", "b", None, "a"],
    })
    
    fixer = AutoFixer(strategy="moderate")
    fixed = fixer.fix(df)
    
    # Should have no missing values
    assert fixed["numeric"].isna().sum() == 0
    assert fixed["category"].isna().sum() == 0


def test_anomaly_detector() -> None:
    """Test anomaly detection."""
    # Create data with outliers
    X = np.random.randn(100, 2)
    X = np.vstack([X, [[10, 10], [-10, -10]]])  # Add outliers
    
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(X)
    
    is_normal = detector.predict(X)
    
    # Should detect some anomalies
    assert is_normal.sum() < len(X)


def test_data_quality_report() -> None:
    """Test data quality reporting."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5, 6, 7, 8],
    })
    
    validator = DataValidator()
    report = validator.validate(df)
    
    # Clean data should have high quality score
    assert report.quality_score > 80
    assert report.is_clean
    
    # Report should be convertible to dict
    report_dict = report.to_dict()
    assert "quality_score" in report_dict
    assert "n_samples" in report_dict
