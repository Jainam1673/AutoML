"""Comprehensive testing utilities for AutoML.

Provides fixtures, helpers, and utilities for thorough testing.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
)

__all__ = [
    "generate_classification_data",
    "generate_regression_data",
    "generate_multiclass_data",
    "generate_imbalanced_data",
    "generate_high_dimensional_data",
    "generate_sparse_data",
]


def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_features - n_informative,
        n_classes=2,
        random_state=random_state,
    )
    return X, y


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 15,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_informative: Number of informative features
        noise: Noise level
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state,
    )
    return X, y


def generate_multiclass_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic multiclass data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features - 5,
        n_redundant=5,
        n_classes=n_classes,
        random_state=random_state,
    )
    return X, y


def generate_imbalanced_data(
    n_samples: int = 1000,
    imbalance_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate imbalanced classification data.
    
    Args:
        n_samples: Number of samples
        imbalance_ratio: Ratio of minority to majority class
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[1 - imbalance_ratio, imbalance_ratio],
        random_state=random_state,
    )
    return X, y


def generate_high_dimensional_data(
    n_samples: int = 100,
    n_features: int = 1000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate high-dimensional data (n_features > n_samples).
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(50, n_features // 2),
        n_redundant=min(50, n_features // 4),
        random_state=random_state,
    )
    return X, y


def generate_sparse_data(
    n_samples: int = 1000,
    n_features: int = 100,
    density: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sparse data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        density: Fraction of non-zero values
        random_state: Random seed
    
    Returns:
        (X, y) tuple
    """
    from scipy import sparse
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        random_state=random_state,
    )
    
    # Make sparse
    rng = np.random.RandomState(random_state)
    mask = rng.rand(*X.shape) < density
    X = X * mask
    
    return X, y


# Pytest fixtures
@pytest.fixture
def classification_data():
    """Fixture for binary classification data."""
    return generate_classification_data()


@pytest.fixture
def regression_data():
    """Fixture for regression data."""
    return generate_regression_data()


@pytest.fixture
def multiclass_data():
    """Fixture for multiclass data."""
    return generate_multiclass_data()


@pytest.fixture
def imbalanced_data():
    """Fixture for imbalanced data."""
    return generate_imbalanced_data()


@pytest.fixture
def high_dimensional_data():
    """Fixture for high-dimensional data."""
    return generate_high_dimensional_data()
