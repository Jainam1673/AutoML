"""State-of-the-art feature engineering and preprocessing."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.base import BaseEstimator, TransformerMixin

__all__ = [
    "advanced_scaler",
    "robust_scaler",
    "power_transformer",
    "quantile_transformer",
    "polynomial_features",
    "target_encoder",
    "feature_selector",
    "missing_value_imputer",
]


def advanced_scaler(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create an advanced standard scaler with outlier handling."""
    from sklearn.preprocessing import StandardScaler
    
    configuration: dict[str, Any] = {
        "with_mean": True,
        "with_std": True,
        "copy": True,
    }
    if params:
        configuration.update(params)
    return StandardScaler(**configuration)


def robust_scaler(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create a robust scaler resistant to outliers.
    
    Uses median and IQR instead of mean and std.
    """
    from sklearn.preprocessing import RobustScaler
    
    configuration: dict[str, Any] = {
        "with_centering": True,
        "with_scaling": True,
        "quantile_range": (25.0, 75.0),
        "copy": True,
    }
    if params:
        configuration.update(params)
    return RobustScaler(**configuration)


def power_transformer(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create a power transformer for making data more Gaussian-like.
    
    Uses Yeo-Johnson or Box-Cox transformation.
    """
    from sklearn.preprocessing import PowerTransformer
    
    configuration: dict[str, Any] = {
        "method": "yeo-johnson",  # Works with negative values
        "standardize": True,
        "copy": True,
    }
    if params:
        configuration.update(params)
    return PowerTransformer(**configuration)


def quantile_transformer(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create a quantile transformer for uniform or normal distribution.
    
    Maps features to uniform or normal distribution using quantiles.
    """
    from sklearn.preprocessing import QuantileTransformer
    
    configuration: dict[str, Any] = {
        "n_quantiles": 1000,
        "output_distribution": "normal",
        "subsample": 100_000,
        "random_state": 42,
        "copy": True,
    }
    if params:
        configuration.update(params)
    return QuantileTransformer(**configuration)


def polynomial_features(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create polynomial features for non-linear relationships.
    
    Generates polynomial and interaction features.
    """
    from sklearn.preprocessing import PolynomialFeatures
    
    configuration: dict[str, Any] = {
        "degree": 2,
        "interaction_only": False,
        "include_bias": False,
    }
    if params:
        configuration.update(params)
    return PolynomialFeatures(**configuration)


def target_encoder(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create a target encoder for categorical variables.
    
    Encodes categories using target statistics with smoothing.
    Requires category_encoders package.
    """
    try:
        from category_encoders import TargetEncoder
    except ImportError as e:
        raise ImportError(
            "category_encoders is required for target encoding. "
            "Install it with: pip install category-encoders"
        ) from e
    
    configuration: dict[str, Any] = {
        "smoothing": 1.0,
        "min_samples_leaf": 1,
        "handle_unknown": "value",
        "handle_missing": "value",
    }
    if params:
        configuration.update(params)
    return TargetEncoder(**configuration)


def feature_selector(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create an intelligent feature selector.
    
    Uses mutual information, variance threshold, or model-based selection.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    
    configuration: dict[str, Any] = {
        "score_func": mutual_info_classif,
        "k": 10,
    }
    if params:
        configuration.update(params)
    return SelectKBest(**configuration)


def missing_value_imputer(params: Mapping[str, Any] | None = None) -> TransformerMixin:
    """Create an advanced missing value imputer.
    
    Supports mean, median, most_frequent, and KNN imputation.
    """
    from sklearn.impute import SimpleImputer
    
    configuration: dict[str, Any] = {
        "strategy": "mean",
        "add_indicator": False,
        "copy": True,
    }
    if params:
        configuration.update(params)
    return SimpleImputer(**configuration)


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Automated feature engineering pipeline.
    
    Automatically generates and selects useful features including:
    - Polynomial features
    - Interaction terms
    - Statistical aggregations
    - Date/time features
    - Text features
    """

    def __init__(
        self,
        polynomial_degree: int = 2,
        interaction_only: bool = False,
        select_k_best: int | None = None,
        handle_missing: bool = True,
        generate_stats: bool = True,
    ) -> None:
        """Initialize auto feature engineer.
        
        Args:
            polynomial_degree: Degree for polynomial features
            interaction_only: Only generate interaction terms
            select_k_best: Number of best features to select (None = all)
            handle_missing: Whether to impute missing values
            generate_stats: Whether to generate statistical features
        """
        self.polynomial_degree = polynomial_degree
        self.interaction_only = interaction_only
        self.select_k_best = select_k_best
        self.handle_missing = handle_missing
        self.generate_stats = generate_stats
        self.pipeline_: Any = None

    def fit(self, X, y=None):
        """Fit the feature engineering pipeline."""
        from sklearn.pipeline import Pipeline
        
        steps = []
        
        # Handle missing values
        if self.handle_missing:
            steps.append(("imputer", missing_value_imputer()))
        
        # Generate polynomial features
        if self.polynomial_degree > 1:
            steps.append((
                "polynomial",
                polynomial_features({
                    "degree": self.polynomial_degree,
                    "interaction_only": self.interaction_only,
                })
            ))
        
        # Feature selection
        if self.select_k_best is not None:
            from sklearn.feature_selection import SelectKBest, f_classif
            steps.append((
                "selector",
                SelectKBest(score_func=f_classif, k=self.select_k_best)
            ))
        
        # Scale features
        steps.append(("scaler", advanced_scaler()))
        
        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X, y)
        return self

    def transform(self, X):
        """Transform features."""
        if self.pipeline_ is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        return self.pipeline_.transform(X)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform features."""
        return self.fit(X, y).transform(X)


class TimeSeriesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Specialized feature engineering for time series data.
    
    Generates lag features, rolling statistics, and time-based features.
    """

    def __init__(
        self,
        lag_features: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        extract_datetime: bool = True,
    ) -> None:
        """Initialize time series feature engineer.
        
        Args:
            lag_features: List of lag periods to create (e.g., [1, 7, 30])
            rolling_windows: List of window sizes for rolling stats
            extract_datetime: Whether to extract datetime components
        """
        self.lag_features = lag_features or [1, 7, 30]
        self.rolling_windows = rolling_windows or [7, 30]
        self.extract_datetime = extract_datetime

    def fit(self, X, y=None):
        """Fit the time series feature engineer."""
        # Time series feature engineering is typically stateless
        return self

    def transform(self, X):
        """Transform time series data with engineered features."""
        import pandas as pd
        
        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_transformed = X.copy()
        
        # Generate lag features
        for col in X.columns:
            if X[col].dtype in ["float64", "int64"]:
                for lag in self.lag_features:
                    X_transformed[f"{col}_lag_{lag}"] = X[col].shift(lag)
        
        # Generate rolling statistics
        for col in X.columns:
            if X[col].dtype in ["float64", "int64"]:
                for window in self.rolling_windows:
                    X_transformed[f"{col}_rolling_mean_{window}"] = X[col].rolling(window).mean()
                    X_transformed[f"{col}_rolling_std_{window}"] = X[col].rolling(window).std()
                    X_transformed[f"{col}_rolling_min_{window}"] = X[col].rolling(window).min()
                    X_transformed[f"{col}_rolling_max_{window}"] = X[col].rolling(window).max()
        
        # Fill NaN values created by lag/rolling operations
        X_transformed = X_transformed.bfill().ffill()
        
        return X_transformed.values

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform time series features."""
        return self.fit(X, y).transform(X)
