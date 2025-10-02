"""State-of-the-art model explainability and interpretability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np

__all__ = [
    "ExplainerProtocol",
    "SHAPExplainer",
    "ShapExplainer",
    "LIMEExplainer",
    "FeatureImportanceExplainer",
    "create_explainer",
]


class ExplainerProtocol(Protocol):
    """Protocol for model explainers."""

    def explain_instance(
        self,
        instance: np.ndarray,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Explain a single prediction."""
        ...

    def explain_global(self, **kwargs: Any) -> Mapping[str, Any]:
        """Provide global model explanations."""
        ...


@dataclass
class ExplanationResult:
    """Container for explanation results."""

    feature_importance: Mapping[str, float]
    explanation_type: str
    metadata: Mapping[str, Any]


class SHAPExplainer:
    """SHAP (SHapley Additive exPlanations) explainer.
    
    Provides both local and global explanations using game-theoretic
    Shapley values. Works with any model type.
    """

    def __init__(
        self,
        model: Any,
        background_data: np.ndarray | None = None,
        algorithm: str = "auto",
    ) -> None:
        """Initialize SHAP explainer.
        
        Args:
            model: Trained model to explain
            background_data: Background dataset for computing SHAP values
            algorithm: SHAP algorithm ('tree', 'kernel', 'deep', 'linear', 'auto')
        """
        self.model = model
        self.background_data = background_data
        self.algorithm = algorithm
        self.explainer_: Any = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize the appropriate SHAP explainer."""
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "shap is required for SHAP explanations. "
                "Install it with: pip install shap"
            ) from e

        if self.algorithm == "auto":
            # Automatically detect model type
            if hasattr(self.model, "tree_"):
                self.explainer_ = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "coef_"):
                self.explainer_ = shap.LinearExplainer(self.model, self.background_data)
            else:
                self.explainer_ = shap.KernelExplainer(
                    self.model.predict,
                    self.background_data,
                )
        elif self.algorithm == "tree":
            self.explainer_ = shap.TreeExplainer(self.model)
        elif self.algorithm == "kernel":
            self.explainer_ = shap.KernelExplainer(
                self.model.predict,
                self.background_data,
            )
        elif self.algorithm == "linear":
            self.explainer_ = shap.LinearExplainer(self.model, self.background_data)
        elif self.algorithm == "deep":
            self.explainer_ = shap.DeepExplainer(self.model, self.background_data)
        else:
            raise ValueError(f"Unknown SHAP algorithm: {self.algorithm}")

    def explain_instance(
        self,
        instance: np.ndarray,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Explain a single prediction using SHAP values.
        
        Args:
            instance: Single instance to explain
            **kwargs: Additional arguments for SHAP explainer
            
        Returns:
            Dictionary containing SHAP values and visualization data
        """
        if self.explainer_ is None:
            raise ValueError("Explainer not initialized")

        # Compute SHAP values
        shap_values = self.explainer_.shap_values(instance.reshape(1, -1))

        # Extract feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            feature_importance = {
                f"feature_{i}": float(np.abs(shap_values[0][0][i]))
                for i in range(len(shap_values[0][0]))
            }
        else:
            feature_importance = {
                f"feature_{i}": float(np.abs(shap_values[0][i]))
                for i in range(len(shap_values[0]))
            }

        return {
            "shap_values": shap_values,
            "feature_importance": feature_importance,
            "base_value": self.explainer_.expected_value,
            "method": "SHAP",
        }

    def explain_global(
        self,
        X: np.ndarray | None = None,
        max_samples: int = 100,
    ) -> Mapping[str, Any]:
        """Provide global explanations using SHAP values.
        
        Args:
            X: Dataset to compute global explanations (uses background if None)
            max_samples: Maximum number of samples to use
            
        Returns:
            Dictionary containing global feature importance
        """
        if self.explainer_ is None:
            raise ValueError("Explainer not initialized")

        data = X if X is not None else self.background_data
        if data is None:
            raise ValueError("No data available for global explanation")

        # Limit samples for computational efficiency
        if len(data) > max_samples:
            indices = np.random.choice(len(data), max_samples, replace=False)
            data = data[indices]

        # Compute SHAP values
        shap_values = self.explainer_.shap_values(data)

        # Aggregate to global importance
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            global_importance = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values],
                axis=0,
            )
        else:
            global_importance = np.abs(shap_values).mean(axis=0)

        feature_importance = {
            f"feature_{i}": float(global_importance[i])
            for i in range(len(global_importance))
        }

        return {
            "feature_importance": feature_importance,
            "method": "SHAP Global",
            "samples_used": len(data),
        }


class LIMEExplainer:
    """LIME (Local Interpretable Model-agnostic Explanations) explainer.
    
    Explains individual predictions by fitting local linear models.
    """

    def __init__(
        self,
        model: Any,
        training_data: np.ndarray,
        mode: str = "classification",
        feature_names: list[str] | None = None,
    ) -> None:
        """Initialize LIME explainer.
        
        Args:
            model: Trained model to explain
            training_data: Training data for LIME
            mode: 'classification' or 'regression'
            feature_names: Optional feature names
        """
        self.model = model
        self.training_data = training_data
        self.mode = mode
        self.feature_names = feature_names
        self.explainer_: Any = None
        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        """Initialize LIME explainer."""
        try:
            from lime import lime_tabular
        except ImportError as e:
            raise ImportError(
                "lime is required for LIME explanations. "
                "Install it with: pip install lime"
            ) from e

        if self.mode == "classification":
            self.explainer_ = lime_tabular.LimeTabularExplainer(
                self.training_data,
                mode="classification",
                feature_names=self.feature_names,
            )
        else:
            self.explainer_ = lime_tabular.LimeTabularExplainer(
                self.training_data,
                mode="regression",
                feature_names=self.feature_names,
            )

    def explain_instance(
        self,
        instance: np.ndarray,
        num_features: int = 10,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Explain a single prediction using LIME.
        
        Args:
            instance: Single instance to explain
            num_features: Number of features to include in explanation
            **kwargs: Additional arguments for LIME
            
        Returns:
            Dictionary containing LIME explanation
        """
        if self.explainer_ is None:
            raise ValueError("Explainer not initialized")

        # Get prediction function
        if self.mode == "classification":
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict

        # Generate explanation
        explanation = self.explainer_.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
            **kwargs,
        )

        # Extract feature importance
        feature_importance = dict(explanation.as_list())

        return {
            "feature_importance": feature_importance,
            "intercept": explanation.intercept[0] if hasattr(explanation, "intercept") else None,
            "score": explanation.score if hasattr(explanation, "score") else None,
            "method": "LIME",
        }

    def explain_global(self, **kwargs: Any) -> Mapping[str, Any]:
        """LIME is designed for local explanations only."""
        return {
            "message": "LIME is designed for local explanations. Use explain_instance() instead.",
            "method": "LIME",
        }


class FeatureImportanceExplainer:
    """Tree-based feature importance explainer.
    
    Extracts native feature importance from tree-based models.
    """

    def __init__(self, model: Any, feature_names: list[str] | None = None) -> None:
        """Initialize feature importance explainer.
        
        Args:
            model: Trained tree-based model
            feature_names: Optional feature names
        """
        self.model = model
        self.feature_names = feature_names

    def explain_instance(self, instance: np.ndarray, **kwargs: Any) -> Mapping[str, Any]:
        """Feature importance is global, not instance-specific."""
        return {
            "message": "Feature importance is a global metric. Use explain_global() instead.",
            "method": "Feature Importance",
        }

    def explain_global(self, **kwargs: Any) -> Mapping[str, Any]:
        """Get feature importance from model.
        
        Returns:
            Dictionary containing feature importance scores
        """
        # Extract feature importance
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_).flatten()
        else:
            raise ValueError("Model does not have feature_importances_ or coef_ attribute")

        # Create feature names if not provided
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        feature_importance = {
            name: float(importance)
            for name, importance in zip(feature_names, importances)
        }

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )

        return {
            "feature_importance": feature_importance,
            "method": "Native Feature Importance",
        }


def create_explainer(
    model: Any,
    method: str = "shap",
    background_data: np.ndarray | None = None,
    **kwargs: Any,
) -> SHAPExplainer | LIMEExplainer | FeatureImportanceExplainer:
    """Factory function to create explainers.
    
    Args:
        model: Trained model to explain
        method: Explanation method ('shap', 'lime', 'importance')
        background_data: Background data for explainer
        **kwargs: Additional arguments for explainer
        
    Returns:
        Initialized explainer
    """
    if method.lower() == "shap":
        return SHAPExplainer(model, background_data, **kwargs)
    elif method.lower() == "lime":
        if background_data is None:
            raise ValueError("background_data required for LIME")
        return LIMEExplainer(model, background_data, **kwargs)
    elif method.lower() == "importance":
        return FeatureImportanceExplainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown explainer method: {method}")


# Alias for common typo/convention
ShapExplainer = SHAPExplainer
