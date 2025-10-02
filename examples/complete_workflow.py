"""
AutoML Advanced Example: Complete Workflow
===========================================

This example demonstrates the full capabilities of the cutting-edge AutoML platform.
"""

# Import necessary libraries
import numpy as np
from automl.core.engine import default_engine
from automl.core.config import (
    AutoMLConfig,
    DatasetConfig,
    PipelineConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessorConfig,
)

# ============================================================================
# Example 1: Basic AutoML Workflow
# ============================================================================

print("=" * 80)
print("Example 1: Basic AutoML with Random Forest")
print("=" * 80)

# Create configuration
config = AutoMLConfig(
    run_name="basic_iris_rf",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[
            PreprocessorConfig(name="standard_scaler"),
        ],
        model=ModelConfig(
            name="random_forest_classifier",
            base_params={"n_estimators": 100, "random_state": 42},
            search_space=[
                {"max_depth": 5, "min_samples_split": 2},
                {"max_depth": 10, "min_samples_split": 5},
                {"max_depth": 15, "min_samples_split": 10},
            ],
        ),
    ),
    optimizer=OptimizerConfig(
        name="random_search",
        cv_folds=5,
        scoring="accuracy",
        params={"max_trials": 10, "random_seed": 42},
    ),
)

# Run AutoML
engine = default_engine()
results = engine.run(config)

print(f"\nBest Score: {results['best_score']:.4f}")
print(f"Best Parameters: {results['best_params']}")
print(f"Total Candidates: {len(results['candidates'])}")

# ============================================================================
# Example 2: Advanced Optimization with Optuna
# ============================================================================

print("\n" + "=" * 80)
print("Example 2: Advanced Optimization with Optuna TPE")
print("=" * 80)

# Register Optuna optimizer
from automl.optimizers.optuna_optimizer import OptunaOptimizer, OptunaSettings

engine.register_optimizer(
    "optuna",
    lambda params=None: OptunaOptimizer(
        event_bus=engine.instrumentation.events,
        settings=OptunaSettings(**(params or {})),
    ),
    description="Optuna-based hyperparameter optimization with TPE",
)

config_optuna = AutoMLConfig(
    run_name="optuna_iris_xgb",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[
            PreprocessorConfig(name="standard_scaler"),
            PreprocessorConfig(name="pca", params={"n_components": 3}),
        ],
        model=ModelConfig(
            name="random_forest_classifier",
            base_params={"random_state": 42},
            search_space=[
                {"n_estimators": 100, "max_depth": 5},
                {"n_estimators": 200, "max_depth": 10},
                {"n_estimators": 300, "max_depth": 15},
                {"n_estimators": 500, "max_depth": 20},
            ],
        ),
    ),
    optimizer=OptimizerConfig(
        name="optuna",
        cv_folds=5,
        scoring="accuracy",
        params={
            "n_trials": 20,
            "sampler": "tpe",
            "pruner": "hyperband",
            "n_jobs": -1,
            "show_progress_bar": True,
        },
    ),
)

results_optuna = engine.run(config_optuna)

print(f"\nBest Score: {results_optuna['best_score']:.4f}")
print(f"Best Parameters: {results_optuna['best_params']}")

# ============================================================================
# Example 3: Model Explainability
# ============================================================================

print("\n" + "=" * 80)
print("Example 3: Model Explainability with SHAP")
print("=" * 80)

# Load dataset for explanation
from automl.datasets.builtin import iris_dataset

dataset = iris_dataset()
X, y = dataset.features, dataset.target

# Train a model
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create explainer
from automl.explainability import create_explainer

explainer = create_explainer(
    model=model,
    method="importance",  # Use 'shap' for SHAP values (requires shap package)
    background_data=X,
)

# Get global explanations
global_explanation = explainer.explain_global()
print("\nFeature Importance:")
for feature, importance in list(global_explanation["feature_importance"].items())[:5]:
    print(f"  {feature}: {importance:.4f}")

# ============================================================================
# Example 4: Event-Based Monitoring
# ============================================================================

print("\n" + "=" * 80)
print("Example 4: Event-Based Monitoring")
print("=" * 80)

from automl.core.events import CandidateEvaluated, RunCompleted

# Subscribe to events
def on_candidate_evaluated(event: CandidateEvaluated):
    print(f"  Candidate {event.candidate_index}: Score = {event.score:.4f}")

def on_run_completed(event: RunCompleted):
    print(f"\n✓ Run completed! Best score: {event.best_score:.4f}")
    print(f"  Total candidates: {event.candidate_count}")

engine.instrumentation.events.subscribe(CandidateEvaluated, on_candidate_evaluated)
engine.instrumentation.events.subscribe(RunCompleted, on_run_completed)

# Run with monitoring
config_monitored = AutoMLConfig(
    run_name="monitored_run",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        model=ModelConfig(
            name="logistic_regression",
            search_space=[
                {"C": 0.1, "penalty": "l2"},
                {"C": 1.0, "penalty": "l2"},
                {"C": 10.0, "penalty": "l2"},
            ],
        ),
    ),
    optimizer=OptimizerConfig(
        name="random_search",
        cv_folds=3,
        scoring="accuracy",
        params={"max_trials": 5},
    ),
)

print("\nRunning with event monitoring...")
results_monitored = engine.run(config_monitored)

# ============================================================================
# Example 5: Custom Component Registration
# ============================================================================

print("\n" + "=" * 80)
print("Example 5: Custom Component Registration")
print("=" * 80)

# Register custom preprocessor
from sklearn.preprocessing import MinMaxScaler

def minmax_scaler(params=None):
    return MinMaxScaler(**(params or {}))

engine.register_preprocessor(
    "minmax_scaler",
    minmax_scaler,
    description="Min-Max feature scaling to [0, 1]",
)

# Register custom model
from sklearn.neighbors import KNeighborsClassifier

def knn_classifier(params=None):
    defaults = {"n_neighbors": 5, "weights": "uniform"}
    if params:
        defaults.update(params)
    return KNeighborsClassifier(**defaults)

engine.register_model(
    "knn_classifier",
    knn_classifier,
    description="K-Nearest Neighbors classifier",
)

# Use custom components
config_custom = AutoMLConfig(
    run_name="custom_components",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[PreprocessorConfig(name="minmax_scaler")],
        model=ModelConfig(
            name="knn_classifier",
            search_space=[
                {"n_neighbors": 3, "weights": "uniform"},
                {"n_neighbors": 5, "weights": "distance"},
                {"n_neighbors": 7, "weights": "uniform"},
            ],
        ),
    ),
    optimizer=OptimizerConfig(
        name="random_search",
        cv_folds=5,
        scoring="accuracy",
        params={"max_trials": 5},
    ),
)

results_custom = engine.run(config_custom)
print(f"\nCustom Model Best Score: {results_custom['best_score']:.4f}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("Summary of Results")
print("=" * 80)

all_results = [
    ("Basic Random Forest", results),
    ("Optuna TPE", results_optuna),
    ("Monitored Run", results_monitored),
    ("Custom Components", results_custom),
]

for name, res in all_results:
    print(f"{name:.<40} {res['best_score']:.4f}")

print("\n✨ All examples completed successfully!")
print("=" * 80)
