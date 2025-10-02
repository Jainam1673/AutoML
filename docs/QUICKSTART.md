# 🚀 AutoML Quick Start Guide

## Installation

### Basic Installation
```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .
```

### With Optional Features
```bash
# GPU support (XGBoost, LightGBM, CatBoost with CUDA)
uv sync --extra gpu

# All features
uv sync --extra all
```

## Usage Examples

### 1. Command-Line Interface

#### Basic Run
```bash
# Run with configuration file
uv run automl run --config configs/iris_classification.yaml

# With GPU acceleration
uv run automl run --config configs/gpu_accelerated.yaml --gpu

# With custom settings
uv run automl run \
  --config configs/iris_classification.yaml \
  --n-trials 100 \
  --n-jobs -1 \
  --verbose
```

#### Validate Configuration
```bash
uv run automl validate configs/iris_classification.yaml
```

#### System Information
```bash
uv run automl info
```

### 2. Python API - Basic Usage

```python
from automl.core.engine import default_engine
from automl.core.config import (
    AutoMLConfig, DatasetConfig, PipelineConfig,
    ModelConfig, OptimizerConfig, PreprocessorConfig
)

# Create engine
engine = default_engine()

# Configure experiment
config = AutoMLConfig(
    run_name="my_experiment",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[
            PreprocessorConfig(name="standard_scaler")
        ],
        model=ModelConfig(
            name="random_forest_classifier",
            base_params={"random_state": 42},
            search_space=[
                {"n_estimators": 100, "max_depth": 5},
                {"n_estimators": 200, "max_depth": 10},
            ]
        )
    ),
    optimizer=OptimizerConfig(
        name="random_search",
        cv_folds=5,
        scoring="accuracy",
        params={"max_trials": 20}
    )
)

# Run optimization
results = engine.run(config)
print(f"Best Score: {results['best_score']:.4f}")
```

### 3. Advanced Optimization with Optuna

```python
from automl.optimizers.optuna_optimizer import OptunaOptimizer, OptunaSettings

# Register Optuna optimizer
engine.register_optimizer(
    "optuna",
    lambda params=None: OptunaOptimizer(
        event_bus=engine.instrumentation.events,
        settings=OptunaSettings(**(params or {}))
    )
)

# Use Optuna with TPE sampler
config = AutoMLConfig(
    run_name="optuna_experiment",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        model=ModelConfig(
            name="xgboost_classifier",
            search_space=[
                {"max_depth": 3, "learning_rate": 0.1},
                {"max_depth": 5, "learning_rate": 0.05},
                {"max_depth": 10, "learning_rate": 0.01},
            ]
        )
    ),
    optimizer=OptimizerConfig(
        name="optuna",
        params={
            "n_trials": 50,
            "sampler": "tpe",  # Tree-structured Parzen Estimator
            "pruner": "hyperband",
            "n_jobs": -1
        }
    )
)

results = engine.run(config)
```

### 4. GPU-Accelerated Models

```python
from automl.models.boosting import (
    xgboost_classifier,
    lightgbm_classifier,
    catboost_classifier
)

# Register GPU-enabled XGBoost
engine.register_model(
    "xgboost_gpu",
    lambda params=None: xgboost_classifier({
        **(params or {}),
        "tree_method": "gpu_hist",
        "device": "cuda"
    })
)

# Register GPU-enabled LightGBM
engine.register_model(
    "lightgbm_gpu",
    lambda params=None: lightgbm_classifier({
        **(params or {}),
        "device": "gpu"
    })
)

# Register GPU-enabled CatBoost
engine.register_model(
    "catboost_gpu",
    lambda params=None: catboost_classifier({
        **(params or {}),
        "task_type": "GPU"
    })
)

# Use GPU model
config = AutoMLConfig(
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        model=ModelConfig(name="xgboost_gpu")
    ),
    optimizer=OptimizerConfig(name="optuna")
)
```

### 5. Advanced Preprocessing

```python
from automl.pipelines.advanced import (
    robust_scaler,
    power_transformer,
    polynomial_features,
    AutoFeatureEngineer
)

# Register advanced preprocessors
engine.register_preprocessor("robust_scaler", robust_scaler)
engine.register_preprocessor("power_transformer", power_transformer)
engine.register_preprocessor("polynomial_features", polynomial_features)

# Use in pipeline
config = AutoMLConfig(
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[
            PreprocessorConfig(name="robust_scaler"),
            PreprocessorConfig(
                name="polynomial_features",
                params={"degree": 2, "interaction_only": True}
            ),
        ],
        model=ModelConfig(name="logistic_regression")
    ),
    optimizer=OptimizerConfig(name="random_search")
)
```

### 6. Ensemble Learning

```python
from automl.models.ensemble import (
    AutoEnsembleClassifier,
    create_stacking_ensemble
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Create auto-ensemble
def auto_ensemble_factory(params=None):
    return AutoEnsembleClassifier(
        ensemble_strategy="stacking",
        optimize_weights=True,
        cv=5,
        n_jobs=-1
    )

engine.register_model("auto_ensemble", auto_ensemble_factory)

# Use ensemble in AutoML
config = AutoMLConfig(
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        model=ModelConfig(name="auto_ensemble")
    ),
    optimizer=OptimizerConfig(name="optuna")
)
```

### 7. Model Explainability

```python
from automl.explainability import create_explainer
from automl.datasets.builtin import iris_dataset

# Load data and train model
dataset = iris_dataset()
X, y = dataset.features, dataset.target

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Create SHAP explainer
explainer = create_explainer(
    model=model,
    method="shap",
    background_data=X[:100]  # Sample for SHAP
)

# Explain single instance
instance_explanation = explainer.explain_instance(X[0])
print("Feature Importance:", instance_explanation["feature_importance"])

# Get global explanations
global_explanation = explainer.explain_global(X)
print("Global Importance:", global_explanation["feature_importance"])

# Feature importance (tree-based)
importance_explainer = create_explainer(
    model=model,
    method="importance"
)
importance = importance_explainer.explain_global()
print("Native Importance:", importance["feature_importance"])
```

### 8. Event-Based Monitoring

```python
from automl.core.events import (
    RunStarted, RunCompleted, CandidateEvaluated
)

# Define event handlers
def on_run_started(event):
    print(f"🚀 Run {event.run_id} started")
    print(f"   Config hash: {event.config_hash}")

def on_candidate_evaluated(event):
    print(f"📊 Candidate {event.candidate_index}: {event.score:.4f}")

def on_run_completed(event):
    print(f"✅ Run {event.run_id} completed")
    print(f"   Best score: {event.best_score:.4f}")
    print(f"   Candidates evaluated: {event.candidate_count}")

# Subscribe to events
engine.instrumentation.events.subscribe(RunStarted, on_run_started)
engine.instrumentation.events.subscribe(CandidateEvaluated, on_candidate_evaluated)
engine.instrumentation.events.subscribe(RunCompleted, on_run_completed)

# Run with monitoring
results = engine.run(config)
```

### 9. Multi-Objective Optimization

```python
from automl.optimizers.optuna_optimizer import (
    MultiObjectiveOptunaOptimizer,
    MultiObjectiveSettings
)

# Register multi-objective optimizer
engine.register_optimizer(
    "multi_objective",
    lambda params=None: MultiObjectiveOptunaOptimizer(
        settings=MultiObjectiveSettings(
            objectives=["accuracy", "f1_weighted"],
            n_trials=100,
            **(params or {})
        )
    )
)

# Use multi-objective optimization
config = AutoMLConfig(
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        model=ModelConfig(name="random_forest_classifier")
    ),
    optimizer=OptimizerConfig(
        name="multi_objective",
        scoring=["accuracy", "f1_weighted"]
    )
)

results = engine.run(config)
```

### 10. Custom Components

```python
# Register custom dataset
def my_dataset():
    from automl.datasets.base import DatasetBundle
    import numpy as np
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 3, 100)
    
    return DatasetBundle(features=X, target=y)

engine.register_dataset("my_dataset", my_dataset)

# Register custom preprocessor
from sklearn.preprocessing import MinMaxScaler

def minmax_scaler(params=None):
    return MinMaxScaler(**(params or {}))

engine.register_preprocessor("minmax", minmax_scaler)

# Register custom model
from sklearn.svm import SVC

def svc_model(params=None):
    defaults = {"kernel": "rbf", "C": 1.0}
    if params:
        defaults.update(params)
    return SVC(**defaults)

engine.register_model("svc", svc_model)

# Use custom components
config = AutoMLConfig(
    dataset=DatasetConfig(name="my_dataset"),
    pipeline=PipelineConfig(
        preprocessors=[PreprocessorConfig(name="minmax")],
        model=ModelConfig(name="svc")
    ),
    optimizer=OptimizerConfig(name="optuna")
)
```

## Configuration File Format

### Basic Configuration (YAML)

```yaml
run_name: "my_experiment"

dataset:
  name: "iris"

pipeline:
  preprocessors:
    - name: "standard_scaler"
      params:
        with_mean: true
        with_std: true
    
    - name: "pca"
      params:
        n_components: 3

  model:
    name: "xgboost_classifier"
    base_params:
      n_estimators: 100
      random_state: 42
    
    search_space:
      - max_depth: 3
        learning_rate: 0.1
      - max_depth: 5
        learning_rate: 0.05
      - max_depth: 10
        learning_rate: 0.01

optimizer:
  name: "optuna"
  cv_folds: 5
  scoring: "accuracy"
  params:
    n_trials: 50
    sampler: "tpe"
    pruner: "hyperband"
    n_jobs: -1
```

## Tips & Best Practices

### 1. Choose the Right Optimizer
- **Random Search**: Quick baseline, small search spaces
- **Optuna TPE**: General purpose, good for most cases
- **Optuna CMA-ES**: Continuous parameter spaces
- **Multi-Objective**: When optimizing multiple metrics

### 2. Preprocessing Pipeline
- Always scale features for linear models
- Use robust scaling for outlier-heavy data
- Consider polynomial features for non-linear relationships
- Use PCA for dimensionality reduction

### 3. Model Selection
- **Random Forest**: Good baseline, handles mixed data
- **XGBoost/LightGBM**: Best for tabular data
- **CatBoost**: Excellent for categorical features
- **Ensembles**: When you need the best performance

### 4. GPU Acceleration
- Enable for large datasets (>10K samples)
- CatBoost generally fastest on GPU
- XGBoost good for very large datasets
- Ensure CUDA is properly installed

### 5. Cross-Validation
- Use 5-10 folds for most datasets
- Use stratified folds for imbalanced data
- More folds = more reliable but slower

### 6. Search Space Design
- Start broad, then narrow
- Include sensible defaults
- Test 10-20 configurations minimum
- Use logarithmic scales for learning rates

## Troubleshooting

### Import Errors
```bash
# Install missing dependencies
uv sync --extra all

# Or specific features
uv sync --extra gpu
```

### GPU Not Detected
```python
# Check CUDA availability
import torch
print(torch.cuda.is_available())

# Verify XGBoost GPU
import xgboost as xgb
print(xgb.get_config()['use_cuda'])
```

### Slow Performance
- Enable parallel processing (`n_jobs=-1`)
- Use GPU acceleration
- Reduce search space size
- Use early stopping
- Sample large datasets

## Next Steps

1. **Explore Examples**: Check `examples/complete_workflow.py`
2. **Read Documentation**: See `docs/FEATURES.md`
3. **Try Configurations**: Use configs in `configs/`
4. **Extend**: Add custom models and preprocessors
5. **Contribute**: Submit PRs for new features!

---

**Happy AutoML-ing! 🚀**
