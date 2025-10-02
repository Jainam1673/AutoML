# 🚀 AutoML - State-of-the-Art Automated Machine Learning

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A **hyper-modern**, **over-engineered**, **cutting-edge** AutoML platform that combines the latest technologies in machine learning, hyperparameter optimization, and distributed computing. Built for Python 3.13+ with the `uv` packaging ecosystem.

## ✨ Features

### 🎯 Core Capabilities

- **🔥 State-of-the-Art Optimization**
  - Optuna with TPE, CMA-ES, NSGA-II samplers
  - Hyperband and Successive Halving pruning
  - Multi-objective optimization support
  - Bayesian optimization with Gaussian Processes

- **🤖 Advanced Model Support**
  - XGBoost, LightGBM, CatBoost (GPU-enabled)
  - Scikit-learn ensemble methods
  - Neural Architecture Search (NAS)
  - Automated ensemble & stacking
  - Custom model registration

- **⚡ High-Performance Computing**
  - GPU acceleration for gradient boosting
  - Distributed optimization with Ray Tune
  - Parallel cross-validation
  - Multi-process hyperparameter search

- **🔧 Feature Engineering**
  - Polynomial features & interactions
  - Target encoding for categoricals
  - Robust & power transformations
  - Automated feature selection
  - Missing value imputation strategies

- **📊 Model Explainability**
  - SHAP values for any model
  - LIME for local interpretability
  - Feature importance analysis
  - Model interpretation dashboards

- **🎨 Beautiful CLI & UI**
  - Rich terminal interface with progress bars
  - Real-time optimization monitoring
  - Configuration validation
  - Experiment tracking integration

### 🏭 Production-Ready Exabyte-Scale Features

- **🌐 Distributed Computing**
  - Ray Tune for distributed hyperparameter search (1000s of trials in parallel)
  - Dask for out-of-core computation (datasets larger than RAM)
  - Cloud storage integration (S3, GCS, Azure Blob)
  - Horizontal scaling across clusters

- **📊 Experiment Tracking & Registry**
  - MLflow for experiment tracking and model versioning
  - Centralized artifact storage (S3/GCS)
  - Model lifecycle management (Staging → Production)
  - A/B testing support

- **⚡ Production Model Serving**
  - FastAPI REST API with async support
  - Redis distributed caching (10-100x speedup)
  - Batch prediction endpoints
  - Horizontal autoscaling with Kubernetes

- **🔍 Monitoring & Observability**
  - Prometheus metrics collection
  - Grafana dashboards for visualization
  - Model drift detection (KS test)
  - Performance degradation alerts
  - Real-time health checks

- **💾 Data Processing at Scale**
  - Streaming data support for infinite datasets
  - Chunked processing for exabyte-scale files
  - Incremental model training
  - Parquet/Arrow for columnar efficiency

- **🔌 Extensible Architecture**
  - Plugin system for custom components
  - Event-driven instrumentation
  - Type-safe configuration with Pydantic
  - Thread-safe component registry

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Jainam1673/AutoML.git
cd AutoML

# Install with uv (fastest, recommended)
uv pip install -r requirements.txt

# Or install with pip
pip install -r requirements.txt

# Verify installation
python -c "import sys; sys.path.insert(0, 'src'); from automl import __version__; print(f'✅ AutoML {__version__}')"
```

> 📖 **Detailed installation guide:** See [INSTALL.md](INSTALL.md) for complete instructions, troubleshooting, and GPU setup.

**What gets installed:** 164 packages including scikit-learn, xgboost, lightgbm, catboost, optuna, mlflow, streamlit, fastapi, and more.

**Requirements:** Python 3.13+

### Basic Usage

```bash
# Run AutoML with a configuration file
python -m automl run --config configs/iris_classification.yaml

# Or use the CLI directly
automl run --config configs/iris_classification.yaml

# Validate configuration
automl validate configs/iris_classification.yaml

# Display system information
automl info

# Show version
automl version
```

### Python API Example

```python
from automl.core.engine import default_engine
from automl.core.config import (
    AutoMLConfig,
    DatasetConfig,
    PipelineConfig,
    ModelConfig,
    OptimizerConfig,
    PreprocessorConfig,
)

# Create configuration
config = AutoMLConfig(
    run_name="iris_classification",
    dataset=DatasetConfig(name="iris"),
    pipeline=PipelineConfig(
        preprocessors=[
            PreprocessorConfig(name="standard_scaler"),
            PreprocessorConfig(name="pca", params={"n_components": 3}),
        ],
        model=ModelConfig(
            name="xgboost_classifier",
            base_params={"n_estimators": 100},
            search_space=[
                {"max_depth": 5, "learning_rate": 0.1},
                {"max_depth": 10, "learning_rate": 0.05},
            ],
        ),
    ),
    optimizer=OptimizerConfig(
        name="optuna",
        cv_folds=5,
        scoring="accuracy",
        params={"n_trials": 50, "sampler": "tpe"},
    ),
)

# Run optimization
engine = default_engine()
results = engine.run(config)

print(f"Best Score: {results['best_score']:.4f}")
print(f"Best Parameters: {results['best_params']}")
```

## 📦 Dependencies

### Core Dependencies
- **NumPy 2.1+** - High-performance numerical computing
- **Pandas 2.2+** - Data manipulation and analysis
- **Scikit-learn 1.5+** - Machine learning algorithms
- **Optuna 4.1+** - Hyperparameter optimization
- **XGBoost 2.1+** - Gradient boosting framework
- **LightGBM 4.5+** - Fast gradient boosting
- **CatBoost 1.2+** - Gradient boosting with categorical support
- **Pydantic 2.10+** - Data validation and settings
- **Rich 13.9+** - Beautiful terminal output
- **Typer 0.15+** - Modern CLI framework

### Optional Dependencies

```bash
# GPU acceleration
uv sync --extra gpu  # PyTorch, CUDA, cuML

# Distributed computing
uv sync --extra distributed  # Ray, Dask

# Vision tasks
uv sync --extra vision  # timm, torchvision, albumentations

# NLP tasks
uv sync --extra nlp  # transformers, sentence-transformers

# Time series
uv sync --extra timeseries  # Prophet, NeuralProphet, Darts

# AutoGluon integration
uv sync --extra autogluon

# REST API
uv sync --extra api  # FastAPI, Redis, Celery

# Everything
uv sync --extra all
```

## 📁 Project Structure

```
automl/
├── src/automl/
│   ├── core/              # Core engine and orchestration
│   │   ├── engine.py      # Main AutoML engine
│   │   ├── config.py      # Pydantic configuration models
│   │   ├── events.py      # Event system for monitoring
│   │   └── registry.py    # Component registry
│   ├── datasets/          # Dataset providers
│   │   ├── base.py        # Dataset abstractions
│   │   └── builtin.py     # Built-in datasets
│   ├── models/            # Model factories
│   │   ├── sklearn.py     # Scikit-learn models
│   │   ├── boosting.py    # XGBoost, LightGBM, CatBoost
│   │   └── ensemble.py    # Ensemble strategies
│   ├── optimizers/        # Hyperparameter optimizers
│   │   ├── random_search.py    # Random search
│   │   └── optuna_optimizer.py # Optuna-based optimizers
│   ├── pipelines/         # Pipeline builders
│   │   ├── sklearn.py     # Scikit-learn preprocessing
│   │   └── advanced.py    # Advanced feature engineering
│   ├── explainability/    # Model interpretation
│   │   └── __init__.py    # SHAP, LIME, feature importance
│   └── cli.py             # Command-line interface
├── configs/               # Example configurations
│   ├── iris_classification.yaml
│   ├── advanced_ensemble.yaml
│   └── gpu_accelerated.yaml
├── examples/              # Usage examples
│   └── complete_workflow.py
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
└── pyproject.toml        # Project metadata

```

## 🎯 Configuration Examples

### Basic Classification
```yaml
run_name: "iris_classification"

dataset:
  name: "iris"

pipeline:
  preprocessors:
    - name: "standard_scaler"
  
  model:
    name: "xgboost_classifier"
    base_params:
      n_estimators: 100
    search_space:
      - max_depth: 5
        learning_rate: 0.1
      - max_depth: 10
        learning_rate: 0.05

optimizer:
  name: "optuna"
  cv_folds: 5
  scoring: "accuracy"
  params:
    n_trials: 50
```

### GPU-Accelerated
```yaml
run_name: "gpu_boosting"

pipeline:
  model:
    name: "catboost_classifier"
    base_params:
      task_type: "GPU"
      devices: "0"
      iterations: 1000

optimizer:
  params:
    n_trials: 200
    sampler: "cmaes"
    n_jobs: 4
```

## 🧪 Model Explainability

```python
from automl.explainability import create_explainer

# Create SHAP explainer
explainer = create_explainer(
    model=trained_model,
    method="shap",
    background_data=X_train,
)

# Explain single prediction
explanation = explainer.explain_instance(X_test[0])

# Get global feature importance
global_importance = explainer.explain_global()
```

## 🔧 Advanced Features

### Event-Based Monitoring
```python
from automl.core.events import CandidateEvaluated

def on_candidate_evaluated(event: CandidateEvaluated):
    print(f"Trial {event.candidate_index}: {event.score:.4f}")

engine.instrumentation.events.subscribe(
    CandidateEvaluated, 
    on_candidate_evaluated
)
```

### Custom Components
```python
# Register custom preprocessor
engine.register_preprocessor(
    "my_scaler",
    my_scaler_factory,
    description="Custom scaling strategy"
)

# Register custom model
engine.register_model(
    "my_model",
    my_model_factory,
    description="Custom model implementation"
)
```

## 🏗️ Development

```bash
# Install development dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run ruff format src/

# Type checking
uv run mypy src/

# Build documentation
uv sync --group docs
uv run mkdocs serve
```

## 📊 Benchmarks

Coming soon: Performance comparisons with AutoGluon, TPOT, and H2O AutoML.

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and code of conduct.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with love using cutting-edge technologies:
- **Optuna** - Hyperparameter optimization framework
- **XGBoost, LightGBM, CatBoost** - Gradient boosting frameworks
- **Ray** - Distributed computing
- **SHAP** - Model explainability
- **Rich** - Beautiful terminal UI
- **Pydantic** - Data validation
- **uv** - Blazing fast Python package manager

## 📮 Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ and over-engineering**
