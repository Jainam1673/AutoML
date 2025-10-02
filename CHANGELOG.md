# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of AutoML framework
- Core engine with factory and registry patterns
- Event-driven architecture with pub/sub system
- Pydantic-based configuration system
- Advanced hyperparameter optimization with Optuna
  - TPE, CMA-ES, QMCS, NSGA-II samplers
  - Hyperband and successive halving pruning
  - Multi-objective optimization support
- GPU-enabled gradient boosting models
  - XGBoost with GPU support
  - LightGBM with GPU support
  - CatBoost with automatic categorical handling
- Advanced preprocessing pipelines
  - Robust scalers and transformers
  - Automated feature engineering
  - Time series feature extraction
  - Polynomial features and interactions
- Ensemble strategies
  - Voting ensembles (hard/soft)
  - Stacking ensembles with meta-learners
  - Auto-ensemble with intelligent model selection
- Model explainability layer
  - SHAP integration (Tree, Kernel, Linear, Deep)
  - LIME for local interpretability
  - Feature importance analysis
- Production utilities
  - Structured logging system
  - Model serialization with versioning
  - Comprehensive data validation
- Beautiful CLI with Rich and Typer
  - Progress bars and spinners
  - Colored output and tables
  - Config validation command
  - System info command
- Comprehensive documentation
  - README with quick start
  - Feature documentation
  - Quick start guide
  - Contributing guidelines
  - Security policy

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- Pydantic v2 compatibility issues
- Type safety issues in ensemble models
- Missing imports for collections.abc types
- Data validation edge cases

### Security
- Added security policy
- Input validation for all user-provided data
- Safe model serialization practices

## [0.1.0] - 2025-01-XX

### Added
- Initial public release
- Core AutoML functionality
- 18 registered models
- 6 preprocessors
- 3 optimizers
- Production-ready utilities
- Comprehensive documentation

[Unreleased]: https://github.com/Jainam1673/AutoML/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Jainam1673/AutoML/releases/tag/v0.1.0
