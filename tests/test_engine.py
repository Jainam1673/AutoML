"""Basic tests for AutoML engine."""

import pytest
from automl.core.engine import AutoMLEngine, default_engine
from automl.core.config import AutoMLConfig


def test_engine_initialization():
    """Test that engine can be initialized."""
    engine = AutoMLEngine()
    assert engine is not None
    assert hasattr(engine, 'datasets')
    assert hasattr(engine, 'models')
    assert hasattr(engine, 'optimizers')
    assert hasattr(engine, 'preprocessors')


def test_default_engine_has_components():
    """Test that default engine has registered components."""
    engine = default_engine()
    
    # Check datasets
    assert 'iris' in engine.datasets._items
    
    # Check models (should have 18+)
    assert 'logistic_regression' in engine.models._items
    assert 'xgboost_classifier' in engine.models._items
    assert 'lightgbm_classifier' in engine.models._items
    assert 'catboost_classifier' in engine.models._items
    assert 'auto_ensemble_classifier' in engine.models._items
    
    # Check optimizers (should have 3)
    assert 'random_search' in engine.optimizers._items
    assert 'optuna' in engine.optimizers._items
    assert 'optuna_multiobjective' in engine.optimizers._items
    
    # Check preprocessors (should have 6+)
    assert 'standard_scaler' in engine.preprocessors._items
    assert 'robust_scaler' in engine.preprocessors._items
    assert 'power_transformer' in engine.preprocessors._items


def test_config_parsing():
    """Test that config can be parsed."""
    config_dict = {
        "dataset": {"name": "iris"},
        "pipeline": {
            "preprocessors": [],
            "model": {
                "name": "logistic_regression",
                "base_params": {},
                "search_space": [{}]
            }
        },
        "optimizer": {
            "name": "random_search",
            "params": {"n_iterations": 10},
            "cv_folds": 3,
            "scoring": "accuracy"
        }
    }
    
    config = AutoMLConfig(**config_dict)
    assert config.dataset.name == "iris"
    assert config.pipeline.model.name == "logistic_regression"
    assert config.optimizer.name == "random_search"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
