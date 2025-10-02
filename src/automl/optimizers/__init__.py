"""Optimization backends for hyperparameter, architecture, and hardware-aware search.

Provides cutting-edge optimization strategies:
- random_search: Simple random search baseline
- optuna_optimizer: Advanced optimization with Optuna (TPE, CMA-ES, NSGA-II)
"""

from __future__ import annotations

from . import optuna_optimizer, random_search

__all__ = ["random_search", "optuna_optimizer"]
