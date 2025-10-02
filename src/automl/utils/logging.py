"""Structured logging utilities for AutoML."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

__all__ = ["setup_logging", "get_logger", "log_experiment", "log_metrics"]


def setup_logging(
    level: str | int = "INFO",
    log_file: Path | str | None = None,
    format_string: str | None = None,
) -> None:
    """Configure structured logging for AutoML.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"
        )
    
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Configure root logger
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True,
    )
    
    # Silence noisy libraries
    logging.getLogger("optuna").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module.
    
    Args:
        name: Name of the logger (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_experiment(
    logger: logging.Logger,
    run_id: str,
    config: dict[str, Any],
) -> None:
    """Log experiment configuration.
    
    Args:
        logger: Logger instance
        run_id: Unique run identifier
        config: Experiment configuration
    """
    logger.info("=" * 80)
    logger.info(f"Starting experiment: {run_id}")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config}")


def log_metrics(
    logger: logging.Logger,
    metrics: dict[str, float],
    prefix: str = "",
) -> None:
    """Log metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metric name -> value
        prefix: Optional prefix for metric names
    """
    logger.info(f"{prefix}Metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.6f}")
        else:
            logger.info(f"  {name}: {value}")
