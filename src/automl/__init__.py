"""Top-level package for the AutoML platform."""

from __future__ import annotations

from ._version import __version__

__all__ = [
    "__version__",
    # Core
    "AutoMLEngine",
    # Validation
    "DataValidator",
    "AutoFixer",
    # Security
    "ModelEncryption",
    "AuditLogger",
    "ComplianceChecker",
    # Ensemble
    "GreedyEnsembleSelection",
    "CaruanaEnsemble",
    # Meta-learning
    "MetaLearner",
    "WarmStarter",
    # Benchmarks
    "BenchmarkSuite",
    "LeaderboardManager",
    # Dashboard
    "AutoMLDashboard",
    "run_dashboard",
]

# Import core components
try:
    from .core import AutoMLEngine
except ImportError:
    pass

# Import validation components
try:
    from .validation import DataValidator, AutoFixer
except ImportError:
    pass

# Import security components
try:
    from .security import ModelEncryption, AuditLogger, ComplianceChecker
except ImportError:
    pass

# Import ensemble components
try:
    from .ensemble.advanced import GreedyEnsembleSelection, CaruanaEnsemble
except ImportError:
    pass

# Import meta-learning components
try:
    from .metalearning import MetaLearner, WarmStarter
except ImportError:
    pass

# Import benchmark components
try:
    from .benchmarks import BenchmarkSuite, LeaderboardManager
except ImportError:
    pass

# Import dashboard components
try:
    from .dashboard import AutoMLDashboard, run_dashboard
except ImportError:
    pass
