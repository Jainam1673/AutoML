# 📦 Installation Guide

## Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/Jainam1673/AutoML.git
cd AutoML

# Install with uv (fastest method)
uv pip install -r requirements.txt

# Or install with pip
pip install -r requirements.txt

# Verify installation
python -c "import sys; sys.path.insert(0, 'src'); from automl import __version__; print(f'✅ AutoML {__version__} installed successfully!')"
```

## What Gets Installed

### Core Scientific Computing (10 packages)
- **numpy** 2.3.3 - Numerical computing
- **pandas** 2.3.3 - Data manipulation
- **scikit-learn** 1.7.2 - Machine learning
- **scipy** 1.16.2 - Scientific computing

### ML Frameworks (3 packages)
- **xgboost** 3.0.5 - Gradient boosting
- **lightgbm** 4.6.0 - Fast gradient boosting
- **catboost** 1.2.8 - Gradient boosting with categorical support

### Optimization & Config (5 packages)
- **optuna** 4.5.0 - Hyperparameter optimization
- **pydantic** 2.11.9 - Data validation
- **hydra-core** 1.3.2 - Configuration management
- **omegaconf** 2.3.0 - YAML configuration
- **pyyaml** 6.0.3 - YAML parser

### MLOps & Tracking (2 packages)
- **mlflow** 2.20.5 - Experiment tracking
- **shap** 0.48.0 - Model explainability

### UI & Visualization (4 packages)
- **streamlit** 1.50.0 - Interactive dashboards
- **plotly** 6.0.0 - Interactive plots
- **rich** 14.1.0 - Terminal UI
- **typer** 0.19.2 - CLI framework

### API & Production (4 packages)
- **fastapi** 0.119.2 - REST API
- **uvicorn** 0.37.0 - ASGI server
- **redis** 5.2.1 - Caching
- **prometheus-client** 0.22.0 - Monitoring

### Security (1 package)
- **cryptography** 45.0.7 - Model encryption

### Testing & Quality (5 packages)
- **pytest** 8.4.0 - Testing framework
- **pytest-cov** 6.1.0 - Coverage reports
- **hypothesis** 6.132.0 - Property testing
- **ruff** 0.13.2 - Linting & formatting
- **mypy** 1.17.1 - Type checking

### Additional Tools (6 packages)
- **openml** 0.16.0 - Dataset repository
- **cloudpickle** 3.1.1 - Serialization
- **joblib** 1.5.2 - Parallel processing
- **tqdm** 4.67.1 - Progress bars
- **click** 8.3.0 - CLI utilities
- **pyyaml** 6.0.3 - YAML support

**Total: 164 packages** (including dependencies)

## Python Version Requirements

- **Python 3.13+** required
- Tested on Python 3.13.7

## Platform Support

✅ **Linux** - Fully supported
✅ **macOS** - Fully supported  
✅ **Windows** - Supported via WSL2 or native

## Installation Methods

### Method 1: Using uv (Fastest ⚡)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

### Method 2: Using pip (Traditional)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Method 3: Using conda

```bash
# Create conda environment
conda create -n automl python=3.13
conda activate automl

# Install dependencies
pip install -r requirements.txt
```

## Verify Installation

```bash
# Test core imports
python -c "
import sys
sys.path.insert(0, 'src')
from automl import __version__
from automl.core import AutoMLEngine
from automl.validation import DataValidator
from automl.security import ModelEncryption
from automl.dashboard import AutoMLDashboard
from automl.benchmarks import BenchmarkSuite
print('✅ All imports successful!')
print(f'AutoML version: {__version__}')
"
```

## Optional: GPU Support

**Note:** GPU packages (torch, CUDA libraries) are not yet compatible with Python 3.13.  
Use Python 3.12 if you need GPU acceleration:

```bash
# Switch to Python 3.12
conda create -n automl-gpu python=3.12
conda activate automl-gpu

# Install base requirements
pip install -r requirements.txt

# Install GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Troubleshooting

### ImportError: No module named 'automl'

Make sure to add `src` to Python path:
```python
import sys
sys.path.insert(0, 'src')
from automl import AutoMLEngine
```

Or install in development mode:
```bash
pip install -e .
```

### Package conflicts

If you encounter dependency conflicts, try:
```bash
# Clear cache
uv cache clean
# or
pip cache purge

# Reinstall
uv pip install --force-reinstall -r requirements.txt
```

### Python version issues

Check your Python version:
```bash
python --version  # Should be 3.13+
```

If using wrong version:
```bash
# Use specific Python
python3.13 -m pip install -r requirements.txt
```

## Development Installation

For contributors:
```bash
# Install all dependencies including dev tools
pip install -r requirements.txt

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# Run tests
pytest tests/

# Run linting
ruff check src/
ruff format src/

# Type checking
mypy src/
```

## Updating Dependencies

To update all packages to latest versions:
```bash
# With uv
uv pip install --upgrade -r requirements.txt
uv pip freeze > requirements.txt

# With pip
pip install --upgrade -r requirements.txt
pip freeze > requirements.txt
```

## Uninstall

```bash
# Remove all packages
pip uninstall -r requirements.txt -y

# Or just delete virtual environment
rm -rf .venv
```

## Support

For installation issues, please:
1. Check this guide first
2. Search existing [GitHub Issues](https://github.com/Jainam1673/AutoML/issues)
3. Create a new issue with:
   - Your Python version
   - Your OS
   - Error messages
   - Output of `pip list` or `uv pip list`

---

**Made with ❤️ using Python 3.13 and uv**
