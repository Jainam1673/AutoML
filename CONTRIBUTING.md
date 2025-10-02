# Contributing to AutoML

Thank you for your interest in contributing to AutoML! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Jainam1673/AutoML.git
   cd AutoML
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync --all-extras
   ```

4. **Set up pre-commit hooks** (optional)
   ```bash
   uv run pre-commit install
   ```

## Code Style

We follow strict Python coding standards:

- **Type hints**: All functions must have complete type annotations
- **Docstrings**: Use Google-style docstrings for all public APIs
- **Formatting**: Code is automatically formatted with `ruff`
- **Linting**: All code must pass `ruff check` and `mypy` type checking

### Running Code Quality Checks

```bash
# Format code
uv run ruff format src/

# Lint code
uv run ruff check src/

# Type check
uv run mypy src/automl --ignore-missing-imports
```

## Testing

We aim for high test coverage:

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=automl --cov-report=html

# Run specific test file
uv run pytest tests/test_engine.py -v
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, well-documented code
   - Add tests for new functionality
   - Update documentation as needed

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
   
   We follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation changes
   - `refactor:` - Code refactoring
   - `test:` - Adding tests
   - `chore:` - Maintenance tasks

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Ensure all CI checks pass

## Adding New Features

### Adding a New Model

1. Create model factory in `src/automl/models/`
2. Follow the `ModelFactory` protocol
3. Register in `default_engine()` in `src/automl/core/engine.py`
4. Add tests in `tests/models/`
5. Update documentation

Example:
```python
def my_model_classifier(params: Mapping[str, Any] | None = None) -> BaseEstimator:
    """Create my custom classifier."""
    from my_library import MyClassifier
    
    defaults = {
        "param1": 100,
        "param2": 0.1,
    }
    return MyClassifier(**_merge_params(defaults, params))
```

### Adding a New Optimizer

1. Create optimizer class in `src/automl/optimizers/`
2. Inherit from `Optimizer` base class
3. Implement `optimize()` method
4. Register in `default_engine()`
5. Add tests and documentation

### Adding a New Preprocessor

1. Create preprocessor factory in `src/automl/pipelines/`
2. Follow the `PreprocessorFactory` protocol
3. Register in `default_engine()`
4. Add tests and documentation

## Documentation

- Keep `README.md` up to date
- Update `docs/FEATURES.md` when adding features
- Add examples to `docs/QUICKSTART.md`
- Maintain inline docstrings

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Focus on what is best for the community

## Getting Help

- Open an issue for bug reports or feature requests
- Join discussions in GitHub Discussions
- Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to AutoML! 🎉
