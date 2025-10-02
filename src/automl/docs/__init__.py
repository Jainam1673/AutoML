"""Automated documentation generation.

Generates comprehensive documentation from code, docstrings,
and examples.
"""

from __future__ import annotations

import ast
import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DocGenerator",
    "APIDocGenerator",
    "ExampleGenerator",
]

logger = logging.getLogger(__name__)


@dataclass
class FunctionDoc:
    """Documentation for a function."""
    
    name: str
    signature: str
    docstring: str
    parameters: list[tuple[str, str, str]] = field(default_factory=list)  # (name, type, description)
    returns: str = ""
    examples: list[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"### `{self.name}`",
            "",
            f"```python",
            self.signature,
            "```",
            "",
        ]
        
        if self.docstring:
            lines.extend([self.docstring, ""])
        
        if self.parameters:
            lines.append("**Parameters:**")
            lines.append("")
            for name, typ, desc in self.parameters:
                lines.append(f"- `{name}` (*{typ}*): {desc}")
            lines.append("")
        
        if self.returns:
            lines.extend([
                "**Returns:**",
                "",
                self.returns,
                "",
            ])
        
        if self.examples:
            lines.append("**Examples:**")
            lines.append("")
            for example in self.examples:
                lines.append("```python")
                lines.append(example)
                lines.append("```")
                lines.append("")
        
        return "\n".join(lines)


@dataclass
class ClassDoc:
    """Documentation for a class."""
    
    name: str
    docstring: str
    methods: list[FunctionDoc] = field(default_factory=list)
    attributes: list[tuple[str, str, str]] = field(default_factory=list)  # (name, type, description)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"## `{self.name}`",
            "",
        ]
        
        if self.docstring:
            lines.extend([self.docstring, ""])
        
        if self.attributes:
            lines.append("**Attributes:**")
            lines.append("")
            for name, typ, desc in self.attributes:
                lines.append(f"- `{name}` (*{typ}*): {desc}")
            lines.append("")
        
        if self.methods:
            lines.append("**Methods:**")
            lines.append("")
            for method in self.methods:
                lines.append(method.to_markdown())
        
        return "\n".join(lines)


@dataclass
class ModuleDoc:
    """Documentation for a module."""
    
    name: str
    docstring: str
    classes: list[ClassDoc] = field(default_factory=list)
    functions: list[FunctionDoc] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """Convert to Markdown format."""
        lines = [
            f"# Module: `{self.name}`",
            "",
        ]
        
        if self.docstring:
            lines.extend([self.docstring, ""])
        
        if self.classes:
            lines.append("## Classes")
            lines.append("")
            for cls in self.classes:
                lines.append(cls.to_markdown())
                lines.append("")
        
        if self.functions:
            lines.append("## Functions")
            lines.append("")
            for func in self.functions:
                lines.append(func.to_markdown())
                lines.append("")
        
        return "\n".join(lines)


class DocGenerator:
    """Generate documentation from Python code."""
    
    def __init__(self, source_dir: Path, output_dir: Path) -> None:
        """Initialize documentation generator.
        
        Args:
            source_dir: Source code directory
            output_dir: Output documentation directory
        """
        self.source_dir = source_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def parse_function(self, func: Any) -> FunctionDoc:
        """Parse function documentation.
        
        Args:
            func: Function object
        
        Returns:
            Function documentation
        """
        name = func.__name__
        signature = str(inspect.signature(func))
        docstring = inspect.getdoc(func) or ""
        
        # Parse docstring for parameters and returns
        parameters = []
        returns = ""
        
        # Simple docstring parsing (could use docstring_parser library)
        lines = docstring.split("\n")
        in_params = False
        in_returns = False
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Args:") or line.startswith("Parameters:"):
                in_params = True
                in_returns = False
                continue
            elif line.startswith("Returns:"):
                in_params = False
                in_returns = True
                continue
            elif line.startswith("Raises:") or line.startswith("Examples:"):
                in_params = False
                in_returns = False
                continue
            
            if in_params and line:
                # Parse parameter line
                if ":" in line:
                    param_part, desc_part = line.split(":", 1)
                    param_name = param_part.strip()
                    parameters.append((param_name, "Any", desc_part.strip()))
            
            if in_returns and line:
                returns += line + " "
        
        return FunctionDoc(
            name=name,
            signature=f"{name}{signature}",
            docstring=docstring,
            parameters=parameters,
            returns=returns.strip(),
        )
    
    def parse_class(self, cls: Any) -> ClassDoc:
        """Parse class documentation.
        
        Args:
            cls: Class object
        
        Returns:
            Class documentation
        """
        name = cls.__name__
        docstring = inspect.getdoc(cls) or ""
        
        # Parse methods
        methods = []
        for method_name, method in inspect.getmembers(cls, inspect.isfunction):
            if not method_name.startswith("_"):
                methods.append(self.parse_function(method))
        
        return ClassDoc(
            name=name,
            docstring=docstring,
            methods=methods,
        )
    
    def parse_module(self, module_path: Path) -> ModuleDoc:
        """Parse module documentation.
        
        Args:
            module_path: Path to Python module
        
        Returns:
            Module documentation
        """
        # Read and parse AST
        source = module_path.read_text()
        tree = ast.parse(source)
        
        # Get module docstring
        docstring = ast.get_docstring(tree) or ""
        
        # Import module dynamically
        import importlib.util
        
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            return ModuleDoc(name=module_path.stem, docstring=docstring)
        
        # Parse classes and functions
        classes = []
        functions = []
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                classes.append(self.parse_class(obj))
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                functions.append(self.parse_function(obj))
        
        return ModuleDoc(
            name=module_path.stem,
            docstring=docstring,
            classes=classes,
            functions=functions,
        )
    
    def generate_docs(self) -> None:
        """Generate documentation for all modules."""
        logger.info(f"Generating docs from {self.source_dir}")
        
        # Find all Python files
        python_files = list(self.source_dir.rglob("*.py"))
        
        for py_file in python_files:
            # Skip __pycache__ and tests
            if "__pycache__" in str(py_file) or "test_" in py_file.name:
                continue
            
            try:
                # Parse module
                module_doc = self.parse_module(py_file)
                
                # Generate Markdown
                markdown = module_doc.to_markdown()
                
                # Write to file
                rel_path = py_file.relative_to(self.source_dir)
                output_path = self.output_dir / rel_path.with_suffix(".md")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                output_path.write_text(markdown)
                
                logger.info(f"Generated docs for {py_file.name}")
            
            except Exception as e:
                logger.warning(f"Failed to generate docs for {py_file}: {e}")


class APIDocGenerator:
    """Generate API documentation."""
    
    def __init__(self, output_path: Path) -> None:
        """Initialize API doc generator.
        
        Args:
            output_path: Output file path
        """
        self.output_path = output_path
    
    def generate(self, package_name: str = "automl") -> None:
        """Generate API documentation.
        
        Args:
            package_name: Package name
        """
        import importlib
        
        # Import package
        package = importlib.import_module(package_name)
        
        lines = [
            f"# {package_name.upper()} API Reference",
            "",
            "Complete API documentation for AutoML.",
            "",
            "## Table of Contents",
            "",
        ]
        
        # Generate TOC
        for module_name in dir(package):
            if not module_name.startswith("_"):
                lines.append(f"- [{module_name}](#{module_name})")
        
        lines.append("")
        
        # Generate module docs
        for module_name in dir(package):
            if not module_name.startswith("_"):
                try:
                    module = getattr(package, module_name)
                    
                    lines.append(f"## {module_name}")
                    lines.append("")
                    
                    if hasattr(module, "__doc__") and module.__doc__:
                        lines.append(module.__doc__)
                        lines.append("")
                
                except Exception as e:
                    logger.warning(f"Failed to document {module_name}: {e}")
        
        # Write
        self.output_path.write_text("\n".join(lines))
        logger.info(f"Generated API docs at {self.output_path}")


class ExampleGenerator:
    """Generate example code and notebooks."""
    
    def __init__(self, output_dir: Path) -> None:
        """Initialize example generator.
        
        Args:
            output_dir: Output directory
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_quickstart(self) -> None:
        """Generate quickstart example."""
        code = '''"""Quickstart example for AutoML.

Train your first AutoML model in minutes!
"""

from automl import AutoMLEngine
from sklearn.datasets import load_iris

# Load data
X, y = load_iris(return_X_y=True)

# Create engine
engine = AutoMLEngine(task_type="classification")

# Train
engine.fit(X, y, n_trials=50, timeout=300)

# Predict
predictions = engine.predict(X)

# Get best model
best_model = engine.get_best_model()

print(f"Best model: {best_model}")
print(f"Accuracy: {engine.best_score_:.4f}")
'''
        
        path = self.output_dir / "quickstart.py"
        path.write_text(code)
        logger.info(f"Generated quickstart example at {path}")
    
    def generate_advanced_example(self) -> None:
        """Generate advanced example with all features."""
        code = '''"""Advanced AutoML example with all features.

Demonstrates production-ready ML pipeline with:
- Distributed training
- Experiment tracking
- Model serving
- Monitoring
"""

from automl import AutoMLEngine
from automl.optimizers import RayTuneOptimizer
from automl.tracking import MLflowTracker
from automl.serving import ModelServer
from automl.monitoring import MetricsCollector
from sklearn.datasets import load_breast_cancer

# 1. Load data
X, y = load_breast_cancer(return_X_y=True)

# 2. Setup distributed optimizer
optimizer = RayTuneOptimizer(
    n_trials=100,
    resources_per_trial={"cpu": 2, "gpu": 0},
)

# 3. Setup experiment tracking
tracker = MLflowTracker(
    experiment_name="breast_cancer_classification",
    tracking_uri="http://localhost:5000",
)

# 4. Create engine with production features
engine = AutoMLEngine(
    task_type="classification",
    optimizer=optimizer,
    tracker=tracker,
)

# 5. Train with monitoring
with MetricsCollector() as metrics:
    engine.fit(X, y, n_trials=100, timeout=3600)
    
    # Log metrics
    metrics.log_metric("best_score", engine.best_score_)

# 6. Serve model
server = ModelServer(model=engine.get_best_model())
app = server.create_app()

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000

# 7. Make predictions via API
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"features": X[0].tolist()},
)

print(f"Prediction: {response.json()}")
'''
        
        path = self.output_dir / "advanced_production.py"
        path.write_text(code)
        logger.info(f"Generated advanced example at {path}")
    
    def generate_all_examples(self) -> None:
        """Generate all examples."""
        self.generate_quickstart()
        self.generate_advanced_example()
        logger.info(f"Generated all examples in {self.output_dir}")
