"""Model and data serialization utilities."""

from __future__ import annotations

import cloudpickle
import joblib
from pathlib import Path
from typing import Any

__all__ = [
    "save_model",
    "load_model",
    "save_artifact",
    "load_artifact",
    "ModelSerializer",
]


def save_model(
    model: Any,
    path: Path | str,
    compress: bool | int = 3,
) -> None:
    """Save a trained model to disk.
    
    Args:
        model: Trained model to save
        path: Output file path
        compress: Compression level (0-9, True=3, False=0)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, path, compress=compress)


def load_model(path: Path | str) -> Any:
    """Load a trained model from disk.
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded model
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    
    return joblib.load(path)


def save_artifact(
    artifact: Any,
    path: Path | str,
    protocol: int = -1,
) -> None:
    """Save arbitrary Python object using cloudpickle.
    
    Useful for saving complex objects like pipelines with custom components.
    
    Args:
        artifact: Object to save
        path: Output file path
        protocol: Pickle protocol version (-1 for latest)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        cloudpickle.dump(artifact, f, protocol=protocol)


def load_artifact(path: Path | str) -> Any:
    """Load artifact saved with cloudpickle.
    
    Args:
        path: Path to saved artifact
        
    Returns:
        Loaded artifact
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    
    with open(path, "rb") as f:
        return cloudpickle.load(f)


class ModelSerializer:
    """Advanced model serialization with metadata.
    
    Saves models with versioning information and metadata.
    """
    
    def __init__(self, base_dir: Path | str = "models"):
        """Initialize model serializer.
        
        Args:
            base_dir: Base directory for model storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: Any,
        name: str,
        version: str = "latest",
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save model with metadata.
        
        Args:
            model: Model to save
            name: Model name
            version: Model version
            metadata: Additional metadata
            
        Returns:
            Path to saved model
        """
        model_dir = self.base_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.joblib"
        save_model(model, model_path)
        
        # Save metadata
        if metadata:
            import json
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
        
        return model_path
    
    def load(
        self,
        name: str,
        version: str = "latest",
    ) -> tuple[Any, dict[str, Any] | None]:
        """Load model with metadata.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Tuple of (model, metadata)
        """
        model_dir = self.base_dir / name / version
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model not found: {name}/{version}")
        
        # Load model
        model_path = model_dir / "model.joblib"
        model = load_model(model_path)
        
        # Load metadata
        metadata = None
        metadata_path = model_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path) as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def list_models(self) -> list[tuple[str, str]]:
        """List all saved models.
        
        Returns:
            List of (model_name, version) tuples
        """
        models = []
        for model_dir in self.base_dir.glob("*/"):
            if model_dir.is_dir():
                for version_dir in model_dir.glob("*/"):
                    if version_dir.is_dir():
                        models.append((model_dir.name, version_dir.name))
        return models
