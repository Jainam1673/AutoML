"""Multi-modal learning support for vision, NLP, and tabular data.

Combines different data modalities for unified predictions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    "MultiModalModel",
    "VisionEncoder",
    "TextEncoder",
    "TabularEncoder",
    "FusionLayer",
]

logger = logging.getLogger(__name__)

# Only define classes if torch is available
if not TORCH_AVAILABLE:
    # Create stub classes that raise ImportError
    class _StubClass:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required for multi-modal features. Install with: pip install torch")
    
    VisionEncoder = _StubClass  # type: ignore
    TextEncoder = _StubClass  # type: ignore
    TabularEncoder = _StubClass  # type: ignore
    FusionLayer = _StubClass  # type: ignore
    MultiModalModel = _StubClass  # type: ignore
    
else:
    # Actual implementations when torch is available
    
    class VisionEncoder(nn.Module):
        """Vision encoder using efficient architectures."""
        
        def __init__(self, input_channels: int = 3, output_dim: int = 512) -> None:
            """Initialize vision encoder.
            
            Args:
                input_channels: Number of input channels (3 for RGB)
                output_dim: Output embedding dimension
            """
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for VisionEncoder")
            
            super().__init__()
            
            # Simple CNN encoder (can be replaced with ViT, ResNet, etc.)
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, output_dim),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.
            
            Args:
                x: Input images (B, C, H, W)
            
            Returns:
                Vision embeddings (B, output_dim)
            """
            return self.encoder(x)


    class TextEncoder(nn.Module):
    """Text encoder using transformers or RNNs."""
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        output_dim: int = 512,
    ) -> None:
        """Initialize text encoder.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Word embedding dimension
            output_dim: Output embedding dimension
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TextEncoder")
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            output_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token IDs (B, seq_len)
        
        Returns:
            Text embeddings (B, output_dim)
        """
        embedded = self.embedding(x)  # (B, seq_len, embedding_dim)
        _, (hidden, _) = self.lstm(embedded)  # (2*num_layers, B, output_dim//2)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, output_dim)
        
        return hidden


class TabularEncoder(nn.Module):
    """Tabular data encoder with embeddings for categoricals."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dims: list[int] | None = None,
    ) -> None:
        """Initialize tabular encoder.
        
        Args:
            input_dim: Number of input features
            output_dim: Output embedding dimension
            hidden_dims: Hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for TabularEncoder")
        
        super().__init__()
        
        hidden_dims = hidden_dims or [256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features (B, input_dim)
        
        Returns:
            Tabular embeddings (B, output_dim)
        """
        return self.encoder(x)


class FusionLayer(nn.Module):
    """Multi-modal fusion layer."""
    
    def __init__(
        self,
        modality_dims: dict[str, int],
        fusion_dim: int = 512,
        fusion_type: str = "concat",
    ) -> None:
        """Initialize fusion layer.
        
        Args:
            modality_dims: Dictionary of {modality_name: embedding_dim}
            fusion_dim: Dimension after fusion
            fusion_type: "concat", "add", "attention", or "gated"
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for FusionLayer")
        
        super().__init__()
        
        self.fusion_type = fusion_type
        self.modality_dims = modality_dims
        
        if fusion_type == "concat":
            total_dim = sum(modality_dims.values())
            self.fusion = nn.Linear(total_dim, fusion_dim)
        
        elif fusion_type == "add":
            # Ensure all dimensions are same
            assert len(set(modality_dims.values())) == 1
            self.fusion = nn.Linear(list(modality_dims.values())[0], fusion_dim)
        
        elif fusion_type == "attention":
            # Cross-attention between modalities
            dim = list(modality_dims.values())[0]
            self.attention = nn.MultiheadAttention(dim, num_heads=8)
            self.fusion = nn.Linear(dim, fusion_dim)
        
        elif fusion_type == "gated":
            # Gated fusion
            self.gates = nn.ModuleDict({
                name: nn.Linear(dim, dim)
                for name, dim in modality_dims.items()
            })
            total_dim = sum(modality_dims.values())
            self.fusion = nn.Linear(total_dim, fusion_dim)
    
    def forward(self, modality_embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality embeddings.
        
        Args:
            modality_embeddings: Dict of {modality_name: embedding_tensor}
        
        Returns:
            Fused embedding
        """
        if self.fusion_type == "concat":
            # Concatenate all modalities
            embeddings = [modality_embeddings[name] for name in self.modality_dims.keys()]
            fused = torch.cat(embeddings, dim=1)
            return self.fusion(fused)
        
        elif self.fusion_type == "add":
            # Element-wise addition
            embeddings = [modality_embeddings[name] for name in self.modality_dims.keys()]
            fused = torch.stack(embeddings).mean(dim=0)
            return self.fusion(fused)
        
        elif self.fusion_type == "attention":
            # Attention-based fusion
            embeddings = [modality_embeddings[name] for name in self.modality_dims.keys()]
            stacked = torch.stack(embeddings)  # (num_modalities, B, dim)
            attended, _ = self.attention(stacked, stacked, stacked)
            fused = attended.mean(dim=0)  # (B, dim)
            return self.fusion(fused)
        
        elif self.fusion_type == "gated":
            # Gated fusion
            gated_embeddings = []
            for name, embedding in modality_embeddings.items():
                gate = torch.sigmoid(self.gates[name](embedding))
                gated_embeddings.append(gate * embedding)
            
            fused = torch.cat(gated_embeddings, dim=1)
            return self.fusion(fused)
        
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class MultiModalModel(nn.Module):
    """Multi-modal model combining vision, text, and tabular data."""
    
    def __init__(
        self,
        modalities: dict[str, dict[str, Any]],
        fusion_dim: int = 512,
        num_classes: int = 2,
        fusion_type: str = "concat",
    ) -> None:
        """Initialize multi-modal model.
        
        Args:
            modalities: Dict of modality configs, e.g.:
                {
                    "vision": {"input_channels": 3, "output_dim": 512},
                    "text": {"vocab_size": 10000, "output_dim": 512},
                    "tabular": {"input_dim": 50, "output_dim": 512},
                }
            fusion_dim: Fusion layer dimension
            num_classes: Number of output classes
            fusion_type: Type of fusion ("concat", "add", "attention", "gated")
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for MultiModalModel")
        
        super().__init__()
        
        self.modalities = modalities
        
        # Create encoders
        self.encoders = nn.ModuleDict()
        modality_dims = {}
        
        if "vision" in modalities:
            config = modalities["vision"]
            self.encoders["vision"] = VisionEncoder(
                input_channels=config.get("input_channels", 3),
                output_dim=config.get("output_dim", 512),
            )
            modality_dims["vision"] = config.get("output_dim", 512)
        
        if "text" in modalities:
            config = modalities["text"]
            self.encoders["text"] = TextEncoder(
                vocab_size=config.get("vocab_size", 10000),
                embedding_dim=config.get("embedding_dim", 128),
                output_dim=config.get("output_dim", 512),
            )
            modality_dims["text"] = config.get("output_dim", 512)
        
        if "tabular" in modalities:
            config = modalities["tabular"]
            self.encoders["tabular"] = TabularEncoder(
                input_dim=config["input_dim"],
                output_dim=config.get("output_dim", 512),
                hidden_dims=config.get("hidden_dims"),
            )
            modality_dims["tabular"] = config.get("output_dim", 512)
        
        # Fusion layer
        self.fusion = FusionLayer(modality_dims, fusion_dim, fusion_type)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Args:
            inputs: Dict of {modality_name: input_tensor}
        
        Returns:
            Class logits
        """
        # Encode each modality
        embeddings = {}
        for modality_name, encoder in self.encoders.items():
            if modality_name in inputs:
                embeddings[modality_name] = encoder(inputs[modality_name])
        
        # Fuse modalities
        fused = self.fusion(embeddings)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
