"""Pydantic-backed configuration models for orchestrating AutoML runs."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field, model_validator

__all__ = [
    "DatasetConfig",
    "PreprocessorConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PipelineConfig",
    "AutoMLConfig",
]


class DatasetConfig(BaseModel):
    name: str = Field(..., description="Dataset provider identifier")


class PreprocessorConfig(BaseModel):
    name: str = Field(..., description="Preprocessor factory identifier")
    params: Mapping[str, Any] | None = Field(default=None)


class ModelConfig(BaseModel):
    name: str = Field(..., description="Model factory identifier")
    base_params: Mapping[str, Any] | None = Field(default=None)
    search_space: Sequence[Mapping[str, Any]] | None = Field(default=None)

    @model_validator(mode="after")
    def ensure_unique_search_space(self) -> "ModelConfig":
        if self.search_space:
            seen = set()
            for entry in self.search_space:
                fingerprint = tuple(sorted(entry.items()))
                if fingerprint in seen:
                    msg = f"Duplicate search-space entry detected: {entry}"
                    raise ValueError(msg)
                seen.add(fingerprint)
        return self


class OptimizerConfig(BaseModel):
    name: str = Field(..., description="Optimizer identifier")
    params: Mapping[str, Any] | None = Field(default=None)
    cv_folds: int = Field(default=3, ge=2, le=20)
    scoring: str = Field(default="accuracy")


class PipelineConfig(BaseModel):
    preprocessors: Sequence[PreprocessorConfig] = Field(default_factory=tuple)
    model: ModelConfig


class AutoMLConfig(BaseModel):
    run_name: str = Field(default="automl-run")
    dataset: DatasetConfig
    pipeline: PipelineConfig
    optimizer: OptimizerConfig

    def hash(self) -> str:
        payload = self.model_dump_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
