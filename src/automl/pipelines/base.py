"""Pipeline assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

__all__ = ["PreprocessorFactory", "PipelineFactory", "PipelineArtifact"]

PreprocessorFactory = Callable[[Mapping[str, object] | None], TransformerMixin]


@dataclass(slots=True)
class PipelineArtifact:
    """Represents a trained pipeline and its evaluation score."""

    pipeline: Pipeline
    score: float
    params: Mapping[str, object]


class PipelineFactory:
    """Build sklearn pipelines from registered preprocessors and models."""

    def __init__(
        self,
        preprocessor_factories: Sequence[Callable[[Mapping[str, object] | None], TransformerMixin]],
        model_factory: Callable[[Mapping[str, object] | None], BaseEstimator],
    ) -> None:
        self._preprocessor_factories = list(preprocessor_factories)
        self._model_factory = model_factory

    def build(
        self,
        *,
        preprocessor_overrides: Sequence[Mapping[str, object] | None] | None = None,
        model_params: Mapping[str, object] | None = None,
    ) -> Pipeline:
        steps: list[tuple[str, TransformerMixin | BaseEstimator]] = []
        overrides = list(preprocessor_overrides or [])
        if overrides and len(overrides) != len(self._preprocessor_factories):
            msg = "Overrides must match number of preprocessors."
            raise ValueError(msg)

        for index, factory in enumerate(self._preprocessor_factories):
            params = overrides[index] if overrides else None
            transformer = factory(params)
            steps.append((f"prep_{index}", transformer))

        model = self._model_factory(model_params)
        steps.append(("model", model))
        return Pipeline(steps)

    def describe(self) -> Iterable[str]:  # pragma: no cover - convenience helper
        for idx, factory in enumerate(self._preprocessor_factories):
            yield f"prep_{idx}: {factory.__module__}.{factory.__name__}"
        yield f"model: {self._model_factory.__module__}.{self._model_factory.__name__}"
