"""Optional local cross-encoder reranking boundary."""

from __future__ import annotations

from typing import Any, Protocol

from local_lke.errors import ProviderUnavailableError


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]: ...


class LocalCrossEncoderReranker:
    """Lazy sentence-transformers CrossEncoder; no network service is used."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        try:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            predictions = self._model.predict([(query, document) for document in documents])
            return [float(value) for value in predictions]
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Cannot run local reranker '{self.model_name}'. "
                "Download it once or disable reranking.",
                component="reranker",
            ) from exc
