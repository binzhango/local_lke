"""Local Hugging Face embeddings and a deterministic test implementation."""

import hashlib
import math
import re
from typing import Protocol

from langchain_huggingface import HuggingFaceEmbeddings

from local_lke.errors import ProviderUnavailableError


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def check_initialization(self) -> str: ...


class LocalHuggingFaceEmbeddings:
    """Lazy local sentence-transformers adapter."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client: HuggingFaceEmbeddings | None = None

    def _get_client(self) -> HuggingFaceEmbeddings:
        if self._client is None:
            try:
                self._client = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    encode_kwargs={"normalize_embeddings": True},
                )
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Cannot initialize local embedding model '{self.model_name}'. "
                    "Ensure it is downloaded or allow Hugging Face to download it once.",
                    component="embeddings",
                ) from exc
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._get_client().embed_documents(texts)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Embedding documents with '{self.model_name}' failed.",
                component="embeddings",
            ) from exc

    def embed_query(self, text: str) -> list[float]:
        try:
            return self._get_client().embed_query(text)
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Embedding a query with '{self.model_name}' failed.",
                component="embeddings",
            ) from exc

    def check_initialization(self) -> str:
        vector = self.embed_query("embedding health check")
        if not vector:
            raise ProviderUnavailableError(
                "The embedding model returned an empty vector.", component="embeddings"
            )
        return f"embedding model initialized ({len(vector)} dimensions)"


class DeterministicFakeEmbeddings:
    """Small normalized hashing embedder with no model or network dependencies."""

    def __init__(self, dimensions: int = 64) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def check_initialization(self) -> str:
        return f"fake embedding initialized ({self.dimensions} dimensions)"

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
