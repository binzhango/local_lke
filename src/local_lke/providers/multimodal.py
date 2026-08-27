"""Optional local joint text/image embedding providers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageStat

from local_lke.errors import ProviderUnavailableError


class MultimodalEmbeddingProvider(Protocol):
    model_id: str
    revision: str
    dimension: int
    normalized: bool

    def embed_text(self, text: str) -> list[float]: ...

    def embed_images(self, paths: list[Path]) -> list[list[float]]: ...

    def check_initialization(self) -> str: ...


class LocalCLIPEmbeddings:
    """Lazy sentence-transformers CLIP adapter; no image leaves the machine."""

    def __init__(self, model_id: str, *, revision: str, dimension: int) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dimension = dimension
        self.normalized = True
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._client = SentenceTransformer(self.model_id, revision=self.revision)
            except Exception as exc:
                raise ProviderUnavailableError(
                    f"Cannot initialize local multimodal model '{self.model_id}'. "
                    "Download it once or keep multimodal retrieval disabled.",
                    component="multimodal_embeddings",
                ) from exc
        return self._client

    def embed_text(self, text: str) -> list[float]:
        return self._encode([text])[0]

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        images: list[Image.Image] = []
        try:
            for path in paths:
                with Image.open(path) as source:
                    images.append(source.convert("RGB").copy())
            return self._encode(images)
        finally:
            for image in images:
                image.close()

    def _encode(self, items: Sequence[object]) -> list[list[float]]:
        try:
            client = self._get_client()
            encoded = client.encode(items, normalize_embeddings=True)  # type: ignore[attr-defined]
            vectors = [list(map(float, row)) for row in encoded]
        except ProviderUnavailableError:
            raise
        except Exception as exc:
            raise ProviderUnavailableError(
                f"Multimodal embedding with '{self.model_id}' failed.",
                component="multimodal_embeddings",
            ) from exc
        if any(len(vector) != self.dimension for vector in vectors):
            raise ProviderUnavailableError(
                f"Multimodal model returned a dimension other than {self.dimension}.",
                component="multimodal_embeddings",
            )
        return vectors

    def check_initialization(self) -> str:
        vector = self.embed_text("multimodal health check")
        return f"multimodal model initialized ({len(vector)} dimensions)"


class DeterministicFakeMultimodalEmbeddings:
    """Color-aware joint space for deterministic image ranking tests."""

    model_id = "deterministic-color-joint-v1"
    revision = "1"
    dimension = 8
    normalized = True

    def embed_text(self, text: str) -> list[float]:
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        vector = [
            float(bool(tokens & {"red", "crimson"})),
            float(bool(tokens & {"green", "emerald"})),
            float(bool(tokens & {"blue", "azure"})),
            float(bool(tokens & {"bright", "white"})),
            float(bool(tokens & {"dark", "black"})),
            float(bool(tokens & {"wide", "landscape"})),
            float(bool(tokens & {"tall", "portrait"})),
            0.25,
        ]
        return _normalize(vector)

    def embed_images(self, paths: list[Path]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                mean = ImageStat.Stat(rgb).mean
                width, height = rgb.size
            brightness = sum(mean) / (3 * 255)
            vectors.append(
                _normalize(
                    [
                        mean[0] / 255,
                        mean[1] / 255,
                        mean[2] / 255,
                        brightness,
                        1 - brightness,
                        width / max(width, height),
                        height / max(width, height),
                        0.25,
                    ]
                )
            )
        return vectors

    def check_initialization(self) -> str:
        return "fake multimodal embedding initialized (8 dimensions)"


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector
