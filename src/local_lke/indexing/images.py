"""Safe optional image ingestion and joint text/image retrieval."""

from __future__ import annotations

import hashlib
import warnings
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import UUID

from PIL import Image, UnidentifiedImageError

from local_lke.errors import IndexingError
from local_lke.indexing.repository import SqlAlchemyIndexRepository
from local_lke.ingestion.safety import normalize_filename, store_upload
from local_lke.models import (
    EmbeddingModality,
    EmbeddingProfileResponse,
    ImageAssetResponse,
    ImageSearchHit,
    ImageSearchResponse,
)
from local_lke.providers import MultimodalEmbeddingProvider
from local_lke.settings import Settings

IMAGE_TYPES = {
    ".png": ("image/png", "PNG"),
    ".jpg": ("image/jpeg", "JPEG"),
    ".jpeg": ("image/jpeg", "JPEG"),
    ".webp": ("image/webp", "WEBP"),
}


class MultimodalIndexingService:
    def __init__(
        self,
        repository: SqlAlchemyIndexRepository,
        embeddings: MultimodalEmbeddingProvider,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.settings = settings

    def profile(self) -> EmbeddingProfileResponse:
        if self.embeddings.dimension != self.settings.multimodal_dimension:
            raise IndexingError(
                "Configured multimodal dimension does not match the local model",
                code="multimodal_dimension_mismatch",
            )
        return self.repository.get_or_create_profile(
            modality=EmbeddingModality.MULTIMODAL,
            model_id=self.embeddings.model_id,
            revision=self.embeddings.revision,
            dimension=self.embeddings.dimension,
            normalized=self.embeddings.normalized,
        )

    def ingest(
        self,
        *,
        collection_id: UUID,
        filename: str,
        content_type: str | None,
        content: bytes,
    ) -> ImageAssetResponse:
        safe_name, media_type, width, height = validate_image(
            filename,
            content_type,
            content,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )
        digest = hashlib.sha256(content).hexdigest()
        existing = self.repository.find_image(collection_id, digest)
        profile = self.profile()
        if existing is not None:
            if not self.repository.has_image_embedding(existing.id, profile.id):
                vector = self.embeddings.embed_images(
                    [self.repository.get_image_path(existing.id)]
                )[0]
                self._write_embedding(existing.id, profile.id, vector)
            self.repository.activate_profile(
                collection_id, profile.id, EmbeddingModality.MULTIMODAL
            )
            return existing
        stored = store_upload(
            content,
            root=self.settings.upload_directory,
            collection_id=collection_id,
            filename=safe_name,
        )
        image = self.repository.create_image(
            collection_id=str(collection_id),
            filename=safe_name,
            media_type=media_type,
            storage_path=str(stored),
            width=width,
            height=height,
            sha256=digest,
            content_fingerprint=bytes.fromhex(digest),
        )
        vector = self.embeddings.embed_images([stored])[0]
        self._write_embedding(image.id, profile.id, vector)
        self.repository.activate_profile(
            collection_id, profile.id, EmbeddingModality.MULTIMODAL
        )
        return image

    def search_text(self, collection_id: UUID, query: str, top_k: int) -> ImageSearchResponse:
        profile = self._active_profile(collection_id)
        vector = self.embeddings.embed_text(query)
        self._validate_vector(vector, profile.dimension)
        return ImageSearchResponse(
            profile=profile,
            hits=[
                ImageSearchHit(image=image, rank=rank, score=round(score, 8))
                for rank, (image, score) in enumerate(
                    self.repository.search_images(collection_id, profile.id, vector, top_k),
                    start=1,
                )
            ],
        )

    def search_image(
        self,
        collection_id: UUID,
        filename: str,
        content_type: str | None,
        content: bytes,
        top_k: int,
    ) -> ImageSearchResponse:
        safe_name, _media_type, _width, _height = validate_image(
            filename,
            content_type,
            content,
            max_bytes=self.settings.max_upload_bytes,
            max_pixels=self.settings.max_image_pixels,
        )
        profile = self._active_profile(collection_id)
        suffix = Path(safe_name).suffix
        with NamedTemporaryFile(prefix="lke-image-query-", suffix=suffix, delete=False) as temp:
            temporary = Path(temp.name)
            temp.write(content)
        try:
            vector = self.embeddings.embed_images([temporary])[0]
        finally:
            temporary.unlink(missing_ok=True)
        self._validate_vector(vector, profile.dimension)
        return ImageSearchResponse(
            profile=profile,
            hits=[
                ImageSearchHit(image=image, rank=rank, score=round(score, 8))
                for rank, (image, score) in enumerate(
                    self.repository.search_images(collection_id, profile.id, vector, top_k),
                    start=1,
                )
            ],
        )

    def get_content_path(self, image_id: UUID) -> Path:
        return self.repository.get_image_path(image_id)

    def _active_profile(self, collection_id: UUID) -> EmbeddingProfileResponse:
        profile = self.repository.get_active_profile(
            collection_id, EmbeddingModality.MULTIMODAL
        )
        if profile is None:
            raise IndexingError(
                "The collection has no multimodal index; upload an image first.",
                code="multimodal_index_not_ready",
            )
        return profile

    def _write_embedding(self, image_id: UUID, profile_id: UUID, vector: list[float]) -> None:
        self._validate_vector(vector, self.settings.multimodal_dimension)
        self.repository.write_image_embedding(image_id, profile_id, vector)

    @staticmethod
    def _validate_vector(vector: list[float], dimension: int) -> None:
        if len(vector) != dimension:
            raise IndexingError(
                f"Multimodal vector has {len(vector)} dimensions; expected {dimension}",
                code="multimodal_dimension_mismatch",
            )


def validate_image(
    filename: str,
    declared_content_type: str | None,
    content: bytes,
    *,
    max_bytes: int,
    max_pixels: int,
) -> tuple[str, str, int, int]:
    safe_name = normalize_filename(filename)
    expected = IMAGE_TYPES.get(Path(safe_name).suffix.lower())
    if expected is None:
        raise IndexingError(
            "Only .png, .jpg, .jpeg, and .webp images are supported",
            code="unsupported_image_type",
        )
    if not content:
        raise IndexingError("Empty images cannot be ingested", code="empty_image")
    if len(content) > max_bytes:
        raise IndexingError(
            f"Image exceeds the configured {max_bytes}-byte limit",
            code="file_too_large",
        )
    declared = (declared_content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if declared not in {expected[0], "application/octet-stream", ""}:
        raise IndexingError(
            "Declared MIME type does not match the image extension",
            code="image_mime_mismatch",
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                width, height = image.size
                if width * height > max_pixels:
                    raise IndexingError(
                        f"Image exceeds the configured {max_pixels}-pixel limit",
                        code="image_too_large",
                    )
                if image.format != expected[1]:
                    raise IndexingError(
                        "Decoded image format does not match its extension",
                        code="image_mime_mismatch",
                    )
                image.verify()
            with Image.open(BytesIO(content)) as image:
                image.convert("RGB").load()
    except IndexingError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Warning) as exc:
        raise IndexingError(
            "The image is malformed, unsafe, or exceeds the pixel limit",
            code="invalid_image",
        ) from exc
    return safe_name, expected[0], width, height
