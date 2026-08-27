"""Application service coordinating safe uploads, parsing, and version persistence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from local_lke.errors import IngestionError
from local_lke.ingestion.chunking import chunk_elements
from local_lke.ingestion.parsers import parse_document
from local_lke.ingestion.safety import store_upload, validate_upload
from local_lke.models import (
    ChunkStrategy,
    CollectionResponse,
    DocumentResponse,
    IngestionJobResponse,
    JobStatus,
    ParserPreviewResponse,
    ParserStrategy,
)
from local_lke.settings import Settings
from local_lke.storage import IngestionRepository

PIPELINE_SCHEMA_VERSION = "chapter-02-v1"


class IngestionService:
    def __init__(self, repository: IngestionRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    def startup(self) -> int:
        """Mark jobs abandoned by an earlier process as explicitly retryable."""
        return self.repository.recover_running_jobs()

    def check_health(self) -> str:
        return self.repository.check_health()

    def create_collection(
        self, name: str, owner_principal_id: str | None = None
    ) -> CollectionResponse:
        return self.repository.create_collection(name, owner_principal_id)

    def list_collections(self) -> list[CollectionResponse]:
        return self.repository.list_collections()

    def ingest(
        self,
        *,
        collection_id: UUID,
        filename: str,
        content_type: str | None,
        content: bytes,
        parser_strategy: ParserStrategy,
        chunk_strategy: ChunkStrategy,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionJobResponse:
        job = self.enqueue(
            collection_id=collection_id,
            filename=filename,
            content_type=content_type,
            content=content,
            parser_strategy=parser_strategy,
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return self.process(job.id)

    def enqueue(
        self,
        *,
        collection_id: UUID,
        filename: str,
        content_type: str | None,
        content: bytes,
        parser_strategy: ParserStrategy,
        chunk_strategy: ChunkStrategy,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestionJobResponse:
        self.repository.get_collection(collection_id)
        safe_name, media_type = validate_upload(
            filename,
            content_type,
            content,
            max_bytes=self.settings.max_upload_bytes,
        )
        size = chunk_size or self.settings.chunk_size
        overlap = self.settings.chunk_overlap if chunk_overlap is None else chunk_overlap
        _validate_chunk_configuration(size, overlap)
        stored_path = store_upload(
            content,
            root=self.settings.upload_directory,
            collection_id=collection_id,
            filename=safe_name,
        )
        return self.repository.create_job(
            collection_id=str(collection_id),
            filename=safe_name.casefold(),
            display_filename=safe_name,
            storage_path=str(stored_path),
            media_type=media_type,
            parser_strategy=parser_strategy.value,
            chunk_strategy=chunk_strategy.value,
            chunk_size=size,
            chunk_overlap=overlap,
            status=JobStatus.QUEUED.value,
            progress=0,
        )

    def process(self, job_id: UUID) -> IngestionJobResponse:
        """Run one queued ingestion job; errors are persisted, never leaked."""
        return self._process(job_id)

    def retry(self, job_id: UUID) -> IngestionJobResponse:
        job_input = self.repository.get_job_input(job_id)
        if job_input["status"] not in {
            JobStatus.FAILED.value,
            JobStatus.INTERRUPTED.value,
        }:
            raise IngestionError(
                "Only failed or interrupted jobs can be retried",
                code="job_not_retryable",
            )
        self.repository.update_job(
            job_id,
            status=JobStatus.QUEUED.value,
            progress=0,
            error_code=None,
            error_message=None,
            finished_at=None,
        )
        return self._process(job_id)

    def get_job(self, job_id: UUID) -> IngestionJobResponse:
        return self.repository.get_job(job_id)

    def list_documents(self, collection_id: UUID) -> list[DocumentResponse]:
        self.repository.get_collection(collection_id)
        return self.repository.list_documents(collection_id)

    def preview(self, version_id: UUID) -> ParserPreviewResponse:
        return self.repository.get_preview(version_id)

    def delete_document(self, document_id: UUID, reason: str) -> DocumentResponse:
        cleaned_reason = " ".join(reason.split()) or "deleted by user"
        return self.repository.soft_delete_document(document_id, cleaned_reason[:255])

    def _process(self, job_id: UUID) -> IngestionJobResponse:
        job_input = self.repository.get_job_input(job_id)
        self.repository.update_job(
            job_id,
            status=JobStatus.RUNNING.value,
            progress=10,
            started_at=datetime.now(UTC),
            finished_at=None,
        )
        try:
            path = Path(str(job_input["storage_path"]))
            if not path.is_file():
                raise IngestionError(
                    "The stored upload is missing; upload the file again",
                    code="stored_upload_missing",
                )
            content_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            configuration = pipeline_configuration(
                parser_strategy=ParserStrategy(str(job_input["parser_strategy"])),
                chunk_strategy=ChunkStrategy(str(job_input["chunk_strategy"])),
                chunk_size=int(job_input["chunk_size"]),
                chunk_overlap=int(job_input["chunk_overlap"]),
            )
            pipeline_hash = hash_pipeline_configuration(configuration)
            collection_id = UUID(str(job_input["collection_id"]))
            document = self.repository.find_document(collection_id, str(job_input["filename"]))
            if document is None:
                document = self.repository.create_document(
                    collection_id,
                    str(job_input["filename"]),
                    str(job_input["display_filename"]),
                )
            existing = self.repository.find_version(document.id, content_sha256, pipeline_hash)
            if existing is not None:
                return self.repository.update_job(
                    job_id,
                    document_id=str(document.id),
                    version_id=str(existing.id),
                    status=JobStatus.COMPLETED.value,
                    progress=100,
                    skipped=True,
                    element_count=existing.element_count,
                    chunk_count=existing.chunk_count,
                    warning_count=existing.warning_count,
                    finished_at=datetime.now(UTC),
                )

            version_id = uuid4()
            self.repository.update_job(job_id, document_id=str(document.id), progress=35)
            parsed = parse_document(
                path,
                media_type=str(job_input["media_type"]),
                version_id=version_id,
                strategy=ParserStrategy(str(job_input["parser_strategy"])),
            )
            self.repository.update_job(job_id, progress=65, element_count=len(parsed.elements))
            chunks, chunk_warnings = chunk_elements(
                parsed.elements,
                version_id=version_id,
                strategy=ChunkStrategy(str(job_input["chunk_strategy"])),
                chunk_size=int(job_input["chunk_size"]),
                chunk_overlap=int(job_input["chunk_overlap"]),
            )
            if not chunks:
                raise IngestionError(
                    "Parsing succeeded but produced no usable chunks",
                    code="empty_chunks",
                )
            warning_count = len(parsed.warnings) + len(chunk_warnings)
            self.repository.update_job(
                job_id,
                progress=85,
                chunk_count=len(chunks),
                warning_count=warning_count,
            )
            version = self.repository.persist_version(
                document_id=document.id,
                version_id=version_id,
                content_sha256=content_sha256,
                pipeline_hash=pipeline_hash,
                pipeline_configuration=configuration,
                media_type=str(job_input["media_type"]),
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                parser_strategy=str(job_input["parser_strategy"]),
                storage_path=str(path),
                elements=parsed.elements,
                chunks=chunks,
                warning_count=warning_count,
            )
            return self.repository.update_job(
                job_id,
                document_id=str(document.id),
                version_id=str(version.id),
                status=JobStatus.COMPLETED.value,
                progress=100,
                element_count=len(parsed.elements),
                chunk_count=len(chunks),
                warning_count=warning_count,
                finished_at=datetime.now(UTC),
            )
        except IngestionError as exc:
            return self.repository.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                error_code=exc.code,
                error_message=str(exc),
                finished_at=datetime.now(UTC),
            )
        except Exception as exc:
            return self.repository.update_job(
                job_id,
                status=JobStatus.FAILED.value,
                error_code="unexpected_ingestion_error",
                error_message=f"Ingestion failed safely: {type(exc).__name__}",
                finished_at=datetime.now(UTC),
            )


def pipeline_configuration(
    *,
    parser_strategy: ParserStrategy,
    chunk_strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> dict[str, str | int]:
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "parser": "normalized-v1",
        "parser_strategy": parser_strategy.value,
        "chunk_strategy": chunk_strategy.value,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "semantic_algorithm": "local-tfidf-adjacent-v1",
    }


def hash_pipeline_configuration(configuration: Mapping[str, object]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_chunk_configuration(size: int, overlap: int) -> None:
    if size < 100 or size > 100_000:
        raise IngestionError("chunk_size must be between 100 and 100000", code="invalid_chunking")
    if overlap < 0 or overlap >= size:
        raise IngestionError(
            "chunk_overlap must be non-negative and smaller than chunk_size",
            code="invalid_chunking",
        )
