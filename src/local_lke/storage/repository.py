"""Repository interface and SQLAlchemy implementation for ingestion state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy import Engine, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, selectinload, sessionmaker

from local_lke.errors import IngestionError, NotFoundError
from local_lke.models import (
    ChunkStrategy,
    CollectionResponse,
    DocumentElement,
    DocumentResponse,
    DocumentVersionResponse,
    IngestedChunk,
    IngestionJobResponse,
    JobStatus,
    ParserPreviewResponse,
    ParserStrategy,
)
from local_lke.storage.models import (
    ChunkRecord,
    CollectionRecord,
    DocumentElementRecord,
    DocumentVersionRecord,
    IngestionJobRecord,
    LogicalDocumentRecord,
    PipelineConfigurationRecord,
)


class IngestionRepository(Protocol):
    """Persistence contract used by the ingestion service."""

    def check_health(self) -> str: ...

    def create_collection(self, name: str) -> CollectionResponse: ...

    def list_collections(self) -> list[CollectionResponse]: ...

    def get_collection(self, collection_id: UUID) -> CollectionResponse: ...

    def create_job(self, **values: Any) -> IngestionJobResponse: ...

    def get_job(self, job_id: UUID) -> IngestionJobResponse: ...

    def update_job(self, job_id: UUID, **values: Any) -> IngestionJobResponse: ...

    def recover_running_jobs(self) -> int: ...

    def get_job_input(self, job_id: UUID) -> Mapping[str, Any]: ...

    def find_document(self, collection_id: UUID, filename: str) -> DocumentResponse | None: ...

    def create_document(
        self, collection_id: UUID, filename: str, display_filename: str
    ) -> DocumentResponse: ...

    def find_version(
        self, document_id: UUID, content_sha256: str, pipeline_hash: str
    ) -> DocumentVersionResponse | None: ...

    def persist_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        content_sha256: str,
        pipeline_hash: str,
        pipeline_configuration: Mapping[str, Any],
        media_type: str,
        parser_name: str,
        parser_version: str,
        parser_strategy: str,
        storage_path: str,
        elements: Sequence[DocumentElement],
        chunks: Sequence[IngestedChunk],
        warning_count: int,
    ) -> DocumentVersionResponse: ...

    def list_documents(self, collection_id: UUID) -> list[DocumentResponse]: ...

    def get_preview(self, version_id: UUID) -> ParserPreviewResponse: ...

    def soft_delete_document(self, document_id: UUID, reason: str) -> DocumentResponse: ...


class SqlAlchemyIngestionRepository:
    def __init__(self, sessions: sessionmaker[Session], engine: Engine) -> None:
        self.sessions = sessions
        self.engine = engine

    def check_health(self) -> str:
        with self.engine.connect() as connection:
            statement = (
                "SELECT sqlite_version()"
                if self.engine.dialect.name == "sqlite"
                else "SELECT version()"
            )
            version = connection.exec_driver_sql(statement).scalar_one()
        return str(version).split(",", maxsplit=1)[0]

    def create_collection(self, name: str) -> CollectionResponse:
        normalized = " ".join(name.split())
        if not normalized:
            raise IngestionError("Collection name cannot be blank", code="invalid_collection")
        with self.sessions.begin() as session:
            record = CollectionRecord(name=normalized)
            session.add(record)
            try:
                session.flush()
            except IntegrityError as exc:
                raise IngestionError(
                    f"A collection named '{normalized}' already exists",
                    code="collection_exists",
                ) from exc
            return _collection_response(record)

    def list_collections(self) -> list[CollectionResponse]:
        with self.sessions() as session:
            records = session.scalars(select(CollectionRecord).order_by(CollectionRecord.name))
            return [_collection_response(item) for item in records]

    def get_collection(self, collection_id: UUID) -> CollectionResponse:
        with self.sessions() as session:
            record = session.get(CollectionRecord, str(collection_id))
            if record is None:
                raise NotFoundError("Collection not found", component="collection")
            return _collection_response(record)

    def create_job(self, **values: Any) -> IngestionJobResponse:
        with self.sessions.begin() as session:
            record = IngestionJobRecord(**values)
            session.add(record)
            session.flush()
            return _job_response(record)

    def get_job(self, job_id: UUID) -> IngestionJobResponse:
        with self.sessions() as session:
            record = session.get(IngestionJobRecord, str(job_id))
            if record is None:
                raise NotFoundError("Ingestion job not found", component="ingestion")
            return _job_response(record)

    def update_job(self, job_id: UUID, **values: Any) -> IngestionJobResponse:
        values["updated_at"] = datetime.now(UTC)
        with self.sessions.begin() as session:
            record = session.get(IngestionJobRecord, str(job_id))
            if record is None:
                raise NotFoundError("Ingestion job not found", component="ingestion")
            for name, value in values.items():
                setattr(record, name, value)
            session.flush()
            return _job_response(record)

    def recover_running_jobs(self) -> int:
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            result = session.execute(
                update(IngestionJobRecord)
                .where(IngestionJobRecord.status == JobStatus.RUNNING.value)
                .values(
                    status=JobStatus.INTERRUPTED.value,
                    error_code="process_interrupted",
                    error_message=(
                        "The application stopped while this job was running; retry it explicitly."
                    ),
                    finished_at=now,
                    updated_at=now,
                )
            )
            return int(result.rowcount or 0) if isinstance(result, CursorResult) else 0

    def get_job_input(self, job_id: UUID) -> Mapping[str, Any]:
        with self.sessions() as session:
            record = session.get(IngestionJobRecord, str(job_id))
            if record is None:
                raise NotFoundError("Ingestion job not found", component="ingestion")
            return {
                "collection_id": UUID(record.collection_id),
                "filename": record.filename,
                "display_filename": record.display_filename,
                "storage_path": record.storage_path,
                "media_type": record.media_type,
                "parser_strategy": record.parser_strategy,
                "chunk_strategy": record.chunk_strategy,
                "chunk_size": record.chunk_size,
                "chunk_overlap": record.chunk_overlap,
                "status": record.status,
            }

    def find_document(self, collection_id: UUID, filename: str) -> DocumentResponse | None:
        with self.sessions() as session:
            statement = (
                select(LogicalDocumentRecord)
                .where(
                    LogicalDocumentRecord.collection_id == str(collection_id),
                    LogicalDocumentRecord.filename == filename,
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
                .options(selectinload(LogicalDocumentRecord.versions))
            )
            record = session.scalar(statement)
            return _document_response(record) if record is not None else None

    def create_document(
        self, collection_id: UUID, filename: str, display_filename: str
    ) -> DocumentResponse:
        with self.sessions.begin() as session:
            record = LogicalDocumentRecord(
                collection_id=str(collection_id),
                filename=filename,
                display_filename=display_filename,
            )
            session.add(record)
            session.flush()
            return _document_response(record)

    def find_version(
        self, document_id: UUID, content_sha256: str, pipeline_hash: str
    ) -> DocumentVersionResponse | None:
        with self.sessions() as session:
            record = session.scalar(
                select(DocumentVersionRecord).where(
                    DocumentVersionRecord.document_id == str(document_id),
                    DocumentVersionRecord.content_sha256 == content_sha256,
                    DocumentVersionRecord.pipeline_hash == pipeline_hash,
                )
            )
            return _version_response(record) if record is not None else None

    def persist_version(
        self,
        *,
        document_id: UUID,
        version_id: UUID,
        content_sha256: str,
        pipeline_hash: str,
        pipeline_configuration: Mapping[str, Any],
        media_type: str,
        parser_name: str,
        parser_version: str,
        parser_strategy: str,
        storage_path: str,
        elements: Sequence[DocumentElement],
        chunks: Sequence[IngestedChunk],
        warning_count: int,
    ) -> DocumentVersionResponse:
        with self.sessions.begin() as session:
            configuration = session.get(PipelineConfigurationRecord, pipeline_hash)
            if configuration is None:
                configuration = PipelineConfigurationRecord(
                    pipeline_hash=pipeline_hash,
                    schema_version=str(pipeline_configuration["schema_version"]),
                    configuration=dict(pipeline_configuration),
                )
                session.add(configuration)

            session.execute(
                update(DocumentVersionRecord)
                .where(
                    DocumentVersionRecord.document_id == str(document_id),
                    DocumentVersionRecord.active.is_(True),
                )
                .values(active=False, inactive_reason="superseded by a newer version")
            )
            record = DocumentVersionRecord(
                id=str(version_id),
                document_id=str(document_id),
                content_sha256=content_sha256,
                pipeline_hash=pipeline_hash,
                media_type=media_type,
                parser_name=parser_name,
                parser_version=parser_version,
                parser_strategy=parser_strategy,
                storage_path=storage_path,
                active=True,
                status="complete",
                element_count=len(elements),
                chunk_count=len(chunks),
                warning_count=warning_count,
            )
            session.add(record)
            session.add_all(
                DocumentElementRecord(
                    id=str(element.element_id),
                    version_id=str(version_id),
                    ordinal=element.ordinal,
                    category=element.category,
                    text=element.text,
                    page_number=element.page_number,
                    heading_path=list(element.heading_path),
                    locator=element.locator,
                    element_metadata=dict(element.metadata),
                )
                for element in elements
            )
            # Chunks reference parent elements. Flush their immutable provenance
            # rows first so SQLite and PostgreSQL both enforce the foreign key.
            session.flush()
            session.add_all(
                ChunkRecord(
                    id=chunk.chunk_id,
                    version_id=str(version_id),
                    parent_element_id=(
                        str(chunk.parent_element_id) if chunk.parent_element_id else None
                    ),
                    ordinal=chunk.ordinal,
                    strategy=chunk.strategy.value,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    heading_path=list(chunk.heading_path),
                    locator=chunk.locator,
                    character_count=chunk.character_count,
                    token_count=chunk.token_count,
                    flags=list(chunk.flags),
                )
                for chunk in chunks
            )
            session.flush()
            return _version_response(record)

    def list_documents(self, collection_id: UUID) -> list[DocumentResponse]:
        with self.sessions() as session:
            records = session.scalars(
                select(LogicalDocumentRecord)
                .where(LogicalDocumentRecord.collection_id == str(collection_id))
                .options(selectinload(LogicalDocumentRecord.versions))
                .order_by(LogicalDocumentRecord.created_at)
            )
            return [_document_response(item) for item in records]

    def get_preview(self, version_id: UUID) -> ParserPreviewResponse:
        with self.sessions() as session:
            record = session.scalar(
                select(DocumentVersionRecord)
                .where(DocumentVersionRecord.id == str(version_id))
                .options(
                    selectinload(DocumentVersionRecord.elements),
                    selectinload(DocumentVersionRecord.chunks),
                )
            )
            if record is None:
                raise NotFoundError("Document version not found", component="document")
            return ParserPreviewResponse(
                version=_version_response(record),
                elements=[_element_response(item) for item in record.elements],
                chunks=[_chunk_response(item) for item in record.chunks],
            )

    def soft_delete_document(self, document_id: UUID, reason: str) -> DocumentResponse:
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            record = session.scalar(
                select(LogicalDocumentRecord)
                .where(LogicalDocumentRecord.id == str(document_id))
                .options(selectinload(LogicalDocumentRecord.versions))
            )
            if record is None:
                raise NotFoundError("Document not found", component="document")
            record.deleted_at = now
            for version in record.versions:
                if version.active:
                    version.active = False
                    version.inactive_reason = reason
            session.flush()
            return _document_response(record)


def _collection_response(record: CollectionRecord) -> CollectionResponse:
    return CollectionResponse(id=UUID(record.id), name=record.name, created_at=record.created_at)


def _version_response(record: DocumentVersionRecord) -> DocumentVersionResponse:
    return DocumentVersionResponse(
        id=UUID(record.id),
        document_id=UUID(record.document_id),
        content_sha256=record.content_sha256,
        pipeline_hash=record.pipeline_hash,
        media_type=record.media_type,
        parser_name=record.parser_name,
        parser_version=record.parser_version,
        parser_strategy=ParserStrategy(record.parser_strategy),
        active=record.active,
        inactive_reason=record.inactive_reason,
        status=record.status,
        element_count=record.element_count,
        chunk_count=record.chunk_count,
        warning_count=record.warning_count,
        created_at=record.created_at,
    )


def _document_response(record: LogicalDocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        id=UUID(record.id),
        collection_id=UUID(record.collection_id),
        filename=record.filename,
        display_filename=record.display_filename,
        deleted_at=record.deleted_at,
        created_at=record.created_at,
        versions=[_version_response(version) for version in record.versions],
    )


def _job_response(record: IngestionJobRecord) -> IngestionJobResponse:
    return IngestionJobResponse(
        id=UUID(record.id),
        collection_id=UUID(record.collection_id),
        document_id=UUID(record.document_id) if record.document_id else None,
        version_id=UUID(record.version_id) if record.version_id else None,
        filename=record.filename,
        status=JobStatus(record.status),
        progress=record.progress,
        skipped=record.skipped,
        element_count=record.element_count,
        chunk_count=record.chunk_count,
        warning_count=record.warning_count,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _element_response(record: DocumentElementRecord) -> DocumentElement:
    return DocumentElement(
        element_id=UUID(record.id),
        document_version_id=UUID(record.version_id),
        ordinal=record.ordinal,
        category=record.category,
        text=record.text,
        locator=record.locator,
        page_number=record.page_number,
        heading_path=tuple(record.heading_path),
        metadata=record.element_metadata,
    )


def _chunk_response(record: ChunkRecord) -> IngestedChunk:
    return IngestedChunk(
        chunk_id=record.id,
        document_version_id=UUID(record.version_id),
        parent_element_id=UUID(record.parent_element_id) if record.parent_element_id else None,
        ordinal=record.ordinal,
        strategy=ChunkStrategy(record.strategy),
        text=record.text,
        locator=record.locator,
        page_number=record.page_number,
        heading_path=tuple(record.heading_path),
        character_count=record.character_count,
        token_count=record.token_count,
        flags=tuple(record.flags),
    )
