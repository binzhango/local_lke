"""SQLAlchemy persistence for embedding profiles, nodes, jobs, and images."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Engine, func, select, update
from sqlalchemy.orm import Session, selectinload, sessionmaker

from local_lke.errors import IndexingError, NotFoundError
from local_lke.models import (
    EmbeddingModality,
    EmbeddingProfileResponse,
    ImageAssetResponse,
    IndexingJobResponse,
    JobStatus,
    NodeGranularity,
)
from local_lke.storage.models import (
    ChunkRecord,
    CollectionIndexProfileRecord,
    DocumentElementRecord,
    DocumentVersionRecord,
    EmbeddingProfileRecord,
    ImageAssetRecord,
    ImageEmbeddingRecord,
    IndexingJobRecord,
    LogicalDocumentRecord,
    VectorNodeRecord,
)


@dataclass(frozen=True)
class IndexableChunk:
    id: str
    ordinal: int
    text: str
    locator: str
    token_count: int
    parent_element_id: str | None


@dataclass(frozen=True)
class IndexableVersion:
    collection_id: UUID
    document_id: UUID
    version_id: UUID
    active: bool
    status: str
    chunks: tuple[IndexableChunk, ...]
    parent_text: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class NodeWrite:
    id: str
    collection_id: UUID
    document_id: UUID
    version_id: UUID
    chunk_id: str
    parent_element_id: str | None
    profile_id: UUID
    granularity: NodeGranularity
    unit_ordinal: int
    text: str
    locator: str
    token_count: int
    embedding: list[float]


@dataclass(frozen=True)
class NodeHit:
    id: str
    chunk_id: str
    document_id: UUID
    version_id: UUID
    parent_element_id: str | None
    profile_id: UUID
    granularity: NodeGranularity
    unit_ordinal: int
    text: str
    locator: str
    token_count: int
    score: float


class SqlAlchemyIndexRepository:
    def __init__(self, sessions: sessionmaker[Session], engine: Engine) -> None:
        self.sessions = sessions
        self.engine = engine

    def vector_health(self, expected_dimension: int) -> str:
        if self.engine.dialect.name == "sqlite":
            return f"SQLite deterministic vector fallback ({expected_dimension} dimensions)"
        with self.engine.connect() as connection:
            available = connection.exec_driver_sql(
                "SELECT default_version FROM pg_available_extensions WHERE name='vector'"
            ).scalar_one_or_none()
            if available is None:
                raise IndexingError(
                    "pgvector is not visible to PostgreSQL 18; reinstall or relink the "
                    "Homebrew pgvector formula, then rerun migrations.",
                    code="pgvector_unavailable",
                )
            installed = connection.exec_driver_sql(
                "SELECT extversion FROM pg_extension WHERE extname='vector'"
            ).scalar_one_or_none()
            if installed is None:
                raise IndexingError(
                    "pgvector is available but not enabled; run 'make migrate'.",
                    code="pgvector_not_enabled",
                )
            schema_type = connection.exec_driver_sql(
                "SELECT format_type(a.atttypid, a.atttypmod) "
                "FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid "
                "WHERE c.relname='vector_nodes' AND a.attname='embedding'"
            ).scalar_one_or_none()
            expected_type = f"vector({expected_dimension})"
            if schema_type != expected_type:
                raise IndexingError(
                    f"Database embedding column is {schema_type or 'missing'}, but settings "
                    f"require {expected_type}; create a compatible migration or restore the "
                    "configured dimension.",
                    code="embedding_dimension_mismatch",
                )
            distance = connection.exec_driver_sql(
                "SELECT '[1,0,0]'::vector <=> '[1,0,0]'::vector"
            ).scalar_one()
        if float(distance) != 0:
            raise IndexingError("pgvector cosine round trip failed", code="pgvector_roundtrip")
        return f"pgvector {installed}; {expected_type}; cosine round trip ok"

    def get_or_create_profile(
        self,
        *,
        modality: EmbeddingModality,
        model_id: str,
        revision: str,
        dimension: int,
        normalized: bool,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> EmbeddingProfileResponse:
        with self.sessions.begin() as session:
            statement = select(EmbeddingProfileRecord).where(
                EmbeddingProfileRecord.modality == modality.value,
                EmbeddingProfileRecord.model_id == model_id,
                EmbeddingProfileRecord.revision == revision,
                EmbeddingProfileRecord.dimension == dimension,
                EmbeddingProfileRecord.normalized.is_(normalized),
                EmbeddingProfileRecord.document_prefix == document_prefix,
                EmbeddingProfileRecord.query_prefix == query_prefix,
            )
            record = session.scalar(statement)
            if record is None:
                record = EmbeddingProfileRecord(
                    modality=modality.value,
                    model_id=model_id,
                    revision=revision,
                    dimension=dimension,
                    normalized=normalized,
                    document_prefix=document_prefix,
                    query_prefix=query_prefix,
                )
                session.add(record)
                session.flush()
            return _profile_response(record)

    def get_profile(self, profile_id: UUID) -> EmbeddingProfileResponse:
        with self.sessions() as session:
            record = session.get(EmbeddingProfileRecord, str(profile_id))
            if record is None:
                raise NotFoundError("Embedding profile not found", component="indexing")
            return _profile_response(record)

    def get_active_profile(
        self, collection_id: UUID, modality: EmbeddingModality
    ) -> EmbeddingProfileResponse | None:
        with self.sessions() as session:
            record = session.scalar(
                select(EmbeddingProfileRecord)
                .join(
                    CollectionIndexProfileRecord,
                    CollectionIndexProfileRecord.profile_id == EmbeddingProfileRecord.id,
                )
                .where(
                    CollectionIndexProfileRecord.collection_id == str(collection_id),
                    CollectionIndexProfileRecord.modality == modality.value,
                    CollectionIndexProfileRecord.active.is_(True),
                )
            )
            return _profile_response(record) if record is not None else None

    def activate_profile(
        self, collection_id: UUID, profile_id: UUID, modality: EmbeddingModality
    ) -> None:
        now = datetime.now(UTC)
        with self.sessions.begin() as session:
            session.execute(
                update(CollectionIndexProfileRecord)
                .where(
                    CollectionIndexProfileRecord.collection_id == str(collection_id),
                    CollectionIndexProfileRecord.modality == modality.value,
                )
                .values(active=False)
            )
            mapping = session.get(
                CollectionIndexProfileRecord,
                {"collection_id": str(collection_id), "profile_id": str(profile_id)},
            )
            if mapping is None:
                mapping = CollectionIndexProfileRecord(
                    collection_id=str(collection_id),
                    profile_id=str(profile_id),
                    modality=modality.value,
                    active=True,
                    activated_at=now,
                )
                session.add(mapping)
            else:
                mapping.active = True
                mapping.activated_at = now

    def load_version(self, version_id: UUID) -> IndexableVersion:
        with self.sessions() as session:
            version = session.scalar(
                select(DocumentVersionRecord)
                .where(DocumentVersionRecord.id == str(version_id))
                .options(
                    selectinload(DocumentVersionRecord.chunks),
                    selectinload(DocumentVersionRecord.elements),
                    selectinload(DocumentVersionRecord.document),
                )
            )
            if version is None:
                raise NotFoundError("Document version not found", component="indexing")
            document = version.document
            return IndexableVersion(
                collection_id=UUID(document.collection_id),
                document_id=UUID(document.id),
                version_id=UUID(version.id),
                active=version.active and document.deleted_at is None,
                status=version.status,
                chunks=tuple(
                    IndexableChunk(
                        id=item.id,
                        ordinal=item.ordinal,
                        text=item.text,
                        locator=item.locator,
                        token_count=item.token_count,
                        parent_element_id=item.parent_element_id,
                    )
                    for item in version.chunks
                ),
                parent_text={
                    item.id: (item.text, item.locator)
                    for item in version.elements
                    if item.text.strip()
                },
            )

    def active_version_ids(self, collection_id: UUID) -> list[UUID]:
        with self.sessions() as session:
            values = session.scalars(
                select(DocumentVersionRecord.id)
                .join(
                    LogicalDocumentRecord,
                    LogicalDocumentRecord.id == DocumentVersionRecord.document_id,
                )
                .where(
                    LogicalDocumentRecord.collection_id == str(collection_id),
                    LogicalDocumentRecord.deleted_at.is_(None),
                    DocumentVersionRecord.active.is_(True),
                    DocumentVersionRecord.status == "complete",
                )
            )
            return [UUID(item) for item in values]

    def get_or_create_job(
        self, version: IndexableVersion, profile_id: UUID
    ) -> IndexingJobResponse:
        with self.sessions.begin() as session:
            record = session.scalar(
                select(IndexingJobRecord).where(
                    IndexingJobRecord.version_id == str(version.version_id),
                    IndexingJobRecord.profile_id == str(profile_id),
                )
            )
            if record is None:
                record = IndexingJobRecord(
                    collection_id=str(version.collection_id),
                    version_id=str(version.version_id),
                    profile_id=str(profile_id),
                    status=JobStatus.QUEUED.value,
                )
                session.add(record)
                session.flush()
            return _job_response(record)

    def update_job(self, job_id: UUID, **values: Any) -> IndexingJobResponse:
        values["updated_at"] = datetime.now(UTC)
        with self.sessions.begin() as session:
            record = session.get(IndexingJobRecord, str(job_id))
            if record is None:
                raise NotFoundError("Indexing job not found", component="indexing")
            for name, value in values.items():
                setattr(record, name, value)
            session.flush()
            return _job_response(record)

    def get_job(self, job_id: UUID) -> IndexingJobResponse:
        with self.sessions() as session:
            record = session.get(IndexingJobRecord, str(job_id))
            if record is None:
                raise NotFoundError("Indexing job not found", component="indexing")
            return _job_response(record)

    def list_jobs(self, collection_id: UUID) -> list[IndexingJobResponse]:
        with self.sessions() as session:
            records = session.scalars(
                select(IndexingJobRecord)
                .where(IndexingJobRecord.collection_id == str(collection_id))
                .order_by(IndexingJobRecord.created_at.desc())
            )
            return [_job_response(item) for item in records]

    def existing_node_ids(self, version_id: UUID, profile_id: UUID) -> set[str]:
        with self.sessions() as session:
            return set(
                session.scalars(
                    select(VectorNodeRecord.id).where(
                        VectorNodeRecord.version_id == str(version_id),
                        VectorNodeRecord.profile_id == str(profile_id),
                    )
                )
            )

    def write_nodes(self, nodes: list[NodeWrite]) -> None:
        with self.sessions.begin() as session:
            for node in nodes:
                if session.get(VectorNodeRecord, node.id) is not None:
                    continue
                session.add(
                    VectorNodeRecord(
                        id=node.id,
                        collection_id=str(node.collection_id),
                        document_id=str(node.document_id),
                        version_id=str(node.version_id),
                        chunk_id=node.chunk_id,
                        parent_element_id=node.parent_element_id,
                        profile_id=str(node.profile_id),
                        granularity=node.granularity.value,
                        unit_ordinal=node.unit_ordinal,
                        text=node.text,
                        locator=node.locator,
                        token_count=node.token_count,
                        embedding=node.embedding,
                        active=False,
                    )
                )

    def delete_version_nodes(self, version_id: UUID, profile_id: UUID) -> None:
        with self.sessions.begin() as session:
            session.query(VectorNodeRecord).filter(
                VectorNodeRecord.version_id == str(version_id),
                VectorNodeRecord.profile_id == str(profile_id),
            ).delete(synchronize_session=False)

    def activate_version(
        self,
        version: IndexableVersion,
        profile_id: UUID,
        expected_nodes: int,
    ) -> None:
        with self.sessions.begin() as session:
            count = session.scalar(
                select(func.count(VectorNodeRecord.id)).where(
                    VectorNodeRecord.version_id == str(version.version_id),
                    VectorNodeRecord.profile_id == str(profile_id),
                )
            )
            if int(count or 0) != expected_nodes:
                raise IndexingError(
                    "Incomplete embedding batches cannot become searchable",
                    code="incomplete_index",
                )
            session.execute(
                update(VectorNodeRecord)
                .where(VectorNodeRecord.document_id == str(version.document_id))
                .values(active=False)
            )
            session.execute(
                update(VectorNodeRecord)
                .where(
                    VectorNodeRecord.version_id == str(version.version_id),
                    VectorNodeRecord.profile_id == str(profile_id),
                )
                .values(active=True)
            )

    def search_nodes(
        self,
        *,
        collection_id: UUID,
        profile_id: UUID,
        query_vector: list[float],
        granularities: tuple[NodeGranularity, ...],
        limit: int,
    ) -> list[NodeHit]:
        with self.sessions() as session:
            statement = (
                select(VectorNodeRecord, DocumentVersionRecord)
                .join(
                    DocumentVersionRecord,
                    DocumentVersionRecord.id == VectorNodeRecord.version_id,
                )
                .join(
                    LogicalDocumentRecord,
                    LogicalDocumentRecord.id == VectorNodeRecord.document_id,
                )
                .where(
                    VectorNodeRecord.collection_id == str(collection_id),
                    VectorNodeRecord.profile_id == str(profile_id),
                    VectorNodeRecord.granularity.in_([item.value for item in granularities]),
                    VectorNodeRecord.active.is_(True),
                    DocumentVersionRecord.active.is_(True),
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
            )
            if self.engine.dialect.name == "postgresql":
                distance = VectorNodeRecord.embedding.cosine_distance(query_vector)
                pg_rows = session.execute(
                    statement.add_columns(distance.label("distance"))
                    .order_by(distance, VectorNodeRecord.id)
                    .limit(limit)
                )
                return [_node_hit(row[0], 1 - float(row[2])) for row in pg_rows]
            rows = session.execute(statement).all()
            scored = [
                _node_hit(record, _cosine(query_vector, list(record.embedding)))
                for record, _version in rows
            ]
            scored.sort(key=lambda item: (-item.score, item.id))
            return scored[:limit]

    def sentence_window(self, hit: NodeHit, radius: int) -> tuple[str, int]:
        with self.sessions() as session:
            records = session.scalars(
                select(VectorNodeRecord)
                .join(ChunkRecord, ChunkRecord.id == VectorNodeRecord.chunk_id)
                .where(
                    VectorNodeRecord.profile_id == str(hit.profile_id),
                    VectorNodeRecord.version_id == str(hit.version_id),
                    VectorNodeRecord.granularity == NodeGranularity.SENTENCE.value,
                    VectorNodeRecord.active.is_(True),
                )
                .order_by(ChunkRecord.ordinal, VectorNodeRecord.unit_ordinal)
            ).all()
            position = next(
                (index for index, item in enumerate(records) if item.id == hit.id),
                None,
            )
            if position is None:
                return hit.text, hit.token_count
            window = records[max(0, position - radius) : position + radius + 1]
            return " ".join(item.text for item in window), sum(
                item.token_count for item in window
            )

    def parent_context(self, hit: NodeHit) -> tuple[str, str, int] | None:
        if hit.parent_element_id is None:
            return None
        with self.sessions() as session:
            parent = session.get(DocumentElementRecord, hit.parent_element_id)
            if parent is None or not parent.text.strip():
                return None
            return parent.text, parent.locator, max(1, len(parent.text.split()))

    def index_counts(self, collection_id: UUID, profile_id: UUID | None) -> tuple[int, int, int]:
        if profile_id is None:
            return 0, 0, self._active_chunk_count(collection_id)
        with self.sessions() as session:
            active_nodes = int(
                session.scalar(
                    select(func.count(VectorNodeRecord.id)).where(
                        VectorNodeRecord.collection_id == str(collection_id),
                        VectorNodeRecord.profile_id == str(profile_id),
                        VectorNodeRecord.active.is_(True),
                    )
                )
                or 0
            )
            active_chunks = int(
                session.scalar(
                    select(func.count(func.distinct(VectorNodeRecord.chunk_id))).where(
                        VectorNodeRecord.collection_id == str(collection_id),
                        VectorNodeRecord.profile_id == str(profile_id),
                        VectorNodeRecord.granularity == NodeGranularity.CHUNK.value,
                        VectorNodeRecord.active.is_(True),
                    )
                )
                or 0
            )
        missing = max(0, self._active_chunk_count(collection_id) - active_chunks)
        return active_nodes, active_chunks, missing

    def _active_chunk_count(self, collection_id: UUID) -> int:
        with self.sessions() as session:
            return int(
                session.scalar(
                    select(func.count(ChunkRecord.id))
                    .join(
                        DocumentVersionRecord,
                        DocumentVersionRecord.id == ChunkRecord.version_id,
                    )
                    .join(
                        LogicalDocumentRecord,
                        LogicalDocumentRecord.id == DocumentVersionRecord.document_id,
                    )
                    .where(
                        LogicalDocumentRecord.collection_id == str(collection_id),
                        LogicalDocumentRecord.deleted_at.is_(None),
                        DocumentVersionRecord.active.is_(True),
                    )
                )
                or 0
            )

    def find_image(self, collection_id: UUID, sha256: str) -> ImageAssetResponse | None:
        with self.sessions() as session:
            record = session.scalar(
                select(ImageAssetRecord).where(
                    ImageAssetRecord.collection_id == str(collection_id),
                    ImageAssetRecord.sha256 == sha256,
                    ImageAssetRecord.deleted_at.is_(None),
                )
            )
            return _image_response(record) if record is not None else None

    def create_image(self, **values: Any) -> ImageAssetResponse:
        with self.sessions.begin() as session:
            record = ImageAssetRecord(**values)
            session.add(record)
            session.flush()
            return _image_response(record)

    def get_image_path(self, image_id: UUID) -> Path:
        with self.sessions() as session:
            record = session.get(ImageAssetRecord, str(image_id))
            if record is None or record.deleted_at is not None:
                raise NotFoundError("Image not found", component="multimodal")
            return Path(record.storage_path)

    def has_image_embedding(self, image_id: UUID, profile_id: UUID) -> bool:
        with self.sessions() as session:
            return (
                session.get(
                    ImageEmbeddingRecord,
                    {"image_id": str(image_id), "profile_id": str(profile_id)},
                )
                is not None
            )

    def write_image_embedding(
        self, image_id: UUID, profile_id: UUID, embedding: list[float]
    ) -> None:
        with self.sessions.begin() as session:
            session.add(
                ImageEmbeddingRecord(
                    image_id=str(image_id), profile_id=str(profile_id), embedding=embedding
                )
            )

    def search_images(
        self, collection_id: UUID, profile_id: UUID, vector: list[float], limit: int
    ) -> list[tuple[ImageAssetResponse, float]]:
        with self.sessions() as session:
            statement = (
                select(ImageAssetRecord, ImageEmbeddingRecord)
                .join(ImageEmbeddingRecord, ImageEmbeddingRecord.image_id == ImageAssetRecord.id)
                .where(
                    ImageAssetRecord.collection_id == str(collection_id),
                    ImageAssetRecord.deleted_at.is_(None),
                    ImageEmbeddingRecord.profile_id == str(profile_id),
                )
            )
            if self.engine.dialect.name == "postgresql":
                distance = ImageEmbeddingRecord.embedding.cosine_distance(vector)
                pg_rows = session.execute(
                    statement.add_columns(distance).order_by(distance).limit(limit)
                )
                return [(_image_response(row[0]), 1 - float(row[2])) for row in pg_rows]
            rows = session.execute(statement).all()
            scored = [
                (_image_response(image), _cosine(vector, list(embedding.embedding)))
                for image, embedding in rows
            ]
            scored.sort(key=lambda item: (-item[1], str(item[0].id)))
            return scored[:limit]


def _profile_response(record: EmbeddingProfileRecord) -> EmbeddingProfileResponse:
    return EmbeddingProfileResponse(
        id=UUID(record.id),
        modality=EmbeddingModality(record.modality),
        model_id=record.model_id,
        revision=record.revision,
        dimension=record.dimension,
        normalized=record.normalized,
        document_prefix=record.document_prefix,
        query_prefix=record.query_prefix,
        created_at=record.created_at,
    )


def _job_response(record: IndexingJobRecord) -> IndexingJobResponse:
    return IndexingJobResponse(
        id=UUID(record.id),
        collection_id=UUID(record.collection_id),
        version_id=UUID(record.version_id),
        profile_id=UUID(record.profile_id),
        status=JobStatus(record.status),
        progress=record.progress,
        total_nodes=record.total_nodes,
        embedded_nodes=record.embedded_nodes,
        embedding_calls=record.embedding_calls,
        skipped=record.skipped,
        error_code=record.error_code,
        error_message=record.error_message,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _node_hit(record: VectorNodeRecord, score: float) -> NodeHit:
    return NodeHit(
        id=record.id,
        chunk_id=record.chunk_id,
        document_id=UUID(record.document_id),
        version_id=UUID(record.version_id),
        parent_element_id=record.parent_element_id,
        profile_id=UUID(record.profile_id),
        granularity=NodeGranularity(record.granularity),
        unit_ordinal=record.unit_ordinal,
        text=record.text,
        locator=record.locator,
        token_count=record.token_count,
        score=score,
    )


def _image_response(record: ImageAssetRecord) -> ImageAssetResponse:
    return ImageAssetResponse(
        id=UUID(record.id),
        collection_id=UUID(record.collection_id),
        filename=record.filename,
        media_type=record.media_type,
        width=record.width,
        height=record.height,
        sha256=record.sha256,
        created_at=record.created_at,
        content_url=f"/api/v1/images/{record.id}/content",
    )


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return -1.0
    denominator = math.sqrt(sum(item * item for item in left)) * math.sqrt(
        sum(item * item for item in right)
    )
    if not denominator:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / denominator
