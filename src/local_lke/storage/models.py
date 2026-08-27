"""Chapter 2 relational schema represented as SQLAlchemy models."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(UTC)


def uuid_string() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    pass


class CollectionRecord(Base):
    __tablename__ = "collections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    documents: Mapped[list["LogicalDocumentRecord"]] = relationship(back_populates="collection")


class LogicalDocumentRecord(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection_id", "filename", name="uq_documents_collection_filename"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    display_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    collection: Mapped[CollectionRecord] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersionRecord"]] = relationship(
        back_populates="document", order_by="DocumentVersionRecord.created_at"
    )


class PipelineConfigurationRecord(Base):
    __tablename__ = "pipeline_configurations"

    pipeline_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    configuration: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class DocumentVersionRecord(Base):
    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "content_sha256",
            "pipeline_hash",
            name="uq_versions_idempotency",
        ),
        Index(
            "uq_versions_one_active",
            "document_id",
            unique=True,
            postgresql_where=text("active = true"),
            sqlite_where=text("active = 1"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_hash: Mapped[str] = mapped_column(
        ForeignKey("pipeline_configurations.pipeline_hash"), nullable=False
    )
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_name: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(50), nullable=False)
    parser_strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    inactive_reason: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(30), default="complete", nullable=False)
    element_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[LogicalDocumentRecord] = relationship(back_populates="versions")
    elements: Mapped[list["DocumentElementRecord"]] = relationship(
        back_populates="version", order_by="DocumentElementRecord.ordinal"
    )
    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="version", order_by="ChunkRecord.ordinal"
    )


class DocumentElementRecord(Base):
    __tablename__ = "document_elements"
    __table_args__ = (UniqueConstraint("version_id", "ordinal", name="uq_elements_order"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    locator: Mapped[str] = mapped_column(String(255), nullable=False)
    element_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    version: Mapped[DocumentVersionRecord] = relationship(back_populates="elements")


class ChunkRecord(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("version_id", "ordinal", name="uq_chunks_order"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    parent_element_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_elements.id", ondelete="RESTRICT")
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy: Mapped[str] = mapped_column(String(30), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    locator: Mapped[str] = mapped_column(String(255), nullable=False)
    character_count: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    flags: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    version: Mapped[DocumentVersionRecord] = relationship(back_populates="chunks")


class IngestionJobRecord(Base):
    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id"))
    version_id: Mapped[str | None] = mapped_column(ForeignKey("document_versions.id"))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    display_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_path: Mapped[str] = mapped_column(Text, nullable=False)
    media_type: Mapped[str] = mapped_column(String(100), nullable=False)
    parser_strategy: Mapped[str] = mapped_column(String(20), nullable=False)
    chunk_strategy: Mapped[str] = mapped_column(String(30), nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="queued", nullable=False)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    skipped: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    element_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    warning_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(80))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class StructuredTableRecord(Base):
    __tablename__ = "structured_tables"
    __table_args__ = (
        UniqueConstraint("collection_id", "physical_name", name="uq_structured_physical_name"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=uuid_string)
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("collections.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="RESTRICT"), nullable=False
    )
    version_id: Mapped[str] = mapped_column(
        ForeignKey("document_versions.id", ondelete="RESTRICT"), nullable=False, unique=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    physical_name: Mapped[str] = mapped_column(String(63), nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_definition: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
