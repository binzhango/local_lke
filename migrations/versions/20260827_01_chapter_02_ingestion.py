"""Create the Chapter 2 versioned-ingestion schema."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_01"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "collections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False, unique=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "pipeline_configurations",
        sa.Column("pipeline_hash", sa.String(64), primary_key=True),
        sa.Column("schema_version", sa.String(32), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("display_filename", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("collection_id", "filename", name="uq_documents_collection_filename"),
    )
    op.create_index("ix_documents_collection_id", "documents", ["collection_id"])
    op.create_table(
        "document_versions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("pipeline_hash", sa.String(64), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("parser_name", sa.String(100), nullable=False),
        sa.Column("parser_version", sa.String(50), nullable=False),
        sa.Column("parser_strategy", sa.String(20), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("inactive_reason", sa.String(255)),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("element_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["pipeline_hash"], ["pipeline_configurations.pipeline_hash"]),
        sa.UniqueConstraint(
            "document_id",
            "content_sha256",
            "pipeline_hash",
            name="uq_versions_idempotency",
        ),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index(
        "uq_versions_one_active",
        "document_versions",
        ["document_id"],
        unique=True,
        postgresql_where=sa.text("active = true"),
        sqlite_where=sa.text("active = 1"),
    )
    op.create_table(
        "document_elements",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("category", sa.String(80), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("heading_path", sa.JSON(), nullable=False),
        sa.Column("locator", sa.String(255), nullable=False),
        sa.Column("element_metadata", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("version_id", "ordinal", name="uq_elements_order"),
    )
    op.create_index("ix_document_elements_version_id", "document_elements", ["version_id"])
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("parent_element_id", sa.String(36)),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("strategy", sa.String(30), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer()),
        sa.Column("heading_path", sa.JSON(), nullable=False),
        sa.Column("locator", sa.String(255), nullable=False),
        sa.Column("character_count", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("flags", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["parent_element_id"], ["document_elements.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("version_id", "ordinal", name="uq_chunks_order"),
    )
    op.create_index("ix_chunks_version_id", "chunks", ["version_id"])
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("document_id", sa.String(36)),
        sa.Column("version_id", sa.String(36)),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("display_filename", sa.String(255), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("parser_strategy", sa.String(20), nullable=False),
        sa.Column("chunk_strategy", sa.String(30), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("skipped", sa.Boolean(), nullable=False),
        sa.Column("element_count", sa.Integer(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column("warning_count", sa.Integer(), nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"]),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"]),
    )
    op.create_index("ix_ingestion_jobs_collection_id", "ingestion_jobs", ["collection_id"])


def downgrade() -> None:
    op.drop_index("ix_ingestion_jobs_collection_id", table_name="ingestion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_index("ix_chunks_version_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_index("ix_document_elements_version_id", table_name="document_elements")
    op.drop_table("document_elements")
    op.drop_index("uq_versions_one_active", table_name="document_versions")
    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_table("document_versions")
    op.drop_index("ix_documents_collection_id", table_name="documents")
    op.drop_table("documents")
    op.drop_table("pipeline_configurations")
    op.drop_table("collections")
