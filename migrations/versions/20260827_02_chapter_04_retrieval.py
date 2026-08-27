"""Add Chapter 4 lexical retrieval and structured-table metadata."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_02"
down_revision: str | None = "20260827_01"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute(
            "ALTER TABLE chunks ADD COLUMN search_vector tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', coalesce(text, ''))) STORED"
        )
        op.execute(
            "CREATE INDEX ix_chunks_search_vector_gin ON chunks USING gin (search_vector)"
        )
    op.create_table(
        "structured_tables",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False, unique=True),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("physical_name", sa.String(63), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("schema_definition", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint(
            "collection_id", "physical_name", name="uq_structured_physical_name"
        ),
    )
    op.create_index("ix_structured_tables_collection_id", "structured_tables", ["collection_id"])


def downgrade() -> None:
    op.drop_index("ix_structured_tables_collection_id", table_name="structured_tables")
    op.drop_table("structured_tables")
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_chunks_search_vector_gin")
        op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS search_vector")
