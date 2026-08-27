"""Add Chapter 3 pgvector indexes and multimodal assets after Chapter 4."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20260827_03"
down_revision: str | None = "20260827_02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TEXT_DIMENSION = 384
MULTIMODAL_DIMENSION = 512
VECTOR_INDEX_COLUMNS = (
    "collection_id",
    "document_id",
    "version_id",
    "chunk_id",
    "profile_id",
    "active",
)


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embedding_profiles",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("modality", sa.String(20), nullable=False),
        sa.Column("model_id", sa.String(255), nullable=False),
        sa.Column("revision", sa.String(120), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("normalized", sa.Boolean(), nullable=False),
        sa.Column("document_prefix", sa.Text(), nullable=False),
        sa.Column("query_prefix", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint(
            "modality", "model_id", "revision", "dimension", "normalized",
            "document_prefix", "query_prefix", name="uq_embedding_profile_contract",
        ),
    )
    op.create_table(
        "collection_index_profiles",
        sa.Column("collection_id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), primary_key=True),
        sa.Column("modality", sa.String(20), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("collection_id", "profile_id", name="uq_collection_profile"),
    )
    op.create_index(
        "uq_collection_one_active_profile_per_modality",
        "collection_index_profiles",
        ["collection_id", "modality"],
        unique=True,
        postgresql_where=sa.text("active = true"),
        sqlite_where=sa.text("active = 1"),
    )
    vector_type: sa.types.TypeEngine[object]
    image_vector_type: sa.types.TypeEngine[object]
    if bind.dialect.name == "postgresql":
        vector_type = Vector(TEXT_DIMENSION)
        image_vector_type = Vector(MULTIMODAL_DIMENSION)
    else:
        vector_type = sa.JSON()
        image_vector_type = sa.JSON()
    op.create_table(
        "vector_nodes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("document_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False),
        sa.Column("parent_element_id", sa.String(36)),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("granularity", sa.String(20), nullable=False),
        sa.Column("unit_ordinal", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("locator", sa.String(255), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("embedding", vector_type, nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["chunk_id"], ["chunks.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["parent_element_id"], ["document_elements.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="RESTRICT"),
    )
    for column in VECTOR_INDEX_COLUMNS:
        op.create_index(f"ix_vector_nodes_{column}", "vector_nodes", [column])
    if bind.dialect.name == "postgresql":
        op.create_index(
            "ix_vector_nodes_embedding_hnsw", "vector_nodes", ["embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )
    op.create_table(
        "indexing_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("version_id", sa.String(36), nullable=False),
        sa.Column("profile_id", sa.String(36), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False),
        sa.Column("total_nodes", sa.Integer(), nullable=False),
        sa.Column("embedded_nodes", sa.Integer(), nullable=False),
        sa.Column("embedding_calls", sa.Integer(), nullable=False),
        sa.Column("skipped", sa.Boolean(), nullable=False),
        sa.Column("error_code", sa.String(80)),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["version_id"], ["document_versions.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("version_id", "profile_id", name="uq_index_job_version_profile"),
    )
    op.create_index("ix_indexing_jobs_collection_id", "indexing_jobs", ["collection_id"])
    op.create_index("ix_indexing_jobs_version_id", "indexing_jobs", ["version_id"])
    op.create_table(
        "image_assets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(100), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("content_fingerprint", sa.LargeBinary(), nullable=False),
        sa.Column("deleted_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("collection_id", "sha256", name="uq_image_collection_content"),
    )
    op.create_index("ix_image_assets_collection_id", "image_assets", ["collection_id"])
    op.create_table(
        "image_embeddings",
        sa.Column("image_id", sa.String(36), primary_key=True),
        sa.Column("profile_id", sa.String(36), primary_key=True),
        sa.Column("embedding", image_vector_type, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["image_assets.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["profile_id"], ["embedding_profiles.id"], ondelete="RESTRICT"),
        sa.UniqueConstraint("image_id", "profile_id", name="uq_image_embedding_profile"),
    )
    if bind.dialect.name == "postgresql":
        op.create_index(
            "ix_image_embeddings_embedding_hnsw", "image_embeddings", ["embedding"],
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.drop_index("ix_image_embeddings_embedding_hnsw", table_name="image_embeddings")
    op.drop_table("image_embeddings")
    op.drop_index("ix_image_assets_collection_id", table_name="image_assets")
    op.drop_table("image_assets")
    op.drop_index("ix_indexing_jobs_version_id", table_name="indexing_jobs")
    op.drop_index("ix_indexing_jobs_collection_id", table_name="indexing_jobs")
    op.drop_table("indexing_jobs")
    if bind.dialect.name == "postgresql":
        op.drop_index("ix_vector_nodes_embedding_hnsw", table_name="vector_nodes")
    for column in reversed(VECTOR_INDEX_COLUMNS):
        op.drop_index(f"ix_vector_nodes_{column}", table_name="vector_nodes")
    op.drop_table("vector_nodes")
    op.drop_index(
        "uq_collection_one_active_profile_per_modality",
        table_name="collection_index_profiles",
    )
    op.drop_table("collection_index_profiles")
    op.drop_table("embedding_profiles")
