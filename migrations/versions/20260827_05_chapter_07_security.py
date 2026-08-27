"""Add Chapter 7 collection ACLs and metadata-only audit events."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_05"
down_revision: str | None = "20260827_04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "collection_access",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("collection_id", sa.String(36), nullable=False),
        sa.Column("principal_id", sa.String(120), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("granted_by", sa.String(120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["collection_id"], ["collections.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "collection_id", "principal_id", name="uq_collection_access_principal"
        ),
        sa.CheckConstraint(
            "role IN ('owner', 'editor', 'viewer')",
            name="ck_collection_access_role",
        ),
    )
    op.create_index("ix_collection_access_collection_id", "collection_access", ["collection_id"])
    op.create_index("ix_collection_access_principal_id", "collection_access", ["principal_id"])
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("principal_id", sa.String(120), nullable=False),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("resource_kind", sa.String(40), nullable=False),
        sa.Column("resource_id", sa.String(120)),
        sa.Column("outcome", sa.String(20), nullable=False),
        sa.Column("detail", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "outcome IN ('allowed', 'denied')",
            name="ck_audit_events_outcome",
        ),
    )
    op.create_index("ix_audit_events_principal_id", "audit_events", ["principal_id"])
    op.create_index("ix_audit_events_action", "audit_events", ["action"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_audit_events_created_at", table_name="audit_events")
    op.drop_index("ix_audit_events_action", table_name="audit_events")
    op.drop_index("ix_audit_events_principal_id", table_name="audit_events")
    op.drop_table("audit_events")
    op.drop_index("ix_collection_access_principal_id", table_name="collection_access")
    op.drop_index("ix_collection_access_collection_id", table_name="collection_access")
    op.drop_table("collection_access")
