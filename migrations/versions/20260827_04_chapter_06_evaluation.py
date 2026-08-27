"""Add Chapter 6 immutable evaluation datasets and persisted runs."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260827_04"
down_revision: str | None = "20260827_03"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "evaluation_datasets",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("cases", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("name", "version", name="uq_evaluation_dataset_version"),
        sa.UniqueConstraint("content_sha256", name="uq_evaluation_dataset_content"),
    )
    op.create_index("ix_evaluation_datasets_name", "evaluation_datasets", ["name"])
    op.create_table(
        "evaluation_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("dataset_id", sa.String(36), nullable=False),
        sa.Column("baseline_run_id", sa.String(36)),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("configuration_sha256", sa.String(64), nullable=False),
        sa.Column("configuration", sa.JSON(), nullable=False),
        sa.Column("metrics", sa.JSON(), nullable=False),
        sa.Column("case_results", sa.JSON(), nullable=False),
        sa.Column("gate", sa.JSON(), nullable=False),
        sa.Column("error_message", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["evaluation_datasets.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["baseline_run_id"], ["evaluation_runs.id"], ondelete="RESTRICT"
        ),
    )
    op.create_index("ix_evaluation_runs_dataset_id", "evaluation_runs", ["dataset_id"])


def downgrade() -> None:
    op.drop_index("ix_evaluation_runs_dataset_id", table_name="evaluation_runs")
    op.drop_table("evaluation_runs")
    op.drop_index("ix_evaluation_datasets_name", table_name="evaluation_datasets")
    op.drop_table("evaluation_datasets")
