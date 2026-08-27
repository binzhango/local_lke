from __future__ import annotations

import pytest
from pydantic import ValidationError

from local_lke.errors import RetrievalError
from local_lke.ingestion import IngestionService
from local_lke.models import (
    MetadataOperator,
    StructuredAggregation,
    StructuredFilter,
    StructuredOrder,
    StructuredQueryPlan,
    StructuredQueryRequest,
)
from local_lke.providers import FakeChatProvider
from local_lke.retrieval import StructuredDataService, StructuredPlanParser
from local_lke.settings import Settings
from local_lke.storage import SqlAlchemyIngestionRepository


def _service(
    ingestion: IngestionService, settings: Settings
) -> StructuredDataService:
    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    return StructuredDataService(
        ingestion.repository,
        settings,
        StructuredPlanParser(FakeChatProvider()),
    )


def test_csv_schema_inference_aggregate_query_and_provenance(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Sales")
    service = _service(ingestion, settings)
    table = service.ingest_csv(
        collection_id=collection.id,
        filename="sales.csv",
        content=(
            b"Region,Revenue,Active,Sale Date\n"
            b"East,100,true,2026-01-01\n"
            b"West,250,false,2026-01-02\n"
            b"East,75,true,2026-01-03\n"
        ),
    )
    response = service.query(
        StructuredQueryRequest(
            table_id=table.id,
            question="What is total revenue by region?",
            plan=StructuredQueryPlan(
                group_by=["region"],
                aggregations=[
                    StructuredAggregation(function="sum", column="revenue", alias="total")
                ],
                order_by=[StructuredOrder(column="total", direction="desc")],
                limit=10,
            ),
        )
    )

    types = {column.name: column.data_type for column in table.columns}
    assert types == {
        "region": "text",
        "revenue": "integer",
        "active": "boolean",
        "sale_date": "date",
    }
    assert response.rows == [
        {"region": "West", "total": 250},
        {"region": "East", "total": 175},
    ]
    assert "SELECT" in response.sql_preview.upper()
    assert response.provenance["version_id"] == str(table.version_id)
    assert response.truncated is False


def test_structured_filter_and_hard_result_limit(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Limits")
    service = _service(ingestion, settings)
    table = service.ingest_csv(
        collection_id=collection.id,
        filename="items.csv",
        content=b"name,amount\na,1\nb,2\nc,3\n",
    )
    response = service.query(
        StructuredQueryRequest(
            table_id=table.id,
            question="Amounts above one",
            plan=StructuredQueryPlan(
                projections=["name", "amount"],
                filters=[
                    StructuredFilter(column="amount", operator=MetadataOperator.GT, value=1)
                ],
                order_by=[StructuredOrder(column="amount")],
                limit=1,
            ),
        )
    )
    assert response.rows == [{"name": "b", "amount": 2}]
    assert response.truncated is True


def test_structured_plans_reject_raw_sql_and_unknown_columns(
    ingestion: IngestionService, settings: Settings
) -> None:
    with pytest.raises(ValidationError):
        StructuredQueryPlan.model_validate({"raw_sql": "DROP TABLE collections"})

    collection = ingestion.create_collection("Safety")
    service = _service(ingestion, settings)
    table = service.ingest_csv(
        collection_id=collection.id,
        filename="safe.csv",
        content=b"name,value\nalpha,1\n",
    )
    with pytest.raises(RetrievalError, match="unknown columns"):
        service.query(
            StructuredQueryRequest(
                table_id=table.id,
                question="Unsafe",
                plan=StructuredQueryPlan(projections=["name; DROP TABLE collections"]),
            )
        )

    injection = service.query(
        StructuredQueryRequest(
            table_id=table.id,
            question="Literal injection attempt",
            plan=StructuredQueryPlan(
                projections=["name"],
                filters=[
                    StructuredFilter(
                        column="name",
                        operator=MetadataOperator.EQ,
                        value="' OR 1=1 --",
                    )
                ],
            ),
        )
    )
    assert injection.rows == []
    assert service.list_tables(collection.id)[0].id == table.id

    with pytest.raises(RetrievalError, match="does not match integer"):
        service.query(
            StructuredQueryRequest(
                table_id=table.id,
                question="Wrong type",
                plan=StructuredQueryPlan(
                    filters=[
                        StructuredFilter(
                            column="value",
                            operator=MetadataOperator.GT,
                            value="not-a-number",
                        )
                    ]
                ),
            )
        )


def test_csv_validation_rejects_duplicates_and_excess_rows(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("CSV validation")
    service = _service(ingestion, settings)
    with pytest.raises(RetrievalError, match="unique"):
        service.ingest_csv(
            collection_id=collection.id,
            filename="duplicate.csv",
            content=b"Name,name\na,b\n",
        )

    limited = settings.model_copy(update={"structured_max_csv_rows": 1})
    with pytest.raises(RetrievalError, match="too many rows"):
        _service(ingestion, limited).ingest_csv(
            collection_id=collection.id,
            filename="large.csv",
            content=b"name\na\nb\n",
        )


def test_structured_queries_follow_active_version_and_delete_lifecycle(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Structured lifecycle")
    service = _service(ingestion, settings)
    old = service.ingest_csv(
        collection_id=collection.id,
        filename="status.csv",
        content=b"name,value\nold,1\n",
    )
    current = service.ingest_csv(
        collection_id=collection.id,
        filename="status.csv",
        content=b"name,value\ncurrent,2\n",
    )
    plan = StructuredQueryPlan(projections=["name", "value"])

    with pytest.raises(RetrievalError, match="not found"):
        service.query(
            StructuredQueryRequest(table_id=old.id, question="Old", plan=plan)
        )
    assert service.query(
        StructuredQueryRequest(table_id=current.id, question="Current", plan=plan)
    ).rows == [{"name": "current", "value": 2}]

    ingestion.delete_document(current.document_id, "structured lifecycle test")
    assert service.list_tables(collection.id) == []
    with pytest.raises(RetrievalError, match="not found"):
        service.query(
            StructuredQueryRequest(table_id=current.id, question="Deleted", plan=plan)
        )
