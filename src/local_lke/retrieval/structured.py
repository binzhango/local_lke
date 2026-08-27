"""Safe CSV ingestion and allowlisted SQLAlchemy Core structured queries."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from io import StringIO
from typing import Any, Literal
from uuid import UUID, uuid4

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    asc,
    desc,
    func,
    select,
    update,
)
from sqlalchemy.exc import SQLAlchemyError

from local_lke.errors import RetrievalError
from local_lke.ingestion.safety import normalize_filename
from local_lke.models import (
    MetadataOperator,
    StructuredColumn,
    StructuredQueryPlan,
    StructuredQueryRequest,
    StructuredQueryResponse,
    StructuredTableResponse,
)
from local_lke.providers import ChatProvider
from local_lke.retrieval.planning import parse_model_plan
from local_lke.settings import Settings
from local_lke.storage.models import (
    DocumentVersionRecord,
    LogicalDocumentRecord,
    PipelineConfigurationRecord,
    StructuredTableRecord,
)
from local_lke.storage.repository import SqlAlchemyIngestionRepository

CSV_PIPELINE = {
    "schema_version": "chapter-04-csv-v1",
    "parser": "python-csv-rfc4180",
    "type_inference": "bounded-column-v1",
}
CSV_PIPELINE_HASH = hashlib.sha256(
    json.dumps(CSV_PIPELINE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class StructuredPlanParser:
    """Ask a model for JSON, then reduce it to a Pydantic allowlisted plan."""

    def __init__(self, chat: ChatProvider) -> None:
        self.chat = chat

    def plan(
        self, question: str, table: StructuredTableResponse
    ) -> StructuredQueryPlan:
        schema = "\n".join(
            f"- {column.name}: {column.data_type}; {column.description}"
            for column in table.columns
        )
        prompt = f"""Translate the question into JSON only. Never output SQL.
Allowed keys: projections, filters, group_by, aggregations, order_by, limit.
Filter operators: eq, ne, in, contains, gt, gte, lt, lte.
Aggregation functions: count, sum, avg, min, max.
Use only these columns:
{schema}

Question: {question}
JSON plan:"""
        return parse_model_plan(self.chat.generate(prompt), StructuredQueryPlan)


class StructuredDataService:
    def __init__(
        self,
        repository: SqlAlchemyIngestionRepository,
        settings: Settings,
        planner: StructuredPlanParser,
    ) -> None:
        self.repository = repository
        self.settings = settings
        self.planner = planner

    def ingest_csv(
        self,
        *,
        collection_id: UUID,
        filename: str,
        content: bytes,
    ) -> StructuredTableResponse:
        self.repository.get_collection(collection_id)
        safe_name = normalize_filename(filename)
        if not safe_name.casefold().endswith(".csv"):
            raise RetrievalError("Structured uploads must use .csv", code="unsupported_type")
        if not content or len(content) > self.settings.max_upload_bytes:
            raise RetrievalError(
                "CSV is empty or exceeds the configured upload limit",
                code="invalid_csv_size",
            )
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise RetrievalError("CSV must use UTF-8 encoding", code="invalid_encoding") from exc
        source_headers, rows = _read_csv(decoded, self.settings)
        columns = _infer_columns(source_headers, rows)
        typed_rows = [_typed_row(row, columns) for row in rows]
        content_hash = hashlib.sha256(content).hexdigest()

        with self.repository.sessions.begin() as session:
            existing = session.scalar(
                select(StructuredTableRecord)
                .join(
                    DocumentVersionRecord,
                    StructuredTableRecord.version_id == DocumentVersionRecord.id,
                )
                .join(
                    LogicalDocumentRecord,
                    DocumentVersionRecord.document_id == LogicalDocumentRecord.id,
                )
                .where(
                    StructuredTableRecord.collection_id == str(collection_id),
                    StructuredTableRecord.filename == safe_name.casefold(),
                    StructuredTableRecord.content_sha256 == content_hash,
                    DocumentVersionRecord.active.is_(True),
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
                .order_by(StructuredTableRecord.created_at.desc())
            )
            if existing is not None:
                return _table_response(existing)

            document = session.scalar(
                select(LogicalDocumentRecord).where(
                    LogicalDocumentRecord.collection_id == str(collection_id),
                    LogicalDocumentRecord.filename == safe_name.casefold(),
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
            )
            if document is None:
                document = LogicalDocumentRecord(
                    collection_id=str(collection_id),
                    filename=safe_name.casefold(),
                    display_filename=safe_name,
                )
                session.add(document)
                session.flush()
            configuration = session.get(PipelineConfigurationRecord, CSV_PIPELINE_HASH)
            if configuration is None:
                session.add(
                    PipelineConfigurationRecord(
                        pipeline_hash=CSV_PIPELINE_HASH,
                        schema_version=str(CSV_PIPELINE["schema_version"]),
                        configuration=CSV_PIPELINE,
                    )
                )
            session.execute(
                update(DocumentVersionRecord)
                .where(
                    DocumentVersionRecord.document_id == document.id,
                    DocumentVersionRecord.active.is_(True),
                )
                .values(active=False, inactive_reason="superseded by a newer CSV version")
            )
            table_id = uuid4()
            version_id = uuid4()
            physical_name = f"lke_csv_{table_id.hex[:24]}"
            version = DocumentVersionRecord(
                id=str(version_id),
                document_id=document.id,
                content_sha256=content_hash,
                pipeline_hash=CSV_PIPELINE_HASH,
                media_type="text/csv",
                parser_name="python-csv",
                parser_version="stdlib",
                parser_strategy="fast",
                storage_path="database://structured-table",
                active=True,
                status="complete",
                element_count=len(rows),
                chunk_count=0,
                warning_count=0,
            )
            session.add(version)
            session.flush()

            sql_table = _sql_table(physical_name, columns)
            connection = session.connection()
            sql_table.create(connection)
            if typed_rows:
                connection.execute(
                    sql_table.insert(),
                    [
                        {
                            **row,
                            "_lke_row_number": index,
                            "_lke_version_id": str(version_id),
                        }
                        for index, row in enumerate(typed_rows, start=1)
                    ],
                )
            record = StructuredTableRecord(
                id=str(table_id),
                collection_id=str(collection_id),
                document_id=document.id,
                version_id=str(version_id),
                filename=safe_name.casefold(),
                physical_name=physical_name,
                content_sha256=content_hash,
                row_count=len(rows),
                schema_definition=[column.model_dump(mode="json") for column in columns],
            )
            session.add(record)
            session.flush()
            return _table_response(record)

    def list_tables(self, collection_id: UUID) -> list[StructuredTableResponse]:
        self.repository.get_collection(collection_id)
        with self.repository.sessions() as session:
            records = session.scalars(
                select(StructuredTableRecord)
                .join(
                    DocumentVersionRecord,
                    StructuredTableRecord.version_id == DocumentVersionRecord.id,
                )
                .join(
                    LogicalDocumentRecord,
                    DocumentVersionRecord.document_id == LogicalDocumentRecord.id,
                )
                .where(
                    StructuredTableRecord.collection_id == str(collection_id),
                    DocumentVersionRecord.active.is_(True),
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
                .order_by(StructuredTableRecord.created_at)
            )
            return [_table_response(record) for record in records]

    def query(self, request: StructuredQueryRequest) -> StructuredQueryResponse:
        table = self._get_table(request.table_id)
        plan = request.plan or self.planner.plan(request.question, table)
        statement = _compile_plan(table, plan, self.settings.structured_max_rows)
        preview = str(statement.compile(dialect=self.repository.engine.dialect))
        try:
            with self.repository.engine.connect() as connection:
                transaction = connection.begin()
                try:
                    if self.repository.engine.dialect.name == "postgresql":
                        connection.exec_driver_sql("SET TRANSACTION READ ONLY")
                        connection.exec_driver_sql(
                            "SET LOCAL statement_timeout = "
                            f"{self.settings.structured_statement_timeout_ms}"
                        )
                    result = connection.execute(statement)
                    columns = list(result.keys())
                    raw_rows = result.mappings().all()
                finally:
                    transaction.rollback()
        except SQLAlchemyError as exc:
            raise RetrievalError(
                "The validated structured query could not be executed",
                code="structured_execution_failed",
            ) from exc
        result_limit = min(plan.limit, self.settings.structured_max_rows)
        truncated = len(raw_rows) > result_limit
        raw_rows = raw_rows[:result_limit]
        rows = [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in raw_rows
        ]
        return StructuredQueryResponse(
            table=table,
            plan=plan,
            sql_preview=preview,
            columns=columns,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
            provenance={
                "collection_id": str(table.collection_id),
                "document_id": str(table.document_id),
                "version_id": str(table.version_id),
                "filename": table.filename,
            },
        )

    def _get_table(self, table_id: UUID) -> StructuredTableResponse:
        with self.repository.sessions() as session:
            record = session.scalar(
                select(StructuredTableRecord)
                .join(
                    DocumentVersionRecord,
                    StructuredTableRecord.version_id == DocumentVersionRecord.id,
                )
                .join(
                    LogicalDocumentRecord,
                    DocumentVersionRecord.document_id == LogicalDocumentRecord.id,
                )
                .where(
                    StructuredTableRecord.id == str(table_id),
                    DocumentVersionRecord.active.is_(True),
                    LogicalDocumentRecord.deleted_at.is_(None),
                )
            )
            if record is None:
                raise RetrievalError(
                    "Structured table not found", code="structured_table_not_found"
                )
            return _table_response(record)


def _read_csv(decoded: str, settings: Settings) -> tuple[list[str], list[dict[str, str]]]:
    try:
        reader = csv.DictReader(StringIO(decoded, newline=""), strict=True)
        headers = reader.fieldnames
        if not headers or any(not header.strip() for header in headers):
            raise RetrievalError("CSV requires non-blank headers", code="invalid_csv_schema")
        if len(headers) > settings.structured_max_columns:
            raise RetrievalError("CSV has too many columns", code="invalid_csv_schema")
        if len(set(header.casefold() for header in headers)) != len(headers):
            raise RetrievalError("CSV headers must be unique", code="invalid_csv_schema")
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row:
                raise RetrievalError("CSV row has more values than headers", code="invalid_csv")
            rows.append(
                {
                    header: value if (value := row.get(header)) is not None else ""
                    for header in headers
                }
            )
            if len(rows) > settings.structured_max_csv_rows:
                raise RetrievalError("CSV has too many rows", code="invalid_csv_size")
    except csv.Error as exc:
        raise RetrievalError("CSV is malformed", code="invalid_csv") from exc
    if not rows:
        raise RetrievalError("CSV requires at least one data row", code="invalid_csv")
    return list(headers), rows


def _infer_columns(headers: list[str], rows: list[dict[str, str]]) -> list[StructuredColumn]:
    normalized = _normalize_headers(headers)
    columns: list[StructuredColumn] = []
    for source_name, name in zip(headers, normalized, strict=True):
        values = [row[source_name].strip() for row in rows if row[source_name].strip()]
        data_type = _infer_type(values)
        columns.append(
            StructuredColumn(
                name=name,
                source_name=source_name,
                data_type=data_type,
                nullable=len(values) != len(rows),
                description=f"CSV column '{source_name}' inferred as {data_type}.",
            )
        )
    return columns


def _normalize_headers(headers: Sequence[str]) -> list[str]:
    names: list[str] = []
    for index, header in enumerate(headers, start=1):
        name = re.sub(r"[^A-Za-z0-9_]+", "_", header.strip()).strip("_").casefold()
        if not name or name[0].isdigit():
            name = f"column_{index}_{name}".rstrip("_")
        if name.startswith("_lke_"):
            name = f"column_{name.lstrip('_')}"
        base = name[:50]
        suffix = 2
        while name in names:
            name = f"{base}_{suffix}"
            suffix += 1
        names.append(name)
    return names


def _infer_type(
    values: Sequence[str],
) -> Literal["integer", "float", "boolean", "date", "text"]:
    if values and all(value.casefold() in {"true", "false"} for value in values):
        return "boolean"
    if values and all(_is_int(value) for value in values):
        return "integer"
    if values and all(_is_float(value) for value in values):
        return "float"
    if values and all(_is_date(value) for value in values):
        return "date"
    return "text"


def _typed_row(row: Mapping[str, str], columns: Sequence[StructuredColumn]) -> dict[str, Any]:
    typed: dict[str, Any] = {}
    for column in columns:
        value = row[column.source_name].strip()
        typed[column.name] = _coerce(value, column.data_type) if value else None
    return typed


def _coerce(value: str, data_type: str) -> object:
    if data_type == "boolean":
        return value.casefold() == "true"
    if data_type == "integer":
        return int(value)
    if data_type == "float":
        return float(value)
    if data_type == "date":
        return date.fromisoformat(value)
    return value


def _sql_table(name: str, columns: Sequence[StructuredColumn]) -> Table:
    types = {
        "boolean": Boolean,
        "integer": Integer,
        "float": Float,
        "date": Date,
        "text": Text,
    }
    return Table(
        name,
        MetaData(),
        Column("_lke_row_number", Integer, primary_key=True),
        Column("_lke_version_id", String(36), nullable=False),
        *[
            Column(column.name, types[column.data_type](), nullable=column.nullable)
            for column in columns
        ],
    )


def _compile_plan(table: StructuredTableResponse, plan: StructuredQueryPlan, max_rows: int) -> Any:
    sql_table = _sql_table(table.physical_name, table.columns)
    available = {column.name: sql_table.c[column.name] for column in table.columns}
    aggregation_aliases = {aggregation.alias for aggregation in plan.aggregations}
    referenced = {
        *plan.projections,
        *plan.group_by,
        *(item.column for item in plan.filters),
        *(item.column for item in plan.aggregations if item.column),
        *(item.column for item in plan.order_by if item.column not in aggregation_aliases),
    }
    unknown = referenced - set(available)
    if unknown:
        raise RetrievalError(
            f"Plan references unknown columns: {', '.join(sorted(unknown))}",
            code="invalid_structured_plan",
        )
    expressions: list[Any] = [available[name] for name in plan.group_by]
    if plan.projections:
        expressions.extend(
            available[name] for name in plan.projections if name not in plan.group_by
        )
    numeric_types = {
        column.name
        for column in table.columns
        if column.data_type in {"integer", "float"}
    }
    column_types = {column.name: column.data_type for column in table.columns}
    aggregation_expressions: dict[str, Any] = {}
    for item in plan.aggregations:
        if item.function in {"sum", "avg"} and item.column not in numeric_types:
            raise RetrievalError(
                f"{item.function} requires a numeric column",
                code="invalid_structured_plan",
            )
        target = available[item.column] if item.column else None
        expression = (
            func.count()
            if item.function == "count" and target is None
            else getattr(func, item.function)(target)
        ).label(item.alias)
        aggregation_expressions[item.alias] = expression
        expressions.append(expression)
    if not expressions:
        expressions = list(available.values())
    if plan.aggregations:
        invalid_projection = set(plan.projections) - set(plan.group_by)
        if invalid_projection:
            raise RetrievalError(
                "Projected columns must be grouped when aggregations are used",
                code="invalid_structured_plan",
            )
    statement = select(*expressions).select_from(sql_table)
    for condition in plan.filters:
        column = available[condition.column]
        value = _typed_filter_value(condition.value, column_types[condition.column])
        if condition.operator is MetadataOperator.CONTAINS and column_types[
            condition.column
        ] != "text":
            raise RetrievalError(
                "contains is only valid for text columns",
                code="invalid_structured_plan",
            )
        if condition.operator in {
            MetadataOperator.GT,
            MetadataOperator.GTE,
            MetadataOperator.LT,
            MetadataOperator.LTE,
        } and column_types[condition.column] == "boolean":
            raise RetrievalError(
                "boolean columns do not support ordering comparisons",
                code="invalid_structured_plan",
            )
        if condition.operator is MetadataOperator.EQ:
            predicate = column == value
        elif condition.operator is MetadataOperator.NE:
            predicate = column != value
        elif condition.operator is MetadataOperator.IN:
            if not isinstance(value, list):
                raise RetrievalError("in requires a list", code="invalid_structured_plan")
            predicate = column.in_(value)
        elif condition.operator is MetadataOperator.CONTAINS:
            predicate = column.contains(str(value), autoescape=True)
        elif condition.operator is MetadataOperator.GT:
            predicate = column > value
        elif condition.operator is MetadataOperator.GTE:
            predicate = column >= value
        elif condition.operator is MetadataOperator.LT:
            predicate = column < value
        else:
            predicate = column <= value
        statement = statement.where(predicate)
    if plan.group_by:
        statement = statement.group_by(*(available[name] for name in plan.group_by))
    for order in plan.order_by:
        expression = aggregation_expressions.get(order.column)
        if expression is None:
            expression = available.get(order.column)
        if expression is None:
            raise RetrievalError(
                f"Unknown order column '{order.column}'",
                code="invalid_structured_plan",
            )
        ordering = desc(expression) if order.direction == "desc" else asc(expression)
        statement = statement.order_by(ordering)
    safe_limit = min(plan.limit, max_rows)
    return statement.limit(safe_limit + 1)


def _table_response(record: StructuredTableRecord) -> StructuredTableResponse:
    return StructuredTableResponse(
        id=UUID(record.id),
        collection_id=UUID(record.collection_id),
        document_id=UUID(record.document_id),
        version_id=UUID(record.version_id),
        filename=record.filename,
        physical_name=record.physical_name,
        content_sha256=record.content_sha256,
        row_count=record.row_count,
        columns=[StructuredColumn.model_validate(item) for item in record.schema_definition],
        created_at=record.created_at,
    )


def _typed_filter_value(value: object, data_type: str) -> object:
    if isinstance(value, list):
        return [_typed_filter_value(item, data_type) for item in value]
    try:
        if data_type == "integer":
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            return int(value)
        if data_type == "float":
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError
            return float(value)
        if data_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.casefold() in {"true", "false"}:
                return value.casefold() == "true"
            raise ValueError
        if data_type == "date":
            if not isinstance(value, str):
                raise ValueError
            return date.fromisoformat(value)
        if not isinstance(value, str):
            raise ValueError
        return value
    except (TypeError, ValueError) as exc:
        raise RetrievalError(
            f"Filter value does not match {data_type} column type",
            code="invalid_structured_plan",
        ) from exc


def _is_int(value: str) -> bool:
    try:
        int(value)
        return "." not in value
    except ValueError:
        return False


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _json_scalar(value: object) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)
