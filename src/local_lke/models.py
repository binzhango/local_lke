"""Stable public domain models for the RAG API."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    DEGRADED = "degraded"
    ERROR = "error"


class RetrievalStrategy(StrEnum):
    DENSE = "dense"
    HYBRID = "hybrid"
    STRUCTURED = "structured"


class QueryRoute(StrEnum):
    SIMPLE_LOOKUP = "simple_lookup"
    BROAD_SYNTHESIS = "broad_synthesis"
    MULTI_PART = "multi_part"
    STRUCTURED = "structured"


class RewriteStrategy(StrEnum):
    NONE = "none"
    STEP_BACK = "step_back"
    HYDE = "hyde"


class MetadataOperator(StrEnum):
    EQ = "eq"
    NE = "ne"
    IN = "in"
    CONTAINS = "contains"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"


class MetadataCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "filename",
        "media_type",
        "parser_strategy",
        "chunk_strategy",
        "page_number",
        "created_at",
    ]
    operator: MetadataOperator
    value: str | int | float | list[str] | list[int]

    @model_validator(mode="after")
    def validate_operator_and_value(self) -> "MetadataCondition":
        if self.operator is MetadataOperator.IN and not isinstance(self.value, list):
            raise ValueError("the 'in' operator requires a list")
        if self.operator is MetadataOperator.CONTAINS and self.field not in {
            "filename",
            "media_type",
            "parser_strategy",
            "chunk_strategy",
        }:
            raise ValueError("contains is only valid for string metadata")
        if (
            self.field == "page_number"
            and (
                (
                    self.operator is MetadataOperator.IN
                    and (
                        not isinstance(self.value, list)
                        or not all(isinstance(item, int) and item >= 1 for item in self.value)
                    )
                )
                or (
                    self.operator is not MetadataOperator.IN
                    and (
                        not isinstance(self.value, (int, float))
                        or isinstance(self.value, bool)
                        or self.value < 1
                    )
                )
            )
        ):
            raise ValueError("page_number filters require positive integer values")
        string_fields = {"filename", "media_type", "parser_strategy", "chunk_strategy"}
        if self.field in string_fields:
            if self.operator not in {
                MetadataOperator.EQ,
                MetadataOperator.NE,
                MetadataOperator.IN,
                MetadataOperator.CONTAINS,
            }:
                raise ValueError("string metadata supports eq, ne, in, and contains")
            values = self.value if isinstance(self.value, list) else [self.value]
            if not all(isinstance(item, str) for item in values):
                raise ValueError("string metadata requires string values")
        enum_values = {
            "parser_strategy": {"fast", "hi_res"},
            "chunk_strategy": {"recursive", "markdown", "semantic"},
            "media_type": {"text/plain", "text/markdown", "application/pdf"},
        }
        if self.field in enum_values and self.operator in {
            MetadataOperator.EQ,
            MetadataOperator.NE,
            MetadataOperator.IN,
        }:
            values = self.value if isinstance(self.value, list) else [self.value]
            if not set(values) <= enum_values[self.field]:
                raise ValueError(f"unsupported {self.field} value")
        return self


class MetadataFilterPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conditions: list[MetadataCondition] = Field(default_factory=list, max_length=12)
    allow_unfiltered_fallback: bool = False


class QueryTransformPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_query: str
    normalized_query: str
    route: QueryRoute
    subqueries: list[str] = Field(min_length=1, max_length=4)
    rewrite: RewriteStrategy = RewriteStrategy.NONE
    rewritten_query: str | None = None
    rationale: str


class SourceDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    content: str = Field(min_length=1)
    media_type: str = "text/plain"


class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    text: str = Field(min_length=1)
    ordinal: int = Field(ge=0)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    rank: int = Field(ge=1)
    score: float | None = None


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=500)
    document_version_id: UUID | None = None
    title: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    collection_id: UUID | None = None
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    metadata_filter: MetadataFilterPlan | None = None
    infer_metadata_filter: bool = False
    rewrite: RewriteStrategy = RewriteStrategy.NONE


class TraceSummary(BaseModel):
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    retrieval: "AdvancedRetrievalTrace | None" = None


class AnswerResponse(BaseModel):
    status: AnswerStatus
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    trace: TraceSummary

    @model_validator(mode="after")
    def answered_responses_have_citations(self) -> "AnswerResponse":
        if self.status is AnswerStatus.ANSWERED and not self.citations:
            raise ValueError("answered responses require at least one citation")
        return self


class ComponentHealth(BaseModel):
    status: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    components: dict[str, ComponentHealth]


class ErrorDetail(BaseModel):
    code: str
    message: str
    component: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ParserStrategy(StrEnum):
    FAST = "fast"
    HI_RES = "hi_res"


class ChunkStrategy(StrEnum):
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class DocumentElement(BaseModel):
    model_config = ConfigDict(frozen=True)

    element_id: UUID
    document_version_id: UUID
    ordinal: int = Field(ge=0)
    category: str = Field(min_length=1)
    text: str
    locator: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    heading_path: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class IngestedChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(min_length=64, max_length=64)
    document_version_id: UUID
    parent_element_id: UUID | None = None
    ordinal: int = Field(ge=0)
    strategy: ChunkStrategy
    text: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    heading_path: tuple[str, ...] = ()
    character_count: int = Field(ge=1)
    token_count: int = Field(ge=1)
    flags: tuple[str, ...] = ()


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CollectionResponse(BaseModel):
    id: UUID
    name: str
    created_at: datetime


class DocumentVersionResponse(BaseModel):
    id: UUID
    document_id: UUID
    content_sha256: str
    pipeline_hash: str
    media_type: str
    parser_name: str
    parser_version: str
    parser_strategy: ParserStrategy
    active: bool
    inactive_reason: str | None = None
    status: str
    element_count: int
    chunk_count: int
    warning_count: int
    created_at: datetime


class DocumentResponse(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    display_filename: str
    deleted_at: datetime | None = None
    created_at: datetime
    versions: list[DocumentVersionResponse] = Field(default_factory=list)


class IngestionJobResponse(BaseModel):
    id: UUID
    collection_id: UUID
    document_id: UUID | None = None
    version_id: UUID | None = None
    filename: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    skipped: bool = False
    element_count: int = 0
    chunk_count: int = 0
    warning_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class ParserPreviewResponse(BaseModel):
    version: DocumentVersionResponse
    elements: list[DocumentElement]
    chunks: list[IngestedChunk]


class ActiveChunk(BaseModel):
    """One queryable chunk from a non-deleted document's active version."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    collection_id: UUID
    document_id: UUID
    version_id: UUID
    filename: str
    media_type: str
    parser_strategy: str
    chunk_strategy: str
    ordinal: int
    text: str
    locator: str
    page_number: int | None = None
    heading_path: tuple[str, ...] = ()
    token_count: int
    created_at: datetime


class RetrievalCandidateTrace(BaseModel):
    chunk_id: str
    document_id: UUID
    version_id: UUID
    filename: str
    locator: str
    matched_subqueries: list[str] = Field(default_factory=list)
    matched_terms: list[str] = Field(default_factory=list)
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    fused_rank: int | None = None
    rrf_score: float | None = None
    rerank_before: int | None = None
    rerank_after: int | None = None
    rerank_score: float | None = None


class ContextManifestEntry(BaseModel):
    chunk_id: str
    document_id: UUID
    version_id: UUID
    locator: str
    decision: Literal["included", "excluded", "truncated"]
    reason: str
    token_count: int = Field(ge=0)
    covered_subqueries: list[str] = Field(default_factory=list)


class AnswerabilityTrace(BaseModel):
    sufficient: bool
    score: float = Field(ge=0, le=1)
    threshold: float = Field(ge=0, le=1)
    term_coverage: float = Field(ge=0, le=1)
    subquery_coverage: float = Field(ge=0, le=1)
    evidence_strength: float = Field(ge=0, le=1)
    reason: str
    initial_failure_reason: str | None = None
    corrective_attempted: bool = False
    corrective_strategy: RetrievalStrategy | None = None


class AdvancedRetrievalTrace(BaseModel):
    strategy: RetrievalStrategy
    transform: QueryTransformPlan
    metadata_filter: MetadataFilterPlan
    metadata_fallback_used: bool = False
    candidates: list[RetrievalCandidateTrace] = Field(default_factory=list)
    context_manifest: list[ContextManifestEntry] = Field(default_factory=list)
    answerability: AnswerabilityTrace
    reranker_latency_ms: float = Field(default=0, ge=0)
    reranker_top_gain: float = 0


class StructuredColumn(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    source_name: str
    data_type: Literal["integer", "float", "boolean", "date", "text"]
    nullable: bool
    description: str


class StructuredTableResponse(BaseModel):
    id: UUID
    collection_id: UUID
    document_id: UUID
    version_id: UUID
    filename: str
    physical_name: str
    content_sha256: str
    row_count: int = Field(ge=0)
    columns: list[StructuredColumn]
    created_at: datetime


class StructuredFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    operator: MetadataOperator
    value: str | int | float | bool | list[str] | list[int]


class StructuredAggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: Literal["count", "sum", "avg", "min", "max"]
    column: str | None = None
    alias: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")

    @model_validator(mode="after")
    def require_column_for_non_count(self) -> "StructuredAggregation":
        if self.function != "count" and self.column is None:
            raise ValueError(f"{self.function} requires a column")
        return self


class StructuredOrder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    direction: Literal["asc", "desc"] = "asc"


class StructuredQueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projections: list[str] = Field(default_factory=list, max_length=50)
    filters: list[StructuredFilter] = Field(default_factory=list, max_length=20)
    group_by: list[str] = Field(default_factory=list, max_length=20)
    aggregations: list[StructuredAggregation] = Field(default_factory=list, max_length=20)
    order_by: list[StructuredOrder] = Field(default_factory=list, max_length=10)
    limit: int = Field(default=50, ge=1, le=1000)


class StructuredQueryRequest(BaseModel):
    table_id: UUID
    question: str = Field(min_length=2, max_length=2000)
    plan: StructuredQueryPlan | None = None


class StructuredQueryResponse(BaseModel):
    table: StructuredTableResponse
    plan: StructuredQueryPlan
    sql_preview: str
    columns: list[str]
    rows: list[dict[str, str | int | float | bool | None]]
    row_count: int = Field(ge=0)
    truncated: bool
    provenance: dict[str, str]
