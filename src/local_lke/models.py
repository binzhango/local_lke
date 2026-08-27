"""Stable public domain models for the RAG API."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal
from uuid import UUID, uuid4

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


class OutputMode(StrEnum):
    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    EVIDENCE_ONLY = "evidence_only"


class StructuredSchemaName(StrEnum):
    FACT_LIST = "fact_list"
    COMPARISON = "comparison"


class CitationLocatorKind(StrEnum):
    MARKDOWN = "markdown_heading"
    TEXT = "text_range"
    PDF = "pdf_element"
    IMAGE = "image"
    TABLE = "table_rows"
    GENERIC = "generic"


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


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

    citation_id: str = Field(default="", pattern=r"^$|^C[1-9][0-9]*$")
    source_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    locator: str = Field(min_length=1)
    excerpt: str = Field(min_length=1, max_length=500)
    document_version_id: UUID | None = None
    title: str | None = None
    locator_detail: "CitationLocator | None" = None


class CitationLocator(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: CitationLocatorKind
    label: str = Field(min_length=1)
    heading_path: tuple[str, ...] = ()
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    page_number: int | None = Field(default=None, ge=1)
    element_id: UUID | None = None
    image_id: UUID | None = None
    table_id: UUID | None = None
    row_start: int | None = Field(default=None, ge=1)
    row_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> "CitationLocator":
        if (
            self.start_line is not None
            and self.end_line is not None
            and self.end_line < self.start_line
        ):
            raise ValueError("end_line must be greater than or equal to start_line")
        if (
            self.row_start is not None
            and self.row_end is not None
            and self.row_end < self.row_start
        ):
            raise ValueError("row_end must be greater than or equal to row_start")
        return self


class ConfidenceExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: ConfidenceLevel
    rationale: str = Field(min_length=1, max_length=500)
    calibration: str = (
        "Qualitative evidence assessment; this level is not a calibrated probability."
    )


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    statement: str = Field(min_length=1)
    citation_ids: list[str] = Field(min_length=1)


class FactListAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal[StructuredSchemaName.FACT_LIST] = StructuredSchemaName.FACT_LIST
    summary: str = Field(min_length=1)
    facts: list[CitedClaim] = Field(min_length=1, max_length=20)


class ComparisonItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    details: list[str] = Field(min_length=1, max_length=20)
    citation_ids: list[str] = Field(min_length=1)


class ComparisonAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal[StructuredSchemaName.COMPARISON] = StructuredSchemaName.COMPARISON
    summary: str = Field(min_length=1)
    items: list[ComparisonItem] = Field(min_length=1, max_length=20)


StructuredAnswer = Annotated[
    FactListAnswer | ComparisonAnswer,
    Field(discriminator="schema_name"),
]


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    collection_id: UUID | None = None
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    metadata_filter: MetadataFilterPlan | None = None
    infer_metadata_filter: bool = False
    rewrite: RewriteStrategy = RewriteStrategy.NONE
    output_mode: OutputMode = OutputMode.CONVERSATIONAL
    schema_name: StructuredSchemaName | None = None

    @model_validator(mode="after")
    def validate_output_selection(self) -> "QueryRequest":
        if self.output_mode is OutputMode.STRUCTURED and self.schema_name is None:
            raise ValueError("schema_name is required for structured output")
        if self.output_mode is not OutputMode.STRUCTURED and self.schema_name is not None:
            raise ValueError("schema_name is only valid for structured output")
        return self


class GenerationTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_version: str
    output_mode: OutputMode
    schema_name: StructuredSchemaName | None = None
    attempts: int = Field(default=0, ge=0)
    repair_attempts: int = Field(default=0, ge=0)
    native_structured_output: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    degraded_reason: str | None = None
    model_output_committed: bool = False


class TraceSummary(BaseModel):
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    retrieval: "AdvancedRetrievalTrace | None" = None
    generation: GenerationTrace | None = None


class AnswerResponse(BaseModel):
    status: AnswerStatus
    answer: str = Field(min_length=1)
    structured_result: StructuredAnswer | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: ConfidenceExplanation
    uncovered_subquestions: list[str] = Field(default_factory=list)
    route: QueryRoute | None = None
    trace_id: UUID = Field(default_factory=uuid4)
    warnings: list[str] = Field(default_factory=list)
    trace: TraceSummary

    @model_validator(mode="after")
    def validate_answer_contract(self) -> "AnswerResponse":
        if self.status in {AnswerStatus.ANSWERED, AnswerStatus.DEGRADED}:
            if not self.citations:
                raise ValueError("answered and degraded responses require at least one citation")
            citation_ids = [item.citation_id for item in self.citations]
            if any(not item for item in citation_ids):
                raise ValueError("answered and degraded citations require citation_id")
            if len(citation_ids) != len(set(citation_ids)):
                raise ValueError("citation_id values must be unique")
        if self.structured_result is not None:
            if self.trace.generation is None:
                raise ValueError("structured results require a generation trace")
            if self.trace.generation.output_mode is not OutputMode.STRUCTURED:
                raise ValueError("structured_result requires structured output mode")
            if self.structured_result.schema_name is not self.trace.generation.schema_name:
                raise ValueError("structured_result does not match the selected schema")
            allowed = {item.citation_id for item in self.citations}
            referenced = structured_citation_ids(self.structured_result)
            if not referenced <= allowed:
                raise ValueError("structured_result references an unavailable citation")
        if self.status is AnswerStatus.ABSTAINED and self.structured_result is not None:
            raise ValueError("abstained responses cannot contain a structured result")
        return self


def structured_citation_ids(value: StructuredAnswer) -> set[str]:
    if isinstance(value, FactListAnswer):
        return {citation for fact in value.facts for citation in fact.citation_ids}
    return {citation for item in value.items for citation in item.citation_ids}


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


class EmbeddingModality(StrEnum):
    TEXT = "text"
    MULTIMODAL = "multimodal"


class NodeGranularity(StrEnum):
    SENTENCE = "sentence"
    CHUNK = "chunk"
    SECTION = "section"


class ExpansionStrategy(StrEnum):
    NONE = "none"
    SENTENCE_WINDOW = "sentence_window"
    PARENT = "parent"
    MULTI = "multi"


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


class EmbeddingProfileResponse(BaseModel):
    id: UUID
    modality: EmbeddingModality
    model_id: str
    revision: str
    dimension: int = Field(ge=1)
    normalized: bool
    document_prefix: str = ""
    query_prefix: str = ""
    created_at: datetime


class IndexingJobResponse(BaseModel):
    id: UUID
    collection_id: UUID
    version_id: UUID
    profile_id: UUID
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    total_nodes: int = Field(ge=0)
    embedded_nodes: int = Field(ge=0)
    embedding_calls: int = Field(ge=0)
    skipped: bool = False
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class IndexStateResponse(BaseModel):
    collection_id: UUID
    active_profile: EmbeddingProfileResponse | None = None
    active_nodes: int = Field(ge=0)
    active_chunks: int = Field(ge=0)
    missing_active_chunks: int = Field(ge=0)
    jobs: list[IndexingJobResponse] = Field(default_factory=list)


class VectorSearchRequest(BaseModel):
    collection_id: UUID
    question: str = Field(min_length=2, max_length=2000)
    profile_id: UUID | None = None
    top_k: int = Field(default=5, ge=1, le=50)
    expansion: ExpansionStrategy = ExpansionStrategy.NONE
    token_budget: int | None = Field(default=None, ge=32, le=1_000_000)
    sentence_window: int = Field(default=2, ge=0, le=20)


class VectorSearchCandidate(BaseModel):
    node_id: str
    chunk_id: str
    document_id: UUID
    version_id: UUID
    granularity: NodeGranularity
    rank: int = Field(ge=1)
    score: float
    locator: str
    child_text: str
    context_text: str
    trigger_node_id: str
    token_count: int = Field(ge=1)
    included: bool
    decision: str


class VectorSearchResponse(BaseModel):
    profile: EmbeddingProfileResponse
    candidates: list[VectorSearchCandidate]
    final_context: list[VectorSearchCandidate]
    final_token_count: int = Field(ge=0)


class ImageAssetResponse(BaseModel):
    id: UUID
    collection_id: UUID
    filename: str
    media_type: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    sha256: str
    created_at: datetime
    content_url: str


class ImageSearchHit(BaseModel):
    image: ImageAssetResponse
    rank: int = Field(ge=1)
    score: float


class ImageSearchResponse(BaseModel):
    profile: EmbeddingProfileResponse
    hits: list[ImageSearchHit]


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
