"""Stable public domain models for the RAG API."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    DEGRADED = "degraded"
    ERROR = "error"


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


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class TraceSummary(BaseModel):
    timings_ms: dict[str, float] = Field(default_factory=dict)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


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
