"""Stable public domain models for the RAG API."""

from enum import StrEnum

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

