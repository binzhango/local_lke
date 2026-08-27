"""Public Chapter 6 evaluation contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from local_lke.models import (
    AnswerStatus,
    OutputMode,
    RetrievalStrategy,
    RewriteStrategy,
    StructuredSchemaName,
)


class EvaluationFault(StrEnum):
    NONE = "none"
    CHAT_UNAVAILABLE = "chat_unavailable"
    EMPTY_OUTPUT = "empty_output"
    MALFORMED_OUTPUT = "malformed_output"


class EvaluationExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_source_ids: list[str] = Field(default_factory=list, max_length=50)
    relevant_chunk_ids: list[str] = Field(default_factory=list, max_length=100)
    answer_contains: list[str] = Field(default_factory=list, max_length=30)
    acceptable_statuses: list[AnswerStatus] = Field(
        default_factory=lambda: [AnswerStatus.ANSWERED]
    )

    @model_validator(mode="after")
    def require_an_expectation(self) -> "EvaluationExpectation":
        if not (
            self.relevant_source_ids
            or self.relevant_chunk_ids
            or self.answer_contains
            or self.acceptable_statuses != [AnswerStatus.ANSWERED]
        ):
            raise ValueError("at least one explicit expectation is required")
        return self


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9_.-]+$")
    question: str = Field(min_length=2, max_length=2000)
    collection_id: UUID | None = None
    strategy: RetrievalStrategy = RetrievalStrategy.DENSE
    rewrite: RewriteStrategy = RewriteStrategy.NONE
    top_k: int = Field(default=3, ge=1, le=20)
    output_mode: OutputMode = OutputMode.CONVERSATIONAL
    schema_name: StructuredSchemaName | None = None
    fault: EvaluationFault = EvaluationFault.NONE
    expectation: EvaluationExpectation

    @model_validator(mode="after")
    def validate_route_and_output(self) -> "EvaluationCase":
        if self.strategy is RetrievalStrategy.STRUCTURED:
            raise ValueError("structured table evaluation uses a separate dataset contract")
        if self.collection_id is None and self.strategy is not RetrievalStrategy.DENSE:
            raise ValueError("fixture evaluation supports only dense retrieval")
        if self.output_mode is OutputMode.STRUCTURED and self.schema_name is None:
            raise ValueError("schema_name is required for structured output")
        if self.output_mode is not OutputMode.STRUCTURED and self.schema_name is not None:
            raise ValueError("schema_name is only valid for structured output")
        return self


class EvaluationDatasetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def unique_case_ids(self) -> "EvaluationDatasetCreate":
        case_ids = [item.case_id for item in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("case_id values must be unique within a dataset")
        return self


class EvaluationDatasetResponse(BaseModel):
    id: UUID
    name: str
    description: str
    version: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: list[EvaluationCase]
    created_at: datetime


class EvaluationThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_case_pass_rate: float = Field(default=1.0, ge=0, le=1)
    min_recall_at_k: float | None = Field(default=None, ge=0, le=1)
    min_answer_match_rate: float | None = Field(default=None, ge=0, le=1)
    min_status_match_rate: float | None = Field(default=None, ge=0, le=1)
    max_p95_latency_ms: float | None = Field(default=None, ge=0)
    max_metric_decline: float = Field(default=0.0, ge=0, le=1)
    max_latency_increase_ms: float | None = Field(default=None, ge=0)


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: UUID
    baseline_run_id: UUID | None = None
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)


class EvaluationCaseResult(BaseModel):
    case_id: str
    status: AnswerStatus
    fault: EvaluationFault
    latency_ms: float = Field(ge=0)
    retrieved_source_ids: list[str]
    retrieved_chunk_ids: list[str]
    reciprocal_rank: float | None = Field(default=None, ge=0, le=1)
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_k: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)
    answer_match: bool | None = None
    status_match: bool
    passed: bool
    failures: list[str] = Field(default_factory=list)
    trace_id: UUID


class EvaluationMetrics(BaseModel):
    case_count: int = Field(ge=1)
    case_pass_rate: float = Field(ge=0, le=1)
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    mean_reciprocal_rank: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_k: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)
    answer_match_rate: float | None = Field(default=None, ge=0, le=1)
    status_match_rate: float = Field(ge=0, le=1)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)


class RegressionGate(BaseModel):
    passed: bool
    failures: list[str] = Field(default_factory=list)
    baseline_run_id: UUID | None = None
    deltas: dict[str, float] = Field(default_factory=dict)


class EvaluationRunResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    baseline_run_id: UUID | None = None
    status: Literal["running", "completed", "failed"]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, object]
    metrics: EvaluationMetrics | None = None
    case_results: list[EvaluationCaseResult] = Field(default_factory=list)
    gate: RegressionGate | None = None
    error_message: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class EvaluationComparison(BaseModel):
    baseline: EvaluationRunResponse
    candidate: EvaluationRunResponse
    deltas: dict[str, float]


class ProviderCapabilityProfile(BaseModel):
    profile_version: Literal["chapter6.provider.v1"] = "chapter6.provider.v1"
    provider: str
    model: str
    base_url: str
    parser_structured_output: bool = True
    native_structured_output: bool
    streaming: bool = True
    bounded_repair_attempts: int = Field(ge=0, le=3)
    supported_faults: list[EvaluationFault]
