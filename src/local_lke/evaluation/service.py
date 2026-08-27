"""Persisted, deterministic evaluation execution and regression gates."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from local_lke.errors import EvaluationError, ProviderUnavailableError
from local_lke.evaluation.models import (
    EvaluationCase,
    EvaluationCaseResult,
    EvaluationComparison,
    EvaluationDatasetCreate,
    EvaluationDatasetResponse,
    EvaluationFault,
    EvaluationMetrics,
    EvaluationRunRequest,
    EvaluationRunResponse,
    EvaluationThresholds,
    ProviderCapabilityProfile,
    RegressionGate,
)
from local_lke.generation import GenerationService
from local_lke.models import AnswerResponse, QueryRequest
from local_lke.providers import ChatProvider
from local_lke.rag import RAGPipeline
from local_lke.retrieval import AdvancedRetrievalService
from local_lke.settings import Settings
from local_lke.storage.models import EvaluationDatasetRecord, EvaluationRunRecord


class EvaluationService:
    """Own immutable datasets and execute synchronous, inspectable evaluation runs."""

    def __init__(
        self,
        sessions: sessionmaker[Session],
        settings: Settings,
        pipeline: RAGPipeline,
        retrieval: AdvancedRetrievalService,
    ) -> None:
        self.sessions = sessions
        self.settings = settings
        self.pipeline = pipeline
        self.retrieval = retrieval

    def create_dataset(self, payload: EvaluationDatasetCreate) -> EvaluationDatasetResponse:
        name = " ".join(payload.name.split())
        canonical_cases = [case.model_dump(mode="json") for case in payload.cases]
        digest = _sha256(
            {
                "name": name,
                "description": payload.description,
                "cases": canonical_cases,
            }
        )
        with self.sessions.begin() as session:
            existing = session.scalar(
                select(EvaluationDatasetRecord).where(
                    EvaluationDatasetRecord.content_sha256 == digest
                )
            )
            if existing is not None:
                return _dataset_response(existing)
            latest = session.scalar(
                select(func.max(EvaluationDatasetRecord.version)).where(
                    EvaluationDatasetRecord.name == name
                )
            )
            record = EvaluationDatasetRecord(
                name=name,
                description=payload.description,
                version=int(latest or 0) + 1,
                content_sha256=digest,
                cases=canonical_cases,
            )
            session.add(record)
            session.flush()
            return _dataset_response(record)

    def list_datasets(self) -> list[EvaluationDatasetResponse]:
        with self.sessions() as session:
            records = session.scalars(
                select(EvaluationDatasetRecord).order_by(
                    EvaluationDatasetRecord.name, EvaluationDatasetRecord.version.desc()
                )
            )
            return [_dataset_response(item) for item in records]

    def get_dataset(self, dataset_id: UUID) -> EvaluationDatasetResponse:
        with self.sessions() as session:
            record = session.get(EvaluationDatasetRecord, str(dataset_id))
            if record is None:
                raise EvaluationError("Evaluation dataset not found", code="dataset_not_found")
            return _dataset_response(record)

    def run(self, request: EvaluationRunRequest) -> EvaluationRunResponse:
        dataset = self.get_dataset(request.dataset_id)
        baseline = (
            self.get_run(request.baseline_run_id)
            if request.baseline_run_id is not None
            else None
        )
        if baseline is not None and (
            baseline.status != "completed" or baseline.metrics is None
        ):
            raise EvaluationError(
                "The baseline run must be completed successfully",
                code="baseline_incomplete",
            )
        if baseline is not None and baseline.dataset_id != dataset.id:
            raise EvaluationError(
                "The baseline run belongs to a different immutable dataset version",
                code="baseline_dataset_mismatch",
            )
        configuration = self._configuration(dataset, request.thresholds)
        configuration_sha256 = _sha256(configuration)
        with self.sessions.begin() as session:
            record = EvaluationRunRecord(
                dataset_id=str(dataset.id),
                baseline_run_id=(str(baseline.id) if baseline is not None else None),
                status="running",
                configuration_sha256=configuration_sha256,
                configuration=configuration,
                metrics={},
                case_results=[],
                gate={},
            )
            session.add(record)
            session.flush()
            run_id = UUID(record.id)
        try:
            results = [self._run_case(case) for case in dataset.cases]
            metrics = _aggregate(results)
            gate = _gate(metrics, request.thresholds, baseline)
        except Exception as exc:
            with self.sessions.begin() as session:
                failed = session.get(EvaluationRunRecord, str(run_id))
                assert failed is not None
                failed.status = "failed"
                failed.error_message = _safe_error(exc)
                failed.finished_at = datetime.now(UTC)
            raise EvaluationError(
                "Evaluation execution failed; inspect the persisted run",
                code="evaluation_run_failed",
            ) from exc
        with self.sessions.begin() as session:
            completed = session.get(EvaluationRunRecord, str(run_id))
            assert completed is not None
            completed.status = "completed"
            completed.metrics = metrics.model_dump(mode="json")
            completed.case_results = [item.model_dump(mode="json") for item in results]
            completed.gate = gate.model_dump(mode="json")
            completed.finished_at = datetime.now(UTC)
        return self.get_run(run_id)

    def list_runs(self, dataset_id: UUID | None = None) -> list[EvaluationRunResponse]:
        with self.sessions() as session:
            statement = select(EvaluationRunRecord)
            if dataset_id is not None:
                statement = statement.where(
                    EvaluationRunRecord.dataset_id == str(dataset_id)
                )
            records = session.scalars(statement.order_by(EvaluationRunRecord.created_at.desc()))
            return [_run_response(item) for item in records]

    def get_run(self, run_id: UUID) -> EvaluationRunResponse:
        with self.sessions() as session:
            record = session.get(EvaluationRunRecord, str(run_id))
            if record is None:
                raise EvaluationError("Evaluation run not found", code="run_not_found")
            return _run_response(record)

    def compare(self, baseline_run_id: UUID, candidate_run_id: UUID) -> EvaluationComparison:
        baseline = self.get_run(baseline_run_id)
        candidate = self.get_run(candidate_run_id)
        if baseline.dataset_id != candidate.dataset_id:
            raise EvaluationError(
                "Runs from different immutable dataset versions cannot be compared",
                code="comparison_dataset_mismatch",
            )
        if baseline.metrics is None or candidate.metrics is None:
            raise EvaluationError(
                "Only completed evaluation runs can be compared",
                code="comparison_incomplete",
            )
        return EvaluationComparison(
            baseline=baseline,
            candidate=candidate,
            deltas=_metric_deltas(baseline.metrics, candidate.metrics),
        )

    def provider_profile(self) -> ProviderCapabilityProfile:
        return ProviderCapabilityProfile(
            provider="openai-compatible-local",
            model=self.settings.chat_model,
            base_url=str(self.settings.chat_base_url),
            native_structured_output=self.settings.generation_native_structured_output,
            bounded_repair_attempts=self.settings.generation_max_repair_attempts,
            supported_faults=list(EvaluationFault),
        )

    def _run_case(self, case: EvaluationCase) -> EvaluationCaseResult:
        started = perf_counter()
        response = self._execute(case)
        latency_ms = round((perf_counter() - started) * 1000, 3)
        ranked_sources, ranked_chunks = _ranked_retrieval(response)
        expected = case.expectation
        relevance_expected = [
            *(f"source:{item}" for item in expected.relevant_source_ids),
            *(f"chunk:{item}" for item in expected.relevant_chunk_ids),
        ]
        relevance = _retrieval_metrics(
            expected.relevant_source_ids,
            expected.relevant_chunk_ids,
            ranked_sources,
            ranked_chunks,
        )
        cited = [
            *(f"source:{item.source_id}" for item in response.citations),
            *(f"chunk:{item.chunk_id}" for item in response.citations),
        ]
        citation_precision, citation_recall = _precision_recall(relevance_expected, cited)
        answer_match = (
            all(
                needle.casefold() in response.answer.casefold()
                for needle in expected.answer_contains
            )
            if expected.answer_contains
            else None
        )
        status_match = response.status in expected.acceptable_statuses
        failures: list[str] = []
        if relevance[1] is not None and relevance[1] < 1:
            failures.append("not all labelled relevant evidence was retrieved")
        if answer_match is False:
            failures.append("answer omitted one or more required phrases")
        if not status_match:
            failures.append(
                f"status {response.status.value} was outside the acceptable status set"
            )
        if citation_recall is not None and citation_recall < 1:
            failures.append("citations omitted labelled relevant evidence")
        return EvaluationCaseResult(
            case_id=case.case_id,
            status=response.status,
            fault=case.fault,
            latency_ms=latency_ms,
            retrieved_source_ids=ranked_sources,
            retrieved_chunk_ids=ranked_chunks,
            reciprocal_rank=relevance[0],
            recall_at_k=relevance[1],
            ndcg_at_k=relevance[2],
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            answer_match=answer_match,
            status_match=status_match,
            passed=not failures,
            failures=failures,
            trace_id=response.trace_id,
        )

    def _execute(self, case: EvaluationCase) -> AnswerResponse:
        if case.fault is EvaluationFault.NONE:
            pipeline = self.pipeline
            retrieval = self.retrieval
        else:
            fault_chat = _FaultChatProvider(self.pipeline.chat, case.fault)
            pipeline = RAGPipeline(
                chat=fault_chat,
                embeddings=self.pipeline.embeddings,
                default_top_k=self.pipeline.default_top_k,
                documents=self.pipeline.documents,
                generation_max_repair_attempts=self.settings.generation_max_repair_attempts,
                generation_native_structured_output=False,
            )
            retrieval = AdvancedRetrievalService(
                repository=self.retrieval.repository,
                embeddings=self.retrieval.embeddings,
                chat=fault_chat,
                settings=self.settings,
                reranker=self.retrieval.reranker,
                metadata_planner=None,
                indexing=self.retrieval.indexing,
                generation=GenerationService(
                    fault_chat,
                    max_repair_attempts=self.settings.generation_max_repair_attempts,
                    prefer_native_structured_output=False,
                ),
            )
        if case.collection_id is None and case.strategy.value == "dense":
            return pipeline.query(
                case.question,
                case.top_k,
                output_mode=case.output_mode,
                schema_name=case.schema_name,
            )
        return retrieval.query(
            QueryRequest(
                question=case.question,
                top_k=case.top_k,
                collection_id=case.collection_id,
                strategy=case.strategy,
                rewrite=case.rewrite,
                output_mode=case.output_mode,
                schema_name=case.schema_name,
            )
        )

    def _configuration(
        self,
        dataset: EvaluationDatasetResponse,
        thresholds: EvaluationThresholds,
    ) -> dict[str, Any]:
        return {
            "contract_version": "chapter6.evaluation.v1",
            "dataset_sha256": dataset.content_sha256,
            "provider_profile": self.provider_profile().model_dump(mode="json"),
            "embedding_model": self.settings.embedding_model,
            "embedding_revision": self.settings.embedding_model_revision,
            "retrieval": {
                "candidate_limit": self.settings.retrieval_candidate_limit,
                "context_tokens": self.settings.retrieval_context_tokens,
                "answerability_threshold": self.settings.retrieval_answerability_threshold,
                "reranker_enabled": self.settings.reranker_enabled,
            },
            "thresholds": thresholds.model_dump(mode="json"),
        }


class _FaultChatProvider:
    def __init__(self, delegate: ChatProvider, fault: EvaluationFault) -> None:
        self.delegate = delegate
        self.fault = fault

    def check_models(self) -> str:
        return self.delegate.check_models()

    def check_completion(self) -> str:
        return self.delegate.check_completion()

    def generate(self, prompt: str) -> str:
        if self.fault is EvaluationFault.CHAT_UNAVAILABLE:
            raise ProviderUnavailableError(
                "Injected Chapter 6 chat outage", component="evaluation.fault"
            )
        if self.fault is EvaluationFault.EMPTY_OUTPUT:
            return ""
        if self.fault is EvaluationFault.MALFORMED_OUTPUT:
            return "{malformed"
        return self.delegate.generate(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        output = self.generate(prompt)
        if output:
            yield output

    def generate_structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return schema.model_validate_json(self.generate(prompt))


def _dataset_response(record: EvaluationDatasetRecord) -> EvaluationDatasetResponse:
    return EvaluationDatasetResponse(
        id=UUID(record.id),
        name=record.name,
        description=record.description,
        version=record.version,
        content_sha256=record.content_sha256,
        cases=[EvaluationCase.model_validate(item) for item in record.cases],
        created_at=record.created_at,
    )


def _run_response(record: EvaluationRunRecord) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        id=UUID(record.id),
        dataset_id=UUID(record.dataset_id),
        baseline_run_id=(UUID(record.baseline_run_id) if record.baseline_run_id else None),
        status=cast(Literal["running", "completed", "failed"], record.status),
        configuration_sha256=record.configuration_sha256,
        configuration=record.configuration,
        metrics=(EvaluationMetrics.model_validate(record.metrics) if record.metrics else None),
        case_results=[EvaluationCaseResult.model_validate(item) for item in record.case_results],
        gate=(RegressionGate.model_validate(record.gate) if record.gate else None),
        error_message=record.error_message,
        created_at=record.created_at,
        finished_at=record.finished_at,
    )


def _ranked_retrieval(response: AnswerResponse) -> tuple[list[str], list[str]]:
    if response.trace.retrieved:
        ordered = sorted(response.trace.retrieved, key=lambda item: item.rank)
        return (
            _unique([item.chunk.source_id for item in ordered]),
            _unique([item.chunk.chunk_id for item in ordered]),
        )
    if response.trace.retrieval is not None:
        included = [
            item
            for item in response.trace.retrieval.context_manifest
            if item.decision in {"included", "truncated"}
        ]
        return (
            [f"version:{item.version_id}" for item in included],
            [item.chunk_id for item in included],
        )
    return (
        _unique([item.source_id for item in response.citations]),
        _unique([item.chunk_id for item in response.citations]),
    )


def _ranking_metrics(
    expected: Sequence[str], ranked: Sequence[str]
) -> tuple[float | None, float | None, float | None]:
    relevant = set(expected)
    if not relevant:
        return None, None, None
    deduped_ranked = _unique(list(ranked))
    first = next(
        (rank for rank, item in enumerate(deduped_ranked, start=1) if item in relevant),
        None,
    )
    reciprocal_rank = 0.0 if first is None else 1 / first
    hits = [1 if item in relevant else 0 for item in deduped_ranked]
    recall = len(relevant & set(deduped_ranked)) / len(relevant)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), len(deduped_ranked))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return round(reciprocal_rank, 6), round(recall, 6), round(ndcg, 6)


def _retrieval_metrics(
    expected_sources: Sequence[str],
    expected_chunks: Sequence[str],
    ranked_sources: Sequence[str],
    ranked_chunks: Sequence[str],
) -> tuple[float | None, float | None, float | None]:
    relevant = {
        *(f"source:{item}" for item in expected_sources),
        *(f"chunk:{item}" for item in expected_chunks),
    }
    if not relevant:
        return None, None, None
    matched: set[str] = set()
    rank_hits: list[int] = []
    for index in range(max(len(ranked_sources), len(ranked_chunks))):
        identities: set[str] = set()
        if index < len(ranked_sources):
            identities.add(f"source:{ranked_sources[index]}")
        if index < len(ranked_chunks):
            identities.add(f"chunk:{ranked_chunks[index]}")
        current = identities & relevant
        matched.update(current)
        rank_hits.append(int(bool(current)))
    first = next(
        (rank for rank, hit in enumerate(rank_hits, start=1) if hit),
        None,
    )
    reciprocal_rank = 0.0 if first is None else 1 / first
    recall = len(matched) / len(relevant)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(rank_hits, start=1))
    ideal_hits = min(len(relevant), len(rank_hits))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / ideal if ideal else 0.0
    return round(reciprocal_rank, 6), round(recall, 6), round(ndcg, 6)


def _precision_recall(
    expected: Sequence[str], actual: Sequence[str]
) -> tuple[float | None, float | None]:
    relevant = set(expected)
    if not relevant:
        return None, None
    observed = set(actual)
    precision = len(relevant & observed) / len(observed) if observed else 0.0
    recall = len(relevant & observed) / len(relevant)
    return round(precision, 6), round(recall, 6)


def _aggregate(results: Sequence[EvaluationCaseResult]) -> EvaluationMetrics:
    latencies = sorted(item.latency_ms for item in results)
    return EvaluationMetrics(
        case_count=len(results),
        case_pass_rate=_ratio(sum(item.passed for item in results), len(results)),
        recall_at_k=_mean(item.recall_at_k for item in results),
        mean_reciprocal_rank=_mean(item.reciprocal_rank for item in results),
        ndcg_at_k=_mean(item.ndcg_at_k for item in results),
        citation_precision=_mean(item.citation_precision for item in results),
        citation_recall=_mean(item.citation_recall for item in results),
        answer_match_rate=_mean(
            float(item.answer_match) if item.answer_match is not None else None
            for item in results
        ),
        status_match_rate=_ratio(sum(item.status_match for item in results), len(results)),
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
    )


def _gate(
    metrics: EvaluationMetrics,
    thresholds: EvaluationThresholds,
    baseline: EvaluationRunResponse | None,
) -> RegressionGate:
    failures: list[str] = []
    checks = {
        "case_pass_rate": thresholds.min_case_pass_rate,
        "recall_at_k": thresholds.min_recall_at_k,
        "answer_match_rate": thresholds.min_answer_match_rate,
        "status_match_rate": thresholds.min_status_match_rate,
    }
    for name, minimum in checks.items():
        value = getattr(metrics, name)
        if minimum is not None and (value is None or value < minimum):
            failures.append(f"{name} is below its minimum {minimum:.3f}")
    if (
        thresholds.max_p95_latency_ms is not None
        and metrics.p95_latency_ms > thresholds.max_p95_latency_ms
    ):
        failures.append("p95_latency_ms exceeds its configured maximum")
    deltas: dict[str, float] = {}
    if baseline is not None and baseline.metrics is not None:
        deltas = _metric_deltas(baseline.metrics, metrics)
        for name in (
            "case_pass_rate",
            "recall_at_k",
            "mean_reciprocal_rank",
            "ndcg_at_k",
            "citation_precision",
            "citation_recall",
            "answer_match_rate",
            "status_match_rate",
        ):
            delta = deltas.get(name)
            if delta is not None and delta < -thresholds.max_metric_decline:
                failures.append(f"{name} regressed by {abs(delta):.3f}")
        latency_delta = deltas.get("p95_latency_ms")
        if (
            latency_delta is not None
            and thresholds.max_latency_increase_ms is not None
            and latency_delta > thresholds.max_latency_increase_ms
        ):
            failures.append(f"p95_latency_ms increased by {latency_delta:.3f} ms")
    return RegressionGate(
        passed=not failures,
        failures=failures,
        baseline_run_id=(baseline.id if baseline is not None else None),
        deltas=deltas,
    )


def _metric_deltas(
    baseline: EvaluationMetrics, candidate: EvaluationMetrics
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for name in EvaluationMetrics.model_fields:
        if name == "case_count":
            continue
        before = getattr(baseline, name)
        after = getattr(candidate, name)
        if before is not None and after is not None:
            deltas[name] = round(float(after) - float(before), 6)
    return deltas


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _mean(values: Sequence[float | None] | Any) -> float | None:
    available = [float(item) for item in values if item is not None]
    if not available:
        return None
    return round(sum(available) / len(available), 6)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    index = max(0, math.ceil(len(values) * percentile) - 1)
    return round(values[index], 3)


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, EvaluationError):
        return str(exc)
    return f"{type(exc).__name__}: evaluation case execution failed"
