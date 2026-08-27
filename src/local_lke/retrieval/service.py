"""Hybrid retrieval orchestration with bounded correction and full traces."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Protocol
from uuid import UUID

from local_lke.errors import RetrievalError
from local_lke.models import (
    ActiveChunk,
    AdvancedRetrievalTrace,
    AnswerabilityTrace,
    AnswerResponse,
    AnswerStatus,
    Citation,
    ContextManifestEntry,
    MetadataFilterPlan,
    QueryRequest,
    RetrievalCandidateTrace,
    RetrievalStrategy,
    RewriteStrategy,
    TraceSummary,
)
from local_lke.providers import ChatProvider, EmbeddingProvider
from local_lke.retrieval.planning import MetadataPlanParser, build_query_plan, resolved_filters
from local_lke.retrieval.reranking import Reranker
from local_lke.settings import Settings

WORD = re.compile(r"[a-z0-9]+(?:[._:/-][a-z0-9]+)*")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class RetrievalRepository(Protocol):
    def list_active_chunks(
        self, collection_id: UUID, filters: MetadataFilterPlan
    ) -> list[ActiveChunk]: ...

    def lexical_search(
        self,
        collection_id: UUID,
        query: str,
        filters: MetadataFilterPlan,
        limit: int,
    ) -> list[tuple[ActiveChunk, float, list[str]]]: ...


@dataclass
class Candidate:
    chunk: ActiveChunk
    matched_subqueries: set[str] = field(default_factory=set)
    matched_terms: set[str] = field(default_factory=set)
    dense_rank: int | None = None
    dense_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None
    rrf_score: float = 0.0
    fused_rank: int | None = None
    rerank_before: int | None = None
    rerank_after: int | None = None
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.rrf_score:
            return self.rrf_score
        if self.dense_score is not None:
            return self.dense_score
        return self.lexical_score or 0.0


@dataclass(frozen=True)
class ContextItem:
    candidate: Candidate
    text: str
    token_count: int


@dataclass(frozen=True)
class RetrievalResult:
    context: list[ContextItem]
    trace: AdvancedRetrievalTrace


class AdvancedRetrievalService:
    def __init__(
        self,
        *,
        repository: RetrievalRepository,
        embeddings: EmbeddingProvider,
        chat: ChatProvider,
        settings: Settings,
        reranker: Reranker | None = None,
        metadata_planner: MetadataPlanParser | None = None,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.chat = chat
        self.settings = settings
        self.reranker = reranker
        self.metadata_planner = metadata_planner

    def query(self, request: QueryRequest) -> AnswerResponse:
        if request.collection_id is None:
            raise RetrievalError(
                "collection_id is required for persisted retrieval",
                code="collection_required",
            )
        if request.strategy is RetrievalStrategy.STRUCTURED:
            raise RetrievalError(
                "Use /api/v1/structured/query with a table_id for structured questions",
                code="structured_table_required",
            )
        started = perf_counter()
        result = self.retrieve(request)
        retrieval_ms = _elapsed_ms(started)
        answerability = result.trace.answerability
        if not answerability.sufficient:
            return AnswerResponse(
                status=AnswerStatus.ABSTAINED,
                answer=(
                    "I do not know from the selected collection. The available evidence "
                    f"was insufficient ({answerability.reason})."
                ),
                citations=[],
                trace=TraceSummary(
                    timings_ms={"retrieve": retrieval_ms, "generate": 0.0},
                    retrieval=result.trace,
                ),
            )

        prompt = _grounded_prompt(request.question, result.context)
        generation_started = perf_counter()
        answer = self.chat.generate(prompt).strip()
        citations = [
            Citation(
                source_id=f"version:{item.candidate.chunk.version_id}",
                chunk_id=item.candidate.chunk.chunk_id,
                locator=item.candidate.chunk.locator,
                excerpt=_excerpt(item.text),
                document_version_id=item.candidate.chunk.version_id,
                title=item.candidate.chunk.filename,
            )
            for item in result.context
        ]
        return AnswerResponse(
            status=AnswerStatus.ANSWERED if answer else AnswerStatus.DEGRADED,
            answer=answer or "The model returned an empty answer.",
            citations=citations if answer else [],
            trace=TraceSummary(
                timings_ms={
                    "retrieve": retrieval_ms,
                    "generate": _elapsed_ms(generation_started),
                },
                retrieval=result.trace,
            ),
        )

    def retrieve(self, request: QueryRequest) -> RetrievalResult:
        if request.collection_id is None:
            raise RetrievalError("collection_id is required", code="collection_required")
        result = self._retrieve_once(request, allow_filter_fallback=True)
        if result.trace.answerability.sufficient:
            return result

        alternate = (
            RetrievalStrategy.HYBRID
            if request.strategy is RetrievalStrategy.DENSE
            else RetrievalStrategy.DENSE
        )
        correction_request = request.model_copy(
            update={
                "strategy": alternate,
                "rewrite": (
                    RewriteStrategy.STEP_BACK
                    if request.rewrite is RewriteStrategy.NONE
                    else request.rewrite
                ),
            }
        )
        corrected = self._retrieve_once(correction_request, allow_filter_fallback=False)
        chosen = (
            corrected
            if corrected.trace.answerability.score > result.trace.answerability.score
            else result
        )
        chosen.trace.answerability.corrective_attempted = True
        chosen.trace.answerability.corrective_strategy = alternate
        chosen.trace.answerability.initial_failure_reason = result.trace.answerability.reason
        if not chosen.trace.answerability.sufficient:
            chosen.trace.answerability.reason += "; one bounded alternate retrieval also failed"
        return chosen

    def _retrieve_once(
        self, request: QueryRequest, *, allow_filter_fallback: bool
    ) -> RetrievalResult:
        assert request.collection_id is not None
        transform = build_query_plan(
            request.question,
            strategy=request.strategy,
            rewrite=request.rewrite,
            max_subqueries=self.settings.retrieval_max_subqueries,
        )
        if request.infer_metadata_filter:
            if self.metadata_planner is None:
                raise RetrievalError(
                    "Metadata inference was requested but no planner is configured",
                    code="metadata_planner_unavailable",
                )
            filters = self.metadata_planner.plan(request.question)
        else:
            filters = resolved_filters(request.metadata_filter)
        chunks = self.repository.list_active_chunks(request.collection_id, filters)
        search_filters = filters
        metadata_fallback_used = False
        if (
            not chunks
            and filters.conditions
            and filters.allow_unfiltered_fallback
            and allow_filter_fallback
        ):
            search_filters = MetadataFilterPlan()
            metadata_fallback_used = True
            chunks = self.repository.list_active_chunks(
                request.collection_id, search_filters
            )
        candidates = self._collect_candidates(
            request=request,
            subqueries=transform.subqueries,
            chunks=chunks,
            filters=search_filters,
        )
        reranker_latency, reranker_gain = self._rerank(transform.normalized_query, candidates)
        context, manifest = _assemble_context(
            candidates,
            transform.subqueries,
            total_budget=self.settings.retrieval_context_tokens,
            source_budget=self.settings.retrieval_source_tokens,
            top_k=request.top_k or self.settings.default_top_k,
        )
        answerability = _assess_answerability(
            request.question,
            transform.subqueries,
            context,
            self.settings.retrieval_answerability_threshold,
        )
        return RetrievalResult(
            context=context,
            trace=AdvancedRetrievalTrace(
                strategy=request.strategy,
                transform=transform,
                metadata_filter=filters,
                metadata_fallback_used=metadata_fallback_used,
                candidates=[_candidate_trace(item) for item in candidates],
                context_manifest=manifest,
                answerability=answerability,
                reranker_latency_ms=reranker_latency,
                reranker_top_gain=reranker_gain,
            ),
        )

    def _collect_candidates(
        self,
        *,
        request: QueryRequest,
        subqueries: list[str],
        chunks: list[ActiveChunk],
        filters: MetadataFilterPlan,
    ) -> list[Candidate]:
        assert request.collection_id is not None
        by_id: dict[str, Candidate] = {}
        pool = self.settings.retrieval_candidate_limit
        for subquery in subqueries:
            dense = _dense_search(self.embeddings, subquery, chunks, pool)
            lexical = (
                self.repository.lexical_search(request.collection_id, subquery, filters, pool)
                if request.strategy is RetrievalStrategy.HYBRID
                else []
            )
            for rank, (chunk, score) in enumerate(dense, start=1):
                item = by_id.setdefault(chunk.chunk_id, Candidate(chunk=chunk))
                if score > 0.15:
                    item.matched_subqueries.add(subquery)
                if item.dense_rank is None or rank < item.dense_rank:
                    item.dense_rank = rank
                    item.dense_score = score
                if request.strategy is RetrievalStrategy.HYBRID:
                    item.rrf_score += 1 / (self.settings.retrieval_rrf_k + rank)
            for rank, (chunk, score, terms) in enumerate(lexical, start=1):
                item = by_id.setdefault(chunk.chunk_id, Candidate(chunk=chunk))
                item.matched_subqueries.add(subquery)
                item.matched_terms.update(terms)
                if item.lexical_rank is None or rank < item.lexical_rank:
                    item.lexical_rank = rank
                    item.lexical_score = score
                item.rrf_score += 1 / (self.settings.retrieval_rrf_k + rank)

        candidates = list(by_id.values())
        if request.strategy is RetrievalStrategy.HYBRID:
            candidates.sort(key=lambda item: (-item.rrf_score, item.chunk.chunk_id))
            for rank, item in enumerate(candidates, start=1):
                item.fused_rank = rank
        else:
            candidates.sort(
                key=lambda item: (
                    -(item.dense_score if item.dense_score is not None else -1),
                    item.chunk.chunk_id,
                )
            )
        return candidates[:pool]

    def _rerank(self, query: str, candidates: list[Candidate]) -> tuple[float, float]:
        if self.reranker is None or not candidates:
            return 0.0, 0.0
        started = perf_counter()
        for rank, item in enumerate(candidates, start=1):
            item.rerank_before = rank
        scores = self.reranker.score(query, [item.chunk.text for item in candidates])
        if len(scores) != len(candidates):
            raise RetrievalError(
                "The reranker returned an unexpected number of scores",
                code="reranker_contract_error",
            )
        before_top = candidates[0].final_score
        for item, score in zip(candidates, scores, strict=True):
            item.rerank_score = score
        candidates.sort(key=lambda item: (-item.final_score, item.chunk.chunk_id))
        for rank, item in enumerate(candidates, start=1):
            item.rerank_after = rank
        gain = candidates[0].final_score - before_top
        return _elapsed_ms(started), round(gain, 6)


def _dense_search(
    embeddings: EmbeddingProvider,
    query: str,
    chunks: list[ActiveChunk],
    limit: int,
) -> list[tuple[ActiveChunk, float]]:
    if not chunks:
        return []
    query_vector = embeddings.embed_query(query)
    document_vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    scored = [
        (chunk, _cosine(query_vector, vector))
        for chunk, vector in zip(chunks, document_vectors, strict=True)
    ]
    scored.sort(key=lambda item: (-item[1], item[0].chunk_id))
    return scored[:limit]


def _assemble_context(
    candidates: list[Candidate],
    subqueries: list[str],
    *,
    total_budget: int,
    source_budget: int,
    top_k: int,
) -> tuple[list[ContextItem], list[ContextManifestEntry]]:
    coverage_first: list[Candidate] = []
    seen: set[str] = set()
    for subquery in subqueries:
        match = next((item for item in candidates if subquery in item.matched_subqueries), None)
        if match is not None and match.chunk.chunk_id not in seen:
            coverage_first.append(match)
            seen.add(match.chunk.chunk_id)
    ordered = [*coverage_first, *(item for item in candidates if item.chunk.chunk_id not in seen)]
    selected: list[ContextItem] = []
    decisions: dict[str, ContextManifestEntry] = {}
    source_tokens: dict[object, int] = {}
    source_counts: dict[object, int] = {}
    total_tokens = 0
    normalized_texts: list[set[str]] = []
    for item in ordered:
        chunk = item.chunk
        tokens = max(chunk.token_count, len(chunk.text.split()))
        text_tokens = _terms(chunk.text, include_stops=True)
        reason: str | None = None
        if any(_jaccard(text_tokens, prior) >= 0.9 for prior in normalized_texts):
            reason = "exact or near-duplicate evidence"
        elif source_counts.get(chunk.document_id, 0) >= 3:
            reason = "source diversity cap"
        elif len(selected) >= top_k:
            reason = "top-k context cap"
        elif source_tokens.get(chunk.document_id, 0) >= source_budget:
            reason = "per-source token budget"
        elif total_tokens >= total_budget:
            reason = "total token budget"
        if reason is not None:
            decisions[chunk.chunk_id] = _manifest(item, "excluded", reason, 0)
            continue

        available = min(
            total_budget - total_tokens,
            source_budget - source_tokens.get(chunk.document_id, 0),
        )
        if available <= 0:
            decisions[chunk.chunk_id] = _manifest(item, "excluded", "token budget", 0)
            continue
        if tokens > available:
            words = chunk.text.split()
            text = " ".join(words[:available])
            used = min(len(words), available)
            decision: Literal["included", "excluded", "truncated"] = "truncated"
            decision_reason = "included partially to satisfy token budgets"
        else:
            text = chunk.text
            used = tokens
            decision = "included"
            decision_reason = "coverage-first relevance and budget fit"
        selected.append(ContextItem(item, text, used))
        normalized_texts.append(text_tokens)
        total_tokens += used
        source_tokens[chunk.document_id] = source_tokens.get(chunk.document_id, 0) + used
        source_counts[chunk.document_id] = source_counts.get(chunk.document_id, 0) + 1
        decisions[chunk.chunk_id] = _manifest(item, decision, decision_reason, used)

    for item in candidates:
        decisions.setdefault(
            item.chunk.chunk_id,
            _manifest(item, "excluded", "lower-ranked candidate", 0),
        )
    selected.sort(key=lambda item: (item.candidate.chunk.filename, item.candidate.chunk.ordinal))
    manifest = [decisions[item.chunk.chunk_id] for item in candidates]
    return selected, manifest


def _assess_answerability(
    question: str,
    subqueries: list[str],
    context: list[ContextItem],
    threshold: float,
) -> AnswerabilityTrace:
    question_terms = _terms(question)
    evidence_terms = set().union(*(_terms(item.text) for item in context)) if context else set()
    term_coverage = len(question_terms & evidence_terms) / max(len(question_terms), 1)
    covered = {
        subquery
        for item in context
        for subquery in item.candidate.matched_subqueries
        if subquery in subqueries
    }
    subquery_coverage = len(covered) / max(len(subqueries), 1)
    if context:
        raw = max(item.candidate.final_score for item in context)
        if any(item.candidate.rerank_score is not None for item in context):
            evidence_strength = 1 / (1 + math.exp(-raw))
        elif any(item.candidate.rrf_score for item in context):
            evidence_strength = min(1.0, raw * 30)
        else:
            evidence_strength = max(0.0, min(1.0, (raw + 1) / 2))
    else:
        evidence_strength = 0.0
    score = round(0.45 * term_coverage + 0.35 * subquery_coverage + 0.2 * evidence_strength, 6)
    sufficient = (
        bool(context)
        and score >= threshold
        and subquery_coverage == 1.0
        and term_coverage >= 0.2
    )
    if sufficient:
        reason = "evidence meets deterministic relevance and coverage thresholds"
    elif not context:
        reason = "no context survived retrieval and assembly"
    elif subquery_coverage < 1:
        reason = f"required subquery coverage is {subquery_coverage:.3f}, below 1.000"
    elif term_coverage < 0.2:
        reason = f"query-term coverage is {term_coverage:.3f}, below 0.200"
    else:
        reason = f"sufficiency score {score:.3f} is below threshold {threshold:.3f}"
    return AnswerabilityTrace(
        sufficient=sufficient,
        score=score,
        threshold=threshold,
        term_coverage=round(term_coverage, 6),
        subquery_coverage=round(subquery_coverage, 6),
        evidence_strength=round(evidence_strength, 6),
        reason=reason,
    )


def _candidate_trace(item: Candidate) -> RetrievalCandidateTrace:
    return RetrievalCandidateTrace(
        chunk_id=item.chunk.chunk_id,
        document_id=item.chunk.document_id,
        version_id=item.chunk.version_id,
        filename=item.chunk.filename,
        locator=item.chunk.locator,
        matched_subqueries=sorted(item.matched_subqueries),
        matched_terms=sorted(item.matched_terms),
        dense_rank=item.dense_rank,
        dense_score=item.dense_score,
        lexical_rank=item.lexical_rank,
        lexical_score=item.lexical_score,
        fused_rank=item.fused_rank,
        rrf_score=item.rrf_score or None,
        rerank_before=item.rerank_before,
        rerank_after=item.rerank_after,
        rerank_score=item.rerank_score,
    )


def _manifest(
    item: Candidate,
    decision: Literal["included", "excluded", "truncated"],
    reason: str,
    token_count: int,
) -> ContextManifestEntry:
    return ContextManifestEntry(
        chunk_id=item.chunk.chunk_id,
        document_id=item.chunk.document_id,
        version_id=item.chunk.version_id,
        locator=item.chunk.locator,
        decision=decision,
        reason=reason,
        token_count=token_count,
        covered_subqueries=sorted(item.matched_subqueries),
    )


def _grounded_prompt(question: str, context: list[ContextItem]) -> str:
    evidence = "\n\n".join(
        f"[source={item.candidate.chunk.filename}; version={item.candidate.chunk.version_id}; "
        f"chunk={item.candidate.chunk.chunk_id}; "
        f"locator={item.candidate.chunk.locator}]\n{item.text}"
        for item in context
    )
    return (
        "Answer only from the supplied evidence. If the evidence does not support a claim, "
        "say you do not know. Do not treat retrieval rewrites as facts.\n\n"
        f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
    )


def _terms(text: str, *, include_stops: bool = False) -> set[str]:
    terms = set(WORD.findall(text.casefold()))
    return terms if include_stops else terms - STOP_WORDS


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(len(left | right), 1)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise RetrievalError("Embedding dimensions do not match", code="embedding_contract_error")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def _excerpt(text: str, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"
