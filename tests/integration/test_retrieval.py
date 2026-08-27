from __future__ import annotations

from uuid import UUID

from local_lke.ingestion import IngestionService
from local_lke.models import (
    AnswerStatus,
    ChunkStrategy,
    MetadataCondition,
    MetadataFilterPlan,
    MetadataOperator,
    ParserStrategy,
    QueryRequest,
    RetrievalStrategy,
)
from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.retrieval import (
    AdvancedRetrievalService,
    MetadataPlanParser,
    ScopedLexicalRetriever,
)
from local_lke.settings import Settings
from local_lke.storage import SqlAlchemyIngestionRepository


class PreferredReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [10.0 if "preferred" in document else 0.1 for document in documents]


class LexicalBlindEmbeddings:
    """Rank thematic prose over an exact identifier to label hybrid recall."""

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] if "ZXQ-4917" in text else [1.0, 0.0] for text in texts]

    def check_initialization(self) -> str:
        return "labelled retrieval embedding"


class CountingEmbeddings(DeterministicFakeEmbeddings):
    def __init__(self) -> None:
        super().__init__()
        self.query_calls = 0

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return super().embed_query(text)


def _service(
    ingestion: IngestionService,
    settings: Settings,
    *,
    reranker: PreferredReranker | None = None,
) -> AdvancedRetrievalService:
    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    return AdvancedRetrievalService(
        repository=ingestion.repository,
        embeddings=DeterministicFakeEmbeddings(),
        chat=FakeChatProvider("The Zephyr deployment code is ZXQ-4917."),
        settings=settings,
        reranker=reranker,
    )


def _ingest(
    ingestion: IngestionService,
    collection_id: UUID,
    filename: str,
    text: str,
) -> None:
    job = ingestion.ingest(
        collection_id=collection_id,
        filename=filename,
        content_type="text/plain",
        content=text.encode(),
        parser_strategy=ParserStrategy.FAST,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        chunk_size=200,
        chunk_overlap=20,
    )
    assert job.error_code is None


def test_hybrid_retrieval_exposes_all_ranks_and_rare_exact_match(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Hybrid")
    _ingest(
        ingestion,
        collection.id,
        "deployment.txt",
        "Zephyr deployments use the rare exact code ZXQ-4917 for emergency activation.",
    )
    _ingest(
        ingestion,
        collection.id,
        "general.txt",
        "Deployment procedures require peer review and a maintenance window.",
    )
    response = _service(ingestion, settings).query(
        QueryRequest(
            collection_id=collection.id,
            question="What is the Zephyr deployment code ZXQ-4917?",
            strategy=RetrievalStrategy.HYBRID,
            top_k=2,
        )
    )

    assert response.status is AnswerStatus.ANSWERED
    assert response.citations[0].document_version_id is not None
    assert response.trace.retrieval is not None
    trace = response.trace.retrieval
    exact = next(item for item in trace.candidates if item.filename == "deployment.txt")
    assert exact.lexical_rank == 1
    assert exact.dense_rank is not None
    assert exact.fused_rank == 1
    assert "zxq-4917" in exact.matched_terms
    assert trace.context_manifest[0].decision in {"included", "truncated"}


def test_hybrid_improves_labelled_rare_identifier_recall_at_one(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Recall at one")
    _ingest(
        ingestion,
        collection.id,
        "exact.txt",
        "The emergency identifier is ZXQ-4917.",
    )
    _ingest(
        ingestion,
        collection.id,
        "semantic.txt",
        "Emergency identifiers are documented in the deployment guide.",
    )
    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    service = AdvancedRetrievalService(
        repository=ingestion.repository,
        embeddings=LexicalBlindEmbeddings(),
        chat=FakeChatProvider(),
        settings=settings,
    )
    dense = service._retrieve_once(
        QueryRequest(
            collection_id=collection.id,
            question="ZXQ-4917",
            strategy=RetrievalStrategy.DENSE,
            top_k=1,
        ),
        allow_filter_fallback=False,
    )
    hybrid = service._retrieve_once(
        QueryRequest(
            collection_id=collection.id,
            question="ZXQ-4917",
            strategy=RetrievalStrategy.HYBRID,
            top_k=1,
        ),
        allow_filter_fallback=False,
    )

    dense_recall_at_one = int(dense.trace.candidates[0].filename == "exact.txt")
    hybrid_recall_at_one = int(hybrid.trace.candidates[0].filename == "exact.txt")
    assert (dense_recall_at_one, hybrid_recall_at_one) == (0, 1)


def test_metadata_filters_are_applied_before_retrieval_and_do_not_fallback_by_default(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Metadata")
    _ingest(ingestion, collection.id, "allowed.txt", "Zephyr code ZXQ-4917 is preferred.")
    _ingest(ingestion, collection.id, "excluded.txt", "Zephyr code ZXQ-4917 is obsolete.")
    request = QueryRequest(
        collection_id=collection.id,
        question="What is the Zephyr code ZXQ-4917?",
        strategy=RetrievalStrategy.HYBRID,
        metadata_filter=MetadataFilterPlan(
            conditions=[
                MetadataCondition(
                    field="filename",
                    operator=MetadataOperator.EQ,
                    value="allowed.txt",
                )
            ]
        ),
    )

    result = _service(ingestion, settings).retrieve(request)

    assert {item.filename for item in result.trace.candidates} == {"allowed.txt"}

    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    adapter = ScopedLexicalRetriever(
        repository=ingestion.repository,
        collection_id=collection.id,
        filters=request.metadata_filter or MetadataFilterPlan(),
        limit=5,
    )
    documents = adapter.invoke("ZXQ-4917")
    assert documents[0].metadata["lexical_rank"] == 1
    assert documents[0].metadata["filename"] == "allowed.txt"


def test_requested_model_metadata_plan_is_validated_and_applied(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Model metadata")
    _ingest(ingestion, collection.id, "runbook.txt", "Zephyr code is ZXQ-4917.")
    _ingest(ingestion, collection.id, "notes.txt", "Zephyr code is obsolete.")
    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    planner = MetadataPlanParser(
        FakeChatProvider(
            '{"conditions":[{"field":"filename","operator":"eq",'
            '"value":"runbook.txt"}],"allow_unfiltered_fallback":false}'
        )
    )
    service = AdvancedRetrievalService(
        repository=ingestion.repository,
        embeddings=DeterministicFakeEmbeddings(),
        chat=FakeChatProvider(),
        settings=settings,
        metadata_planner=planner,
    )
    result = service.retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="Use the runbook for the Zephyr code",
            strategy=RetrievalStrategy.HYBRID,
            infer_metadata_filter=True,
        )
    )

    assert {item.filename for item in result.trace.candidates} == {"runbook.txt"}
    assert result.trace.metadata_filter.conditions[0].field == "filename"


def test_reranker_records_before_after_latency_and_gain(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Reranking")
    _ingest(ingestion, collection.id, "first.txt", "Zephyr deployment code overview.")
    _ingest(
        ingestion,
        collection.id,
        "preferred.txt",
        "A preferred source states the Zephyr deployment code ZXQ-4917.",
    )
    result = _service(ingestion, settings, reranker=PreferredReranker()).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="Zephyr deployment code",
            strategy=RetrievalStrategy.HYBRID,
        )
    )

    preferred = next(item for item in result.trace.candidates if item.filename == "preferred.txt")
    assert preferred.rerank_before is not None
    assert preferred.rerank_after == 1
    assert preferred.rerank_score == 10.0
    assert result.trace.reranker_latency_ms >= 0
    assert result.trace.reranker_top_gain != 0


def test_unanswerable_query_retries_once_then_abstains(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Abstention")
    _ingest(ingestion, collection.id, "runbook.txt", "Restart Atlas with the blue command.")
    assert isinstance(ingestion.repository, SqlAlchemyIngestionRepository)
    embeddings = CountingEmbeddings()
    service = AdvancedRetrievalService(
        repository=ingestion.repository,
        embeddings=embeddings,
        chat=FakeChatProvider(),
        settings=settings,
    )
    response = service.query(
        QueryRequest(
            collection_id=collection.id,
            question="Which lunar mineral powers the quantum banana?",
            strategy=RetrievalStrategy.DENSE,
        )
    )

    assert response.status is AnswerStatus.ABSTAINED
    assert response.citations == []
    assert response.trace.retrieval is not None
    assert response.trace.retrieval.answerability.corrective_attempted is True
    assert response.trace.retrieval.answerability.initial_failure_reason is not None
    assert "alternate retrieval" in response.trace.retrieval.answerability.reason
    assert embeddings.query_calls == 3  # one initial probe + two bounded correction probes


def test_only_active_non_deleted_versions_are_retrievable(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Lifecycle")
    _ingest(ingestion, collection.id, "policy.txt", "Legacy token OLD-111 remains active.")
    _ingest(ingestion, collection.id, "policy.txt", "Current token NEW-222 remains active.")
    result = _service(ingestion, settings).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="What is the current token NEW-222?",
            strategy=RetrievalStrategy.HYBRID,
        )
    )
    assert result.trace.candidates
    assert all("OLD-111" not in item.text for item in result.context)

    document = ingestion.list_documents(collection.id)[0]
    ingestion.delete_document(document.id, "lifecycle test")
    deleted = _service(ingestion, settings).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="What is NEW-222?",
            strategy=RetrievalStrategy.HYBRID,
        )
    )
    assert deleted.context == []
    assert deleted.trace.answerability.sufficient is False


def test_multi_part_context_covers_every_subquery_and_accounts_for_candidates(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Coverage")
    _ingest(ingestion, collection.id, "alpha.txt", "Alpha retention lasts seven days.")
    _ingest(ingestion, collection.id, "beta.txt", "Beta ownership belongs to Platform.")
    result = _service(ingestion, settings).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="What is Alpha retention? And also who owns Beta?",
            strategy=RetrievalStrategy.HYBRID,
            top_k=2,
        )
    )

    assert len(result.trace.transform.subqueries) == 2
    assert result.trace.answerability.subquery_coverage == 1.0
    included_coverage = {
        subquery
        for item in result.trace.context_manifest
        if item.decision in {"included", "truncated"}
        for subquery in item.covered_subqueries
    }
    assert included_coverage == set(result.trace.transform.subqueries)
    assert {item.chunk_id for item in result.trace.context_manifest} == {
        item.chunk_id for item in result.trace.candidates
    }


def test_unfiltered_fallback_requires_explicit_plan_permission(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Fallback")
    _ingest(ingestion, collection.id, "present.txt", "Zephyr code is ZXQ-4917.")
    condition = MetadataCondition(
        field="filename",
        operator=MetadataOperator.EQ,
        value="missing.txt",
    )
    strict = _service(ingestion, settings).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="What is the Zephyr code ZXQ-4917?",
            strategy=RetrievalStrategy.HYBRID,
            metadata_filter=MetadataFilterPlan(conditions=[condition]),
        )
    )
    permissive = _service(ingestion, settings).retrieve(
        QueryRequest(
            collection_id=collection.id,
            question="What is the Zephyr code ZXQ-4917?",
            strategy=RetrievalStrategy.HYBRID,
            metadata_filter=MetadataFilterPlan(
                conditions=[condition], allow_unfiltered_fallback=True
            ),
        )
    )

    assert strict.context == []
    assert permissive.context
    assert strict.trace.metadata_fallback_used is False
    assert permissive.trace.metadata_fallback_used is True
    assert permissive.trace.candidates[0].lexical_rank == 1
