from uuid import UUID

from local_lke.indexing import IndexingService, SqlAlchemyIndexRepository
from local_lke.ingestion import IngestionService
from local_lke.models import (
    ChunkStrategy,
    ExpansionStrategy,
    JobStatus,
    ParserStrategy,
    VectorSearchRequest,
)
from local_lke.providers import DeterministicFakeEmbeddings
from local_lke.settings import Settings
from local_lke.storage import SqlAlchemyIngestionRepository


class CountingEmbeddings(DeterministicFakeEmbeddings):
    def __init__(self, dimensions: int = 64, fail_call: int | None = None) -> None:
        super().__init__(dimensions)
        self.calls = 0
        self.fail_call = fail_call

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        if self.fail_call == self.calls:
            raise RuntimeError("synthetic failed batch")
        return super().embed_documents(texts)


def _ingest(
    ingestion: IngestionService,
    collection_id: UUID,
    content: bytes,
    *,
    chunk_size: int = 200,
    filename: str = "index-guide.md",
):
    return ingestion.ingest(
        collection_id=collection_id,
        filename=filename,
        content_type="text/markdown",
        content=content,
        parser_strategy=ParserStrategy.FAST,
        chunk_strategy=ChunkStrategy.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=20,
    )


def _service(
    ingestion: IngestionService,
    settings: Settings,
    embeddings: DeterministicFakeEmbeddings,
) -> IndexingService:
    repository = ingestion.repository
    assert isinstance(repository, SqlAlchemyIngestionRepository)
    return IndexingService(
        SqlAlchemyIndexRepository(repository.sessions, repository.engine),
        embeddings,
        settings,
    )


def test_unchanged_version_performs_zero_new_embedding_calls(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Index idempotency")
    job = _ingest(
        ingestion,
        collection.id,
        b"# Operations\n\nPriority-one incidents are acknowledged in fifteen minutes.",
    )
    assert job.version_id is not None
    embeddings = CountingEmbeddings()
    indexing = _service(ingestion, settings, embeddings)

    first = indexing.index_version(job.version_id)
    calls = embeddings.calls
    second = indexing.index_version(job.version_id)
    state = indexing.state(collection.id)

    assert first.status is JobStatus.COMPLETED
    assert second.skipped is True
    assert embeddings.calls == calls
    assert state.missing_active_chunks == 0
    assert state.active_chunks == job.chunk_count


def test_failed_batch_retry_reuses_completed_batches(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Index retry")
    job = _ingest(
        ingestion,
        collection.id,
        (
            b"# Retry\n\nAlpha is first. Beta is second. Gamma is third. "
            b"Delta is fourth. Epsilon is fifth."
        ),
    )
    assert job.version_id is not None
    embeddings = CountingEmbeddings(fail_call=2)
    indexing = _service(
        ingestion,
        settings.model_copy(update={"embedding_batch_size": 1}),
        embeddings,
    )

    failed = indexing.index_version(job.version_id)
    completed_before_retry = failed.embedded_nodes
    embeddings.fail_call = None
    retried = indexing.index_version(job.version_id)

    assert failed.status is JobStatus.FAILED
    assert failed.error_code == "embedding_batch_failed"
    assert completed_before_retry == 1
    assert retried.status is JobStatus.COMPLETED
    assert retried.embedded_nodes == retried.total_nodes
    assert retried.embedding_calls == retried.total_nodes


def test_sentence_window_parent_dedup_and_token_budget(
    ingestion: IngestionService,
    indexing: IndexingService,
) -> None:
    collection = ingestion.create_collection("Expansion")
    job = _ingest(
        ingestion,
        collection.id,
        (
            b"# Incident policy\n\nThe alert opens the incident. The commander acknowledges "
            b"priority one in fifteen minutes. The responder posts updates every thirty "
            b"minutes. The final review records the timeline and corrective actions."
        ),
        chunk_size=120,
    )
    assert job.version_id is not None
    indexed = indexing.index_version(job.version_id)
    assert indexed.status is JobStatus.COMPLETED

    window = indexing.search(
        VectorSearchRequest(
            collection_id=collection.id,
            question="commander acknowledges priority one in fifteen minutes",
            expansion=ExpansionStrategy.SENTENCE_WINDOW,
            top_k=3,
            sentence_window=1,
            token_budget=80,
        )
    )
    parent = indexing.search(
        VectorSearchRequest(
            collection_id=collection.id,
            question="priority one incident policy timeline corrective actions",
            expansion=ExpansionStrategy.PARENT,
            top_k=5,
            token_budget=32,
        )
    )

    window_context = window.final_context[0].context_text
    assert "fifteen minutes" in window_context
    assert window_context.index("alert") < window_context.index("commander")
    assert window.final_context[0].trigger_node_id == window.final_context[0].node_id
    assert window.final_token_count <= 80
    assert parent.final_token_count <= 32
    assert len({item.context_text for item in parent.final_context}) == len(
        parent.final_context
    )
    assert any("duplicate" in item.decision for item in parent.candidates)


def test_new_version_replaces_active_vectors_and_deletion_deactivates_them(
    ingestion: IngestionService,
    indexing: IndexingService,
) -> None:
    collection = ingestion.create_collection("Index lifecycle")
    first = _ingest(ingestion, collection.id, b"# Code\n\nThe code is OLD-100.")
    assert first.version_id is not None
    indexing.index_version(first.version_id)
    second = _ingest(ingestion, collection.id, b"# Code\n\nThe code is NEW-200.")
    assert second.version_id is not None
    indexing.index_version(second.version_id)

    result = indexing.search(
        VectorSearchRequest(
            collection_id=collection.id,
            question="What is the code NEW-200?",
            top_k=5,
        )
    )
    document = ingestion.list_documents(collection.id)[0]
    ingestion.delete_document(document.id, "retention request")
    state = indexing.state(collection.id)

    assert all(item.version_id == second.version_id for item in result.candidates)
    assert state.active_nodes == 0
    assert state.active_chunks == 0


def test_dimension_mismatch_is_visible_and_never_activates_partial_index(
    ingestion: IngestionService, settings: Settings
) -> None:
    collection = ingestion.create_collection("Dimension mismatch")
    job = _ingest(ingestion, collection.id, b"# Dimensions\n\nVectors must agree.")
    assert job.version_id is not None
    indexing = _service(ingestion, settings, DeterministicFakeEmbeddings(32))

    result = indexing.index_version(job.version_id)
    state = indexing.state(collection.id)

    assert result.status is JobStatus.FAILED
    assert result.error_code == "embedding_dimension_mismatch"
    assert state.active_nodes == 0


def test_labelled_fine_retrieval_meets_recall_at_three_baseline(
    ingestion: IngestionService,
    indexing: IndexingService,
) -> None:
    """The Chapter 3 deterministic baseline is Recall@3 = 1.0."""
    collection = ingestion.create_collection("Recall baseline")
    fixtures = (
        ("cedar.md", b"# Cedar\n\nThe cedar release code is ORBIT-731.", "ORBIT-731"),
        ("harbor.md", b"# Harbor\n\nThe harbor release code is TIDAL-284.", "TIDAL-284"),
        ("summit.md", b"# Summit\n\nThe summit release code is AURORA-956.", "AURORA-956"),
    )
    labelled_queries: list[tuple[str, UUID]] = []
    for filename, content, answer in fixtures:
        job = _ingest(ingestion, collection.id, content, filename=filename)
        assert job.version_id is not None
        indexing.index_version(job.version_id)
        labelled_queries.append((f"Which release uses code {answer}?", job.version_id))

    recovered = 0
    for question, expected_version_id in labelled_queries:
        result = indexing.search(
            VectorSearchRequest(
                collection_id=collection.id,
                question=question,
                top_k=3,
                expansion=ExpansionStrategy.NONE,
            )
        )
        if expected_version_id in {item.version_id for item in result.final_context}:
            recovered += 1

    recall_at_three = recovered / len(labelled_queries)
    assert recall_at_three == 1.0
