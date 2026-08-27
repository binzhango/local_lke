"""Batched persistent indexing and context-expanding vector retrieval."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from uuid import UUID

from langchain_core.documents import Document

from local_lke.errors import IndexingError
from local_lke.indexing.repository import (
    IndexableVersion,
    NodeHit,
    NodeWrite,
    SqlAlchemyIndexRepository,
)
from local_lke.models import (
    EmbeddingModality,
    EmbeddingProfileResponse,
    ExpansionStrategy,
    IndexingJobResponse,
    IndexStateResponse,
    JobStatus,
    NodeGranularity,
    VectorSearchCandidate,
    VectorSearchRequest,
    VectorSearchResponse,
)
from local_lke.providers import EmbeddingProvider
from local_lke.settings import Settings

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[\u3002\uff01\uff1f])")


@dataclass(frozen=True)
class PendingNode:
    id: str
    chunk_id: str
    parent_element_id: str | None
    granularity: NodeGranularity
    unit_ordinal: int
    text: str
    locator: str
    token_count: int


class IndexingService:
    def __init__(
        self,
        repository: SqlAlchemyIndexRepository,
        embeddings: EmbeddingProvider,
        settings: Settings,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self.settings = settings

    def check_health(self) -> str:
        return self.repository.vector_health(self.settings.embedding_dimension)

    def profile(self) -> EmbeddingProfileResponse:
        return self.repository.get_or_create_profile(
            modality=EmbeddingModality.TEXT,
            model_id=self.embeddings.model_id,
            revision=self.embeddings.revision,
            dimension=self.settings.embedding_dimension,
            normalized=self.embeddings.normalized,
            document_prefix=self.embeddings.document_prefix,
            query_prefix=self.embeddings.query_prefix,
        )

    def index_version(self, version_id: UUID, *, force: bool = False) -> IndexingJobResponse:
        version = self.repository.load_version(version_id)
        if not version.active or version.status != "complete":
            raise IndexingError(
                "Only a complete active document version can be indexed",
                code="version_not_indexable",
            )
        profile = self.profile()
        nodes = _build_nodes(version, profile.id)
        if not nodes:
            raise IndexingError("The version has no indexable nodes", code="empty_index")
        job = self.repository.get_or_create_job(version, profile.id)
        if force:
            self.repository.delete_version_nodes(version_id, profile.id)
        existing = self.repository.existing_node_ids(version_id, profile.id)
        missing = [item for item in nodes if item.id not in existing]
        if not missing and len(existing) == len(nodes):
            self.repository.activate_version(version, profile.id, len(nodes))
            self.repository.activate_profile(
                version.collection_id, profile.id, EmbeddingModality.TEXT
            )
            return self.repository.update_job(
                job.id,
                status=JobStatus.COMPLETED.value,
                progress=100,
                total_nodes=len(nodes),
                embedded_nodes=len(nodes),
                skipped=True,
                error_code=None,
                error_message=None,
            )
        calls = 0 if force else job.embedding_calls
        self.repository.update_job(
            job.id,
            status=JobStatus.RUNNING.value,
            progress=int(100 * len(existing) / len(nodes)),
            total_nodes=len(nodes),
            embedded_nodes=len(existing),
            skipped=False,
            error_code=None,
            error_message=None,
        )
        try:
            for start in range(0, len(missing), self.settings.embedding_batch_size):
                batch = missing[start : start + self.settings.embedding_batch_size]
                vectors = self.embeddings.embed_documents([item.text for item in batch])
                calls += 1
                if len(vectors) != len(batch):
                    raise IndexingError(
                        "The embedding provider returned an unexpected batch size",
                        code="embedding_batch_contract",
                    )
                writes: list[NodeWrite] = []
                for node, vector in zip(batch, vectors, strict=True):
                    if len(vector) != profile.dimension:
                        raise IndexingError(
                            f"Embedding dimension {len(vector)} does not match profile "
                            f"dimension {profile.dimension}",
                            code="embedding_dimension_mismatch",
                        )
                    writes.append(
                        NodeWrite(
                            id=node.id,
                            collection_id=version.collection_id,
                            document_id=version.document_id,
                            version_id=version.version_id,
                            chunk_id=node.chunk_id,
                            parent_element_id=node.parent_element_id,
                            profile_id=profile.id,
                            granularity=node.granularity,
                            unit_ordinal=node.unit_ordinal,
                            text=node.text,
                            locator=node.locator,
                            token_count=node.token_count,
                            embedding=vector,
                        )
                    )
                self.repository.write_nodes(writes)
                complete = len(existing) + min(start + len(batch), len(missing))
                self.repository.update_job(
                    job.id,
                    progress=int(95 * complete / len(nodes)),
                    embedded_nodes=complete,
                    embedding_calls=calls,
                )
            self.repository.activate_version(version, profile.id, len(nodes))
            self.repository.activate_profile(
                version.collection_id, profile.id, EmbeddingModality.TEXT
            )
            return self.repository.update_job(
                job.id,
                status=JobStatus.COMPLETED.value,
                progress=100,
                total_nodes=len(nodes),
                embedded_nodes=len(nodes),
                embedding_calls=calls,
                skipped=False,
            )
        except IndexingError as exc:
            return self.repository.update_job(
                job.id,
                status=JobStatus.FAILED.value,
                embedding_calls=calls,
                error_code=exc.code,
                error_message=str(exc),
            )
        except Exception as exc:
            return self.repository.update_job(
                job.id,
                status=JobStatus.FAILED.value,
                embedding_calls=calls,
                error_code="embedding_batch_failed",
                error_message=f"Indexing failed safely: {type(exc).__name__}",
            )

    def index_collection(self, collection_id: UUID) -> list[IndexingJobResponse]:
        return [
            self.index_version(version_id)
            for version_id in self.repository.active_version_ids(collection_id)
        ]

    def state(self, collection_id: UUID) -> IndexStateResponse:
        profile = self.repository.get_active_profile(collection_id, EmbeddingModality.TEXT)
        counts = self.repository.index_counts(
            collection_id, profile.id if profile is not None else None
        )
        return IndexStateResponse(
            collection_id=collection_id,
            active_profile=profile,
            active_nodes=counts[0],
            active_chunks=counts[1],
            missing_active_chunks=counts[2],
            jobs=self.repository.list_jobs(collection_id),
        )

    def search(self, request: VectorSearchRequest) -> VectorSearchResponse:
        profile = (
            self.repository.get_profile(request.profile_id)
            if request.profile_id is not None
            else self.repository.get_active_profile(request.collection_id, EmbeddingModality.TEXT)
        )
        if profile is None:
            raise IndexingError(
                "The collection has no active text embedding profile; index a version first.",
                code="index_not_ready",
            )
        query_vector = self.embeddings.embed_query(request.question)
        if len(query_vector) != profile.dimension:
            raise IndexingError(
                "The query embedding is incompatible with the selected profile",
                code="embedding_dimension_mismatch",
            )
        granularities = _search_granularities(request.expansion)
        hits = self.repository.search_nodes(
            collection_id=request.collection_id,
            profile_id=profile.id,
            query_vector=query_vector,
            granularities=granularities,
            limit=max(request.top_k * 4, request.top_k),
        )
        budget = request.token_budget or self.settings.retrieval_context_tokens
        candidates, final, token_count = self._expand_and_pack(hits, request, budget)
        return VectorSearchResponse(
            profile=profile,
            candidates=candidates,
            final_context=final,
            final_token_count=token_count,
        )

    def _expand_and_pack(
        self,
        hits: list[NodeHit],
        request: VectorSearchRequest,
        budget: int,
    ) -> tuple[list[VectorSearchCandidate], list[VectorSearchCandidate], int]:
        candidates: list[VectorSearchCandidate] = []
        final: list[VectorSearchCandidate] = []
        seen_context: set[str] = set()
        used = 0
        for rank, hit in enumerate(hits, start=1):
            context, locator, tokens, key = self._expanded_context(hit, request)
            decision = "candidate"
            included = False
            if key in seen_context:
                decision = "duplicate expanded parent/window; best child retained"
            elif len(final) >= request.top_k:
                decision = "excluded by top-k context cap"
            elif used >= budget:
                decision = "excluded by token budget"
            else:
                available = budget - used
                if tokens > available:
                    words = context.split()
                    context = " ".join(words[:available])
                    tokens = min(available, len(words))
                    decision = "included with deterministic token truncation"
                else:
                    decision = "included after expansion and deduplication"
                if tokens > 0:
                    included = True
                    seen_context.add(key)
                    used += tokens
            candidate = VectorSearchCandidate(
                node_id=hit.id,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                version_id=hit.version_id,
                granularity=hit.granularity,
                rank=rank,
                score=round(hit.score, 8),
                locator=locator,
                child_text=hit.text,
                context_text=context,
                trigger_node_id=hit.id,
                token_count=max(1, tokens),
                included=included,
                decision=decision,
            )
            candidates.append(candidate)
            if included:
                final.append(candidate)
        return candidates, final, used

    def _expanded_context(
        self, hit: NodeHit, request: VectorSearchRequest
    ) -> tuple[str, str, int, str]:
        if request.expansion is ExpansionStrategy.SENTENCE_WINDOW:
            text, tokens = self.repository.sentence_window(hit, request.sentence_window)
            return text, hit.locator, tokens, f"window:{hit.chunk_id}:{text}"
        if request.expansion is ExpansionStrategy.PARENT:
            parent = self.repository.parent_context(hit)
            if parent is not None:
                return parent[0], parent[1], parent[2], f"parent:{hit.parent_element_id}"
        if request.expansion is ExpansionStrategy.MULTI:
            if hit.granularity is NodeGranularity.SENTENCE:
                text, tokens = self.repository.sentence_window(hit, request.sentence_window)
                return text, hit.locator, tokens, f"window:{hit.chunk_id}:{text}"
            parent = self.repository.parent_context(hit)
            if parent is not None:
                return parent[0], parent[1], parent[2], f"parent:{hit.parent_element_id}"
        return hit.text, hit.locator, hit.token_count, f"node:{hit.id}"

    def as_retriever(
        self,
        collection_id: UUID,
        *,
        top_k: int = 5,
        expansion: ExpansionStrategy = ExpansionStrategy.NONE,
    ) -> PersistentDenseRetriever:
        return PersistentDenseRetriever(self, collection_id, top_k, expansion)


class PersistentDenseRetriever:
    """Minimal LangChain-compatible retriever with ``invoke`` semantics."""

    def __init__(
        self,
        service: IndexingService,
        collection_id: UUID,
        top_k: int,
        expansion: ExpansionStrategy,
    ) -> None:
        self.service = service
        self.collection_id = collection_id
        self.top_k = top_k
        self.expansion = expansion

    def invoke(self, query: str) -> list[Document]:
        response = self.service.search(
            VectorSearchRequest(
                collection_id=self.collection_id,
                question=query,
                top_k=self.top_k,
                expansion=self.expansion,
            )
        )
        return [
            Document(
                page_content=item.context_text,
                metadata={
                    "chunk_id": item.chunk_id,
                    "version_id": str(item.version_id),
                    "locator": item.locator,
                    "score": item.score,
                    "trigger_node_id": item.trigger_node_id,
                },
            )
            for item in response.final_context
        ]

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


def _build_nodes(version: IndexableVersion, profile_id: UUID) -> list[PendingNode]:
    nodes: list[PendingNode] = []
    first_for_parent: dict[str, str] = {}
    for chunk in version.chunks:
        nodes.append(
            _pending(
                profile_id,
                chunk.id,
                chunk.parent_element_id,
                NodeGranularity.CHUNK,
                0,
                chunk.text,
                chunk.locator,
                chunk.token_count,
            )
        )
        sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(chunk.text) if part.strip()]
        for ordinal, sentence in enumerate(sentences):
            nodes.append(
                _pending(
                    profile_id,
                    chunk.id,
                    chunk.parent_element_id,
                    NodeGranularity.SENTENCE,
                    ordinal,
                    sentence,
                    chunk.locator,
                    max(1, len(sentence.split())),
                )
            )
        if chunk.parent_element_id is not None:
            first_for_parent.setdefault(chunk.parent_element_id, chunk.id)
    for parent_id, chunk_id in first_for_parent.items():
        parent = version.parent_text.get(parent_id)
        if parent is None:
            continue
        nodes.append(
            _pending(
                profile_id,
                chunk_id,
                parent_id,
                NodeGranularity.SECTION,
                0,
                parent[0],
                parent[1],
                max(1, len(parent[0].split())),
            )
        )
    return nodes


def _pending(
    profile_id: UUID,
    chunk_id: str,
    parent_element_id: str | None,
    granularity: NodeGranularity,
    ordinal: int,
    text: str,
    locator: str,
    token_count: int,
) -> PendingNode:
    identity = f"{profile_id}:{chunk_id}:{granularity.value}:{ordinal}:{text}"
    return PendingNode(
        id=hashlib.sha256(identity.encode()).hexdigest(),
        chunk_id=chunk_id,
        parent_element_id=parent_element_id,
        granularity=granularity,
        unit_ordinal=ordinal,
        text=text,
        locator=locator,
        token_count=max(1, token_count),
    )


def _search_granularities(
    expansion: ExpansionStrategy,
) -> tuple[NodeGranularity, ...]:
    if expansion is ExpansionStrategy.SENTENCE_WINDOW:
        return (NodeGranularity.SENTENCE,)
    if expansion is ExpansionStrategy.PARENT:
        return (NodeGranularity.SENTENCE, NodeGranularity.CHUNK)
    if expansion is ExpansionStrategy.MULTI:
        return tuple(NodeGranularity)
    return (NodeGranularity.CHUNK,)
