"""Inspectable four-stage in-memory RAG pipeline."""

from collections.abc import Iterator
from time import perf_counter
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from local_lke.generation import GenerationEvidence, GenerationRequest, GenerationService
from local_lke.generation.locators import citation_locator
from local_lke.models import (
    AnswerResponse,
    Citation,
    OutputMode,
    QueryRoute,
    RetrievedChunk,
    SourceDocument,
    StructuredSchemaName,
)
from local_lke.providers import ChatProvider, EmbeddingProvider
from local_lke.rag.documents import (
    chunk_to_langchain_document,
    langchain_document_to_chunk,
    load_fixture_documents,
)
from local_lke.rag.splitting import split_documents


class RAGPipeline:
    """One shared pipeline for the API, CLI health checks, and Gradio."""

    def __init__(
        self,
        *,
        chat: ChatProvider,
        embeddings: EmbeddingProvider,
        default_top_k: int = 3,
        documents: list[SourceDocument] | None = None,
        generation_max_repair_attempts: int = 1,
        generation_native_structured_output: bool = False,
    ) -> None:
        self.chat = chat
        self.embeddings = embeddings
        self.default_top_k = default_top_k
        self._provided_documents = documents
        self._documents: list[SourceDocument] = []
        self._vector_store: InMemoryVectorStore | None = None
        self._preparation_timings: dict[str, float] = {}
        self.generation = GenerationService(
            chat,
            max_repair_attempts=generation_max_repair_attempts,
            prefer_native_structured_output=generation_native_structured_output,
        )

    @property
    def documents(self) -> list[SourceDocument]:
        if not self._documents:
            self._documents = self._provided_documents or load_fixture_documents()
        return list(self._documents)

    def prepare(self) -> None:
        if self._vector_store is not None:
            return

        started = perf_counter()
        if not self._documents:
            self._documents = self._provided_documents or load_fixture_documents()
        loaded = perf_counter()
        chunks = split_documents(self._documents)
        split = perf_counter()

        vector_store = InMemoryVectorStore(embedding=cast(Embeddings, self.embeddings))
        vector_store.add_documents([chunk_to_langchain_document(chunk) for chunk in chunks])
        embedded = perf_counter()
        self._vector_store = vector_store
        self._preparation_timings = {
            "load": _elapsed_ms(started, loaded),
            "split": _elapsed_ms(loaded, split),
            "embed": _elapsed_ms(split, embedded),
        }

    def retrieve(
        self, question: str, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], float]:
        self.prepare()
        assert self._vector_store is not None
        started = perf_counter()
        matches = self._vector_store.similarity_search_with_score(
            question, k=top_k or self.default_top_k
        )
        elapsed = _elapsed_ms(started, perf_counter())
        retrieved = [
            RetrievedChunk(
                chunk=langchain_document_to_chunk(document),
                rank=rank,
                score=float(score),
            )
            for rank, (document, score) in enumerate(matches, start=1)
        ]
        return retrieved, elapsed

    def query(
        self,
        question: str,
        top_k: int | None = None,
        *,
        output_mode: OutputMode = OutputMode.CONVERSATIONAL,
        schema_name: StructuredSchemaName | None = None,
    ) -> AnswerResponse:
        retrieved, retrieval_ms = self.retrieve(question, top_k)
        started = perf_counter()
        response = self.generation.generate(
            self._generation_request(
                question,
                retrieved,
                retrieval_ms,
                output_mode=output_mode,
                schema_name=schema_name,
            )
        )
        response.trace.timings_ms["generate"] = _elapsed_ms(started, perf_counter())
        response.trace.retrieved = retrieved
        return response

    def stream_query(
        self,
        question: str,
        top_k: int | None = None,
        *,
        output_mode: OutputMode = OutputMode.CONVERSATIONAL,
        schema_name: StructuredSchemaName | None = None,
    ) -> Iterator[tuple[str, Any]]:
        retrieved, retrieval_ms = self.retrieve(question, top_k)
        yield "retrieval", {
            "chunks": [item.model_dump(mode="json") for item in retrieved],
            "timing_ms": retrieval_ms,
        }
        started = perf_counter()
        request = self._generation_request(
            question,
            retrieved,
            retrieval_ms,
            output_mode=output_mode,
            schema_name=schema_name,
        )
        response = (
            self.generation.generate_streaming(request)
            if retrieved
            else self.generation.generate(request)
        )
        response.trace.timings_ms["generate"] = _elapsed_ms(started, perf_counter())
        response.trace.retrieved = retrieved
        if response.status.value in {"answered", "degraded"}:
            for offset in range(0, len(response.answer), 48):
                yield "delta", response.answer[offset : offset + 48]
        yield "completion", response

    def _generation_request(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        retrieval_ms: float,
        *,
        output_mode: OutputMode,
        schema_name: StructuredSchemaName | None,
    ) -> GenerationRequest:
        timings = dict(self._preparation_timings)
        timings.update({"retrieve": retrieval_ms, "generate": 0.0})
        documents = {item.source_id: item for item in self.documents}
        evidence = []
        for index, item in enumerate(retrieved, start=1):
            source = documents.get(item.chunk.source_id)
            citation = Citation(
                citation_id=f"C{index}",
                source_id=item.chunk.source_id,
                chunk_id=item.chunk.chunk_id,
                locator=item.chunk.locator,
                excerpt=_excerpt(item.chunk.text),
                title=source.title if source else None,
                locator_detail=citation_locator(
                    item.chunk.locator,
                    media_type=source.media_type if source else "text/plain",
                ),
            )
            evidence.append(GenerationEvidence(citation=citation, text=item.chunk.text))
        return GenerationRequest(
            question=question,
            evidence=evidence,
            output_mode=output_mode,
            schema_name=schema_name,
            route=QueryRoute.SIMPLE_LOOKUP,
            answerability=(
                "evidence passed the retrieval gate"
                if retrieved
                else "no supporting evidence was retrieved"
            ),
            sufficient=bool(retrieved),
            timings_ms=timings,
        )


def _elapsed_ms(started: float, finished: float) -> float:
    return round((finished - started) * 1000, 3)


def _excerpt(text: str, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"
