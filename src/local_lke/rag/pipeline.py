"""Inspectable four-stage in-memory RAG pipeline."""

from collections.abc import Iterator
from time import perf_counter
from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from local_lke.models import (
    AnswerResponse,
    AnswerStatus,
    Citation,
    RetrievedChunk,
    SourceDocument,
    TraceSummary,
)
from local_lke.providers import ChatProvider, EmbeddingProvider
from local_lke.rag.documents import (
    chunk_to_langchain_document,
    langchain_document_to_chunk,
    load_fixture_documents,
)
from local_lke.rag.prompting import build_grounded_prompt
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
    ) -> None:
        self.chat = chat
        self.embeddings = embeddings
        self.default_top_k = default_top_k
        self._provided_documents = documents
        self._documents: list[SourceDocument] = []
        self._vector_store: InMemoryVectorStore | None = None
        self._preparation_timings: dict[str, float] = {}

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

    def query(self, question: str, top_k: int | None = None) -> AnswerResponse:
        retrieved, retrieval_ms = self.retrieve(question, top_k)
        if not retrieved:
            return self._answer_response(
                answer="I do not know because no supporting evidence was retrieved.",
                status=AnswerStatus.ABSTAINED,
                retrieved=[],
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )
        prompt = build_grounded_prompt(question, retrieved)
        started = perf_counter()
        answer = self.chat.generate(prompt).strip()
        generation_ms = _elapsed_ms(started, perf_counter())
        status = AnswerStatus.ANSWERED if answer else AnswerStatus.DEGRADED
        return self._answer_response(
            answer=answer or "The model returned an empty answer.",
            status=status,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    def stream_query(
        self, question: str, top_k: int | None = None
    ) -> Iterator[tuple[str, Any]]:
        retrieved, retrieval_ms = self.retrieve(question, top_k)
        yield "retrieval", {
            "chunks": [item.model_dump(mode="json") for item in retrieved],
            "timing_ms": retrieval_ms,
        }
        if not retrieved:
            response = self._answer_response(
                answer="I do not know because no supporting evidence was retrieved.",
                status=AnswerStatus.ABSTAINED,
                retrieved=[],
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
            )
            yield "completion", response
            return

        prompt = build_grounded_prompt(question, retrieved)
        started = perf_counter()
        tokens: list[str] = []
        for delta in self.chat.stream(prompt):
            tokens.append(delta)
            yield "delta", delta
        response = self._answer_response(
            answer="".join(tokens).strip() or "The model returned an empty answer.",
            status=AnswerStatus.ANSWERED if tokens else AnswerStatus.DEGRADED,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            generation_ms=_elapsed_ms(started, perf_counter()),
        )
        yield "completion", response

    def _answer_response(
        self,
        *,
        answer: str,
        status: AnswerStatus,
        retrieved: list[RetrievedChunk],
        retrieval_ms: float,
        generation_ms: float,
    ) -> AnswerResponse:
        timings = dict(self._preparation_timings)
        timings.update({"retrieve": retrieval_ms, "generate": generation_ms})
        citations = [
            Citation(
                source_id=item.chunk.source_id,
                chunk_id=item.chunk.chunk_id,
                locator=item.chunk.locator,
                excerpt=_excerpt(item.chunk.text),
            )
            for item in retrieved
        ]
        return AnswerResponse(
            status=status,
            answer=answer,
            citations=citations,
            trace=TraceSummary(timings_ms=timings, retrieved=retrieved),
        )


def _elapsed_ms(started: float, finished: float) -> float:
    return round((finished - started) * 1000, 3)


def _excerpt(text: str, limit: int = 280) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"
