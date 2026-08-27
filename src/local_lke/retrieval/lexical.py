"""LangChain-compatible adapter for collection-scoped lexical retrieval."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

from local_lke.models import MetadataFilterPlan


class ScopedLexicalRetriever(BaseRetriever):
    """Expose PostgreSQL/BM25 results through LangChain's retriever contract."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repository: Any
    collection_id: UUID
    filters: MetadataFilterPlan = Field(default_factory=MetadataFilterPlan)
    limit: int = 20

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        hits = self.repository.lexical_search(
            self.collection_id,
            query,
            self.filters,
            self.limit,
        )
        return [
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "document_id": str(chunk.document_id),
                    "version_id": str(chunk.version_id),
                    "filename": chunk.filename,
                    "locator": chunk.locator,
                    "lexical_rank": rank,
                    "lexical_score": score,
                    "matched_terms": matched_terms,
                },
            )
            for rank, (chunk, score, matched_terms) in enumerate(hits, start=1)
        ]
