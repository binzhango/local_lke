# Chapters 1, 2, and 4 Architecture

Chapter 1 is a measured naive-RAG baseline. It separates the four stages so later
chapters can improve one stage without changing the public answer contract.

| RAG stage | Responsibility | Implementation |
|---|---|---|
| Data preparation | Load normalized English fixtures and split stable chunks | `rag/documents.py`, `rag/splitting.py` |
| Indexing | Embed chunks locally and put them in memory | `providers/embeddings.py`, `rag/pipeline.py` |
| Retrieval | Similarity-search top-k chunks and preserve rank/score | `rag/pipeline.py` |
| Generation | Delimit untrusted evidence, call local chat, attach citations | `rag/prompting.py`, `providers/chat.py`, `rag/pipeline.py` |

FastAPI and the mounted Gradio `Blocks` workbench receive the same `RAGPipeline`
instance. LangChain `Document` objects exist only at the vector-store boundary;
Pydantic models define the stable public API.

Chapter 2 adds a separate ingestion path without silently changing the Chapter 1
query baseline:

| Ingestion stage | Responsibility | Implementation |
|---|---|---|
| Upload boundary | Normalize filenames; validate size, extension, MIME, encoding/signature, encryption, and PDF integrity; store below generated IDs | `ingestion/safety.py` |
| Parsing | Normalize Markdown headings/source lines, text paragraphs, and Unstructured PDF elements/pages | `ingestion/parsers.py` |
| Chunking | Produce stable, parent-linked chunks using recursive, Markdown-aware, or experimental sentence-semantic strategies | `ingestion/chunking.py` |
| Versioning | Hash content and pipeline configuration; skip identical work; atomically activate one immutable version | `ingestion/service.py`, `storage/repository.py` |
| Persistence | Store collections, documents, versions, elements, chunks, pipeline configs, and recoverable jobs | `storage/models.py`, `migrations/` |

FastAPI and Gradio share both the `RAGPipeline` and `IngestionService`. The query
pipeline remains available as the measured in-memory fixture baseline.

Chapter 4 intentionally skips the Chapter 3 milestone and adds a bounded bridge
from Chapter 2 chunks:

| Retrieval stage | Responsibility | Implementation |
|---|---|---|
| Plan | Normalize, route, decompose, and validate metadata | `retrieval/planning.py`, `models.py` |
| Recall | On-demand dense cosine plus lifecycle-scoped PostgreSQL FTS/BM25 | `retrieval/service.py`, `storage/repository.py`, `migrations/` |
| Fusion/precision | RRF identity fusion and optional local cross-encoder | `retrieval/service.py`, `retrieval/reranking.py` |
| Context | Coverage-first dedupe/diversity/token packing with manifest | `retrieval/service.py` |
| Correction | Deterministic sufficiency, one alternate retrieval, abstention | `retrieval/service.py` |
| Structured | CSV typing/provenance and Pydantic-to-SQLAlchemy compilation | `retrieval/structured.py` |

The absence of Chapter 3 is explicit: no persistent dense vectors, ANN index,
multimodal store, or embedding-profile activation exists. This keeps the Chapter
4 feature set honest while preserving a clear seam for later indexing work.

## Trace timings

Fixture answers record `load`, `split`, `embed`, `retrieve`, and `generate`.
Persisted answers additionally expose query transformation, metadata plan,
dense/lexical/fused/reranked positions, context inclusion/exclusion/truncation,
reranker latency/gain, sufficiency features, and corrective strategy.
