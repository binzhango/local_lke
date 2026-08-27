# Chapters 1–2 Architecture

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
pipeline remains the measured in-memory fixture baseline until Chapter 3 embeds
persisted chunks into PostgreSQL/pgvector.

## Trace timings

Every answer records milliseconds for `load`, `split`, `embed`, `retrieve`, and
`generate`. The workbench Trace tab also shows retrieval rank, similarity score,
source metadata, and the exact chunk text used as evidence.
