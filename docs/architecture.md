# Chapter 1 Architecture

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

The implementation is intentionally single-process, single-collection, and
in-memory. Restarting the process rebuilds the fixture index. Persistent
PostgreSQL 18 and pgvector indexing are later milestones, not hidden Chapter 1
dependencies.

## Trace timings

Every answer records milliseconds for `load`, `split`, `embed`, `retrieve`, and
`generate`. The workbench Trace tab also shows retrieval rank, similarity score,
source metadata, and the exact chunk text used as evidence.

