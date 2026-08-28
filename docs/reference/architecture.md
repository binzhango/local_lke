# Chapters 1-7 Architecture

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

Chapter 3 makes Chapter 2's active versions persistently searchable:

| Indexing stage | Responsibility | Implementation |
|---|---|---|
| Profile | Version model, revision, dimension, normalization, prefixes, modality | `providers/embeddings.py`, `indexing/repository.py` |
| Build | Derive sentence/chunk/section nodes and embed in resumable batches | `indexing/service.py` |
| Activate | Keep partial builds hidden; atomically replace stale active nodes | `indexing/repository.py`, `storage/models.py` |
| Search/expand | HNSW cosine candidates, windows, parents, multi-granularity, budget | `indexing/service.py` |
| Multimodal | Validate images and run optional local joint-space retrieval | `indexing/images.py`, `providers/multimodal.py` |

Chapter 4 consumes that stable seam and adds bounded advanced retrieval:

| Retrieval stage | Responsibility | Implementation |
|---|---|---|
| Plan | Normalize, route, decompose, and validate metadata | `retrieval/planning.py`, `models.py` |
| Recall | Persistent-first dense cosine plus lifecycle-scoped PostgreSQL FTS/BM25 | `indexing/`, `retrieval/service.py`, `storage/repository.py`, `migrations/` |
| Fusion/precision | RRF identity fusion and optional local cross-encoder | `retrieval/service.py`, `retrieval/reranking.py` |
| Context | Coverage-first dedupe/diversity/token packing with manifest | `retrieval/service.py` |
| Correction | Deterministic sufficiency, one alternate retrieval, abstention | `retrieval/service.py` |
| Structured | CSV typing/provenance and Pydantic-to-SQLAlchemy compilation | `retrieval/structured.py` |

The old on-demand dense path remains a compatibility fallback when a collection
has no complete compatible index. It is not used to disguise a failed or
dimension-incompatible active profile: index health and job errors remain
visible.

## Trace timings

Fixture answers record `load`, `split`, `embed`, `retrieve`, and `generate`.
Persisted answers additionally expose query transformation, metadata plan,
dense/lexical/fused/reranked positions, context inclusion/exclusion/truncation,
reranker latency/gain, sufficiency features, and corrective strategy.

Chapter 5 turns generation from unvalidated text into a stable application boundary:

| Generation stage | Responsibility | Implementation |
|---|---|---|
| Evidence registry | Assign `C1..Cn` only to active retrieved evidence and typed locators | `generation/service.py`, `generation/locators.py` |
| Prompt boundary | Separate policy, contract, answerability, manifest, encoded untrusted evidence, and question | `generation/prompting.py` |
| Output contract | Validate conversational claims or allowlisted `fact_list`/`comparison` JSON | `models.py`, `generation/service.py` |
| Repair | Return sanitized validation categories to one bounded retry | `generation/service.py` |
| Degradation | Produce cited extracts on provider/schema failure; preserve no-evidence abstention | `generation/service.py` |
| Presentation | Serialize one API/SSE contract and escape generated/source Markdown in Gradio | `web/api.py`, `web/workbench.py` |

Model citation IDs are references into the application registry, not source
identifiers accepted from the model. Only the application resolves a validated
ID to its retrieved source version, chunk, locator, and excerpt.

Chapter 6 adds a control plane around the unchanged query boundary:

| Evaluation stage | Responsibility | Implementation |
|---|---|---|
| Dataset | Canonicalize, hash, version, and persist immutable labelled cases | `evaluation/models.py`, `evaluation/service.py`, `storage/models.py` |
| Execute | Run fixture or collection queries and isolate controlled provider faults | `evaluation/service.py` |
| Measure | Score retrieval, answer phrases, citations, statuses, and latency | `evaluation/service.py` |
| Gate | Apply absolute thresholds and same-dataset baseline deltas | `evaluation/service.py` |
| Deliver | Expose typed API and an Evaluation workbench tab | `web/api.py`, `web/workbench.py` |

Evaluation consumes the public `AnswerResponse`; it does not reach into raw model
output or replace application citation resolution. A completed run and a passed
gate remain distinct states.

Chapter 7 places a governance boundary around every `/api/v1` operation:

| Security stage | Responsibility | Implementation |
|---|---|---|
| Authenticate | Validate configured bearer tokens with constant-time digest comparison | `security/service.py`, `settings.py` |
| Authorize | Resolve nested resources to collections and enforce viewer/editor/owner permissions | `security/service.py`, `web/api.py` |
| Govern | Reserve cross-collection evaluation and audit inspection for administrators | `web/api.py` |
| Audit | Persist metadata-only allow/deny decisions without tokens, prompts, or evidence | `security/service.py`, `storage/models.py` |
| Deliver safely | Publish bearer security in OpenAPI and omit direct-service Gradio in secure mode | `web/app.py`, `web/api.py` |

Authentication is disabled for the original loopback-only learning workflow. Once
enabled, collection creation and owner assignment share one database transaction;
all document, version, job, index, image, structured-table, citation, and query
paths resolve back to that collection before invoking a service.
