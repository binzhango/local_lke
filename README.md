# Local LKE

Local LKE is an English, executable companion for learning Retrieval-Augmented
Generation (RAG). Chapters 1-7 now provide a cited baseline, versioned ingestion,
persistent pgvector and multimodal indexing, and hybrid, metadata-aware,
corrective retrieval, validated generation, regression evaluation, and an opt-in
governed API boundary. One FastAPI and Gradio application exposes collections,
ingestion and indexing jobs, retrieval-stage traces, context manifests, image
search, CSV schemas, safe SQL previews, cited answers, and persisted evaluation runs.

The deterministic suite creates isolated SQLite stores and a temporary local
PostgreSQL 18 cluster; it needs no model server or network access. Interactive
answers use a model served locally by LM Studio, `llama-server`, or another
OpenAI-compatible endpoint.

## Prerequisites

- macOS or Linux
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management
- PostgreSQL 18 and pgvector (`brew install postgresql@18 pgvector` on macOS)
- Optional for interactive answers: LM Studio or `llama-server`

The application uses the explicit Homebrew PostgreSQL 18 binary directory at
`/opt/homebrew/opt/postgresql@18/bin` by default. No older PostgreSQL
installation is required.

## One-command foundation setup

```bash
./scripts/init_environment.sh
```

The script installs Python 3.12 through `uv`, installs exactly the locked
dependencies, creates `.env` from the safe example when needed, and runs an
offline end-to-end RAG check. It does not overwrite an existing `.env`.

Start and initialize PostgreSQL once:

```bash
brew services start postgresql@18
make init-postgres
```

`make init-postgres` checks the explicit PostgreSQL 18 binaries, creates the
passwordless local `local_lke` database only when absent, and applies Alembic
migrations. The committed database URL contains no password.

Then configure and run the local model:

1. In LM Studio, download/load an instruction model and start **Developer →
   Local Server**.
2. Copy the model ID shown by `http://127.0.0.1:1234/v1/models` into
   `LKE_CHAT_MODEL` in `.env`.
3. Run the provider checks and application:

```bash
make doctor
make serve
```

Open [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app). API documentation
is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Try this bundled question:

> How quickly does Atlas acknowledge a priority-one incident?

The answer should say **15 minutes** and cite `fixture:atlas-support`.

## Commands

```bash
make demo-setup # one-time Python, dependency, and database setup for all chapters
make demo       # one cumulative server; browse chapter-tagged endpoints in /docs
make demo-secure # restart the cumulative server with disposable Chapter 7 tokens
make init       # lower-level environment bootstrap
make init-postgres # lower-level database initialization
make migrate    # apply pending Alembic migrations
make serve      # FastAPI and Gradio in one process
make demo-chapter CHAPTER=4 # optional focus on the same cumulative app
make doctor     # models endpoint, chat completion, and embedding checks
make test       # deterministic tests; network sockets are disabled
make test-live  # optional provider smoke test (see docs/quick-start.md)
make lint       # Ruff
make typecheck  # strict mypy
make check      # lint, typecheck, and deterministic tests
uv run lke openapi  # export .artifacts/openapi.json
```

The shortest demo flow is one setup and one server:

```bash
make demo-setup  # once
make demo        # every normal demo
```

Open `/docs` and expand the Chapter 1–7 tags; no chapter checkout, dependency
installation, or service restart is needed. `make demo-secure` is the separate
Chapter 7 authorization demonstration because secure mode intentionally disables
direct-service Gradio callbacks. The older `make demo CHAPTER=4` form remains an
optional focused guide. See [Chapter demo launcher](docs/chapter-demos.md).

## Chapters 2-7 capabilities and boundaries

- Named collections and `.md`, `.txt`, and `.pdf` uploads
- Immutable versions with SHA-256 content and pipeline hashes
- Markdown hierarchy, text source lines, and PDF page/element preservation
- Recursive, Markdown-aware, and experimental sentence-semantic chunking
- Inspectable persistent jobs, parser previews, versions, and chunks
- Safe local single-user upload boundary with size, MIME, filename, and PDF checks
- Versioned local BGE embedding profiles, bounded resumable batches, and
  transactional index activation
- Persistent pgvector text nodes with HNSW cosine search and visible index health
- Sentence-window, parent-child, and multi-granularity context expansion under a
  token budget
- Optional local CLIP text-to-image and image-to-image search over validated
  PNG, JPEG, and WebP assets
- Active-version PostgreSQL English full-text search with a generated `tsvector`
  and GIN index
- Persistent-first local dense search with a compatibility fallback, RRF hybrid
  fusion, and optional local cross-encoder reranking
- Allowlisted metadata filters, bounded decomposition, step-back/HyDE probes,
  context manifests, one corrective retry, and evidence-based abstention
- Validated CSV schema inference and SQLAlchemy-compiled read-only structured queries
- The no-collection API path still preserves the Chapter 1 fixture baseline
- Conversational, allowlisted structured JSON, and evidence-only generation modes
- Stable citation IDs resolved only from the retrieved active-evidence registry
- Pydantic validation, one bounded schema repair, and validation-safe traces
- Extractive cited degradation for model/format failure and generation-free abstention
- Versioned trust-boundary prompts and sanitized Gradio answer/source rendering
- Immutable labelled evaluation datasets with content hashes and dataset versions
- Persisted per-case runs with Recall@k, MRR, nDCG, answer, citation, status, and latency metrics
- Absolute and baseline-relative regression gates over identical dataset versions
- Run-local chat outage, empty-output, and malformed-output fault injection
- Declarative provider capability profiles separated from live health checks
- Optional constant-time bearer authentication with redacted configuration
- Per-collection owner, editor, and viewer authorization on every API resource path
- Metadata-only allow/deny audit evidence and administrator-only evaluation controls
- Secure API mode intentionally disables direct-service Gradio callbacks

See the [Chapter 1 learning notes](docs/chapter-01-knowledge-guide.md),
[detailed implementation report](docs/chapter-01-implementation.md),
[quick start](docs/quick-start.md), [four-stage code map](docs/architecture.md),
and [initial RAG issue baseline](docs/blog-coverage.md).
For this milestone, see the [Chapter 2 learning notes](docs/chapter-02-knowledge-guide.md),
[implementation report](docs/chapter-02-implementation.md), and
[ingestion operations guide](docs/chapter-02-ingestion.md).
Chapter 3 adds the [indexing and embeddings learning guide](docs/chapter-03-knowledge-guide.md),
[implementation report](docs/chapter-03-implementation.md), and
[indexing operations guide](docs/chapter-03-indexing.md).
Chapter 4 adds the [advanced retrieval learning guide](docs/chapter-04-knowledge-guide.md),
[implementation report](docs/chapter-04-implementation.md), and
[retrieval operations guide](docs/chapter-04-retrieval.md).
Chapter 5 adds the [validated generation learning guide](docs/chapter-05-knowledge-guide.md),
[implementation report](docs/chapter-05-implementation.md), and
[generation operations guide](docs/chapter-05-generation.md).
Chapter 6 adds the [evaluation learning guide](docs/chapter-06-knowledge-guide.md),
[implementation report](docs/chapter-06-implementation.md), and
[evaluation operations guide](docs/chapter-06-evaluation.md).
Chapter 7 adds the [security and governance learning guide](docs/chapter-07-knowledge-guide.md),
[implementation report](docs/chapter-07-implementation.md), and
[security operations guide](docs/chapter-07-security.md).

## API

- `GET /healthz` — component-level chat, embedding, database, and vector health
- `POST/GET /api/v1/collections` — create and list collections
- `POST /api/v1/collections/{id}/documents` — safely ingest one or more files
- `GET /api/v1/jobs/{id}` and `POST /api/v1/jobs/{id}/retry` — inspect/retry jobs
- `GET /api/v1/collections/{id}/documents` — list documents and version history
- `GET /api/v1/document-versions/{id}/preview` — inspect elements and chunks
- `DELETE /api/v1/documents/{id}` — soft-delete and deactivate versions
- `POST /api/v1/document-versions/{id}/index` — build or resume one text index
- `POST /api/v1/collections/{id}/index` and `GET .../index-state` — build/inspect
  the active collection index
- `POST /api/v1/retrieval-lab` — dense candidates plus bounded context expansion
- `POST /api/v1/collections/{id}/images` and `/images/search/{text,image}` —
  validate, index, and search local image assets
- `POST /api/v1/query` — fixture baseline or collection-scoped dense/hybrid answer
- `POST /api/v1/query/stream` — ordered `start`, `retrieval`, `delta`,
  `completion`, and `error` SSE events
- `GET /api/v1/sources/{source_id}` — citation target for bundled sources
- `POST/GET /api/v1/collections/{id}/structured-tables` — ingest/list safe CSV tables
- `POST /api/v1/structured/query` — execute a validated compiled structured plan
- `POST/GET /api/v1/evaluations/datasets` — create/list immutable labelled datasets
- `POST/GET /api/v1/evaluations/runs` — execute/list persisted evaluation runs
- `GET /api/v1/evaluations/compare` — compare same-dataset baseline and candidate runs
- `GET /api/v1/evaluations/provider-profile` — inspect configured generation capabilities
- `GET/PUT /api/v1/collections/{id}/access` — inspect or grant collection access
- `DELETE /api/v1/collections/{id}/access/{principal}` — revoke non-owner access
- `GET /api/v1/audit-events` — inspect metadata-only security decisions as an admin

`POST /api/v1/query` and `/query/stream` accept `output_mode` values
`conversational`, `structured`, or `evidence_only`. Structured mode additionally
requires the allowlisted `schema_name` `fact_list` or `comparison`.

## Future LKE extension

- [ ] **LKE — Large Language Model Knowledge Management Expert System**
  - Ingest and version private organizational documents and structured data.
  - Route questions across lexical, vector, structured, and graph retrieval.
  - Produce grounded answers with citations, abstention, traces, and evaluations.
  - Remain local-first with PostgreSQL 18, pgvector, Apache AGE, and local models.

This extension begins after the chapter-by-chapter RAG foundation is complete.
