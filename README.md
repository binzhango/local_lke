# Local LKE

Local LKE is an English, executable companion for learning Retrieval-Augmented
Generation (RAG). Chapter 2 adds safe, versioned ingestion for Markdown, text,
and PDF files to the Chapter 1 in-memory RAG baseline. One FastAPI and Gradio
application exposes collections, ingestion jobs, parser previews, chunks,
version history, and cited chat.

The deterministic suite creates isolated SQLite stores and a temporary local
PostgreSQL 18 cluster; it needs no model server or network access. Interactive
answers use a model served locally by LM Studio, `llama-server`, or another
OpenAI-compatible endpoint.

## Prerequisites

- macOS or Linux
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management
- PostgreSQL 18 (`brew install postgresql@18` on macOS)
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
make init       # bootstrap a clean checkout
make init-postgres # create and migrate the local PostgreSQL 18 database
make migrate    # apply pending Alembic migrations
make serve      # FastAPI and Gradio in one process
make doctor     # models endpoint, chat completion, and embedding checks
make test       # deterministic tests; network sockets are disabled
make test-live  # optional provider smoke test (see docs/quick-start.md)
make lint       # Ruff
make typecheck  # strict mypy
make check      # lint, typecheck, and deterministic tests
uv run lke openapi  # export .artifacts/openapi.json
```

## Chapter 2 capabilities and boundaries

- Named collections and `.md`, `.txt`, and `.pdf` uploads
- Immutable versions with SHA-256 content and pipeline hashes
- Markdown hierarchy, text source lines, and PDF page/element preservation
- Recursive, Markdown-aware, and experimental sentence-semantic chunking
- Inspectable persistent jobs, parser previews, versions, and chunks
- Safe local single-user upload boundary with size, MIME, filename, and PDF checks
- Chapter 1 chat still uses its in-memory fixture index; persisted chunk indexing
  begins in Chapter 3
- Authentication and multi-user ACLs remain out of scope

See the [Chapter 1 learning notes](docs/chapter-01-knowledge-guide.md),
[detailed implementation report](docs/chapter-01-implementation.md),
[quick start](docs/quick-start.md), [four-stage code map](docs/architecture.md),
and [initial RAG issue baseline](docs/blog-coverage.md).
For this milestone, see the [Chapter 2 learning notes](docs/chapter-02-knowledge-guide.md),
[implementation report](docs/chapter-02-implementation.md), and
[ingestion operations guide](docs/chapter-02-ingestion.md).

## API

- `GET /healthz` — component-level chat and embedding health
- `POST/GET /api/v1/collections` — create and list collections
- `POST /api/v1/collections/{id}/documents` — safely ingest one or more files
- `GET /api/v1/jobs/{id}` and `POST /api/v1/jobs/{id}/retry` — inspect/retry jobs
- `GET /api/v1/collections/{id}/documents` — list documents and version history
- `GET /api/v1/document-versions/{id}/preview` — inspect elements and chunks
- `DELETE /api/v1/documents/{id}` — soft-delete and deactivate versions
- `POST /api/v1/query` — synchronous cited answer
- `POST /api/v1/query/stream` — ordered `start`, `retrieval`, `delta`,
  `completion`, and `error` SSE events
- `GET /api/v1/sources/{source_id}` — citation target for bundled sources

## Future LKE extension

- [ ] **LKE — Large Language Model Knowledge Management Expert System**
  - Ingest and version private organizational documents and structured data.
  - Route questions across lexical, vector, structured, and graph retrieval.
  - Produce grounded answers with citations, abstention, traces, and evaluations.
  - Remain local-first with PostgreSQL 18, pgvector, Apache AGE, and local models.

This extension begins after the chapter-by-chapter RAG foundation is complete.
