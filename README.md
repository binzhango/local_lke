# Local LKE

Local LKE is an English, executable companion for learning Retrieval-Augmented
Generation (RAG). Chapter 1 is a deliberately small, in-memory baseline that
makes the four stages—load, index, retrieve, and generate—visible through one
FastAPI and Gradio application.

The bundled fixtures and deterministic tests need no database, model server, or
network access. Interactive answers use a model served locally by LM Studio,
`llama-server`, or another OpenAI-compatible endpoint.

## Prerequisites

- macOS or Linux
- [`uv`](https://docs.astral.sh/uv/) for Python and dependency management
- Optional for interactive answers: LM Studio or `llama-server`

PostgreSQL is not used by the Chapter 1 baseline. Later chapters will support
Homebrew `postgresql@18`; no older PostgreSQL installation is required.

## One-command foundation setup

```bash
./scripts/init_environment.sh
```

The script installs Python 3.12 through `uv`, installs exactly the locked
dependencies, creates `.env` from the safe example when needed, and runs an
offline end-to-end RAG check. It does not overwrite an existing `.env`.

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
make serve      # FastAPI and Gradio in one process
make doctor     # models endpoint, chat completion, and embedding checks
make test       # deterministic tests; network sockets are disabled
make test-live  # optional provider smoke test (see docs/quick-start.md)
make lint       # Ruff
make typecheck  # strict mypy
make check      # lint, typecheck, and deterministic tests
uv run lke openapi  # export .artifacts/openapi.json
```

## Chapter 1 boundaries

- Two small English fixtures only
- One in-memory LangChain vector store
- Local Hugging Face embeddings
- Local OpenAI-compatible chat model
- Single user and single collection
- No uploads, PostgreSQL, authentication, or background jobs yet

See [the detailed quick start](docs/quick-start.md), [the four-stage code map](docs/architecture.md),
and [the initial RAG issue baseline](docs/blog-coverage.md).

## API

- `GET /healthz` — component-level chat and embedding health
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
