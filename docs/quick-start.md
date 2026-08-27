# Chapters 1-5 Quick Start

## Automated foundation setup

From a clean checkout, run:

```bash
./scripts/init_environment.sh
```

This script is safe to run again. It requires `uv`, ensures Python 3.12 exists,
runs `uv sync --locked`, creates `.env` only when it is missing, and executes a
deterministic RAG doctor check. The check uses fake local providers and therefore
does not contact the internet or require a model server.

Chapter 2 also requires the local PostgreSQL 18 service:

```bash
brew install postgresql@18 pgvector
brew services start postgresql@18
make init-postgres
```

The last command creates `local_lke` only when absent and applies the Alembic
migrations. See [the ingestion guide](chapter-02-ingestion.md) for alternate
paths, parser dependencies, and database troubleshooting.

If `uv` is not installed, follow the official instructions at
<https://docs.astral.sh/uv/getting-started/installation/> and rerun the script.

## LM Studio

1. Install LM Studio and download an instruction-tuned model that fits your
   machine.
2. Load the model.
3. Open **Developer → Local Server** and start the OpenAI-compatible server. The
   common default is `http://127.0.0.1:1234/v1`.
4. Open `http://127.0.0.1:1234/v1/models` and copy the exact model `id`.
5. Update `.env`:

```dotenv
LKE_CHAT_BASE_URL=http://127.0.0.1:1234/v1
LKE_CHAT_MODEL=the-exact-loaded-model-id
LKE_CHAT_API_KEY=lm-studio
```

6. Verify all providers and start the workbench:

```bash
make doctor
make serve
```

`make doctor` checks the models endpoint, sends a minimal completion, and
initializes the local embedding model. It also verifies PostgreSQL 18, pgvector,
the fixed vector dimensions, and a vector round trip. The first embedding
initialization may download `BAAI/bge-small-en-v1.5`; later runs use the local
cache. The optional CLIP model loads only when image indexing is used.

## Serving an Unsloth GGUF

An Unsloth-exported GGUF is a model file, not a server by itself. Use either of
these local OpenAI-compatible options:

- Import the GGUF into LM Studio, load it, and use the LM Studio steps above.
- Start llama.cpp's `llama-server` with your GGUF and an explicit model alias:

```bash
llama-server \
  --model /absolute/path/to/model.gguf \
  --alias local-unsloth \
  --host 127.0.0.1 \
  --port 8080
```

Then configure:

```dotenv
LKE_CHAT_BASE_URL=http://127.0.0.1:8080/v1
LKE_CHAT_MODEL=local-unsloth
LKE_CHAT_API_KEY=local
```

Do not bind a model server to `0.0.0.0` unless you intentionally configure
network access and authentication. Local LKE also defaults to `127.0.0.1`.

## Tests and contracts

The normal suite cannot open network sockets:

```bash
make check
```

To opt into the real provider smoke test after starting your server:

```bash
LKE_RUN_LIVE_TESTS=1 make test-live
```

Export the current OpenAPI contract for inspection:

```bash
uv run lke openapi
```

The generated `.artifacts/openapi.json` is a local build artifact and is ignored
by Git.

## Chapter 3 index and first query

Create a collection and ingest a text/Markdown/PDF document in the Documents
tab. Successful ingestion automatically schedules its persistent text index.
Open Retrieval Lab, refresh collections, select that collection, inspect index
state, and compare sentence-window, parent, and multi-granularity expansion.
Each result exposes the triggering child, expanded context, locator, duplicate
decision, and token total.

For image retrieval, open Multimodal Search and upload a PNG, JPEG, or WebP.
Search with either text or another image. Returned assets include provenance;
the application does not claim the text-only answer model inspected them.

Then choose Chapter 4 `hybrid` collection retrieval. Its stage comparison
exposes persistent dense, lexical, fused, optional reranked, and final context
decisions.

For tabular data, open Structured Data and upload UTF-8 CSV. Copy the returned
table ID into the query form. You may provide a Pydantic-shaped plan or let the
configured local model propose JSON. The application always validates and
compiles the plan; it never executes model SQL.

See [Chapter 3 indexing operations](chapter-03-indexing.md) and
[Chapter 4 retrieval operations](chapter-04-retrieval.md) for request examples
and tuning controls.

## Chapter 5 validated generation

In Chat or Retrieval Lab, choose conversational, structured, or evidence-only
output. Structured mode also requires `fact_list` or `comparison`. Status,
warnings, validation/degradation state, and citations remain separate.

Prompt-plus-parser mode is the compatible default. Enable native JSON Schema
only when your local server supports it:

```dotenv
LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=false
LKE_GENERATION_MAX_REPAIR_ATTEMPTS=1
```

See [Chapter 5 generation operations](chapter-05-generation.md) for API examples,
status handling, structured-output troubleshooting, and the streaming contract.

## Troubleshooting

### The configured model is not loaded

Use the exact `id` returned by `/v1/models`. Display names and filenames are not
always the same as the API model ID.

### The local server cannot be reached

Confirm the server is running, its port matches `LKE_CHAT_BASE_URL`, and the URL
ends in `/v1`. `make doctor` reports the failing component without exposing the
API key.

### Embedding initialization fails

The default embedding model must be downloaded once. Ensure you have network
access for that first initialization, or point `LKE_EMBEDDING_MODEL` to an
already-cached sentence-transformers model.

### PostgreSQL cannot be reached

Run `/opt/homebrew/opt/postgresql@18/bin/pg_isready`, then start the service with
`brew services start postgresql@18`. Run `make init-postgres` again to create the
database and apply migrations; the command is idempotent.

### pgvector is unavailable or has the wrong dimension

Run the extension query and safe Homebrew reinstall procedure in the
[Chapter 3 indexing guide](chapter-03-indexing.md). Do not pad, truncate, or cast
vectors to hide a dimension mismatch; use the matching model or an explicit
schema migration.
