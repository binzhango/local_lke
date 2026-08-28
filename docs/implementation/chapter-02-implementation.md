# Chapter 2 Implementation Report: Safe, Versioned Ingestion

This report records the code and acceptance evidence for the Chapter 2
milestone. The companion [learning notes](../learning/chapter-02-knowledge-guide.md) explain
the concepts; the [operations guide](../operations/chapter-02-ingestion.md) covers setup and
parser choices.

## Result

The application now creates collections and ingests `.md`, `.txt`, and `.pdf`
files into PostgreSQL 18. Every successful chunk is linked to an immutable
document version and normalized source element. Re-uploading identical content
under the same pipeline configuration skips parsing and chunk persistence.

Chapter 1 cited chat remains deliberately unchanged and in memory. Chapter 3
will embed persisted chunks and make collections queryable.

## Delivered components

- PostgreSQL 18, SQLAlchemy, psycopg, Alembic, and one initial relational migration
- Repository boundary isolating persistence from ingestion orchestration
- Collections, logical documents, immutable versions, normalized elements,
  chunks, pipeline configurations, and recoverable jobs
- Startup recovery from `running` to explicit `interrupted` state
- Safe filename normalization and generated-ID storage paths
- Extension, MIME, signature/encoding, per-file, per-batch, encrypted-PDF, and
  malformed-PDF validation
- Direct Markdown/text parsers and Unstructured PDF `fast`/`hi_res` parsing
- Page, source-line, heading, element-category, table, order, and parent metadata
- Recursive, heading-aware, and local TF-IDF semantic-experiment chunking
- Stable SHA-256 chunk IDs, content hashes, and pipeline hashes
- Atomic one-active-version replacement, idempotent skips, and soft deletion
- Collection, upload, job, retry, history, preview, and deletion API contracts
- Gradio collection management, multi-file upload, job polling/retry, version
  history, parser preview, and chunk inspection
- Explicit PostgreSQL doctor checks and idempotent local database bootstrap script

## Acceptance evidence

The deterministic suite covers:

- a real empty PostgreSQL 18 cluster and Alembic migration;
- Markdown headings/source lines and plain-text paragraph locators;
- multipage PDF element/category/table normalization and blank-element omission;
- malformed/encrypted PDFs, traversal filenames, MIME mismatch, and file limits;
- all three chunking strategies, provenance, stable IDs, and repeated chunks;
- identical-ingestion skip behavior and pipeline-change version replacement;
- failed-job inspection, explicit retry, interruption recovery, and soft deletion;
- API lifecycle and structured errors;
- Gradio construction and ingestion callback flow;
- the unchanged Chapter 1 query, streaming, citation, and provider behavior.

Run the gate with:

```bash
make check
bash -n scripts/init_environment.sh scripts/init_postgres.sh
uv run lke doctor --skip-providers --skip-database
uv run lke openapi
```

The live model test remains opt-in and is not part of the offline milestone gate.

## Residual limitations

- PDF OCR and layout fidelity vary with the file and host dependencies.
- The semantic strategy is a deterministic TF-IDF experiment, not an embedding
  benchmark; Chapter 3 measures retrieval quality.
- Persisted uploads are not searchable by the RAG pipeline until Chapter 3.
- API jobs run after the HTTP response in the single local process; Gradio calls
  the same service directly. Persistence and retry semantics are in place, but a
  durable external worker queue is not.
- Authentication, multi-user authorization, antivirus scanning, and hosted
  deployment remain outside the local single-user threat model.
