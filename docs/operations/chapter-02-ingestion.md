# Chapter 2: Safe, Versioned Ingestion

Chapter 2 turns user files into inspectable, immutable ingestion records. It
does not embed the new chunks into the query index yet; PostgreSQL/pgvector
indexing is the Chapter 3 milestone.

## PostgreSQL 18 setup

The default connection is `postgresql+psycopg://localhost/local_lke`, which uses
the current local operating-system user and commits no password. The doctor
looks for PostgreSQL 18 under `/opt/homebrew/opt/postgresql@18/bin`.

```bash
brew install postgresql@18
brew services start postgresql@18
make init-postgres
make doctor
```

To use a different local installation, set `LKE_POSTGRES_BIN_DIRECTORY` and
`LKE_DATABASE_URL` in `.env`. Apply later schema changes with `make migrate`.
Useful diagnostics are:

```bash
/opt/homebrew/opt/postgresql@18/bin/pg_isready
/opt/homebrew/opt/postgresql@18/bin/psql --dbname local_lke
uv run alembic current
```

“Connection refused” means the service is not running. “Database does not
exist” means `make init-postgres` has not created it. Peer-authentication errors
usually mean the URL names a database user different from the current local
user.

## Safe upload boundary

Only `.md`, `.txt`, and `.pdf` are accepted. The system checks extension,
declared MIME, content signature/encoding, configured file and batch sizes,
path traversal, malformed PDFs, and encryption before selecting a parser.
Uploads are stored below generated collection/upload IDs; filenames never
select a destination directory and uploaded content is never executed.

The SHA-256 digest is computed before parsing. Document text is not logged by
default. A parser failure produces an inspectable failed job and never an active
version.

## Parser choices

- Markdown parsing records each heading path and exact source-line locator.
- Plain text parsing records paragraph line ranges and rejects non-UTF-8 input.
- PDF `fast` uses Unstructured's quick text extraction and is the default.
- PDF `hi_res` asks Unstructured for layout-aware extraction. It is slower and
  may require Poppler, Tesseract, and platform inference libraries. It can
  improve tables and layout but does not guarantee OCR or reading-order fidelity.

PDF elements retain page number, category, reading-order ordinal, heading
context, and bounded scalar metadata. Tables remain individual `Table`
elements when Unstructured identifies them.

## Chunk strategy comparison

For a fixture containing `# Operations`, `## Recovery`, and two recovery
paragraphs:

| Strategy | Boundary behavior | Best Chapter 2 use |
|---|---|---|
| `recursive` | Splits each normalized element using paragraph, line, sentence, word, then character separators | General text and a predictable size ceiling |
| `markdown` | Uses the same safe recursive sizing while every chunk carries its parsed heading path | Markdown documents where section context matters |
| `semantic` | Experimental local TF-IDF similarity detects adjacent sentence topic shifts without a model or network call | Comparing topic-aware boundaries before embedding evaluation exists |

Every chunk records its immutable version, parent element, heading path, page or
line locator, ordinal, character count, approximate token count, and a stable
SHA-256 ID derived from version, strategy, order, and text. Exact repeated chunks
are omitted and counted as warnings.

## Version and job lifecycle

The pipeline hash covers the schema version, parser strategy, chunk strategy,
size, and overlap. Re-uploading the same normalized filename with the same
content and pipeline hash returns a completed `skipped` job referencing the
existing version and performs zero parser/chunk writes. Content or pipeline
changes create a new immutable version and atomically deactivate its predecessor.

Jobs move through `queued`, `running`, and `completed` or `failed`. On process
startup, abandoned `running` jobs become `interrupted`; only failed or
interrupted jobs accept explicit retry. Soft deletion retains provenance and
job history while deactivating every searchable version.

Use the Documents tab at `/app` for collection creation, multi-file upload,
parser/chunker selection, version history, and parser/chunk inspection. The same
operations are available under `/api/v1` and documented at `/docs`.
