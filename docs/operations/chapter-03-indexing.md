# Chapter 3 Indexing Operations Guide

This guide operates the persistent text and optional image indexes added in
Chapter 3. It assumes Chapter 2 ingestion is already working.

## 1. Install PostgreSQL 18 and pgvector

On Apple Silicon Homebrew:

```bash
brew install postgresql@18 pgvector
brew services start postgresql@18
```

Verify that PostgreSQL 18 can see the extension files:

```bash
/opt/homebrew/opt/postgresql@18/bin/pg_config --version
/opt/homebrew/opt/postgresql@18/bin/psql postgres -c \
  "SELECT name, default_version, installed_version FROM pg_available_extensions WHERE name = 'vector';"
```

If the query returns no row after a PostgreSQL upgrade, reinstall pgvector so
Homebrew builds/links extension artifacts for the active PostgreSQL formula:

```bash
brew reinstall pgvector
brew unlink pgvector
brew link pgvector
```

Then confirm that `vector.control` exists under the directory reported by:

```bash
/opt/homebrew/opt/postgresql@18/bin/pg_config --sharedir
```

Do not copy extension binaries between PostgreSQL major versions. Reinstalling
is safer because server extension modules must match the running server ABI.

On Linux, install the pgvector package built for the exact PostgreSQL major
version, or build pgvector against that version's `pg_config`.

## 2. Initialize and diagnose

```bash
make init-postgres
uv run lke doctor --skip-providers
```

`make init-postgres` creates the local `local_lke` database only when absent and
runs all Alembic migrations. The Chapter 3 migration executes `CREATE EXTENSION
IF NOT EXISTS vector`, creates fixed-dimension columns, and adds HNSW cosine
indexes.

The doctor output must report the database and vector index as healthy. It checks
extension availability, installed version, schema dimensions, and a vector
round trip. Useful direct checks are:

```sql
SELECT extversion FROM pg_extension WHERE extname = 'vector';
SELECT format_type(a.atttypid, a.atttypmod)
FROM pg_attribute AS a
JOIN pg_class AS c ON c.oid = a.attrelid
WHERE c.relname IN ('vector_nodes', 'image_embeddings')
  AND a.attname = 'embedding';
```

Expected types are `vector(384)` for text and `vector(512)` for images.

## 3. Configure embedding profiles

The safe defaults in `.env.example` are:

```dotenv
LKE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LKE_EMBEDDING_MODEL_REVISION=main
LKE_EMBEDDING_DIMENSION=384
LKE_EMBEDDING_NORMALIZE=true
LKE_EMBEDDING_DOCUMENT_PREFIX=
LKE_EMBEDDING_QUERY_PREFIX="Represent this sentence for searching relevant passages: "
LKE_EMBEDDING_BATCH_SIZE=32

LKE_MULTIMODAL_MODEL=sentence-transformers/clip-ViT-B-32
LKE_MULTIMODAL_MODEL_REVISION=main
LKE_MULTIMODAL_DIMENSION=512
```

Model, revision, dimension, normalization, and prefixes are one compatibility
contract. Changing any field creates a distinct profile namespace and requires
a new build. Never point a 768-dimensional model at the migrated 384-dimensional
text column: doctor will reject that configuration. A dimension change requires
an explicit schema migration or separate fixed-dimension table.

The providers run locally. No source text or image is sent to a remote embedding
endpoint.

## 4. Build and inspect a text index

Start the application:

```bash
make serve
```

Upload a document in the workbench Documents tab. Successful uploads are indexed
in the background. For explicit control, call:

```bash
curl -X POST \
  http://127.0.0.1:8000/api/v1/document-versions/VERSION_ID/index

curl -X POST \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/index

curl \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/index-state
```

The state response includes the active profile, total/active/missing nodes, and
latest job progress. Retrieve an individual job with:

```bash
curl http://127.0.0.1:8000/api/v1/indexing-jobs/JOB_ID
```

A failed batch is safe: already completed batches stay persisted, new nodes stay
inactive, and a retry resumes instead of embedding completed node IDs again.
Use `?force=true` on the version-index endpoint only when you intentionally need
a replacement build under the same profile.

## 5. Run the retrieval laboratory

```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieval-lab \
  -H 'Content-Type: application/json' \
  -d '{
    "collection_id": "COLLECTION_ID",
    "question": "What is the escalation window?",
    "top_k": 5,
    "expansion": "sentence_window",
    "token_budget": 800
  }'
```

Expansion strategies:

| Value | Behavior |
|---|---|
| `none` | Return matched nodes as indexed |
| `sentence_window` | Add bounded neighboring sentences in source order |
| `parent` | Retrieve a small child and return its containing chunk |
| `multi` | Search sentence, chunk, and section granularities |

Inspect `trigger_node_id`, score, locator, expansion decision, and token count.
Repeated children can point at the same parent; the response retains only the
best-scoring parent and records duplicate decisions. Results that would exceed
the budget are visible but not silently packed.

In Gradio, open **Retrieval Lab**, select a collection, build/refresh the index,
and compare the same query across strategies. The Chapter 4 collection query
path also uses the persistent dense index when its active profile is compatible.

## 6. Index and search images

Image support is optional and lazy. Accepted formats are PNG, JPEG, and WebP.
The service validates bytes, MIME, extension, decoded format, dimensions, pixel
limits, and RGB decoding before storage.

Upload:

```bash
curl -X POST \
  -F 'file=@/absolute/path/example.png;type=image/png' \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/images
```

Text-to-image search:

```bash
curl -X POST \
  -F 'query=a blue architectural diagram' \
  -F 'top_k=5' \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/images/search/text
```

Image-to-image search:

```bash
curl -X POST \
  -F 'file=@/absolute/path/query.webp;type=image/webp' \
  -F 'top_k=5' \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/images/search/image
```

Each hit includes a local content URL and provenance. Search proves joint-space
similarity only; it does not mean a text-only chat model saw or understood the
image. Keep the returned images as assets/citations or configure a separate
vision-language generation stage later.

## 7. Lifecycle behavior

- Identical content and profile: stable node IDs, zero new embedding calls.
- New active document version: build inactive nodes, then atomically replace the
  old active set after complete success.
- Failed build: old active version remains searchable.
- Soft delete: text vectors for the document become inactive.
- Profile change: new namespace; old vectors never mix with new query vectors.
- Image duplicate: content hash makes ingestion idempotent within a collection.

These invariants matter more than whether an individual ANN query is fast. A
fast index that exposes partial or stale content is not a correct retrieval
system.

## 8. HNSW operations and tuning

PostgreSQL uses the cosine operator class and `<=>` distance. HNSW offers good
query latency without a training phase, at the cost of memory and build time.
Tune only against labelled queries:

- higher `m`: more graph edges, memory, build time, and often recall;
- higher `ef_construction`: slower builds and often better graph quality;
- higher query `hnsw.ef_search`: more candidates, slower queries, often better
  Recall@k.

Compare ANN output with exact search on a representative corpus before changing
defaults. Reindex after index-definition changes. Do not infer answer quality
from latency alone.

## 9. Troubleshooting

### `vector` is unavailable

Confirm the server major version and `pg_config`, reinstall pgvector, restart
PostgreSQL, then rerun `make init-postgres`. Availability is cluster-level;
installation via `CREATE EXTENSION` is database-level.

### Dimension mismatch

Restore the configured model/dimension that matches the schema, or create a new
migration/table for the new dimension. Do not truncate, pad, or cast embeddings.

### Index remains incomplete

Read the indexing job error and cursor. Correct the local provider or input
problem, then call the version index endpoint again. Completed batches are
reused. Search continues using the prior active index.

### First model load is slow

Sentence Transformers may download weights once. After that, use the local
cache. Text-only use does not initialize CLIP.

### Image rejected

Check extension, declared MIME, actual decoded format, file byte limit, pixel
count, and whether Pillow can decode it safely. Renaming a file does not change
its actual format.

## 10. Verification commands

```bash
make check
LKE_RUN_LIVE_TESTS=1 make test-live
uv run lke openapi
```

The ordinary suite is deterministic and disables network sockets. The live
suite is opt-in because it exercises locally configured real providers.

For the theory behind normalization, model selection, CLIP, vector databases,
FAISS/Milvus comparisons, HNSW, and context expansion, read
[Chapter 3 learning guide](../learning/chapter-03-knowledge-guide.md). For the exact
implementation and test boundaries, read
[Chapter 3 implementation report](../implementation/chapter-03-implementation.md).
