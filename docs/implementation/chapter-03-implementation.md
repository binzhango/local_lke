# Chapter 3 Implementation Report: Persistent and Multimodal Indexing

## Outcome

Chapter 3 turns the immutable chunks created in Chapter 2 into durable local
search indexes. Text nodes use a versioned embedding profile and PostgreSQL
`vector(384)` columns with HNSW cosine indexes. Images use a separate local CLIP
profile and `vector(512)` table. Index jobs are resumable, incomplete builds are
not searchable, and an active profile is selected per collection.

Chapter 4 happened to be implemented first in this repository. This milestone
therefore fills its indexing seam without replacing its public query contract:
collection retrieval now prefers the persistent Chapter 3 index and retains the
old bounded on-demand dense path as a compatibility fallback when no index is
ready.

## Architecture

```text
Chapter 2 active document version
            |
            v
   IndexingService.index_version
      |        |          |
      |        |          +-- section nodes
      |        +------------- sentence nodes
      +---------------------- chunk nodes
            |
            v
 bounded local embedding batches
            |
            v
 inactive vector_nodes + persisted batch cursor
            |
        complete build
            |
            v
 transactional activation + collection/profile selection
            |
     +------+----------------------+
     |                             |
 retrieval laboratory       Chapter 4 dense recall
 child -> window/parent      persistent-first, safe fallback
```

The image path is deliberately separate:

```text
validated local PNG/JPEG/WebP
       -> RGB decode + metadata
       -> local CLIP image vector
       -> image_assets + image_embeddings
       -> text-to-image or image-to-image search
```

The application returns the matching image and provenance. It does not claim
that the configured text-only answer model inspected pixels.

## Embedding profiles are compatibility contracts

`embedding_profiles` records the fields that determine vector meaning:

- provider and model ID;
- immutable model revision label;
- vector dimension;
- normalization policy;
- document and query instruction prefixes;
- modality.

The stable profile identity namespaces every vector. A collection's active
mapping lives in `collection_index_profiles`, so vectors from different models,
dimensions, prefixes, or normalization policies cannot silently mix. Startup
health compares the configured dimension with the migrated schema and performs
a vector round trip. A mismatch is a visible failure rather than a partially
working search system.

The default text profile is `BAAI/bge-small-en-v1.5`, revision `main`, 384
dimensions, normalized, with the BGE retrieval query instruction. Document
prefixes remain empty. `LocalHuggingFaceEmbeddings` loads the model lazily and
never sends document text to a remote embedding endpoint. Tests inject a
deterministic provider with the same contract.

## Storage model

| Table | Purpose |
|---|---|
| `embedding_profiles` | Immutable text or image embedding contract |
| `collection_index_profiles` | Active profile for a collection and modality |
| `vector_nodes` | Chunk, sentence, and section nodes plus text vectors |
| `indexing_jobs` | Resumable status, batch cursor, counts, and errors |
| `image_assets` | Validated image identity, dimensions, MIME, hash, and provenance |
| `image_embeddings` | Joint-space image vectors under a multimodal profile |

`vector_nodes.embedding` is `vector(384)` on PostgreSQL; image embeddings use
`vector(512)`. Both have HNSW cosine indexes. SQLite receives JSON variants only
for deterministic unit and integration tests; PostgreSQL 18 plus pgvector is the
production contract for this chapter.

Each text node retains collection, document, immutable version, chunk, section,
source locator, granularity, ordinal, and parent relationships. Stable IDs hash
the source identity together with the profile contract. That makes an unchanged
reindex a zero-embedding operation while preventing incompatible reuse.

## Index lifecycle

1. Resolve the immutable document version and current text profile.
2. Create or resume its `indexing_jobs` row.
3. Derive chunk, sentence, and section nodes deterministically.
4. Skip node IDs that already have complete vectors.
5. Embed the remaining texts in bounded batches.
6. Persist each successful batch and advance the job cursor.
7. Leave all new nodes inactive while any batch is incomplete or failed.
8. On success, atomically deactivate stale nodes for that document, activate the
   completed version, and select the profile for the collection.

Retry resumes at the last committed batch. `force=true` starts a replacement
build. A new active version removes stale active-vector references. Soft-deleting
a document also deactivates its vectors. Collection indexing walks only active
versions. These rules keep incomplete and obsolete data out of search results.

## Retrieval and expansion

`PersistentDenseRetriever` provides a LangChain-compatible `invoke()` and
`get_relevant_documents()` interface over the repository. PostgreSQL orders by
pgvector cosine distance; SQLite tests use an exact cosine implementation.

The retrieval laboratory accepts:

- collection and optional profile;
- query and `top_k`;
- `none`, `sentence_window`, `parent`, or `multi` expansion;
- a strict context token budget.

Sentence nodes retrieve the smallest evidence. Sentence-window expansion adds
bounded neighboring sentences in document order. Parent expansion resolves a
child to its complete chunk. Multi-granularity search admits sentence, chunk,
and section nodes so fine questions and broad synthesis questions can use
different retrieval units.

Expansion deduplicates repeated parent contexts. The retained item carries the
best child score and the exact child locator that triggered it. Packing stops at
the token budget and reports included, duplicate, or over-budget decisions. The
retrieval response includes total tokens and the embedding profile, making the
experiment reproducible.

## Multimodal boundary

`ImageIndexingService` accepts `.png`, `.jpg`, `.jpeg`, and `.webp` only. It
checks normalized names, declared MIME, extension, magic/decoded format,
maximum bytes, maximum pixel count, Pillow decompression-bomb protection, and
successful RGB decoding. Files are stored below generated identifiers; public
responses never expose storage paths.

The optional `sentence-transformers/clip-ViT-B-32` provider embeds both text and
images in the same 512-dimensional space. Content hashes make identical uploads
idempotent. Text-to-image and image-to-image endpoints return ranked image IDs,
scores, content URLs, filenames, MIME types, dimensions, and collection
provenance. The provider loads only when the image feature is used, so text RAG
works without paying the multimodal model's memory cost.

## API and workbench

New text-index routes:

- `POST /api/v1/document-versions/{version_id}/index`
- `POST /api/v1/collections/{collection_id}/index`
- `GET /api/v1/collections/{collection_id}/index-state`
- `GET /api/v1/indexing-jobs/{job_id}`
- `POST /api/v1/retrieval-lab`

New image routes:

- `POST /api/v1/collections/{collection_id}/images`
- `POST /api/v1/collections/{collection_id}/images/search/text`
- `POST /api/v1/collections/{collection_id}/images/search/image`
- `GET /api/v1/images/{image_id}/content`

Document upload automatically indexes a successfully ingested version. Retry of
a completed ingestion job also ensures its index. The workbench exposes index
state/profile, build controls, expansion strategy, candidates, duplicate
decisions, token totals, image upload, and both image query modes.

## PostgreSQL and pgvector readiness

The Alembic migration enables `vector`, creates the profile/lifecycle tables,
and builds HNSW indexes. Health and `lke doctor` report four separate properties:

1. PostgreSQL can be reached.
2. `vector` is visible in `pg_available_extensions`.
3. the extension is installed in the database;
4. configured dimensions match the schema and a vector can round-trip.

This distinguishes “the package is installed on disk” from “this database has
the extension and compatible schema.” See
[Chapter 3 indexing operations](../operations/chapter-03-indexing.md) for recovery commands.

## Mapping vector-store concepts to pgvector

| Concept | Local LKE implementation |
|---|---|
| Milvus Collection | PostgreSQL table plus collection foreign key |
| Schema | Alembic-managed SQLAlchemy model and fixed vector typmod |
| Partition | Collection/version/profile predicates and SQL indexes |
| Alias | `collection_index_profiles` active mapping |
| Index | PostgreSQL HNSW cosine operator class |
| ANN search | `ORDER BY embedding <=> query LIMIT k` |
| FAISS metadata sidecar | Relational columns and foreign keys beside each vector |
| Flush/load lifecycle | committed batches followed by transactional activation |

FAISS and Milvus remain conceptual comparisons, not hidden dependencies or
additional live backends. PostgreSQL is appropriate here because the corpus,
version lifecycle, metadata, lexical search, and vectors share one local
transaction boundary.

## Verification

The deterministic suite covers:

- profile idempotency and zero embeddings for unchanged input;
- persisted batch retry and transactional activation;
- replacement and soft deletion;
- schema/profile dimension mismatch;
- correct child ranking, parent/window order, deduplication, and token budgets;
- labelled fine and broad retrieval fixtures;
- safe image validation and deterministic text/image ranking;
- API indexing and multimodal flows;
- PostgreSQL 18 migration, vector extension, schema dimensions, and HNSW indexes.

The marked live-provider test initializes the real local embedding adapter. It
is opt-in because the first run may need model weights. Chapter 3 records
Recall@k on labelled deterministic fixtures rather than treating ANN speed as
proof of retrieval quality.

## Boundaries

- HNSW tuning remains workload-specific; the defaults are a safe starting point,
  not a universal optimum.
- The local image path retrieves assets but does not add a vision-language answer
  model.
- OCR and unusually complex PDF layout remain residual ingestion risks.
- Authentication, multi-user ACLs, remote model endpoints, and distributed vector
  infrastructure remain outside the local single-user scope.
- Changing the vector dimension requires a new migration/table namespace; it is
  intentionally not an in-place cast.

The deeper theory, model history, Milvus/FAISS examples, and optimization
algorithms are in the [Chapter 3 learning guide](../learning/chapter-03-knowledge-guide.md).
