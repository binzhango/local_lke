# Chapter 4 Implementation Report

## Outcome

Chapter 4 added advanced retrieval and safe structured queries directly on top
of the Chapter 2 persistence milestone. Historical sequencing matters: the
Chapter 4 commit landed before Chapter 3, so its original dense implementation
used an explicit on-demand bridge. Chapter 3 has since filled that seam with
persistent pgvector/HNSW and embedding-profile lifecycle support while retaining
the bridge as a compatibility fallback.

Persisted Chapter 2 chunks are now queryable through:

- persistent-first local dense embeddings with a bounded on-demand fallback;
- PostgreSQL English full-text search with a generated `tsvector` and GIN index;
- identity-safe reciprocal-rank fusion;
- an optional local cross-encoder;
- allowlisted metadata filters;
- bounded transformation/decomposition;
- coverage-aware, budgeted context assembly;
- deterministic answerability and one corrective attempt;
- validated CSV-to-SQLAlchemy structured queries.

## Architecture

```text
QueryRequest
  -> QueryTransformPlan
  -> active-version metadata prefilter
  -> dense pool -----------+
  -> lexical pool ---------+-> identity/RRF fusion
                              -> optional cross-encoder
                              -> coverage-first context manifest
                              -> sufficiency assessment
                              -> answer | one alternate retrieval | abstain

CSV bytes
  -> validation + schema/type inference
  -> generated collection-scoped SQL table + immutable version
  -> optional model JSON
  -> StructuredQueryPlan validation
  -> SQLAlchemy Core Select
  -> read-only/time-limited/row-limited execution
```

## Retrieval controls

The repository selects only chunks for:

- the requested collection;
- a non-deleted logical document;
- the active document version;
- a completed version.

Metadata filters cover filename, media type, parser strategy, chunk strategy,
page number, and version creation time. Fields and operators are enumerated.
Unfiltered fallback is opt-in per plan.
Clients may provide the plan or explicitly request local-model inference; model
JSON crosses the same Pydantic boundary and never becomes SQL.

PostgreSQL uses the generated indexed `chunks.search_vector`. SQLite uses a
deterministic BM25 fallback so unit/integration tests need no server. Dense
retrieval prefers the compatible active Chapter 3 profile and HNSW index; an
unindexed collection can still use bounded on-demand cosine similarity. Both
channels fetch `LKE_RETRIEVAL_CANDIDATE_LIMIT`; RRF uses
`LKE_RETRIEVAL_RRF_K`.

The optional cross-encoder is lazy and disabled by default. Enabling it requires
the model to be cached or available for a one-time download. Traces retain ranks
before and after reranking plus latency and score delta.

## Context and answerability

The assembler reserves one evidence candidate per subquery before optional
support. It deduplicates near-identical text, limits one document to three
chunks, respects per-source/total token budgets, and records a decision for
every candidate.

The sufficiency policy measures query-term coverage, required-subquery coverage,
and evidence strength. One alternate strategy is allowed. There is no web search
or unbounded agent loop. Failure after correction returns an explicit abstention
with no citations.

## Structured safety

CSV tables have generated physical names and document-version provenance. A
Pydantic plan can describe projection, filters, grouping, ordering, aggregation,
and a row limit. Unknown fields are forbidden. SQLAlchemy Core owns every SQL
identifier and operator; user/model values remain parameters.

PostgreSQL execution uses:

```sql
SET TRANSACTION READ ONLY;
SET LOCAL statement_timeout = <configured integer>;
```

The compiler emits only `Select` objects and adds a hard limit. Raw SQL is absent
from the request schema and never executed.

## API and UI

`POST /api/v1/query` remains backward compatible: no `collection_id` uses the
Chapter 1 fixture baseline. A persisted request supplies `collection_id` plus
`dense` or `hybrid` strategy. The trace includes transformation, filter, route,
channel/fused/reranked ranks, manifest, and answerability.

New structured endpoints:

- `POST /api/v1/collections/{id}/structured-tables`
- `GET /api/v1/collections/{id}/structured-tables`
- `POST /api/v1/structured/query`

The Gradio Retrieval Lab exposes strategy/rewrite/filter controls and one
comparison record containing dense, lexical, fused, reranked, and final stages.
The Structured Data panel exposes inferred schema, safe plan, SQL preview, rows,
and provenance.

## Acceptance evidence

| Requirement | Deterministic evidence |
|---|---|
| rare exact lexical recall | `test_hybrid_retrieval_exposes_all_ranks_and_rare_exact_match` |
| metadata prefilter/fallback | `test_metadata_filters_are_applied...`, `test_unfiltered_fallback_requires...` |
| RRF/rerank trace | `test_reranker_records_before_after_latency_and_gain` |
| active lifecycle | `test_only_active_non_deleted_versions_are_retrievable` |
| multi-part coverage/manifest | `test_multi_part_context_covers_every_subquery...` |
| bounded correction/abstention | `test_unanswerable_query_retries_once_then_abstains` |
| CSV schema/aggregate/provenance | `test_csv_schema_inference_aggregate_query_and_provenance` |
| row limit/read-only plan | `test_structured_filter_and_hard_result_limit` |
| injection/raw SQL rejection | `test_structured_plans_reject_raw_sql_and_unknown_columns` |
| API/Gradio contracts | Chapter 4 API tests and workbench smoke test |
| real PostgreSQL 18 FTS | `test_migrations_apply_to_an_empty_postgresql_18_database` |

Run the full gate with `make check`.

## Residual limits

- Unindexed collections still use linear/on-demand dense retrieval; build a
  Chapter 3 index before scaling the corpus.
- English PostgreSQL stemming is appropriate for the current English learning
  corpus; multilingual FTS needs language-aware configuration.
- Type inference is conservative and does not infer currency, timezone, or
  domain enums.
- Model-produced plans can still be semantically wrong even when structurally
  safe; the UI exposes the plan and SQL preview for inspection.
- The answerability threshold needs corpus-specific labeled calibration.
- Cross-encoder model download/inference is optional and excluded from the
  network-disabled deterministic suite.
