# Chapter 4 Retrieval Operations

## Apply the migration

```bash
make init-postgres
```

The migration adds a generated English `tsvector`, its GIN index, and structured
table metadata. It does not install pgvector because Chapter 3 is intentionally
not part of this milestone.

## Query persisted documents

First ingest `.md`, `.txt`, or `.pdf` through the Chapter 2 Documents panel or
API. Then send a collection-scoped request:

```json
{
  "collection_id": "00000000-0000-0000-0000-000000000000",
  "question": "What is the Zephyr deployment code?",
  "strategy": "hybrid",
  "rewrite": "none",
  "top_k": 3,
  "metadata_filter": {
    "conditions": [
      {"field": "filename", "operator": "contains", "value": "runbook"}
    ],
    "allow_unfiltered_fallback": false
  },
  "infer_metadata_filter": false
}
```

Use `dense` for paraphrase-heavy questions over a small corpus. Use `hybrid`
when exact names, IDs, terminology, or mixed lexical/semantic wording matters.
Hybrid performs both searches and adds fusion latency. Reranking adds one local
model inference per candidate and should be enabled only after measurement.

The response trace is the primary debugging surface. Inspect:

- `transform`: original/normalized query, route, subqueries, rewrite;
- `metadata_filter`: the exact interpreted plan;
- `candidates`: dense, lexical, fused, and reranked ranks/scores;
- `context_manifest`: every inclusion, exclusion, and truncation;
- `answerability`: features, threshold, correction, and final decision.

## Metadata filters

Allowed fields:

- `filename`
- `media_type`
- `parser_strategy`
- `chunk_strategy`
- `page_number`
- `created_at`

Allowed operators are `eq`, `ne`, `in`, `contains`, `gt`, `gte`, `lt`, and
`lte`. `created_at` values must be ISO-8601 strings. Invalid fields/types fail
closed. If a restrictive model/client plan may be discarded, it must explicitly
set `allow_unfiltered_fallback` to true; otherwise no-result means no-result.
Set `infer_metadata_filter=true` to ask the local model for JSON; the same
Pydantic allowlist is applied before retrieval.

## Query transformations

- `none`: normalized original query only.
- `step_back`: adds one general-background retrieval probe.
- `hyde`: adds one hypothetical answer-style retrieval probe.

Multi-part questions are decomposed to at most
`LKE_RETRIEVAL_MAX_SUBQUERIES`. Generated probes are never treated as factual
evidence. There is no recursive planning loop.

## Enable the optional reranker

```dotenv
LKE_RERANKER_ENABLED=true
LKE_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

The model must exist in the local Hugging Face cache or be downloaded once.
Compare labeled Recall@k/ranking metrics and `reranker_latency_ms` before and
after enabling it.

## Upload and query CSV data

Upload through the Structured Data panel or:

```bash
curl -F 'file=@/absolute/path/sales.csv;type=text/csv' \
  http://127.0.0.1:8000/api/v1/collections/COLLECTION_ID/structured-tables
```

Then submit a validated plan:

```json
{
  "table_id": "00000000-0000-0000-0000-000000000000",
  "question": "What is total revenue by region?",
  "plan": {
    "projections": [],
    "filters": [],
    "group_by": ["region"],
    "aggregations": [
      {"function": "sum", "column": "revenue", "alias": "total_revenue"}
    ],
    "order_by": [{"column": "total_revenue", "direction": "desc"}],
    "limit": 20
  }
}
```

Omit `plan` to ask the configured local model for JSON. The result is still
Pydantic-validated and compiled by SQLAlchemy; the model never supplies SQL.

## Limits and troubleshooting

Configuration:

```dotenv
LKE_RETRIEVAL_CANDIDATE_LIMIT=20
LKE_RETRIEVAL_CONTEXT_TOKENS=1800
LKE_RETRIEVAL_SOURCE_TOKENS=900
LKE_RETRIEVAL_MAX_SUBQUERIES=4
LKE_RETRIEVAL_RRF_K=60
LKE_RETRIEVAL_ANSWERABILITY_THRESHOLD=0.34
LKE_STRUCTURED_MAX_ROWS=100
LKE_STRUCTURED_MAX_CSV_ROWS=50000
LKE_STRUCTURED_MAX_COLUMNS=100
LKE_STRUCTURED_STATEMENT_TIMEOUT_MS=3000
```

If an answer abstains, inspect term/subquery coverage before lowering the
threshold. Lowering it may trade false abstentions for unsupported answers. If
CSV inference chooses text, normalize the source values; mixed-type columns are
intentionally inferred as text. If an aggregate plan fails, verify numeric types
and ensure every projected non-aggregate column is in `group_by`.

## Security boundary

This is a local single-user workbench, not a multi-tenant database gateway.
Nevertheless, structured execution is read-only and bounded, raw SQL is not an
input, identifiers are allowlisted, values are parameters, and provenance is
returned. Keep the service bound to `127.0.0.1`; authentication and row-level ACLs
remain future work.
