# Chapter 6 Implementation Report: Evaluation Control Plane

## Outcome

Chapter 6 adds an executable evaluation control plane over the Chapter 5 answer
contract. It persists immutable labelled datasets and run snapshots, evaluates
fixture or collection queries, injects bounded provider faults, computes
deterministic retrieval/answer/citation/latency metrics, and applies absolute and
baseline-relative regression gates.

## Architecture

```text
immutable dataset version + SHA-256
  -> case input and labelled expectations
  -> normal or run-local fault provider
  -> existing RAG query boundary
  -> public AnswerResponse + trace ID
  -> deterministic per-case metrics
  -> aggregate metrics
  -> absolute thresholds + optional baseline deltas
  -> persisted completed run and pass/fail gate
```

`evaluation/models.py` owns the public dataset, case, metric, provider-profile,
run, comparison, and gate contracts. `evaluation/service.py` owns canonical
hashing, execution, fault isolation, scoring, persistence, and comparison.

## Persistence

Migration `20260827_04` adds:

- `evaluation_datasets`, with unique name/version and content digest;
- `evaluation_runs`, with dataset and optional self-referential baseline foreign
  keys, configuration snapshot, metrics, per-case results, gate, and lifecycle.

Cases and results are JSON snapshots validated through Pydantic at both write and
read boundaries. This keeps the complete run evidence immutable without making
metric-specific relational columns part of the migration contract.

## Execution

Normal cases use the same `RAGPipeline` or `AdvancedRetrievalService` instance as
the API. Fault cases construct a run-local provider wrapper and generation
service while reusing the same documents, embeddings, retrieval repository,
reranker, index, and settings. Shared provider state is never modified.

Supported faults are unavailable chat, empty output, and malformed output. They
exercise the real Chapter 5 repair/degradation boundary rather than fabricating a
post-hoc status.

An unexpected execution error marks the persisted run `failed` with a safe error
category. Provider raw output, prompts, evidence bodies, and tracebacks are not
stored in evaluation error fields.

## Metrics and gates

Metrics are calculated from application-owned source/chunk identities and the
validated public response:

- Recall@k, mean reciprocal rank, and binary nDCG@k;
- citation precision and recall;
- exact required-phrase answer match;
- public status match and complete case pass rate;
- end-to-end p50 and p95 latency.

The default gate requires every case to pass. Optional absolute thresholds can
be set per run. Baseline comparison accepts only the same immutable dataset
version and records candidate-minus-baseline deltas. Quality and latency
tolerances are configured independently.

## Delivery surfaces

FastAPI exposes dataset create/list/get, run create/list/get, comparison, and
provider-profile endpoints. The mounted Gradio workbench adds an Evaluation tab
for the same service. OpenAPI contains the typed Chapter 6 schemas.

The provider capability profile records configuration and supported controlled
faults. It does not claim a live server supports a feature merely because an API
shape exists; live health remains `make doctor` and `/healthz`.

## Acceptance evidence

| Requirement | Evidence |
|---|---|
| immutable versioned datasets | API persistence test and unique migration constraints |
| deterministic retrieval metrics | `test_ranking_metrics_use_labelled_relevance` |
| absolute regression gate | `test_aggregate_and_absolute_gate_are_deterministic` |
| provider outage degradation | Chapter 6 API run test with `chat_unavailable` |
| persisted per-case run evidence | run GET equality assertion |
| baseline comparison | same-dataset candidate comparison assertion |
| provider capability contract | provider-profile API assertion |
| typed public API | OpenAPI assertion |
| PostgreSQL 18 clean migration | migration integration test |
| workbench construction | existing Gradio smoke suite |

Run the complete gate with `make check`.

## Residual limits

- Runs are synchronous and bounded to 1,000 cases.
- Required-phrase matching is deterministic, not semantic equivalence.
- Relevance is binary; no graded judgements are accepted.
- Structured tables and multimodal search retain deterministic tests but are not
  yet routed through this dataset schema.
- The provider profile is declarative configuration evidence; it is not a live
  conformance suite across multiple vendors.
- Evaluation endpoints inherit the application's local single-user boundary;
  authentication and multi-tenant authorization remain out of scope.
