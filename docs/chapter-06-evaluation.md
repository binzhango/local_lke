# Chapter 6 Evaluation Operations Guide

## Create an immutable dataset

```bash
curl -s http://127.0.0.1:8000/api/v1/evaluations/datasets \
  -H 'content-type: application/json' \
  -d '{
    "name": "Atlas release gate",
    "description": "Fixture answer and outage behavior",
    "cases": [
      {
        "case_id": "atlas-p1",
        "question": "How quickly does Atlas acknowledge a priority-one incident?",
        "top_k": 1,
        "expectation": {
          "relevant_source_ids": ["fixture:atlas-support"],
          "answer_contains": ["15 minutes"],
          "acceptable_statuses": ["answered"]
        }
      },
      {
        "case_id": "atlas-p1-outage",
        "question": "How quickly does Atlas acknowledge a priority-one incident?",
        "top_k": 1,
        "fault": "chat_unavailable",
        "expectation": {
          "relevant_source_ids": ["fixture:atlas-support"],
          "acceptable_statuses": ["degraded"]
        }
      }
    ]
  }'
```

Save the returned dataset `id`. Reposting identical content returns the existing
version. Any content change creates the next version for that name.

For persisted retrieval, set `collection_id` and choose `dense` or `hybrid`.
Label stable chunk IDs when source-version IDs would be too broad.

## Run the dataset

```bash
curl -s http://127.0.0.1:8000/api/v1/evaluations/runs \
  -H 'content-type: application/json' \
  -d '{
    "dataset_id": "DATASET_UUID",
    "thresholds": {
      "min_case_pass_rate": 1.0,
      "min_recall_at_k": 1.0,
      "min_status_match_rate": 1.0,
      "max_p95_latency_ms": 5000
    }
  }'
```

The endpoint runs synchronously and persists both success and failure state.
Inspect:

- `configuration_sha256` and `configuration` for reproducibility;
- `metrics` for aggregate quality and latency;
- `case_results` for ranked evidence, public status, faults, and failures;
- `gate.passed` for the release decision;
- `trace_id` for correlation with answer traces.

An HTTP 201 response does not imply the regression gate passed. Read
`gate.passed` explicitly.

## Compare against a baseline

Pass a completed run from the same dataset version:

```bash
curl -s http://127.0.0.1:8000/api/v1/evaluations/runs \
  -H 'content-type: application/json' \
  -d '{
    "dataset_id": "DATASET_UUID",
    "baseline_run_id": "BASELINE_RUN_UUID",
    "thresholds": {
      "min_case_pass_rate": 1.0,
      "max_metric_decline": 0.01,
      "max_latency_increase_ms": 100
    }
  }'
```

Retrieve a side-by-side comparison with:

```bash
curl -sG http://127.0.0.1:8000/api/v1/evaluations/compare \
  --data-urlencode baseline_run_id=BASELINE_RUN_UUID \
  --data-urlencode candidate_run_id=CANDIDATE_RUN_UUID
```

Positive quality deltas mean improvement. Positive latency deltas mean slower
execution. Cross-dataset-version comparison is rejected.

## Inspect datasets, runs, and provider capability

```bash
curl -s http://127.0.0.1:8000/api/v1/evaluations/datasets
curl -s http://127.0.0.1:8000/api/v1/evaluations/runs
curl -s http://127.0.0.1:8000/api/v1/evaluations/provider-profile
```

The provider profile is configuration evidence, not a live compatibility probe.
Use `make doctor` for live provider health. The profile records whether native
structured output is enabled, the bounded repair budget, and the supported
fault scenarios.

## Workbench flow

Open the Evaluation tab:

1. Inspect the provider profile.
2. Edit and create the JSON dataset.
3. Copy the returned dataset ID.
4. Set absolute thresholds and optionally a baseline run ID.
5. Run the evaluation and inspect per-case failures before aggregate metrics.

## Choosing labels

- Prefer chunk IDs for precise retrieval acceptance.
- Prefer source IDs when any passage from a small source is sufficient.
- Use short, stable `answer_contains` values for exact business facts.
- Include `abstained` when missing evidence is the correct behavior.
- Include `degraded` only when the application should return cited extracts
  after a controlled provider or schema failure.

Do not encode prose style preferences as exact phrases. The default evaluator is
designed for facts and contracts, not semantic grading.

## Verification

```bash
make migrate
make check
```

The deterministic suite includes dataset persistence, provider outage injection,
metrics, gates, comparison, OpenAPI, Gradio, and PostgreSQL 18 migration coverage.
