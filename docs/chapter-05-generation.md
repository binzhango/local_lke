# Chapter 5 Generation Operations Guide

## Choose an output mode

All query requests default to conversational output.

```bash
curl -s http://127.0.0.1:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{
    "question": "How quickly does Atlas acknowledge a priority-one incident?",
    "top_k": 1,
    "output_mode": "conversational"
  }'
```

Structured mode requires an allowlisted schema:

```bash
curl -s http://127.0.0.1:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{
    "question": "List the documented Atlas support facts.",
    "top_k": 3,
    "output_mode": "structured",
    "schema_name": "fact_list"
  }'
```

Supported schema names are:

- `fact_list`: `summary` plus cited `facts`;
- `comparison`: `summary` plus cited subject/detail `items`.

Evidence-only mode does not call the chat model:

```bash
curl -s http://127.0.0.1:8000/api/v1/query \
  -H 'content-type: application/json' \
  -d '{
    "question": "What is the incident acknowledgement target?",
    "top_k": 3,
    "output_mode": "evidence_only"
  }'
```

The same fields work with collection-scoped dense/hybrid retrieval and with
`POST /api/v1/query/stream`.

## Read a response

Check these fields before displaying the answer:

- `status`: answered, abstained, degraded, or error;
- `answer`: backward-compatible text or extractive display value;
- `structured_result`: typed object when structured generation succeeds;
- `citations`: application-resolved evidence used by the result;
- `uncovered_subquestions`: requested parts without evidence;
- `warnings`: repair, bypass, or degradation notices;
- `trace_id`: identity for this response;
- `trace.generation`: prompt version, attempts, mode, schema, safe validation
  categories, and commit/degradation state.

Do not interpret `confidence.level` as a numeric probability. Its
`calibration` field explicitly describes the qualitative meaning.

## Local model configuration

Parser mode is the default and works with the broadest range of local
OpenAI-compatible servers:

```dotenv
LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=false
LKE_GENERATION_MAX_REPAIR_ATTEMPTS=1
```

Enable native JSON Schema only if the configured server supports the relevant
OpenAI-compatible response format:

```dotenv
LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=true
```

Restart `make serve` after changing environment variables.

The repair setting accepts zero through three. Keep it small. A higher value
increases latency and can conceal a server that does not follow the selected
schema.

## Status handling

### `answered`

The response passed its selected Pydantic schema and citation-integrity checks,
or evidence-only mode returned retrieved extracts. Confirm citations are present.

### `abstained`

The retrieval gate found an evidence gap. The model was not called. Use
`uncovered_subquestions` and the answer's safe next action to decide which
document or collection needs improvement.

### `degraded`

Retrieval found sufficient evidence, but synthesis was unavailable or invalid.
The answer contains application-selected cited excerpts. Inspect:

- `warnings`;
- `trace.generation.degraded_reason`;
- `trace.generation.validation_errors`;
- provider health at `GET /healthz`.

Do not silently relabel degraded output as answered in a client.

### SSE `error`

An interrupted stream emits `error` without `completion`. Discard any local UI
buffer for that request. Local LKE itself does not expose partial model JSON as
answer deltas.

## Troubleshooting

### Native structured output fails

Set:

```dotenv
LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=false
```

Then restart the service. Parser mode still enforces the same public schema.
Native support varies among LM Studio, `llama-server`, and other compatible
servers and versions.

### Every answer is degraded with validation errors

1. Confirm the configured model is an instruction-following chat model.
2. Use parser mode.
3. Keep temperature at the provider's configured deterministic value.
4. Inspect validation categories, not hidden/raw output.
5. Try `evidence_only` to verify retrieval independently of synthesis.
6. Do not add JSON-extraction heuristics; fix model compatibility or schema size.

### Empty response or connection failure

Run:

```bash
make doctor
```

Confirm the local server is running and `LKE_CHAT_MODEL` exactly matches a model
ID returned by the local `/v1/models` endpoint. A synchronous query with
sufficient evidence should return a cited degraded response while the model is
unavailable.

### Structured request receives HTTP 422

Structured mode requires `schema_name`. Schema names are allowlisted. A schema
name is invalid for conversational or evidence-only mode.

### A model cites an unknown ID

The invalid ID is rejected and cannot appear as a returned Citation. One repair
is attempted; repeated failure becomes a cited extractive degradation using the
real retrieval registry.

### Gradio displays escaped formatting

This is intentional for untrusted generated and source text. The workbench
escapes Markdown/HTML controls and renders citations separately. Clients that
want rich formatting must use a reviewed sanitizer and preserve the separate
citation contract.

## Verification commands

```bash
make lint
make typecheck
make test
make check
```

Optional live-provider coverage remains marked `live`:

```bash
make test-live
```

The deterministic suite does not require a model server or network access.

## Security boundary

- Retrieved instructions are untrusted evidence.
- The model receives no arbitrary filesystem, shell, SQL, network, or graph tool.
- Citation URLs and source identity come from application state, not model text.
- Invalid raw model output is not returned in traces.
- Gradio escapes generated/source content before Markdown rendering.
- The application remains loopback-bound, local, and single-user; it does not
  provide multi-user authorization.
