# Chapter 5 Implementation Report: Validated Generation Contracts

## Outcome

Chapter 5 replaces free-form answer completion with one validated generation
service shared by the Chapter 1 fixture pipeline and Chapter 4 persisted
retrieval. It supports cited conversational answers, two allowlisted structured
schemas, evidence-only output, bounded repair, explicit abstention, extractive
degradation, safe SSE completion, and escaped Gradio presentation.

No Chapter 6 evaluation control plane was started.

## Architecture

```text
retrieved active context
  -> application citation registry (C1..Cn)
  -> typed locator + excerpt
  -> chapter5.generation.v1 prompt
       system policy
       selected JSON Schema
       answerability and uncovered parts
       retrieval manifest
       JSON-encoded untrusted evidence
       JSON-encoded user question
  -> native structured output (opt-in) or prompt/parser mode
  -> Pydantic validation
  -> citation allowlist validation
       | valid                    | invalid/provider failure
       v                          v
  answered response       one bounded repair -> cited degradation
```

## Public contract

`QueryRequest` adds:

- `output_mode`: `conversational`, `structured`, or `evidence_only`;
- `schema_name`: required only for structured mode and allowlisted to
  `fact_list` or `comparison`.

`AnswerResponse` retains the backward-compatible `answer` string and adds:

- `structured_result`, discriminated by `schema_name`;
- application-resolved citations with `citation_id` and `locator_detail`;
- qualitative confidence explanation;
- uncovered subquestions;
- selected query route;
- trace ID;
- warnings;
- generation trace with prompt version, mode, attempts, safe validation errors,
  native/parser path, degradation reason, and commit state.

Answered and degraded responses require cited evidence. Abstentions can have no
citations and cannot contain a structured result.

## Citation integrity

The generation service accepts `GenerationEvidence` objects. Each must be both
retrieved and active and must have a unique stable ID. The model can reference
only these IDs. Unknown IDs cause validation failure and never become public
source objects.

Fixture citations resolve to bundled source IDs. Persisted citations resolve to
the immutable active document version, chunk, filename, locator, and excerpt.
Typed locators cover Markdown headings, text ranges, PDF page/elements, images,
structured rows, and a legacy generic form.

## Prompt and injection boundary

`generation/prompting.py` owns the versioned prompt. Policy, schema,
answerability, manifest, evidence, and question are separate sections. Evidence
and question values are JSON strings; angle brackets are Unicode-escaped so
retrieved text cannot inject a fake closing delimiter.

The policy tells the model that retrieved commands, role changes, tool requests,
and output instructions are data. The model receives no arbitrary tools.

## Structured output paths

Parser mode is the default. It requires exactly one JSON value and validates it
with Pydantic. It does not extract JSON substrings or strip code fences.

When `LKE_GENERATION_NATIVE_STRUCTURED_OUTPUT=true`, the LangChain provider uses
`with_structured_output(..., method="json_schema")`. This is opt-in because local
OpenAI-compatible servers differ. Both paths end at the same Pydantic and
citation-integrity checks.

Safe validation summaries contain field paths and error categories, not raw
model output or evidence. The default repair budget is one. Exhaustion returns
a cited extractive `degraded` response.

## Failure behavior

| Condition | Result |
|---|---|
| retrieval evidence insufficient | `abstained`; model not called |
| evidence-only selected | `answered`; model not called |
| connection/timeout/provider failure with evidence | cited `degraded` extract |
| empty output | repair, then cited degradation if repeated |
| invalid JSON/missing field/wrong type | one repair, then degradation |
| fabricated citation ID | repair, then degradation; ID never escapes |
| inactive/unretrieved evidence supplied | rejected generation contract |
| interrupted model stream | SSE `error`; no partial answer or `completion` |

Confidence is intentionally qualitative. The response says it is not a
calibrated probability.

## API and Gradio

Synchronous and SSE endpoints serialize the same AnswerResponse. SSE `start`
records mode/schema. Model JSON is buffered until provider completion, schema
validation, and citation validation. Only validated answer text is emitted as
display deltas.

The Gradio Chat and Retrieval Lab expose output-mode and schema controls. Answer
status and warnings are visible. Generated text and citation fields are escaped
before Markdown rendering; application-owned source URLs render separately.
Raw prompts and invalid model output are never shown.

## Dependency correction

The workbench uses Gradio 5's event API and type surface. The earlier broad
`<7` dependency range allowed the lock to resolve Gradio 6 even though the
application targeted Gradio 5. Chapter 5 narrows the declared range to
`gradio>=5.49,<6` and refreshes `uv.lock`, making runtime and strict type checks
reproducible.

## Acceptance evidence

| Requirement | Deterministic evidence |
|---|---|
| public answer/citation invariants | `test_models.py` contract tests |
| typed structured result | `test_allowlisted_fact_list_schema_is_a_typed_public_result` |
| prompt boundaries and hostile evidence | `test_prompt_keeps_policy_manifest_question_and_malicious_evidence_separate` |
| invalid JSON/missing/wrong type | parametrized bounded-repair test |
| repeated repair failure | extractive-degradation unit test |
| fabricated citation rejection | citation-registry unit test |
| inactive/unretrieved rejection | generation-evidence integrity tests |
| evidence-only bypass | no-model-call unit and API tests |
| no-answer remains abstained | insufficient-evidence unit plus Chapter 4 retrieval tests |
| provider outage degradation | integration error-path test |
| interrupted stream | SSE error-without-completion integration test |
| API structured/evidence modes | Chapter 5 API integration test |
| hostile rendered output | Gradio smoke test |
| schema in OpenAPI | API OpenAPI contract test |

Run the complete deterministic gate with `make check`.

## Residual limits

- Native JSON Schema support depends on the configured local server; parser mode
  is the portable default.
- Qualitative confidence is not outcome-calibrated.
- Prompt delimitation and escaped rendering reduce attack surface but are not a
  substitute for authorization or adversarial evaluation.
- No arbitrary model tools are exposed. Tool-calling concepts are documented,
  while application actions remain explicit and allowlisted.
- Structured schemas are intentionally small. New schemas require code, tests,
  public API review, and a prompt-version decision.
- Cross-provider capability profiles and evaluation/fault dashboards belong to
  Chapter 6.
