# Chapter 6 Knowledge Guide: Evaluation and Regression Control

Chapter 6 turns the earlier deterministic tests into a product-facing evaluation
control plane. The central idea is simple: a RAG change is not better because an
individual demo looks better. It is better only when it improves a fixed,
representative dataset without unacceptable regressions in retrieval, answers,
citations, failure behavior, or latency.

## 1. What an evaluation unit contains

An evaluation case keeps inputs and expectations together:

- a stable case ID and question;
- an optional persisted collection and retrieval configuration;
- the desired output mode;
- labelled relevant source or chunk identities;
- required answer phrases;
- acceptable public statuses;
- an optional controlled provider fault.

The labels should describe observable application behavior. They do not contain
hidden chain-of-thought, raw provider output, or executable instructions.

## 2. Why datasets are immutable

Changing the questions or labels changes the test. Comparing a new run against
an old run is meaningful only when both use the exact same dataset version.

Local LKE canonicalizes the dataset content, computes a SHA-256 digest, and
assigns a monotonically increasing version per dataset name. Submitting the same
content again is idempotent. Changing the description, cases, or expectations
creates a new immutable version. Cross-version run comparison is rejected.

This prevents a common evaluation mistake: silently editing a difficult case and
then claiming the new system improved on the old benchmark.

## 3. Retrieval metrics

Chapter 6 uses labelled source or chunk identities rather than string similarity
to decide relevance.

### Recall@k

Recall@k is the fraction of labelled relevant identities present in the final
retrieved context:

```text
Recall@k = relevant identities retrieved / all labelled relevant identities
```

It answers whether the generator had access to the evidence it needed. A low
score is primarily a retrieval or context-assembly problem.

### Mean reciprocal rank

Reciprocal rank is `1 / rank` of the first relevant result, or zero when no
relevant result appears. Mean reciprocal rank averages that value across labelled
cases. It rewards putting useful evidence early.

### nDCG@k

Normalized discounted cumulative gain rewards relevant evidence at higher ranks.
The current implementation uses binary relevance because the dataset contract
labels evidence as relevant or not relevant. Graded relevance can be added later
only with a new public contract and migration.

## 4. Answer and citation metrics

Answer match is deliberately deterministic. Every configured phrase must appear
case-insensitively in the public answer. This is useful for stable facts, IDs,
dates, status words, and other acceptance-critical content. It is not a semantic
equivalence judge.

Citation precision asks what fraction of returned citation identities are
labelled relevant. Citation recall asks what fraction of labelled relevant
identities were cited. Status match verifies that the response is `answered`,
`abstained`, or `degraded` exactly as the case permits.

These dimensions are separate on purpose. A fluent answer can have poor
retrieval. A factually correct extract can be a valid degraded response during a
provider outage. An abstention can be the correct outcome when evidence is
missing.

## 5. Latency metrics

Each case records end-to-end application latency. Runs report nearest-rank p50
and p95 values. Absolute latency limits are useful in a controlled environment.
Relative latency gates should include an explicit tolerance because operating
system scheduling, warm caches, and model initialization introduce normal
variation.

Latency is therefore not compared to a baseline by default. A run must opt in
with `max_latency_increase_ms`.

## 6. Provider fault evaluation

The normal success path is not enough. Chapter 6 provides four explicit modes:

| Mode | Behavior under test |
|---|---|
| `none` | configured provider behaves normally |
| `chat_unavailable` | generation raises a provider-unavailable error |
| `empty_output` | provider returns an empty completion on every attempt |
| `malformed_output` | provider returns invalid contract data on every attempt |

Faults are applied through a run-local provider wrapper. The shared application
provider is not mutated, so one evaluation cannot contaminate another request.
Retrieval still runs normally. With sufficient evidence, Chapter 5 should turn
these generation failures into cited `degraded` output after its bounded policy.

## 7. Regression gates

A run may enforce absolute thresholds:

- minimum case pass rate;
- minimum Recall@k;
- minimum answer-match rate;
- minimum status-match rate;
- maximum p95 latency.

When a baseline run is supplied, the gate also measures candidate-minus-baseline
deltas. Quality metrics may decline only by `max_metric_decline`. Latency may
increase only when an explicit `max_latency_increase_ms` permits it.

The run itself can complete while its gate fails. This distinction matters:
execution success means metrics were measured; gate success means the measured
result satisfies the release policy.

## 8. Why no model judge is enabled by default

LLM-as-judge can help evaluate tone, completeness, and semantic equivalence, but
it adds another model, prompt, cost, bias, and version to the experiment. It can
also make a supposedly offline gate depend on provider availability.

Chapter 6 keeps the default path deterministic and network-free. A future judge
should be an explicitly versioned optional metric with saved judge identity,
prompt, rubric, and raw-safe result—not a hidden replacement for labelled
retrieval and citation checks.

## 9. Reading a failure

Start with the per-case result instead of the aggregate:

1. If Recall@k is low, inspect retrieval candidates and context packing.
2. If recall is high but citation recall is low, inspect generation claims and
   citation selection.
3. If citations are correct but answer match fails, inspect synthesis or an
   overly narrow phrase expectation.
4. If status match fails under a fault, inspect repair/degradation policy.
5. If only latency fails, rerun in a controlled warm environment before changing
   architecture.

The saved trace ID connects the case result to the same public answer trace used
by the API and workbench.

## 10. Current boundary

Chapter 6 evaluates fixture and unstructured collection query paths. Structured
table execution and multimodal search have deterministic tests but do not yet
share this dataset schema. Runs execute synchronously and are bounded to 1,000
cases. Multi-worker scheduling, graded relevance, semantic judges, production
traffic sampling, and authorization remain future work.
