# RAG Issue Coverage

Chapter 1 establishes a measured naive-RAG baseline. No issue is claimed as
resolved or mitigated yet; subsequent chapters must provide tests and evidence
before changing a status.

| # | Issue | Status after Chapter 1 | Baseline observation |
|---:|---|---|---|
| 1 | Document ingestion quality | Open | Only two controlled fixtures are loaded. |
| 2 | Chunking quality | Open | One recursive strategy is observable but not evaluated. |
| 3 | Embedding suitability | Open | One small local embedding model is used. |
| 4 | Retrieval relevance | Open | Vector top-k is traced without a relevance benchmark. |
| 5 | Missing lexical retrieval | Open | There is no keyword or hybrid search. |
| 6 | Missing reranking | Open | Similarity order is passed directly to generation. |
| 7 | Context construction | Open | Evidence is delimited but not optimized. |
| 8 | Hallucination and grounding | Open | Citations are attached, but faithfulness is not measured. |
| 9 | Abstention reliability | Open | The schema supports abstention; policy quality is unevaluated. |
| 10 | Evaluation coverage | Open | Known-answer tests are not a full RAG evaluation set. |
| 11 | Observability and cost | Open | Stage timings exist; token and resource metrics do not. |
| 12 | Knowledge relationships | Open | There is no graph retrieval or entity model. |

