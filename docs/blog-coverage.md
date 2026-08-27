# RAG Issue Coverage

Chapter 4 moves nine canonical reliability issues beyond the Chapter 1
naive-RAG baseline. “Mitigated” means a concrete local control and deterministic
test exist; it does not mean the underlying research problem is eliminated.

| # | Blog issue | Status after Chapter 4 | Evidence and residual risk |
|---:|---|---|---|
| 1 | Corpus lacks the answer | Mitigated | Deterministic sufficiency, one bounded alternate retrieval, and tested abstention prevent unsupported generation. Threshold calibration remains corpus-specific. |
| 2 | Answer missing from top results | Mitigated | Dense plus PostgreSQL lexical recall, identity-safe RRF, and optional measured cross-encoding rescue rare IDs and semantic variants. Persistent ANN is absent because Chapter 3 was skipped. |
| 3 | Evidence lost during assembly | Mitigated | Coverage-first packing, exact/near-duplicate removal, source diversity, token budgets, and complete manifests expose every context decision. Parent/window expansion remains a skipped Chapter 3 feature. |
| 4 | Complex content is corrupted | Partially mitigated | Markdown headings, source lines, PDF pages/categories, and table elements are preserved and golden-tested. OCR, columns, and unusual layouts still vary by host and document. |
| 5 | Chunk granularity mismatch | Partially mitigated | Recursive, heading-aware, and experimental sentence-semantic strategies preserve parent provenance. Multi-granularity/parent-window retrieval remains absent because Chapter 3 was skipped. |
| 6 | Multi-part answer incomplete | Mitigated | Fixed-fan-out decomposition reserves evidence per subquery and reports explicit subquery coverage before answering. |
| 7 | Structured data treated as prose | Mitigated | CSV schemas retain types and version provenance; model JSON becomes an allowlisted SQLAlchemy `Select` under read-only/time/row limits. |
| 8 | Wrong output format | Open | Output contracts and repair begin in Chapter 5. |
| 9 | Ingestion too costly or slow | Partially mitigated | Content/pipeline hashes make unchanged re-ingestion perform zero parser and chunk writes; on-demand dense embedding is deliberately small-corpus only because Chapter 3 was skipped. |
| 10 | Provider fallback differs | Open | Capability profiles and fault injection begin in Chapter 6. |
| 11 | Security/privacy risks | Partially mitigated | Upload checks plus allowlisted structured plans, bound values, read-only transactions, and time/row limits protect local boundaries. Authentication and multi-user ACLs remain out of scope. |
| 12 | No evaluation control plane | Open | The deterministic suite covers ingestion controls; a user-facing evaluation plane begins in Chapter 6. |
