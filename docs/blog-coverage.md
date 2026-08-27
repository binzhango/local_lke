# RAG Issue Coverage

Chapters 2-5 move ten canonical reliability issues beyond the Chapter 1
naive-RAG baseline. “Mitigated” means a concrete local control and deterministic
test exist; it does not mean the underlying research problem is eliminated.

| # | Blog issue | Status after Chapter 4 | Evidence and residual risk |
|---:|---|---|---|
| 1 | Corpus lacks the answer | Mitigated | Deterministic sufficiency, one bounded alternate retrieval, and tested abstention prevent unsupported generation. Threshold calibration remains corpus-specific. |
| 2 | Answer missing from top results | Mitigated | Persistent HNSW dense recall plus PostgreSQL lexical recall, identity-safe RRF, and optional measured cross-encoding rescue rare IDs and semantic variants. Labelled Recall@k fixtures guard the dense profile. |
| 3 | Evidence lost during assembly | Mitigated | Sentence windows and parent expansion retain broader evidence; best-child dedupe, source-aware packing, token budgets, and complete manifests expose every decision. Calibration remains corpus-specific. |
| 4 | Complex content is corrupted | Mitigated for approved fixtures | Markdown headings/source lines, PDF pages/categories/tables, and validated PNG/JPEG/WebP metadata are preserved and golden-tested. OCR, columns, and unusual layouts remain residual risks. |
| 5 | Chunk granularity mismatch | Mitigated | Fine sentence/chunk candidates pass lookup fixtures; section/parent expansion passes broad-answer fixtures without duplicate parents and retains triggering-child provenance. |
| 6 | Multi-part answer incomplete | Mitigated | Fixed-fan-out decomposition reserves evidence per subquery and reports explicit subquery coverage before answering. |
| 7 | Structured data treated as prose | Mitigated | CSV schemas retain types and version provenance; model JSON becomes an allowlisted SQLAlchemy `Select` under read-only/time/row limits. |
| 8 | Wrong output format | Mitigated | Conversational and allowlisted JSON schemas validate through Pydantic; malformed, missing, and wrong-type outputs receive one bounded repair, then a typed extractive degradation. Provider-specific native-schema quirks remain model-dependent. |
| 9 | Ingestion too costly or slow | Mitigated locally | Content/pipeline hashes make unchanged re-ingestion perform zero parser/chunk writes; stable profile-scoped node IDs make unchanged reindexing perform zero embedding calls. Large distributed rebuilds remain out of scope. |
| 10 | Provider fallback differs | Partially mitigated | Chapter 5 detects generation connection, stream, empty-output, and schema failures and returns validated degradation. Multi-provider capability profiles and broader fault injection remain Chapter 6 work. |
| 11 | Security/privacy risks | Partially mitigated | Upload checks, safe structured plans, encoded prompt boundaries, application-owned citation/tool policy, and escaped Gradio rendering protect the local path. Authentication, multi-user ACLs, and adversarial evaluation remain out of scope. |
| 12 | No evaluation control plane | Open | The deterministic suite covers ingestion controls; a user-facing evaluation plane begins in Chapter 6. |
