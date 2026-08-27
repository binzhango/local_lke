# RAG Issue Coverage

Chapters 2-7 move the canonical reliability issues beyond the Chapter 1
naive-RAG baseline. “Mitigated” means a concrete local control and deterministic
test exist; it does not mean the underlying research problem is eliminated.

| # | Blog issue | Status after Chapter 7 | Evidence and residual risk |
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
| 10 | Provider fallback differs | Mitigated for the local contract | Chapter 5 returns validated degradation; Chapter 6 records the configured provider capability profile and deterministically injects unavailable, empty, and malformed generation behavior. Live cross-vendor conformance remains provider-specific. |
| 11 | Security/privacy risks | Mitigated for the governed local API | Earlier upload, plan, prompt, citation, and rendering controls remain. Chapter 7 adds opt-in bearer authentication, collection-scoped owner/editor/viewer permissions, nested-resource authorization, admin-only evaluation, and metadata-only audit evidence. Token provisioning is static configuration; TLS, enterprise identity, adversarial evaluation, and remote deployment hardening remain external. |
| 12 | No evaluation control plane | Mitigated | Immutable labelled datasets, persisted per-case runs, retrieval/answer/citation/status/latency metrics, fault scenarios, same-version comparisons, and regression gates are exposed through API and Gradio. Semantic judges and production sampling remain future work. |
