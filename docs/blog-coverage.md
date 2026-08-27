# RAG Issue Coverage

Chapter 2 moves four canonical reliability issues beyond the Chapter 1
naive-RAG baseline. “Mitigated” means a concrete local control and deterministic
test exist; it does not mean the underlying research problem is eliminated.

| # | Blog issue | Status after Chapter 2 | Evidence and residual risk |
|---:|---|---|---|
| 1 | Corpus lacks the answer | Open | Answerability and corrective retrieval begin in Chapter 4. |
| 2 | Answer missing from top results | Open | Hybrid recall and reranking begin in Chapter 4. |
| 3 | Evidence lost during assembly | Open | Parent expansion and context packing begin in Chapters 3–4. |
| 4 | Complex content is corrupted | Partially mitigated | Markdown headings, source lines, PDF pages/categories, and table elements are preserved and golden-tested. OCR, columns, and unusual layouts still vary by host and document. |
| 5 | Chunk granularity mismatch | Partially mitigated | Recursive, heading-aware, and experimental sentence-semantic strategies preserve parent provenance. Retrieval impact is not measured until Chapter 3. |
| 6 | Multi-part answer incomplete | Open | Decomposition and coverage validation begin in Chapter 4. |
| 7 | Structured data treated as prose | Open | Safe structured CSV ingestion begins in Chapter 4. |
| 8 | Wrong output format | Open | Output contracts and repair begin in Chapter 5. |
| 9 | Ingestion too costly or slow | Partially mitigated | Content/pipeline hashes make unchanged re-ingestion perform zero parser and chunk writes; batched embedding remains for Chapter 3. |
| 10 | Provider fallback differs | Open | Capability profiles and fault injection begin in Chapter 6. |
| 11 | Security/privacy risks | Partially mitigated | Loopback binding plus extension/MIME/signature, UTF-8, path, size, encrypted-PDF, and malformed-PDF checks protect the local upload boundary. Authentication and multi-user ACLs remain out of scope. |
| 12 | No evaluation control plane | Open | The deterministic suite covers ingestion controls; a user-facing evaluation plane begins in Chapter 6. |
