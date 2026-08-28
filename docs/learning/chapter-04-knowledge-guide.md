# Chapter 4 Knowledge Guide: Advanced and Structured Retrieval

This chapter turns retrieval from “find the nearest chunks” into an explicit,
inspectable decision system. It is based on all five Chapter 4 source lessons:

1. `11_hybrid_search.md` — sparse/dense retrieval and fusion;
2. `12_query_construction.md` — metadata and graph query construction;
3. `13_text2sql.md` — Text-to-SQL architecture and failure modes;
4. `14_query_rewriting.md` — rewriting, decomposition, HyDE, step-back, and routing;
5. `15_advanced_retrieval_techniques.md` — reranking, compression, and corrective RAG.

Historical sequencing note: the Chapter 4 milestone originally landed before
Chapter 3 and therefore made no claim of persistent pgvector/HNSW, multimodal
retrieval, or embedding-profile lifecycle support. Its dense path embedded
active Chapter 2 chunks on demand. Chapter 3 now supplies those missing
capabilities, and Chapter 4 prefers the compatible persistent index while
retaining the original bridge for collections that have not yet been indexed.

## 1. Why one retrieval method is not enough

A user’s words and a source’s words can relate in several different ways:

- exact identity: `ZXQ-4917`, a function name, product SKU, or person’s name;
- lexical overlap: “deployment policy” appears in both query and source;
- semantic equivalence: “car” and “automobile” differ lexically but mean nearly
  the same thing;
- structured intent: “highest revenue by region” is an aggregation and ordering
  operation, not passage similarity;
- compositional intent: one question asks for several separately supported facts.

Dense similarity alone is weak on exact identifiers and can return a fluent but
wrong thematic match. Lexical search alone has a vocabulary gap. Treating a CSV
as prose destroys its types and aggregation semantics. Treating a multi-part
question as one vector can retrieve evidence for only the dominant clause.

Chapter 4 therefore separates:

```text
understand query
    -> prefilter candidates
    -> retrieve through one or more channels
    -> fuse identities and ranks
    -> rerank a bounded pool
    -> assemble coverage-aware context
    -> assess sufficiency
    -> answer, correct once, or abstain
```

Each arrow has a typed input/output and a trace. That is important: when an
answer is bad, “the RAG system failed” is not a useful diagnosis. We need to know
whether the failure was query interpretation, recall, fusion, reranking,
assembly, or answerability.

## 2. Sparse retrieval

### 2.1 Sparse vectors and the bag-of-words model

A sparse or lexical vector normally has one dimension per vocabulary term.
Most dimensions are zero for any one document. For an eight-dimensional example:

```text
dense storage:  [0, 0, 0, 5, 0, 0, 0, 9]
dictionary:     {3: 5, 7: 9}
COO:            (dimension=8, indices=[3, 7], values=[5, 9])
```

Dictionary/key-value and coordinate-list (COO) forms avoid storing the zeros.
The underlying bag-of-words assumption discards word order and grammar. Each
dimension is interpretable, but it cannot inherently know that two different
dimensions are synonyms.

TF-IDF is the classic weighting family. BM25 adds term-frequency saturation and
document-length normalization:

$$
score(Q,D)=\sum_{q_i\in Q} IDF(q_i)
\frac{f(q_i,D)(k_1+1)}
{f(q_i,D)+k_1(1-b+b\frac{|D|}{avgdl})}
$$

- `IDF(q_i)` discounts terms that occur in many documents.
- `f(q_i,D)` is the term frequency in this document.
- `|D| / avgdl` compares this document’s length to the corpus average.
- `k1` controls saturation: 100 repetitions should not be ten times as valuable
  as 10 repetitions.
- `b` controls how strongly document length is normalized.

Lexical retrieval is cheap, explainable, training-free, and excellent for rare
tokens. It does not understand synonyms unless stemming, expansion, or another
semantic mechanism bridges them. “Out of vocabulary” also needs nuance:
traditional fixed vocabularies may ignore unknown terms, while an inverted index
built from the current corpus can index new literal strings. Modern dense models
usually split unseen words into BPE/WordPiece subwords rather than dropping them.

### 2.2 This repository’s lexical path

PostgreSQL stores a generated English `tsvector` beside every chunk and indexes
it with GIN. Queries use `plainto_tsquery('english', ...)`, `@@`, and
`ts_rank_cd`. Only chunks belonging to a non-deleted document’s active,
completed version in the requested collection can match.

The trace retains:

- lexical rank;
- lexical score;
- literal matched query terms;
- immutable chunk, document, and version identity.

SQLite tests use a deterministic BM25 implementation with the same lifecycle
scope. It is a test/portable fallback, not a claim that SQLite and PostgreSQL
stemming/ranking are identical.

## 3. Dense retrieval

A dense embedding maps text to a relatively low-dimensional array in a learned
continuous semantic space:

```text
[0.89, -0.12, 0.77, ..., -0.45]
```

Individual dimensions are usually not directly interpretable. Their value comes
from geometry: related texts should point in similar directions or occupy nearby
regions. Word2Vec’s often-cited `king - man + woman ≈ queen` illustrates the
idea of learned relational directions; GloVe, BERT-family encoders, and modern
sentence transformers extend the broader representation-learning family.

Dense retrieval can connect paraphrases and related concepts even with little
word overlap. Costs and limits include:

- model initialization and inference;
- model/version dependence;
- less interpretable dimensions;
- possible failures on exact rare identifiers;
- domain mismatch between training data and the local corpus;
- changing embedding models invalidates comparisons with old vectors.

Current collection retrieval first queries the compatible active Chapter 3
profile through pgvector/HNSW. If a collection has no ready index, active chunks
are embedded on demand and cosine-scored in the application. That compatibility
path is correct for a small learning corpus, but it is O(number of active chunks)
per uncached query. Build the persistent index before scaling the corpus.

## 4. Hybrid retrieval and fusion

Hybrid search runs dense and sparse retrieval independently over a wider pool,
then combines their results. It seeks both lexical precision and semantic
generalization.

### 4.1 Weighted score fusion

After normalizing heterogeneous scores to a comparable range:

$$
hybrid = \alpha\,dense + (1-\alpha)\,sparse
$$

This provides direct business control: exact keyword search can dominate product
catalog lookup, while semantics can dominate broad question answering. The hard
part is reliable normalization. Cosine similarity, BM25, inner product, and
cross-encoder logits have different distributions. Min-max normalization is
sensitive to the current candidate pool; fixed calibration can drift across
models and corpora.

### 4.2 Reciprocal Rank Fusion

RRF avoids raw-score comparability and uses rank only:

$$
RRF(d)=\sum_i \frac{1}{c + rank_i(d)}
$$

`c` is commonly 60. A larger value smooths the difference among early ranks; a
smaller value emphasizes the first few positions. A document appearing high in
both lists accumulates two contributions. A document found by only one channel
can still enter the final list.

RRF’s strength is robust zero-shot fusion. Its limitation is exactly what makes
it robust: it throws away the magnitude of the original scores. A barely-first
result and an overwhelmingly-first result contribute the same rank term.

The implementation deduplicates by immutable `chunk_id`, not text or array
position, and records both channel ranks, both channel scores, RRF score, and
fused rank. Stable identity is essential: concatenating result lists can present
the same evidence twice and waste context.

### 4.3 The source’s Milvus/BGE-M3 example

The source demonstrates a different persistence choice that remains useful
conceptually:

- BGE-M3 emits dense and sparse representations in one call.
- A Milvus collection defines a primary key, scalar metadata, one
  `SPARSE_FLOAT_VECTOR`, and one fixed-dimensional `FLOAT_VECTOR`.
- The sparse field uses `SPARSE_INVERTED_INDEX`; the dense field uses an ANN
  index (`AUTOINDEX` in the example) with inner product.
- Data fields must align with schema order during batch insertion.
- `flush()` makes buffered inserts durable/searchable immediately.
- Scalar filters such as category scope both searches.
- Two `AnnSearchRequest` objects are fused by `RRFRanker(k=60)`.

The example’s rank change—one item can be third in each single list but fifth
after fusion—follows from the union: other items can have a very high rank in one
channel and still accumulate enough reciprocal-rank mass. Final order depends on
the ranks of every identity in both lists, not one item’s isolated position.

Local LKE uses PostgreSQL rather than Milvus, but preserves the same separation
between channel-specific retrieval and fusion.

## 5. Reranking

Initial retrieval is optimized for recall: cheaply get a plausible pool.
Reranking is optimized for precision: spend more work on a much smaller set.

### 5.1 RRF as a zero-shot reranker

RRF can itself be described as reranking several lists. It is extremely cheap and
requires no learned model, but sees only ranks and no query-document interaction.

### 5.2 RankLLM

An LLM-based reranker places the query and numbered document summaries in a
prompt and asks the model to return ordered IDs and relevance scores, often as
JSON. It can reason at a high semantic level, but adds generation latency, token
cost, output-validation requirements, and position/order sensitivity. A model
must never be allowed to invent IDs outside the candidate allowlist.

### 5.3 Cross-encoder

A cross-encoder jointly encodes each pair:

```text
[CLS] query [SEP] candidate document [SEP]
```

Because attention crosses the query/document boundary at every layer, it can
model fine interactions that a bi-encoder’s independent vectors miss. The cost
is one full inference per candidate. A typical pipeline retrieves 20–50 cheap
candidates, cross-encodes that bounded pool, and returns a small final top-k.
Common MS MARCO MiniLM/TinyBERT cross-encoders represent the latency/quality
tradeoff.

Local LKE exposes a lazy, local sentence-transformers cross-encoder. It is off by
default so a normal checkout does not unexpectedly download a model. When
enabled, traces retain pre/post ranks, scores, latency, and a top-result score
delta. Measure downstream recall/ranking quality and latency before enabling it
by habit.

### 5.4 ColBERT late interaction

ColBERT is a middle point between a bi-encoder and cross-encoder:

1. independently encode every query and document token;
2. for each query token, find its maximum similarity to any document token
   (`MaxSim`);
3. sum those maxima into the document score.

Document token embeddings can be precomputed, while token-level late interaction
retains more detail than comparing one pooled vector. It is more expensive than
single-vector ANN search and often cheaper than full joint cross-encoding.

| Method | Interaction | Typical cost | Main use |
|---|---|---:|---|
| RRF | ranks only | very low | merge heterogeneous recall |
| RankLLM | generative semantic judgment | medium/high | high-value small pools |
| Cross-encoder | full query-document attention | high per candidate | top-k precision |
| ColBERT | token-level late interaction | medium | fine-grained retrieval/rerank |

## 6. Metadata query construction

Semi-structured documents attach fields such as source, author, date, category,
page, and duration. A self-query retriever asks an LLM to split a natural-language
request into:

- a semantic query string; and
- a structured metadata filter (and sometimes limit/order intent).

The source’s LangChain flow is:

1. flatten messy loader metadata into stable primitive fields;
2. define every field with `AttributeInfo(name, description, type)`;
3. give those descriptions and document-content semantics to a query constructor;
4. validate the resulting generic structured query;
5. translate it with a vector-store-specific translator;
6. execute semantic search under the translated filter.

The Bilibili example flattens nested owner/stat metadata to `title`, `author`,
`source`, integer `view_count`, and integer `length`. Clear types and descriptions
are not cosmetic: they are the model’s schema. Temperature zero improves
repeatability for plan generation but does not make the plan correct.

Why does “shortest video” return the wrong answer in the source example? The
generated plan is essentially blank semantic search plus `limit=1`; it does not
express `ORDER BY length ASC`. Limiting an unordered similarity result is not a
minimum operation. The later JSON instruction example fixes this by producing an
allowlisted `{sort_by: length, order: asc}` plan executed by application code.

Local LKE makes the safety boundary explicit:

- fields are a `Literal` allowlist;
- operators are an enum (`eq`, `ne`, `in`, `contains`, `gt`, `gte`, `lt`, `lte`);
- Pydantic rejects unknown fields and malformed value shapes;
- SQLAlchemy expressions, not string concatenation, apply the filter;
- filtering occurs before dense and lexical ranking;
- dropping filters is forbidden unless `allow_unfiltered_fallback=true`;
- the interpreted filter is returned in the trace.

This design can accept a model-produced JSON object, but the model never receives
an SQL execution capability.

## 7. Text-to-Cypher and graph routing

Cypher expresses graph patterns such as:

```cypher
(:Person {name: "Tomaz"})-[:LIVES_IN]->(:Country {name: "Slovenia"})
```

A Text-to-Cypher chain receives the graph schema, maps a natural-language
question to a graph query, executes it, and optionally verbalizes the rows.
Relationship direction, labels, and property names are schema-dependent, so a
strong model and schema grounding matter. The same production rule applies as
SQL: generated graph text is not inherently safe. Constrain labels,
relationships, operations, scope, cardinality, and read-only execution before
running it.

Chapter 4 does not add a graph database; it records graph routing as a future
explicit path rather than misrouting graph questions to vector search.

## 8. Text-to-SQL: concepts and hardened implementation

### 8.1 Core challenges

- **Schema hallucination:** the model invents tables or columns.
- **Relationship errors:** valid-looking joins use the wrong keys or cardinality.
- **Business ambiguity:** “last month,” “customer,” and “spend” need local
  definitions.
- **Dialect mismatch:** SQLite, PostgreSQL, and MySQL differ.
- **Safety:** generated SQL can read too much, consume resources, or mutate data.
- **Result semantics:** a syntactically valid query can still answer the wrong
  business question.

### 8.2 Grounding a SQL planner

The source recommends progressively richer grounding:

1. exact DDL, including types, keys, and foreign-key relationships;
2. a small set of high-quality question/SQL examples;
3. a retrieved schema knowledge base containing DDL, table/column descriptions,
   business synonyms, and complex Q-SQL examples;
4. execution feedback and bounded correction.

Its reference architecture stores three knowledge types—`ddl`, `qsql`, and
`description`—in one vector collection, retrieves relevant items with BGE-M3,
then orders the prompt context as schema, descriptions, and examples. A
coordinator retrieves knowledge, generates SQL, executes it, feeds database
errors back for correction, caps retries, and limits returned rows. Returning
column names plus row dictionaries preserves structure for downstream use.

This architecture is intentionally unwrapped to make failures attributable to
knowledge retrieval, generation, or execution. High-level frameworks are useful,
but opaque internal translation made the earlier “shortest video” error harder
to diagnose.

### 8.3 Why Local LKE never executes model SQL

The educational source’s string-SQL approach illustrates the architecture, but
checking `startswith('SELECT')` and appending `LIMIT` is not a sufficient security
boundary. Comments, CTEs, multiple statements, nested limits, dialect features,
and resource-heavy reads complicate textual inspection.

Local LKE instead uses this pipeline:

```text
natural-language question
  -> model JSON (optional)
  -> Pydantic StructuredQueryPlan
  -> table/column/type allowlist validation
  -> SQLAlchemy Core Select construction
  -> read-only transaction + timeout + hard limit
  -> rows + parameterized SQL preview + provenance
```

The plan can express:

- projections;
- typed filters;
- grouping;
- `count`, `sum`, `avg`, `min`, and `max`;
- ordering by columns or aggregate aliases;
- bounded result limit.

There is no `raw_sql` field, and extra fields are forbidden. Unknown columns,
non-numeric `sum`/`avg`, and ungrouped aggregate projections are rejected. Values
remain bound parameters, so text resembling SQL remains data. PostgreSQL runs
the statement in a read-only transaction with `statement_timeout`; every backend
gets a hard application limit.

### 8.4 CSV as structured provenance

CSV ingestion validates:

- `.csv` filename and upload size;
- UTF-8 (including optional BOM);
- nonblank, case-insensitively unique headers;
- fixed maximum columns and rows;
- consistent row width and at least one data row.

Headers become safe unique identifiers. Each column is inferred conservatively
as boolean, integer, float, ISO date, or text; blanks make it nullable. Each
generated description states the original header and inferred type. A physical
table name is generated from a UUID, never from user input. Internal row number
and source-version columns preserve provenance. A metadata record links table,
collection, logical document, immutable version, content hash, schema, and row
count.

## 9. Query translation

Raw questions are not always good retrieval probes. Query translation reduces
the mismatch between user language and source language.

### 9.1 Direct structured instructions

Instead of free-form rewriting, an LLM can emit a small JSON instruction. The
source fixes the “shortest video” case by allowlisting `length`/`view_count` and
`asc`/`desc`. Application code performs the sort. This is more inspectable than
hoping a generic retriever infers an ordering primitive.

### 9.2 Multi-query decomposition

A complex question can contain several intents. Multi-query retrieval:

1. decomposes it into simpler subquestions;
2. retrieves for each subquestion, ideally in parallel;
3. merges and identity-deduplicates candidates;
4. assembles context that covers every required part;
5. reports missing coverage instead of silently answering only one part.

More queries are not automatically better. They increase embedding/search work,
candidate duplication, and the chance of drifting from the original intent.
Local LKE caps fan-out at four and never recursively decomposes generated text.

### 9.3 Step-back prompting

Step-back prompting first asks for a more general principle, then applies that
principle to the original detail. In the source’s ideal-gas example, the system
first retrieves `PV=nRT`, then reasons about doubling temperature and increasing
volume eightfold to obtain one-quarter pressure.

The retrieved general background can improve reasoning, but it must not replace
evidence for the concrete question. Local LKE labels the step-back text as a
retrieval probe; generated text never enters the evidence set by itself.

### 9.4 HyDE

Hypothetical Document Embeddings address the query/document style gap:

1. a generator writes a plausible ideal answer passage;
2. an encoder embeds that passage;
3. retrieval finds real documents near the hypothetical passage.

This turns short-query-to-document matching into document-style-to-document
matching and needs no retriever fine-tuning. The hypothetical text can be
factually wrong. It is a search instrument, never a source. Only retrieved real
chunks may support the final answer. Local LKE exposes a named, bounded HyDE-style
probe and records it in the transform trace.

## 10. Query routing

Routing chooses a data source, component, or prompt based on intent.

- **Data source:** product vectors, order-history SQL, or an investment graph.
- **Component:** cheap FAQ retrieval versus a more capable agent/tool path.
- **Prompt:** math reasoning, code generation, or domain-specific answering.

### 10.1 LLM classification routing

Define closed route labels and descriptions, ask the model for one label, validate
it, then let application code dispatch. In LangChain Expression Language (LCEL),
`prompt | llm | parser` forms the classifier pipeline, while `RunnableBranch`
acts like allowlisted `if/elif/else` dispatch. LCEL can also preserve the original
question while computing the topic in parallel.

### 10.2 Embedding-similarity routing

Embed detailed route descriptions in advance, embed the user query, choose the
highest cosine similarity, and dispatch through an application-owned route map.
It is usually faster than LLM classification but can confuse routes with similar
vocabulary and needs an uncertainty threshold/default route.

LlamaIndex models data sources as `QueryEngineTool` objects with descriptive
text. `RouterQueryEngine` and selectors such as `LLMSingleSelector` or
`PydanticSingleSelector` select tools. Its “semantic routing” is primarily LLM
understanding of tool descriptions rather than a separate raw cosine router.

Local LKE records four explicit intent routes: simple lookup, broad synthesis,
multi-part, and structured. The application, not generated code, owns dispatch.

## 11. Contextual compression

Retrieved chunks often contain a relevant sentence surrounded by noise. Passing
all of it increases tokens, latency, cost, and distraction. Compression means:

- **content extraction:** keep only relevant sentences/passages; or
- **document filtering:** keep/drop the whole candidate.

LangChain’s `ContextualCompressionRetriever` wraps a base retriever and a
`DocumentCompressor`. Examples include:

- `LLMChainExtractor` for relevant-span extraction;
- `LLMChainFilter` for whole-document decisions;
- `EmbeddingsFilter` for a cheaper similarity threshold.

`DocumentCompressorPipeline` composes transformations in order. The source shows
a custom ColBERT `BaseDocumentCompressor` followed by `LLMChainExtractor`:
retrieve 20, rerank, then extract. Studying the base interface and existing
compressors is the reliable way to add an unsupported reranker.

The example’s repeated output can arise because overlapping source chunks carry
the same sentence into the candidate pool and extraction preserves it more than
once. Deduplicate by stable identity and normalized/near-duplicate content after
expansion/compression, while retaining the best-scoring provenance.

LlamaIndex’s `SentenceEmbeddingOptimizer` performs sentence-level postprocessing:
split each node into sentences, compare each sentence to the query, and retain
the highest-similarity sentences.

Local LKE avoids generative extraction in this milestone. It performs exact and
high-Jaccard near-duplicate removal, per-document diversity caps, and token-budget
truncation. The context manifest makes every loss visible.

## 12. Coverage-aware context assembly

Candidate order is not the same as usable context order. Assembly must answer:

- Does every subquery have evidence?
- Are several chunks copies or overlapping windows?
- Is one document crowding out independent corroboration?
- Will a long source consume the entire prompt budget?
- Was evidence omitted or truncated, and why?

Local LKE uses coverage-first packing:

1. reserve the best candidate for each subquery;
2. add optional support in final-rank order;
3. deduplicate exact/near-identical token sets;
4. cap one document at three chunks;
5. enforce per-source and total token budgets;
6. truncate only when a useful final chunk can partially fit;
7. restore source/ordinal order for the final evidence block.

Every candidate appears once in the manifest as `included`, `excluded`, or
`truncated`, with reason, token count, version, locator, and covered subqueries.
This is how an evidence-loss bug becomes observable rather than anecdotal.

## 13. Corrective RAG and answerability

Corrective RAG (C-RAG) rejects the assumption that retrieved material is always
useful. The source’s full pattern is:

1. **Retrieve** candidate documents.
2. **Assess** each as correct, incorrect, or ambiguous.
3. **Act**:
   - correct: split into knowledge strips, filter noise, recombine, answer;
   - incorrect: rewrite and obtain outside knowledge, commonly web search;
   - ambiguous: obtain more evidence before answering.

Graph-style orchestration such as LangGraph is convenient for conditional paths
and loops, but every loop requires an explicit termination rule.

Local LKE is local-first and deliberately prohibits web fallback. Its
deterministic sufficiency score combines:

- query-term coverage;
- required-subquery coverage;
- normalized evidence strength.

It additionally requires some lexical term support, which is conservative and
reduces accidental hash/semantic matches in the compatibility bridge. When the
initial evidence is insufficient, exactly one alternate retrieval is allowed:
dense switches to hybrid, or hybrid switches to dense with a step-back probe.
The higher-scoring result is retained. If it still fails, the system abstains
without citations and records threshold, features, correction strategy, and
final reason.

The threshold is a policy, not a universal truth. Calibrate it on labeled local
answerable and deliberately unanswerable questions. Track false answers and
false abstentions separately.

## 14. Evaluation

Measure stages independently:

- Recall@k: does the candidate pool contain the labeled evidence?
- MRR/nDCG: how early and how consistently is relevant evidence ranked?
- exact-identifier recall: does lexical search rescue rare tokens?
- subquery coverage: does every required aspect reach final context?
- context precision: how much included context is actually useful?
- answerable precision/recall: do supported questions answer and unsupported ones
  abstain?
- latency: dense embedding, lexical query, fusion, reranker, assembly, generation;
- cost/resources: candidate count, model calls, and prompt tokens.

Reranking “improves” retrieval only if a labeled metric improves or stays equal
within an acceptable latency budget. A larger model score is not itself evidence
of better retrieval.

## 15. Code map

| Concept | Implementation |
|---|---|
| PostgreSQL English FTS + GIN | `migrations/versions/20260827_02_chapter_04_retrieval.py` |
| lifecycle-scoped chunks, metadata predicates, BM25 fallback | `storage/repository.py` |
| typed plans/traces/manifests/structured contracts | `models.py` |
| bounded normalization, decomposition, step-back, HyDE | `retrieval/planning.py` |
| dense scoring, RRF, correction, answerability | `retrieval/service.py` |
| local cross-encoder boundary | `retrieval/reranking.py` |
| CSV inference and compiled read-only queries | `retrieval/structured.py` |
| strategy/query/structured API | `web/api.py` |
| comparison lab and structured panel | `web/workbench.py` |

## 16. Source-coverage checklist

The guide explicitly preserves the source material’s important concepts:

- sparse representation, COO/dictionary examples, TF-IDF/BM25 parameters;
- dense geometry, semantic matching, OOV/subword nuance;
- RRF and normalized weighted fusion;
- BGE-M3/Milvus schema, indexes, filter, flush, requests, and rank behavior;
- self-query field descriptions, flattening, translators, temperature, and the
  minimum-vs-limit failure;
- Cypher workflow;
- Text-to-SQL hallucination, schema grounding, examples, RAG knowledge types,
  correction, row limits, and inspectability;
- JSON query instructions, multi-query, step-back, HyDE;
- data/component/prompt routing, LLM/embedding routing, LCEL, and LlamaIndex;
- RRF, RankLLM, cross-encoder, and ColBERT tradeoffs;
- extraction/filter compression, LangChain compressor pipeline, custom
  compressor boundary, and LlamaIndex sentence optimization;
- C-RAG retrieve/assess/act branches and the local no-web adaptation.

That source coverage is only the conceptual baseline. The repository adds the
production controls the examples need: active-version scope, immutable
provenance, parameterized plans, hard budgets, bounded retries, complete traces,
and deterministic tests.
