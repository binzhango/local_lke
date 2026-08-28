# Chapter 3 Learning Notes: Embeddings, Vector Databases, Multimodal Search, and Index Optimization

Chapter 2 produced safe, versioned chunks. Chapter 3 explains how those chunks
become searchable vectors, how vector indexes trade exactness for speed, how text
and images can share a retrieval space, and how small retrieval units can expand
into useful generation context.

These notes cover the complete conceptual content of the five source sections:

1. Vector embeddings
2. Multimodal embeddings
3. Vector databases and FAISS
4. Milvus concepts and multimodal practice
5. Sentence-window, structured, and recursive index optimization

The source guide uses BGE, Visualized-BGE, FAISS, Milvus, LangChain, and
LlamaIndex examples. This repository implements the same principles with local
BGE text embeddings, an optional local CLIP-compatible model, PostgreSQL 18,
and pgvector. A concept being explained here does not imply that every source
framework or database is installed.

## 1. The Chapter 3 pipeline

```text
active versioned chunks
        |
        v
embedding profile -- model/revision/dimension/prefix/normalization contract
        |
        v
sentence + chunk + section vectors
        |
        v
pgvector exact/ANN search -- small child matches
        |
        v
window/parent expansion -- deduplication -- token budget
        |
        v
retrieval candidates and final context
```

Indexing and retrieval are different workloads:

- Indexing is mostly offline. It embeds documents, persists progress, builds an
  index, and activates a complete namespace.
- Retrieval is online. It embeds one query, searches the compatible namespace,
  expands context, and returns ranked evidence.

Both sides must use the same embedding contract. Vectors from incompatible
models or dimensions do not belong in one comparison space.

## 2. What an embedding is

An **embedding** converts a complex object—text, image, audio, or video—into a
fixed-length dense numeric vector.

```text
data object -> embedding model -> [0.16, 0.29, -0.88, ...]
```

The output is not intended to reproduce every input detail. It is a learned
representation whose geometry makes selected relationships measurable. For a
retrieval model, semantic neighbors should usually be close and unrelated
objects should usually be far apart.

Three things must not be conflated:

- The source object contains the original information.
- The model defines what relationships its vector space preserves.
- The vector is a compressed coordinate in that learned space.

The vector dimension is fixed for a model, commonly hundreds or thousands of
numbers. A higher dimension can represent more distinctions, but costs more
storage, memory bandwidth, index-build time, and query computation. Dimension
alone is not a quality score.

## 3. Similarity and distance geometry

### 3.1 Cosine similarity

For vectors $a$ and $b$:

```text
cosine(a, b) = (a · b) / (||a|| ||b||)
```

Cosine similarity measures their angle. A value near `1` means their directions
are strongly aligned; `0` means orthogonal; a negative value means opposed
directions. Semantic retrieval commonly ranks larger cosine similarity first.

pgvector's cosine operator returns a **distance**:

```text
cosine distance = 1 - cosine similarity
```

Therefore the SQL query orders the smallest distance first, while the API
converts it back into a larger-is-better score.

### 3.2 Dot product

```text
dot(a, b) = sum(a[i] * b[i])
```

If both vectors are normalized to unit length, dot product equals cosine
similarity. Without normalization, magnitude affects the result. A system must
match the distance metric to the model's training and encoding instructions.

### 3.3 Euclidean distance

```text
L2(a, b) = sqrt(sum((a[i] - b[i])^2))
```

Euclidean distance measures straight-line distance. Smaller is closer. For
normalized vectors, cosine, dot-product, and L2 rankings are closely related,
but they are not interchangeable for arbitrary unnormalized vectors.

### 3.4 Why normalization is part of the index contract

Normalizing vectors makes every vector length one. This prevents document
magnitude from silently changing cosine-equivalent dot-product ranking. It also
means a later model configuration that disables normalization is not the same
profile, even if the model name and dimension remain unchanged.

## 4. Embeddings in the RAG loop

The source chapter describes four steps.

1. **Offline index construction:** split documents, embed every chunk, and store
   vectors beside identifiers and metadata.
2. **Online query encoding:** embed the user's question with the same compatible
   model contract.
3. **Similarity search:** compare the query vector with indexed document vectors.
4. **Top-k recall:** return the nearest chunks as candidate context for the LLM.

Embedding quality controls which candidates are available downstream. A strong
generator cannot cite a relevant passage that retrieval never returned. A poor
embedding model can pollute context with superficially related but factually
irrelevant passages.

The same model is necessary but not sufficient. Query and document encodings may
also require different instruction prefixes. For example, BGE retrieval models
may recommend a query instruction while documents receive no prefix. Applying a
query prefix to documents—or omitting a required query instruction—changes the
space and can reduce recall.

This repository records:

- model ID;
- immutable or explicit revision;
- vector dimension;
- normalization setting;
- document prefix;
- query prefix;
- modality.

Those fields define an **embedding profile**. Collections activate profiles
atomically so incompatible namespaces never mix.

## 5. How text embedding technology evolved

### 5.1 Static word embeddings

**Word2Vec (2013)** and **GloVe (2014)** assign one fixed vector to each word.

Word2Vec learns from local context windows:

- CBOW predicts a center word from surrounding words.
- Skip-gram predicts surrounding words from a center word.

Its learned geometry made analogical operations such as
`king - man + woman ≈ queen` famous. GloVe adds global word-word co-occurrence
statistics.

The central limitation is context independence. The word “apple” receives the
same vector in a sentence about fruit and a sentence about the company. A fixed
lexical vector cannot directly represent polysemy.

### 5.2 Contextual embeddings

The Transformer introduced self-attention, allowing a token representation to
depend on every relevant token in the sequence. BERT uses stacked bidirectional
Transformer encoders, so a word obtains different representations in different
contexts.

The conceptual advance is not simply “larger word vectors.” It is dynamic
contextualization: the representation is computed for this occurrence in this
sequence.

### 5.3 BERT's self-supervised objectives

The source explains two original BERT tasks.

**Masked Language Modeling (MLM):**

- choose roughly 15% of input tokens;
- hide or alter them using the masking procedure;
- predict the original tokens from bidirectional context.

MLM forces the encoder to learn deep token-context relationships from unlabeled
text.

**Next Sentence Prediction (NSP):**

- construct sentence pairs `A` and `B`;
- in half the examples, `B` is the real next sentence;
- in the other half, `B` is sampled elsewhere;
- predict `IsNext` versus `NotNext`.

NSP aimed to teach inter-sentence coherence. Later work such as RoBERTa found
that the original task could be too easy or counterproductive. Many later models
changed or removed it. SBERT-style retrieval models also require objectives that
directly produce useful sentence-level geometry; raw token pretraining alone is
not a guarantee of good nearest-neighbor retrieval.

### 5.4 Metric learning and contrastive learning

Modern retrieval embeddings optimize relative relationships.

**Metric learning** uses related pairs—question/answer, title/body, duplicate
queries—and trains positives to rank closer than negatives. The important goal
is the ordering, not forcing every positive score to exactly `1` and every
negative to exactly `0`.

**Contrastive learning** often uses an anchor, positive, and negative:

```text
distance(anchor, positive) < distance(anchor, negative)
```

Hard negatives are especially valuable because they resemble the query but do
not supply the correct evidence. Poor negative sampling can produce a model that
looks good on easy comparisons and fails on real retrieval ambiguity.

## 6. What RAG demands from an embedding model

RAG introduced needs beyond generic word representation.

### 6.1 Domain adaptation

Legal, medical, scientific, and internal business corpora contain specialized
terms and relationships. A general model may need domain fine-tuning or explicit
retrieval instructions. Public leaderboard strength does not prove private
corpus performance.

### 6.2 Multiple granularities and modalities

A knowledge system may embed:

- short sentences for precise fact lookup;
- chunks for normal passages;
- sections or documents for broad synthesis;
- source code;
- tables or table descriptions;
- images and text in a shared multimodal space.

One model may not be optimal for all of them. Chapter 3 therefore separates the
text profile from the multimodal profile.

### 6.3 Efficiency and hybrid retrieval

Model size and dimension affect indexing throughput, storage, memory, and
latency. Dense retrieval also complements rather than eliminates lexical search.
BGE-M3 is cited as an example of the trend toward dense, sparse, and multi-vector
functions. Chapter 4 combines the Chapter 3 dense index with PostgreSQL lexical
search rather than assuming one retrieval signal is universally sufficient.

## 7. Selecting a text embedding model

### 7.1 MTEB is a starting point

The Massive Text Embedding Benchmark reports many tasks, including retrieval,
classification, clustering, and ranking. Its visualization encodes several
tradeoffs:

- horizontal position: number of model parameters;
- vertical position: average task score;
- bubble size: embedding dimension;
- bubble color: maximum token length.

For RAG, inspect retrieval performance for the relevant language and domain.
An average score can hide weak retrieval results.

### 7.2 Selection dimensions

Evaluate at least:

| Dimension | Question |
|---|---|
| Task | Is retrieval—not only classification—strong? |
| Language | Does it support every corpus and query language? |
| Model size | Does it fit local RAM/VRAM and latency needs? |
| Vector dimension | What storage and ANN cost does it create? |
| Maximum tokens | Is it compatible with the chunking policy? |
| Publisher/revision | Is the artifact trustworthy and reproducible? |
| Cost | What are download, hardware, indexing, and operational costs? |

The chunk size must stay within the embedding tokenizer's maximum length. A
character count is only an approximation of tokens.

### 7.3 Private evaluation is the final selector

The source recommends an iterative process:

1. Choose several plausible baseline models.
2. Build a private evaluation set of real questions and labelled relevant chunks.
3. Measure retrieval accuracy and relevance.
4. Change one variable at a time—model, prefix, normalization, or chunking.
5. Select the model that wins on the real workload and deployment budget.

Chapter 3 records a deterministic Recall@k fixture as a baseline. Chapter 6 will
provide the full evaluation control plane.

## 8. Why multimodal embeddings are needed

Text-only embeddings cannot compare a sentence with an image because the two
encoders normally produce unrelated spaces. The source calls this separation a
“modality wall.”

A multimodal embedding model maps different data types into one shared space:

```text
"a running dog" --text encoder--+
                                   +--> nearby shared vectors
running-dog.jpg --image encoder---+
```

The central technical problem is **cross-modal alignment**: teaching the model
that semantically matching objects from different modalities should be close.

## 9. CLIP's dual-encoder pattern

CLIP uses two encoders:

- a text encoder;
- an image encoder, often a ResNet or Vision Transformer family architecture.

Both project outputs into the same vector dimension. During training, a batch
contains matching image-text pairs. The objective increases similarity for the
correct diagonal pairs and decreases it for the incorrect cross-pairs.

```text
             text 1  text 2  text 3
image 1        +       -       -
image 2        -       +       -
image 3        -       -       +
```

This large-scale contrastive training enables **zero-shot classification**. To
classify an image as a cat, dog, or car, encode prompts such as “a photo of a
cat” and select the closest text vector. Classification becomes retrieval.

Dual encoders are efficient because corpus images can be embedded offline.
Their limitation is that independent encoders may miss fine-grained token/patch
interactions that a heavier cross-modal model could inspect.

## 10. Beyond CLIP and Visualized-BGE

The source mentions several directions:

- BLIP-family models emphasize richer image-text understanding and generation.
- ALIGN demonstrated large-scale learning from noisy web image-text pairs.
- `bge-visualized-m3` adds visual capability to a BGE-M3 text foundation.

The “M3” description highlights:

- multilingual text support across more than 100 languages;
- multiple retrieval functions, including dense and multi-vector modes;
- multiple text granularities, up to long inputs such as 8192 tokens on the text
  side for the cited model family.

Visualized-BGE extracts image patch tokens, maps them into the text model's token
dimension, and jointly processes image and text tokens through the BGE
Transformer. It can encode:

- text only;
- image only;
- an image and text together.

When vectors are normalized, matrix multiplication computes cosine similarity.
The source example compares image/image, image/joint, text/joint, and joint/joint
vectors. A combined image-text query can change the vector away from the pure
image vector. Therefore querying with the same source image plus text need not
produce a similarity of exactly `1` against the stored pure-image embedding.

This repository chooses a separate local CLIP-compatible profile because it is
widely supported by sentence-transformers and keeps multimodal loading optional.
It does not claim architectural equivalence with Visualized-BGE.

## 11. Safe multimodal behavior in this repository

Approved image types are PNG, JPEG, and WebP. The boundary checks:

- extension and declared MIME compatibility;
- decoded image format;
- safe filename normalization;
- byte limit;
- pixel limit and decompression-bomb warnings;
- full RGB decode before persistence;
- SHA-256 idempotency.

Image vectors live in a separate fixed-dimension table and profile. This avoids
mixing text vectors with multimodal vectors merely because both are arrays of
floats.

The API supports text-to-image and image-to-image search. It returns image IDs,
dimensions, hashes, content URLs, ranks, and scores. It does **not** pass images
to the text-only chat model or pretend that model visually inspected them.

## 12. Why a vector database exists

Brute-force comparison is simple for hundreds of vectors. At millions or
billions, storing, filtering, updating, and searching high-dimensional arrays
becomes a specialized systems problem.

A vector database provides:

- high-dimensional vector storage;
- similarity search;
- approximate-nearest-neighbor indexes such as HNSW or IVF;
- insert, update, and delete lifecycle operations;
- scalar metadata filters;
- range and grouping queries in capable engines;
- framework integration, monitoring, and operational controls;
- in distributed products, horizontal scale and fault tolerance.

“Milliseconds at billion scale” is a product- and workload-dependent goal, not
a universal guarantee. Dimension, hardware, filters, index parameters, recall
target, concurrency, and data distribution all matter.

## 13. Vector databases and relational databases are complementary

Traditional relational databases excel at typed structured data, equality/range
predicates, joins, constraints, and ACID transactions. Vector search ranks by
geometric similarity.

The source contrasts vector and traditional databases, but the boundary is no
longer absolute. PostgreSQL with pgvector combines:

- relational metadata and transaction semantics;
- full-text search;
- vector columns and ANN indexes.

This is why Local LKE does not run a second vector service. Collection,
document-version, chunk, profile, vector, and lifecycle state can change in one
database transaction. Large distributed vector systems may still be preferable
when independent scale, specialized availability, or billion-vector workloads
justify the operational cost.

## 14. A conceptual vector-database architecture

The source divides the system into four layers:

1. **Storage layer:** vectors, scalar metadata, persistence, and distribution.
2. **Index layer:** HNSW, LSH, PQ, IVF, index building, and tuning.
3. **Query layer:** ANN execution, filters, hybrid requests, and optimization.
4. **Service layer:** clients, monitoring, logs, connection management, and security.

The separation helps diagnose failures. A correct vector can still be missing
from the index; a good ANN candidate can still be removed by a bad filter; a
correct query can still fail because a service namespace was not loaded.

## 15. Families of nearest-neighbor techniques

### 15.1 Tree-based methods

Annoy uses random-projection trees. Trees partition space so a query explores a
small subset rather than scanning every vector. Tree methods often work better
at moderate dimensions than at extreme dimensionality.

### 15.2 Hash-based methods

Locality-Sensitive Hashing chooses hash functions that place nearby objects in
the same buckets with high probability. Search examines matching or neighboring
buckets. It trades memory and probabilistic recall for speed.

### 15.3 Graph-based methods

HNSW constructs a multi-layer navigable small-world graph. Search starts in a
sparse upper layer, moves quickly toward the target region, and refines in denser
lower layers.

Important parameters include:

- `M`: maximum or target neighbor connections per node; higher values usually
  improve recall and memory cost.
- `ef_construction`: candidate breadth while building; higher values usually
  improve graph quality and build cost.
- `ef_search`: breadth at query time; higher values usually improve recall and
  latency cost.

Local LKE creates HNSW cosine indexes with explicit construction parameters.
Production tuning must use measured recall and latency.

### 15.4 Quantization methods

IVF first clusters vectors into coarse lists. A query probes the closest lists
instead of every vector. Product Quantization compresses subvectors into codebook
entries, reducing memory and accelerating approximate distance calculations.

Quantization may be followed by exact or higher-precision **result refinement**
on a candidate set. Compression improves scale at the cost of approximation.

## 16. Main vector index tradeoffs

| Index | Principle | Strength | Cost/risk |
|---|---|---|---|
| FLAT | Compare every vector | Exact recall | Linear query work |
| IVF_FLAT | Search selected coarse clusters | Good throughput balance | Misses unprobed clusters |
| IVF_SQ8/PQ | IVF plus quantization | Lower memory, faster scans | More approximation |
| HNSW | Navigate a layered neighbor graph | Low latency, high recall | High memory and build cost |
| DiskANN | SSD-optimized graph search | Data can exceed RAM | Disk-aware operational complexity |

No index is universally best. Choose through data scale, memory, latency,
throughput, update rate, and measured recall.

## 17. Survey of vector-store products

The source introduces:

- **Pinecone:** managed/serverless service emphasizing operational simplicity,
  scale, availability, and low latency.
- **Milvus:** open-source distributed vector database with multiple indexes,
  GPU options, and large-scale architecture.
- **Qdrant:** Rust-based open-source engine emphasizing performance, filtering,
  and quantization.
- **Weaviate:** open-source database with GraphQL and many AI integrations.
- **Chroma:** lightweight local-first store suited to prototypes and education.
- **FAISS:** an indexing and clustering library rather than a database service.

The guide recommends Chroma or FAISS for learning/small prototypes and a
specialized service for million-scale, high-concurrency, frequently updated, or
complex-filter workloads. Local LKE instead chooses pgvector because its local
scope benefits from relational lifecycle transactions and one service.

## 18. FAISS as a local vector library

FAISS is a high-performance similarity-search and clustering library. In the
source LangChain example:

1. Create `Document` objects.
2. Embed their text with a Hugging Face model.
3. Build a FAISS vector store.
4. Save the index locally.
5. Reload it with the same embedding model.
6. Search for the nearest document.

FAISS persistence commonly includes:

- a `.faiss` binary index;
- a serialized document store/mapping such as `.pkl`.

The example's `allow_dangerous_deserialization=True` is a real warning:
pickle-like files can execute code during deserialization. Never load an
untrusted index artifact merely to make an example run.

The LangChain construction path is conceptually layered:

- `from_documents` extracts text and metadata.
- `from_texts` calls `embedding.embed_documents`.
- internal `__from` chooses an empty FAISS index and prepares the docstore.
- internal `__add` adds numeric arrays, stores documents, and creates the mapping
  from FAISS integer positions to document IDs.

That ID mapping is essential. An ANN result is only a vector position until the
system reconnects it to text, metadata, and provenance.

The source also asks the learner to inspect LlamaIndex's readable JSON
persistence and implement reload/search. This reinforces the same lesson: an
index includes vectors plus identifiers, metadata, and storage metadata—not only
a matrix of floats.

## 19. Milvus deployment concepts

Milvus targets production-scale distributed search rather than an embedded
library. The cited standalone deployment uses Docker Compose and three services:

- Milvus standalone;
- etcd for metadata;
- MinIO for object storage.

The default client port is `19530`. `docker compose down` stops containers while
retaining volumes; `docker compose down -v` also removes the stored data and is
therefore destructive.

Local LKE does not deploy Milvus. These details are retained because they explain
the operational difference between an algorithm library, a database extension,
and a distributed vector service.

## 20. Milvus data organization

The source uses a library analogy:

- **Collection:** the library, comparable to a table.
- **Partition:** a library area, a logical subset used to reduce search scope.
- **Schema:** the catalog rules defining every field.
- **Entity:** one stored record.
- **Alias:** a movable application-facing name for a collection.

### 20.1 Schema and fields

A collection schema can contain:

- exactly one primary-key field;
- one or more vector fields;
- scalar fields such as strings, numbers, booleans, or JSON.

The source's news example includes article identity, text metadata, image URL,
image embedding, summary dense embedding, and summary sparse embedding. Multiple
vector fields enable multimodal and hybrid retrieval, but each field still has a
defined dimension and metric.

### 20.2 Partitions

Every Milvus collection starts with `_default`. Additional logical partitions
can group categories or dates. Searching selected partitions reduces candidate
scope and allows bulk load/unload/delete operations. The cited limit is 1024
partitions per collection. Excessive partitioning can itself become an
operational burden.

### 20.3 Aliases and safe cutover

An alias can point to `collection_v1`, then atomically switch to a fully built
`collection_v2`. The application keeps using the alias and never observes a
partially rebuilt collection.

Local LKE maps this idea to collection/profile activation:

1. write all vector batches inactive;
2. verify the expected node count;
3. activate the complete version and profile transactionally;
4. deactivate stale references.

## 21. Milvus indexes and search

Milvus supports scalar indexes such as inverted or bitmap indexes and vector
indexes such as FLAT, IVF, HNSW, and DiskANN. An index is a materialized data
structure that consumes storage and often memory to reduce query work.

The source HNSW example uses cosine distance with:

```text
M = 16
efConstruction = 256
ef search = 128
```

The exact values are examples, not universal defaults. After index creation, a
Milvus collection must be loaded before search; a PostgreSQL index does not have
the same explicit collection-load lifecycle.

Basic ANN search specifies:

- vector field (`anns_field`);
- one or more query vectors (`data`);
- result count (`limit` or top-k);
- metric and index search parameters.

## 22. Enhanced vector retrieval modes

### 22.1 Filtered search

Apply scalar predicates and then search the eligible subset—for example, similar
products below a price or relevant documents from a particular year/category.
Filters increase precision but can reduce recall if metadata is missing or
incorrect.

### 22.2 Range search

Return every result inside a distance/similarity interval instead of an arbitrary
top-k. Face verification and anomaly detection are common examples. Thresholds
must be calibrated for the model and population.

### 22.3 Multi-vector hybrid search

Run several vector searches in parallel—for example dense text, sparse text, and
image vectors—then fuse results. Milvus offers RRF and weighted rankers. Chapter
4 performs dense/lexical RRF above PostgreSQL; the principle is the same even
though the storage API differs.

### 22.4 Grouping search

Group by a field such as `document_id` so top results do not all come from one
book, video creator, or source. Local LKE applies parent deduplication and source
caps during context assembly. Grouping increases diversity but may hide multiple
independently relevant passages from one source if configured too aggressively.

## 23. The source Milvus multimodal exercise

The full exercise:

1. Wrap Visualized-BGE in an encoder.
2. Discover the model's vector dimension from a sample image.
3. Create a collection with auto-increment ID, vector, and image-path fields.
4. Encode each image and batch insert vectors plus paths.
5. Create an HNSW cosine index.
6. Load the collection.
7. Encode an image-plus-text query.
8. Search top five and return image paths.
9. Render a panorama of query and results.
10. Release and drop the demonstration collection.

The example's top result is the query image itself, but its similarity is about
`0.94`, not exactly `1`, because stored data uses a pure-image vector while the
query combines image features with the text “a dragon.” The query semantics have
shifted the vector.

Local LKE replaces filesystem paths in public results with bounded content URLs.
It keeps stored paths internal and never exposes them as user-facing metadata.

## 24. Mapping Milvus/FAISS concepts to pgvector

| Source concept | Local LKE mapping |
|---|---|
| Milvus Collection | relational `collections` plus vector rows scoped by `collection_id` |
| Collection Schema | Alembic/SQLAlchemy tables and constraints |
| Entity primary key | stable node/image IDs |
| Vector field | fixed-dimension pgvector column |
| Scalar fields | document/version/chunk/profile columns and joins |
| Partition | collection/profile/version predicates; no physical Milvus partition |
| Alias | atomically active profile and active document version |
| FLAT | exact pgvector scan when ANN is not selected by the planner |
| HNSW | PostgreSQL HNSW `vector_cosine_ops` index |
| Filtered search | SQL predicates and lifecycle joins |
| Grouping | parent deduplication and context diversity controls |
| FAISS docstore mapping | foreign keys from vectors to chunks/elements/versions |

PostgreSQL's planner decides whether to use HNSW based on query shape, table
size, filters, and cost estimates. Creating an ANN index does not prove every
query uses it; use `EXPLAIN (ANALYZE, BUFFERS)` for real tuning.

## 25. The chunk-size contradiction

Small chunks are precise retrieval targets but may lack explanatory context.
Large chunks provide context but dilute embedding meaning and introduce noise.

Chapter 3 resolves this by separating the unit used for **matching** from the
unit supplied for **generation**:

```text
index small -> retrieve precise child -> expand bounded context
```

This is a family of strategies, not one algorithm.

## 26. Sentence-window retrieval

The source LlamaIndex process is exact and worth remembering.

### 26.1 Index phase

1. Split a document into individual sentences.
2. Create one `TextNode` per sentence.
3. For node `i`, select:

```text
nodes[max(0, i-window) : min(i+window+1, len(nodes))]
```

4. Join those sentences into a window string.
5. Store the window and original sentence in node metadata.
6. Exclude the window/original-text metadata keys from embedding and LLM metadata
   rendering.

Exclusion is crucial: only the precise sentence should create the retrieval
vector. The large window is stored for later replacement, not folded into every
sentence embedding.

### 26.2 Retrieval and post-processing

1. Similarity-search sentence nodes.
2. `MetadataReplacementPostProcessor` replaces each matched sentence with its
   stored window.
3. Send expanded windows to generation.

The source compares sentence-window retrieval with a regular 512-token splitter
on an AMOC climate question. Both answers identify projected decline, but the
window answer retains more surrounding uncertainty details: low confidence in
quantitative projections, short observations, and uncertainty in historical
reconstruction. One example does not prove universal superiority; it shows the
mechanism's intended benefit.

Local LKE stores sentence nodes and expands neighbors by stable chunk and
sentence ordinal. It exposes the child that triggered the window.

## 27. Parent-child retrieval

Parent-child retrieval indexes small children while retaining a link to a larger
element, section, or document.

At query time:

1. rank children;
2. resolve each child's parent;
3. keep the best child score for each repeated parent;
4. return the parent under a token budget;
5. preserve the triggering child ID and locator.

This prevents five sibling matches from inserting the same parent five times.
It also prevents a broad parent from hiding which precise child caused recall.

## 28. Multi-granularity indexing

Different questions need different units:

- sentence nodes: precise facts and paraphrases;
- chunk nodes: normal passage lookup;
- section nodes: broad synthesis and topic summaries.

Multi-granularity search can query all three spaces in one compatible profile.
The result must deduplicate overlapping contexts and preserve the best trigger.
More nodes increase embedding and storage cost, so idempotent batch indexing is
important.

## 29. Structured metadata indexes

As a corpus grows to hundreds of PDFs, global top-k search can be noisy and
expensive. Metadata such as filename, date, heading, author, year, quarter, or
document type can first restrict the eligible subset.

For “Summarize AI discussion in the 2023 Q2 report”:

1. filter to report documents for year 2023 and quarter Q2;
2. vector-search “AI discussion” inside that subset.

Markdown heading-aware chunking already creates useful structural metadata.
Metadata quality is therefore part of retrieval quality.

Chapter 4 implements allowlisted metadata filters. The application never passes
an arbitrary model-generated filter language directly to SQL.

## 30. Recursive retrieval across structured sources

The source workbook example has one sheet per movie year.

### 30.1 LlamaIndex demonstration

For each sheet:

- build a `PandasQueryEngine`;
- create a summary `IndexNode` such as “contains movies from 1994”;
- add the summary to a top-level vector index;
- map the node's `index_id` to the sheet query engine.

The `RecursiveRetriever` first retrieves the summary pointer, enters the matching
sheet engine, generates Pandas code, executes it, and returns the result.

### 30.2 Critical security warning

The source explicitly warns that experimental `PandasQueryEngine` may generate
Python and execute it with `eval()`. Without a strong sandbox, model-generated
code can execute arbitrary operations. It is unsuitable for this local project.

### 30.3 Safer two-index design

The source proposes:

1. a summary index used only for routing;
2. a content index whose records carry `sheet_name` metadata;
3. retrieve the best summary;
4. apply the returned sheet name as a metadata filter to content search.

This separates route selection from data execution. Local LKE goes further for
CSV: Chapter 4 validates a Pydantic query plan, allowlists columns/operators,
compiles SQLAlchemy expressions, binds values, and enforces read-only limits. It
never executes Python or raw model SQL.

## 31. Framework philosophy

The source deliberately mixes LangChain, LlamaIndex, Milvus, and custom code.
Its lesson is to learn the mechanism rather than memorize one framework API.

- Frameworks accelerate common paths.
- Every framework has abstractions and security boundaries.
- Real requirements may need lower-level composition.
- Understanding nodes, vectors, mappings, filters, activation, and post-processing
  transfers across libraries.

Official framework documentation remains the best reference for version-specific
APIs. These notes focus on durable system ideas.

## 32. Persistent indexing lifecycle in Local LKE

### 32.1 Profiles and fixed dimensions

Text vectors use a 384-dimensional pgvector column for
`BAAI/bge-small-en-v1.5`. Multimodal vectors use a separate 512-dimensional
column for a CLIP-compatible model. A different dimension requires an explicit
schema migration; silently inserting it is prohibited.

### 32.2 Bounded batches and resume

Index jobs record:

- total nodes;
- successfully embedded nodes;
- successful provider calls;
- progress;
- status and safe error.

Each completed batch commits separately as inactive rows. A retry reads existing
node IDs and embeds only missing batches. An unchanged fully indexed version
performs zero new embedding calls.

### 32.3 Transactional activation

Incomplete rows never become searchable. Activation verifies the expected node
count, deactivates stale document vectors, activates the complete namespace, and
switches the collection profile. Retrieval also joins active document versions
and non-deleted logical documents.

### 32.4 Incremental lifecycle

- Add: index the new active version.
- Replace: inactive old versions cannot be recalled; complete new vectors replace
  active references.
- Delete: soft deletion retains provenance but deactivates vectors.
- Reindex: force rebuilding the version/profile namespace.
- Model change: create a new embedding profile instead of mixing vectors.

## 33. Retrieval laboratory interpretation

The persistent vector lab exposes:

- selected embedding profile;
- child node granularity;
- child score and rank;
- trigger node ID;
- expanded context and locator;
- duplicate-parent/window decisions;
- included/excluded decision;
- final token count.

This separation matters because a retrieval score belongs to the indexed child,
not automatically to every sentence in its expanded parent.

## 34. Chapter 3 evaluation baseline

Important tests answer different questions:

| Test | Failure it catches |
|---|---|
| Recall@k labelled fixture | relevant child is not recalled |
| Parent/window ordering | wrong neighbor or parent expansion |
| Deduplication | repeated parent consumes context |
| Token budget | expansion overflows generation budget |
| Idempotency | unchanged input is re-embedded |
| Batch retry | completed work is lost after one failure |
| Dimension mismatch | incompatible vectors enter a namespace |
| Replacement/deletion | stale vectors remain searchable |
| Image ranking | shared text/image space is wired incorrectly |
| Image validation | malformed or oversized media is accepted |

ANN recall should eventually be compared with an exact search baseline. A good
answer is not a substitute for retrieval metrics because generation can mask a
retrieval regression.

## 35. Common mistakes

1. Mixing vectors from different models because dimensions happen to match.
2. Changing query prefixes without changing the profile.
3. Using cosine-style scores on unnormalized vectors without checking model docs.
4. Treating higher dimension as automatically better.
5. Selecting a model solely from an average leaderboard score.
6. Re-embedding the corpus on every query.
7. Activating vectors before every batch succeeds.
8. Returning a parent without the child that triggered it.
9. Expanding every candidate before deduplication and token budgeting.
10. Loading an untrusted FAISS pickle with dangerous deserialization enabled.
11. Executing model-generated Python or raw SQL.
12. Claiming a text-only chat model inspected retrieved images.
13. Creating an HNSW index without measuring whether the planner uses it.
14. Reporting similarity scores without defining metric and direction.

## 36. Glossary

| Term | Meaning |
|---|---|
| Embedding | Fixed-length learned vector representation |
| Dense vector | Vector with most dimensions populated |
| Sparse vector | High-dimensional vector with mostly zero values |
| Dimension | Number of numeric coordinates |
| Normalization | Scaling a vector to unit length |
| Cosine similarity | Directional alignment between vectors |
| ANN | Approximate nearest-neighbor search |
| Recall@k | Fraction of labelled relevant items found in top k |
| HNSW | Hierarchical Navigable Small World graph index |
| IVF | Inverted-file coarse-cluster index |
| PQ | Product Quantization compression |
| Profile | Versioned embedding compatibility contract |
| Namespace | Isolated set of mutually comparable vectors |
| Child | Small indexed retrieval unit |
| Parent | Larger context resolved after child retrieval |
| Sentence window | Center sentence plus bounded neighbors |
| Cross-modal alignment | Shared geometry across text, image, or other types |
| Dual encoder | Independent encoders projected into one space |
| Zero-shot | Applying learned alignment without task-specific fine-tuning |

## 37. Further reading cited by Chapter 3

- Lewis et al. (2020), *Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks*
- BERT and RoBERTa material cited by the embedding section
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Milvus](https://github.com/milvus-io/milvus) and its architecture/index docs
- LlamaIndex, *Building Performant RAG Applications for Production*
- LlamaIndex sentence-window metadata-replacement example
- LlamaIndex recursive retriever and structured hierarchical retrieval examples
- [pgvector](https://github.com/pgvector/pgvector)

## 38. What I should retain

1. Embeddings create task-specific geometry, not lossless semantic truth.
2. Model, revision, dimension, normalization, and prefixes form one contract.
3. Public benchmarks select candidates; private Recall@k selects the deployment.
4. Dense, lexical, structured, and multimodal search solve different problems.
5. A vector database reconnects numeric neighbors to metadata and provenance.
6. ANN trades exact recall for speed and must be measured against exact search.
7. HNSW parameters trade memory/build/query time for recall.
8. CLIP learns a shared text/image space through batch contrastive alignment.
9. Image retrieval does not give a text-only generator visual understanding.
10. Retrieve small children and expand bounded parents/windows for generation.
11. Deduplicate expanded parents while preserving the best triggering child.
12. Persist batch progress and activate only complete indexes.
13. Never execute model-generated Python or raw SQL as a retrieval shortcut.
14. Framework APIs change; vector, mapping, lifecycle, and security principles last.
