# Chapter 2 Learning Notes: Document Loading and Text Chunking

Chapter 1 proved the basic RAG loop with controlled files. Chapter 2 studies the
first production-quality stage: turning long, varied documents into safe,
traceable, retrieval-ready chunks.

These notes survey the main document-loading and chunking techniques, then map
each technique to this repository. A technique being explained here does not
imply that it is already implemented.

## 1. Where chunking fits in RAG

Text chunking divides a loaded document into smaller units. Those units become
the basic inputs to embedding, vector indexing, retrieval, context assembly,
and generation.

```text
source file → parser → normalized elements → chunker → chunks → embeddings/index
```

Parsing and chunking answer different questions:

- Parsing asks: what content and structure did the source contain?
- Chunking asks: what units should later retrieval compare and return?

Keeping them separate matters. A parser can preserve pages, headings, tables,
lists, and source lines before a chunking experiment changes boundaries. Every
derived chunk should still point back to its source version and normalized
element.

This is an ETL pipeline: extract source content, transform it into a consistent
representation, and load it with enough provenance to reproduce the result.
Poor parsing or chunking cannot be repaired reliably by a better retriever.

## 2. Why chunking is necessary

### 2.1 The embedding model has an input limit

An embedding model converts a chunk into one fixed-size vector. It has a hard
token limit. If a chunk exceeds that limit, the provider may reject it or,
worse, truncate it silently. The resulting vector then represents only part of
the source.

The safe rule is:

```text
chunk token count ≤ embedding model input limit
```

This limit must be checked for the actual embedding model and tokenizer. A
character count is only an approximation because token-to-character ratios vary
by language, punctuation, code, and model vocabulary.

A model with a 512-token input window provides a concrete illustration: a chunk
beyond that limit may be truncated and its embedding cannot represent the
omitted text. Treat the number as model-specific, not as a universal embedding
limit.

### 2.2 The generation model has a context limit

The chat model must fit all of the following in one context window:

- system and application instructions;
- conversation history;
- the user question;
- all retrieved chunks and their metadata;
- space for the generated answer.

If each retrieved chunk is large, only a few chunks fit. That reduces evidence
breadth and can prevent a multi-part question from receiving coverage across
several sources.

### 2.3 A legal maximum is not a good target size

A model accepting a very long input does not mean every chunk should approach
that maximum. Chunk size is a retrieval-quality decision, not only a validation
constraint.

#### Embedding information dilution

Most text embedding models use a Transformer-style process:

1. Tokenization converts text into tokens.
2. The encoder produces a contextual vector for every token.
3. Pooling compresses all token vectors into one vector for the entire chunk.

Common pooling methods include using the special `[CLS]` representation or
mean-pooling all token representations. `[CLS]` is a special token used by
BERT-style encoders whose final state is trained to aggregate sequence-level
information.

Pooling is lossy. A single vector of fixed dimensionality must summarize every
fact and topic in the chunk. As a chunk grows and contains more semantic points,
the representation becomes more general. A small but decisive detail may have
too little influence on the final vector, reducing retrieval precision.

#### Topic dilution

A good retrieval chunk normally focuses on one coherent topic. Suppose one
large operations chunk combines deployment steps, billing policy, and incident
response. A query about incident acknowledgement may not match the mixed vector
strongly enough, even though the answer is present. Three topic-focused chunks
produce a much clearer match.

This produces a common failure mode:

```text
the corpus contains the answer
        ↓
the answer is inside an over-broad chunk
        ↓
the chunk vector is dominated by several topics
        ↓
the relevant chunk is not retrieved
```

#### Lost in the middle during generation

Even if several oversized chunks fit inside a long-context LLM, the evidence can
be buried in noise. Long-context models often use information near the beginning
and end more reliably than information in the middle. This positional behavior
is commonly called “lost in the middle.” More context is therefore not
automatically better context.

The practical goal is high signal-to-noise evidence: enough local context to
interpret a fact, but not so much unrelated material that retrieval and
generation lose focus.

## 3. The central chunk-size tradeoff

Smaller chunks usually improve match precision, but they can separate a fact
from its definition, condition, exception, or heading. Larger chunks preserve
local coherence, but they dilute embeddings and consume more of the generation
budget.

Overlap copies boundary text into neighboring chunks. It can preserve continuity
when a sentence or idea crosses a boundary, but excessive overlap creates:

- duplicate embeddings and storage;
- repetitive retrieval results;
- wasted context tokens;
- misleadingly strong evidence caused by the same passage appearing repeatedly.

There is no universal best size or overlap. The values depend on document
structure, language, embedding tokenizer, question granularity, retrieval
strategy, and generation budget. They should be treated as versioned pipeline
configuration and evaluated on representative questions.

## 4. Paragraph-aware fixed-size chunking

The simplest family aims for a target size and overlap. LangChain’s
`CharacterTextSplitter` is often described as fixed-size chunking, but its
default behavior is more accurately paragraph-aware adaptive chunking.

### 4.1 Two-stage algorithm

1. Split the text with a separator. The default separator is `"\n\n"`, so the
   initial units are paragraphs.
2. Merge consecutive units while monitoring accumulated length. When adding the
   next unit would exceed `chunk_size`, emit the current chunk and retain enough
   prior material to satisfy `chunk_overlap`.

Important consequences:

- Paragraph integrity is preferred over exact chunk length.
- Chunk sizes vary around the target.
- A single paragraph larger than the limit may remain oversized and produce a
  warning because this strategy does not recursively split it.
- Overlap is applied during the merge stage to preserve continuity.

### 4.2 Strengths and weaknesses

Strengths:

- simple and fast;
- low computational cost;
- predictable enough for logs, preprocessing, and already-uniform text;
- easy to reason about and visualize.

Weaknesses:

- can cut across semantic boundaries;
- can combine unrelated short paragraphs;
- can leave a very long paragraph above the intended limit;
- knows nothing about headings, tables, code structure, or topic changes.

This strategy is a useful baseline, not a general optimum.

An illustrative LangChain configuration is:

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=200,
    chunk_overlap=10,
)
chunks = splitter.split_documents(documents)
```

Use `split_text(...)` for one string and `split_documents(...)` for a sequence of
LangChain `Document` objects.

## 5. Recursive character chunking

`RecursiveCharacterTextSplitter` improves the size guarantee by trying a
hierarchy of increasingly fine separators.

Typical English defaults are:

```python
["\n\n", "\n", " ", ""]
```

An application can add sentence punctuation before spaces or characters:

```python
["\n\n", "\n", ". ", " ", ""]
```

### 5.1 Algorithm, step by step

For the current text segment:

1. Scan the ordered separator list and choose the first separator that actually
   occurs in the text. If none occurs, use the final fallback, usually the empty
   string for character-level splitting.
2. Split the text using that separator.
3. Add fragments within the size limit to a temporary `_good_splits` batch.
4. When an oversized fragment appears:
   - merge and emit the accumulated acceptable fragments;
   - if finer separators remain, recursively split the oversized fragment with
     the remaining separator list;
   - if no separator remains, retain the oversized fragment rather than recurse
     forever.
5. Merge and emit the final accumulated fragments.

The `no separators remain` condition is the recursion termination rule. The
batching behavior is also important: acceptable fragments are merged only when
an oversized fragment forces a flush or when the input ends.

### 5.2 Difference from paragraph-aware fixed splitting

Both approaches prefer natural boundaries and use the same merging/overlap
concept. Their key difference appears when one initial unit is too large:

- `CharacterTextSplitter` may warn and preserve the oversized paragraph.
- `RecursiveCharacterTextSplitter` tries line, sentence, word, and finally
  character boundaries until the fragment fits or no fallback remains.

Recursive splitting therefore balances structural preservation with stronger
size control.

### 5.3 Multilingual separators

Languages without ordinary ASCII word boundaries require tailored separators.
Useful additions include:

```python
[
    "\n\n", "\n", " ",
    ".", ",",
    "\u200b",       # zero-width space used in some Thai/Japanese text
    "\uff0c",       # full-width comma
    "\u3001",       # ideographic comma
    "\uff0e",       # full-width full stop
    "\u3002",       # ideographic full stop
    "",
]
```

The separator order expresses preference: preserve paragraphs before lines,
sentences before words, and words before characters.

### 5.4 Programming-language-aware recursion

Code should be split around syntax-level boundaries where possible. LangChain
provides language presets through
`RecursiveCharacterTextSplitter.from_language(...)`. Depending on the language,
the hierarchy can prefer classes, functions, methods, control-flow statements,
lines, and finally characters. This produces chunks that better match how code
is read and retrieved than arbitrary character windows.

## 6. Embedding-based semantic chunking

Semantic chunking tries to split where the topic changes rather than at a fixed
separator or character count. LangChain’s experimental `SemanticChunker` is a
representative implementation.

### 6.1 Canonical workflow

1. Sentence splitting: divide the input with sentence-ending punctuation such
   as periods, question marks, and exclamation marks.
2. Context-aware embedding: for every sentence, combine it with neighboring
   sentences before embedding. With the default `buffer_size=1`, the embedded
   unit contains the previous sentence, current sentence, and next sentence
   where available. This gives each vector local context instead of representing
   an isolated sentence.
3. Distance calculation: compute cosine distance between every pair of adjacent
   context-aware embeddings.
4. Breakpoint identification: estimate a dynamic threshold and mark unusually
   large semantic jumps.
5. Chunk assembly: split the original sentence sequence at those breakpoints and
   merge the sentences within each region.

For adjacent vectors `v_i` and `v_(i+1)`, a typical distance is:

```text
d_i = 1 - cosine_similarity(v_i, v_(i+1))
```

A larger `d_i` means a larger semantic change and a more plausible boundary.

### 6.2 Breakpoint threshold methods

`breakpoint_threshold_type` controls how an unusually large change is detected.

| Method | Decision rule | Typical default | When it helps |
|---|---|---:|---|
| `percentile` | Split when a distance exceeds the selected percentile of all adjacent distances | 95th percentile | General default; selects roughly the most extreme changes |
| `standard_deviation` | Split above `mean + N × standard deviation` | `N = 3` | Data whose distances are reasonably summarized by mean and spread |
| `interquartile` | Split above `Q3 + N × IQR`, where `IQR = Q3 - Q1` | `N = 1.5` | More robust treatment of skew and outliers |
| `gradient` | Compute the rate of change of adjacent distances, then apply a percentile threshold to that gradient | 95th percentile | Dense, internally similar material such as legal or medical text, where a change in the distance pattern is more revealing than an absolute distance |

`breakpoint_threshold_amount` configures the percentile or multiplier. Changing
it changes chunk count and granularity: a stricter threshold yields fewer,
larger chunks; a more permissive threshold yields more, smaller chunks.

### 6.3 Benefits and costs

Benefits:

- chunks can remain internally topic-coherent;
- boundaries are not limited to formatting conventions;
- semantically meaningful changes can be found in plain prose.

Costs and risks:

- every sentence neighborhood requires embedding work during ingestion;
- results depend on the embedding model, buffer size, sentence splitter, and
  threshold method;
- the resulting sizes are less predictable;
- very short documents provide too few distance observations for robust
  statistics;
- changing the embedding model or breakpoint configuration must change the
  pipeline hash and trigger a new version.

The canonical LangChain experiment has this shape:

```python
from langchain_experimental.text_splitter import SemanticChunker

splitter = SemanticChunker(
    embeddings,
    buffer_size=1,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=95,
)
chunks = splitter.split_documents(documents)
```

The embedding adapter should normalize vectors when the selected model expects
cosine-normalized output. The exact import path and accepted parameters are
version-sensitive because this splitter is experimental.

### 6.4 What this repository implements

This repository’s Chapter 2 `semantic` option is deliberately different. It
uses local TF-IDF vectors for adjacent sentences and a deterministic similarity
threshold. It requires no model download or network call, so the normal test
suite remains offline.

That option is a semantic-boundary experiment, not an implementation of
LangChain `SemanticChunker`. It does not provide context-window buffering or the
four statistical breakpoint modes above. A future embedding-based experiment
must version the embedding model, buffer size, sentence rule, threshold type,
and threshold amount in the pipeline configuration.

## 7. Structure-aware chunking

Documents with explicit structure—Markdown, HTML, LaTeX, and source code—offer
boundaries that are often more reliable than length alone.

### 7.1 Markdown heading hierarchy

LangChain’s `MarkdownHeaderTextSplitter` follows a two-part idea:

1. Define which markers represent levels, such as `# → Header 1` and
   `## → Header 2`.
2. Walk the document and group content under the current heading path until the
   next heading at the same or higher level.

Each logical section receives metadata representing its full address. A chunk
inside “Chapter 3: Model Evaluation → 3.2 Metrics” can carry both headings. This
metadata explains where the passage belongs even if the heading text is not
repeated in every chunk.

An illustrative header mapping is:

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter

header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)
sections = header_splitter.split_text(markdown_text)
```

### 7.2 Why structure and recursive size control should be combined

Heading-only splitting can produce an enormous section. The robust pattern is:

1. split into logical sections and attach heading-path metadata;
2. recursively split any oversized section;
3. copy the heading metadata to every smaller child chunk.

This preserves the macro structure while respecting embedding and generation
limits. The repository follows this principle by parsing Markdown into ordered
elements with heading paths, then applying bounded chunking within each element.

## 8. Unstructured: partition first, then chunk elements

Unstructured separates document understanding from chunk assembly.

### 8.1 Partitioning

Partitioning converts PDF, HTML, and other source formats into ordered semantic
elements such as:

- `Title`;
- `NarrativeText`;
- `ListItem`;
- `Table`;
- headers, footers, and other layout elements.

The element labels and metadata preserve more source meaning than flattening the
whole document into plain text.

### 8.2 Element-based chunking

Unstructured offers two principal strategies:

- `basic` combines consecutive elements until `max_characters` is reached. If a
  single element is too large, it text-splits that element.
- `by_title` adds section awareness. A `Title` starts a new chunk, so elements
  from different titled sections are not combined into the same chunk.

Partitioning and chunking may be performed in one call or as separate stages.
The latter is easier to inspect and version: first understand the document,
then experiment with how its elements should be combined.

This repository uses Unstructured for PDF partitioning but normalizes the
elements into its own `DocumentElement` model and applies its own chunking
strategies. It does not currently call Unstructured’s `basic` or `by_title`
chunker directly.

## 9. LlamaIndex: nodes and transformation pipelines

LlamaIndex models parsed content as nodes. Chunking is one transformation in a
larger node-processing pipeline.

### 9.1 Node parser families

- Structure-aware parsers include `MarkdownNodeParser`, `JSONNodeParser`, and
  `CodeSplitter`.
- Semantic parsers include `SemanticSplitterNodeParser`, which detects embedding
  changes between sentences.
- `SentenceWindowNodeParser` creates a small node for each sentence but stores
  the neighboring `N` sentences in metadata. Retrieval can match the precise
  sentence while generation receives its broader window. This is an early form
  of the “small unit for recall, larger context for generation” pattern.
- Conventional parsers include `TokenTextSplitter` and `SentenceSplitter`.

### 9.2 Composable transformations

A pipeline can first create Markdown section nodes and then sentence-split each
section. Each child node retains source and structural metadata. This resembles
the repository’s parse-elements-first design and prepares for parent-child
retrieval in Chapter 3.

### 9.3 Interoperability

LlamaIndex’s `LangchainNodeParser` can wrap a LangChain text splitter, allowing a
LangChain boundary algorithm to participate in a LlamaIndex transformation
pipeline. The frameworks use different public abstractions—LangChain documents
and LlamaIndex nodes—but the chunking ideas are transferable.

## 10. Visual inspection with ChunkViz

ChunkViz is a simple visualization tool that colors chunk boundaries and
overlap regions for a supplied document and configuration. It is useful for
spotting:

- boundaries that split a sentence or definition;
- excessive overlap;
- one abnormally large chunk;
- highly uneven chunk sizes;
- a heading separated from its content.

Visual inspection is diagnostic, not an evaluation metric. Do not upload
private documents to an external visualization service unless its privacy and
retention behavior are acceptable. A local equivalent can display the same
boundaries from stored parser previews.

## 11. Strategy comparison

| Strategy | Boundary signal | Size predictability | Compute cost | Main strength | Main risk |
|---|---|---:|---:|---|---|
| Paragraph-aware character | Separator plus target length | Medium | Very low | Simple baseline | Oversized paragraphs and topic mixing |
| Recursive character | Ordered separators plus target length | High | Low | Strong size control while preferring natural boundaries | Still heuristic rather than topic-aware |
| Embedding semantic | Adjacent contextual embedding distance | Low | High | Topic-coherent prose chunks | Model/config sensitivity and ingestion cost |
| Markdown structure | Heading hierarchy | Low until recursively refined | Low | Precise section metadata | Large sections need a second splitter |
| Unstructured `basic` | Ordered document elements and max characters | Medium | Medium | Respects parsed elements | Can combine different sections |
| Unstructured `by_title` | Title elements plus max characters | Medium | Medium | Prevents cross-section mixing | Depends on title-detection fidelity |
| Sentence window | Sentence for retrieval, neighbors for context | High at retrieval unit | Medium | Precise recall with broader generation context | More metadata and parent/window handling |

In practice these strategies are composed. For example: partition a PDF into
elements, begin a new logical group at each title, recursively split oversized
groups, retrieve small children, and expand to a parent or sentence window for
generation.

## 12. How to choose and evaluate a strategy

Start from the application’s questions and evidence granularity, not an
arbitrary popular chunk size.

1. Identify the smallest passage that usually answers one question.
2. Identify what surrounding text is required to interpret that passage.
3. Keep chunks below the embedding tokenizer’s actual limit.
4. Reserve generation tokens for instructions, question, multiple evidence
   chunks, citations, and the answer.
5. Preserve source structure and stable locators before tuning size.
6. Compare configurations on the same evaluation set.

Useful measurements include:

- chunk count and size distribution;
- percentage of chunks above the embedding limit;
- duplicate/near-duplicate rate caused by overlap;
- retrieval Recall@k, MRR, and nDCG;
- evidence coverage for multi-part questions;
- context precision or irrelevant-token ratio;
- ingestion time, embedding calls, and storage size;
- answer accuracy, citation validity, and latency.

A visual preview can reveal obviously bad boundaries, but retrieval and answer
evaluation determine whether a strategy actually helps.

## 13. Repository mapping and current boundaries

| Source concept | Repository status after Chapter 2 |
|---|---|
| Normalized structure before chunking | Implemented for Markdown, UTF-8 text, and Unstructured PDF elements |
| Paragraph-aware fixed splitting | Explained as a baseline; no separate public strategy option |
| Recursive splitting | Implemented with configurable size and overlap |
| Multilingual/code presets | Explained conceptually; not exposed in Chapter 2 configuration |
| Embedding-based `SemanticChunker` | Explained conceptually; not implemented |
| Local semantic experiment | Implemented with deterministic adjacent-sentence TF-IDF similarity |
| Markdown heading metadata plus bounded refinement | Implemented through heading-aware parsing followed by bounded per-element splitting |
| Unstructured `basic` / `by_title` | Explained conceptually; PDF partitioning is implemented, but these chunkers are not called directly |
| LlamaIndex node parsers and sentence windows | Framework comparison only; parent/window retrieval begins in Chapter 3 |
| Visual chunk inspection | Implemented locally through parser/chunk previews in Gradio |
| Retrieval impact measurement | Deferred to Chapter 3, when persisted chunks enter the index |

Every persisted project chunk records its immutable document version, parent
element, strategy, ordinal, heading path, page or line locator, character count,
approximate token count, stable hash ID, and quality flags. Content and pipeline
hashes make experiments reproducible and prevent duplicate writes.

## 14. Safety and versioning around chunking

Uploaded content is untrusted data. Before parsing, the project validates the
extension, declared MIME, basic content signature or UTF-8 encoding, configured
file and batch sizes, filename/path safety, PDF integrity, and encryption. It
stores uploads under generated IDs and never executes document content.

The pipeline hash versions the parser/chunker contract. Together with the file’s
SHA-256 content hash, it distinguishes:

1. same bytes and same pipeline: skip repeated parsing and chunk writes;
2. changed bytes: create a new immutable content version;
3. same bytes but changed pipeline: create a new derived version.

Exactly one version is active for a logical document. Older versions remain for
provenance and reproducibility. Failed or interrupted jobs remain inspectable
but never create searchable active versions.

## 15. Debugging checklist

When ingestion or retrieval looks wrong, inspect in this order:

1. Did the parser preserve element order, titles, pages, lines, and tables?
2. Is the correct document and pipeline version active?
3. Do heading paths and parent element IDs identify the right source context?
4. Are any chunks empty, duplicated, boilerplate-heavy, or above the embedding
   token limit?
5. Are chunk size and overlap producing coherent boundaries?
6. For semantic splitting, are sentence rules, embedding model, buffer, and
   breakpoint threshold appropriate for the corpus?
7. Does retrieval miss the relevant chunk, or does generation ignore a retrieved
   chunk because the assembled context is too noisy?

This order separates ingestion defects from indexing, retrieval, and generation
defects instead of treating every bad answer as a model problem.

## 16. Concept coverage index

| Topic | Covered here |
|---|---|
| Definition of text chunking | Section 1 |
| Embedding and LLM context limits | Section 2 |
| Pooling loss, topic dilution, and lost in the middle | Sections 2–3 |
| Fixed/character chunking internals and overlap | Section 4 |
| Recursive separator selection, batching, recursion, and termination | Section 5 |
| Multilingual and programming-language separators | Section 5 |
| Semantic sentence buffering, cosine distance, and chunk assembly | Section 6 |
| Percentile, standard deviation, IQR, and gradient breakpoints | Section 6 |
| Markdown heading metadata and two-stage splitting | Section 7 |
| Unstructured partitioning plus `basic` and `by_title` | Section 8 |
| LlamaIndex node parsers, transformations, windows, and interoperability | Section 9 |
| ChunkViz | Section 10 |
| Cross-strategy tradeoffs and evaluation | Sections 11–12 |
| Exact repository implementation versus conceptual material | Sections 13–14 |

## References

- Nelson F. Liu et al., [Lost in the Middle: How Language Models Use Long
  Contexts](https://arxiv.org/abs/2307.03172)
- LangChain, Unstructured, LlamaIndex, and ChunkViz APIs and defaults should be
  checked against the installed versions before copying example code.
