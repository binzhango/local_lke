# Chapter 2 Learning Notes: Loading and Chunking

Chapter 1 proved the RAG loop with controlled files. Chapter 2 addresses the
less glamorous but decisive question: how does an arbitrary document become
reliable evidence?

## 1. Ingestion is an ETL pipeline

A document loader does more than return text. It extracts bytes from a format,
transforms them into normalized elements, and loads those elements plus their
provenance into durable storage. If a parser loses a page, heading, table
boundary, or reading-order clue, retrieval cannot reconstruct it later.

The durable unit is therefore not “some text.” It is:

- the logical document;
- an immutable version of its exact bytes and pipeline configuration;
- ordered elements such as titles, paragraphs, and tables;
- derived chunks linked back to their parent element and locator.

This is the practical meaning of “garbage in, garbage out” for RAG.

## 2. Parsing and chunking are separate decisions

Parsing answers “what structure did the source contain?” Chunking answers “what
units should later retrieval compare?” Keeping them separate lets us preview
parser fidelity, change a chunking policy without pretending the source changed,
and trace every chunk back to the normalized evidence that created it.

Markdown already exposes heading structure. Plain text exposes reliable lines
but little document semantics. PDFs contain positioned elements whose apparent
reading order may differ from the stored order, especially with columns,
headers, scans, and tables. That is why PDF extraction has both a fast path and
a slower layout-aware path—and why neither is treated as perfect.

## 3. The chunk-size tradeoff

Small chunks improve retrieval precision but may omit the explanation around a
fact. Large chunks preserve context but can dilute the embedding and waste the
generation context window. Overlap can protect boundary facts, but too much
overlap produces duplicate evidence and misleadingly repetitive retrieval.

The implemented strategies make those tradeoffs visible:

- Recursive splitting prefers paragraphs, lines, sentences, words, then
  characters while enforcing a size ceiling.
- Heading-aware splitting preserves the Markdown section path, then recursively
  sizes content inside that structure.
- Experimental semantic splitting uses local TF-IDF similarity to detect topic
  shifts between adjacent sentences. It is deterministic and network-free, but
  lexical similarity is only a proxy for meaning.

Chapter 3 will test the retrieval consequences and introduce small-child /
large-parent context expansion. Chapter 2 only guarantees that multiple
strategies are reproducible and preserve provenance.

## 4. Idempotency is both performance and correctness

Re-uploading an unchanged file should not create a second set of elements and
chunks. The content hash identifies the exact bytes; the pipeline hash identifies
the parser/chunker contract. Together they distinguish three cases:

1. Same bytes and same pipeline: skip all repeated work.
2. Different bytes: create a new immutable content version.
3. Same bytes but changed pipeline: create a new derived version for the new
   interpretation.

Exactly one version is active for a logical document. Old versions remain for
provenance, debugging, and reproducibility.

## 5. Uploaded content is untrusted data

An upload endpoint is a security boundary even for a loopback-only application.
The filename cannot select a filesystem path. Extension and MIME must agree with
basic content evidence. Size limits are applied before expensive parsing.
Encrypted and malformed PDFs fail closed. Text must decode as UTF-8. Parsed
content is data and is never executed.

These controls reduce risk for a local single-user system. They do not provide
authentication, tenant isolation, malware scanning, or complete protection from
prompt injection; later chapters address the relevant downstream controls.

## 6. What to inspect when ingestion looks wrong

Start with the parser preview, not retrieval. Check element order, page/line
locators, heading paths, table boundaries, and warnings. Then inspect chunk size,
overlap, parent IDs, repeated-content removal, and active version. Only after
those look correct does it make sense to debug embeddings or ranking.
