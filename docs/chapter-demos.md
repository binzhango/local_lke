# Simplified Chapter Demo

## One setup, one cumulative service

Local LKE is cumulative. Do not install dependencies or initialize a separate
database for each chapter.

Run setup once:

```bash
brew services start postgresql@18
make demo-setup
```

Then start one service whenever you want to demo Chapters 1–7:

```bash
make demo
```

Open:

- `http://127.0.0.1:8000/app` for the interactive workbench;
- `http://127.0.0.1:8000/docs` for FastAPI endpoints grouped under Chapter 1
  through Chapter 7 tags.

The start command checks migrations but does not reinstall Python or libraries.
The current database, collections, indexes, and evaluation datasets are reused.

## Tagged FastAPI walkthrough

| Tag | Demo focus |
|---|---|
| Chapter 1 · Baseline RAG | health, bundled fixture query, SSE, citations |
| Chapter 2 · Ingestion | collections, safe upload, jobs, versions, previews |
| Chapter 3 · Indexing | pgvector state, retrieval lab, images |
| Chapter 4 · Retrieval | hybrid/corrective query and structured tables |
| Chapter 5 · Generation | output modes, validation, evidence, citations |
| Chapter 6 · Evaluation | datasets, runs, faults, comparisons, gates |
| Chapter 7 · Security | collection grants/revocation and audit events |

Some endpoints belong to multiple chapters and appear under every relevant tag.
For example, `/api/v1/query` demonstrates the Chapter 1 baseline, Chapter 4
retrieval, and Chapter 5 output contract.

## Secure Chapter 7 demonstration

Chapter 7 needs a separate secure-mode restart because the normal workbench calls
services directly and is deliberately not mounted behind decorative UI auth.

Stop `make demo`, then run:

```bash
make demo-secure
```

The launcher prints disposable administrator and member tokens. Open `/docs`,
select **Authorize**, paste one token, and exercise the Chapter 7-tagged endpoints.
Tokens change on each run and are never written to disk.

## Optional focused guidance

The existing focused launcher is still available, but it uses the same cumulative
code and locked environment:

```bash
./scripts/demo_chapter.sh list
make demo-chapter CHAPTER=4
```

It prints a short walkthrough for the selected chapter. It does not install a
chapter-specific dependency set or create a chapter-specific database.

## Overrides

```bash
LKE_DEMO_PORT=8010 make demo
LKE_DEMO_PORT=8010 make demo-secure
```

The demo binds to `127.0.0.1` by default so it is not unexpectedly exposed on the
network.
