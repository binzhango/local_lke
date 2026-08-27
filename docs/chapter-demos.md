# Chapter Demo Launcher

## Start a demo

Local LKE is cumulative: the current service contains every completed chapter.
The launcher selects the appropriate delivery mode and prints a focused path
through the capabilities introduced by one chapter.

```bash
./scripts/demo_chapter.sh list
./scripts/demo_chapter.sh 4
```

The equivalent Make target is:

```bash
make demo CHAPTER=4
```

Chapters 1–6 start the Gradio workbench at `/app`. Chapter 7 generates disposable
administrator and member bearer tokens, starts the secure API-only mode, and
points to `/docs`. The tokens live only in the launcher process environment and
change on every run.

## Chapter menu

| Chapter | Demo focus | Prerequisites |
|---:|---|---|
| 1 | fixture-based cited RAG and stage trace | optional local model server |
| 2 | upload safety, parsing, chunking, immutable versions | PostgreSQL 18 |
| 3 | persistent pgvector, context expansion, image retrieval | PostgreSQL 18; local embedding models |
| 4 | hybrid/corrective/structured retrieval | PostgreSQL 18; indexed collection |
| 5 | validated generation modes, citation registry, degradation | PostgreSQL 18; local chat model for synthesis |
| 6 | datasets, faults, metrics, comparisons, gates | PostgreSQL 18 |
| 7 | bearer authentication, collection ACLs, audit events | PostgreSQL 18 |

The service can still start when the chat provider is unavailable. Model-backed
answers will degrade or report provider health; evidence-only and deterministic
evaluation paths remain useful for demonstrations.

## Environment overrides

```bash
LKE_DEMO_PORT=8010 ./scripts/demo_chapter.sh 5
LKE_DEMO_HOST=127.0.0.1 ./scripts/demo_chapter.sh 7
LKE_DEMO_SKIP_MIGRATE=1 ./scripts/demo_chapter.sh 3
```

The launcher binds to `127.0.0.1:8000` by default even if `.env` contains a
different host or port. This prevents a demo from unexpectedly exposing itself
on the network. `LKE_DEMO_SKIP_MIGRATE=1` is intended only when the database is
already at the current migration head.

## Database setup

Chapters 2–7 apply current migrations before starting. If PostgreSQL is not
ready, the launcher stops with the normal setup command:

```bash
brew services start postgresql@18
make init-postgres
```

Chapter 1 skips migration and can demonstrate the bundled fixture baseline even
while database health is degraded.
