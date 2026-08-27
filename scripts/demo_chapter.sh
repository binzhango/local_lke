#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHAPTER="${1:-}"

usage() {
  cat <<'EOF'
Usage: ./scripts/demo_chapter.sh <chapter>

Start the cumulative Local LKE service with the delivery mode and demo notes for
one chapter. Supported chapters: 1, 2, 3, 4, 5, 6, 7.

Environment overrides:
  LKE_DEMO_HOST          Bind host (default: 127.0.0.1)
  LKE_DEMO_PORT          Bind port (default: 8000)
  LKE_DEMO_SKIP_MIGRATE  Set to 1 to skip migrations for Chapters 2-7

Use "list" instead of a chapter number to print the chapter menu.
EOF
}

chapter_menu() {
  cat <<'EOF'
1  Naive cited RAG baseline
2  Safe versioned ingestion and chunk inspection
3  Persistent pgvector and multimodal indexing
4  Hybrid, corrective, and structured retrieval
5  Validated generation, citations, and safe degradation
6  Persisted evaluation datasets, runs, faults, and regression gates
7  Bearer authentication, collection ACLs, and audit evidence
EOF
}

if [[ "${CHAPTER}" == "list" ]]; then
  chapter_menu
  exit 0
fi

if [[ ! "${CHAPTER}" =~ ^[1-7]$ ]]; then
  usage >&2
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run ./scripts/init_environment.sh first." >&2
  exit 1
fi

cd "${PROJECT_DIR}"

export LKE_HOST="${LKE_DEMO_HOST:-127.0.0.1}"
export LKE_PORT="${LKE_DEMO_PORT:-8000}"

if [[ "${CHAPTER}" == "7" ]]; then
  if command -v openssl >/dev/null 2>&1; then
    ADMIN_TOKEN="$(openssl rand -hex 24)"
    MEMBER_TOKEN="$(openssl rand -hex 24)"
  else
    ADMIN_TOKEN="$(LC_ALL=C od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
    MEMBER_TOKEN="$(LC_ALL=C od -An -N24 -tx1 /dev/urandom | tr -d ' \n')"
  fi
  export LKE_AUTH_ENABLED=true
  export LKE_AUTH_CREDENTIALS_JSON="[{\"principal_id\":\"demo-admin\",\"display_name\":\"Demo administrator\",\"global_role\":\"admin\",\"token\":\"${ADMIN_TOKEN}\"},{\"principal_id\":\"demo-member\",\"display_name\":\"Demo member\",\"global_role\":\"member\",\"token\":\"${MEMBER_TOKEN}\"}]"
else
  export LKE_AUTH_ENABLED=false
fi

if [[ "${CHAPTER}" != "1" && "${LKE_DEMO_SKIP_MIGRATE:-0}" != "1" ]]; then
  echo "Applying database migrations for the Chapter ${CHAPTER} demo..."
  if ! uv run --locked lke migrate; then
    echo >&2
    echo "Database setup failed. Start PostgreSQL 18 and run 'make init-postgres'," >&2
    echo "then retry this launcher." >&2
    exit 1
  fi
fi

BASE_URL="http://${LKE_HOST}:${LKE_PORT}"

echo
echo "Local LKE · Chapter ${CHAPTER} demo"
echo "=================================="

case "${CHAPTER}" in
  1)
    cat <<EOF
Focus: naive cited RAG baseline
Open:  ${BASE_URL}/app
Try:   How quickly does Atlas acknowledge a priority-one incident?
Watch: the answer, source citation, retrieved chunks, and stage timings.
EOF
    ;;
  2)
    cat <<EOF
Focus: safe versioned ingestion and chunk inspection
Open:  ${BASE_URL}/app
Try:   create a collection, upload fixtures/atlas-support.md, then inspect its
       job, immutable version, parsed elements, source locators, and chunks.
EOF
    ;;
  3)
    cat <<EOF
Focus: persistent text and image indexing
Open:  ${BASE_URL}/app
Try:   index the Chapter 2 collection, inspect index health, run retrieval lab,
       and compare sentence-window, parent, and multi-granularity expansion.
EOF
    ;;
  4)
    cat <<EOF
Focus: advanced retrieval
Open:  ${BASE_URL}/app
Try:   compare dense and hybrid search, inspect RRF/reranking/context decisions,
       then upload a CSV and execute an allowlisted structured query.
EOF
    ;;
  5)
    cat <<EOF
Focus: validated generation
Open:  ${BASE_URL}/app
Try:   compare conversational, fact_list structured, and evidence-only output;
       inspect citation IDs, validation attempts, and degradation behavior.
EOF
    ;;
  6)
    cat <<EOF
Focus: deterministic evaluation and regression gates
Open:  ${BASE_URL}/app
Try:   create the provided Atlas evaluation dataset, run it, inject provider
       faults, inspect per-case metrics, then compare a candidate to a baseline.
EOF
    ;;
  7)
    cat <<EOF
Focus: governed API security
Open:  ${BASE_URL}/docs
Note:  /app is intentionally disabled in secure mode.

Disposable credentials for this process:
  admin token:  ${ADMIN_TOKEN}
  member token: ${MEMBER_TOKEN}

Create a collection as the member:
  curl -sS -X POST '${BASE_URL}/api/v1/collections' \\
    -H 'Authorization: Bearer ${MEMBER_TOKEN}' \\
    -H 'Content-Type: application/json' \\
    -d '{"name":"Chapter 7 private demo"}'

Then use the returned collection ID to grant access and inspect audit events.
Credentials disappear when this service stops.
EOF
    ;;
esac

echo
echo "Provider note: model-backed answers require the configured local model server."
echo "Stop the demo with Ctrl-C."
echo

exec uv run --locked lke serve
