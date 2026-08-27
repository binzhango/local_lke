#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMAND="${1:-start}"

usage() {
  cat <<'EOF'
Usage: ./scripts/demo.sh [setup|start|secure|help]

  setup   Install the locked environment and initialize PostgreSQL once.
  start   Start one cumulative Chapters 1-7 demo server (default).
  secure  Start the same server in Chapter 7 secure API mode with temporary tokens.
  help    Show this message.

After setup, start/secure never reinstall Python or project dependencies.
FastAPI endpoints are grouped by chapter at http://127.0.0.1:8000/docs.
EOF
}

if [[ "${COMMAND}" == "help" || "${COMMAND}" == "--help" || "${COMMAND}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ ! "${COMMAND}" =~ ^(setup|start|secure)$ ]]; then
  usage >&2
  exit 2
fi

cd "${PROJECT_DIR}"

if [[ "${COMMAND}" == "setup" ]]; then
  ./scripts/init_environment.sh
  echo
  echo "Initializing the single cumulative Chapters 1-7 database..."
  if ! ./scripts/init_postgres.sh; then
    echo >&2
    echo "PostgreSQL must be running before setup can finish." >&2
    echo "On Homebrew: brew services start postgresql@18" >&2
    echo "Then rerun: make demo-setup" >&2
    exit 1
  fi
  echo
  echo "Demo setup is complete. Start everything with: make demo"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Run 'make demo-setup' once." >&2
  exit 1
fi

export LKE_HOST="${LKE_DEMO_HOST:-127.0.0.1}"
export LKE_PORT="${LKE_DEMO_PORT:-8000}"

if [[ "${COMMAND}" == "secure" ]]; then
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

echo "Checking the cumulative database schema..."
if ! uv run --locked lke migrate; then
  echo >&2
  echo "The one-time demo setup is incomplete. Run: make demo-setup" >&2
  exit 1
fi

BASE_URL="http://${LKE_HOST}:${LKE_PORT}"
echo
echo "Local LKE cumulative Chapters 1-7 demo"
echo "======================================"
echo "Chapter-tagged FastAPI: ${BASE_URL}/docs"

if [[ "${COMMAND}" == "secure" ]]; then
  cat <<EOF
Secure API mode:         enabled
Gradio workbench:        intentionally disabled
Administrator token:    ${ADMIN_TOKEN}
Member token:           ${MEMBER_TOKEN}

Use FastAPI's Authorize button with either disposable token.
EOF
else
  cat <<EOF
Gradio workbench:        ${BASE_URL}/app
Security mode:           disabled for the interactive cumulative walkthrough

Expand Chapter 1 through Chapter 7 in /docs to demo each capability without
restarting the service or installing anything chapter by chapter.
EOF
fi

echo
echo "Stop the demo with Ctrl-C."
echo
exec uv run --locked lke serve
