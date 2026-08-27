#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POSTGRES_BIN_DIR="${LKE_POSTGRES_BIN_DIRECTORY:-/opt/homebrew/opt/postgresql@18/bin}"
PSQL="${POSTGRES_BIN_DIR}/psql"
CREATEDB="${POSTGRES_BIN_DIR}/createdb"
PG_ISREADY="${POSTGRES_BIN_DIR}/pg_isready"

cd "${PROJECT_DIR}"

for executable in "${PSQL}" "${CREATEDB}" "${PG_ISREADY}"; do
  if [[ ! -x "${executable}" ]]; then
    echo "PostgreSQL 18 executable not found: ${executable}"
    echo "Install it with 'brew install postgresql@18' or set LKE_POSTGRES_BIN_DIRECTORY."
    exit 1
  fi
done

if ! "${PG_ISREADY}" --quiet; then
  echo "PostgreSQL 18 is not accepting local connections."
  echo "On Homebrew, start it with: brew services start postgresql@18"
  exit 1
fi

if ! "${PSQL}" --dbname postgres --tuples-only --no-align \
  --command "SELECT 1 FROM pg_database WHERE datname = 'local_lke'" | grep -qx "1"; then
  "${CREATEDB}" local_lke
  echo "Created database local_lke."
else
  echo "Database local_lke already exists."
fi

uv run --locked lke migrate
echo "PostgreSQL 18 foundation is ready."
