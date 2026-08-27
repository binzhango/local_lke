#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "[1/4] Ensuring Python 3.12 is available..."
uv python install 3.12

echo "[2/4] Installing locked project and development dependencies..."
uv sync --locked

if [[ ! -f .env ]]; then
  echo "[3/4] Creating .env from the safe example..."
  cp .env.example .env
else
  echo "[3/4] Keeping the existing .env file..."
fi

echo "[4/4] Running offline foundation checks..."
uv run --locked lke doctor --skip-providers --skip-database

echo
echo "Foundation is ready."
echo "1. Run 'make init-postgres' to create/migrate the local PostgreSQL 18 database."
echo "2. Load a model in LM Studio and enable its local server."
echo "3. Set LKE_CHAT_MODEL in .env to the model identifier exposed by that server."
echo "4. Run 'make doctor', then 'make serve'."
echo "5. Open http://127.0.0.1:8000/app"
