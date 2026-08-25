.PHONY: init serve doctor test test-live lint typecheck check

init:
	./scripts/init_environment.sh

serve:
	uv run --locked lke serve

doctor:
	uv run --locked lke doctor

test:
	uv run --locked pytest -m "not live"

test-live:
	uv run --locked pytest -m live --force-enable-socket

lint:
	uv run --locked ruff check .

typecheck:
	uv run --locked mypy

check: lint typecheck test
