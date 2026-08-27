.PHONY: init init-postgres migrate serve demo demo-setup demo-secure demo-chapter doctor test test-live lint typecheck check

init:
	./scripts/init_environment.sh

init-postgres:
	./scripts/init_postgres.sh

migrate:
	uv run --locked lke migrate

serve:
	uv run --locked lke serve

demo:
	@if test -n "$(CHAPTER)"; then ./scripts/demo_chapter.sh "$(CHAPTER)"; else ./scripts/demo.sh start; fi

demo-setup:
	./scripts/demo.sh setup

demo-secure:
	./scripts/demo.sh secure

demo-chapter:
	@test -n "$(CHAPTER)" || (echo "Usage: make demo-chapter CHAPTER=1..7" && exit 2)
	./scripts/demo_chapter.sh "$(CHAPTER)"

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
