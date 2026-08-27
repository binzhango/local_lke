"""Local LKE command-line entry point."""

import argparse
import json
import subprocess
from pathlib import Path

import uvicorn
from alembic import command
from alembic.config import Config

from local_lke.errors import ProviderUnavailableError
from local_lke.factory import (
    create_indexing_services,
    create_ingestion_service,
    create_pipeline,
)
from local_lke.logging import configure_logging
from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings, get_settings
from local_lke.web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lke", description="Local LKE RAG workbench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("serve", help="Run FastAPI and Gradio in one process")
    doctor = subparsers.add_parser("doctor", help="Check configuration and local providers")
    doctor.add_argument(
        "--skip-providers",
        action="store_true",
        help="Run only deterministic foundation checks",
    )
    doctor.add_argument(
        "--skip-database",
        action="store_true",
        help="Skip PostgreSQL binary and connection checks",
    )
    subparsers.add_parser("migrate", help="Apply database migrations")
    openapi = subparsers.add_parser("openapi", help="Export the OpenAPI contract")
    openapi.add_argument("--output", default=".artifacts/openapi.json")
    return parser


def main() -> None:
    configure_logging()
    arguments = build_parser().parse_args()
    settings = get_settings()
    if arguments.command == "serve":
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port)
    elif arguments.command == "doctor":
        raise SystemExit(
            run_doctor(
                settings,
                skip_providers=arguments.skip_providers,
                skip_database=arguments.skip_database,
            )
        )
    elif arguments.command == "migrate":
        migrate(settings)
    elif arguments.command == "openapi":
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(create_app(settings).openapi(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(f"OpenAPI contract written to {output}")


def run_doctor(settings: Settings, *, skip_providers: bool, skip_database: bool = False) -> int:
    print(json.dumps({"configuration": settings.redacted_summary}, indent=2))
    if skip_providers:
        pipeline = RAGPipeline(
            chat=FakeChatProvider(),
            embeddings=DeterministicFakeEmbeddings(),
            default_top_k=settings.default_top_k,
        )
        response = pipeline.query("How quickly does Atlas acknowledge a priority-one incident?")
        print(
            json.dumps(
                {
                    "foundation": "ok",
                    "answer_status": response.status,
                    "citations": [item.source_id for item in response.citations],
                },
                indent=2,
            )
        )
        failures = 0
    else:
        pipeline = create_pipeline(settings)
        checks = [
            ("models", pipeline.chat.check_models),
            ("completion", pipeline.chat.check_completion),
            ("embeddings", pipeline.embeddings.check_initialization),
        ]
        failures = 0
        for name, check in checks:
            try:
                print(f"[ok] {name}: {check()}")
            except ProviderUnavailableError as exc:
                failures += 1
                print(f"[unavailable] {name}: {exc}")

    if not skip_database:
        psql = settings.postgres_bin_directory / "psql"
        if not psql.is_file():
            failures += 1
            print(f"[unavailable] postgresql: expected PostgreSQL 18 binary at {psql}")
        else:
            version_result = subprocess.run(
                [str(psql), "--version"], capture_output=True, text=True, check=False
            )
            print(f"[ok] postgresql binary: {version_result.stdout.strip()}")
        try:
            ingestion = create_ingestion_service(settings)
            print(f"[ok] database: {ingestion.check_health()}")
        except Exception as exc:
            failures += 1
            print(f"[unavailable] database: {exc}")
        else:
            try:
                indexing, _multimodal = create_indexing_services(
                    settings, pipeline, ingestion
                )
                print(f"[ok] vector index: {indexing.check_health()}")
            except Exception as exc:
                failures += 1
                print(f"[unavailable] vector index: {exc}")
    return 1 if failures else 0


def migrate(settings: Settings) -> None:
    configuration = Config("alembic.ini")
    configuration.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
    command.upgrade(configuration, "head")
    print("Database migrations applied.")
