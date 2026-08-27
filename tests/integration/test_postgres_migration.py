from __future__ import annotations

import getpass
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from local_lke.ingestion import IngestionService
from local_lke.models import ChunkStrategy, JobStatus, ParserStrategy
from local_lke.settings import Settings
from local_lke.storage import SqlAlchemyIngestionRepository, create_session_factory

POSTGRES_BIN = Path("/opt/homebrew/opt/postgresql@18/bin")
EXPECTED_TABLES = {
    "alembic_version",
    "chunks",
    "collections",
    "document_elements",
    "document_versions",
    "documents",
    "ingestion_jobs",
    "pipeline_configurations",
}


@pytest.mark.skipif(
    not (POSTGRES_BIN / "initdb").is_file(),
    reason="Homebrew PostgreSQL 18 is not installed",
)
def test_migrations_apply_to_an_empty_postgresql_18_database() -> None:
    port = 55_000 + (os.getpid() % 500)
    with TemporaryDirectory(prefix="lke-pg18-", dir="/tmp") as temporary:
        root = Path(temporary)
        data = root / "data"
        socket_directory = root / "socket"
        socket_directory.mkdir()
        _run(
            POSTGRES_BIN / "initdb",
            "-D",
            data,
            "--auth=trust",
            "--no-locale",
            "--encoding=UTF8",
            "--username",
            getpass.getuser(),
        )
        options = f"-F -k {socket_directory} -p {port} -c listen_addresses=''"
        _run(
            POSTGRES_BIN / "pg_ctl",
            "-D",
            data,
            "-l",
            root / "postgres.log",
            "-o",
            options,
            "-w",
            "start",
        )
        try:
            _run(
                POSTGRES_BIN / "createdb",
                "-h",
                socket_directory,
                "-p",
                str(port),
                "local_lke",
            )
            database_url = (
                f"postgresql+psycopg://{getpass.getuser()}@/local_lke"
                f"?host={quote(str(socket_directory))}&port={port}&connect_timeout=5"
            )
            configuration = Config("alembic.ini")
            configuration.set_main_option("sqlalchemy.url", database_url)
            command.upgrade(configuration, "head")

            engine = create_engine(database_url)
            try:
                assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES
                indexes = inspect(engine).get_indexes("document_versions")
                assert any(
                    item["name"] == "uq_versions_one_active" and item["unique"] for item in indexes
                )
                settings = Settings(
                    _env_file=None,
                    database_url=database_url,
                    upload_directory=root / "uploads",
                )
                service = IngestionService(
                    SqlAlchemyIngestionRepository(create_session_factory(engine), engine),
                    settings,
                )
                collection = service.create_collection("PostgreSQL acceptance")
                job = service.ingest(
                    collection_id=collection.id,
                    filename="acceptance.txt",
                    content_type="text/plain",
                    content=b"PostgreSQL preserves immutable chunk provenance.",
                    parser_strategy=ParserStrategy.FAST,
                    chunk_strategy=ChunkStrategy.RECURSIVE,
                    chunk_size=200,
                    chunk_overlap=20,
                )
                assert job.status is JobStatus.COMPLETED
                assert job.version_id is not None
                assert service.preview(job.version_id).chunks[0].locator == "lines:1-1"
            finally:
                engine.dispose()
        finally:
            _run(POSTGRES_BIN / "pg_ctl", "-D", data, "-m", "fast", "-w", "stop")


def _run(executable: Path, *arguments: object) -> None:
    result = subprocess.run(
        [str(executable), *(str(item) for item in arguments)],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    if result.returncode:
        raise AssertionError(
            f"{executable.name} failed ({result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
