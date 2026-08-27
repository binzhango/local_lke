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

from local_lke.indexing import IndexingService, SqlAlchemyIndexRepository
from local_lke.ingestion import IngestionService
from local_lke.models import (
    ChunkStrategy,
    JobStatus,
    ParserStrategy,
    QueryRequest,
    RetrievalStrategy,
    StructuredQueryPlan,
    StructuredQueryRequest,
)
from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.retrieval import (
    AdvancedRetrievalService,
    StructuredDataService,
    StructuredPlanParser,
)
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
    "indexing_jobs",
    "embedding_profiles",
    "collection_index_profiles",
    "vector_nodes",
    "image_assets",
    "image_embeddings",
    "pipeline_configurations",
    "structured_tables",
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
                chunk_indexes = inspect(engine).get_indexes("chunks")
                assert any(
                    item["name"] == "ix_chunks_search_vector_gin"
                    and item["dialect_options"]["postgresql_using"] == "gin"
                    for item in chunk_indexes
                )
                vector_indexes = inspect(engine).get_indexes("vector_nodes")
                assert any(
                    item["name"] == "ix_vector_nodes_embedding_hnsw"
                    and item["dialect_options"]["postgresql_using"] == "hnsw"
                    for item in vector_indexes
                )
                with engine.connect() as connection:
                    assert connection.exec_driver_sql(
                        "SELECT extversion FROM pg_extension WHERE extname='vector'"
                    ).scalar_one()
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
                persistent = IndexingService(
                    SqlAlchemyIndexRepository(create_session_factory(engine), engine),
                    DeterministicFakeEmbeddings(384),
                    settings,
                )
                indexed = persistent.index_version(job.version_id)
                assert indexed.status is JobStatus.COMPLETED
                assert persistent.state(collection.id).missing_active_chunks == 0
                retrieval = AdvancedRetrievalService(
                    repository=service.repository,
                    embeddings=DeterministicFakeEmbeddings(),
                    chat=FakeChatProvider("PostgreSQL preserves immutable chunk provenance."),
                    settings=settings,
                    indexing=persistent,
                )
                retrieved = retrieval.retrieve(
                    QueryRequest(
                        collection_id=collection.id,
                        question="What preserves immutable chunk provenance?",
                        strategy=RetrievalStrategy.HYBRID,
                    )
                )
                assert retrieved.trace.candidates[0].lexical_rank == 1
                assert retrieved.trace.candidates[0].matched_terms
                structured = StructuredDataService(
                    service.repository,
                    settings,
                    StructuredPlanParser(FakeChatProvider()),
                )
                table = structured.ingest_csv(
                    collection_id=collection.id,
                    filename="acceptance.csv",
                    content=b"name,value\nalpha,1\nbeta,2\n",
                )
                rows = structured.query(
                    StructuredQueryRequest(
                        table_id=table.id,
                        question="List values",
                        plan=StructuredQueryPlan(
                            projections=["name", "value"], limit=1
                        ),
                    )
                )
                assert rows.rows == [{"name": "alpha", "value": 1}]
                assert rows.truncated is True
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
