import os
from collections.abc import Iterator
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import pytest
from fastapi.testclient import TestClient

from local_lke.ingestion import IngestionService
from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.storage import (
    Base,
    SqlAlchemyIngestionRepository,
    create_database_engine,
    create_session_factory,
)
from local_lke.web import create_app


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        chat_base_url="http://127.0.0.1:1234/v1",
        chat_model="test-model",
        embedding_model="test-embeddings",
        database_url=f"sqlite+pysqlite:///{tmp_path / 'test.db'}",
        upload_directory=tmp_path / "uploads",
    )


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline(
        chat=FakeChatProvider(),
        embeddings=DeterministicFakeEmbeddings(),
        default_top_k=2,
    )


@pytest.fixture
def ingestion(settings: Settings) -> IngestionService:
    engine = create_database_engine(settings.database_url)
    Base.metadata.create_all(engine)
    return IngestionService(
        SqlAlchemyIngestionRepository(create_session_factory(engine), engine),
        settings,
    )


@pytest.fixture
def client(
    settings: Settings, pipeline: RAGPipeline, ingestion: IngestionService
) -> Iterator[TestClient]:
    with TestClient(create_app(settings, pipeline, ingestion)) as test_client:
        yield test_client
