import os
from collections.abc import Iterator

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import pytest
from fastapi.testclient import TestClient

from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        chat_base_url="http://127.0.0.1:1234/v1",
        chat_model="test-model",
        embedding_model="test-embeddings",
    )


@pytest.fixture
def pipeline() -> RAGPipeline:
    return RAGPipeline(
        chat=FakeChatProvider(),
        embeddings=DeterministicFakeEmbeddings(),
        default_top_k=2,
    )


@pytest.fixture
def client(settings: Settings, pipeline: RAGPipeline) -> Iterator[TestClient]:
    with TestClient(create_app(settings, pipeline)) as test_client:
        yield test_client
