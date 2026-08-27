import os

import pytest

from local_lke.factory import create_pipeline
from local_lke.providers import LocalHuggingFaceEmbeddings
from local_lke.settings import Settings


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("LKE_RUN_LIVE_TESTS") != "1",
    reason="set LKE_RUN_LIVE_TESTS=1 to test a running local model",
)
def test_live_local_provider_smoke() -> None:
    pipeline = create_pipeline(Settings())

    assert pipeline.chat.check_models()
    assert pipeline.chat.check_completion()
    assert pipeline.embeddings.check_initialization()


@pytest.mark.live
@pytest.mark.skipif(
    os.getenv("LKE_RUN_LIVE_TESTS") != "1",
    reason="set LKE_RUN_LIVE_TESTS=1 to initialize the real local BGE model",
)
def test_default_bge_profile_initializes_at_the_schema_dimension() -> None:
    settings = Settings(_env_file=None)
    embeddings = LocalHuggingFaceEmbeddings(
        settings.embedding_model,
        revision=settings.embedding_model_revision,
        normalized=settings.embedding_normalize,
        document_prefix=settings.embedding_document_prefix,
        query_prefix=settings.embedding_query_prefix,
    )

    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert embeddings.check_initialization() == "embedding model initialized (384 dimensions)"
