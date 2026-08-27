import os

import pytest

from local_lke.factory import create_pipeline
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
