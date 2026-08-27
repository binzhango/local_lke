from collections.abc import Iterator

from fastapi.testclient import TestClient

from local_lke.errors import ProviderUnavailableError
from local_lke.ingestion import IngestionService
from local_lke.providers import DeterministicFakeEmbeddings, FakeChatProvider
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web import create_app


class UnavailableChat(FakeChatProvider):
    def check_models(self) -> str:
        raise ProviderUnavailableError(
            "Start the local model server and load the configured model.",
            component="chat.models",
        )

    def generate(self, prompt: str) -> str:
        del prompt
        raise ProviderUnavailableError(
            "Start the local model server and load the configured model.",
            component="chat.completion",
        )

    def stream(self, prompt: str) -> Iterator[str]:
        del prompt
        raise ProviderUnavailableError(
            "Start the local model server and load the configured model.",
            component="chat.completion",
        )
        yield "unreachable"


def test_missing_model_maps_to_actionable_error_without_stack_trace(
    ingestion: IngestionService,
) -> None:
    pipeline = RAGPipeline(
        chat=UnavailableChat(),
        embeddings=DeterministicFakeEmbeddings(),
    )
    settings = Settings(_env_file=None, chat_model="missing-model")
    with TestClient(create_app(settings, pipeline, ingestion)) as client:
        response = client.post(
            "/api/v1/query",
            json={"question": "What is the priority-one acknowledgement target?"},
        )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "provider_unavailable",
        "message": "Start the local model server and load the configured model.",
        "component": "chat.completion",
    }
    assert "traceback" not in response.text.lower()


def test_missing_model_health_is_degraded_not_a_crash(
    ingestion: IngestionService,
) -> None:
    pipeline = RAGPipeline(
        chat=UnavailableChat(),
        embeddings=DeterministicFakeEmbeddings(),
    )
    settings = Settings(_env_file=None, chat_model="missing-model")
    with TestClient(create_app(settings, pipeline, ingestion)) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["components"]["chat"]["status"] == "unavailable"
    assert "Start the local model server" in response.json()["components"]["chat"]["detail"]
