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


class InterruptedChat(FakeChatProvider):
    def stream(self, prompt: str) -> Iterator[str]:
        del prompt
        yield '{"answer":"partial'
        raise ProviderUnavailableError(
            "The local model stream was interrupted.",
            component="chat.completion",
        )


def test_missing_model_degrades_to_validated_cited_evidence_without_stack_trace(
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

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["citations"][0]["citation_id"] == "C1"
    assert payload["trace"]["generation"]["degraded_reason"] == (
        "local model synthesis was unavailable"
    )
    assert payload["warnings"]
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


def test_interrupted_stream_emits_error_and_never_false_completion(
    ingestion: IngestionService,
) -> None:
    pipeline = RAGPipeline(
        chat=InterruptedChat(),
        embeddings=DeterministicFakeEmbeddings(),
    )
    settings = Settings(_env_file=None)
    with TestClient(create_app(settings, pipeline, ingestion)) as client:
        response = client.post(
            "/api/v1/query/stream",
            json={"question": "What is the priority-one acknowledgement target?"},
        )

    events = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]
    assert response.status_code == 200
    assert events == ["start", "retrieval", "error"]
    assert "completion" not in events
    assert "partial" not in response.text
