from fastapi.testclient import TestClient

QUESTION = "How quickly does Atlas acknowledge a priority-one incident?"


def test_health_has_component_status(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["components"]) == {"chat", "embeddings"}


def test_query_returns_the_stable_answer_contract(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": QUESTION, "top_k": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "answered"
    assert payload["citations"][0]["source_id"] == "fixture:atlas-support"
    assert payload["trace"]["retrieved"][0]["rank"] == 1


def test_stream_has_ordered_sse_events(client: TestClient) -> None:
    response = client.post("/api/v1/query/stream", json={"question": QUESTION, "top_k": 1})
    event_names = [
        line.removeprefix("event: ")
        for line in response.text.splitlines()
        if line.startswith("event: ")
    ]

    assert response.status_code == 200
    assert event_names[0:2] == ["start", "retrieval"]
    assert "delta" in event_names
    assert event_names[-1] == "completion"


def test_validation_errors_are_structured(client: TestClient) -> None:
    response = client.post("/api/v1/query", json={"question": ""})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"
    assert "traceback" not in response.text.lower()


def test_source_citation_target_is_readable(client: TestClient) -> None:
    response = client.get("/api/v1/sources/fixture%3Aatlas-support")

    assert response.status_code == 200
    assert "15 minutes" in response.text


def test_openapi_contains_query_contract(client: TestClient) -> None:
    contract = client.app.openapi()

    assert "/api/v1/query" in contract["paths"]
    assert "AnswerResponse" in contract["components"]["schemas"]

