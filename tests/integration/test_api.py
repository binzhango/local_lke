from fastapi.testclient import TestClient

QUESTION = "How quickly does Atlas acknowledge a priority-one incident?"


def test_health_has_component_status(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["components"]) == {"database", "chat", "embeddings"}


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


def test_collection_upload_job_preview_and_version_history(client: TestClient) -> None:
    created = client.post("/api/v1/collections", json={"name": "Engineering"})
    collection_id = created.json()["id"]

    uploaded = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files={"files": ("guide.md", b"# Runbook\n\nRestart safely.", "text/markdown")},
        data={"parser_strategy": "fast", "chunk_strategy": "markdown"},
    )

    assert created.status_code == 201
    assert uploaded.status_code == 202
    job = uploaded.json()[0]
    assert job["status"] == "queued"

    status = client.get(f"/api/v1/jobs/{job['id']}")
    documents = client.get(f"/api/v1/collections/{collection_id}/documents")
    version_id = status.json()["version_id"]
    preview = client.get(f"/api/v1/document-versions/{version_id}/preview")

    assert status.json()["progress"] == 100
    assert status.json()["chunk_count"] >= 1
    assert documents.json()[0]["versions"][0]["active"] is True
    assert preview.json()["elements"][0]["heading_path"] == ["Runbook"]
    assert preview.json()["chunks"][0]["document_version_id"] == version_id


def test_upload_security_error_is_structured(client: TestClient) -> None:
    collection_id = client.post("/api/v1/collections", json={"name": "Security"}).json()["id"]

    response = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files={"files": ("fake.pdf", b"not a pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "mime_mismatch"
    assert "traceback" not in response.text.lower()


def test_batch_is_fully_validated_before_any_job_is_queued(client: TestClient) -> None:
    collection_id = client.post(
        "/api/v1/collections", json={"name": "Atomic validation"}
    ).json()["id"]

    response = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files=[
            ("files", ("valid.txt", b"Valid text.", "text/plain")),
            ("files", ("invalid.exe", b"not supported", "application/octet-stream")),
        ],
    )

    assert response.status_code == 422
    documents = client.get(f"/api/v1/collections/{collection_id}/documents")
    assert documents.json() == []
