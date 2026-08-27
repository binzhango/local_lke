from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

QUESTION = "How quickly does Atlas acknowledge a priority-one incident?"


def test_health_has_component_status(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()["components"]) == {
        "database",
        "chat",
        "embeddings",
        "vector_index",
    }


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
    query_schema = contract["components"]["schemas"]["QueryRequest"]
    assert "output_mode" in query_schema["properties"]
    assert "schema_name" in query_schema["properties"]


def test_query_supports_validated_structured_and_evidence_only_modes(
    client: TestClient,
) -> None:
    structured = client.post(
        "/api/v1/query",
        json={
            "question": QUESTION,
            "top_k": 1,
            "output_mode": "structured",
            "schema_name": "fact_list",
        },
    )
    evidence = client.post(
        "/api/v1/query",
        json={"question": QUESTION, "top_k": 1, "output_mode": "evidence_only"},
    )

    assert structured.status_code == 200
    structured_payload = structured.json()
    assert structured_payload["structured_result"]["schema_name"] == "fact_list"
    assert structured_payload["structured_result"]["facts"][0]["citation_ids"] == ["C1"]
    assert structured_payload["trace"]["generation"]["model_output_committed"] is True
    assert evidence.status_code == 200
    evidence_payload = evidence.json()
    assert evidence_payload["structured_result"] is None
    assert evidence_payload["trace"]["generation"]["attempts"] == 0
    assert evidence_payload["warnings"] == [
        "Evidence-only mode bypassed model synthesis."
    ]


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


def test_persisted_hybrid_query_returns_stage_trace_and_versioned_source(
    client: TestClient,
) -> None:
    collection_id = client.post(
        "/api/v1/collections", json={"name": "Retrieval API"}
    ).json()["id"]
    client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files={
            "files": (
                "zephyr.txt",
                b"The Zephyr deployment code is ZXQ-4917.",
                "text/plain",
            )
        },
        data={"chunk_strategy": "recursive"},
    )

    response = client.post(
        "/api/v1/query",
        json={
            "collection_id": collection_id,
            "question": "What is the Zephyr deployment code ZXQ-4917?",
            "strategy": "hybrid",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["trace"]["retrieval"]["strategy"] == "hybrid"
    assert payload["trace"]["retrieval"]["candidates"][0]["lexical_rank"] == 1
    citation = payload["citations"][0]
    source = client.get(f"/api/v1/sources/{citation['source_id']}")
    assert source.status_code == 200
    assert "ZXQ-4917" in source.text


def test_structured_upload_and_query_api_never_accepts_raw_sql(client: TestClient) -> None:
    collection_id = client.post(
        "/api/v1/collections", json={"name": "Structured API"}
    ).json()["id"]
    upload = client.post(
        f"/api/v1/collections/{collection_id}/structured-tables",
        files={"file": ("sales.csv", b"region,revenue\nEast,100\nWest,250\n", "text/csv")},
    )
    table_id = upload.json()["id"]
    query = client.post(
        "/api/v1/structured/query",
        json={
            "table_id": table_id,
            "question": "Largest revenue",
            "plan": {
                "projections": ["region", "revenue"],
                "filters": [],
                "group_by": [],
                "aggregations": [],
                "order_by": [{"column": "revenue", "direction": "desc"}],
                "limit": 1,
            },
        },
    )
    rejected = client.post(
        "/api/v1/structured/query",
        json={
            "table_id": table_id,
            "question": "Destroy data",
            "plan": {"raw_sql": "DROP TABLE collections"},
        },
    )

    assert upload.status_code == 201
    assert query.status_code == 200
    assert query.json()["rows"] == [{"region": "West", "revenue": 250}]
    assert rejected.status_code == 422
    assert rejected.json()["error"]["code"] == "validation_error"


def test_upload_builds_persistent_index_and_retrieval_lab_exposes_context_decisions(
    client: TestClient,
) -> None:
    collection_id = client.post(
        "/api/v1/collections", json={"name": "Persistent index API"}
    ).json()["id"]
    upload = client.post(
        f"/api/v1/collections/{collection_id}/documents",
        files={
            "files": (
                "policy.txt",
                b"Priority one is acknowledged in fifteen minutes. Updates follow every hour.",
                "text/plain",
            )
        },
        data={"chunk_strategy": "recursive"},
    )
    state = client.get(f"/api/v1/collections/{collection_id}/index-state")
    lab = client.post(
        "/api/v1/retrieval-lab",
        json={
            "collection_id": collection_id,
            "question": "When is priority one acknowledged?",
            "top_k": 3,
            "expansion": "sentence_window",
            "token_budget": 100,
        },
    )

    assert upload.status_code == 202
    assert state.json()["missing_active_chunks"] == 0
    assert state.json()["active_profile"]["dimension"] == 64
    assert lab.status_code == 200
    assert lab.json()["final_context"][0]["trigger_node_id"]
    assert lab.json()["final_token_count"] <= 100


def test_multimodal_api_returns_provenance_without_chat_claims(client: TestClient) -> None:
    collection_id = client.post(
        "/api/v1/collections", json={"name": "Multimodal API"}
    ).json()["id"]
    output = BytesIO()
    Image.new("RGB", (24, 24), (255, 0, 0)).save(output, format="PNG")
    content = output.getvalue()
    uploaded = client.post(
        f"/api/v1/collections/{collection_id}/images",
        files={"file": ("red.png", content, "image/png")},
    )
    searched = client.post(
        f"/api/v1/collections/{collection_id}/images/search/text",
        data={"query": "red image", "top_k": 1},
    )
    content_response = client.get(uploaded.json()["content_url"])

    assert uploaded.status_code == 201
    assert searched.status_code == 200
    assert searched.json()["hits"][0]["image"]["id"] == uploaded.json()["id"]
    assert "answer" not in searched.json()
    assert content_response.status_code == 200
