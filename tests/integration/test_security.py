import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from local_lke.indexing import IndexingService, MultimodalIndexingService
from local_lke.ingestion import IngestionService
from local_lke.rag import RAGPipeline
from local_lke.settings import Settings
from local_lke.web import create_app

ADMIN_TOKEN = "admin-token-000000"
ALICE_TOKEN = "alice-token-000000"
BOB_TOKEN = "bob-token-00000000"


@pytest.fixture
def secure_client(
    settings: Settings,
    pipeline: RAGPipeline,
    ingestion: IngestionService,
    indexing: IndexingService,
    multimodal: MultimodalIndexingService,
) -> Iterator[TestClient]:
    credentials = [
        {
            "principal_id": "admin",
            "display_name": "Administrator",
            "global_role": "admin",
            "token": ADMIN_TOKEN,
        },
        {
            "principal_id": "alice",
            "display_name": "Alice",
            "global_role": "member",
            "token": ALICE_TOKEN,
        },
        {
            "principal_id": "bob",
            "display_name": "Bob",
            "global_role": "member",
            "token": BOB_TOKEN,
        },
    ]
    secure_settings = settings.model_copy(
        update={
            "auth_enabled": True,
            "auth_credentials_json": SecretStr(json.dumps(credentials)),
        }
    )
    with TestClient(
        create_app(
            secure_settings,
            pipeline,
            ingestion,
            indexing=indexing,
            multimodal=multimodal,
        )
    ) as client:
        yield client


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_secure_mode_requires_bearer_but_keeps_health_public(
    secure_client: TestClient,
) -> None:
    health = secure_client.get("/healthz")
    missing = secure_client.get("/api/v1/collections")
    invalid = secure_client.get(
        "/api/v1/collections", headers=_auth("invalid-token-0000")
    )

    assert health.status_code == 200
    assert missing.status_code == 401
    assert missing.headers["www-authenticate"] == "Bearer"
    assert missing.json()["error"]["code"] == "authentication_required"
    assert invalid.status_code == 401
    assert "/app" not in {
        getattr(route, "path", None) for route in secure_client.app.routes
    }
    assert "HTTPBearer" in secure_client.app.openapi()["components"]["securitySchemes"]


def test_owner_viewer_editor_and_admin_boundaries_are_audited(
    secure_client: TestClient,
) -> None:
    created = secure_client.post(
        "/api/v1/collections",
        headers=_auth(ALICE_TOKEN),
        json={"name": "Alice private knowledge"},
    )
    collection_id = created.json()["id"]

    hidden = secure_client.get("/api/v1/collections", headers=_auth(BOB_TOKEN))
    denied_query = secure_client.post(
        "/api/v1/query",
        headers=_auth(BOB_TOKEN),
        json={
            "collection_id": collection_id,
            "question": "What secret is in Alice's collection?",
            "strategy": "hybrid",
        },
    )
    viewer_grant = secure_client.put(
        f"/api/v1/collections/{collection_id}/access",
        headers=_auth(ALICE_TOKEN),
        json={"principal_id": "bob", "role": "viewer"},
    )
    visible = secure_client.get("/api/v1/collections", headers=_auth(BOB_TOKEN))
    viewer_write = secure_client.post(
        f"/api/v1/collections/{collection_id}/documents",
        headers=_auth(BOB_TOKEN),
        files={"files": ("private.txt", b"Private evidence.", "text/plain")},
    )
    editor_grant = secure_client.put(
        f"/api/v1/collections/{collection_id}/access",
        headers=_auth(ALICE_TOKEN),
        json={"principal_id": "bob", "role": "editor"},
    )
    editor_write = secure_client.post(
        f"/api/v1/collections/{collection_id}/documents",
        headers=_auth(BOB_TOKEN),
        files={"files": ("private.txt", b"Private evidence.", "text/plain")},
    )
    member_evaluation = secure_client.get(
        "/api/v1/evaluations/datasets", headers=_auth(BOB_TOKEN)
    )
    admin_collections = secure_client.get(
        "/api/v1/collections", headers=_auth(ADMIN_TOKEN)
    )
    audit = secure_client.get(
        "/api/v1/audit-events", headers=_auth(ADMIN_TOKEN)
    )

    assert created.status_code == 201
    assert hidden.json() == []
    assert denied_query.status_code == 403
    assert viewer_grant.json()["role"] == "viewer"
    assert [item["id"] for item in visible.json()] == [collection_id]
    assert viewer_write.status_code == 403
    assert editor_grant.json()["role"] == "editor"
    assert editor_write.status_code == 202
    assert member_evaluation.status_code == 403
    assert [item["id"] for item in admin_collections.json()] == [collection_id]
    assert audit.status_code == 200
    assert any(
        event["action"] == "document.upload" and event["outcome"] == "denied"
        for event in audit.json()
    )
    serialized_audit = audit.text
    assert ALICE_TOKEN not in serialized_audit
    assert BOB_TOKEN not in serialized_audit
    assert "What secret" not in serialized_audit


def test_only_owner_or_admin_can_manage_access(secure_client: TestClient) -> None:
    collection_id = secure_client.post(
        "/api/v1/collections",
        headers=_auth(ALICE_TOKEN),
        json={"name": "ACL management"},
    ).json()["id"]
    secure_client.put(
        f"/api/v1/collections/{collection_id}/access",
        headers=_auth(ALICE_TOKEN),
        json={"principal_id": "bob", "role": "editor"},
    )

    editor_acl = secure_client.get(
        f"/api/v1/collections/{collection_id}/access",
        headers=_auth(BOB_TOKEN),
    )
    owner_acl = secure_client.get(
        f"/api/v1/collections/{collection_id}/access",
        headers=_auth(ALICE_TOKEN),
    )
    revoked = secure_client.delete(
        f"/api/v1/collections/{collection_id}/access/bob",
        headers=_auth(ALICE_TOKEN),
    )
    hidden_again = secure_client.get(
        "/api/v1/collections", headers=_auth(BOB_TOKEN)
    )

    assert editor_acl.status_code == 403
    assert {item["role"] for item in owner_acl.json()} == {"owner", "editor"}
    assert revoked.status_code == 204
    assert hidden_again.json() == []
