"""Local bearer authentication, collection ACLs, and metadata-only audit events."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Annotated, Any, cast
from uuid import UUID

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from local_lke.errors import AuthenticationError, AuthorizationError, NotFoundError
from local_lke.security.models import (
    AuditEventResponse,
    CollectionAccessResponse,
    CollectionRole,
    GlobalRole,
    Permission,
    Principal,
)
from local_lke.settings import Settings
from local_lke.storage.models import (
    AuditEventRecord,
    CollectionAccessRecord,
    CollectionRecord,
    DocumentVersionRecord,
    ImageAssetRecord,
    IndexingJobRecord,
    IngestionJobRecord,
    LogicalDocumentRecord,
    StructuredTableRecord,
)

_bearer = HTTPBearer(auto_error=False)
_credential_adapter = TypeAdapter(list[dict[str, Any]])
_ROLE_PERMISSIONS = {
    CollectionRole.VIEWER: {Permission.READ},
    CollectionRole.EDITOR: {Permission.READ, Permission.WRITE},
    CollectionRole.OWNER: {Permission.READ, Permission.WRITE, Permission.MANAGE},
}


class SecurityService:
    def __init__(self, sessions: sessionmaker[Session], settings: Settings) -> None:
        self.sessions = sessions
        self.enabled = settings.auth_enabled
        self._principals_by_digest = self._load_credentials(settings)
        self.local_principal = Principal(
            principal_id="local-user",
            display_name="Local user",
            global_role=GlobalRole.ADMIN,
        )

    def _load_credentials(self, settings: Settings) -> dict[bytes, Principal]:
        raw = settings.auth_credentials_json.get_secret_value()
        try:
            entries = _credential_adapter.validate_python(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError("LKE_AUTH_CREDENTIALS_JSON must be a JSON array") from exc
        principals: dict[bytes, Principal] = {}
        principal_ids: set[str] = set()
        for entry in entries:
            token = entry.get("token")
            principal_id = entry.get("principal_id")
            display_name = entry.get("display_name")
            if not isinstance(token, str) or len(token) < 16:
                raise ValueError(
                    "Every Chapter 7 bearer token must contain at least 16 characters"
                )
            if not isinstance(principal_id, str) or not isinstance(display_name, str):
                raise ValueError(
                    "Every Chapter 7 credential requires principal_id and display_name"
                )
            principal = Principal(
                principal_id=principal_id,
                display_name=display_name,
                global_role=entry.get("global_role", "member"),
            )
            digest = hashlib.sha256(token.encode()).digest()
            if digest in principals or principal.principal_id in principal_ids:
                raise ValueError("Chapter 7 principal IDs and bearer tokens must be unique")
            principals[digest] = principal
            principal_ids.add(principal.principal_id)
        if self.enabled and not principals:
            raise ValueError("Authentication is enabled but no bearer credentials are configured")
        if self.enabled and not any(
            item.global_role is GlobalRole.ADMIN for item in principals.values()
        ):
            raise ValueError("Authentication requires at least one admin principal")
        return principals

    def configured_principal_ids(self) -> set[str]:
        return {item.principal_id for item in self._principals_by_digest.values()}

    def authenticate(
        self,
        request: Request,
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    ) -> Principal:
        if not self.enabled:
            request.state.principal = self.local_principal
            return self.local_principal
        if credentials is None or credentials.scheme.casefold() != "bearer":
            raise AuthenticationError("A bearer token is required")
        candidate = hashlib.sha256(credentials.credentials.encode()).digest()
        principal = next(
            (
                configured
                for digest, configured in self._principals_by_digest.items()
                if hmac.compare_digest(candidate, digest)
            ),
            None,
        )
        if principal is None:
            raise AuthenticationError("The bearer token is invalid")
        request.state.principal = principal
        return principal

    def principal(self, request: Request) -> Principal:
        return cast(Principal, request.state.principal)

    def authorize_collection(
        self,
        principal: Principal,
        collection_id: UUID,
        permission: Permission,
        *,
        action: str,
    ) -> None:
        if not self.enabled:
            self.audit(principal, action, "collection", str(collection_id), "allowed")
            return
        with self.sessions() as session:
            exists = session.get(CollectionRecord, str(collection_id)) is not None
            role_value = session.scalar(
                select(CollectionAccessRecord.role).where(
                    CollectionAccessRecord.collection_id == str(collection_id),
                    CollectionAccessRecord.principal_id == principal.principal_id,
                )
            )
        if not exists:
            raise NotFoundError("Collection not found", component="collection")
        if principal.global_role is GlobalRole.ADMIN:
            self.audit(principal, action, "collection", str(collection_id), "allowed")
            return
        role = CollectionRole(role_value) if role_value is not None else None
        allowed = role is not None and permission in _ROLE_PERMISSIONS[role]
        self.audit(
            principal,
            action,
            "collection",
            str(collection_id),
            "allowed" if allowed else "denied",
        )
        if not allowed:
            raise AuthorizationError("The principal does not have access to this collection")

    def require_admin(self, principal: Principal, *, action: str) -> None:
        allowed = not self.enabled or principal.global_role is GlobalRole.ADMIN
        self.audit(principal, action, "system", None, "allowed" if allowed else "denied")
        if not allowed:
            raise AuthorizationError("Administrator access is required")

    def accessible_collection_ids(self, principal: Principal) -> set[UUID] | None:
        if not self.enabled or principal.global_role is GlobalRole.ADMIN:
            return None
        with self.sessions() as session:
            values = session.scalars(
                select(CollectionAccessRecord.collection_id).where(
                    CollectionAccessRecord.principal_id == principal.principal_id
                )
            )
            return {UUID(item) for item in values}

    def grant(
        self,
        actor: Principal,
        collection_id: UUID,
        principal_id: str,
        role: CollectionRole,
    ) -> CollectionAccessResponse:
        if principal_id not in self.configured_principal_ids():
            raise AuthorizationError("The target principal is not configured")
        if role is CollectionRole.OWNER:
            raise AuthorizationError("Ownership transfer is not supported")
        with self.sessions.begin() as session:
            record = session.scalar(
                select(CollectionAccessRecord).where(
                    CollectionAccessRecord.collection_id == str(collection_id),
                    CollectionAccessRecord.principal_id == principal_id,
                )
            )
            if record is None:
                record = CollectionAccessRecord(
                    collection_id=str(collection_id),
                    principal_id=principal_id,
                    role=role.value,
                    granted_by=actor.principal_id,
                )
                session.add(record)
            else:
                if record.role == CollectionRole.OWNER.value:
                    raise AuthorizationError("The collection owner cannot be replaced")
                record.role = role.value
                record.granted_by = actor.principal_id
            session.flush()
            response = _access_response(record)
        self.audit(actor, "collection.access.grant", "collection", str(collection_id), "allowed")
        return response

    def revoke(self, actor: Principal, collection_id: UUID, principal_id: str) -> None:
        with self.sessions.begin() as session:
            record = session.scalar(
                select(CollectionAccessRecord).where(
                    CollectionAccessRecord.collection_id == str(collection_id),
                    CollectionAccessRecord.principal_id == principal_id,
                )
            )
            if record is not None and record.role == CollectionRole.OWNER.value:
                raise AuthorizationError("The collection owner cannot be revoked")
            session.execute(
                delete(CollectionAccessRecord).where(
                    CollectionAccessRecord.collection_id == str(collection_id),
                    CollectionAccessRecord.principal_id == principal_id,
                )
            )
        self.audit(actor, "collection.access.revoke", "collection", str(collection_id), "allowed")

    def list_access(self, collection_id: UUID) -> list[CollectionAccessResponse]:
        with self.sessions() as session:
            records = session.scalars(
                select(CollectionAccessRecord)
                .where(CollectionAccessRecord.collection_id == str(collection_id))
                .order_by(CollectionAccessRecord.role, CollectionAccessRecord.principal_id)
            )
            return [_access_response(item) for item in records]

    def list_audit_events(self, limit: int = 100) -> list[AuditEventResponse]:
        with self.sessions() as session:
            records = session.scalars(
                select(AuditEventRecord)
                .order_by(AuditEventRecord.created_at.desc())
                .limit(limit)
            )
            return [_audit_response(item) for item in records]

    def audit(
        self,
        principal: Principal,
        action: str,
        resource_kind: str,
        resource_id: str | None,
        outcome: str,
        detail: dict[str, str | int | bool | None] | None = None,
    ) -> None:
        if not self.enabled:
            return
        with self.sessions.begin() as session:
            session.add(
                AuditEventRecord(
                    principal_id=principal.principal_id,
                    action=action,
                    resource_kind=resource_kind,
                    resource_id=resource_id,
                    outcome=outcome,
                    detail=detail or {},
                )
            )

    def collection_for_job(self, job_id: UUID, *, indexing: bool = False) -> UUID:
        model = IndexingJobRecord if indexing else IngestionJobRecord
        return self._direct_collection(model, job_id, "Job")

    def collection_for_image(self, image_id: UUID) -> UUID:
        return self._direct_collection(ImageAssetRecord, image_id, "Image")

    def collection_for_table(self, table_id: UUID) -> UUID:
        return self._direct_collection(StructuredTableRecord, table_id, "Structured table")

    def collection_for_document(self, document_id: UUID) -> UUID:
        return self._direct_collection(LogicalDocumentRecord, document_id, "Document")

    def collection_for_version(self, version_id: UUID) -> UUID:
        with self.sessions() as session:
            value = session.scalar(
                select(LogicalDocumentRecord.collection_id)
                .join(
                    DocumentVersionRecord,
                    DocumentVersionRecord.document_id == LogicalDocumentRecord.id,
                )
                .where(DocumentVersionRecord.id == str(version_id))
            )
        if value is None:
            raise NotFoundError("Document version not found", component="security")
        return UUID(value)

    def _direct_collection(self, model: Any, resource_id: UUID, label: str) -> UUID:
        with self.sessions() as session:
            value = session.scalar(
                select(model.collection_id).where(model.id == str(resource_id))
            )
        if value is None:
            raise NotFoundError(f"{label} not found", component="security")
        return UUID(value)


def _access_response(record: CollectionAccessRecord) -> CollectionAccessResponse:
    return CollectionAccessResponse(
        collection_id=UUID(record.collection_id),
        principal_id=record.principal_id,
        role=CollectionRole(record.role),
        granted_by=record.granted_by,
        created_at=record.created_at,
    )


def _audit_response(record: AuditEventRecord) -> AuditEventResponse:
    return AuditEventResponse(
        id=UUID(record.id),
        principal_id=record.principal_id,
        action=record.action,
        resource_kind=record.resource_kind,
        resource_id=record.resource_id,
        outcome=cast(Any, record.outcome),
        detail=record.detail,
        created_at=record.created_at,
    )
