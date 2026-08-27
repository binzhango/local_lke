"""Typed Chapter 7 authentication, authorization, and audit contracts."""

from datetime import datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class GlobalRole(StrEnum):
    ADMIN = "admin"
    MEMBER = "member"


class CollectionRole(StrEnum):
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"


class Permission(StrEnum):
    READ = "read"
    WRITE = "write"
    MANAGE = "manage"


class Principal(BaseModel):
    model_config = ConfigDict(frozen=True)

    principal_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._@-]+$")
    display_name: str = Field(min_length=1, max_length=120)
    global_role: GlobalRole = GlobalRole.MEMBER


class CollectionGrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal_id: str = Field(min_length=1, max_length=120, pattern=r"^[A-Za-z0-9._@-]+$")
    role: Literal[CollectionRole.EDITOR, CollectionRole.VIEWER]


class CollectionAccessResponse(BaseModel):
    collection_id: UUID
    principal_id: str
    role: CollectionRole
    granted_by: str
    created_at: datetime


class AuditEventResponse(BaseModel):
    id: UUID
    principal_id: str
    action: str
    resource_kind: str
    resource_id: str | None
    outcome: Literal["allowed", "denied"]
    detail: dict[str, str | int | bool | None]
    created_at: datetime
