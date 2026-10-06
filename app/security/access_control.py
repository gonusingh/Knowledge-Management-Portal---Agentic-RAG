"""Demo identity resolution and Qdrant document authorization policy."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from qdrant_client.http import models


class Role(StrEnum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMINISTRATOR = "administrator"


class Classification(StrEnum):
    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    CONFIDENTIAL = "CONFIDENTIAL"
    RESTRICTED = "RESTRICTED"


ROLE_CLASSIFICATIONS: dict[Role, frozenset[Classification]] = {
    Role.VIEWER: frozenset({Classification.PUBLIC}),
    Role.OPERATOR: frozenset({Classification.PUBLIC, Classification.INTERNAL}),
    Role.ADMINISTRATOR: frozenset(Classification),
}

CLASSIFICATION_ROLES: dict[Classification, frozenset[Role]] = {
    Classification.PUBLIC: frozenset(Role),
    Classification.INTERNAL: frozenset({Role.OPERATOR, Role.ADMINISTRATOR}),
    Classification.CONFIDENTIAL: frozenset({Role.ADMINISTRATOR}),
    Classification.RESTRICTED: frozenset({Role.ADMINISTRATOR}),
}


class UserContext(BaseModel):
    """Server-resolved principal used to construct retrieval ACL filters."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    roles: tuple[Role, ...]
    tenant: str


DEMO_USERS: dict[str, UserContext] = {
    "alice": UserContext(user_id="user-001", roles=(Role.VIEWER,), tenant="nimbuspay"),
    "bob": UserContext(user_id="user-002", roles=(Role.OPERATOR,), tenant="nimbuspay"),
    "carol": UserContext(
        user_id="user-003",
        roles=(Role.ADMINISTRATOR,),
        tenant="nimbuspay",
    ),
}


class DocumentAccessMetadata(BaseModel):
    """Trusted ACL metadata assigned by an ingestion/admin process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    classification: Classification
    allowed_roles: tuple[Role, ...]
    tenant: str
    service: str | None = None
    effective_date: str | None = None
    version: str | None = None


def build_document_access_metadata(
    *,
    document_id: str,
    classification: Classification,
    tenant: str = "nimbuspay",
    service: str | None = None,
    effective_date: str | None = None,
    version: str | None = None,
) -> DocumentAccessMetadata:
    """Derive the allowed roles from the trusted classification policy."""
    return DocumentAccessMetadata(
        document_id=document_id,
        classification=classification,
        allowed_roles=tuple(sorted(CLASSIFICATION_ROLES[classification], key=str)),
        tenant=tenant,
        service=service,
        effective_date=effective_date,
        version=version,
    )


def build_authorization_filter(
    user: UserContext,
    *,
    filter_type: str | None = None,
) -> models.Filter:
    """Build a fail-closed Qdrant filter from a resolved user principal."""
    allowed_classifications = {
        classification
        for role in user.roles
        for classification in ROLE_CLASSIFICATIONS[role]
    }
    must_conditions = [
        models.FieldCondition(
            key="tenant",
            match=models.MatchValue(value=user.tenant),
        ),
        models.FieldCondition(
            key="classification",
            match=models.MatchAny(
                any=sorted(classification.value for classification in allowed_classifications)
            ),
        ),
        models.FieldCondition(
            key="allowed_roles",
            match=models.MatchAny(any=[role.value for role in user.roles]),
        ),
    ]
    if filter_type is not None:
        must_conditions.append(
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value=filter_type),
            )
        )

    return models.Filter(must=must_conditions)


def scope_thread_id(thread_id: str | None, user: UserContext) -> str:
    """Bind conversation checkpoints to the current tenant/user/role set."""
    role_scope = "+".join(sorted(role.value for role in user.roles))
    prefix = f"{user.tenant}:{user.user_id}:{role_scope}:"
    if thread_id and thread_id.startswith(prefix):
        return thread_id
    return f"{prefix}{thread_id or 'new'}"