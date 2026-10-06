"""Apply trusted document-level ACL policy to existing Qdrant chunks."""

from app.ingestion.trusted_policy import (
    classification_for_document,
    trusted_chunk_classifications_for_document,
    trusted_sections_for_document,
)
from app.security.access_control import (
    Classification,
    build_document_access_metadata,
)
from app.services.retrieval import vector_search
from qdrant_client.http import models


def reindex_document_policy(document_id: str, tenant: str = "nimbuspay") -> int:
    """Update stored ACL payloads without re-embedding unchanged document text."""
    chunk_classifications = trusted_chunk_classifications_for_document(document_id)
    if trusted_sections_for_document(document_id):
        raise ValueError(
            "This document has section-level ACLs; re-run trusted batch ingestion "
            "to preserve its per-section authorization policy."
        )

    vector_search.ensure_collection_exists()
    offset = None
    points_by_classification: dict[str, list[str | int]] = {}
    stored_chunk_indexes: set[int] = set()
    all_point_ids: list[str | int] = []
    while True:
        points, next_offset = vector_search.client.scroll(
            collection_name=vector_search.COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                    models.FieldCondition(
                        key="tenant",
                        match=models.MatchValue(value=tenant),
                    ),
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=bool(chunk_classifications),
            with_vectors=False,
        )
        if not points and offset is None:
            raise LookupError(
                f"No stored chunks found for document {document_id} in tenant {tenant}."
            )

        if points:
            for point in points:
                all_point_ids.append(point.id)
                if chunk_classifications:
                    payload = point.payload or {}
                    chunk_index = payload.get("chunk_index")
                    if not isinstance(chunk_index, int):
                        raise ValueError(
                            f"Stored chunk {point.id} is missing a valid chunk_index."
                        )
                    classification = chunk_classifications.get(chunk_index)
                    if classification is None:
                        raise ValueError(
                            f"No trusted section ACL is configured for chunk "
                            f"{chunk_index} of document {document_id}."
                        )
                    stored_chunk_indexes.add(chunk_index)
                    points_by_classification.setdefault(
                        classification.value, []
                    ).append(point.id)

        if next_offset is None:
            break
        offset = next_offset

    if chunk_classifications and stored_chunk_indexes != set(chunk_classifications):
        missing_indexes = sorted(set(chunk_classifications) - stored_chunk_indexes)
        unexpected_indexes = sorted(stored_chunk_indexes - set(chunk_classifications))
        raise ValueError(
            f"Stored chunk indexes do not match trusted policy for {document_id}; "
            f"missing={missing_indexes}, unexpected={unexpected_indexes}."
        )

    if chunk_classifications:
        for classification_value, point_ids in points_by_classification.items():
            access_metadata = build_document_access_metadata(
                document_id=document_id,
                classification=Classification(classification_value),
                tenant=tenant,
            )
            vector_search.client.set_payload(
                collection_name=vector_search.COLLECTION_NAME,
                payload={
                    "classification": access_metadata.classification.value,
                    "allowed_roles": [
                        role.value for role in access_metadata.allowed_roles
                    ],
                },
                points=point_ids,
                wait=True,
            )
        return sum(len(point_ids) for point_ids in points_by_classification.values())

    classification = classification_for_document(document_id)
    access_metadata = build_document_access_metadata(
        document_id=document_id,
        classification=classification,
        tenant=tenant,
    )
    vector_search.client.set_payload(
        collection_name=vector_search.COLLECTION_NAME,
        payload={
            "classification": access_metadata.classification.value,
            "allowed_roles": [role.value for role in access_metadata.allowed_roles],
        },
        points=all_point_ids,
        wait=True,
    )

    return len(all_point_ids)
