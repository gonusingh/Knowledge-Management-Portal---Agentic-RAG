"""
app/services/retrieval/vector_search.py — Qdrant search.

Two responsibilities:
1. upsert_chunks()  — embed chunks and store them in Qdrant (ingestion time)
2. search()         — embed a query and find the closest chunks (query time)

Both share the same embedding function, since the query and the documents
MUST be embedded with the exact same model to be comparable at all —
mixing embedding models would produce vectors in two different, unrelated
"meaning spaces" that cosine similarity can't meaningfully compare.
"""

import uuid
import re

import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.security.access_control import UserContext, build_authorization_filter

# --- One-time setup ---------------------------------------------------

genai.configure(api_key=settings.gemini_api_key)

client = QdrantClient(
    url=settings.qdrant_cluster_endpoint,
    api_key=settings.qdrant_api_key,
)

COLLECTION_NAME = "enterprise_rag_docs"
EMBEDDING_DIMENSIONS = 3072  # gemini-embedding-2-preview output size
_payload_indexes_ready_for: tuple[int, str] | None = None


def _get_embedding(text: str) -> list[float]:
    """
    Turns a piece of text into a 3072-number vector using Gemini.
    Used for BOTH document chunks (at ingestion) and user queries
    (at search time) — same function, same model, so the resulting
    vectors live in the same comparable "meaning space".
    """
    result = genai.embed_content(
        model="models/gemini-embedding-2-preview",
        content=text,
    )
    return result["embedding"]


def ensure_collection_exists() -> None:
    """
    Creates the Qdrant collection if it doesn't already exist.
    distance=models.Distance.COSINE is the literal config line where the
    cosine similarity concept becomes real: this tells Qdrant "when
    searching, rank results by the angle between vectors" (Concept 6),
    not by raw Euclidean distance.
    """
    global _payload_indexes_ready_for
    client_collection = (id(client), COLLECTION_NAME)
    if _payload_indexes_ready_for == client_collection:
        return

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=EMBEDDING_DIMENSIONS,
                distance=models.Distance.COSINE,
            ),
        )

    collection_info = client.get_collection(collection_name=COLLECTION_NAME)
    existing_payload_schema = collection_info.payload_schema or {}
    for field_name in ("tenant", "classification", "allowed_roles", "type", "document_id"):
        if field_name not in existing_payload_schema:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )

    _payload_indexes_ready_for = client_collection


def upsert_chunks(chunks: list[dict]) -> None:
    """
    Embeds and uploads a list of chunk dicts (from chunk_document()) into
    Qdrant. "Upsert" = insert if new, update if the same chunk_id already
    exists — safe to re-run without creating duplicates.

    Args:
        chunks: list of {"chunk_id", "text", "source_file", "type", "chunk_index"}
    """
    if not chunks:
        return

    ensure_collection_exists()

    point_ids = {
        chunk["chunk_id"]: str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))
        for chunk in chunks
    }
    existing_points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=list(point_ids.values()),
        with_payload=True,
        with_vectors=True,
    )
    existing_by_id = {str(point.id): point for point in existing_points}

    points = []
    for chunk in chunks:
        point_id = point_ids[chunk["chunk_id"]]
        existing_point = existing_by_id.get(point_id)
        if (
            existing_point is not None
            and existing_point.payload is not None
            and existing_point.payload.get("text") == chunk["text"]
            and existing_point.vector is not None
        ):
            vector = existing_point.vector
        else:
            vector = _get_embedding(chunk["text"])

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                # The "payload" is Qdrant's term for metadata stored
                # alongside the vector. We store the chunk's own text
                # here too, so search results come back with the actual
                # content, not just an ID we'd have to look up separately.
                # chunk_id (the original human-readable string) is kept
                # here too, for tracing/citations, even though it's not
                # the actual Qdrant point ID anymore.
                payload={
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source_file": chunk["source_file"],
                    "type": chunk["type"],
                    "chunk_index": chunk["chunk_index"],
                    "document_id": chunk["document_id"],
                    "classification": chunk["classification"],
                    "allowed_roles": chunk["allowed_roles"],
                    "tenant": chunk["tenant"],
                    "service": chunk["service"],
                    "effective_date": chunk["effective_date"],
                    "version": chunk["version"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)


def document_exists(document_id: str, tenant: str) -> bool:
    """Check whether this content-addressed document already has stored chunks."""
    return get_document_metadata(document_id, tenant) is not None


def list_documents(tenant: str) -> list[dict]:
    """Return the unique indexed documents for a tenant with their stored ACL metadata."""
    ensure_collection_exists()
    unique_documents: dict[str, dict] = {}
    offset = None

    while True:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant",
                        match=models.MatchValue(value=tenant),
                    )
                ]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = response

        for point in points:
            payload = point.payload or {}
            document_id = payload.get("document_id")
            if not document_id or document_id in unique_documents:
                continue

            unique_documents[document_id] = {
                "document_id": document_id,
                "source_file": payload.get("source_file"),
                "classification": payload.get("classification"),
                "allowed_roles": payload.get("allowed_roles") or [],
                "tenant": payload.get("tenant", tenant),
                "version": payload.get("version"),
            }

        if next_offset is None:
            break
        offset = next_offset

    return sorted(
        unique_documents.values(),
        key=lambda item: (item.get("source_file") or "", item["document_id"]),
    )


def inspect_document_chunks(document_id: str, tenant: str) -> list[dict]:
    """Return all stored chunks for a document, including the ACL metadata needed for debugging."""
    ensure_collection_exists()
    offset = None
    results: list[dict] = []

    while True:
        response = client.scroll(
            collection_name=COLLECTION_NAME,
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
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = response

        for point in points:
            payload = point.payload or {}
            results.append(
                {
                    "document_id": payload.get("document_id"),
                    "source_file": payload.get("source_file"),
                    "chunk_index": payload.get("chunk_index"),
                    "classification": payload.get("classification"),
                    "allowed_roles": payload.get("allowed_roles") or [],
                    "tenant": payload.get("tenant"),
                }
            )

        if next_offset is None:
            break
        offset = next_offset

    return results


def get_document_metadata(document_id: str, tenant: str) -> dict | None:
    """Return stored metadata for an existing document without loading vectors."""
    ensure_collection_exists()
    response = client.scroll(
        collection_name=COLLECTION_NAME,
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
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    points, _ = response
    return points[0].payload if points else None


def delete_document(document_id: str, tenant: str) -> None:
    """Delete every stored chunk for one document in the specified tenant."""
    ensure_collection_exists()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
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
            )
        ),
        wait=True,
    )


def search(
    query: str,
    user: UserContext,
    *,
    document_id: str,
    top_k: int = 50,
    filter_type: str | None = None,
    apply_role_filter: bool = True,
) -> list[dict]:
    """
    Embeds the user's query and searches Qdrant for the top_k closest
    chunks by cosine similarity (HNSW approximate search under the hood).

    Args:
        query: the raw user question
        top_k: how many candidates to return (this project casts a wide
               net here — e.g. 50 — then narrows down with FlashRank
               reranking afterward; see our two-stage retrieval notes)
          user: server-resolved principal; required so every vector search
              applies tenant, classification, and role ACL filters in Qdrant.
          filter_type: optional — additionally restrict by the ingestion type tag.
          document_id: required active document scope. It is combined with,
              never substituted for, the tenant/role authorization filter.

    Returns:
        A list of dicts, each with the chunk's text, metadata, and its
        cosine similarity score (0 to 1 in practice for normalized
        embeddings — see our cosine similarity notes on the score range).
    """
    if not isinstance(document_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", document_id
    ):
        raise ValueError("A valid active document ID is required for retrieval.")

    ensure_collection_exists()
    query_vector = _get_embedding(query)

    # ACL is applied inside Qdrant so unauthorized payload text never leaves
    # the vector store for reranking or context assembly. Missing ACL fields
    # do not match and therefore remain inaccessible until re-ingested.
    qdrant_filter = models.Filter(must=[
        models.FieldCondition(
            key="tenant",
            match=models.MatchValue(value=user.tenant),
        ),
        models.FieldCondition(
            key="document_id",
            match=models.MatchValue(value=document_id),
        ),
    ])
    if apply_role_filter:
        role_filter = build_authorization_filter(user, filter_type=filter_type)
        qdrant_filter = models.Filter(
            must=[
                *(role_filter.must or []),
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                ),
            ]
        )

    # NOTE: client.search() was removed in newer qdrant-client versions,
    # replaced by client.query_points() — same underlying search, just a
    # renamed method with a slightly different response shape (results
    # come back as response.points instead of directly as a list).
    # Query-time retrieval works in the same embedding space as ingestion,
    # so the system compares meaning rather than raw words. Qdrant returns
    # the most similar vectors first, and the reranker then narrows those
    # candidates to the handful that are truly relevant to the question.
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=qdrant_filter,
    )
    results = response.points

    return [
        {
            "chunk_id": hit.payload["chunk_id"],
            "text": hit.payload["text"],
            "source_file": hit.payload["source_file"],
            "type": hit.payload["type"],
            "chunk_index": hit.payload["chunk_index"],
            "score": float(hit.score),  # the cosine similarity score itself
            "document_id": hit.payload["document_id"],
            "classification": hit.payload["classification"],
            "allowed_roles": hit.payload["allowed_roles"],
            "tenant": hit.payload["tenant"],
            "service": hit.payload.get("service"),
            "effective_date": hit.payload.get("effective_date"),
            "version": hit.payload.get("version"),
        }
        for hit in results
    ]