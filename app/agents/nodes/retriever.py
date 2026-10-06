"""
app/agents/nodes/retriever.py — Retriever Node.

Only runs when the Planner classified the query as "technical". Combines
THREE search angles into one candidate pool before reranking:
  1. HyDE — search using an embedded hypothetical answer
  2. Multi-Query — search using several rephrasings of the question
  3. The original query itself — always included too, as a baseline

This is a deliberate cost/latency vs. recall tradeoff: more LLM calls and
more Qdrant searches per request, in exchange for a better chance of
finding the right chunks even when phrasing doesn't closely match the
documents.
"""

from app.services.retrieval.vector_search import search
from app.services.retrieval.reranker import rerank
from app.services.retrieval.hyde import generate_hypothetical_document
from app.services.retrieval.multi_query import expand_query
from app.agents.relevance import MIN_RERANK_SCORE
from app.security.access_control import (
    Classification,
    ROLE_CLASSIFICATIONS,
    Role,
    UserContext,
)
import re


_VALID_ROLES = {role.value for role in Role}


def _deduplicate_candidates(candidate_lists: list[list[dict]]) -> list[dict]:
    """
    Merges several lists of Qdrant search results into one list with no
    duplicate chunks. The same chunk can appear in more than one search
    (e.g. found by both the HyDE search AND a multi-query rephrasing) —
    when that happens, we keep whichever instance had the higher score.
    """
    best_by_chunk_id: dict[str, dict] = {}

    for candidates in candidate_lists:
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            existing = best_by_chunk_id.get(chunk_id)
            if existing is None or candidate["score"] > existing["score"]:
                best_by_chunk_id[chunk_id] = candidate

    return list(best_by_chunk_id.values())


def _is_authorized_for_user(candidate: dict, user: UserContext) -> bool | None:
    """Return whether trusted ACL metadata grants access; None means malformed."""
    classification_value = candidate.get("classification")
    if not isinstance(classification_value, str):
        return None
    try:
        classification = Classification(classification_value)
    except ValueError:
        return None
    allowed_roles = candidate.get("allowed_roles")
    if (
        not isinstance(allowed_roles, list)
        or not allowed_roles
        or any(
            not isinstance(role, str) or role not in _VALID_ROLES
            for role in allowed_roles
        )
    ):
        return None

    role_allows_classification = any(
        classification in ROLE_CLASSIFICATIONS[role]
        for role in user.roles
    )
    metadata_allows_role = any(
        role.value in allowed_roles
        for role in user.roles
    )
    return role_allows_classification and metadata_allows_role


def _has_relevant_restricted_evidence(
    *,
    user_query: str,
    search_queries: list[str],
    user: UserContext,
    active_document_id: str,
) -> bool:
    """Check restricted matches locally without returning their text to graph state."""
    candidate_lists = [
        search(
            query=query,
            user=user,
            top_k=20,
            document_id=active_document_id,
            apply_role_filter=False,
        )
        for query in search_queries
    ]
    restricted_candidates = [
        candidate
        for candidate in _deduplicate_candidates(candidate_lists)
        if _is_authorized_for_user(candidate, user) is False
    ]
    if not restricted_candidates:
        return False

    ranked_restricted_candidates = rerank(
        original_query=user_query,
        candidates=restricted_candidates,
        active_document_id=active_document_id,
        top_n=8,
    )
    return any(
        isinstance(candidate.get("rerank_score"), (int, float))
        and candidate["rerank_score"] >= MIN_RERANK_SCORE
        for candidate in ranked_restricted_candidates
    )


def retriever_node(state: dict) -> dict:
    """
    Args:
        state: reads state["user_query"]

    Returns:
        {"reranked_chunks": [...]} — the final, precise top-N chunks
        the Responder node will see.
    """
    user_query = state["user_query"]
    user = UserContext.model_validate(state["user_context"])
    active_document_id = state.get("active_document_id")
    # Trace has no append reducer so it resets at the start of each turn;
    # carry forward this turn's planner entry before adding retrieval steps.
    trace = list(state.get("trace", []))

    if not isinstance(active_document_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", active_document_id
    ):
        return {
            "reranked_chunks": [],
            "restricted_evidence_found": False,
            "trace": [
                *trace,
                "Retriever abstained: missing or invalid active document ID",
            ],
        }

    # The allowed ACL set is enforced at query time. If a user asks a question
    # that exists only in a restricted section, the retrieval layer should still
    # return zero authorized results rather than a noisy "closest public chunk".
    all_candidate_lists = []
    search_queries = [user_query]

    # --- Angle 1: the original query, always included as a baseline ---
    original_results = search(
        query=user_query,
        user=user,
        top_k=20,
        document_id=active_document_id,
    )
    all_candidate_lists.append(original_results)
    trace.append(
        f"Qdrant authorization filter applied (tenant={user.tenant}, "
        f"roles={','.join(role.value for role in user.roles)}); "
        f"results scoped to active document {active_document_id}"
    )
    trace.append(f"Authorized vector results (original query): {len(original_results)}")

    # --- Angle 2: HyDE — search using a hypothetical answer instead of
    #     the short raw question ---
    hypothetical_doc = generate_hypothetical_document(user_query)
    search_queries.append(hypothetical_doc)
    hyde_results = search(
        query=hypothetical_doc,
        user=user,
        top_k=20,
        document_id=active_document_id,
    )
    all_candidate_lists.append(hyde_results)
    trace.append(f"HyDE authorized results: {len(hyde_results)}")

    # --- Angle 3: Multi-Query — search using several rephrasings ---
    rephrasings = expand_query(user_query, n=2)
    search_queries.extend(rephrasings)
    trace.append(f"Multi-Query: generated {len(rephrasings)} rephrasings — {rephrasings}")
    for rephrasing in rephrasings:
        rephrase_results = search(
            query=rephrasing,
            user=user,
            top_k=20,
            document_id=active_document_id,
        )
        all_candidate_lists.append(rephrase_results)
        trace.append(f"  -> authorized results for rephrasing: {len(rephrase_results)}")

    # Merge all search results into one deduplicated pool.
    combined_candidates = _deduplicate_candidates(all_candidate_lists)
    trace.append(f"Merged + deduplicated: {len(combined_candidates)} unique candidates")

    reranked_chunks = []
    if not combined_candidates:
        trace.append(
            "Retriever abstained: no authorized chunks found in the active document"
        )
    else:
        # Reranking ALWAYS uses the original user_query — never the
        # hypothetical HyDE text or any rephrased version.
        reranked_chunks = rerank(
            original_query=user_query,
            candidates=combined_candidates,
            active_document_id=active_document_id,
            top_n=8,
        )
        trace.append(
            f"FlashRank reranked against original query — kept top "
            f"{len(reranked_chunks)} of {len(combined_candidates)} candidates"
        )

    for chunk in reranked_chunks:
        trace.append(
            "Selected chunk: "
            f"source={chunk['source_file']}, "
            f"document_id={chunk.get('document_id', 'unknown')}, "
            f"chunk_index={chunk.get('chunk_index', 'unknown')}, "
            f"rerank_score={chunk['rerank_score']:.4f}"
        )

    scored_chunks = [
        chunk["rerank_score"]
        for chunk in reranked_chunks
        if isinstance(chunk.get("rerank_score"), (int, float))
    ]
    authorized_evidence_is_weak = not reranked_chunks or (
        bool(scored_chunks) and max(scored_chunks) < MIN_RERANK_SCORE
    )
    restricted_evidence_found = False
    if authorized_evidence_is_weak:
        restricted_evidence_found = _has_relevant_restricted_evidence(
            user_query=user_query,
            search_queries=search_queries,
            user=user,
            active_document_id=active_document_id,
        )

    return {
        "reranked_chunks": reranked_chunks,
        "restricted_evidence_found": restricted_evidence_found,
        "trace": trace,
    }
