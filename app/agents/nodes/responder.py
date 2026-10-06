"""
app/agents/nodes/responder.py — Responder Node.

The LAST node before the graph ends. Runs for BOTH paths (technical and
conversational) — the difference is just whether reranked_chunks is
populated or empty by the time this node runs.
"""

from app.agents.relevance import MIN_RERANK_SCORE
from app.gateway.portkey_client import (
    ACCESS_DENIED_MESSAGE,
    NO_EVIDENCE_MESSAGE,
    call_llm,
)


def _best_rerank_score(chunks: list[dict]) -> float | None:
    """Highest FlashRank score among the chunks, or None if none are scored."""
    scores = [
        chunk["rerank_score"]
        for chunk in chunks
        if isinstance(chunk.get("rerank_score"), (int, float))
    ]
    return max(scores) if scores else None


def responder_node(state: dict) -> dict:
    """
    Args:
        state: reads state["user_query"], state["reranked_chunks"]
               (may be missing/empty for conversational queries — see
               .get() usage below), and state["conversation_history"]

    Returns:
        {"final_answer": "..."} — the answer that gets sent back to the
        user. We do NOT update conversation_history here; that's handled
        by LangGraph's checkpointer (MemorySaver) automatically saving
        the full state after this node runs (see our Concept 4 notes on
        checkpointing).
    """
    user_query = state["user_query"]
    answer_length = state.get("answer_length")

    # .get() with a default of [] handles the conversational path safely
    # — a conversational query never went through the Retriever node,
    # so "reranked_chunks" won't exist in state at all for that path.
    context_chunks = state.get("reranked_chunks", [])

    # Relevance floor: authorized chunks exist, but if even the best one is
    # scored as essentially unrelated to the question, abstain without an LLM
    # call. Chunks without a score are not judged here (behaviour unchanged).
    best_score = _best_rerank_score(context_chunks) if context_chunks else None
    relevance_is_weak = (
        not context_chunks
        or (best_score is not None and best_score < MIN_RERANK_SCORE)
    )
    if state.get("intent") == "technical" and relevance_is_weak:
        answer = (
            ACCESS_DENIED_MESSAGE
            if state.get("restricted_evidence_found") is True
            else NO_EVIDENCE_MESSAGE
        )
        if not context_chunks:
            reason = "no authorized context reached the model"
        else:
            reason = (
                f"best rerank score {best_score:.4f} is below "
                f"relevance floor {MIN_RERANK_SCORE:.4f}"
            )
        return {
            "final_answer": answer,
            "trace": [
                *state.get("trace", []),
                f"Responder skipped: {reason}; no LLM call",
            ],
        }

    conversation_history = state.get("conversation_history", [])

    answer = call_llm(
        user_query=user_query,
        context_chunks=context_chunks,
        conversation_history=conversation_history,
        answer_length=answer_length,
    )

    # This trace communicates whether the answer was grounded in retrieved
    # documents or whether the model was simply responding to a small-talk
    # prompt without context. It is useful both for product demos and for
    # debugging the route chosen by the planner.
    source = "retrieved context" if context_chunks else "small-talk prompt (no retrieval)"
    trace_line = f"Responder: generated answer via Portkey/Groq, using {source}"

    extra_trace = []
    if best_score is not None:
        extra_trace.append(f"Best rerank score: {best_score:.4f}")

    return {
        "final_answer": answer,
        "trace": [*state.get("trace", []), *extra_trace, trace_line],
    }