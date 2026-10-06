"""
app/services/retrieval/reranker.py — FlashRank cross-encoder reranking.

Stage 2 of the two-stage retrieval pipeline (see our FlashRank notes):
Qdrant already narrowed millions of chunks down to ~50 rough candidates
(bi-encoder, fast). This file re-scores just those 50 by reading the
query and EACH candidate TOGETHER (cross-encoder, slow but precise),
and keeps only the real top few.
"""

import re

from flashrank import Ranker, RerankRequest

# Loading the model happens ONCE, when this module is first imported —
# not on every single request. Re-loading a model from disk on every
# query would add unnecessary latency; this way it stays in memory,
# ready to score requests as they come in.
#
# model_name picks the MiniLM variant from our notes (more accurate than
# TinyBERT, still far faster than a full-size cross-encoder).
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")


def rerank(
    original_query: str,
    candidates: list[dict],
    *,
    active_document_id: str,
    top_n: int = 5,
) -> list[dict]:
    """
    Args:
        original_query: the user's ACTUAL question — never a rewritten/
                         expanded version (see our "always rerank against
                         the original query" gotcha from the FlashRank notes)
        candidates: candidate chunks; only chunks whose document_id matches
                    active_document_id are eligible for reranking.
        active_document_id: required document scope shared with vector retrieval.
        top_n: how many chunks to keep after reranking (these are what
               actually get sent to the LLM — keep this small, e.g. 5)

    Returns:
        The same candidate dicts, but re-ordered by FlashRank's more
        precise relevance score, trimmed down to just top_n, with a
        new "rerank_score" field added to each one.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", active_document_id):
        return []

    candidates = [
        candidate for candidate in candidates
        if candidate.get("document_id") == active_document_id
    ]
    if not candidates:
        return []

    # FlashRank's RerankRequest wants a simple list of passages, each
    # needing at least a "text" field (an "id" is optional but useful
    # so we can match results back to our original candidate dicts).
    passages = [
        {"id": candidate["chunk_id"], "text": candidate["text"]}
        for candidate in candidates
    ]

    rerank_request = RerankRequest(query=original_query, passages=passages)

    # This is the actual cross-encoder pass: for EACH passage, FlashRank
    # runs the query and that passage's text through the model TOGETHER,
    # producing one precise relevance score per passage (unlike Qdrant's
    # pre-computed vector comparison).
    reranked_results = _ranker.rerank(rerank_request)

    # reranked_results comes back already sorted, highest relevance first.
    # We now map each result back to its original full candidate dict
    # (which has source_file, type, etc — not just text), so we don't
    # lose that metadata, and attach the new score.
    candidates_by_id = {c["chunk_id"]: c for c in candidates}

    final_results = []
    for result in reranked_results[:top_n]:
        original_candidate = candidates_by_id[result["id"]]
        final_results.append({
            **original_candidate,
            "rerank_score": float(result["score"]),
        })

    return final_results