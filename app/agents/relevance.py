import os
import math
import re


MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.01"))
RERANK_SCORE_TRACE_PRECISION = 4
MIN_RERANK_SCORE_TOLERANCE = 0.5 * 10 ** -RERANK_SCORE_TRACE_PRECISION
_QUERY_STOP_WORDS = {
    "about", "and", "are", "can", "could", "describe", "did", "do", "does",
    "for", "from", "give", "how", "in", "is", "it", "me", "of", "please",
    "should", "tell", "the", "this", "to", "what", "when", "where", "which",
    "who", "why", "with", "word", "would", "you", "your",
}


def is_relevant_rerank_score(
    score: float,
    minimum_score: float = MIN_RERANK_SCORE,
) -> bool:
    """Treat scores indistinguishable from the displayed floor as equal."""
    return score >= minimum_score - MIN_RERANK_SCORE_TOLERANCE


def has_lexical_evidence(query: str, chunks: list[dict]) -> tuple[bool, int, int]:
    """Check for meaningful exact-term overlap when a cross-encoder score is low."""
    query_terms = {
        term.casefold()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", query)
        if term.casefold() not in _QUERY_STOP_WORDS
    }
    if not query_terms:
        return False, 0, 0

    evidence_terms = {
        term.casefold()
        for chunk in chunks
        for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", chunk.get("text", ""))
    }
    overlap = len(query_terms & evidence_terms)
    minimum_overlap = 1 if len(query_terms) <= 3 else max(
        2,
        math.ceil(len(query_terms) * 0.3),
    )
    return overlap >= minimum_overlap, overlap, len(query_terms)
