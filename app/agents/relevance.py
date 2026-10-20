import os


MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "0.01"))
RERANK_SCORE_TRACE_PRECISION = 4
MIN_RERANK_SCORE_TOLERANCE = 0.5 * 10 ** -RERANK_SCORE_TRACE_PRECISION


def is_relevant_rerank_score(
    score: float,
    minimum_score: float = MIN_RERANK_SCORE,
) -> bool:
    """Treat scores indistinguishable from the displayed floor as equal."""
    return score >= minimum_score - MIN_RERANK_SCORE_TOLERANCE
