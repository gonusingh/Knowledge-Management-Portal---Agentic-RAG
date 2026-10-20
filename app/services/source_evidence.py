"""Parse authorized retrieval-source metadata from the public query trace."""

import re


_SELECTED_SOURCE_PATTERN = re.compile(
    r"^Selected chunk: source=(?P<source>.*?), "
    r"document_id=(?P<document_id>[^,]+), "
    r"chunk_index=(?P<chunk_index>[^,]+), "
    r"rerank_score=(?P<score>-?\d+(?:\.\d+)?)$"
)


def extract_selected_sources(trace: list[str]) -> list[dict[str, str]]:
    """Return deduplicated chunk references emitted after authorized retrieval."""
    sources: list[dict[str, str]] = []
    seen_chunks: set[tuple[str, str]] = set()
    for item in trace:
        match = _SELECTED_SOURCE_PATTERN.match(item)
        if not match:
            continue
        document_id = match.group("document_id")
        chunk_index = match.group("chunk_index")
        key = (document_id, chunk_index)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        sources.append(
            {
                "source": match.group("source"),
                "document_id": document_id,
                "chunk_index": chunk_index,
                "score": match.group("score"),
            }
        )
    return sources


def distinct_source_count(sources: list[dict[str, str]]) -> int:
    """Count distinct source documents, preferring stable document IDs."""
    identities = set()
    for source in sources:
        document_id = source.get("document_id", "").strip()
        if document_id and document_id.casefold() != "unknown":
            identities.add(("document_id", document_id))
            continue
        filename = " ".join(source.get("source", "").casefold().split())
        if filename:
            identities.add(("source_file", filename))
    return len(identities)
