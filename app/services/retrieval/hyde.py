"""
app/services/retrieval/hyde.py — HyDE (Hypothetical Document Embeddings).

Generates a hypothetical answer to the user's query, to be embedded and
searched INSTEAD OF the raw query — never shown to the user, and never
used for reranking (that still uses the original query).
"""

from app.gateway.portkey_client import client
from app.services.token_usage import record_chat_usage

_HYDE_PROMPT = """Write a short, plausible-sounding technical answer to \
the following question, as if it came from internal documentation. It's \
OK if some details are made up — this is only used to improve document \
search, not shown to any user.

Question: {query}

Hypothetical answer:"""


def generate_hypothetical_document(query: str) -> str:
    """
    Args:
        query: the user's original question

    Returns:
        A short, made-up answer text — used ONLY as input to the
        embedding/search step, never shown to the user and never used
        for reranking (which always uses the real original query).
    """
    prompt = _HYDE_PROMPT.format(query=query)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",  # small/cheap model is fine here too
    )
    record_chat_usage("HyDE", response)

    return response.choices[0].message.content
