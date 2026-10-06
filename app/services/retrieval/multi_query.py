"""
app/services/retrieval/multi_query.py — Multi-Query Expansion.

Generates several rephrasings of the user's query, so retrieval can run
against each one and combine results — catching relevant documents that
a single phrasing might miss.
"""

from app.gateway.portkey_client import client
from app.services.token_usage import record_chat_usage

_EXPANSION_PROMPT = """Generate {n} different rephrasings of the \
following question. Each should ask the same thing using different \
words or phrasing. Return ONLY the rephrasings, one per line, no \
numbering, no extra text.

Original question: {query}"""


def expand_query(query: str, n: int = 3) -> list[str]:
    """
    Args:
        query: the user's original question
        n: how many alternate phrasings to generate

    Returns:
        A list of `n` rephrased query strings (does NOT include the
        original query itself — the caller combines both).
    """
    prompt = _EXPANSION_PROMPT.format(n=n, query=query)

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="openai/gpt-oss-20b",
    )
    record_chat_usage("multi-query", response)

    raw_text = response.choices[0].message.content.strip()

    # Split on newlines, and drop any empty lines that might result from
    # extra blank lines in the model's output.
    rephrasings = [line.strip() for line in raw_text.split("\n") if line.strip()]

    return rephrasings[:n]  # defensive trim, in case the model returned extra lines
