"""
app/gateway/portkey_client.py — Portkey/direct Groq LLM integration.

The Responder node calls call_llm() from this file instead of calling
Groq directly when no saved Portkey config is configured. This temporary
local fallback keeps development working when Portkey blocks inline configs.
"""

import re

from openai import OpenAI
from portkey_ai import Portkey

from app.config import settings
from app.services.token_usage import record_chat_usage


MAX_CONTEXT_CHUNKS = 8
MAX_CONTEXT_CHARS = 12000
MAX_FOCUSED_PASSAGES = 3
MAX_FOCUSED_PASSAGE_CHARS = 3000

# Single canonical message used whenever the available document evidence
# does not support an answer.
NO_EVIDENCE_MESSAGE = (
    "I couldn't find enough information in the active document to answer that question."
)
ACCESS_DENIED_MESSAGE = (
    "This information exists in the active document but isn't available to your role."
)

_WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

_QUERY_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def _focused_passages(
    context_chunks: list[dict],
    user_query: str,
) -> list[dict]:
    """Select verbatim document passages that overlap most with the question."""
    query_terms = {
        term.casefold()
        for term in _WORD_PATTERN.findall(user_query)
        if term.casefold() not in _QUERY_STOP_WORDS
    }

    if not query_terms:
        return []

    ranked_passages = []

    for chunk_index, chunk in enumerate(context_chunks):
        for passage_index, passage in enumerate(
            re.split(r"\n\s*\n", chunk["text"].strip())
        ):
            passage = passage.strip()

            if not passage:
                continue

            passage_terms = {
                term.casefold()
                for term in _WORD_PATTERN.findall(passage)
            }

            overlap = len(query_terms & passage_terms)

            if overlap:
                ranked_passages.append(
                    (
                        overlap,
                        chunk_index,
                        passage_index,
                        chunk,
                        passage,
                    )
                )

    ranked_passages.sort(
        key=lambda item: (-item[0], item[1], item[2])
    )

    selected = []
    selected_chars = 0

    for _, _, _, chunk, passage in ranked_passages:
        if len(selected) >= MAX_FOCUSED_PASSAGES:
            break

        remaining_chars = (
            MAX_FOCUSED_PASSAGE_CHARS - selected_chars
        )

        if remaining_chars <= 0:
            break

        if len(passage) > remaining_chars:
            passage = passage[:remaining_chars].rsplit(" ", 1)[0]

        if passage:
            selected.append(
                {
                    "source_file": chunk["source_file"],
                    "text": passage,
                }
            )
            selected_chars += len(passage)

    return selected


def _assemble_context(
    context_chunks: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Build bounded responder context, skipping repeated uploaded content."""
    context_blocks = []
    seen_texts = set()
    used_chars = 0

    for chunk in context_chunks:
        text = chunk["text"].strip()
        normalized_text = " ".join(text.casefold().split())

        if not text or normalized_text in seen_texts:
            continue

        source_header = f"[Source: {chunk['source_file']}]\n"
        separator = "\n\n" if context_blocks else ""

        if len(context_blocks) >= MAX_CONTEXT_CHUNKS:
            break

        available_chars = (
            max_chars
            - used_chars
            - len(separator)
            - len(source_header)
        )

        if available_chars <= 0:
            break

        if len(text) > available_chars:
            truncation_marker = "... [truncated]"

            if available_chars > len(truncation_marker):
                text = (
                    text[
                        : available_chars - len(truncation_marker)
                    ].rstrip()
                    + truncation_marker
                )
            else:
                text = text[:available_chars]

        context_blocks.append(
            f"{source_header}{text}"
        )

        seen_texts.add(normalized_text)
        used_chars += (
            len(separator)
            + len(source_header)
            + len(text)
        )

    return "\n\n".join(context_blocks)


# This config block is where the fallback strategy actually gets wired
# up (see our "Piece 2: how the fallback mechanism works" notes) —
# it's the literal, real version of the example config we discussed:
# try the primary model first, automatically try the fallback if it fails.
_portkey_config = {
    "strategy": {"mode": "fallback"},
    "targets": [
        {
            "provider": "groq",
            "api_key": settings.groq_api_key,
            "override_params": {
                "model": "openai/gpt-oss-120b"
            },
        },
        {
            "provider": "groq",
            "api_key": settings.groq_fallback_api_key,
            "override_params": {
                "model": "openai/gpt-oss-20b"
            },
        },
    ],
}

# Portkey accounts can block inline configs. Use a saved Portkey config when
# available; otherwise use Groq directly for local development.
if settings.portkey_config:
    client = Portkey(
        api_key=settings.portkey_api_key,
        config=settings.portkey_config,
    )
else:
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )


def call_llm(
    user_query: str,
    context_chunks: list[dict],
    conversation_history: list[dict],
    answer_length: str = "long",
) -> str:
    """
    Args:
        user_query: the user's latest question
        context_chunks: the reranked chunks from reranker.rerank() —
                         these get inserted into the system prompt so the
                         LLM has the actual source material to answer from.
                         Empty on the conversational path (the Planner
                         skipped retrieval), which changes the system prompt
                         below.
        conversation_history: prior turns in this conversation, in
                              [{"role": "user"/"assistant", "content": "..."}]
                              format — this is what gives the agent memory
                              across turns (ties into LangGraph's
                              MemorySaver/checkpointing from our notes)
        answer_length: "short" for a concise response or "long" for a
                       detailed response. None leaves the answer length
                       to the model; direct callers default to "long".

    Returns:
        The LLM's answer as a plain string.
    """

    # Two different system prompts, depending on whether retrieval ran.
    #
    # WITH chunks (technical path): the model must answer only from the
    # retrieved context. We include source_file per chunk so the model can
    # cite where each fact came from, and so we could later parse
    # citations out of the answer to show sources in the UI.
    #
    # WITHOUT chunks (conversational path, e.g. "hi", "thanks"): there is
    # no context to ground on, so telling the model "answer ONLY from the
    # context" makes it apologise ("I don't have any context..."). Instead
    # we give it a friendly small-talk prompt, plus a guard so it doesn't
    # start stating technical facts that nothing in the system supports.
    if context_chunks:
        focused_passages = _focused_passages(
            context_chunks,
            user_query,
        )

        focused_text = _assemble_context(
            focused_passages,
            max_chars=MAX_FOCUSED_PASSAGE_CHARS,
        )

        context_text = _assemble_context(
            context_chunks,
            max_chars=MAX_CONTEXT_CHARS - len(focused_text),
        )

        system_content = (
            "Answer using ONLY explicit evidence from the user's active "
            "document. The section labeled QUESTION-RELEVANT EXCERPTS contains "
            "verbatim passages selected from that document and should be "
            "checked first. Treat conversation history only as context for "
            "resolving references; it is not evidence and must never override "
            "the document. For every requested fact, reproduce stated values "
            "and units exactly. Never substitute a typical or remembered "
            "default. If the document does not explicitly support an answer, "
            f"reply with exactly this sentence and nothing else: "
            f"{NO_EVIDENCE_MESSAGE} "
            "Never infer permissions, rules or capabilities the text does not state.\n\n"
            "QUESTION-RELEVANT EXCERPTS (verbatim):\n"
            f"{focused_text or '[No matching passage found]'}"
            "\n\nFULL RETRIEVED DOCUMENT CONTEXT:\n"
            f"CONTEXT:\n{context_text}"
        )

    else:
        system_content = (
            "You are a friendly assistant for a technical documentation "
            "Q&A system. The user's message is small talk or a greeting "
            "that needs no document lookup. Reply briefly and naturally, "
            "and mention that you can answer questions about the uploaded "
            "documents. Do not state technical facts you cannot support."
        )

    # Add length guidance to the system message so the setting changes
    # response detail without changing retrieval, safety checks, or routing.
    if answer_length == "short":
        system_content += (
            " When the document supports an answer, "
            "give the direct answer in 2-4 sentences "
            "and omit secondary details unless needed for correctness."
        )

    elif answer_length == "long":
        system_content += (
            " When the document supports an answer, give a thorough, "
            "clearly structured answer with relevant "
            "explanations and examples, using only facts supported by the "
            "provided context when retrieval was used."
        )

    system_message = {
        "role": "system",
        "content": system_content,
    }

    # Full message list: system instructions + everything said so far in
    # this conversation + the user's new question, in that order —
    # this is the standard shape every chat-style LLM API expects.
    messages = [
        system_message,
        *conversation_history,
        {"role": "user", "content": user_query},
    ]

    # This goes through Portkey when a saved config is present, otherwise it
    # uses the direct Groq client selected above for local development.
    response = client.chat.completions.create(
        messages=messages,
        model="openai/gpt-oss-120b",
        temperature=0.0,
    )

    record_chat_usage("responder", response)

    return response.choices[0].message.content