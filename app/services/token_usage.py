"""Request-scoped tracking of provider-reported chat token usage."""

from contextvars import ContextVar, Token
from typing import Any

_usage_records: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "chat_token_usage_records",
    default=None,
)


def start_token_tracking() -> Token[list[dict[str, Any]] | None]:
    """Start an isolated usage list for the current request context."""
    return _usage_records.set([])


def stop_token_tracking(token: Token[list[dict[str, Any]] | None]) -> None:
    """Restore the previous context value so requests cannot share counts."""
    _usage_records.reset(token)


def record_chat_usage(label: str, response: Any) -> None:
    """Record usage fields returned by an OpenAI-compatible chat response."""
    records = _usage_records.get()
    if records is None:
        return

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    def read_count(attribute: str, key: str) -> int | None:
        value = getattr(usage, attribute, None) if usage is not None else None
        if value is None and isinstance(usage, dict):
            value = usage.get(key)
        return int(value) if value is not None else None

    prompt_tokens = read_count("prompt_tokens", "prompt_tokens")
    completion_tokens = read_count("completion_tokens", "completion_tokens")
    total_tokens = read_count("total_tokens", "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    records.append({
        "label": label,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    })


def format_token_usage() -> str:
    """Summarize exact reported chat tokens, marking any unavailable counts."""
    records = _usage_records.get() or []
    known_records = [record for record in records if record["total_tokens"] is not None]

    prompt_total = sum(record["prompt_tokens"] or 0 for record in known_records)
    completion_total = sum(record["completion_tokens"] or 0 for record in known_records)
    total = sum(record["total_tokens"] or 0 for record in known_records)
    breakdown = ", ".join(
        f"{record['label']} {record['total_tokens']}"
        for record in known_records
    ) or "no provider usage reported"
    unavailable_count = len(records) - len(known_records)
    unavailable_note = (
        f"; usage unavailable for {unavailable_count} chat call(s)"
        if unavailable_count
        else ""
    )

    return (
        f"Chat token usage (provider-reported): {prompt_total} prompt + "
        f"{completion_total} completion = {total} total across {len(records)} call(s) "
        f"[{breakdown}]{unavailable_note}. Excludes guardrail and embedding calls."
    )