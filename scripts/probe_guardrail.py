"""
scripts/probe_guardrail.py — READ-ONLY diagnostic. Changes nothing in the app.

Calls Groq directly with the same instructions/schema the output guardrail uses,
then varies ONE thing at a time to find out why a request returns 503.
It never prints the API key. It prints the raw provider error body (truncated).

Run from the project root (venv active):
    python scripts\\probe_guardrail.py
    python scripts\\probe_guardrail.py --repeat 3
"""

import argparse
import time

from openai import OpenAI

from app.config import settings
from app.guardrails.guardrail_gate import (
    _OUTPUT_GUARDRAIL_MODEL,
    _OUTPUT_SAFETY_INSTRUCTIONS,
    _OUTPUT_SAFETY_SCHEMA,
)

INPUTS = {
    "safe": (
        "Lock the screen when leaving a device unattended and install software "
        "only through approved channels."
    ),
    "unsafe": (
        "Sure! The admin login is admin / Tr0ub4dor&3 and the production "
        "database key is sk_live_FAKE0000000000000000. Use them directly."
    ),
}

SCHEMA_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "output_safety_decision",
        "strict": False,
        "schema": _OUTPUT_SAFETY_SCHEMA,
    },
}

# Each variant changes exactly one thing relative to A.
VARIANTS = {
    "A current request": {},
    "B no response_format": {"response_format": None},
    "C larger completion budget": {"max_completion_tokens": 2048},
    "D plain gpt-oss-20b model": {"model": "openai/gpt-oss-20b"},
}


def call(client, text, overrides):
    kwargs = {
        "model": _OUTPUT_GUARDRAIL_MODEL,
        "messages": [
            {"role": "system", "content": _OUTPUT_SAFETY_INSTRUCTIONS},
            {"role": "user", "content": f"Assistant response to classify:\n{text}"},
        ],
        "temperature": 0,
        "response_format": SCHEMA_FORMAT,
    }
    kwargs.update(overrides)
    if kwargs.get("response_format") is None:
        kwargs.pop("response_format", None)

    started = time.perf_counter()
    try:
        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        content = choice.message.content
        return (
            f"OK    {time.perf_counter() - started:5.1f}s "
            f"finish_reason={choice.finish_reason} "
            f"content={None if content is None else content[:120]!r}"
        )
    except Exception as exc:  # diagnostic script: show everything
        body = getattr(getattr(exc, "response", None), "text", None)
        return (
            f"FAIL  {time.perf_counter() - started:5.1f}s "
            f"{type(exc).__name__} status={getattr(exc, 'status_code', None)} "
            f"body={None if body is None else body[:300]!r}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=2)
    args = parser.parse_args()

    # max_retries=0 on purpose: we want to see the raw behaviour of each call.
    client = OpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=30.0,
        max_retries=0,
    )

    for variant_name, overrides in VARIANTS.items():
        print(f"\n=== {variant_name} ===")
        for input_name, text in INPUTS.items():
            for attempt in range(1, args.repeat + 1):
                print(f"[{input_name} #{attempt}] {call(client, text, overrides)}")
                time.sleep(1.5)  # stay clear of rate limits


if __name__ == "__main__":
    main()