"""
config.py — Centralized environment variable management.

This is the ONLY file in the project that should read from os.environ / .env
directly. Every other file imports the `settings` object from here instead
of calling os.getenv() itself — that way, all config lives in one place,
and if a required key is missing, the app fails LOUDLY at startup instead
of silently crashing later mid-request (a much worse debugging experience).
"""

# pydantic-settings is a small library built on top of Pydantic (already in
# your stack) specifically for loading and validating environment variables.
# We use it instead of plain os.getenv() because it gives us automatic
# type validation and startup-time errors for free.
from pydantic_settings import BaseSettings, SettingsConfigDict

# IMPORTANT: pydantic-settings loads .env into OUR Settings object below,
# but does NOT put those values into the real system environment
# (os.environ) for OTHER libraries to find. Some libraries we depend on
# (like the Groq client NeMo Guardrails uses internally) read directly
# from os.environ themselves, bypassing our Settings object entirely.
# load_dotenv() here explicitly copies everything from .env into the
# real os.environ, so EVERY library — ours or a third-party one — can
# find these values, regardless of how it looks them up.
from dotenv import load_dotenv
load_dotenv(override=True)


class Settings(BaseSettings):
    """
    Each attribute below maps to one line in your .env file.
    pydantic-settings matches them by name automatically (case-insensitive).
    Declaring the type (str) means: if this variable is missing or empty
    when the app starts, Pydantic raises a clear validation error
    immediately — you'll know exactly which key is missing, at startup,
    not three hours later when a user triggers the code path that needed it.
    """

    # --- Groq (primary LLM + fallback) ---
    groq_api_key: str
    groq_fallback_api_key: str

    # --- Portkey LLM Gateway ---
    portkey_api_key: str
    portkey_config: str | None = None

    # --- Qdrant Vector DB ---
    qdrant_api_key: str
    qdrant_cluster_endpoint: str

    # --- Pydantic Logfire (observability) ---
    logfire_token: str

    # --- LangSmith (tracing) ---
    langsmith_tracing: bool = True   # default True if not set in .env
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: str
    langsmith_project: str

    # --- Streamlit UI -> FastAPI backend URL ---
    backend_url: str = "http://localhost:8000"

    # --- Eval judge LLM (kept separate from groq_api_key on purpose —
    #     see our RAGAS notes: this prevents a big eval run from
    #     rate-limiting real production users) ---
    judge_groq: str

    # --- Gemini Embeddings ---
    gemini_api_key: str

    # Header-based principals are only for the local authorization demo.
    # Production deployments must replace this with verified JWT/IAM identity.
    demo_auth_enabled: bool = False

    # Raw responder and guardrail outputs can contain document content;
    # keep diagnostics disabled unless explicitly debugging locally.
    guardrail_debug_logging: bool = False

    # This tells pydantic-settings WHERE to look for the .env file,
    # and to ignore any extra keys in .env we haven't declared above
    # (rather than crashing on them).
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


# This line runs ONCE, the moment this module is first imported anywhere
# in the app. If any required key above is missing from your .env file,
# the app will crash right here, at import time — which is exactly what
# we want: fail at startup, not mid-request.
settings = Settings()