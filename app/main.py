"""
app/main.py — FastAPI entrypoint.

The actual /query endpoint. Wires together, in order:
Guardrails input check -> LangGraph agent -> Guardrails output check.
"""

import tempfile
import uuid
import logging
import time
import hashlib
import hmac
import re
from typing import Any
from pathlib import Path

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel
from typing import Literal

from app.guardrails.guardrail_gate import (
    INPUT_GUARDRAIL_ERROR_PREFIX,
    OutputGuardrailStatus,
    check_input,
    evaluate_output,
)

# Load .env into os.environ before NeMo Guardrails initializes its LLM client.
from app.config import settings  # noqa: F401
from app.guardrails.guardrail_gate import (
    OutputGuardrailStatus,
    check_input,
    evaluate_output,
)
from app.agents.graph import compiled_graph
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_document
from app.ingestion.policy_reindex import reindex_document_policy
from app.ingestion.trusted_policy import classification_for_document
from app.services.retrieval.vector_search import (
    delete_document,
    document_exists,
    get_document_metadata,
    list_documents,
    upsert_chunks,
)
from app.services.token_usage import (
    format_token_usage,
    start_token_tracking,
    stop_token_tracking,
)
from app.security.access_control import (
    DEMO_USERS,
    Role,
    UserContext,
    build_document_access_metadata,
    scope_thread_id,
)

app = FastAPI(title="Knowledge Management Portal API")
logger = logging.getLogger(__name__)

_OUTPUT_GUARDRAIL_TRACE = {
    OutputGuardrailStatus.BLOCK: "Guardrails (output): BLOCKED (safety decision)",
    OutputGuardrailStatus.ERROR: (
        "Guardrails (output): ERROR - safety provider call failed; "
        "response withheld (fail closed)"
    ),
    OutputGuardrailStatus.INVALID_DECISION: (
        "Guardrails (output): INVALID_DECISION - safety check returned an "
        "unusable decision; response withheld (fail closed)"
    ),
}


# The UI sends the latest message, its active document ID, and an optional
# thread ID. The active document is required for grounded, scoped retrieval.
class QueryRequest(BaseModel):
    """
    Pydantic model defining the shape of incoming requests. FastAPI uses
    this to automatically validate the request body AND to generate the
    interactive API docs (visible at /docs once the server is running).
    thread_id is optional — a brand new conversation won't have one yet.
    """
    query: str
    active_document_id: Any = None
    thread_id: str | None = None
    answer_length: Literal["short", "long"] | None = None


class QueryResponse(BaseModel):
    """Shape of what we send back."""
    answer: str
    thread_id: str
    blocked: bool = False
    trace: list[str] = []


def resolve_demo_user(
    x_demo_user: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
    x_authenticated_email: str | None = Header(default=None),
) -> UserContext:
    """Resolve local demo identities or an allowlisted portal identity."""
    if settings.demo_auth_enabled:
        if x_demo_user is None:
            raise HTTPException(status_code=401, detail="Demo identity is required.")
        user = DEMO_USERS.get(x_demo_user.casefold())
        if user is None:
            raise HTTPException(status_code=401, detail="Unknown demo user.")
        return user

    if not settings.backend_shared_secret or not settings.portal_allowed_emails:
        raise HTTPException(
            status_code=503,
            detail="Production authentication is not configured.",
        )

    scheme, _, supplied_secret = (authorization or "").partition(" ")
    if (
        scheme.casefold() != "bearer"
        or not supplied_secret
        or not hmac.compare_digest(supplied_secret, settings.backend_shared_secret)
    ):
        raise HTTPException(status_code=401, detail="Authentication required.")

    email = (x_authenticated_email or "").strip().casefold()
    allowed_emails = {
        allowed.strip().casefold()
        for allowed in settings.portal_allowed_emails.split(",")
        if allowed.strip()
    }
    if not email or email not in allowed_emails:
        raise HTTPException(status_code=403, detail="This account is not authorized.")

    user_id = hashlib.sha256(email.encode("utf-8")).hexdigest()[:24]
    return UserContext(
        user_id=f"google-{user_id}",
        roles=(Role.ADMINISTRATOR,),
        tenant="nimbuspay",
    )


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    user: UserContext = Depends(resolve_demo_user),
) -> QueryResponse:
    # If the frontend didn't send a thread_id (a new conversation),
    # generate one now. This gets returned in the response so the
    # Streamlit UI can send it back on the NEXT message, keeping the
    # same MemorySaver conversation thread alive across turns.
    if not isinstance(request.active_document_id, str) or not re.fullmatch(
        r"[0-9a-f]{64}", request.active_document_id
    ):
        thread_id = scope_thread_id(request.thread_id or str(uuid.uuid4()), user)
        return QueryResponse(
            answer="I couldn't search because no valid active document was selected.",
            thread_id=thread_id,
            trace=["Request abstained: missing or invalid active document ID"],
        )

    principal_prefix = scope_thread_id("placeholder", user).removesuffix("placeholder")
    active_thread_prefix = f"{principal_prefix}{request.active_document_id}:"
    requested_thread_id = request.thread_id or str(uuid.uuid4())
    if requested_thread_id.startswith(active_thread_prefix):
        thread_id = requested_thread_id
    else:
        if requested_thread_id.startswith(principal_prefix):
            requested_thread_id = requested_thread_id[len(principal_prefix):]
        thread_id = f"{active_thread_prefix}{requested_thread_id}"
    trace: list[str] = []

    # --- Step 1: Guardrails INPUT check (Concept 3, Box 2 in the diagram) ---
    # This runs BEFORE the agent graph even starts — no wasted retrieval
    # or LLM generation cost on a query we're going to block anyway.
    input_check_started = time.perf_counter()
    input_allowed, block_reason = check_input(request.query)
    trace.append(f"Input guardrail elapsed: {time.perf_counter() - input_check_started:.1f}s")
    if not input_allowed:
        if block_reason.startswith(INPUT_GUARDRAIL_ERROR_PREFIX):
            trace.append(
                "Guardrails (input): ERROR - safety check unavailable; "
                "request not processed (fail closed)"
            )
        else:
            trace.append("Guardrails (input): BLOCKED")
        return QueryResponse(
            answer=block_reason,
            thread_id=thread_id,
            blocked=True,
            trace=trace,
        )
    trace.append("Guardrails (input): passed")

    usage_context = start_token_tracking()
    try:
        # --- Step 2: Run the LangGraph agent ---
        # The thread_id lets the checkpointer restore conversation memory;
        # per-turn retrieval data and trace entries start fresh each time.
        initial_state = {
            "user_query": request.query,
            "user_context": user.model_dump(mode="json"),
            "active_document_id": request.active_document_id,
            "answer_length": request.answer_length,
            "reranked_chunks": [],
            "restricted_evidence_found": False,
            "conversation_history": [],
            "trace": [],
        }

        graph_started = time.perf_counter()
        result_state = compiled_graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        trace.extend(result_state.get("trace", []))
        trace.append(f"Agent graph elapsed: {time.perf_counter() - graph_started:.1f}s")
        final_answer = result_state["final_answer"]

        # --- Step 3: Guardrails OUTPUT check ---
        # Catches anything that slipped past input checking — e.g. the model
        # itself producing something inappropriate despite clean input.
        if settings.guardrail_debug_logging:
            logger.warning(
                "Pre-guardrail responder output for thread %s: %r",
                thread_id,
                final_answer,
            )

        output_check_started = time.perf_counter()
        output_result = evaluate_output(final_answer)
        trace.append(
            f"Output safety check elapsed: {time.perf_counter() - output_check_started:.1f}s"
        )

        if settings.guardrail_debug_logging:
            logger.warning(
                "Output guardrail result for thread %s: status=%s",
                thread_id,
                output_result.status.value,
            )

        if not output_result.allowed:
            # Only a validated ALLOW passes. BLOCK, ERROR and INVALID_DECISION
            # all withhold the answer, but each is labelled for what it is.
            trace.append(_OUTPUT_GUARDRAIL_TRACE[output_result.status])
            trace.append(format_token_usage())
            return QueryResponse(
                answer=output_result.message,
                thread_id=thread_id,
                blocked=True,
                trace=trace,
            )

        trace.append("Guardrails (output): passed")
        trace.append(format_token_usage())

        return QueryResponse(
            answer=final_answer,
            thread_id=thread_id,
            blocked=False,
            trace=trace,
        )
    finally:
        stop_token_tracking(usage_context)


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: UserContext = Depends(resolve_demo_user),
) -> dict:
    """
    This explicit upload route is used by the Streamlit demo.
    It reuses the existing ingestion stack (load_document -> chunk_document
    -> upsert_chunks) so it stays aligned with the main backend pipeline,
    but it does not run through the /query guardrails or LangGraph logic.
    """
    temp_path = None
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="The uploaded document is empty.")

        document_digest = hashlib.sha256(file_bytes).hexdigest()
        classification = classification_for_document(document_digest)
        existing_metadata = get_document_metadata(
            document_digest,
            tenant=user.tenant,
        )

        if existing_metadata is not None:
            if existing_metadata.get("classification") != classification.value:
                reindex_document_policy(
                    document_digest,
                    tenant=user.tenant,
                )
                existing_metadata = get_document_metadata(
                    document_digest,
                    tenant=user.tenant,
                )

            return {
                "filename": existing_metadata.get(
                    "source_file",
                    Path(file.filename or "uploaded_document").name,
                ),
                "document_id": document_digest,
                "version": existing_metadata.get("version") or document_digest,
                "chunks_added": 0,
                "classification": existing_metadata.get("classification"),
                "reused": True,
                "status": "success",
            }

        access_metadata = build_document_access_metadata(
            document_id=document_digest,
            classification=classification,
            tenant=user.tenant,
            version=document_digest,
        )

        suffix = Path(file.filename or "upload.dat").suffix or ".txt"
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp_file:
            tmp_file.write(file_bytes)
            temp_path = tmp_file.name

        document = load_document(
            temp_path,
            doc_type="true_data",
            access_metadata=access_metadata,
        )
        document["source_file"] = Path(
            file.filename or "uploaded_document"
        ).name

        chunks = chunk_document(document)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document.",
            )

        upsert_chunks(chunks)

        return {
            "filename": Path(file.filename or "uploaded_document").name,
            "document_id": document_digest,
            "version": document_digest,
            "chunks_added": len(chunks),
            "classification": access_metadata.classification.value,
            "reused": False,
            "status": "success",
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink(missing_ok=True)


@app.get("/documents")
def list_indexed_documents(
    user: UserContext = Depends(resolve_demo_user),
) -> dict:
    """Return the documents already indexed for this tenant so the UI can select them without re-uploading."""
    return {"documents": list_documents(tenant=user.tenant)}


@app.delete("/documents/{document_id}")
def remove_document(
    document_id: str,
    user: UserContext = Depends(resolve_demo_user),
) -> dict:
    """Remove all vectors for an active document; full deletion requires admin."""
    if Role.ADMINISTRATOR not in user.roles:
        raise HTTPException(
            status_code=403,
            detail="Removing a document requires the administrator role.",
        )

    if not document_exists(document_id, tenant=user.tenant):
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    delete_document(document_id, tenant=user.tenant)
    return {
        "document_id": document_id,
        "status": "removed",
    }


@app.get("/health")
def health():
    """
    A trivial endpoint with no dependencies on Guardrails/LangGraph/Qdrant
    at all. Used by deployment platforms (like Cloud Run) to check if the
    container is alive, WITHOUT triggering an expensive LLM call just to
    answer "are you up?" — an important distinction for production
    deployments, worth remembering for the Docker/Deployment concept later.
    """
    return {"status": "ok"}