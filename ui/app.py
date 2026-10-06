"""
ui/app.py — Streamlit chat interface.

Talks to the FastAPI backend's /query endpoint. Run with:
    streamlit run ui/app.py
(with the FastAPI backend already running separately, e.g. via
`uvicorn app.main:app --reload`)
"""

import os
from textwrap import dedent

import requests
import streamlit as st

# Same BACKEND_URL pattern as your .env config — defaults to local dev,
# but can point at a deployed Cloud Run URL in production.
BACKEND_URL = os.getenv("BACKEND_URL") or str(
    st.secrets.get("BACKEND_URL", "http://localhost:8000")
)
DEMO_AUTH_ENABLED = os.getenv("DEMO_AUTH_ENABLED", "false").casefold() == "true"
PORTAL_SECRETS = st.secrets.get("portal")
PUBLIC_ROLE_SELECTOR = (
    os.getenv("PUBLIC_ROLE_SELECTOR", "false").casefold() == "true"
    or bool(PORTAL_SECRETS and PORTAL_SECRETS.get("public_role_selector", False))
)

st.set_page_config(
    page_title="Knowledge Management Portal",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

if DEMO_AUTH_ENABLED and not PUBLIC_ROLE_SELECTOR:
    demo_user = None
    authenticated_email = None
    api_headers: dict[str, str] = {}
elif PUBLIC_ROLE_SELECTOR:
    backend_shared_secret = str(
        PORTAL_SECRETS.get("backend_shared_secret", "")
        if PORTAL_SECRETS
        else ""
    )
    if not backend_shared_secret:
        st.error("Public role selection is not configured by the app owner.")
        st.stop()
    demo_user = None
    authenticated_email = None
    api_headers = {"Authorization": f"Bearer {backend_shared_secret}"}
else:
    if not st.user.is_logged_in:
        st.title("Knowledge Management Portal")
        st.write("Sign in with your authorized Google account to continue.")
        st.button("Sign in with Google", on_click=st.login, type="primary")
        st.stop()

    authenticated_email = str(st.user.email).strip().casefold()
    portal_secrets = PORTAL_SECRETS
    if not portal_secrets:
        st.error("Portal authentication is not configured by the app owner.")
        st.stop()

    allowed_emails = {
        str(email).strip().casefold()
        for email in portal_secrets.get("allowed_emails", [])
        if str(email).strip()
    }
    backend_shared_secret = str(portal_secrets.get("backend_shared_secret", ""))
    if not allowed_emails or not backend_shared_secret:
        st.error("Portal authentication is not fully configured by the app owner.")
        st.stop()
    if authenticated_email not in allowed_emails:
        st.error("This Google account is not authorized to use the portal.")
        st.button("Sign out", on_click=st.logout)
        st.stop()

    demo_user = None
    api_headers = {
        "Authorization": f"Bearer {backend_shared_secret}",
        "X-Authenticated-Email": authenticated_email,
    }

# Keep the visual treatment quiet and work-focused: a soft neutral canvas,
# restrained green accents, and clear framing for the existing chat workflow.
st.markdown(
    dedent("""
    <style>
    [data-testid="stAppViewContainer"] {
        background: #f6f8f7;
        color: #202b28;
    }
    [data-testid="stMain"] {
        background: #f6f8f7;
    }
    [data-testid="stSidebar"] {
        background: #edf2ef;
        border-right: 1px solid #dce5e0;
        width: 290px !important;
        min-width: 290px !important;
    }
    [data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        max-width: 980px;
        padding: 1.8rem 2rem 2rem;
    }
    .portal-heading {
        display: flex;
        align-items: center;
        gap: 13px;
        padding-bottom: 16px;
        margin-bottom: 12px;
        border-bottom: 1px solid #d5e0da;
    }
    .portal-mark {
        display: grid;
        place-items: center;
        width: 48px;
        height: 48px;
        flex: 0 0 48px;
        border-radius: 8px;
        background: #dceee5;
        font-size: 25px;
    }
    .portal-heading h1 {
        margin: 0;
        color: #20332b;
        font-size: 28px;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: 0;
    }
    .portal-kicker {
        margin: 0 0 4px;
        color: #527665;
        font-size: 11px;
        font-weight: 700;
        line-height: 1.2;
        letter-spacing: 1px;
    }
    .empty-state {
        padding: 3.1rem 0 1.3rem;
    }
    .empty-state h2 {
        margin: 0 0 6px;
        color: #20332b;
        font-size: 22px;
        font-weight: 650;
        letter-spacing: 0;
    }
    .empty-state p {
        margin: 0;
        color: #68766f;
        font-size: 14px;
    }
    [data-testid="stChatMessage"] {
        border: 0;
        border-bottom: 1px solid #e5ebe7;
        border-radius: 0;
        background: transparent;
        padding: 0.65rem 0.25rem 0.85rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        line-height: 1.65;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #c5d4cb;
        border-radius: 8px;
        background: #ffffff;
        box-shadow: 0 3px 12px rgba(30, 58, 43, 0.06);
    }
    .stButton > button {
        border-radius: 6px;
        font-weight: 600;
        transition: border-color 120ms ease, background-color 120ms ease;
    }
    div[data-testid="stButton"] > button {
        height: auto;
        min-height: 3rem;
        padding: 0.65rem 0.85rem;
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
        justify-content: flex-start;
    }
    [data-testid="stDialog"] {
        border: 1px solid #dce5e0;
        border-radius: 10px;
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.2rem 1rem 1.5rem;
        }
        .portal-heading h1 {
            font-size: 23px;
        }
        .portal-mark {
            width: 42px;
            height: 42px;
            flex-basis: 42px;
        }
    }
    </style>
    <div class="portal-heading">
        <div class="portal-mark">🛡️</div>
        <div>
            <p class="portal-kicker">KNOWLEDGE OPERATIONS</p>
            <h1>Knowledge Management Portal</h1>
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

DEFAULT_DEMO_DOCUMENT = "NimbusPay_Enterprise_Policy_Handbook_Demo.docx"

# Active-document state lives only in this Streamlit session. It is never
# restored from Qdrant, so a restart cannot silently reactivate a document.
if "active_document_id" not in st.session_state:
    st.session_state.active_document_id = None
if "active_document_filename" not in st.session_state:
    st.session_state.active_document_filename = None
if "active_document_version" not in st.session_state:
    st.session_state.active_document_version = None
if "skip_default_restore_once" not in st.session_state:
    st.session_state.skip_default_restore_once = False

# Sidebar upload control: this keeps the demo usable without leaving the
# chat interface, while also making it obvious that the knowledge base can
# be extended by ingesting new source files into the vector store.
with st.sidebar:
    st.subheader("🛡️ Knowledge workspace")
    if DEMO_AUTH_ENABLED or PUBLIC_ROLE_SELECTOR:
        demo_user = st.selectbox(
            "Choose a role",
            options=["alice", "bob", "carol"],
            format_func=lambda user: {
                "alice": "Alice · viewer",
                "bob": "Bob · operator",
                "carol": "Carol · administrator",
            }[user],
        )
        if DEMO_AUTH_ENABLED and not PUBLIC_ROLE_SELECTOR:
            api_headers = {"X-Demo-User": demo_user}
            st.caption("Simulated identity for local development.")
        else:
            api_headers["X-Demo-User"] = demo_user
            st.caption(
                "Public demo: anyone can choose a role. Choosing Administrator "
                "grants access to confidential sections and document management."
            )
        is_administrator = demo_user == "carol"
        active_principal = demo_user
    else:
        st.caption(f"Signed in as {authenticated_email}")
        st.button("Sign out", on_click=st.logout)
        is_administrator = True
        active_principal = authenticated_email

    try:
        with st.spinner("Checking indexed documents..."):
            response = requests.get(
                f"{BACKEND_URL}/documents",
                headers=api_headers,
                params={
                    "restore_default": not st.session_state.skip_default_restore_once
                },
                timeout=30,
            )
            response.raise_for_status()
            indexed_documents = response.json().get("documents", [])
            st.session_state.skip_default_restore_once = False
    except requests.RequestException:
        indexed_documents = []

    if indexed_documents:
        default_document = next(
            (doc for doc in indexed_documents if doc.get("source_file") == DEFAULT_DEMO_DOCUMENT),
            None,
        )
        if default_document and not st.session_state.active_document_id:
            st.session_state.active_document_id = default_document["document_id"]
            st.session_state.active_document_filename = default_document.get("source_file")
            st.session_state.active_document_version = default_document.get("version") or default_document["document_id"]

        document_options = {
            doc["document_id"]: f"{doc.get('source_file') or doc['document_id']} ({doc.get('classification') or 'unknown'})"
            for doc in indexed_documents
        }
        selected_document_id = st.selectbox(
            "Select an existing document",
            options=list(document_options.keys()),
            index=(
                list(document_options.keys()).index(st.session_state.active_document_id)
                if st.session_state.active_document_id in document_options
                else 0
            ),
            format_func=lambda document_id: document_options[document_id],
        )
        if selected_document_id and selected_document_id != st.session_state.active_document_id:
            selected_document = next(
                doc for doc in indexed_documents if doc["document_id"] == selected_document_id
            )
            st.session_state.active_document_id = selected_document["document_id"]
            st.session_state.active_document_filename = selected_document.get("source_file")
            st.session_state.active_document_version = selected_document.get("version") or selected_document["document_id"]
            st.session_state.messages = []
            st.session_state.thread_id = None
            st.session_state.pending_refinement = None
            st.rerun()
        st.caption("The NimbusPay handbook is preloaded by default. Upload a custom file only if you want a different document.")
    else:
        st.caption("No indexed documents found for this tenant yet. The NimbusPay handbook is the default demo document.")

    if st.session_state.active_document_id:
        st.success(f"Active document: **{st.session_state.active_document_filename}**")
        st.caption(
            f"Version: {st.session_state.active_document_version[:12]} · "
            "questions are scoped to this document"
        )
        if st.button(
            "Remove Document",
            type="primary",
            use_container_width=True,
            key="remove_active_document",
            disabled=not is_administrator,
        ):
            try:
                response = requests.delete(
                    f"{BACKEND_URL}/documents/{st.session_state.active_document_id}",
                    headers=api_headers,
                    timeout=30,
                )
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"Document removal failed: {exc}")
            else:
                if (
                    st.session_state.active_document_filename
                    == DEFAULT_DEMO_DOCUMENT
                ):
                    st.session_state.skip_default_restore_once = True
                st.session_state.active_document_id = None
                st.session_state.active_document_filename = None
                st.session_state.active_document_version = None
                st.session_state.messages = []
                st.session_state.thread_id = None
                st.session_state.pending_refinement = None
                st.rerun()
        if not is_administrator:
            st.caption("Document removal requires the Administrator role.")

    st.caption("Upload a source document")
    uploaded_file = st.file_uploader(
        "Add to the knowledge base",
        type=["pdf", "docx", "pptx", "html", "txt"],
    )
    if uploaded_file is not None:
        if st.button("Ingest this file"):
            try:
                with st.spinner("Processing document..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    response = requests.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        headers=api_headers,
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()
                if data["document_id"] != st.session_state.active_document_id:
                    st.session_state.messages = []
                    st.session_state.thread_id = None
                    st.session_state.pending_refinement = None
                st.session_state.active_document_id = data["document_id"]
                st.session_state.active_document_filename = data["filename"]
                st.session_state.active_document_version = data["version"]
                if data["reused"]:
                    st.info(f"Reused the existing indexed version of {data['filename']}.")
                else:
                    st.success(
                        f"Added {data['chunks_added']} {data['classification']} chunks "
                        f"from {data['filename']}."
                    )
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")

# --- Session state setup ---------------------------------------------
# st.session_state persists across Streamlit's top-to-bottom re-runs
# (see explanation above). We initialize both values ONLY if they don't
# already exist yet — this check matters because this init code runs on
# EVERY re-run, and we don't want to wipe existing chat history just
# because the script ran again.
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None  # None until the backend gives us one

if "pending_refinement" not in st.session_state:
    st.session_state.pending_refinement = None

identity_changed = (
    "active_principal" in st.session_state
    and st.session_state.active_principal != active_principal
)
if identity_changed:
    st.session_state.messages = []
    st.session_state.thread_id = None
    st.session_state.pending_refinement = None
st.session_state.active_principal = active_principal

if identity_changed:
    st.info("The conversation was cleared after switching users.")


if not st.session_state.messages:
    welcome_heading = (
        "NimbusPay handbook is ready"
        if st.session_state.active_document_filename in (None, DEFAULT_DEMO_DOCUMENT)
        else "Your document is ready"
    )
    st.markdown(
        dedent(f"""
        <div class="empty-state">
            <h2>{welcome_heading}</h2>
            <p>
                Ask a question about the active document. Answers stay grounded in that document and are limited by your demo role.
                <em>Retrieval may take a little longer because the system uses multi-query search and reranking to improve answer quality.</em>
            </p>
        </div>
        """),
        unsafe_allow_html=True,
    )
    if (
        st.session_state.active_document_id
        and st.session_state.active_document_filename == DEFAULT_DEMO_DOCUMENT
    ):
        st.markdown(
            "<p class='portal-kicker' style='margin: 0 0 10px;'>TRY ASKING</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Public-section examples, available to Alice · Viewer and every demo role."
        )
        suggested_questions = (
            "What are the corporate device rules?",
            "What should employees do when they receive a suspicious email?",
            "What are the service desk support hours?",
        )
        suggested_question = None
        for index, question in enumerate(suggested_questions):
            if st.button(
                question,
                key=f"suggested-question-{index}",
                use_container_width=True,
                type="secondary",
            ):
                suggested_question = question
    else:
        suggested_question = None
else:
    suggested_question = None


# --- Render existing chat history -------------------------------------
# On every re-run, we redraw the FULL conversation so far from
# session_state — this is what makes it look like a persistent chat
# window instead of losing history after each message.
for message_index, message in enumerate(st.session_state.messages):
    avatar = "🛡️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        if message.get("blocked"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])
        if message.get("answer_length"):
            st.caption(f"Refined response · {message['answer_length'].title()}")
        if message.get("trace"):
            with st.expander("View step-by-step"):
                for step in message["trace"]:
                    st.markdown(f"- {step}")

        if (
            message["role"] == "assistant"
            and message.get("question")
            and not message.get("blocked")
            and st.button(
                "Refine answer",
                key=f"refine-answer-{message_index}",
                icon=":material/tune:",
            )
        ):
            st.session_state.pending_refinement = {
                "message_index": message_index,
                "question": message["question"],
            }
            st.rerun()

# --- Handle new user input --------------------------------------------
# st.chat_input() returns None on most re-runs, and only returns the
# typed text on the run where the user actually hit enter.
user_input = st.chat_input(
    "Ask a question about the NimbusPay handbook...",
    disabled=not st.session_state.active_document_id,
)
if suggested_question is not None:
    user_input = suggested_question

if user_input:
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching your knowledge base..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "query": user_input,
                        "active_document_id": st.session_state.active_document_id,
                        "thread_id": st.session_state.thread_id,
                    },
                    headers=api_headers,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                status = exc.response.status_code if exc.response is not None else "connection error"
                st.error(
                    f"The backend request failed ({status}). "
                    "Check the FastAPI terminal for the underlying error."
                )
                st.stop()

        if data.get("blocked"):
            st.warning(data["answer"])
        else:
            st.markdown(data["answer"])
        if data.get("trace"):
            with st.expander("View step-by-step"):
                for step in data["trace"]:
                    st.markdown(f"- {step}")

    st.session_state.messages.extend([
        {"role": "user", "content": user_input},
        {
            "role": "assistant",
            "content": data["answer"],
            "question": user_input,
            "blocked": data.get("blocked", False),
            "trace": data.get("trace", []),
            "answer_length": None,
        },
    ])
    st.session_state.thread_id = data["thread_id"]
    st.rerun()


@st.dialog("Refine this answer")
def refine_answer_dialog() -> None:
    """Regenerate an existing answer with the requested level of detail."""
    refinement = st.session_state.pending_refinement
    st.markdown("Choose a shorter or more detailed version.")
    st.info(refinement["question"])

    answer_length = st.radio(
        "Answer length",
        options=["short", "long"],
        format_func=str.title,
        horizontal=True,
        index=0,
        key="refinement_answer_length",
    )

    cancel_column, submit_column = st.columns(2)
    if cancel_column.button("Cancel", use_container_width=True):
        st.session_state.pending_refinement = None
        st.rerun()

    if submit_column.button("Regenerate answer", type="primary", use_container_width=True):
        try:
            with st.spinner("Regenerating answer..."):
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "query": refinement["question"],
                        "active_document_id": st.session_state.active_document_id,
                        "thread_id": st.session_state.thread_id,
                        "answer_length": answer_length,
                    },
                    headers=api_headers,
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()
        except requests.RequestException as exc:
            status = exc.response.status_code if exc.response is not None else "connection error"
            st.error(
                f"The backend request failed ({status}). "
                "Check the FastAPI terminal for the underlying error."
            )
            return

        if data.get("blocked"):
            st.warning(
                "This revised version was blocked by the safety check. "
                "Your existing answer is unchanged. You can try regenerating "
                "it again or cancel to keep the current answer."
            )
            return

        message_index = refinement["message_index"]
        revised_message = dict(st.session_state.messages[message_index])
        revised_message.update({
            "content": data["answer"],
            "blocked": False,
            "trace": data.get("trace", []),
            "answer_length": answer_length,
        })
        st.session_state.messages[message_index] = revised_message
        st.session_state.thread_id = data["thread_id"]
        st.session_state.pending_refinement = None
        st.rerun()


if st.session_state.pending_refinement is not None:
    refine_answer_dialog()