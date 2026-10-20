"""
ui/app.py — Streamlit chat interface.

Talks to the FastAPI backend's /query endpoint. Run with:
    streamlit run ui/app.py
(with the FastAPI backend already running separately, e.g. via
`uvicorn app.main:app --reload`)
"""

import os
import re
from html import escape
from textwrap import dedent

import requests
import streamlit as st

# Same BACKEND_URL pattern as your .env config — defaults to local dev,
# but can point at a deployed Cloud Run URL in production.
BACKEND_URL = os.getenv("BACKEND_URL") or str(
    st.secrets.get("BACKEND_URL", "http://localhost:8000")
)

_SELECTED_SOURCE_PATTERN = re.compile(
    r"^Selected chunk: source=(?P<source>.*?), document_id=.*?, "
    r"chunk_index=(?P<chunk_index>[^,]+), rerank_score=(?P<score>-?\d+(?:\.\d+)?)$"
)


def extract_selected_sources(trace: list[str]) -> list[dict[str, str]]:
    """Read only source metadata already emitted in the authorized retrieval trace."""
    sources = []
    seen = set()
    for item in trace:
        match = _SELECTED_SOURCE_PATTERN.match(item)
        if not match:
            continue
        source = match.group("source")
        chunk_index = match.group("chunk_index")
        key = (source, chunk_index)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "source": source,
                "chunk_index": chunk_index,
                "score": match.group("score"),
            }
        )
    return sources


def render_answer_evidence(trace: list[str], *, blocked: bool = False) -> None:
    """Present real retrieved-source metadata without manufacturing citations."""
    sources = extract_selected_sources(trace)
    if sources:
        with st.expander(f"Sources · {len(sources)} retrieved chunk(s)"):
            for source in sources:
                st.markdown(
                    "<div class='source-record'>"
                    f"<strong>{escape(source['source'])}</strong>"
                    "<span class='source-meta'>"
                    f"Chunk {escape(source['chunk_index'])} · "
                    f"Rerank score {escape(source['score'])}"
                    "</span></div>",
                    unsafe_allow_html=True,
                )
        return

    if blocked:
        return
    if any("small-talk prompt (no retrieval)" in item for item in trace):
        st.caption("No document sources were used for this response.")
    elif any(
        "Responder skipped" in item or "Retriever abstained" in item
        for item in trace
    ):
        st.caption("No supporting document chunks were selected for this response.")
    elif trace:
        st.caption("No source chunks were included in the response trace.")


DEMO_AUTH_ENABLED = os.getenv("DEMO_AUTH_ENABLED", "false").casefold() == "true"
PUBLIC_ROLE_SELECTOR = os.getenv("PUBLIC_ROLE_SELECTOR", "false").casefold() == "true"
if PUBLIC_ROLE_SELECTOR:
    PORTAL_SECRETS = None
else:
    PORTAL_SECRETS = st.secrets.get("portal")
    PUBLIC_ROLE_SELECTOR = bool(
        PORTAL_SECRETS and PORTAL_SECRETS.get("public_role_selector", False)
    )

st.set_page_config(
    page_title="Knowledge Management Portal",
    page_icon="K",
    initial_sidebar_state="auto",
)

if DEMO_AUTH_ENABLED and not PUBLIC_ROLE_SELECTOR:
    demo_user = None
    authenticated_email = None
    api_headers: dict[str, str] = {}
elif PUBLIC_ROLE_SELECTOR:
    backend_shared_secret = str(
        os.getenv("BACKEND_SHARED_SECRET")
        or (
            PORTAL_SECRETS.get("backend_shared_secret", "")
            if PORTAL_SECRETS
            else ""
        )
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
    backend_shared_secret = str(
        os.getenv("BACKEND_SHARED_SECRET")
        or portal_secrets.get("backend_shared_secret", "")
    )
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

# Keep presentation styles separate from the existing request and session flows.
st.markdown(
    dedent("""
    <style>
    :root {
        --portal-bg: #07111f;
        --portal-sidebar: #0b1728;
        --portal-panel: #0d1b2e;
        --portal-border: #203650;
        --portal-text: #e8eef7;
        --portal-muted: #93a4ba;
        --portal-blue: #3b82f6;
    }
    body {
        font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: var(--portal-bg);
        color: var(--portal-text);
    }
    [data-testid="stSidebar"] {
        background: var(--portal-sidebar);
        border-right: 1px solid var(--portal-border);
        min-width: 300px !important;
        width: 300px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        background: var(--portal-sidebar);
    }
    .block-container {
        max-width: 1160px;
        padding: 2rem 3rem 3rem;
    }
    .portal-heading {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 210px;
        align-items: center;
        gap: 1.5rem;
        min-height: 175px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid #1c3552;
        border-radius: 10px;
        background:
            radial-gradient(ellipse at 90% 80%, rgba(22, 83, 159, 0.18), transparent 40%),
            linear-gradient(115deg, #0d1b2e 0%, #0b192d 65%, #0b1d36 100%);
    }
    .portal-heading h1,
    .assistant-intro h2 {
        color: var(--portal-text);
    }
    .portal-heading h1 {
        margin: 0;
        font-size: clamp(1.2rem, 2vw, 1.45rem);
        font-weight: 650;
        line-height: 1.25;
        letter-spacing: -0.025em;
    }
    .portal-kicker {
        margin: 0 0 0.5rem;
        color: #7eb6ff;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.13em;
    }
    .portal-subtitle {
        margin: 0.3rem 0 0;
        color: var(--portal-muted);
        font-size: 0.88rem;
    }
    .hero-art {
        position: relative;
        height: 145px;
        overflow: hidden;
    }
    .hero-document {
        position: absolute;
        top: 20px;
        left: 55px;
        width: 88px;
        height: 112px;
        padding: 25px 12px 10px;
        border: 1px solid #3b79bc;
        border-radius: 8px;
        background: linear-gradient(155deg, #1b4d83, #102a4a 82%);
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
        animation: document-float 4.2s ease-in-out infinite;
    }
    .hero-document::before {
        position: absolute;
        top: 10px;
        left: 12px;
        width: 27px;
        height: 4px;
        border-radius: 2px;
        background: #64b5ff;
        content: "";
    }
    .hero-document::after {
        position: absolute;
        top: 22px;
        left: 12px;
        width: 54px;
        height: 40px;
        background: repeating-linear-gradient(
            to bottom,
            rgba(157, 205, 249, 0.72) 0 3px,
            transparent 3px 11px
        );
        content: "";
    }
    .hero-document-back {
        position: absolute;
        top: 31px;
        left: 34px;
        width: 88px;
        height: 105px;
        border: 1px solid rgba(65, 128, 195, 0.65);
        border-radius: 8px;
        background: rgba(14, 38, 67, 0.65);
        transform: rotate(-8deg);
        animation: document-blink 3.6s ease-in-out infinite;
    }
    .hero-shield {
        position: absolute;
        top: 3px;
        right: 23px;
        display: grid;
        width: 33px;
        height: 33px;
        place-items: center;
        border: 1px solid #2c5c8b;
        border-radius: 50%;
        background: #102747;
        color: #8fc6ff;
        animation: document-blink 3.6s ease-in-out infinite;
    }
    @keyframes document-float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    @keyframes document-blink {
        0%, 100% { opacity: 0.58; }
        50% { opacity: 1; }
    }
    .trust-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    .trust-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        min-height: 30px;
        padding: 0.3rem 0.6rem;
        border: 1px solid #254664;
        border-radius: 999px;
        background: rgba(8, 22, 39, 0.72);
        color: #a8bdd3;
        font-size: 0.67rem;
        line-height: 1.25;
    }
    .trust-pill svg {
        width: 14px;
        height: 14px;
        flex: 0 0 14px;
        color: #60b9f0;
    }
    .trust-pill strong {
        color: #dce8f5;
        font-weight: 600;
    }
    .sidebar-security {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        margin-top: 1.25rem;
        padding: 0.8rem;
        border: 1px solid #203b59;
        border-radius: 8px;
        background: #0d1b2e;
    }
    .sidebar-security svg {
        width: 19px;
        height: 19px;
        flex: 0 0 19px;
        color: #4cc8d8;
    }
    .sidebar-security strong {
        display: block;
        color: #dce8f5;
        font-size: 0.74rem;
        font-weight: 600;
    }
    .sidebar-security span {
        display: block;
        margin-top: 0.2rem;
        color: var(--portal-muted);
        font-size: 0.67rem;
        line-height: 1.4;
    }
    .portal-brand {
        display: flex;
        gap: 0.8rem;
        align-items: center;
        padding: 0.5rem 0 1.25rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--portal-border);
    }
    .portal-mark {
        display: grid;
        place-items: center;
        width: 36px;
        height: 36px;
        flex: 0 0 36px;
        border: 1px solid #31516f;
        border-radius: 9px;
        background: #10243a;
        color: #a8d7ff;
        font-size: 16px;
        font-weight: 700;
    }
    .portal-brand-name {
        margin: 0;
        color: var(--portal-text);
        font-size: 0.88rem;
        font-weight: 650;
        line-height: 1.3;
    }
    .portal-brand-subtitle {
        margin: 0.2rem 0 0;
        color: var(--portal-muted);
        font-size: 0.72rem;
        line-height: 1.35;
    }
    .workspace-label {
        margin: 1.1rem 0 0.45rem;
        color: #aab9cb;
        font-size: 0.69rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .document-list {
        max-height: 190px;
        overflow-y: auto;
        margin: 0.4rem 0 0.65rem;
        border-top: 1px solid var(--portal-border);
    }
    .document-row {
        padding: 0.55rem 0.1rem;
        border-bottom: 1px solid rgba(32, 54, 80, 0.75);
    }
    .document-name {
        color: var(--portal-text);
        font-size: 0.76rem;
        line-height: 1.4;
        overflow-wrap: anywhere;
    }
    .document-meta {
        margin-top: 0.15rem;
        color: var(--portal-muted);
        font-size: 0.68rem;
    }
    .source-record {
        padding: 0.65rem 0;
        border-bottom: 1px solid var(--portal-border);
        color: var(--portal-text);
        font-size: 0.82rem;
        line-height: 1.5;
        overflow-wrap: anywhere;
    }
    .source-record:last-child {
        border-bottom: 0;
    }
    .source-meta {
        display: block;
        margin-top: 0.15rem;
        color: var(--portal-muted);
        font-size: 0.72rem;
    }
    .assistant-intro {
        padding: 1.35rem 0 1rem;
        text-align: center;
    }
    .assistant-intro h2 {
        margin: 0 0 0.45rem;
        font-size: 1.2rem;
        font-weight: 620;
        letter-spacing: -0.02em;
    }
    .assistant-intro p {
        margin: 0;
        color: var(--portal-muted);
        font-size: 0.9rem;
    }
    .assistant-sparkle {
        display: grid;
        width: 42px;
        height: 42px;
        place-items: center;
        margin: 0 auto 0.75rem;
        border: 1px solid #2869a8;
        border-radius: 50%;
        background: #0b213a;
        color: #8bc7ff;
        font-size: 1.25rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--portal-border);
        border-radius: 10px;
        background: linear-gradient(160deg, rgba(13, 27, 46, 0.82), rgba(8, 20, 35, 0.7));
    }
    [data-testid="stChatMessage"] {
        border: 1px solid var(--portal-border);
        border-radius: 11px;
        background: var(--portal-panel);
        padding: 0.85rem 1rem;
        margin-bottom: 0.8rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        color: var(--portal-text);
        line-height: 1.72;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        overflow-wrap: anywhere;
    }
    [data-testid="stChatMessage"] pre,
    [data-testid="stChatMessage"] table {
        max-width: 100%;
        overflow-x: auto;
    }
    [data-testid="stChatMessage"] pre {
        background: #091626;
    }
    [data-testid="stChatMessage"] [data-testid="stAvatar"] {
        border: 1px solid #31516f;
        background: #132a43;
        color: #b9d9f6;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #2a4562;
        border-radius: 11px;
        background: var(--portal-panel);
        box-shadow: none;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #5596e8;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.18);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--portal-text);
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #8193a9;
    }
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button {
        min-height: 2.35rem;
        border: 1px solid var(--portal-border);
        border-radius: 8px;
        background: #102039;
        color: var(--portal-text);
        font-weight: 600;
        transition: border-color 120ms ease, background-color 120ms ease;
    }
    [data-testid="stButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover {
        border-color: #42698c;
        background: #142a44;
        color: #ffffff;
    }
    [data-testid="stButton"] > button:focus-visible,
    [data-testid="stChatInput"] textarea:focus-visible,
    [data-testid="stSelectbox"] [role="combobox"]:focus-visible {
        outline: 2px solid #79b5ff;
        outline-offset: 2px;
    }
    [data-testid="stElementContainer"][class*="st-key-suggested-question-"] button {
        height: 100%;
        min-height: 3.1rem;
        justify-content: flex-start;
        text-align: left;
        white-space: normal;
        overflow-wrap: anywhere;
        background: var(--portal-panel);
    }
    [data-testid="stChatInputSubmitButton"] {
        width: 38px;
        height: 38px;
        border-radius: 8px !important;
        background: var(--portal-blue) !important;
        color: white !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInputSubmitButton"]:disabled {
        background: #1a2b40 !important;
        color: #8293a8 !important;
        cursor: not-allowed;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--portal-text);
        font-size: 0.9rem;
        font-weight: 620;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: var(--portal-muted);
        font-size: 0.78rem;
        line-height: 1.55;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small {
        border-color: var(--portal-border);
        background: var(--portal-panel);
        color: var(--portal-text);
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        border-style: dashed;
    }
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stRadio"] label {
        color: var(--portal-text);
    }
    [data-testid="stExpander"],
    [data-testid="stDialog"] {
        border: 1px solid var(--portal-border);
        border-radius: 10px;
        background: var(--portal-panel);
    }
    [data-testid="stExpander"] {
        margin-top: 0.45rem;
    }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        color: var(--portal-text);
    }
    [data-testid="stAlert"] {
        border-radius: 9px;
    }
    [data-testid="stMarkdownContainer"] pre,
    [data-testid="stMarkdownContainer"] code {
        border-color: var(--portal-border);
    }
    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
        }
    }
    @media (max-width: 640px) {
        .block-container {
            padding: 1.25rem 1rem 2rem;
        }
        .portal-heading {
            grid-template-columns: minmax(0, 1fr);
            min-height: auto;
            padding: 1.1rem;
            padding-bottom: 1rem;
            margin-bottom: 1rem;
        }
        .portal-heading h1 {
            font-size: 1.2rem;
        }
        .assistant-intro {
            padding-top: 1.1rem;
        }
        .hero-art {
            display: none;
        }
        .trust-strip {
            gap: 0.35rem;
        }
        .trust-pill {
            white-space: normal;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        .hero-document,
        .hero-document-back,
        .hero-shield {
            animation: none;
        }
    }
    </style>
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
if "confirm_remove_active_document" not in st.session_state:
    st.session_state.confirm_remove_active_document = False

# Sidebar upload control: this keeps the demo usable without leaving the
# chat interface, while also making it obvious that the knowledge base can
# be extended by ingesting new source files into the vector store.
with st.sidebar:
    st.markdown(
        """
        <div class="portal-brand">
            <div class="portal-mark">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
                     aria-hidden="true">
                    <path d="M12 3 19 6v5c0 4.6-2.8 8-7 10-4.2-2-7-5.4-7-10V6l7-3Z"/>
                    <path d="m9 12 2 2 4-4"/>
                </svg>
            </div>
            <div>
                <p class="portal-brand-name">Knowledge Management Portal</p>
                <p class="portal-brand-subtitle">Enterprise Knowledge Workspace</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("Workspace")
    if DEMO_AUTH_ENABLED or PUBLIC_ROLE_SELECTOR:
        demo_user = st.selectbox(
            "Demo role · access-control test",
            options=["alice", "bob", "carol"],
            format_func=lambda user: {
                "alice": "Alice — Viewer",
                "bob": "Bob — Operator",
                "carol": "Carol — Administrator",
            }[user],
        )
        if DEMO_AUTH_ENABLED and not PUBLIC_ROLE_SELECTOR:
            api_headers = {"X-Demo-User": demo_user}
            st.caption("Simulated identity for local development.")
        else:
            api_headers["X-Demo-User"] = demo_user
            st.caption(
                "Public demonstration only. Role selection is simulated and is "
                "not production authentication."
            )
        is_administrator = demo_user == "carol"
        active_principal = demo_user
    else:
        st.caption(f"Signed in as {authenticated_email}")
        st.button("Sign out", on_click=st.logout)
        is_administrator = True
        active_principal = authenticated_email

    document_library_error = None
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
    except requests.RequestException as exc:
        indexed_documents = []
        status = exc.response.status_code if exc.response is not None else "connection error"
        document_library_error = (
            f"Document library unavailable ({status}). Check the API connection and retry."
        )

    st.markdown("#### Document library")
    if document_library_error:
        st.warning(document_library_error)
    elif indexed_documents:
        st.caption(f"{len(indexed_documents)} indexed document(s)")
        document_rows = []
        for document in indexed_documents:
            filename = str(document.get("source_file") or "Untitled document")
            classification = str(document.get("classification") or "Classification unavailable")
            document_rows.append(
                "<div class='document-row'>"
                f"<div class='document-name' title='{escape(filename, quote=True)}'>"
                f"{escape(filename)}</div>"
                f"<div class='document-meta'>{escape(classification)}</div>"
                "</div>"
            )
        st.markdown(
            "<div class='document-list'>" + "".join(document_rows) + "</div>",
            unsafe_allow_html=True,
        )
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
            "Active source",
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
    else:
        st.caption("No indexed documents are available for this workspace.")

    if st.session_state.active_document_id:
        st.markdown("#### Active document")
        st.caption(st.session_state.active_document_filename or "Selected document")
        st.caption(
            f"Version: {st.session_state.active_document_version[:12]} · "
            "questions are scoped to this document"
        )
        if st.button(
            "Remove from knowledge base",
            use_container_width=True,
            key="remove_active_document",
            disabled=not is_administrator,
        ):
            st.session_state.confirm_remove_active_document = True
            st.rerun()
        if st.session_state.confirm_remove_active_document:
            st.warning("Remove this document from the knowledge base? This cannot be undone.")
            confirm_column, cancel_column = st.columns(2)
            if cancel_column.button(
                "Cancel",
                use_container_width=True,
                key="cancel_remove_document",
            ):
                st.session_state.confirm_remove_active_document = False
                st.rerun()
            if confirm_column.button(
                "Confirm removal",
                type="primary",
                use_container_width=True,
                key="confirm_remove_document",
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
                    st.session_state.confirm_remove_active_document = False
                    st.rerun()
        if not is_administrator:
            st.caption("Document removal requires the Administrator role.")

    st.markdown("#### Add knowledge")
    st.caption("Upload a source document to index it for authorized questions.")
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
    st.markdown(
        """
        <div class="sidebar-security">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                 stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"
                 aria-hidden="true">
                <path d="M12 3 19 6v5c0 4.6-2.8 8-7 10-4.2-2-7-5.4-7-10V6l7-3Z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
            <div>
                <strong>Access controls</strong>
                <span>Document access is enforced by the backend for each request.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


available_document_ids = {
    document.get("document_id") for document in indexed_documents
}
active_document_count = int(
    bool(st.session_state.active_document_id)
    and st.session_state.active_document_id in available_document_ids
)
latest_assistant_message = next(
    (
        message
        for message in reversed(st.session_state.messages)
        if message.get("role") == "assistant"
    ),
    None,
)
latest_sources = (
    extract_selected_sources(latest_assistant_message.get("trace", []))
    if latest_assistant_message
    else []
)
if document_library_error:
    trusted_sources_value = "Unavailable"
    trusted_sources_detail = "Document library could not be reached."
else:
    trusted_sources_value = f"{len(indexed_documents)} indexed"
    trusted_sources_detail = f"{active_document_count} active in this conversation"

if DEMO_AUTH_ENABLED or PUBLIC_ROLE_SELECTOR:
    current_access = {
        "alice": "Alice · Viewer",
        "bob": "Bob · Operator",
        "carol": "Carol · Administrator",
    }[demo_user]
    access_detail = "Simulated demo role · backend access controls apply"
else:
    current_access = "Authenticated session"
    access_detail = "Requests are subject to backend access controls"

if latest_assistant_message is None:
    evidence_value = "Shown per response"
    evidence_detail = "No response evidence to display yet."
elif latest_assistant_message.get("blocked"):
    evidence_value = "Withheld"
    evidence_detail = "The response was blocked by a safety check."
elif latest_sources:
    evidence_value = f"{len(latest_sources)} source chunk(s)"
    evidence_detail = "Metadata returned in the authorized retrieval trace."
else:
    evidence_value = "No sources returned"
    evidence_detail = "No retrieved source chunks were exposed for the latest answer."

trust_items = (
    (
        "TRUSTED SOURCES",
        trusted_sources_value,
        trusted_sources_detail,
    ),
    ("SECURE ACCESS", current_access, access_detail),
    ("EVIDENCE-BACKED ANSWERS", evidence_value, evidence_detail),
)
trust_icons = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M12 3 19 6v5c0 4.6-2.8 8-7 10-4.2-2-7-5.4-7-10V6l7-3Z"/><path d="m9 12 2 2 4-4"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/><path d="M12 14v3"/></svg>',
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M5 4h14v16H5z"/><path d="M8 8h8M8 12h8M8 16h5"/></svg>',
)
trust_pills = "".join(
    "<div class='trust-pill'>"
    f"{icon}<span><strong>{escape(title)}</strong> · {escape(value)}</span></div>"
    for icon, (title, value, _) in zip(trust_icons, trust_items)
)
shield_icon = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
    'aria-hidden="true"><path d="M12 3 19 6v5c0 4.6-2.8 8-7 10-4.2-2-7-5.4-7-10V6l7-3Z"/>'
    '<path d="m9 12 2 2 4-4"/></svg>'
)
st.markdown(
    "<div class='portal-heading'>"
    "<div class='hero-copy'>"
    "<p class='portal-kicker'>KNOWLEDGE WORKSPACE</p>"
    "<h1>Knowledge Management Portal</h1>"
    "<p class='portal-subtitle'>Enterprise Knowledge Workspace · Find answers across your authorized documents.</p>"
    f"<div class='trust-strip'>{trust_pills}</div>"
    "</div>"
    "<div class='hero-art' aria-hidden='true'>"
    "<div class='hero-document-back'></div>"
    "<div class='hero-document'></div>"
    f"<div class='hero-shield'>{shield_icon}</div>"
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    with st.container(border=True):
        st.markdown(
            """
            <div class="assistant-intro">
                <div class="assistant-sparkle" aria-hidden="true">✦</div>
                <h2>Ask your knowledge base</h2>
                <p>Get answers from your available documents and authorized knowledge sources.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.active_document_id:
            st.markdown(
                "<p class='portal-kicker'>SUGGESTED PROMPTS</p>",
                unsafe_allow_html=True,
            )
            suggested_question = None
            suggested_questions = (
                "Summarize the selected document.",
                "Find a specific policy or procedure.",
                "Explain a documented technical issue.",
                "Compare two topics covered by the available sources.",
            )
            prompt_columns = st.columns(2)
            for index, question in enumerate(suggested_questions):
                if prompt_columns[index % 2].button(
                    question,
                    key=f"suggested-question-{index}",
                    use_container_width=True,
                    type="secondary",
                ):
                    suggested_question = question
        else:
            suggested_question = None
            st.info(
                "Select an indexed document from the workspace sidebar to start a conversation."
            )
else:
    suggested_question = None


# --- Render existing chat history -------------------------------------
# On every re-run, we redraw the FULL conversation so far from
# session_state — this is what makes it look like a persistent chat
# window instead of losing history after each message.
for message_index, message in enumerate(st.session_state.messages):
    is_assistant = message["role"] == "assistant"
    avatar = ":material/auto_awesome:" if is_assistant else ":material/person:"
    with st.chat_message(message["role"], avatar=avatar):
        st.caption("Knowledge Assistant" if is_assistant else "You")
        if message.get("blocked"):
            st.warning(message["content"])
        else:
            if is_assistant:
                st.markdown("**Answer**")
            st.markdown(message["content"])
        if message.get("answer_length"):
            st.caption(f"Refined response · {message['answer_length'].title()}")
        render_answer_evidence(
            message.get("trace", []),
            blocked=message.get("blocked", False),
        )
        if message.get("trace"):
            with st.expander("Retrieval and safety details"):
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
    "Ask a question about the selected document...",
    disabled=not st.session_state.active_document_id,
)
if suggested_question is not None:
    user_input = suggested_question

if user_input:
    with st.chat_message("user", avatar=":material/person:"):
        st.caption("You")
        st.markdown(user_input)
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.caption("Knowledge Assistant")
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
            st.markdown("**Answer**")
            st.markdown(data["answer"])
        render_answer_evidence(
            data.get("trace", []),
            blocked=data.get("blocked", False),
        )
        if data.get("trace"):
            with st.expander("Retrieval and safety details"):
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