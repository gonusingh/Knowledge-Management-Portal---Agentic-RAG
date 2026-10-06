# Prototype Document Authorization

## Current Boundary

The application applies tenant, classification, and role filters inside
Qdrant before any vector results are returned to the retriever, reranker, or
responder. Missing ACL metadata does not match the filter. Technical queries
with no authorized context return a fixed message without calling the responder
LLM.

When authorized results are absent or all fall below the relevance floor, the
retriever performs a second, tenant- and active-document-scoped search without
the role filter. It reranks only chunks whose valid ACL metadata excludes the
current role, locally, and passes only a boolean to graph state. Restricted
chunk text is never placed in graph state, the trace, or the answer-generation
LLM context. A relevant restricted match produces a fixed access-denied
response; otherwise the existing no-evidence response is retained. This
intentionally reveals that a relevant topic exists in restricted material and
is suitable for the demo, not for deployments that require topic-existence
confidentiality.

This is a local prototype, not production authentication. `X-Demo-User` selects
one of three server-defined demo principals, but a caller can forge that header.
Do not expose this mode to untrusted users or deploy it as an authorization
system. Replace `resolve_demo_user` with a verified JWT or identity-provider
dependency before deployment.

Conversation checkpoint IDs are scoped by tenant, user, role set, and active
document; the UI clears local chat history when switching demo principals or
documents. This prevents a prior answer from being reused across document or
role changes. The selected document ID is held only in Streamlit session state
and is not restored after an application restart.

## Test Policy

| Role | PUBLIC | INTERNAL | CONFIDENTIAL | RESTRICTED |
| --- | --- | --- | --- | --- |
| viewer | Allow | Deny | Deny | Deny |
| operator | Allow | Allow | Deny | Deny |
| administrator | Allow | Allow | Allow | Allow |

The application has exactly these three roles. `RESTRICTED` is a document
classification, not a separate role, and is visible only to Administrators.

All documents are scoped to a tenant. The batch ingestion script assigns
classification from its trusted folder policy (`true_data` is PUBLIC and
`noisy_data` is INTERNAL by default); it never derives access from document
text. The default NimbusPay handbook demo document has trusted paragraph ranges
keyed to its SHA-256 hash: public service information, internal operator
runbooks, and confidential administrator guidance. Gaps inherit the
document's RESTRICTED classification, which is administrator-only. The
section metadata comes from server-side ingestion configuration, not from the
document's classification labels.

Reindex verifies that stored chunk indexes exactly match the trusted policy
before changing any ACL payload.
UI uploads default to RESTRICTED, and classification/roles cannot be supplied
by the UI. The document content hash is its stable ID/version; an unchanged
upload already present for the tenant is reused without embedding again.
Retrieval adds the active document ID to the existing
tenant/classification/role filter, so other documents cannot enter that
answer's context.

## Re-index Existing Documents

Existing Qdrant points without `classification`, `allowed_roles`, and `tenant`
are deliberately invisible to every role. Re-run the trusted batch ingestion
script to upsert local DATA files with ACL metadata:

```powershell
.\.venv\Scripts\python.exe scripts\ingest_documents.py
```

The UI's **Remove Document** action deletes all vectors for that document from
Qdrant and clears the current session's active-document state. Full deletion
requires the `administrator` role. Re-indexing a document-level policy for a
mixed-section document is rejected; run the trusted batch ingestion script
instead to preserve its section ACLs. The user must explicitly upload a document
again to make it active after an application restart.

## Local Demo

Set `DEMO_AUTH_ENABLED=true` in the local `.env`, restart FastAPI, and select a
demo principal in the Streamlit sidebar:

- Alice (`viewer`)
- Bob (`operator`)
- Carol (`administrator`)

Run `scripts/ingest_documents.py` before the demo. The UI defaults to the
trusted `DATA/true_data/NimbusPay_Enterprise_Policy_Handbook_Demo.docx`
handbook, which reuses the trusted, section-labeled chunks. Asking the same
operational question as Alice and Bob demonstrates that the viewer receives no
operator runbook context while the operator can.

Run the isolated ACL tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The tests use an in-memory Qdrant collection and verify the three-role policy,
the mixed-classification garden document, direct restricted queries,
semantically rephrased queries, tenant separation, missing-ACL denial, and that
no-context technical queries do not invoke the responder.