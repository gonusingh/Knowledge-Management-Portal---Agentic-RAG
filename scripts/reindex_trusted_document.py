"""Reapply server-controlled ACL metadata to a document already stored in Qdrant.

Usage:
    python scripts/reindex_trusted_document.py <64-character-document-id>
"""

import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.policy_reindex import reindex_document_policy
from app.ingestion.trusted_policy import (
    TRUSTED_DOCUMENT_CLASSIFICATIONS,
    classification_for_document,
    trusted_sections_for_document,
)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: reindex_trusted_document.py <document-id>")

    document_id = sys.argv[1]
    if not re.fullmatch(r"[0-9a-f]{64}", document_id):
        raise SystemExit("Document ID must be a 64-character lowercase SHA-256 hash.")
    if document_id not in TRUSTED_DOCUMENT_CLASSIFICATIONS:
        raise SystemExit(f"No explicit trusted policy is configured for {document_id}.")
    if trusted_sections_for_document(document_id):
        raise SystemExit(
            "This document has section-level ACLs; re-run "
            "scripts/ingest_documents.py to preserve them."
        )

    updated = reindex_document_policy(document_id)
    classification = classification_for_document(document_id)
    print(
        f"Updated ACL metadata for {updated} chunks: "
        f"document={document_id}, classification={classification.value}"
    )


if __name__ == "__main__":
    main()
