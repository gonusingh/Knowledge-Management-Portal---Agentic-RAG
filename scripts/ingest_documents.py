"""
scripts/ingest_documents.py — one-time (or re-run-as-needed) script that
actually populates Qdrant. Run manually, separately from the live API:

    python scripts/ingest_documents.py

Walks DATA/true_data/ and DATA/noisy_data/, and for every file: loads it,
chunks it, and uploads it to Qdrant — tagging each chunk with the correct
"type" metadata based on which folder it came from.
"""

import hashlib
import sys
from pathlib import Path

# This lets the script import from app/ even though it lives in scripts/,
# not inside the app package itself.
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestion.loaders import load_document
from app.ingestion.chunking import chunk_document
from app.services.retrieval.vector_search import (
    delete_document,
    list_documents,
    upsert_chunks,
)
from app.ingestion.trusted_policy import (
    TRUSTED_DOCUMENT_CLASSIFICATIONS,
    trusted_sections_for_document,
)
from app.security.access_control import (
    Classification,
    build_document_access_metadata,
)

DATA_DIR = Path(__file__).parent.parent / "DATA"
DEFAULT_DOCUMENT_NAME = "NimbusPay_Enterprise_Policy_Handbook_Demo.docx"

# This demo is intentionally centered on the trusted Nimbus handbook.
# All other sample documents are removed from the default ingestion path so the
# app starts with a single, role-aware document instead of a noisy multi-doc demo.
FOLDER_TO_TYPE = {
    "true_data": "true_data",
}

# This policy is assigned by the trusted batch-ingestion code, not inferred
# from document text. Review it before placing new sensitive files in DATA/.
FOLDER_TO_CLASSIFICATION = {
    "true_data": Classification.PUBLIC,
}


def ingest_folder(folder_path: Path, doc_type: str) -> int:
    """Ingest only the trusted Nimbus handbook for the default demo setup."""
    total_chunks = 0

    if not folder_path.exists():
        print(f"  (skipping — folder doesn't exist: {folder_path})")
        return 0

    target_file = folder_path / DEFAULT_DOCUMENT_NAME
    if not target_file.exists():
        print(f"  (default handbook not found: {target_file})")
        return 0

    try:
        with target_file.open("rb") as file_handle:
            document_digest = hashlib.file_digest(file_handle, "sha256").hexdigest()

        access_metadata = build_document_access_metadata(
            document_id=document_digest,
            classification=TRUSTED_DOCUMENT_CLASSIFICATIONS.get(
                document_digest,
                FOLDER_TO_CLASSIFICATION[folder_path.name],
            ),
            tenant="nimbuspay",
            version=document_digest,
        )
        document = load_document(
            str(target_file),
            doc_type=doc_type,
            access_metadata=access_metadata,
        )
        trusted_sections = trusted_sections_for_document(document_digest)
        if trusted_sections:
            document["trusted_sections"] = trusted_sections
        chunks = chunk_document(document)
        upsert_chunks(chunks)
        total_chunks += len(chunks)
        print(f"    OK: {target_file.name} -> {len(chunks)} chunks")
    except ValueError as e:
        print(f"    SKIP: {target_file.name}: {e}")

    return total_chunks


def main():
    print("Starting default demo ingestion...\n")

    existing_documents = list_documents(tenant="nimbuspay")
    for document in existing_documents:
        document_id = document.get("document_id")
        source_file = document.get("source_file")
        if document_id and source_file != DEFAULT_DOCUMENT_NAME:
            print(f"Removing extra demo document: {source_file or document_id}")
            delete_document(document_id, tenant="nimbuspay")

    grand_total = 0
    for folder_name, doc_type in FOLDER_TO_TYPE.items():
        folder_path = DATA_DIR / folder_name
        print(f"Processing {folder_name}/ (tagged as type={doc_type}):")
        grand_total += ingest_folder(folder_path, doc_type)
        print()

    print(f"Done. {grand_total} total chunks uploaded to Qdrant.")


if __name__ == "__main__":
    main()