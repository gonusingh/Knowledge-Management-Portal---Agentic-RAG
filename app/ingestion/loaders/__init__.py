"""
app/ingestion/loaders/__init__.py — local document parsers.

Exposes ONE function, load_document(), that the rest of the app calls.
Everything downstream (chunking, embedding) only ever deals with plain
text + a small metadata dict — it never needs to know what file format
the original document was.
"""

import hashlib
import os
from pathlib import Path

# Each of these is a separate library, one per file format, already in
# requirements.txt. We import all of them here since this is the ONE
# place in the app that needs to know about file-format-specific parsing.
import pypdf
import pdfplumber
from docx import Document as DocxDocument
from pptx import Presentation
from bs4 import BeautifulSoup

from app.security.access_control import (
    Classification,
    DocumentAccessMetadata,
    build_document_access_metadata,
)


def _load_pdf(file_path: str) -> str:
    """
    Try the fast parser (pypdf) first. If it returns very little text
    (a strong signal the PDF is scanned/complex and pypdf choked on it),
    fall back to the slower but more robust pdfplumber.
    """
    text_parts = []

    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)

        for page in reader.pages:
            text_parts.append(page.extract_text() or "")

    text = "\n".join(text_parts).strip()

    # Heuristic: fewer than 50 characters total across the whole PDF is
    # almost certainly a failed extraction.
    if len(text) < 50:
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")

        text = "\n".join(text_parts).strip()

    return text


def _load_docx(file_path: str) -> str:
    """
    Extract DOCX paragraphs while preserving paragraph boundaries.

    The chunker uses blank lines (\\n\\n) to identify paragraph
    boundaries, so DOCX paragraphs must be joined using two newlines.
    """
    doc = DocxDocument(file_path)

    return "\n\n".join(
        p.text
        for p in doc.paragraphs
    )


def _load_pptx(file_path: str) -> str:
    prs = Presentation(file_path)
    text_parts = []

    # A .pptx file is a list of slides; each slide is a list of shapes.
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text:
                text_parts.append(shape.text)

    return "\n".join(text_parts)


def _load_html(file_path: str) -> str:
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Keep some structure instead of mashing everything together.
    return soup.get_text(separator="\n").strip()


def _load_txt(file_path: str) -> str:
    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as f:
        return f.read()


# This dict maps file extensions to the function that handles them.
_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".pptx": _load_pptx,
    ".html": _load_html,
    ".htm": _load_html,
    ".txt": _load_txt,
}


def load_document(
    file_path: str,
    doc_type: str = "true_data",
    access_metadata: DocumentAccessMetadata | None = None,
) -> dict:
    """
    The single entry point the rest of the app uses.

    Args:
        file_path: path to the file on disk
        doc_type: "true_data" or "noisy_data"

    Returns:
        {
            "text": "...",
            "source_file": "cronjobs.docx",
            "type": "true_data",
            "document_id": "...",
            "classification": "RESTRICTED",
            "allowed_roles": ["administrator"],
            "tenant": "nimbuspay",
        }
    """
    extension = Path(file_path).suffix.lower()

    if extension not in _LOADERS:
        raise ValueError(
            f"Unsupported file type '{extension}'. "
            f"Supported types: {list(_LOADERS.keys())}"
        )

    loader_function = _LOADERS[extension]
    extracted_text = loader_function(file_path)

    source_file = os.path.basename(file_path)

    with open("{}".format(file_path), "rb") as file_handle:
        file_digest = hashlib.file_digest(
            file_handle,
            "sha256",
        ).hexdigest()

    trusted_access = access_metadata or build_document_access_metadata(
        document_id=file_digest,
        classification=Classification.RESTRICTED,
        version=file_digest,
    )

    return {
        "text": extracted_text,
        "source_file": source_file,
        "type": doc_type,
        **trusted_access.model_dump(mode="json"),
    }
