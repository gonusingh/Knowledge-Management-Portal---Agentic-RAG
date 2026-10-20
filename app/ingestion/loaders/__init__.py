"""
app/ingestion/loaders/__init__.py — local document parsers.

Exposes ONE function, load_document(), that the rest of the app calls.
Everything downstream (chunking, embedding) only ever deals with plain
text + a small metadata dict — it never needs to know what file format
the original document was.
"""

import hashlib
import os
import zipfile
from pathlib import Path

# Each of these is a separate library, one per file format, already in
# requirements.txt. We import all of them here since this is the ONE
# place in the app that needs to know about file-format-specific parsing.
import pypdf
import pdfplumber
from pdfplumber.utils.exceptions import (
    MalformedPDFException,
    PdfminerException,
)
from pdfminer.pdfdocument import PDFEncryptionError
from pdfminer.pdfparser import PDFException, PDFSyntaxError
from docx import Document as DocxDocument
from docx.opc.exceptions import PackageNotFoundError as DocxPackageNotFoundError
from docx.table import Table
from docx.text.paragraph import Paragraph
from pptx import Presentation
from pptx.exc import PackageNotFoundError as PptxPackageNotFoundError
from bs4 import BeautifulSoup

from app.security.access_control import (
    Classification,
    DocumentAccessMetadata,
    build_document_access_metadata,
)


def _load_pdf(file_path: str) -> str:
    """Compare both supported parsers and retain the more complete text."""
    extracted_texts = []
    parser_errors = []

    try:
        with open(file_path, "rb") as file_handle:
            reader = pypdf.PdfReader(file_handle)
            extracted_texts.append(
                "\n\n".join(
                    page.extract_text() or "" for page in reader.pages
                ).strip()
            )
    except (OSError, ValueError, pypdf.errors.PdfReadError) as exc:
        parser_errors.append(exc)

    try:
        with pdfplumber.open(file_path) as pdf:
            extracted_texts.append(
                "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                ).strip()
            )
    except (
        OSError,
        ValueError,
        PDFException,
        PDFSyntaxError,
        PDFEncryptionError,
        MalformedPDFException,
        PdfminerException,
    ) as exc:
        parser_errors.append(exc)

    if not extracted_texts:
        error = ValueError(
            "The PDF could not be parsed by either supported parser."
        )
        if parser_errors:
            raise error from parser_errors[-1]
        raise error

    return max(
        extracted_texts,
        key=lambda candidate: (
            sum(character.isalnum() for character in candidate),
            len(candidate),
        ),
    )


def _load_docx(file_path: str) -> str:
    """
    Extract DOCX paragraphs and table rows while preserving paragraph counts.

    The chunker uses blank lines (\\n\\n) to identify paragraph
    boundaries, so DOCX paragraphs must be joined using two newlines. Table
    rows stay inside the adjacent paragraph to keep trusted section indexes
    aligned with the source document's paragraph sequence.
    """
    try:
        doc = DocxDocument(file_path)
    except (OSError, zipfile.BadZipFile, DocxPackageNotFoundError) as exc:
        raise ValueError(f"Could not parse DOCX document: {exc}") from exc
    body_parts = []
    pending_table_rows = []

    for child in doc.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, doc)
            if paragraph.text.strip():
                style_name = paragraph.style.name if paragraph.style else ""
                paragraph_text = paragraph.text.strip()
                if style_name.startswith("Heading"):
                    try:
                        heading_level = int(style_name.rsplit(" ", 1)[-1])
                    except ValueError:
                        heading_level = 2
                    paragraph_text = f"{'#' * max(1, min(heading_level, 6))} {paragraph_text}"
                elif style_name.startswith("List"):
                    paragraph_text = f"- {paragraph_text}"
                if pending_table_rows:
                    paragraph_text = "\n".join(
                        [*pending_table_rows, paragraph_text]
                    )
                    pending_table_rows.clear()
                body_parts.append(paragraph_text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, doc)
            for row in table.rows:
                cells = [" ".join(cell.text.split()) for cell in row.cells]
                if any(cells):
                    row_text = " | ".join(cells)
                    if body_parts:
                        body_parts[-1] += "\n" + row_text
                    else:
                        pending_table_rows.append(row_text)

    if pending_table_rows:
        if body_parts:
            body_parts[-1] += "\n" + "\n".join(pending_table_rows)
        else:
            body_parts.append("\n".join(pending_table_rows))
    return "\n\n".join(body_parts)


def _load_pptx(file_path: str) -> str:
    try:
        prs = Presentation(file_path)
    except (OSError, zipfile.BadZipFile, PptxPackageNotFoundError) as exc:
        raise ValueError(f"Could not parse PowerPoint document: {exc}") from exc
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
    if not extracted_text.strip():
        if extension == ".pdf":
            raise ValueError(
                "No selectable text could be extracted from this PDF. "
                "It may be scanned or image-only; OCR is not supported."
            )
        raise ValueError("No extractable text was found in the document.")

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
