"""
app/ingestion/chunking/__init__.py

Paragraph-based text splitter with trusted section-level ACL support.

The default behavior remains document-level ACL inheritance.

For mixed-classification documents, the loader/ingestion policy may provide
trusted section metadata through:

document["trusted_sections"] = [
    {
        "start_paragraph": 0,
        "end_paragraph": 12,
        "classification": Classification.INTERNAL,
        "allowed_roles": ["administrator", "operator"],
    },
    ...
]

IMPORTANT:
- Section ACL metadata must come from trusted ingestion configuration.
- Never infer classification or allowed_roles from document text.
- A chunk never crosses a trusted section boundary.
"""

MAX_CHUNK_CHARS = 1500


def _split_into_paragraphs(text: str) -> list[str]:
    """
    Split text on blank lines and remove empty paragraphs.

    DOCX ingestion preserves paragraph boundaries using \\n\\n,
    so each resulting item corresponds to one source paragraph.
    """
    raw_paragraphs = text.split("\n\n")

    return [
        paragraph.strip()
        for paragraph in raw_paragraphs
        if paragraph.strip()
    ]


def _section_for_paragraph(
    paragraph_index: int,
    trusted_sections: list[dict],
) -> dict | None:
    """
    Return the trusted ACL section containing the paragraph.

    Section ranges use:
        start_paragraph = inclusive
        end_paragraph   = exclusive

    Example:
        start=0, end=5
        covers paragraphs 0,1,2,3,4
    """
    for section in trusted_sections:
        start = int(section["start_paragraph"])
        end = int(section["end_paragraph"])

        if start <= paragraph_index < end:
            return section

    return None


def _build_chunk(
    chunk_text: str,
    document: dict,
    chunk_index: int,
    acl_metadata: dict | None,
) -> dict:
    """
    Build one chunk and attach trusted ACL metadata.

    If no section-level ACL metadata exists, fall back to the
    document-level metadata already produced by the trusted loader.
    """

    if acl_metadata is None:
        classification = document["classification"]
        allowed_roles = document["allowed_roles"]
    else:
        classification = acl_metadata["classification"]
        allowed_roles = acl_metadata["allowed_roles"]

    return {
        "chunk_id": f"{document['document_id']}_{chunk_index}",
        "text": chunk_text,
        "source_file": document["source_file"],
        "type": document["type"],
        "chunk_index": chunk_index,
        "document_id": document["document_id"],
        "classification": classification,
        "allowed_roles": allowed_roles,
        "tenant": document["tenant"],
        "service": document["service"],
        "effective_date": document["effective_date"],
        "version": document["version"],
    }


def chunk_document(document: dict) -> list[dict]:
    """
    Split a loaded document into chunks.

    Normal documents:
        Every chunk inherits the document-level ACL.

    Mixed-classification documents:
        The document may contain trusted section-level ACL metadata in
        document["trusted_sections"].

        Chunks never cross section boundaries, preventing an INTERNAL
        chunk from accidentally containing RESTRICTED content.

    trusted_sections must be supplied by trusted ingestion configuration.
    It must NEVER be derived from the document's own text.
    """

    paragraphs = _split_into_paragraphs(document["text"])

    if not paragraphs:
        return []

    trusted_sections = document.get("trusted_sections", [])

    chunks: list[tuple[str, dict | None]] = []

    current_chunk_text = ""
    current_acl: dict | None = None

    def flush_current_chunk() -> None:
        nonlocal current_chunk_text, current_acl

        if current_chunk_text:
            chunks.append(
                (
                    current_chunk_text,
                    current_acl,
                )
            )

            current_chunk_text = ""
            current_acl = None

    for paragraph_index, paragraph in enumerate(paragraphs):

        # Determine trusted ACL metadata for this paragraph.
        paragraph_acl = _section_for_paragraph(
            paragraph_index,
            trusted_sections,
        )

        # If section-level ACL exists, ensure chunks never cross
        # a section boundary.
        if current_chunk_text and paragraph_acl != current_acl:
            flush_current_chunk()

        # Start the current chunk with this paragraph's ACL.
        if not current_chunk_text:
            current_acl = paragraph_acl

        # A single paragraph larger than MAX_CHUNK_CHARS is intentionally
        # preserved as one chunk. We do not split technical content
        # arbitrarily in the middle of a paragraph.
        if len(paragraph) > MAX_CHUNK_CHARS:
            flush_current_chunk()

            chunks.append(
                (
                    paragraph,
                    paragraph_acl,
                )
            )

            continue

        # Would adding this paragraph exceed the limit?
        additional_length = (
            len(paragraph)
            if not current_chunk_text
            else len(paragraph) + 2
        )

        if (
            current_chunk_text
            and len(current_chunk_text) + additional_length
            > MAX_CHUNK_CHARS
        ):
            flush_current_chunk()
            current_acl = paragraph_acl

        if current_chunk_text:
            current_chunk_text += "\n\n" + paragraph
        else:
            current_chunk_text = paragraph

    flush_current_chunk()

    result: list[dict] = []

    for index, (chunk_text, acl_metadata) in enumerate(chunks):
        result.append(
            _build_chunk(
                chunk_text=chunk_text,
                document=document,
                chunk_index=index,
                acl_metadata=acl_metadata,
            )
        )

    return result
