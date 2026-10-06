"""Server-controlled document classification overrides keyed by content hash."""

from app.security.access_control import Classification

DEFAULT_CLASSIFICATION = Classification.RESTRICTED

# Trusted overrides are applied only to the exact content-addressed document
# IDs listed here. They are never derived from filenames or document text.
TRUSTED_DOCUMENT_CLASSIFICATIONS: dict[str, Classification] = {
    "587b89a44343e650a22df598bd567926d61df62a81db50e5cb3ebec09fe07489":
        Classification.PUBLIC,

    "db86ae480023f4e1bf114c3e57ed10fa968a4fb1571cbe7189109e152002fa80":
        Classification.RESTRICTED,

    "4d64def30b98587074f28da245e19811716ac542df9612740bc72de45a443010":
        Classification.RESTRICTED,

    # NimbusPay Enterprise Policy Handbook Demo.
    # Corrected handbook with explicit paragraph boundaries.
    "7e732071f97d97a3eb705124169ab0e037cf0c20e20f500c015938cac0182f33":
        Classification.RESTRICTED,
}


# Trusted per-chunk classifications for documents where classification
# cannot be inferred safely from the document-level classification.
TRUSTED_DOCUMENT_CHUNK_CLASSIFICATIONS: dict[
    str, dict[int, Classification]
] = {
    "4d64def30b98587074f28da245e19811716ac542df9612740bc72de45a443010": {
        0: Classification.PUBLIC,
        1: Classification.PUBLIC,
        2: Classification.CONFIDENTIAL,
        3: Classification.INTERNAL,
        4: Classification.CONFIDENTIAL,
        5: Classification.CONFIDENTIAL,
        6: Classification.CONFIDENTIAL,
        7: Classification.CONFIDENTIAL,
    },
}


# Trusted paragraph-level sections for documents whose content contains
# multiple classifications.
#
# NimbusPay Enterprise Policy Handbook Demo:
#   Paragraphs 0-15  -> PUBLIC
#   Paragraphs 16-28 -> INTERNAL
#   Paragraphs 29-41 -> CONFIDENTIAL
#
# Section ranges use [start_paragraph, end_paragraph), so the end paragraph
# is exclusive.
#
# Exactly three application roles are used:
#   Viewer
#   Operator
#   Administrator
TRUSTED_DOCUMENT_SECTIONS: dict[str, tuple[dict, ...]] = {
    "db86ae480023f4e1bf114c3e57ed10fa968a4fb1571cbe7189109e152002fa80": (
        {
            "start_paragraph": 0,
            "end_paragraph": 19,
            "classification": Classification.PUBLIC,
            "allowed_roles": [
                "administrator",
                "operator",
                "viewer",
            ],
        },
        {
            "start_paragraph": 19,
            "end_paragraph": 37,
            "classification": Classification.INTERNAL,
            "allowed_roles": [
                "administrator",
                "operator",
            ],
        },
        {
            "start_paragraph": 37,
            "end_paragraph": 59,
            "classification": Classification.CONFIDENTIAL,
            "allowed_roles": [
                "administrator",
            ],
        },
    ),

    # NimbusPay Enterprise Policy Handbook Demo.
    "7e732071f97d97a3eb705124169ab0e037cf0c20e20f500c015938cac0182f33": (
        {
            "start_paragraph": 0,
            "end_paragraph": 16,
            "classification": Classification.PUBLIC,
            "allowed_roles": [
                "administrator",
                "operator",
                "viewer",
            ],
        },
        {
            "start_paragraph": 16,
            "end_paragraph": 29,
            "classification": Classification.INTERNAL,
            "allowed_roles": [
                "administrator",
                "operator",
            ],
        },
        {
            "start_paragraph": 29,
            "end_paragraph": 42,
            "classification": Classification.CONFIDENTIAL,
            "allowed_roles": [
                "administrator",
            ],
        },
    ),
}


def classification_for_document(document_id: str) -> Classification:
    """Return an explicit server-side override or the fail-closed default."""
    return TRUSTED_DOCUMENT_CLASSIFICATIONS.get(
        document_id,
        DEFAULT_CLASSIFICATION,
    )


def trusted_sections_for_document(document_id: str) -> list[dict]:
    """Return copied, server-configured section ACLs for an exact document."""
    return [
        dict(section)
        for section in TRUSTED_DOCUMENT_SECTIONS.get(document_id, ())
    ]


def trusted_chunk_classifications_for_document(
    document_id: str,
) -> dict[int, Classification]:
    """Return the trusted ACL classification per stored chunk for a document."""
    return dict(
        TRUSTED_DOCUMENT_CHUNK_CLASSIFICATIONS.get(
            document_id,
            {},
        )
    )