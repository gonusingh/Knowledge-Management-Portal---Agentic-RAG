"""
tests/test_handbook_policy.py

Pins the trusted ACL policy for the NimbusPay demo handbook and proves, with
the REAL loader and chunker, that no sensitive text lands in a lower-access
chunk.

Why the end-to-end test matters:
The DOCX loader preserves paragraph boundaries using blank lines, while the
chunker splits on those blank-line boundaries. This test ensures the trusted
section ranges remain aligned with the actual paragraph structure of the
handbook and that sensitive content cannot inherit a lower-access ACL.
"""

import hashlib
import unittest
from pathlib import Path

from app.ingestion.chunking import (
    _split_into_paragraphs,
    chunk_document,
)
from app.ingestion.loaders import load_document
from app.ingestion.trusted_policy import (
    classification_for_document,
    trusted_sections_for_document,
)
from app.security.access_control import (
    Classification,
    build_document_access_metadata,
)


HANDBOOK_ID = "7e732071f97d97a3eb705124169ab0e037cf0c20e20f500c015938cac0182f33"

HANDBOOK_FILE = (
    Path(__file__).parents[1]
    / "DATA"
    / "true_data"
    / "NimbusPay_Enterprise_Policy_Handbook_Demo.docx"
)


EXPECTED_ROLES = {
    Classification.PUBLIC: {
        "viewer",
        "operator",
        "administrator",
    },
    Classification.INTERNAL: {
        "operator",
        "administrator",
    },
    Classification.CONFIDENTIAL: {
        "administrator",
    },
}


CONFIDENTIAL_MARKERS = [
    "Zebra-Quartz",
    "Meera Iyer",
    "nimbus-demo-token",
    "4.2 million",
    "penetration test",
    "bg-emergency-01",
]


INTERNAL_MARKERS = [
    "Operators may acknowledge",
    "Tuesday and Thursday",
]


def label(chunk: dict) -> str:
    """Return the classification value as a string."""
    value = chunk["classification"]
    return getattr(value, "value", value)


class HandbookTrustedPolicyTests(unittest.TestCase):
    """Tests for the server-controlled trusted handbook ACL policy."""

    def setUp(self):
        self.sections = trusted_sections_for_document(HANDBOOK_ID)

    def test_three_sections_in_public_internal_confidential_order(self):
        self.assertEqual(
            [section["classification"] for section in self.sections],
            [
                Classification.PUBLIC,
                Classification.INTERNAL,
                Classification.CONFIDENTIAL,
            ],
        )

    def test_ranges_are_contiguous_and_cover_the_whole_document(self):
        ranges = [
            (
                section["start_paragraph"],
                section["end_paragraph"],
            )
            for section in self.sections
        ]

        self.assertEqual(
            ranges,
            [
                (0, 16),
                (16, 29),
                (29, 42),
            ],
        )

        for (_, end), (start, _) in zip(
            ranges,
            ranges[1:],
        ):
            self.assertEqual(end, start)

    def test_allowed_roles_match_each_classification(self):
        for section in self.sections:
            with self.subTest(
                classification=section["classification"].value
            ):
                self.assertEqual(
                    set(section["allowed_roles"]),
                    EXPECTED_ROLES[section["classification"]],
                )

    def test_viewer_is_only_allowed_in_the_public_section(self):
        for section in self.sections:
            if section["classification"] is not Classification.PUBLIC:
                self.assertNotIn(
                    "viewer",
                    section["allowed_roles"],
                )

    def test_document_level_classification_stays_restrictive(self):
        self.assertEqual(
            classification_for_document(HANDBOOK_ID),
            Classification.RESTRICTED,
        )


@unittest.skipUnless(
    HANDBOOK_FILE.exists(),
    "Handbook file is not in DATA/true_data on this machine.",
)
class HandbookIngestionBoundaryTests(unittest.TestCase):
    """
    Uses the real DOCX file, real loader, and real chunker.

    This proves that the trusted paragraph ranges map correctly onto the
    actual chunks produced by the ingestion pipeline.
    """

    @classmethod
    def setUpClass(cls):
        access = build_document_access_metadata(
            document_id=HANDBOOK_ID,
            classification=Classification.RESTRICTED,
            tenant="nimbuspay",
            version=HANDBOOK_ID,
        )

        cls.document = load_document(
            str(HANDBOOK_FILE),
            doc_type="true_data",
            access_metadata=access,
        )

        cls.document["trusted_sections"] = (
            trusted_sections_for_document(HANDBOOK_ID)
        )

        cls.chunks = chunk_document(cls.document)

    def test_policy_key_is_the_hash_of_the_file_on_disk(self):
        with HANDBOOK_FILE.open("rb") as file_handle:
            digest = hashlib.file_digest(
                file_handle,
                "sha256",
            ).hexdigest()

        self.assertEqual(
            digest,
            HANDBOOK_ID,
        )

    def test_chunker_sees_the_expected_number_of_paragraphs(self):
        """
        Guards against paragraph-counting drift between the loader and the
        trusted section ranges:

            PUBLIC       0-15
            INTERNAL    16-28
            CONFIDENTIAL 29-41
        """
        self.assertEqual(
            len(
                _split_into_paragraphs(
                    self.document["text"]
                )
            ),
            42,
        )

    def test_all_three_classifications_are_present(self):
        self.assertEqual(
            {label(chunk) for chunk in self.chunks},
            {
                "PUBLIC",
                "INTERNAL",
                "CONFIDENTIAL",
            },
        )

    def test_confidential_text_only_appears_in_confidential_chunks(self):
        for marker in CONFIDENTIAL_MARKERS:
            with self.subTest(marker=marker):
                holders = [
                    chunk
                    for chunk in self.chunks
                    if marker in chunk["text"]
                ]

                self.assertTrue(
                    holders,
                    f"{marker!r} not found in any chunk",
                )

                self.assertEqual(
                    {label(chunk) for chunk in holders},
                    {"CONFIDENTIAL"},
                )

    def test_internal_text_only_appears_in_internal_chunks(self):
        for marker in INTERNAL_MARKERS:
            with self.subTest(marker=marker):
                holders = [
                    chunk
                    for chunk in self.chunks
                    if marker in chunk["text"]
                ]

                self.assertTrue(
                    holders,
                    f"{marker!r} not found in any chunk",
                )

                self.assertEqual(
                    {label(chunk) for chunk in holders},
                    {"INTERNAL"},
                )

    def test_public_chunks_contain_the_public_policy_and_nothing_sensitive(
        self,
    ):
        public_text = "\n".join(
            chunk["text"]
            for chunk in self.chunks
            if label(chunk) == "PUBLIC"
        )

        self.assertIn(
            "corporate devices",
            public_text,
        )

        for marker in (
            CONFIDENTIAL_MARKERS + INTERNAL_MARKERS
        ):
            with self.subTest(marker=marker):
                self.assertNotIn(
                    marker,
                    public_text,
                )

    def test_chunk_roles_match_their_classification(self):
        for chunk in self.chunks:
            roles = set(chunk["allowed_roles"])

            with self.subTest(
                chunk=chunk["chunk_index"],
                classification=label(chunk),
            ):
                if label(chunk) == "PUBLIC":
                    self.assertIn(
                        "viewer",
                        roles,
                    )
                else:
                    self.assertNotIn(
                        "viewer",
                        roles,
                    )

                if label(chunk) == "CONFIDENTIAL":
                    self.assertEqual(
                        roles,
                        {"administrator"},
                    )


if __name__ == "__main__":
    unittest.main()
