import hashlib
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.main as main_module
from app.security.access_control import DEMO_USERS


class DocumentLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_overrides = dict(main_module.app.dependency_overrides)
        main_module.app.dependency_overrides[main_module.resolve_demo_user] = (
            lambda: DEMO_USERS["carol"]
        )
        self.client = TestClient(main_module.app)

    def tearDown(self) -> None:
        main_module.app.dependency_overrides = self.original_overrides

    def test_unchanged_upload_reuses_existing_content_hash_without_upserting(self) -> None:
        content = b"Active document lifecycle test."
        with patch(
            "app.main.get_document_metadata",
            side_effect=[
                None,
                {
                    "source_file": "notes.txt",
                    "version": hashlib.sha256(content).hexdigest(),
                    "classification": "RESTRICTED",
                },
            ],
        ), patch("app.main.upsert_chunks") as upsert:
            first = self.client.post(
                "/upload",
                files={"file": ("notes.txt", content, "text/plain")},
                data={"classification": "PUBLIC", "allowed_roles": "viewer"},
                headers={"X-Demo-User": "carol"},
            )
            second = self.client.post(
                "/upload",
                files={"file": ("renamed.txt", content, "text/plain")},
                headers={"X-Demo-User": "carol"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["document_id"], second.json()["document_id"])
        self.assertEqual(first.json()["version"], second.json()["version"])
        self.assertFalse(first.json()["reused"])
        self.assertTrue(second.json()["reused"])
        self.assertEqual(second.json()["chunks_added"], 0)
        upsert.assert_called_once()
        indexed_chunk = upsert.call_args.args[0][0]
        expected_hash = hashlib.sha256(content).hexdigest()
        self.assertEqual(first.json()["document_id"], expected_hash)
        self.assertEqual(indexed_chunk["version"], expected_hash)
        self.assertEqual(indexed_chunk["classification"], "RESTRICTED")
        self.assertEqual(indexed_chunk["allowed_roles"], ["administrator"])
        self.assertEqual(indexed_chunk["source_file"], "notes.txt")

    def test_remove_requires_admin_and_deletes_document_vectors(self) -> None:
        document_id = "b" * 64
        with patch("app.main.document_exists", return_value=True), patch(
            "app.main.delete_document"
        ) as delete:
            main_module.app.dependency_overrides[main_module.resolve_demo_user] = (
                lambda: DEMO_USERS["bob"]
            )
            forbidden = self.client.delete(
                f"/documents/{document_id}",
                headers={"X-Demo-User": "bob"},
            )

            main_module.app.dependency_overrides[main_module.resolve_demo_user] = (
                lambda: DEMO_USERS["carol"]
            )
            removed = self.client.delete(
                f"/documents/{document_id}",
                headers={"X-Demo-User": "carol"},
            )

        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(removed.status_code, 200)
        delete.assert_called_once_with(document_id, tenant="nimbuspay")

    def test_query_abstains_before_guardrails_or_graph_for_missing_or_invalid_document(self) -> None:
        with patch.object(main_module, "check_input") as check_input, patch.object(
            main_module, "compiled_graph"
        ) as graph:
            for body in (
                {"query": "Question without scope"},
                {"query": "Question with invalid scope", "active_document_id": "invalid"},
                {"query": "Question with non-string scope", "active_document_id": 42},
            ):
                response = self.client.post(
                    "/query",
                    json=body,
                    headers={"X-Demo-User": "carol"},
                )
                self.assertEqual(response.status_code, 200)
                self.assertIn("no valid active document", response.json()["answer"])
                self.assertIn("missing or invalid", response.json()["trace"][0])

        check_input.assert_not_called()
        graph.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
