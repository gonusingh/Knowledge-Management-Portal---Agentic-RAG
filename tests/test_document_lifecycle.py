import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException
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

    def test_public_role_selector_keeps_api_behind_shared_secret(self) -> None:
        with patch.object(main_module.settings, "demo_auth_enabled", False), patch.object(
            main_module.settings,
            "public_role_selector",
            True,
        ), patch.object(
            main_module.settings,
            "backend_shared_secret",
            "test-backend-secret",
        ):
            administrator = main_module.resolve_demo_user(
                "carol",
                "Bearer test-backend-secret",
            )
            self.assertEqual(administrator, DEMO_USERS["carol"])

            with self.assertRaises(HTTPException) as error:
                main_module.resolve_demo_user("carol", "Bearer incorrect")
            self.assertEqual(getattr(error.exception, "status_code", None), 401)

    def test_default_handbook_is_reingested_only_when_missing(self) -> None:
        content = b"Bundled handbook for lifecycle test."
        document_id = hashlib.sha256(content).hexdigest()
        with TemporaryDirectory() as directory:
            handbook_path = Path(directory) / main_module.DEFAULT_DEMO_DOCUMENT
            handbook_path.write_bytes(content)
            with patch.object(
                main_module,
                "DEFAULT_DEMO_DOCUMENT_PATH",
                handbook_path,
            ), patch(
                "app.main.get_document_metadata",
                side_effect=[
                    None,
                    {"document_id": document_id},
                ],
            ), patch(
                "app.main.load_document",
                return_value={"text": "handbook"},
            ), patch(
                "app.main.chunk_document",
                return_value=[{"text": "chunk"}],
            ), patch("app.main.upsert_chunks") as upsert:
                self.assertTrue(main_module.ensure_default_demo_document())
                self.assertFalse(main_module.ensure_default_demo_document())

        upsert.assert_called_once_with([{"text": "chunk"}])


if __name__ == "__main__":
    unittest.main()
