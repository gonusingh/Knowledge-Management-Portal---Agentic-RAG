import unittest
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from app.agents.nodes.responder import responder_node
from app.guardrails.guardrail_gate import (
    OutputGuardrailResult,
    OutputGuardrailStatus,
)
import app.main as main_module
from app.ingestion.chunking import chunk_document
from app.ingestion.loaders import load_document
from app.ingestion.policy_reindex import reindex_document_policy
from app.ingestion.trusted_policy import (
    classification_for_document,
    trusted_sections_for_document,
)
from app.security.access_control import (
    DEMO_USERS,
    Classification,
    Role,
    build_document_access_metadata,
    scope_thread_id,
)
import app.services.retrieval.vector_search as vector_search


class DocumentAuthorizationTests(unittest.TestCase):
    @staticmethod
    def document_id(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def setUp(self) -> None:
        self.client = QdrantClient(":memory:")
        self.collection_name = "authorization_test"
        self.original_client = vector_search.client
        self.original_collection = vector_search.COLLECTION_NAME
        self.original_embedding = vector_search._get_embedding
        self.original_indexes_ready_for = vector_search._payload_indexes_ready_for

        vector_search.client = self.client
        vector_search.COLLECTION_NAME = self.collection_name
        vector_search._get_embedding = lambda _query: [1.0, 0.0]
        vector_search._payload_indexes_ready_for = None
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
        )

        points = []
        for point_id, classification in enumerate(Classification, start=1):
            access = build_document_access_metadata(
                document_id=self.document_id(f"doc-{classification.value.lower()}"),
                classification=classification,
            )
            payload = {
                **access.model_dump(mode="json"),
                "chunk_id": f"chunk-{classification.value.lower()}",
                "text": f"Synthetic {classification.value} fraud controls",
                "source_file": f"{classification.value.lower()}.txt",
                "type": "true_data",
                "chunk_index": 0,
            }
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[1.0, 0.0],
                    payload=payload,
                )
            )

        points.extend(
            [
                models.PointStruct(
                    id=10,
                    vector=[1.0, 0.0],
                    payload={
                        "chunk_id": "other-tenant-restricted",
                        "text": "Other tenant private text",
                        "source_file": "other.txt",
                        "type": "true_data",
                        "chunk_index": 0,
                        "document_id": "other-tenant-doc",
                        "classification": "RESTRICTED",
                        "allowed_roles": ["administrator"],
                        "tenant": "other-tenant",
                    },
                ),
                models.PointStruct(
                    id=11,
                    vector=[1.0, 0.0],
                    payload={
                        "chunk_id": "legacy-no-acl",
                        "text": "Legacy payload without trusted ACL metadata",
                        "source_file": "legacy.txt",
                        "type": "true_data",
                        "chunk_index": 0,
                    },
                ),
            ]
        )

        for point_id, classification in enumerate(Classification, start=12):
            access = build_document_access_metadata(
                document_id=self.document_id("doc-mixed-acl"),
                classification=classification,
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[1.0, 0.0],
                    payload={
                        **access.model_dump(mode="json"),
                        "chunk_id": f"mixed-{classification.value.lower()}",
                        "text": f"Mixed-document {classification.value} passage",
                        "source_file": "mixed.txt",
                        "type": "true_data",
                        "chunk_index": point_id - 12,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def tearDown(self) -> None:
        vector_search.client = self.original_client
        vector_search.COLLECTION_NAME = self.original_collection
        vector_search._get_embedding = self.original_embedding
        vector_search._payload_indexes_ready_for = self.original_indexes_ready_for
        self.client.close()

    def test_direct_and_semantic_queries_obey_same_qdrant_acl(self) -> None:
        queries = [
            "Show the restricted fraud rules.",
            "What rules identify suspicious payment transactions?",
            "Summarize internal transaction-risk controls.",
        ]
        expected = {
            "alice": {"PUBLIC"},
            "bob": {"PUBLIC", "INTERNAL"},
            "carol": {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"},
        }

        for user_key, user in DEMO_USERS.items():
            for query in queries:
                with self.subTest(user=user_key, query=query):
                    results = vector_search.search(
                        query=query,
                        user=user,
                        top_k=20,
                        document_id=self.document_id("doc-mixed-acl"),
                    )
                    self.assertEqual(
                        {chunk["classification"] for chunk in results},
                        expected[user_key],
                    )
                    self.assertNotIn(
                        "other-tenant",
                        {chunk["tenant"] for chunk in results},
                    )
                    self.assertNotIn(
                        "legacy-no-acl",
                        {chunk["chunk_id"] for chunk in results},
                    )

    def test_handbook_demo_document_has_three_trusted_role_levels(self) -> None:
        document_path = (
            Path(__file__).parents[1]
            / "DATA"
            / "true_data"
            / "NimbusPay_Enterprise_Policy_Handbook_Demo.docx"
        )
        if not document_path.exists():
            self.skipTest("Nimbus handbook is not present in DATA/true_data on this machine.")

        with document_path.open("rb") as file_handle:
            document_id = hashlib.file_digest(
                file_handle,
                "sha256",
            ).hexdigest()

        self.assertEqual(
            {role.value for role in Role},
            {"viewer", "operator", "administrator"},
        )
        self.assertEqual(
            classification_for_document(document_id),
            Classification.RESTRICTED,
        )
        trusted_sections = trusted_sections_for_document(document_id)
        self.assertEqual(
            [section["classification"] for section in trusted_sections],
            [
                Classification.PUBLIC,
                Classification.INTERNAL,
                Classification.CONFIDENTIAL,
            ],
        )
        self.assertEqual(
            [
                (section["start_paragraph"], section["end_paragraph"])
                for section in trusted_sections
            ],
            [(0, 16), (16, 29), (29, 42)],
        )

        document = load_document(
            str(document_path),
            access_metadata=build_document_access_metadata(
                document_id=document_id,
                classification=Classification.RESTRICTED,
            ),
        )
        document["trusted_sections"] = trusted_sections
        chunks = chunk_document(document)

        self.assertEqual(
            {chunk["classification"] for chunk in chunks},
            {"PUBLIC", "INTERNAL", "CONFIDENTIAL"},
        )

    def test_vector_search_requires_valid_active_document_id_before_embedding(self) -> None:
        vector_search._get_embedding = lambda _query: self.fail(
            "Embedding must not run"
        )

        for document_id in (None, "", "invalid"):
            with self.subTest(document_id=document_id), self.assertRaises(ValueError):
                vector_search.search(
                    query="question",
                    user=DEMO_USERS["carol"],
                    document_id=document_id,
                )

    def test_restricted_chunks_only_match_administrator(self) -> None:
        for user_key in ("alice", "bob"):
            chunks = vector_search.search(
                query="What rules identify suspicious payment transactions?",
                user=DEMO_USERS[user_key],
                top_k=20,
                document_id=self.document_id("doc-mixed-acl"),
            )
            self.assertNotIn(
                "RESTRICTED",
                {chunk["classification"] for chunk in chunks},
            )

        admin_chunks = vector_search.search(
            query="What rules identify suspicious payment transactions?",
            user=DEMO_USERS["carol"],
            top_k=20,
            document_id=self.document_id("doc-mixed-acl"),
        )
        self.assertIn(
            "RESTRICTED",
            {chunk["classification"] for chunk in admin_chunks},
        )

    def test_active_document_scope_keeps_acl_and_remove_deletes_all_its_chunks(
        self,
    ) -> None:
        scoped_chunks = vector_search.search(
            query="Show the fraud rules.",
            user=DEMO_USERS["carol"],
            top_k=20,
            document_id=self.document_id("doc-mixed-acl"),
        )

        self.assertEqual(
            {chunk["document_id"] for chunk in scoped_chunks},
            {self.document_id("doc-mixed-acl")},
        )
        self.assertTrue(
            vector_search.document_exists(
                self.document_id("doc-mixed-acl"),
                tenant="nimbuspay",
            )
        )

        vector_search.delete_document(
            self.document_id("doc-mixed-acl"),
            tenant="nimbuspay",
        )

        self.assertFalse(
            vector_search.document_exists(
                self.document_id("doc-mixed-acl"),
                tenant="nimbuspay",
            )
        )

        remaining = vector_search.search(
            query="Show the fraud rules.",
            user=DEMO_USERS["carol"],
            top_k=20,
            document_id=self.document_id("doc-mixed-acl"),
        )
        self.assertEqual(remaining, [])

    def test_reingesting_unchanged_chunks_reuses_the_stored_vector(self) -> None:
        embedding_calls = []

        def count_embedding(text: str) -> list[float]:
            embedding_calls.append(text)
            return [1.0, 0.0]

        vector_search._get_embedding = count_embedding

        chunk = {
            "chunk_id": "stable-content-id_0",
            "text": "Unchanged content",
            "source_file": "renamed-source.txt",
            "type": "true_data",
            "chunk_index": 0,
            "document_id": "stable-content-id",
            "classification": "RESTRICTED",
            "allowed_roles": ["administrator"],
            "tenant": "nimbuspay",
            "service": None,
            "effective_date": None,
            "version": "content-version",
        }

        vector_search.upsert_chunks([chunk])
        vector_search.upsert_chunks([chunk])

        self.assertEqual(
            embedding_calls,
            ["Unchanged content"],
        )

    def test_active_document_scope_does_not_expand_current_role_access(self) -> None:
        public_chunks = vector_search.search(
            query="Show the fraud rules.",
            user=DEMO_USERS["bob"],
            top_k=20,
            document_id=self.document_id("doc-restricted"),
        )
        self.assertEqual(public_chunks, [])

    def test_role_changes_filter_the_same_active_document_without_reingestion(
        self,
    ) -> None:
        expected = {
            "alice": {"PUBLIC"},
            "bob": {"PUBLIC", "INTERNAL"},
            "carol": {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"},
        }

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                chunks = vector_search.search(
                    query="Show the passage.",
                    user=user,
                    top_k=20,
                    document_id=self.document_id("doc-mixed-acl"),
                )
                self.assertEqual(
                    {chunk["classification"] for chunk in chunks},
                    expected[user_key],
                )

    def test_trusted_policy_reindexes_acl_without_reembedding(self) -> None:
        document_id = (
            "587b89a44343e650a22df598bd567926d61df62a81db50e5cb3ebec09fe07489"
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=100,
                    vector=[1.0, 0.0],
                    payload={
                        "chunk_id": f"{document_id}_0",
                        "text": (
                            "Default forecast horizon 12 hours. "
                            "Rain threshold 70 percent."
                        ),
                        "source_file": "trusted-test-document.docx",
                        "type": "true_data",
                        "chunk_index": 0,
                        "document_id": document_id,
                        "classification": "RESTRICTED",
                        "allowed_roles": ["administrator"],
                        "tenant": "nimbuspay",
                        "service": None,
                        "effective_date": None,
                        "version": document_id,
                    },
                )
            ],
        )

        before = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[100],
            with_payload=True,
            with_vectors=True,
        )[0]

        self.assertEqual(
            classification_for_document(document_id),
            Classification.PUBLIC,
        )

        updated = reindex_document_policy(document_id)

        after = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[100],
            with_payload=True,
            with_vectors=True,
        )[0]

        self.assertEqual(updated, 1)
        self.assertEqual(after.payload["classification"], "PUBLIC")
        self.assertEqual(
            after.payload["allowed_roles"],
            ["administrator", "operator", "viewer"],
        )
        self.assertEqual(after.vector, before.vector)

        viewer_results = vector_search.search(
            query="forecast horizon and rain probability",
            user=DEMO_USERS["alice"],
            top_k=10,
            document_id=document_id,
        )
        self.assertEqual(
            [chunk["document_id"] for chunk in viewer_results],
            [document_id],
        )

    def test_document_level_reindex_rejects_section_acl_documents(self) -> None:
        document_id = (
            "db86ae480023f4e1bf114c3e57ed10fa968a4fb1571cbe7189109e152002fa80"
        )

        with self.assertRaisesRegex(ValueError, "section-level ACLs"):
            reindex_document_policy(document_id)

    def test_corporate_device_document_trusted_chunk_policy_preserves_role_boundaries(
        self,
    ) -> None:
        document_id = (
            "4d64def30b98587074f28da245e19811716ac542df9612740bc72de45a443010"
        )

        chunk_texts = [
            "Synthetic acceptable-use policy overview.",
            (
                "General IT use and corporate devices: lock unattended screens, "
                "install approved software only, and keep security tools enabled."
            ),
            "Remote work includes mixed viewer, operator, and administrator guidance.",
            (
                "Operational actions: operators check telemetry, acknowledge service "
                "alerts, follow procedures, and record actions."
            ),
            "Access requests and role management includes administrator-only role assignment.",
            "Security incidents and credentials include administrator-only response procedures.",
            "Service desk and audit records include role-specific actions.",
            "Role-based demo matrix and access-control tests.",
        ]

        points = []

        for chunk_index, text in enumerate(chunk_texts):
            access = build_document_access_metadata(
                document_id=document_id,
                classification=Classification.RESTRICTED,
            )

            points.append(
                models.PointStruct(
                    id=100 + chunk_index,
                    vector=[1.0, 0.0],
                    payload={
                        **access.model_dump(mode="json"),
                        "chunk_id": f"{document_id}_{chunk_index}",
                        "text": text,
                        "source_file": (
                            "Enterprise_IT_Access_Role_Based_Policy_3_Roles.docx"
                        ),
                        "type": "true_data",
                        "chunk_index": chunk_index,
                    },
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        updated = reindex_document_policy(document_id)

        self.assertEqual(updated, 8)

        expected_classifications = {
            "alice": {"PUBLIC"},
            "bob": {"PUBLIC", "INTERNAL"},
            "carol": {"PUBLIC", "INTERNAL", "CONFIDENTIAL"},
        }

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                device_results = vector_search.search(
                    query="What are the general rules for using corporate devices?",
                    user=user,
                    top_k=20,
                    document_id=document_id,
                )

                self.assertEqual(
                    {chunk["classification"] for chunk in device_results},
                    expected_classifications[user_key],
                )

                self.assertIn(
                    "corporate devices",
                    "\n".join(
                        chunk["text"] for chunk in device_results
                    ).lower(),
                )

                alert_results = vector_search.search(
                    query="Can I acknowledge an operational alert?",
                    user=user,
                    top_k=20,
                    document_id=document_id,
                )

                alert_text = "\n".join(
                    chunk["text"] for chunk in alert_results
                )

                if user_key == "alice":
                    self.assertNotIn(
                        "acknowledge service alerts",
                        alert_text,
                    )
                else:
                    self.assertIn(
                        "acknowledge service alerts",
                        alert_text,
                    )

                if user_key == "bob":
                    self.assertNotIn(
                        "administrator-only role assignment",
                        alert_text
                        + "\n".join(
                            chunk["text"] for chunk in device_results
                        ),
                    )

    def test_technical_request_with_no_authorized_context_skips_llm(self) -> None:
        with patch("app.agents.nodes.responder.call_llm") as call_llm:
            result = responder_node(
                {
                    "user_query": "Show restricted fraud rules",
                    "intent": "technical",
                    "reranked_chunks": [],
                    "trace": ["Authorized vector results: 0"],
                }
            )

        call_llm.assert_not_called()
        self.assertIn(
            "active document",
            result["final_answer"],
        )
        self.assertIn(
            "no authorized context reached the model",
            result["trace"][-1],
        )

    def test_unknown_user_cannot_choose_an_arbitrary_role(self) -> None:
        self.assertEqual(
            main_module.resolve_demo_user("bob").roles,
            (DEMO_USERS["bob"].roles[0],),
        )

        with self.assertRaises(HTTPException) as error:
            main_module.resolve_demo_user("administrator-from-request")

        self.assertEqual(
            getattr(error.exception, "status_code", None),
            401,
        )

    def test_conversation_checkpoint_is_scoped_to_principal_and_roles(self) -> None:
        alice_thread = scope_thread_id(
            "shared-browser-thread",
            DEMO_USERS["alice"],
        )
        carol_thread = scope_thread_id(
            "shared-browser-thread",
            DEMO_USERS["carol"],
        )

        self.assertNotEqual(
            alice_thread,
            carol_thread,
        )
        self.assertEqual(
            scope_thread_id(
                alice_thread,
                DEMO_USERS["alice"],
            ),
            alice_thread,
        )

    def test_upload_metadata_defaults_to_restricted_and_survives_chunking(self) -> None:
        with TemporaryDirectory() as directory:
            document_path = Path(directory) / "fraud-rules.txt"
            document_path.write_text(
                "Synthetic restricted test data.",
                encoding="utf-8",
            )
            document = load_document(str(document_path))

        chunks = chunk_document(document)

        self.assertEqual(
            document["classification"],
            "RESTRICTED",
        )
        self.assertEqual(
            document["allowed_roles"],
            ["administrator"],
        )
        self.assertNotEqual(
            document["document_id"],
            document["source_file"],
        )
        self.assertEqual(
            document["version"],
            document["document_id"],
        )
        self.assertEqual(
            chunks[0]["tenant"],
            "nimbuspay",
        )
        self.assertEqual(
            chunks[0]["classification"],
            "RESTRICTED",
        )

    def test_query_route_passes_server_resolved_role_to_graph(self) -> None:
        captured_states = []

        class FakeGraph:
            def invoke(self, initial_state, config):
                captured_states.append(initial_state)
                return {
                    "final_answer": "test answer",
                    "trace": [],
                }

        with patch.object(
            main_module,
            "check_input",
            return_value=(True, ""),
        ), \
            patch.object(
                main_module,
                "evaluate_output",
                return_value=OutputGuardrailResult(
                    OutputGuardrailStatus.ALLOW,
                ),
            ), \
            patch.object(
                main_module,
                "compiled_graph",
                FakeGraph(),
            ):
            client = TestClient(main_module.app)

            response = client.post(
                "/query",
                json={
                    "query": "test",
                    "active_document_id": "a" * 64,
                },
                headers={"X-Demo-User": "bob"},
            )

            unknown_user = client.post(
                "/query",
                json={
                    "query": "test",
                    "active_document_id": "a" * 64,
                },
                headers={"X-Demo-User": "administrator-from-request"},
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            captured_states[0]["user_context"]["user_id"],
            "user-002",
        )
        self.assertEqual(
            captured_states[0]["user_context"]["roles"],
            ["operator"],
        )
        self.assertEqual(
            captured_states[0]["active_document_id"],
            "a" * 64,
        )
        self.assertTrue(
            response.json()["thread_id"].startswith(
                "nimbuspay:user-002:operator:"
            )
        )
        self.assertEqual(
            unknown_user.status_code,
            401,
        )


if __name__ == "__main__":
    unittest.main()