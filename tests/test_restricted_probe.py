"""
tests/test_restricted_probe.py

Regression tests for the current restricted-content probe.

Current architecture:

1. Normal retrieval applies tenant + active-document + role ACLs.
2. If authorized evidence is absent/weak, retriever performs a restricted
   probe using apply_role_filter=False.
3. The probe still remains scoped to the same tenant and active document.
4. Unauthorized candidates are identified locally using the user's ACL.
5. Restricted candidates are reranked only to determine whether relevant
   restricted evidence exists.
6. Only restricted_evidence_found (a boolean) is returned in graph state.
7. Restricted document text must never enter graph state.
8. responder_node converts restricted_evidence_found into the access-denied
   message without calling the LLM.

No Gemini, Groq, Portkey, or Qdrant Cloud calls are made by these tests.
"""

import hashlib
import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient, models

import app.services.retrieval.vector_search as vector_search
from app.agents.nodes import retriever, responder
from app.agents.nodes.responder import responder_node
from app.security.access_control import (
    DEMO_USERS,
    Classification,
    build_document_access_metadata,
)


def sha(value: str) -> str:
    """Return a deterministic SHA-256 document ID."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ============================================================================
# VECTOR SEARCH / ACL TESTS
# ============================================================================

class RestrictedProbeVectorSearchTests(unittest.TestCase):
    """
    Tests the two retrieval modes used by the current implementation:

    apply_role_filter=True
        Normal authorized retrieval.

    apply_role_filter=False
        Broad retrieval used internally by the restricted probe.

    Even in broad mode, tenant and active-document isolation must remain.
    """

    def setUp(self) -> None:
        self.client = QdrantClient(":memory:")
        self.collection_name = "restricted_probe_test"

        self.originals = (
            vector_search.client,
            vector_search.COLLECTION_NAME,
            vector_search._get_embedding,
            vector_search._payload_indexes_ready_for,
        )

        vector_search.client = self.client
        vector_search.COLLECTION_NAME = self.collection_name

        # Avoid real Gemini calls.
        vector_search._get_embedding = lambda _query: [1.0, 0.0]

        vector_search._payload_indexes_ready_for = None

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=2,
                distance=models.Distance.COSINE,
            ),
        )

        self.document_id = sha("restricted-probe-document")

        points = []

        # One chunk for every classification.
        for point_id, classification in enumerate(Classification, start=1):
            access = build_document_access_metadata(
                document_id=self.document_id,
                classification=classification,
            )

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=[1.0, 0.0],
                    payload={
                        **access.model_dump(mode="json"),
                        "chunk_id": f"probe-{classification.value.lower()}",
                        "text": f"Synthetic {classification.value} passage",
                        "source_file": "probe.txt",
                        "type": "true_data",
                        "chunk_index": point_id - 1,
                        "service": "NimbusPay",
                        "effective_date": None,
                        "version": "1.0",
                    },
                )
            )

        # Same document but DIFFERENT tenant.
        points.append(
            models.PointStruct(
                id=50,
                vector=[1.0, 0.0],
                payload={
                    "chunk_id": "probe-other-tenant",
                    "text": "Other tenant secret",
                    "source_file": "other.txt",
                    "type": "true_data",
                    "chunk_index": 0,
                    "document_id": self.document_id,
                    "classification": "RESTRICTED",
                    "allowed_roles": ["administrator"],
                    "tenant": "other-tenant",
                    "service": "OtherTenant",
                    "effective_date": None,
                    "version": "1.0",
                },
            )
        )

        # Different document in the SAME tenant.
        other_document_id = sha("another-document")

        other_access = build_document_access_metadata(
            document_id=other_document_id,
            classification=Classification.RESTRICTED,
        )

        points.append(
            models.PointStruct(
                id=51,
                vector=[1.0, 0.0],
                payload={
                    **other_access.model_dump(mode="json"),
                    "chunk_id": "probe-other-document",
                    "text": "Other document secret",
                    "source_file": "another.txt",
                    "type": "true_data",
                    "chunk_index": 0,
                    "service": "NimbusPay",
                    "effective_date": None,
                    "version": "1.0",
                },
            )
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def tearDown(self) -> None:
        (
            vector_search.client,
            vector_search.COLLECTION_NAME,
            vector_search._get_embedding,
            vector_search._payload_indexes_ready_for,
        ) = self.originals

        self.client.close()

    def search(self, user, *, apply_role_filter=True):
        return vector_search.search(
            query="Show the passage.",
            user=user,
            top_k=20,
            document_id=self.document_id,
            apply_role_filter=apply_role_filter,
        )

    def test_authorized_search_returns_only_role_accessible_content(self):
        expected = {
            "alice": {"PUBLIC"},
            "bob": {"PUBLIC", "INTERNAL"},
            "carol": {
                "PUBLIC",
                "INTERNAL",
                "CONFIDENTIAL",
                "RESTRICTED",
            },
        }

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                chunks = self.search(user)

                self.assertEqual(
                    {
                        chunk["classification"]
                        for chunk in chunks
                    },
                    expected[user_key],
                )

    def test_probe_mode_stays_inside_same_tenant(self):
        """
        apply_role_filter=False must NOT disable tenant isolation.
        """

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                chunks = self.search(
                    user,
                    apply_role_filter=False,
                )

                chunk_ids = {
                    chunk["chunk_id"]
                    for chunk in chunks
                }

                self.assertNotIn(
                    "probe-other-tenant",
                    chunk_ids,
                )

                for chunk in chunks:
                    self.assertEqual(
                        chunk["tenant"],
                        user.tenant,
                    )

    def test_probe_mode_stays_inside_active_document(self):
        """
        apply_role_filter=False must NOT disable active-document isolation.
        """

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                chunks = self.search(
                    user,
                    apply_role_filter=False,
                )

                chunk_ids = {
                    chunk["chunk_id"]
                    for chunk in chunks
                }

                self.assertNotIn(
                    "probe-other-document",
                    chunk_ids,
                )

                for chunk in chunks:
                    self.assertEqual(
                        chunk["document_id"],
                        self.document_id,
                    )

    def test_authorized_and_unauthorized_sets_cover_active_document(self):
        """
        Broad retrieval gives the complete active document.

        Normal retrieval gives the authorized subset.

        Therefore the difference between them represents the content the
        current role cannot read.
        """

        all_chunk_ids = {
            f"probe-{classification.value.lower()}"
            for classification in Classification
        }

        for user_key, user in DEMO_USERS.items():
            with self.subTest(user=user_key):
                all_chunks = self.search(
                    user,
                    apply_role_filter=False,
                )

                authorized_chunks = self.search(user)

                all_ids = {
                    chunk["chunk_id"]
                    for chunk in all_chunks
                }

                authorized_ids = {
                    chunk["chunk_id"]
                    for chunk in authorized_chunks
                }

                restricted_ids = all_ids - authorized_ids

                self.assertEqual(
                    authorized_ids & restricted_ids,
                    set(),
                )

                self.assertEqual(
                    authorized_ids | restricted_ids,
                    all_chunk_ids,
                )

    def test_default_authorized_search_remains_unchanged(self):
        alice_results = self.search(DEMO_USERS["alice"])

        self.assertEqual(
            {
                chunk["classification"]
                for chunk in alice_results
            },
            {"PUBLIC"},
        )


# ============================================================================
# RETRIEVER RESTRICTED-PROBE TESTS
# ============================================================================

DOC_ID = "a" * 64
QUESTION = "How can MFA be changed?"
SECRET_TEXT = "SECRET RESTRICTED PASSAGE TEXT"


def restricted_candidate() -> dict:
    """
    Synthetic restricted candidate representing the confidential MFA chunk.

    This candidate exists only inside the mocked restricted-probe execution.
    It must never appear in the returned graph state.
    """

    return {
        "chunk_id": f"{DOC_ID}_7",
        "text": SECRET_TEXT,
        "source_file": "handbook.docx",
        "chunk_index": 7,
        "classification": "CONFIDENTIAL",
        "document_id": DOC_ID,
        "tenant": "nimbuspay",
        "allowed_roles": ["administrator"],
        "score": 0.9,
    }


def authorized_candidate() -> dict:
    """Synthetic strong PUBLIC candidate."""

    return {
        "chunk_id": f"{DOC_ID}_1",
        "text": "Relevant public information.",
        "source_file": "handbook.docx",
        "chunk_index": 1,
        "classification": "PUBLIC",
        "document_id": DOC_ID,
        "tenant": "nimbuspay",
        "allowed_roles": [
            "administrator",
            "operator",
            "viewer",
        ],
        "score": 0.95,
    }


def make_retriever_state(
    *,
    user="alice",
    **overrides,
) -> dict:
    state = {
        "user_query": QUESTION,
        "intent": "technical",
        "reranked_chunks": [],
        "trace": [],
        "user_context": DEMO_USERS[user].model_dump(
            mode="json"
        ),
        "active_document_id": DOC_ID,
    }

    state.update(overrides)

    return state


class RetrieverRestrictedProbeTests(unittest.TestCase):
    """
    Tests the actual restricted-probe implementation in retriever.py.
    """

    def run_retriever(
        self,
        state,
        *,
        normal_candidates=None,
        restricted_candidates=None,
        normal_rerank_score=None,
        restricted_rerank_score=0.88,
    ):
        """
        Stub all retrieval/reranking dependencies.

        The important distinction is:

        normal retrieval:
            apply_role_filter=True

        restricted probe:
            apply_role_filter=False
        """

        if normal_candidates is None:
            normal_candidates = []

        if restricted_candidates is None:
            restricted_candidates = [
                restricted_candidate()
            ]

        search_calls = []

        def fake_search(
            *,
            query,
            user,
            top_k,
            document_id,
            apply_role_filter=True,
        ):
            search_calls.append(
                {
                    "query": query,
                    "user": user,
                    "top_k": top_k,
                    "document_id": document_id,
                    "apply_role_filter": apply_role_filter,
                }
            )

            if apply_role_filter:
                return list(normal_candidates)

            return list(restricted_candidates)

        def fake_rerank(
            *,
            original_query,
            candidates,
            active_document_id,
            top_n,
        ):
            # Restricted probe candidates contain CONFIDENTIAL.
            is_restricted_probe = any(
                candidate.get("classification")
                in {
                    "CONFIDENTIAL",
                    "RESTRICTED",
                    "INTERNAL",
                }
                for candidate in candidates
            )

            if is_restricted_probe:
                if restricted_rerank_score is None:
                    return []

                return [
                    {
                        **candidate,
                        "rerank_score": restricted_rerank_score,
                    }
                    for candidate in candidates
                ]

            # Normal authorized retrieval.
            if normal_rerank_score is None:
                return [
                    {
                        **candidate,
                        "rerank_score": candidate.get(
                            "rerank_score",
                            0.0,
                        ),
                    }
                    for candidate in candidates
                ]

            return [
                {
                    **candidate,
                    "rerank_score": normal_rerank_score,
                }
                for candidate in candidates
            ]

        with (
            patch.object(
                retriever,
                "search",
                side_effect=fake_search,
            ),
            patch.object(
                retriever,
                "rerank",
                side_effect=fake_rerank,
            ),
            patch.object(
                retriever,
                "generate_hypothetical_document",
                return_value="HYPOTHETICAL MFA ANSWER",
            ),
            patch.object(
                retriever,
                "expand_query",
                return_value=[
                    "MFA policy change",
                    "change MFA policy",
                ],
            ),
        ):
            result = retriever.retriever_node(state)

        return result, search_calls

    def test_relevant_restricted_content_sets_flag_true(self):
        result, search_calls = self.run_retriever(
            make_retriever_state(),
            restricted_rerank_score=0.88,
        )

        self.assertTrue(
            result["restricted_evidence_found"]
        )

        # Restricted text must never enter graph state.
        self.assertNotIn(
            SECRET_TEXT,
            repr(result),
        )

        restricted_calls = [
            call
            for call in search_calls
            if call["apply_role_filter"] is False
        ]

        self.assertGreater(
            len(restricted_calls),
            0,
        )

    def test_restricted_probe_uses_active_document_id(self):
        result, search_calls = self.run_retriever(
            make_retriever_state(),
            restricted_rerank_score=0.88,
        )

        self.assertTrue(
            result["restricted_evidence_found"]
        )

        restricted_calls = [
            call
            for call in search_calls
            if call["apply_role_filter"] is False
        ]

        self.assertGreater(
            len(restricted_calls),
            0,
        )

        for call in restricted_calls:
            self.assertEqual(
                call["document_id"],
                DOC_ID,
            )

    def test_restricted_probe_preserves_tenant(self):
        result, search_calls = self.run_retriever(
            make_retriever_state(),
            restricted_rerank_score=0.88,
        )

        self.assertTrue(
            result["restricted_evidence_found"]
        )

        restricted_calls = [
            call
            for call in search_calls
            if call["apply_role_filter"] is False
        ]

        self.assertGreater(
            len(restricted_calls),
            0,
        )

        for call in restricted_calls:
            self.assertEqual(
                call["user"].tenant,
                "nimbuspay",
            )

    def test_restricted_text_never_enters_graph_state(self):
        result, _ = self.run_retriever(
            make_retriever_state(),
            restricted_rerank_score=0.88,
        )

        result_blob = repr(result)

        self.assertNotIn(
            SECRET_TEXT,
            result_blob,
        )

        self.assertNotIn(
            "CONFIDENTIAL",
            result_blob,
        )

        self.assertNotIn(
            "0.88",
            result_blob,
        )

        self.assertEqual(
            result["restricted_evidence_found"],
            True,
        )

    def test_restricted_content_below_relevance_floor_returns_false(self):
        result, _ = self.run_retriever(
            make_retriever_state(),
            restricted_rerank_score=0.0004,
        )

        self.assertFalse(
            result["restricted_evidence_found"]
        )

    def test_no_restricted_candidates_returns_false(self):
        result, _ = self.run_retriever(
            make_retriever_state(),
            restricted_candidates=[],
        )

        self.assertFalse(
            result["restricted_evidence_found"]
        )

    def test_strong_authorized_evidence_skips_restricted_probe(self):
        """
        If authorized evidence is already strong, there is no reason to
        perform the restricted-content existence probe.
        """

        result, search_calls = self.run_retriever(
            make_retriever_state(),
            normal_candidates=[
                authorized_candidate()
            ],
            normal_rerank_score=0.95,
            restricted_rerank_score=0.88,
        )

        self.assertFalse(
            result["restricted_evidence_found"]
        )

        restricted_calls = [
            call
            for call in search_calls
            if call["apply_role_filter"] is False
        ]

        self.assertEqual(
            restricted_calls,
            [],
        )

        self.assertEqual(
            len(result["reranked_chunks"]),
            1,
        )

        self.assertEqual(
            result["reranked_chunks"][0]["rerank_score"],
            0.95,
        )

    def test_invalid_active_document_id_skips_probe(self):
        state = make_retriever_state(
            active_document_id="not-a-valid-document-id"
        )

        result, search_calls = self.run_retriever(
            state,
            restricted_rerank_score=0.88,
        )

        self.assertFalse(
            result["restricted_evidence_found"]
        )

        self.assertEqual(
            search_calls,
            [],
        )

        trace = " ".join(result["trace"]).lower()

        self.assertIn(
            "missing or invalid active document id",
            trace,
        )

    def test_probe_only_runs_when_authorized_evidence_is_weak(self):
        """
        The restricted probe is conditional. It should run when the normal
        authorized path has no useful evidence.
        """

        result, search_calls = self.run_retriever(
            make_retriever_state(),
            normal_candidates=[],
            restricted_rerank_score=0.88,
        )

        self.assertTrue(
            result["restricted_evidence_found"]
        )

        restricted_calls = [
            call
            for call in search_calls
            if call["apply_role_filter"] is False
        ]

        self.assertGreater(
            len(restricted_calls),
            0,
        )

    def test_missing_user_context_fails_before_probe(self):
        state = make_retriever_state()

        del state["user_context"]

        with self.assertRaises(KeyError):
            self.run_retriever(state)


# ============================================================================
# RESPONDER TESTS
# ============================================================================

class ResponderRestrictedProbeTests(unittest.TestCase):
    """
    responder.py receives only the boolean restricted_evidence_found.

    It must never need restricted document text to produce the access-denied
    response.
    """

    def make_state(
        self,
        *,
        chunks=None,
        restricted_evidence_found=False,
    ):
        return {
            "user_query": QUESTION,
            "intent": "technical",
            "reranked_chunks": (
                [] if chunks is None else chunks
            ),
            "trace": [],
            "restricted_evidence_found": (
                restricted_evidence_found
            ),
            "answer_length": None,
        }

    def test_restricted_evidence_returns_access_denied_message(self):
        state = self.make_state(
            restricted_evidence_found=True,
        )

        with patch.object(
            responder,
            "call_llm",
            return_value="LLM ANSWER",
        ) as llm_mock:
            result = responder_node(state)

        self.assertEqual(
            result["final_answer"],
            responder.ACCESS_DENIED_MESSAGE,
        )

        llm_mock.assert_not_called()

    def test_no_restricted_evidence_returns_no_evidence_message(self):
        state = self.make_state(
            restricted_evidence_found=False,
        )

        with patch.object(
            responder,
            "call_llm",
            return_value="LLM ANSWER",
        ) as llm_mock:
            result = responder_node(state)

        self.assertEqual(
            result["final_answer"],
            responder.NO_EVIDENCE_MESSAGE,
        )

        llm_mock.assert_not_called()

    def test_restricted_message_contains_no_secret_text(self):
        state = self.make_state(
            restricted_evidence_found=True,
        )

        with patch.object(
            responder,
            "call_llm",
            return_value="LLM ANSWER",
        ):
            result = responder_node(state)

        result_blob = repr(result)

        self.assertNotIn(
            SECRET_TEXT,
            result_blob,
        )

        self.assertNotIn(
            "CONFIDENTIAL",
            result_blob,
        )

    def test_weak_authorized_context_plus_restricted_flag_returns_access_denied(self):
        weak_chunks = [
            {
                "chunk_id": "weak-1",
                "text": "Irrelevant public text.",
                "source_file": "handbook.docx",
                "chunk_index": 1,
                "classification": "PUBLIC",
                "document_id": DOC_ID,
                "tenant": "nimbuspay",
                "score": 0.1,
                "rerank_score": 0.0003,
            }
        ]

        state = self.make_state(
            chunks=weak_chunks,
            restricted_evidence_found=True,
        )

        with patch.object(
            responder,
            "call_llm",
            return_value="LLM ANSWER",
        ) as llm_mock:
            result = responder_node(state)

        self.assertEqual(
            result["final_answer"],
            responder.ACCESS_DENIED_MESSAGE,
        )

        llm_mock.assert_not_called()

    def test_strong_authorized_context_calls_llm(self):
        strong_chunks = [
            {
                "chunk_id": "authorized-1",
                "text": "Relevant authorized information.",
                "source_file": "handbook.docx",
                "chunk_index": 1,
                "classification": "PUBLIC",
                "document_id": DOC_ID,
                "tenant": "nimbuspay",
                "score": 0.95,
                "rerank_score": 0.95,
                "allowed_roles": [
                    "administrator",
                    "operator",
                    "viewer",
                ],
            }
        ]

        state = self.make_state(
            chunks=strong_chunks,
            restricted_evidence_found=False,
        )

        with patch.object(
            responder,
            "call_llm",
            return_value="LLM ANSWER",
        ) as llm_mock:
            result = responder_node(state)

        self.assertEqual(
            result["final_answer"],
            "LLM ANSWER",
        )

        llm_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()