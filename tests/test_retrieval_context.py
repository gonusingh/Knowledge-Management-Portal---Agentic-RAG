import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents.nodes.retriever import retriever_node
from app.agents.nodes.planner import planner_node
from app.gateway.portkey_client import (
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHUNKS,
    _assemble_context,
    call_llm,
)
from app.security.access_control import DEMO_USERS
from app.services.retrieval.reranker import rerank


class RetrievalContextTests(unittest.TestCase):
    def test_selected_document_always_routes_questions_to_retrieval(self) -> None:
        result = planner_node({
            "user_query": "Hi, what can you do?",
            "active_document_id": "a" * 64,
            "trace": [],
        })

        self.assertEqual(result["intent"], "technical")
        self.assertIn("active-document grounded retrieval", result["trace"][-1])

    def test_rerank_keeps_eight_scored_candidates(self) -> None:
        candidates = [
            {
                "chunk_id": f"chunk-{index}",
                "text": f"Passage {index}",
                "source_file": "economics.docx",
                "document_id": "a" * 64,
                "chunk_index": index,
                "type": "true_data",
                "score": 0.5,
            }
            for index in range(10)
        ]

        class FakeRanker:
            def rerank(self, _request):
                return [
                    {"id": candidate["chunk_id"], "score": 1.0 - index / 10}
                    for index, candidate in enumerate(candidates)
                ]

        with patch("app.services.retrieval.reranker._ranker", FakeRanker()):
            results = rerank(
                "GDP question",
                candidates,
                active_document_id="a" * 64,
                top_n=8,
            )

        self.assertEqual(len(results), 8)
        self.assertEqual(results[0]["rerank_score"], 1.0)
        self.assertAlmostEqual(results[-1]["rerank_score"], 0.3)
        self.assertEqual(results[-1]["chunk_index"], 7)

    def test_reranker_excludes_chunks_outside_active_document(self) -> None:
        candidates = [
            {
                "chunk_id": "active",
                "text": "Active document passage",
                "document_id": "a" * 64,
            },
            {
                "chunk_id": "unrelated",
                "text": "Unrelated passage",
                "document_id": "other-document",
            },
        ]

        class FakeRanker:
            def rerank(self, request):
                return [{"id": passage["id"], "score": 1.0} for passage in request.passages]

        with patch("app.services.retrieval.reranker._ranker", FakeRanker()):
            results = rerank(
                "question",
                candidates,
                active_document_id="a" * 64,
            )

        self.assertEqual([chunk["chunk_id"] for chunk in results], ["active"])

    def test_retriever_traces_selected_source_location_and_score(self) -> None:
        candidates = [
            {
                "chunk_id": f"chunk-{index}",
                "text": f"Passage {index}",
                "source_file": "economics.docx",
                "document_id": "economics-document",
                "chunk_index": index,
                "score": 0.5,
            }
            for index in range(8)
        ]
        selected = [
            {**candidate, "rerank_score": 0.9 - index / 10}
            for index, candidate in enumerate(candidates)
        ]
        with patch(
                 "app.agents.nodes.retriever.generate_hypothetical_document",
                 return_value="Hypothetical GDP answer",
             ), \
             patch(
                 "app.agents.nodes.retriever.expand_query",
                 return_value=["GDP difference 1", "GDP difference 2"],
             ), \
             patch("app.agents.nodes.retriever.rerank", return_value=selected) as rerank_mock, \
             patch("app.agents.nodes.retriever.search", return_value=candidates) as search_mock:
            result = retriever_node({
                "user_query": "What is the difference between real GDP and nominal GDP?",
                "user_context": DEMO_USERS["carol"].model_dump(mode="json"),
                "active_document_id": "a" * 64,
            })

        self.assertEqual(rerank_mock.call_args.kwargs["top_n"], 8)
        self.assertEqual(len(result["reranked_chunks"]), 8)
        self.assertEqual(search_mock.call_count, 4)
        self.assertTrue(all(
            call.kwargs["document_id"] == "a" * 64
            for call in search_mock.call_args_list
        ))
        self.assertIn(
            "Selected chunk: source=economics.docx, document_id=economics-document, "
            "chunk_index=0, rerank_score=0.9000",
            result["trace"],
        )

    def test_retriever_abstains_without_search_for_missing_or_invalid_scope(self) -> None:
        for document_id in (None, "not-a-content-hash"):
            with self.subTest(document_id=document_id), \
                 patch("app.agents.nodes.retriever.search") as search_mock, \
                 patch("app.agents.nodes.retriever.rerank") as rerank_mock:
                state = {
                    "user_query": "question",
                    "user_context": DEMO_USERS["carol"].model_dump(mode="json"),
                    "trace": [],
                }
                if document_id is not None:
                    state["active_document_id"] = document_id
                result = retriever_node(state)

            search_mock.assert_not_called()
            rerank_mock.assert_not_called()
            self.assertEqual(result["reranked_chunks"], [])
            self.assertIn("missing or invalid active document ID", result["trace"][-1])

    def test_retriever_skips_reranking_when_no_authorized_active_document_results(self) -> None:
        with patch("app.agents.nodes.retriever.search", return_value=[]) as search_mock, \
             patch(
                 "app.agents.nodes.retriever.generate_hypothetical_document",
                 return_value="hypothetical",
             ), \
             patch(
                 "app.agents.nodes.retriever.expand_query",
                 return_value=["alternative"],
             ), \
             patch("app.agents.nodes.retriever.rerank") as rerank_mock:
            result = retriever_node({
                "user_query": "question",
                "user_context": DEMO_USERS["carol"].model_dump(mode="json"),
                "active_document_id": "a" * 64,
                "trace": [],
            })

        self.assertEqual(search_mock.call_count, 6)
        self.assertTrue(all(
            call.kwargs["document_id"] == "a" * 64
            for call in search_mock.call_args_list
        ))
        self.assertEqual(
            [call.kwargs.get("apply_role_filter", True) for call in search_mock.call_args_list],
            [True, True, True, False, False, False],
        )
        rerank_mock.assert_not_called()
        self.assertEqual(result["reranked_chunks"], [])
        self.assertFalse(result["restricted_evidence_found"])
        self.assertIn("no authorized chunks", result["trace"][-1])

    def test_restricted_match_sets_only_boolean_state_and_never_enters_trace(self) -> None:
        restricted_chunk = {
            "chunk_id": "confidential-mfa",
            "text": "SECRET RESTRICTED MFA CONTENT",
            "source_file": "handbook.docx",
            "document_id": "a" * 64,
            "chunk_index": 7,
            "classification": "CONFIDENTIAL",
            "allowed_roles": ["administrator"],
            "score": 0.9,
        }

        def search_result(*, apply_role_filter=True, **_kwargs):
            return [] if apply_role_filter else [restricted_chunk]

        def rerank_result(**kwargs):
            self.assertEqual(
                [item["chunk_id"] for item in kwargs["candidates"]],
                ["confidential-mfa"],
            )
            return [{**restricted_chunk, "rerank_score": 0.8}]

        with patch(
            "app.agents.nodes.retriever.search",
            side_effect=search_result,
        ) as search_mock, patch(
            "app.agents.nodes.retriever.generate_hypothetical_document",
            return_value="hypothetical",
        ), patch(
            "app.agents.nodes.retriever.expand_query",
            return_value=["alternative"],
        ), patch(
            "app.agents.nodes.retriever.rerank",
            side_effect=rerank_result,
        ):
            result = retriever_node({
                "user_query": "How can MFA be changed?",
                "user_context": DEMO_USERS["alice"].model_dump(mode="json"),
                "active_document_id": "a" * 64,
                "trace": [],
            })

        self.assertEqual(search_mock.call_count, 6)
        self.assertTrue(result["restricted_evidence_found"])
        self.assertEqual(result["reranked_chunks"], [])
        self.assertNotIn("SECRET RESTRICTED MFA CONTENT", repr(result))
        self.assertNotIn("confidential-mfa", repr(result))

    def test_restricted_detection_requires_rerank_score_at_or_above_floor(self) -> None:
        restricted_chunk = {
            "chunk_id": "confidential-mfa",
            "text": "Restricted MFA passage",
            "source_file": "handbook.docx",
            "document_id": "a" * 64,
            "chunk_index": 7,
            "classification": "CONFIDENTIAL",
            "allowed_roles": ["administrator"],
            "score": 0.9,
        }

        def search_result(*, apply_role_filter=True, **_kwargs):
            return [] if apply_role_filter else [restricted_chunk]

        with patch(
            "app.agents.nodes.retriever.search",
            side_effect=search_result,
        ), patch(
            "app.agents.nodes.retriever.generate_hypothetical_document",
            return_value="hypothetical",
        ), patch(
            "app.agents.nodes.retriever.expand_query",
            return_value=["alternative"],
        ), patch(
            "app.agents.nodes.retriever.rerank",
            return_value=[{**restricted_chunk, "rerank_score": 0.0099}],
        ):
            result = retriever_node({
                "user_query": "How can MFA be changed?",
                "user_context": DEMO_USERS["alice"].model_dump(mode="json"),
                "active_document_id": "a" * 64,
                "trace": [],
            })

        self.assertFalse(result["restricted_evidence_found"])
        self.assertEqual(result["reranked_chunks"], [])

    def test_context_assembly_can_pass_eight_chunks_within_budget(self) -> None:
        chunks = [
            {
                "source_file": "economics.docx",
                "text": f"Economics passage {index}. " + ("x" * 1400),
            }
            for index in range(8)
        ]

        context = _assemble_context(chunks)

        self.assertEqual(MAX_CONTEXT_CHUNKS, 8)
        self.assertEqual(MAX_CONTEXT_CHARS, 12000)
        self.assertLessEqual(len(context), MAX_CONTEXT_CHARS)
        self.assertEqual(context.count("[Source:"), 8)

    def test_context_assembly_truncates_oversized_relevant_chunk_instead_of_dropping_it(self) -> None:
        chunks = [{
            "source_file": "economics.docx",
            "text": "Nominal GDP uses current prices; real GDP removes price changes. "
            + ("x" * (MAX_CONTEXT_CHARS + 4000)),
        }]

        context = _assemble_context(chunks)

        self.assertIn("Nominal GDP uses current prices", context)
        self.assertIn("real GDP removes price changes", context)
        self.assertIn("[truncated]", context)
        self.assertLessEqual(len(context), MAX_CONTEXT_CHARS)

    def test_grounded_answer_prioritizes_verbatim_evidence_and_uses_zero_temperature(self) -> None:
        document_text = (
            "The platform includes weather and irrigation components.\n\n"
            "The default forecast horizon is 12 hours. The rain suppression "
            "probability threshold is 70 percent.\n\n"
            "The dashboard reports zone activity and alerts."
        )
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="12 hours and 70 percent."))]
        )
        with patch("app.gateway.portkey_client.client") as client_mock, patch(
            "app.gateway.portkey_client.record_chat_usage"
        ):
            client_mock.chat.completions.create.return_value = fake_response

            answer = call_llm(
                user_query=(
                    "What is the default forecast horizon and what is the rain "
                    "suppression probability threshold?"
                ),
                context_chunks=[{
                    "source_file": "garden.docx",
                    "text": document_text,
                }],
                conversation_history=[],
            )

        request = client_mock.chat.completions.create.call_args.kwargs
        prompt = request["messages"][0]["content"]
        self.assertEqual(answer, "12 hours and 70 percent.")
        self.assertEqual(request["temperature"], 0.0)
        self.assertLess(
            prompt.index("The default forecast horizon is 12 hours"),
            prompt.index("The platform includes weather and irrigation components"),
        )
        self.assertIn("Never substitute a typical or remembered default", prompt)


if __name__ == "__main__":
    unittest.main()
