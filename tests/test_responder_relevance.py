"""
tests/test_responder_relevance.py

The responder must not ask the LLM to answer from authorized-but-unrelated
chunks. This is a relevance check only: authorization is enforced earlier by
the Qdrant filter and is not touched here.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import app.gateway.portkey_client as portkey
from app.agents.nodes import responder
from app.gateway.portkey_client import ACCESS_DENIED_MESSAGE, NO_EVIDENCE_MESSAGE

QUESTION = "Can I acknowledge an operational alert?"


def chunk(score):
    chunk_data = {"text": "Some authorized passage.", "source_file": "doc.docx"}
    if score is not None:
        chunk_data["rerank_score"] = score
    return chunk_data


class ResponderRelevanceFloorTests(unittest.TestCase):
    def run_node(self, chunks, *, intent="technical", floor=0.01):
        state = {
            "user_query": QUESTION,
            "intent": intent,
            "reranked_chunks": chunks,
            "trace": [],
        }
        with patch.object(responder, "MIN_RERANK_SCORE", floor), patch.object(
            responder, "call_llm", return_value="LLM ANSWER"
        ) as call_llm:
            result = responder.responder_node(state)
        return result, call_llm

    def test_all_chunks_below_floor_abstains_without_calling_the_llm(self):
        result, call_llm = self.run_node([chunk(0.0033), chunk(0.0000)])

        call_llm.assert_not_called()
        self.assertEqual(result["final_answer"], NO_EVIDENCE_MESSAGE)
        self.assertIn("relevance floor", result["trace"][-1])

    def test_relevant_chunk_reaches_the_llm_and_score_is_traced(self):
        result, call_llm = self.run_node([chunk(0.9909), chunk(0.0033)])

        call_llm.assert_called_once()
        self.assertEqual(result["final_answer"], "LLM ANSWER")
        self.assertIn("Best rerank score: 0.9909", result["trace"])

    def test_best_chunk_decides_not_the_worst(self):
        _, call_llm = self.run_node([chunk(0.0033), chunk(0.9)])
        call_llm.assert_called_once()

    def test_score_exactly_at_floor_is_allowed(self):
        _, call_llm = self.run_node([chunk(0.01)], floor=0.01)
        call_llm.assert_called_once()

    def test_unscored_chunks_are_not_judged_by_the_floor(self):
        _, call_llm = self.run_node([chunk(None)])
        call_llm.assert_called_once()

    def test_conversational_path_is_unchanged(self):
        result, call_llm = self.run_node([], intent="conversational")

        call_llm.assert_called_once()
        self.assertEqual(result["final_answer"], "LLM ANSWER")

    def test_no_authorized_chunks_still_abstains(self):
        result, call_llm = self.run_node([])

        call_llm.assert_not_called()
        self.assertEqual(result["final_answer"], NO_EVIDENCE_MESSAGE)
        self.assertIn("no authorized context", result["trace"][-1])

    def test_relevant_restricted_evidence_returns_access_denied_without_llm(self):
        state = {
            "user_query": QUESTION,
            "intent": "technical",
            "reranked_chunks": [],
            "restricted_evidence_found": True,
            "trace": [],
        }
        with patch.object(responder, "call_llm") as call_llm:
            result = responder.responder_node(state)

        call_llm.assert_not_called()
        self.assertEqual(result["final_answer"], ACCESS_DENIED_MESSAGE)

    def test_low_authorized_score_returns_access_denied_when_restricted_match_exists(self):
        state = {
            "user_query": QUESTION,
            "intent": "technical",
            "reranked_chunks": [chunk(0.0033)],
            "restricted_evidence_found": True,
            "trace": [],
        }
        with patch.object(responder, "call_llm") as call_llm:
            result = responder.responder_node(state)

        call_llm.assert_not_called()
        self.assertEqual(result["final_answer"], ACCESS_DENIED_MESSAGE)

    def test_restricted_flag_does_not_override_relevant_authorized_evidence(self):
        state = {
            "user_query": QUESTION,
            "intent": "technical",
            "reranked_chunks": [chunk(0.9)],
            "restricted_evidence_found": True,
            "trace": [],
        }
        with patch.object(responder, "call_llm", return_value="LLM ANSWER") as call_llm:
            result = responder.responder_node(state)

        call_llm.assert_called_once()
        self.assertEqual(result["final_answer"], "LLM ANSWER")


class ResponderPromptAbstainWordingTests(unittest.TestCase):
    def system_prompt_for(self, answer_length):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        with patch.object(portkey, "client", fake_client), patch.object(
            portkey, "record_chat_usage"
        ):
            portkey.call_llm(
                user_query=QUESTION,
                context_chunks=[chunk(0.5)],
                conversation_history=[],
                answer_length=answer_length,
            )
        messages = fake_client.chat.completions.create.call_args.kwargs["messages"]
        return messages[0]["content"]

    def test_prompt_contains_the_exact_abstain_sentence(self):
        for length in ("short", "long", None):
            with self.subTest(answer_length=length):
                self.assertIn(NO_EVIDENCE_MESSAGE, self.system_prompt_for(length))

    def test_length_guidance_applies_only_when_the_document_supports_an_answer(self):
        for length in ("short", "long"):
            with self.subTest(answer_length=length):
                self.assertIn(
                    "When the document supports an answer",
                    self.system_prompt_for(length),
                )


if __name__ == "__main__":
    unittest.main()