"""
tests/test_input_guardrail.py

The input rail calls a capacity-limited Groq model through NeMo. A provider
failure must be retried briefly and then FAIL CLOSED with a clear message
(never an unhandled exception, never an implicit allow).
"""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import app.guardrails.guardrail_gate as gate
import app.main as main_module
from app.guardrails.guardrail_gate import check_input

QUESTION = "What are the general rules for using corporate devices?"
ALLOWED_REPLY = {"role": "assistant", "content": "Corporate devices should be kept updated."}
REFUSAL_REPLY = {"role": "assistant", "content": "I'm sorry, I can't respond to that."}


class InputGuardrailFailClosedTests(unittest.TestCase):
    def run_check(self, side_effect):
        rails = MagicMock()
        rails.generate.side_effect = side_effect
        with patch.object(gate, "_rails", rails), patch.object(
            gate.time, "sleep"
        ) as sleep:
            result = check_input(QUESTION)
        return result, rails, sleep

    def test_normal_allow(self):
        (allowed, reason), rails, sleep = self.run_check([ALLOWED_REPLY])

        self.assertTrue(allowed)
        self.assertEqual(reason, "")
        self.assertEqual(rails.generate.call_count, 1)
        sleep.assert_not_called()

    def test_nemo_refusal_is_a_normal_block_not_an_error(self):
        (allowed, reason), rails, _ = self.run_check([REFUSAL_REPLY])

        self.assertFalse(allowed)
        self.assertNotIn("GUARDRAIL_ERROR", reason)
        self.assertEqual(rails.generate.call_count, 1)

    def test_transient_failure_is_retried_then_allowed(self):
        (allowed, reason), rails, sleep = self.run_check(
            [RuntimeError("503 over capacity"), ALLOWED_REPLY]
        )

        self.assertTrue(allowed)
        self.assertEqual(rails.generate.call_count, 2)
        sleep.assert_called_once_with(1.0)

    def test_persistent_failure_fails_closed_without_raising(self):
        (allowed, reason), rails, sleep = self.run_check(RuntimeError("503 over capacity"))

        self.assertFalse(allowed)
        self.assertTrue(reason.startswith("GUARDRAIL_ERROR"))
        self.assertEqual(rails.generate.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1.0, 2.0])

    def test_failure_log_names_the_state_and_never_logs_the_query(self):
        with self.assertLogs(gate.logger, level="WARNING") as captured:
            self.run_check(RuntimeError("503 over capacity"))

        logged = "\n".join(captured.output)
        self.assertIn("INPUT_GUARDRAIL_ERROR", logged)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(QUESTION, logged)


class QueryRouteInputGuardrailTests(unittest.TestCase):
    def test_input_guardrail_failure_returns_a_clean_response_not_a_500(self):
        class MustNotRun:
            def invoke(self, *_args, **_kwargs):
                raise AssertionError("The graph must not run when the input check fails")

        error_reply = (False, gate._INPUT_GUARDRAIL_ERROR_MESSAGE)
        with patch.object(main_module, "check_input", return_value=error_reply), patch.object(
            main_module, "compiled_graph", MustNotRun()
        ):
            response = TestClient(main_module.app).post(
                "/query",
                json={"query": QUESTION, "active_document_id": "a" * 64},
                headers={"X-Demo-User": "alice"},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["answer"].startswith("GUARDRAIL_ERROR"))
        self.assertTrue(body["blocked"])
        trace_text = "\n".join(body["trace"])
        self.assertIn("Guardrails (input): ERROR", trace_text)
        self.assertNotIn("BLOCKED", trace_text)

    def test_real_input_block_keeps_the_blocked_label(self):
        with patch.object(
            main_module, "check_input", return_value=(False, "Please rephrase.")
        ):
            body = TestClient(main_module.app).post(
                "/query",
                json={"query": QUESTION, "active_document_id": "a" * 64},
                headers={"X-Demo-User": "alice"},
            ).json()

        self.assertEqual(body["answer"], "Please rephrase.")
        self.assertIn("Guardrails (input): BLOCKED", body["trace"])


if __name__ == "__main__":
    unittest.main()