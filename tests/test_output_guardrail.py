"""
tests/test_output_guardrail.py

Unit tests for the explicit output-guardrail states (ALLOW / BLOCK / ERROR /
INVALID_DECISION) and for how /query labels them in the trace.

The Groq client is mocked, so nothing here calls a live API except the two
opt-in tests at the bottom (set RUN_LIVE_GUARDRAIL_TESTS=1 to run them).
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import openai
from fastapi.testclient import TestClient

import app.guardrails.guardrail_gate as gate
import app.main as main_module
from app.guardrails.guardrail_gate import (
    OutputGuardrailResult,
    OutputGuardrailStatus,
    check_output,
    evaluate_output,
)

SAFE_ANSWER = (
    "Lock the screen when leaving a device unattended and install software "
    "only through approved channels."
)
SENTINEL_ANSWER = "SENTINEL-ANSWER-TEXT-MUST-NOT-APPEAR-IN-LOGS"


def fake_completion(content, finish_reason="stop"):
    """Mimic the parts of an OpenAI chat completion that gate.py reads."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content),
            )
        ]
    )


def mocked_client(*, returns=None, raises=None):
    client = MagicMock()
    create = client.chat.completions.create
    if raises is not None:
        create.side_effect = raises
    else:
        create.return_value = returns
    return client


class OutputGuardrailStateTests(unittest.TestCase):
    def run_check(self, *, returns=None, raises=None, answer=SAFE_ANSWER):
        with patch.object(
            gate,
            "_output_guardrail_client",
            mocked_client(returns=returns, raises=raises),
        ):
            return evaluate_output(answer)

    def assert_fails_closed_as(self, result, expected_status):
        self.assertEqual(result.status, expected_status)
        self.assertFalse(result.allowed)
        self.assertTrue(result.message.startswith("GUARDRAIL_ERROR"))

    # 1. Valid ALLOW
    def test_valid_allow_releases_the_answer(self):
        result = self.run_check(
            returns=fake_completion('{"decision": "ALLOW", "reason": ""}')
        )
        self.assertEqual(result.status, OutputGuardrailStatus.ALLOW)
        self.assertTrue(result.allowed)
        self.assertEqual(result.message, "")

    # 2. Valid BLOCK is a normal refusal, not GUARDRAIL_ERROR
    def test_valid_block_is_a_normal_safety_refusal(self):
        result = self.run_check(
            returns=fake_completion(
                '{"decision": "BLOCK", "reason": "Contains private credentials."}'
            )
        )
        self.assertEqual(result.status, OutputGuardrailStatus.BLOCK)
        self.assertFalse(result.allowed)
        self.assertEqual(result.message, "Contains private credentials.")
        self.assertNotIn("GUARDRAIL_ERROR", result.message)

    # 3. Malformed JSON
    def test_malformed_json_fails_closed(self):
        result = self.run_check(returns=fake_completion("not json at all"))
        self.assert_fails_closed_as(result, OutputGuardrailStatus.INVALID_DECISION)

    def test_empty_or_missing_content_fails_closed(self):
        for content in ("", None):
            with self.subTest(content=content):
                result = self.run_check(
                    returns=fake_completion(content, finish_reason="length")
                )
                self.assert_fails_closed_as(
                    result, OutputGuardrailStatus.INVALID_DECISION
                )

    def test_truncated_json_fails_closed(self):
        result = self.run_check(
            returns=fake_completion(
                '{"decision": "ALLOW", "rea',
                finish_reason="length",
            )
        )
        self.assert_fails_closed_as(
            result, OutputGuardrailStatus.INVALID_DECISION
        )

    # 4. Missing decision field
    def test_missing_decision_field_fails_closed(self):
        result = self.run_check(
            returns=fake_completion('{"reason": "looks fine"}')
        )
        self.assert_fails_closed_as(
            result, OutputGuardrailStatus.INVALID_DECISION
        )

    # 5. Invalid decision value (and extra fields, which the schema forbids)
    def test_invalid_decision_value_fails_closed(self):
        for payload in (
            '{"decision": "MAYBE", "reason": ""}',
            '{"decision": "allow", "reason": ""}',
            '{"decision": "ALLOW", "reason": "", "extra": 1}',
        ):
            with self.subTest(payload=payload):
                result = self.run_check(returns=fake_completion(payload))
                self.assert_fails_closed_as(
                    result, OutputGuardrailStatus.INVALID_DECISION
                )

    def test_block_without_reason_fails_closed(self):
        result = self.run_check(
            returns=fake_completion(
                '{"decision": "BLOCK", "reason": "   "}'
            )
        )
        self.assert_fails_closed_as(
            result, OutputGuardrailStatus.INVALID_DECISION
        )

    # 6. Provider / API exception
    def test_provider_exception_is_error_not_block(self):
        result = self.run_check(raises=RuntimeError("provider exploded"))
        self.assert_fails_closed_as(result, OutputGuardrailStatus.ERROR)
        self.assertNotEqual(result.status, OutputGuardrailStatus.BLOCK)

    # 7. Fallback model is used only when the primary provider call fails.
    #
    # The first mocked call represents the primary safeguard model being
    # temporarily unavailable (for example, Groq capacity/503). The second
    # call represents the fallback model returning a valid BLOCK decision.
    def test_fallback_model_used_only_when_primary_provider_fails(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            RuntimeError("primary over capacity"),
            fake_completion(
                '{"decision": "BLOCK", "reason": "Contains credentials."}'
            ),
        ]

        with patch.object(gate, "_output_guardrail_client", client):
            result = evaluate_output(SAFE_ANSWER)

        self.assertEqual(result.status, OutputGuardrailStatus.BLOCK)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    # 8. An invalid decision from the primary model is NOT a provider failure.
    #
    # Therefore the fallback must not be used. We fail closed with
    # INVALID_DECISION rather than trying another model and potentially
    # changing the safety decision.
    def test_fallback_is_not_used_when_primary_answers_with_invalid_decision(
        self,
    ):
        client = MagicMock()
        client.chat.completions.create.return_value = fake_completion(
            "garbage"
        )

        with patch.object(gate, "_output_guardrail_client", client):
            result = evaluate_output(SAFE_ANSWER)

        self.assertEqual(
            result.status,
            OutputGuardrailStatus.INVALID_DECISION,
        )
        self.assertEqual(client.chat.completions.create.call_count, 1)

    # 9. Timeout / rate limit are exposed distinctly by the openai client
    def test_rate_limit_and_timeout_are_errors(self):
        request = httpx.Request(
            "POST",
            "https://api.groq.com/openai/v1/chat/completions",
        )
        rate_limit = openai.RateLimitError(
            "rate limited",
            response=httpx.Response(429, request=request),
            body=None,
        )
        timeout = openai.APITimeoutError(request=request)

        for name, exc in (("rate_limit", rate_limit), ("timeout", timeout)):
            with self.subTest(error=name):
                result = self.run_check(raises=exc)
                self.assert_fails_closed_as(
                    result,
                    OutputGuardrailStatus.ERROR,
                )

    # Backward-compatible wrapper keeps the (allowed, reason) contract
    def test_check_output_wrapper_keeps_tuple_contract(self):
        with patch.object(
            gate,
            "_output_guardrail_client",
            mocked_client(
                returns=fake_completion(
                    '{"decision": "ALLOW", "reason": ""}'
                )
            ),
        ):
            self.assertEqual(check_output(SAFE_ANSWER), (True, ""))

        with patch.object(
            gate,
            "_output_guardrail_client",
            mocked_client(raises=RuntimeError("x")),
        ):
            allowed, reason = check_output(SAFE_ANSWER)

        self.assertFalse(allowed)
        self.assertTrue(reason.startswith("GUARDRAIL_ERROR"))

    # Logging: distinct labels, no answer text, no key
    def test_error_log_names_the_state_and_never_logs_the_answer(self):
        debug_settings = MagicMock()
        debug_settings.guardrail_debug_logging = True

        with patch.object(
            gate,
            "settings",
            debug_settings,
        ), self.assertLogs(
            gate.logger,
            level="WARNING",
        ) as captured:
            self.run_check(
                raises=RuntimeError("provider exploded"),
                answer=SENTINEL_ANSWER,
            )

        logged = "\n".join(captured.output)
        self.assertIn("OUTPUT_GUARDRAIL_ERROR", logged)
        self.assertIn("RuntimeError", logged)
        self.assertNotIn(SENTINEL_ANSWER, logged)

    def test_invalid_decision_log_uses_its_own_label(self):
        with self.assertLogs(gate.logger, level="WARNING") as captured:
            self.run_check(
                returns=fake_completion(
                    "garbage",
                    finish_reason="length",
                )
            )

        logged = "\n".join(captured.output)
        self.assertIn("OUTPUT_GUARDRAIL_INVALID_DECISION", logged)
        self.assertIn("finish_reason=length", logged)


class QueryRouteOutputGuardrailTraceTests(unittest.TestCase):
    """/query must label BLOCK, ERROR and INVALID_DECISION differently."""

    class FakeGraph:
        def invoke(self, initial_state, config):
            return {"final_answer": "graph answer", "trace": []}

    def post_with_output_result(self, output_result):
        with patch.object(
            main_module,
            "check_input",
            return_value=(True, ""),
        ), patch.object(
            main_module,
            "evaluate_output",
            return_value=output_result,
        ), patch.object(
            main_module,
            "compiled_graph",
            self.FakeGraph(),
        ):
            client = TestClient(main_module.app)
            return client.post(
                "/query",
                json={
                    "query": "test",
                    "active_document_id": "a" * 64,
                },
                headers={"X-Demo-User": "alice"},
            ).json()

    def test_allow_passes_the_graph_answer_through(self):
        body = self.post_with_output_result(
            OutputGuardrailResult(OutputGuardrailStatus.ALLOW)
        )
        self.assertEqual(body["answer"], "graph answer")
        self.assertFalse(body["blocked"])
        self.assertIn("Guardrails (output): passed", body["trace"])

    def test_block_is_labelled_blocked(self):
        body = self.post_with_output_result(
            OutputGuardrailResult(
                OutputGuardrailStatus.BLOCK,
                "Unsafe content.",
            )
        )
        self.assertEqual(body["answer"], "Unsafe content.")
        self.assertTrue(body["blocked"])
        self.assertTrue(
            any(
                "BLOCKED (safety decision)" in line
                for line in body["trace"]
            )
        )

    def test_error_and_invalid_decision_are_not_labelled_blocked(self):
        for status, label in (
            (OutputGuardrailStatus.ERROR, "ERROR"),
            (OutputGuardrailStatus.INVALID_DECISION, "INVALID_DECISION"),
        ):
            with self.subTest(status=status.value):
                body = self.post_with_output_result(
                    OutputGuardrailResult(
                        status,
                        gate._GUARDRAIL_ERROR_MESSAGE,
                    )
                )
                self.assertTrue(
                    body["answer"].startswith("GUARDRAIL_ERROR")
                )
                self.assertNotEqual(body["answer"], "graph answer")
                self.assertTrue(body["blocked"])
                trace_text = "\n".join(body["trace"])
                self.assertIn(
                    f"Guardrails (output): {label}",
                    trace_text,
                )
                self.assertNotIn("BLOCKED", trace_text)


@unittest.skipUnless(
    os.getenv("RUN_LIVE_GUARDRAIL_TESTS") == "1",
    "Live Groq test; set RUN_LIVE_GUARDRAIL_TESTS=1 to run.",
)
class LiveOutputGuardrailTests(unittest.TestCase):
    """Opt-in: calls the real safety model. Results can vary by model behaviour."""

    def test_safe_rag_answer_is_allowed(self):
        result = evaluate_output(SAFE_ANSWER)
        self.assertEqual(
            result.status,
            OutputGuardrailStatus.ALLOW,
            result,
        )

    def test_response_leaking_credentials_is_blocked(self):
        unsafe = (
            "Sure! The admin login is admin / Tr0ub4dor&3 and the production "
            "database key is sk_live_FAKE0000000000000000. Use them directly."
        )
        result = evaluate_output(unsafe)
        self.assertEqual(
            result.status,
            OutputGuardrailStatus.BLOCK,
            result,
        )


if __name__ == "__main__":
    unittest.main()