import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.gateway.portkey_client import EmptyLLMResponseError, call_llm
import app.main as main_module


class EmptyModelAnswerTests(unittest.TestCase):
    def test_empty_provider_content_is_not_returned_as_an_answer(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="  ", refusal=None),
                )
            ]
        )

        with patch("app.gateway.portkey_client.client") as client_mock, patch(
            "app.gateway.portkey_client.record_chat_usage"
        ):
            client_mock.chat.completions.create.return_value = response
            with self.assertRaises(EmptyLLMResponseError):
                call_llm(
                    user_query="Tell me about Aarush.",
                    context_chunks=[],
                    conversation_history=[],
                )

    def test_provider_refusal_is_returned_when_completion_content_is_empty(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        refusal="I can't help with that request.",
                    ),
                )
            ]
        )

        with patch("app.gateway.portkey_client.client") as client_mock, patch(
            "app.gateway.portkey_client.record_chat_usage"
        ):
            client_mock.chat.completions.create.return_value = response
            answer = call_llm(
                user_query="Tell me about Aarush.",
                context_chunks=[],
                conversation_history=[],
            )

        self.assertEqual(answer, "I can't help with that request.")

    def test_empty_model_answer_becomes_retryable_api_error(self) -> None:
        class EmptyAnswerGraph:
            def invoke(self, initial_state, config):
                raise EmptyLLMResponseError("empty provider completion")

        with patch.object(main_module, "check_input", return_value=(True, "")), patch.object(
            main_module,
            "compiled_graph",
            EmptyAnswerGraph(),
        ):
            client = TestClient(main_module.app, raise_server_exceptions=False)
            response = client.post(
                "/query",
                json={
                    "query": "Tell me about Aarush.",
                    "active_document_id": "a" * 64,
                },
                headers={"X-Demo-User": "alice"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "The answer provider returned an empty response. Please retry.",
        )


if __name__ == "__main__":
    unittest.main()
