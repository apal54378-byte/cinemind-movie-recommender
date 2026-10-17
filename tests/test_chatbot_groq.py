"""Focused tests for the Groq-based chatbot integration.

These tests use mocks only and never require a real GROQ_API_KEY or a
network connection.
"""

import os
import types
import unittest
from unittest import mock

import httpx
import groq

from backend import llm
from backend import chatbot


_ENV_VARS = (
    "GROQ_API_KEY",
    "AI_API_KEY",
    "GROQ_MODEL",
    "AI_MODEL",
    "GROQ_BASE_URL",
    "GROQ_TIMEOUT",
)


def _request():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls, status_code, message="error"):
    return cls(message, response=httpx.Response(status_code, request=_request()), body=None)


class _EnvIsolated(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ[k] for k in _ENV_VARS if k in os.environ}
        for key in _ENV_VARS:
            os.environ.pop(key, None)
        llm.reset_client()
        llm._deprecation_warned = False

    def tearDown(self):
        for key in _ENV_VARS:
            os.environ.pop(key, None)
        os.environ.update(self._saved)
        llm.reset_client()


class ProviderConfigTests(_EnvIsolated):
    def test_default_model_is_a_current_groq_production_model(self):
        self.assertEqual(llm.get_model(), "openai/gpt-oss-120b")

    def test_groq_model_env_overrides_default(self):
        os.environ["GROQ_MODEL"] = "openai/gpt-oss-20b"
        self.assertEqual(llm.get_model(), "openai/gpt-oss-20b")

    def test_legacy_ai_model_is_used_as_fallback(self):
        os.environ["AI_MODEL"] = "llama-3.1-8b-instant"
        self.assertEqual(llm.get_model(), "llama-3.1-8b-instant")

    def test_groq_api_key_takes_priority(self):
        os.environ["GROQ_API_KEY"] = "gsk_primary"
        os.environ["AI_API_KEY"] = "legacy"
        self.assertEqual(llm.get_api_key(), "gsk_primary")

    def test_legacy_ai_api_key_is_backward_compatible(self):
        os.environ["AI_API_KEY"] = "legacy-key"
        with mock.patch("builtins.print"):
            self.assertEqual(llm.get_api_key(), "legacy-key")

    def test_not_configured_without_key(self):
        self.assertFalse(llm.is_configured())

    def test_configured_with_key_and_sdk(self):
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        self.assertTrue(llm.is_configured())

    def test_default_timeout_and_override(self):
        self.assertEqual(llm.get_timeout(), llm.DEFAULT_GROQ_TIMEOUT)
        os.environ["GROQ_TIMEOUT"] = "12.5"
        self.assertEqual(llm.get_timeout(), 12.5)

    def test_invalid_timeout_falls_back_with_warning(self):
        os.environ["GROQ_TIMEOUT"] = "not-a-number"
        with mock.patch("builtins.print"):
            self.assertEqual(llm.get_timeout(), llm.DEFAULT_GROQ_TIMEOUT)

    def test_provider_status_never_exposes_the_key(self):
        os.environ["GROQ_API_KEY"] = "gsk_super_secret"
        status = llm.provider_status()
        self.assertTrue(status["configured"])
        self.assertNotIn("gsk_super_secret", str(status))
        self.assertNotIn("api_key", status)


class ClientLifecycleTests(_EnvIsolated):
    def test_get_client_requires_a_key(self):
        with self.assertRaises(llm.LLMNotConfigured):
            llm.get_client()

    def test_get_client_creates_once_and_caches(self):
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        sentinel = object()
        with mock.patch.object(llm, "_create_client", return_value=sentinel) as create:
            self.assertIs(llm.get_client(), sentinel)
            self.assertIs(llm.get_client(), sentinel)
        create.assert_called_once()
        _, args, _ = create.mock_calls[0]
        self.assertEqual(args[0], "gsk_mock")

    def test_reset_client_allows_recreation(self):
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", side_effect=[object(), object()]) as create:
            llm.get_client()
            llm.reset_client()
            llm.get_client()
        self.assertEqual(create.call_count, 2)

    def test_client_is_built_with_official_groq_class(self):
        client = object()
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        os.environ["GROQ_TIMEOUT"] = "20"
        with mock.patch.object(llm, "Groq", return_value=client) as groq_cls:
            self.assertIs(llm.get_client(), client)
        groq_cls.assert_called_once_with(api_key="gsk_mock", timeout=20.0)

    def test_client_passes_base_url_when_overridden(self):
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        os.environ["GROQ_BASE_URL"] = "https://gateway.invalid/v1"
        with mock.patch.object(llm, "Groq", return_value=object()) as groq_cls:
            llm.get_client()
        _, kwargs = groq_cls.call_args
        self.assertEqual(kwargs["base_url"], "https://gateway.invalid/v1")

    def test_missing_sdk_raises_unavailable(self):
        with mock.patch.object(llm, "Groq", None):
            with self.assertRaises(llm.LLMUnavailable):
                llm._create_client("gsk_mock", 30.0)


class ChatCompletionTests(_EnvIsolated):
    def _fake_client(self, side_effect=None, content="Hello from Groq"):
        completion = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )
        create = mock.Mock(return_value=completion, side_effect=side_effect)
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        return client, create

    def test_successful_response_is_returned(self):
        client, create = self._fake_client()
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            result = llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertEqual(result, "Hello from Groq")
        _, kwargs = create.call_args
        self.assertEqual(kwargs["model"], llm.DEFAULT_GROQ_MODEL)
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hi"}])

    def test_explicit_model_override_is_forwarded(self):
        client, create = self._fake_client()
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            llm.chat_completion([{"role": "user", "content": "hi"}], model="openai/gpt-oss-20b")
        _, kwargs = create.call_args
        self.assertEqual(kwargs["model"], "openai/gpt-oss-20b")

    def test_authentication_error_is_reported_clearly(self):
        client, _ = self._fake_client(side_effect=_status_error(groq.AuthenticationError, 401))
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertIn("401", str(ctx.exception))

    def test_rate_limit_error_is_reported_clearly(self):
        client, _ = self._fake_client(side_effect=_status_error(groq.RateLimitError, 429))
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertIn("429", str(ctx.exception))

    def test_timeout_is_reported_clearly(self):
        client, _ = self._fake_client(side_effect=groq.APITimeoutError(request=_request()))
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertIn("timed out", str(ctx.exception))

    def test_connection_error_is_reported_clearly(self):
        client, _ = self._fake_client(side_effect=groq.APIConnectionError(request=_request()))
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertIn("connection", str(ctx.exception).lower())

    def test_invalid_model_error_mentions_the_model(self):
        os.environ["GROQ_API_KEY"] = "gsk_mock"
        os.environ["GROQ_MODEL"] = "not-a-real-model"
        client, _ = self._fake_client(side_effect=_status_error(groq.BadRequestError, 400))
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        message = str(ctx.exception)
        self.assertIn("not-a-real-model", message)
        self.assertIn("400", message)

    def test_error_message_does_not_leak_the_key(self):
        os.environ["GROQ_API_KEY"] = "gsk_top_secret_value"
        client, _ = self._fake_client(side_effect=_status_error(groq.AuthenticationError, 401))
        with mock.patch.object(llm, "_create_client", return_value=client):
            with self.assertRaises(llm.LLMUnavailable) as ctx:
                llm.chat_completion([{"role": "user", "content": "hi"}])
        self.assertNotIn("gsk_top_secret_value", str(ctx.exception))


class ChatbotFallbackTests(_EnvIsolated):
    def _selected(self):
        return {"Title": "Interstellar"}

    def _recs(self):
        return [
            {"Title": "Gravity", "Release_Date": "2013-10-04", "Vote_Average": 7.7,
             "Genre": "Science Fiction", "Semantic_Similarity": 0.81},
            {"Title": "The Martian", "Release_Date": "2015-09-30", "Vote_Average": 7.6,
             "Genre": "Science Fiction", "Semantic_Similarity": 0.79},
        ]

    def test_summary_uses_local_fallback_without_key(self):
        reply = chatbot._ai_summarize("movies like Interstellar", self._selected(), self._recs())
        self.assertIn("Gravity", reply)
        self.assertIn("The Martian", reply)

    def test_summary_falls_back_when_groq_unavailable(self):
        with mock.patch.object(llm, "is_configured", return_value=True), \
             mock.patch.object(llm, "chat_completion",
                               side_effect=llm.LLMUnavailable("rate limit")):
            with mock.patch("builtins.print"):
                reply = chatbot._ai_summarize("movies like Interstellar", self._selected(), self._recs())
        self.assertIn("Gravity", reply)

    def test_summary_returns_groq_text_when_available(self):
        with mock.patch.object(llm, "is_configured", return_value=True), \
             mock.patch.object(llm, "chat_completion", return_value="Groq says hi"):
            reply = chatbot._ai_summarize("movies like Interstellar", self._selected(), self._recs())
        self.assertEqual(reply, "Groq says hi")

    def test_general_reply_fallback_without_key(self):
        reply = chatbot._ai_general_reply("hello")
        self.assertIn("movie", reply.lower())

    def test_chat_still_uses_local_engine_and_groq_for_wording(self):
        with mock.patch.object(chatbot.engine, "get_recommended_movie_details",
                               return_value=(self._selected(), self._recs())) as local, \
             mock.patch.object(llm, "is_configured", return_value=True), \
             mock.patch.object(llm, "chat_completion", return_value="Groq phrasing") as completion:
            result = chatbot.chat("Suggest 2 movies like Interstellar")
        local.assert_called_once()
        completion.assert_called_once()
        self.assertEqual(result["reply"], "Groq phrasing")
        self.assertEqual(result["source_movie"]["Title"], "Interstellar")
        self.assertEqual(len(result["recommendations"]), 2)

    def test_chat_never_invents_movies_when_groq_fails(self):
        with mock.patch.object(chatbot.engine, "get_recommended_movie_details",
                               return_value=(self._selected(), self._recs())), \
             mock.patch.object(llm, "is_configured", return_value=True), \
             mock.patch.object(llm, "chat_completion",
                               side_effect=llm.LLMUnavailable("boom")), \
             mock.patch("builtins.print"):
            result = chatbot.chat("Suggest 2 movies like Interstellar")
        titles = {r["Title"] for r in result["recommendations"]}
        self.assertEqual(titles, {"Gravity", "The Martian"})
        self.assertIn("Gravity", result["reply"])


if __name__ == "__main__":
    unittest.main()
