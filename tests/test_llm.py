"""Verify output token limits without contacting an LLM provider."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from agent.llm import get_llm


class LLMTokenBudgetTests(unittest.TestCase):
    def test_default_output_limit_preserves_existing_settings(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("langchain_openai.ChatOpenAI") as constructor:
            result = get_llm()
        constructor.assert_called_once_with(
            model="gpt-4o", temperature=0.1, max_retries=0, max_tokens=512, timeout=30.0,
        )
        self.assertIs(result, constructor.return_value)

    def test_custom_output_limit_preserves_provider_settings(self):
        settings = {
            "OPENAI_MODEL": "qwen/qwen3.8-27b",
            "OPENAI_API_BASE": "https://api.groq.com/openai/v1",
            "LLM_TEMPERATURE": "0.2",
            "LLM_MAX_RETRIES": "4",
            "LLM_MAX_TOKENS": "256",
            "LLM_TIMEOUT_SECONDS": "45",
        }
        with patch.dict(os.environ, settings, clear=True), \
             patch("langchain_openai.ChatOpenAI") as constructor:
            get_llm()
        constructor.assert_called_once_with(
            model="qwen/qwen3.8-27b", temperature=0.2, max_retries=0,
            max_tokens=256, timeout=45.0,
        )

    def test_invalid_output_limits_fail_before_client_construction(self):
        for value in ("", " ", "not-a-number", "12.5", "0", "-1"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"LLM_MAX_TOKENS": value}, clear=True), \
                 patch("langchain_openai.ChatOpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "LLM_MAX_TOKENS must be a positive integer"):
                    get_llm()
                constructor.assert_not_called()

    def test_real_client_request_payload_includes_completion_cap(self):
        # Constructing a client and building its payload perform no HTTP requests.
        settings = {
            "OPENAI_API_KEY": "dummy-test-key",
            "OPENAI_API_BASE": "https://example.invalid/openai/v1",
            "OPENAI_MODEL": "qwen/qwen3.8-27b",
            "LLM_MAX_TOKENS": "256",
        }
        with patch.dict(os.environ, settings, clear=True):
            client = get_llm()
            payload = client._get_request_payload([HumanMessage(content="Return concise JSON.")])
        self.assertEqual(payload["max_completion_tokens"], 256)
        self.assertEqual(payload["model"], settings["OPENAI_MODEL"])
        self.assertEqual(client.request_timeout, 30.0)
        self.assertEqual(client.root_client.timeout, 30.0)
        self.assertEqual(client.root_client.max_retries, 0)
        self.assertNotIn("tools", payload)

    def test_invalid_timeout_rejected_before_constructing_client(self):
        for value in ("", "zero", "0", "-1", "nan", "inf"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"LLM_TIMEOUT_SECONDS": value}, clear=True), \
                 patch("langchain_openai.ChatOpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "LLM_TIMEOUT_SECONDS must be finite and positive"):
                    get_llm()
                constructor.assert_not_called()

    def test_shared_retry_budget_is_validated_without_enabling_sdk_retries(self):
        settings = {"LLM_MAX_RETRIES": "3", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "20"}
        with patch.dict(os.environ, settings, clear=True), \
             patch("langchain_openai.ChatOpenAI") as constructor:
            get_llm()
        self.assertEqual(constructor.call_args.kwargs["max_retries"], 0)

    def test_invalid_retry_wait_budget_rejected_before_constructing_client(self):
        for value in ("", "none", "0", "-1", "nan", "inf"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"LLM_RATE_LIMIT_MAX_WAIT_SECONDS": value}, clear=True), \
                 patch("langchain_openai.ChatOpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "LLM_RATE_LIMIT_MAX_WAIT_SECONDS"):
                    get_llm()
                constructor.assert_not_called()

    def test_invalid_retry_count_rejected_before_constructing_client(self):
        for value in ("", "many", "1.5", "-1"):
            with self.subTest(value=value), \
                 patch.dict(os.environ, {"LLM_MAX_RETRIES": value}, clear=True), \
                 patch("langchain_openai.ChatOpenAI") as constructor:
                with self.assertRaisesRegex(ValueError, "LLM_MAX_RETRIES must be a nonnegative integer"):
                    get_llm()
                constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
