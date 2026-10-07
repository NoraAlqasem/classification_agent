"""Recover from temporary provider quotas with bounded, deterministic waits."""
from __future__ import annotations

import os
import unittest
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import format_datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from openai import APIStatusError, RateLimitError

from agent import llm_retry


def rate_limit_error(message="Rate limit reached. Please try again in 4.26s.", *,
                     headers=None, code="rate_limit_exceeded", error_type="tokens",
                     url="https://example.invalid/v1/chat/completions"):
    request = httpx.Request("POST", url)
    body = {"error": {"message": message, "code": code, "type": error_type}}
    return RateLimitError(message, response=httpx.Response(429, request=request,
                                                          headers=headers or {}), body=body)


class FakeClock:
    def __init__(self):
        self.elapsed = 0.0
        self.wall = 1_700_000_000.0
        self.delays = []

    def monotonic(self):
        return self.elapsed

    def time(self):
        return self.wall + self.elapsed

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.elapsed += seconds


class LLMRetryTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.messages = [HumanMessage(content="Propose a training configuration.")]
        self.response = AIMessage(content='{"updated_config":{"epochs":30},"rationale":"Converge."}')
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {}, clear=True))
        stack.enter_context(patch.object(llm_retry.time, "monotonic", side_effect=self.clock.monotonic))
        stack.enter_context(patch.object(llm_retry.time, "time", side_effect=self.clock.time))
        self.sleep = stack.enter_context(patch.object(llm_retry.time, "sleep", side_effect=self.clock.sleep))
        self.console = stack.enter_context(patch.object(llm_retry.console, "print"))

    def invoke(self, *results):
        invoke = Mock(side_effect=results)
        client = SimpleNamespace(invoke=invoke)
        response = llm_retry.invoke_with_rate_limit_retry(client, self.messages)
        return response, invoke

    def test_success_returns_original_response_without_waiting(self):
        response, invoke = self.invoke(self.response)
        self.assertIs(response, self.response)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_user_groq_message_retries_after_suggested_delay(self):
        message = ("Rate limit reached for model qwen/qwen3.8-27b on input tokens per minute "
                   "(ITPM): Limit 7000, Used 3178, Requested 4319. Please try again in 4.26s.")
        response, invoke = self.invoke(rate_limit_error(message), self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertGreaterEqual(self.clock.delays[0], 4.26)
        self.assertLessEqual(self.clock.delays[0], 5.26)
        self.assertTrue(all(call.args[0] is self.messages for call in invoke.call_args_list))
        self.console.assert_called()

    def test_retry_after_numeric_header_takes_precedence_over_message(self):
        response, invoke = self.invoke(rate_limit_error(headers={"Retry-After": "7.5"}), self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertGreaterEqual(self.clock.delays[0], 7.5)
        self.assertLessEqual(self.clock.delays[0], 8.5)

    def test_retry_after_http_date_uses_current_wall_time(self):
        retry_at = format_datetime(datetime.fromtimestamp(self.clock.wall + 12, timezone.utc), usegmt=True)
        response, invoke = self.invoke(rate_limit_error(headers={"retry-after": retry_at}), self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertGreaterEqual(self.clock.delays[0], 12)
        self.assertLessEqual(self.clock.delays[0], 13)

    def test_past_retry_date_never_causes_a_negative_sleep(self):
        retry_at = format_datetime(datetime.fromtimestamp(self.clock.wall - 10, timezone.utc), usegmt=True)
        response, invoke = self.invoke(rate_limit_error(headers={"retry-after": retry_at}), self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(self.clock.delays, [1.0])

    def test_provider_compound_duration_is_parsed_in_full(self):
        error = rate_limit_error("Minute limit reached. Please try again in 1m4.26s.")
        with patch.dict(os.environ, {"LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "120"}):
            response, invoke = self.invoke(error, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertAlmostEqual(self.clock.delays[0], 65.26)

    def test_millisecond_retry_header_is_supported(self):
        response, invoke = self.invoke(rate_limit_error("Temporary minute limit", headers={"Retry-After-Ms": "4250"}),
                                       self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertAlmostEqual(self.clock.delays[0], 5.25)

    def test_token_reset_duration_is_used_when_no_retry_hint_is_available(self):
        error = rate_limit_error("Temporary minute limit", headers={"x-ratelimit-reset-tokens": "2.5s"})
        response, invoke = self.invoke(error, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertAlmostEqual(self.clock.delays[0], 3.5)

    def test_missing_wait_hint_uses_bounded_exponential_fallback(self):
        error = rate_limit_error("Transient tokens per minute limit")
        response, invoke = self.invoke(error, error, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(self.clock.delays, [5.0, 10.0])

    def test_invalid_retry_header_falls_back_to_provider_message(self):
        for value in ("nonsense", "nan", "inf", "-5"):
            with self.subTest(header=value):
                self.clock.delays.clear()
                response, invoke = self.invoke(rate_limit_error(headers={"Retry-After": value}), self.response)
                self.assertIs(response, self.response)
                self.assertEqual(invoke.call_count, 2)
                self.assertGreaterEqual(self.clock.delays[0], 4.26)
                self.assertLessEqual(self.clock.delays[0], 5.26)

    def test_zero_remaining_tokens_is_transient_and_may_retry(self):
        response, invoke = self.invoke(rate_limit_error(headers={"x-ratelimit-remaining-tokens": "0"}), self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(len(self.clock.delays), 1)

    def test_default_budget_allows_only_two_additional_requests(self):
        errors = [rate_limit_error("Transient minute limit") for _ in range(3)]
        invoke = Mock(side_effect=[*errors, self.response])
        with self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, errors[-1])
        self.assertEqual(invoke.call_count, 3)
        self.assertEqual(self.clock.delays, [5.0, 10.0])

    def test_zero_retries_disables_retry_without_changing_original_error(self):
        error = rate_limit_error()
        invoke = Mock(side_effect=error)
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "0"}), \
             self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, error)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_custom_retry_count_is_total_additional_requests(self):
        error = rate_limit_error("Minute limit; retry shortly")
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "3"}):
            response, invoke = self.invoke(error, error, error, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 4)
        self.assertEqual(self.clock.delays, [5.0, 10.0, 20.0])

    def test_retry_wait_longer_than_budget_does_not_sleep_or_send_again(self):
        error = rate_limit_error(headers={"Retry-After": "61"})
        invoke = Mock(side_effect=[error, self.response])
        with self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, error)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_cumulative_wait_budget_stops_before_next_delay(self):
        errors = [rate_limit_error("Transient minute limit") for _ in range(3)]
        invoke = Mock(side_effect=[*errors, self.response])
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "5", "LLM_RATE_LIMIT_MAX_WAIT_SECONDS": "12"}), \
             self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, errors[1])
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(self.clock.delays, [5.0])

    def test_zero_remaining_requests_is_not_retried(self):
        error = rate_limit_error(headers={"x-ratelimit-remaining-requests": "0"},
                                 url="https://api.groq.com/openai/v1/chat/completions")
        invoke = Mock(side_effect=[error, self.response])
        with self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, error)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_zero_remaining_requests_on_another_provider_can_be_transient(self):
        error = rate_limit_error(headers={"x-ratelimit-remaining-requests": "0", "retry-after": "3"},
                                 url="https://api.openai.com/v1/chat/completions")
        response, invoke = self.invoke(error, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(self.clock.delays, [4.0])

    def test_langchain_wrapped_rate_limit_error_from_user_log_is_retried(self):
        from langchain_openai.chat_models.base import OpenAIRateLimitError
        error = rate_limit_error(url="https://api.groq.com/openai/v1/chat/completions")
        wrapped = OpenAIRateLimitError(message=error.message, response=error.response, body=error.body)
        response, invoke = self.invoke(wrapped, self.response)
        self.assertIs(response, self.response)
        self.assertEqual(invoke.call_count, 2)
        self.assertEqual(self.clock.delays, [5.26])

    def test_single_request_exceeding_minute_limit_is_not_retried(self):
        error = rate_limit_error("Input tokens per minute: Limit 7000, Used 0, Requested 8891. "
                                 "Please try again in 4.26s.")
        invoke = Mock(side_effect=[error, self.response])
        with self.assertRaises(RateLimitError) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, error)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_permanent_quota_and_daily_limits_are_not_retried(self):
        errors = [
            rate_limit_error("You exceeded your current quota", code="insufficient_quota"),
            rate_limit_error("Daily tokens limit reached; try again tomorrow"),
            rate_limit_error("Requests per day (RPD): Limit 1000; try again in 5s."),
            rate_limit_error("Tokens per day (TPD): Limit 10000; try again in 5s."),
        ]
        for error in errors:
            with self.subTest(message=str(error)):
                invoke = Mock(side_effect=error)
                with self.assertRaises(RateLimitError) as raised:
                    llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
                self.assertIs(raised.exception, error)
                invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_other_http_errors_and_string_429_are_not_retried(self):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        errors = [
            APIStatusError("Request too large. Please try again in 4s.",
                           response=httpx.Response(413, request=request), body={"error": {"code": "rate_limit_exceeded"}}),
            APIStatusError("Internal server error", response=httpx.Response(500, request=request), body=None),
            RuntimeError("429: provider temporary rate limit"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                invoke = Mock(side_effect=error)
                with self.assertRaises(type(error)) as raised:
                    llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
                self.assertIs(raised.exception, error)
                invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_malformed_response_is_returned_once_for_callers_to_validate(self):
        malformed = AIMessage(content="{ invalid JSON")
        response, invoke = self.invoke(malformed)
        self.assertIs(response, malformed)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()

    def test_keyboard_interrupt_during_wait_propagates_without_another_request(self):
        error = rate_limit_error()
        interrupt = KeyboardInterrupt("User stopped the rate-limit wait")
        invoke = Mock(side_effect=[error, self.response])
        self.sleep.side_effect = interrupt
        with self.assertRaises(KeyboardInterrupt) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, interrupt)
        invoke.assert_called_once_with(self.messages)

    def test_keyboard_interrupt_during_request_is_never_retried(self):
        interrupt = KeyboardInterrupt("User stopped the provider request")
        invoke = Mock(side_effect=interrupt)
        with self.assertRaises(KeyboardInterrupt) as raised:
            llm_retry.invoke_with_rate_limit_retry(SimpleNamespace(invoke=invoke), self.messages)
        self.assertIs(raised.exception, interrupt)
        invoke.assert_called_once_with(self.messages)
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
