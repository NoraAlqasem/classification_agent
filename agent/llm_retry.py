"""Retry temporary HTTP 429 responses without multiplying SDK retries."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from email.utils import parsedate_to_datetime
import math
import os
import re
import time
from typing import Any

from rich.console import Console

console = Console()
_QUOTA_CODES = frozenset({
    "insufficient_quota", "quota_exceeded", "billing_hard_limit_reached",
    "billing_not_active", "payment_required",
})
_DURATION = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|seconds?|minutes?|hours?|days?|s|m|h|d)", re.I)
_UNITS = {"ms": .001, "s": 1, "second": 1, "seconds": 1,
          "m": 60, "minute": 60, "minutes": 60,
          "h": 3600, "hour": 3600, "hours": 3600,
          "d": 86400, "day": 86400, "days": 86400}


def get_retry_settings() -> tuple[int, float]:
    """Limit extra HTTP attempts and their total planned waiting time."""
    try:
        retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_MAX_RETRIES must be a nonnegative integer") from exc
    if retries < 0:
        raise ValueError("LLM_MAX_RETRIES must be a nonnegative integer")
    try:
        budget = float(os.getenv("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", "60"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_RATE_LIMIT_MAX_WAIT_SECONDS must be finite and positive") from exc
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("LLM_RATE_LIMIT_MAX_WAIT_SECONDS must be finite and positive")
    return retries, budget


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _duration_seconds(value: str) -> float | None:
    """Parse provider reset durations such as 4.26s or 1m2.5s."""
    value = value.strip()
    if not value:
        return None
    matches = list(_DURATION.finditer(value))
    if not matches or _DURATION.sub("", value).strip():
        return None
    seconds = sum(float(match[1]) * _UNITS[match[2].lower()] for match in matches)
    return seconds if math.isfinite(seconds) else None


def _headers(error: Exception) -> dict[str, str]:
    headers = getattr(getattr(error, "response", None), "headers", None)
    return {str(key).lower(): str(value) for key, value in headers.items()} if isinstance(headers, Mapping) else {}


def _message(error: Exception) -> str:
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict) and isinstance(details.get("message"), str):
            return details["message"]
    return str(error)


def _is_groq_response(error: Exception) -> bool:
    request = getattr(error, "request", None)
    if request is None:
        try:
            request = getattr(getattr(error, "response", None), "request", None)
        except RuntimeError:
            return False
    host = getattr(getattr(request, "url", None), "host", None)
    return isinstance(host, str) and host.lower() == "api.groq.com"


def _transient_rate_limit(error: Exception, headers: dict[str, str]) -> bool:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    if status != 429:
        return False
    body = getattr(error, "body", None)
    details = body.get("error", body) if isinstance(body, dict) else {}
    codes = [getattr(error, "code", None)]
    if isinstance(details, dict):
        codes.extend((details.get("code"), details.get("type")))
    if any(str(code).lower() in _QUOTA_CODES for code in codes):
        return False
    # Groq's request header refers to the daily allowance, not the minute rate.
    if _is_groq_response(error) and headers.get("x-ratelimit-remaining-requests", "").strip() == "0":
        return False
    message = _message(error)
    if any(marker in message.lower() for marker in (
        "tokens per day", "requests per day", "daily limit", "daily quota",
        "daily token", "daily request", "insufficient_quota",
    )):
        return False
    limit = re.search(r"\bLimit\s*:?\s*([\d,]+)", message, re.I)
    requested = re.search(r"\bRequested\s*:?\s*([\d,]+)", message, re.I)
    if limit and requested:
        if int(requested[1].replace(",", "")) > int(limit[1].replace(",", "")):
            return False
    return True


def _provider_delay(error: Exception, headers: dict[str, str]) -> float | None:
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        if (seconds := _number(retry_after)) is not None:
            return seconds
        try:
            date = parsedate_to_datetime(retry_after)
            if date.tzinfo is not None:
                return max(0.0, date.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            pass
    if (milliseconds := _number(headers.get("retry-after-ms"))) is not None:
        return milliseconds / 1000
    # The OpenAI-compatible SDK also includes Groq's wait hint in the error body.
    match = re.search(r"try again in\s+([\d.]+\s*[a-z]+(?:\s*[\d.]+\s*[a-z]+)*)", _message(error), re.I)
    if match and (seconds := _duration_seconds(match[1])) is not None:
        return seconds
    reset = headers.get("x-ratelimit-reset-tokens")
    return _duration_seconds(reset) if reset else None


def invoke_with_rate_limit_retry(llm: Any, messages: Sequence[Any]) -> Any:
    """Retry the same analysis, leaving permanent failures and interrupts intact."""
    retries, budget = get_retry_settings()
    waited = 0.0
    for attempt in range(retries + 1):
        try:
            return llm.invoke(messages)
        except Exception as error:
            headers = _headers(error)
            if attempt >= retries or not _transient_rate_limit(error, headers):
                raise
            delay = _provider_delay(error, headers)
            # A small margin avoids retrying just before a provider window resets.
            delay = delay + 1.0 if delay is not None else min(5.0 * (2 ** min(attempt, 4)), 60.0)
            if delay > budget - waited:
                console.print("LLM rate limit requires more waiting than the configured budget; stopping retries.",
                              style="yellow", markup=False, highlight=False)
                raise
            console.print(f"LLM rate limit reached; waiting {delay:.2f}s before retry {attempt + 1}/{retries}.",
                          style="yellow", markup=False, highlight=False)
            time.sleep(delay)
            waited += delay
