"""Shared LLM construction and defensive JSON parsing."""
from __future__ import annotations

import json
import math
import os
from typing import Any

from agent.llm_retry import get_retry_settings


def get_llm():
    try:
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "512"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_MAX_TOKENS must be a positive integer") from exc
    if max_tokens <= 0:
        raise ValueError("LLM_MAX_TOKENS must be a positive integer")
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except (TypeError, ValueError) as exc:
        raise ValueError("LLM_TIMEOUT_SECONDS must be finite and positive") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be finite and positive")
    get_retry_settings()

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        # The shared invocation helper owns the total retry count and wait budget.
        max_retries=0,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def parse_json_response(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    if not isinstance(content, str):
        raise ValueError("Expected a text JSON response")
    raw = content.strip()
    if raw.startswith(chr(96) * 3):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == chr(96) * 3 else lines[1:])
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result
