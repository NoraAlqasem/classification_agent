"""Convert the original readable LLM log without discarding its contents."""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any


IMPORTED_CONSOLE_MARKER = (
    "Imported console excerpt supplied by the user (original request details unavailable)"
)
_EVENTS = re.compile(
    r"(?m)(?:\A|\n)(?:"
    r"(?P<request>={80}\nRequest ID: (?P<request_id>[^\n]*)(?:\n|$))|"
    r"(?P<imported>={80}\n" + re.escape(IMPORTED_CONSOLE_MARKER) + r"(?:\n|$))|"
    r"(?P<completion>--- LLM (?P<kind>response|error) \| Request ID: "
    r"(?P<completion_id>[^\n]*?) \| Stage: (?P<completion_stage>[^\n]*?) ---)(?:\n|$))"
)
_ROLES = re.compile(r"(?m)^\[(system|human|ai|user|assistant|developer|tool|function|chat|message|remove)\]\n")
_FENCED_JSON = re.compile(r"\A```(?:json)?\s*\n(?P<body>[\s\S]*?)\n```\Z", re.IGNORECASE)
_ROLE_ALIASES = {"human": "user", "ai": "assistant"}


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"Nonfinite JSON constant: {value}")


def _finite_float(value: str) -> float:
    decoded = float(value)
    if not math.isfinite(decoded):
        raise ValueError(f"Nonfinite JSON number: {value}")
    return decoded


def structured_content(content: Any) -> Any:
    """Decode an entire JSON object/list or a fenced one; retain other content."""
    if not isinstance(content, str):
        return content
    candidate = content.strip()
    fenced = _FENCED_JSON.fullmatch(candidate)
    if fenced is not None:
        candidate = fenced.group("body")
    try:
        decoded = json.loads(candidate, parse_constant=_reject_nonfinite, parse_float=_finite_float)
    except (ValueError, TypeError, RecursionError):
        return content
    return decoded if isinstance(decoded, (dict, list)) else content


def _stable_id(kind: str, content: str, position: int) -> str:
    digest = hashlib.sha256(f"{kind}:{position}:{content}".encode("utf-8")).hexdigest()[:24]
    return f"legacy-{kind}-{digest}"


def _record(request_id: str, *, stage: str = "unknown", status: str = "pending") -> dict:
    # Keep the useful result fields first in each grouped JSON request.
    return {
        "stage": stage, "attempt": None, "status": status, "response": None,
        "error": None, "request_id": request_id, "started_at": None,
        "finished_at": None, "session": None, "model": None,
        "request": {"messages": []},
    }


def _without_writer_newline(text: str) -> str:
    return text[:-1] if text.endswith("\n") else text


def _metadata(text: str) -> tuple[dict[str, str], str]:
    fields, unparsed = {}, []
    known = {"Started (UTC)", "Finished (UTC)", "Saved (UTC)", "Stage", "Attempt", "Session", "Model"}
    for line in text.splitlines(keepends=True):
        name, separator, value = line.rstrip("\r\n").partition(": ")
        if separator and name in known:
            fields[name] = value
        elif line.strip():
            unparsed.append(line)
    return fields, "".join(unparsed)


def _messages(prompt: str) -> list[dict[str, Any]]:
    markers = list(_ROLES.finditer(prompt))
    if not markers:
        return [{"role": "message", "content": structured_content(prompt)}] if prompt else []
    messages = []
    prefix = prompt[:markers[0].start()]
    if prefix:
        messages.append({"role": "message", "content": structured_content(prefix)})
    for index, marker in enumerate(markers):
        stop = markers[index + 1].start() if index + 1 < len(markers) else len(prompt)
        content = prompt[marker.end():stop]
        # Two newlines separate legacy messages; any earlier whitespace belongs
        # to the original message and must survive conversion.
        if index + 1 < len(markers) and content.endswith("\n\n"):
            content = content[:-2]
        role = marker.group(1)
        messages.append({"role": _ROLE_ALIASES.get(role, role), "content": structured_content(content)})
    return messages


def _request_record(event: re.Match, body: str, position: int) -> dict:
    request_id = event.group("request_id") or _stable_id("request", body, position)
    record = _record(request_id)
    marker = re.search(r"(?m)^Request:(?:\n|$)", body)
    metadata_text = body[:marker.start()] if marker else body
    fields, unparsed = _metadata(metadata_text)
    record.update({
        "stage": fields.get("Stage", "unknown"),
        "started_at": fields.get("Started (UTC)"),
        "session": fields.get("Session"), "model": fields.get("Model"),
    })
    attempt = fields.get("Attempt")
    if attempt is not None:
        try:
            record["attempt"] = int(attempt)
        except ValueError:
            record["attempt"] = attempt
    prompt = _without_writer_newline(body[marker.end():]) if marker else ""
    if unparsed:
        record["request"]["messages"].append({"role": "message", "content": unparsed})
    record["request"]["messages"].extend(_messages(prompt))
    return record


def _completion(record: dict, event: re.Match, body: str) -> None:
    marker_name = "Response" if event.group("kind") == "response" else "Error"
    marker = re.search(rf"(?m)^{marker_name}:(?:\n| )", body)
    metadata_text = body[:marker.start()] if marker else body
    fields, unparsed = _metadata(metadata_text)
    record["finished_at"] = fields.get("Finished (UTC)")
    if event.group("kind") == "response":
        raw = _without_writer_newline(body[marker.end():]) if marker else ""
        if not marker and unparsed:
            raw = _without_writer_newline(unparsed)
        record.update({"status": "success", "response": structured_content(raw),
                       "raw_response": raw, "error": None})
    else:
        raw = _without_writer_newline(body[marker.end():]) if marker else _without_writer_newline(unparsed)
        error_type, separator, message = raw.partition(": ")
        if not separator:
            error_type, message = "UnknownError", raw
        status = "interrupted" if error_type in {"KeyboardInterrupt", "SystemExit", "InterruptedError"} else "error"
        record.update({"status": status, "response": None,
                       "error": {"type": error_type, "message": message}})


def _imported_record(body: str, position: int) -> dict:
    metadata_text, separator, excerpt = body.partition("\n\n")
    if not separator:
        metadata_text, excerpt = "", body
    fields, unparsed = _metadata(metadata_text)
    raw = _without_writer_newline(excerpt)
    if unparsed:
        raw = unparsed + raw
    record = _record(_stable_id("imported", body, position),
                     stage=fields.get("Stage", "unknown"), status="imported")
    record.update({"finished_at": fields.get("Saved (UTC)"),
                   "response": structured_content(raw), "raw_response": raw})
    return record


def convert_legacy_log(text: str) -> list[dict]:
    """Group start/completion entries by request ID, preserving incomplete data."""
    events = list(_EVENTS.finditer(text))
    records, pending = [], {}
    prefix = text[:events[0].start()] if events else text
    if prefix.strip():
        record = _record(_stable_id("unparsed", prefix, 0), status="unparsed")
        record.update({"response": structured_content(prefix), "raw_response": prefix})
        records.append(record)
    for position, event in enumerate(events):
        stop = events[position + 1].start() if position + 1 < len(events) else len(text)
        body = text[event.end():stop]
        if event.group("request"):
            record = _request_record(event, body, position)
            records.append(record)
            pending.setdefault(record["request_id"], []).append(record)
        elif event.group("imported"):
            records.append(_imported_record(body, position))
        else:
            request_id = event.group("completion_id") or _stable_id("orphan", body, position)
            waiting = pending.get(request_id, [])
            if waiting:
                record = waiting.pop(0)
            else:
                record = _record(request_id, stage=event.group("completion_stage"))
                records.append(record)
            _completion(record, event, body)
    return records
