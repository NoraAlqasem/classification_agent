"""Save only evaluation and optimization responses in each session's JSON."""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from rich.console import Console

from agent.llm_log_migration import structured_content
from agent.llm_retry import invoke_with_rate_limit_retry
from agent.session_log_format import sanitize_session_log
from memory.session_paths import allocate_session_directory

DEFAULT_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs" / "resnet"
console = Console()
_ALLOWED_STAGES = frozenset({"evaluation", "optimization"})


def session_log_path(state: Mapping[str, Any]) -> Path:
    """Use the active session or reserve a numbered session when necessary."""
    config = state.get("current_config") or {}
    session = state.get("session_dir") or config.get("out")
    if not session:
        session = str(allocate_session_directory(DEFAULT_RUNS_DIR))
        if isinstance(state, MutableMapping):
            state["session_dir"] = session
    return Path(session) / "llm_responses.json"


def _read_document(path: Path, model: str) -> dict[str, Any]:
    if not path.exists():
        return {"session": {"session_id": path.parent.name, "model": model, "experiments": {}}}
    document = sanitize_session_log(json.loads(path.read_text(encoding="utf-8")))
    if document["session"]["session_id"] != path.parent.name:
        raise ValueError("Existing LLM JSON log belongs to a different session")
    return document


def _pending_nonces(lock) -> dict[str, str]:
    lock.seek(0)
    raw = lock.read()
    if not raw:
        return {}
    pending = json.loads(raw)
    if not isinstance(pending, dict) or any(not isinstance(key, str) or not isinstance(value, str)
                                            for key, value in pending.items()):
        raise ValueError("Invalid temporary LLM request state")
    return pending


def _save_nonces(lock, pending: dict[str, str]) -> None:
    lock.seek(0)
    lock.truncate()
    if pending:
        json.dump(pending, lock)
    lock.flush()


def _write_record(record: dict[str, Any], path: Path) -> None:
    """Atomically save the newest response; lock state holds temporary nonces only."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        with lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            data = _read_document(path, record["model"])
            pending = _pending_nonces(lock)
            experiment_name = f"exp_{record['attempt']:02d}"
            key = f"{experiment_name}:{record['stage']}"
            if record["status"] == "pending":
                pending[key] = record["request_id"]
                _save_nonces(lock, pending)
                data["session"]["model"] = record["model"]
            elif pending.get(key) != record["request_id"]:
                return
            experiment = data["session"]["experiments"].setdefault(
                experiment_name, {"attempt": record["attempt"], "stages": {}},
            )
            experiment["stages"][record["stage"]] = {
                "status": record["status"], "response": record["response"],
            }
            data = sanitize_session_log(data)
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False, default=str)
                handle.write("\n")
            os.replace(temporary, path)
            temporary = None
            if record["status"] != "pending":
                pending.pop(key, None)
                _save_nonces(lock, pending)
    except (OSError, ValueError, TypeError) as exc:
        console.print(f"Unable to save LLM log at {path}: {exc}",
                      style="yellow", markup=False, highlight=False)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def invoke_logged_llm(messages: Sequence[Any], *, state: Mapping[str, Any],
                      stage: str, llm_factory: Callable[[], Any]) -> Any:
    """Record status and parsed response only; report requests are not logged."""
    attempt = state.get("attempt_number", 0)
    if stage not in _ALLOWED_STAGES or type(attempt) is not int or attempt < 1:
        return invoke_with_rate_limit_retry(llm_factory(), messages)
    path = None
    try:
        path = session_log_path(state)
    except (OSError, ValueError, TypeError) as exc:
        console.print(f"Unable to prepare LLM session log: {exc}",
                      style="yellow", markup=False, highlight=False)
    record = {
        "stage": stage, "attempt": attempt, "status": "pending", "response": None,
        "request_id": uuid4().hex, "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
    }
    if path is not None:
        _write_record(record, path)
    try:
        response = invoke_with_rate_limit_retry(llm_factory(), messages)
    except BaseException as exc:
        record.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error")
        if path is not None:
            _write_record(record, path)
        raise
    record.update(status="success", response=structured_content(response.content))
    if path is not None:
        _write_record(record, path)
    return response
