"""Project session LLM responses onto the requested minimal JSON structure."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ALLOWED_STAGES = frozenset({"evaluation", "optimization"})
_RESPONSE_FIELDS = {
    "evaluation": frozenset({"verdict", "target_reached", "issues", "recommendations", "reasoning"}),
    "optimization": frozenset({"updated_config", "rationale"}),
}


def _response(stage: str, value: Any) -> Any:
    if isinstance(value, dict):
        return {key: deepcopy(item) for key, item in value.items() if key in _RESPONSE_FIELDS[stage]}
    return deepcopy(value)


def _request_time(record: dict[str, Any]) -> float | None:
    for key in ("started_at", "finished_at"):
        value = record.get(key)
        if isinstance(value, str):
            try:
                stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return stamp.replace(tzinfo=stamp.tzinfo or timezone.utc).timestamp()
            except (ValueError, OverflowError):
                pass
    return None


def session_log_document(records: list[dict[str, Any]], session_dir: str | Path) -> dict[str, Any]:
    """Keep only the latest evaluation and optimization response per attempt."""
    session = {"session_id": Path(session_dir).name, "model": None, "experiments": {}}
    latest_model_key = None
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("LLM request records must be objects")
        if record.get("model"):
            stamp = _request_time(record)
            model_key = (stamp is not None, stamp or 0.0, position)
            if latest_model_key is None or model_key > latest_model_key:
                session["model"] = record["model"]
                latest_model_key = model_key
        attempt, stage = record.get("attempt"), record.get("stage")
        if type(attempt) is not int or attempt < 1 or stage not in _ALLOWED_STAGES:
            continue
        name = f"exp_{attempt:02d}"
        experiment = session["experiments"].setdefault(name, {"attempt": attempt, "stages": {}})
        experiment["stages"][stage] = {
            "status": record.get("status"), "response": _response(stage, record.get("response")),
        }
    return {"session": session}


def sanitize_session_log(document: dict[str, Any]) -> dict[str, Any]:
    """Remove all fields beyond the session, experiment and response allowlist."""
    if not isinstance(document, dict) or not isinstance(document.get("session"), dict):
        raise ValueError("Existing LLM JSON log has an unsupported structure")
    source = document["session"]
    if not isinstance(source.get("session_id"), str) or not isinstance(source.get("experiments"), dict):
        raise ValueError("Existing LLM JSON log has invalid session metadata")
    session = {"session_id": source["session_id"], "model": deepcopy(source.get("model")), "experiments": {}}
    for name, experiment in source["experiments"].items():
        if not isinstance(experiment, dict) or not isinstance(experiment.get("stages"), dict):
            raise ValueError("Existing LLM JSON log has invalid experiment metadata")
        attempt = experiment.get("attempt")
        if type(attempt) is not int or attempt < 1 or name != f"exp_{attempt:02d}":
            raise ValueError("LLM experiment name does not match its positive attempt")
        stages = {}
        for stage, payload in experiment["stages"].items():
            if stage not in _ALLOWED_STAGES:
                continue
            if (not isinstance(payload, dict) or not isinstance(payload.get("status"), str)
                    or "response" not in payload):
                raise ValueError("Existing LLM JSON log has invalid stage metadata")
            stages[stage] = {"status": payload["status"], "response": _response(stage, payload["response"])}
        if stages:
            session["experiments"][name] = {"attempt": attempt, "stages": stages}
    return {"session": session}


def session_log_records(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Read minimal stage responses without fabricating request metadata."""
    session = sanitize_session_log(document)["session"]
    return [{"attempt": experiment["attempt"], "stage": stage, "model": session["model"],
             "status": deepcopy(payload["status"]), "response": deepcopy(payload["response"])}
            for experiment in session["experiments"].values()
            for stage, payload in experiment["stages"].items()]
