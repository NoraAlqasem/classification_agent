"""Keep LLM context small while retaining measured scores and recent trends."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import os
from typing import Any

from memory.experiment_store import experiment_score
from tools.config_tools import TUNABLE_CONFIG_KEYS

_SCALAR_TYPES = (str, int, float, bool, type(None))
_HISTORY_KEYS = ("train_loss_history", "val_loss_history", "mean_acc_history")


@dataclass(frozen=True)
class ContextLimits:
    history_entries: int = 5
    recent_experiments: int = 3
    max_items: int = 3
    max_classes_per_head: int = 50


def _positive_limit(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    error = f"{name} must be a positive integer; got {raw!r}."
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(error) from exc
    if value <= 0:
        raise ValueError(error)
    return value


def get_context_limits() -> ContextLimits:
    """Read prompt limits from the environment without caching user settings."""
    defaults = ContextLimits()
    return ContextLimits(
        history_entries=_positive_limit("LLM_CONTEXT_HISTORY_ENTRIES", defaults.history_entries),
        recent_experiments=_positive_limit("LLM_CONTEXT_RECENT_EXPERIMENTS", defaults.recent_experiments),
        max_items=_positive_limit("LLM_CONTEXT_MAX_ITEMS", defaults.max_items),
        max_classes_per_head=_positive_limit("LLM_CONTEXT_MAX_CLASSES_PER_HEAD", defaults.max_classes_per_head),
    )


def _scalar_fields(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if isinstance(value, _SCALAR_TYPES)}


def _compact_class_accuracy(records: Any, max_classes: int) -> dict[str, Any] | None:
    if not isinstance(records, list):
        return None
    valid = []
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        label, correct, total, accuracy = (record.get(key) for key in
                                           ("class", "correct", "total", "accuracy"))
        if (not isinstance(label, str) or not label.strip() or label in seen
                or type(correct) is not int or type(total) is not int
                or total <= 0 or not 0 <= correct <= total
                or isinstance(accuracy, bool) or not isinstance(accuracy, (int, float))
                or not math.isfinite(accuracy)
                or not math.isclose(accuracy, correct / total, rel_tol=1e-9, abs_tol=1e-9)):
            continue
        seen.add(label)
        valid.append((label, correct, total, correct / total))
    if not valid:
        return None
    valid.sort(key=lambda row: (row[3], -row[2], row[0]))
    selected = valid[:max_classes]
    return {
        "columns": ["class", "correct", "total", "accuracy"],
        "rows": [[label, correct, total, round(accuracy, 6)]
                 for label, correct, total, accuracy in selected],
        "total_classes": len(valid), "shown_classes": len(selected),
        "omitted_classes": len(valid) - len(selected),
        "selection": "all" if len(selected) == len(valid) else "lowest_accuracy",
        "macro_accuracy": round(sum(row[3] for row in valid) / len(valid), 6),
    }


def _compact_head(values: dict[str, Any], max_classes: int) -> dict[str, Any]:
    result = _scalar_fields(values)
    result.pop("per_class_accuracy", None)
    per_class = _compact_class_accuracy(values.get("per_class_accuracy"), max_classes)
    if per_class is not None:
        result["per_class_accuracy"] = per_class
    return result


def _compact_metrics(metrics: dict[str, Any], include_validation: bool, limits: ContextLimits) -> dict[str, Any]:
    result = _scalar_fields(metrics)
    if isinstance(metrics.get("heads"), dict):
        result["heads"] = {
            head: _compact_head(values, limits.max_classes_per_head) for head, values in metrics["heads"].items()
            if isinstance(values, dict)
        }
    if isinstance(metrics.get("img_size"), (list, tuple)):
        result["img_size"] = list(metrics["img_size"])
    for key in _HISTORY_KEYS:
        values = metrics.get(key)
        if isinstance(values, (list, tuple)):
            result[key] = deepcopy(list(values[-limits.history_entries:]))
            result[f"{key}_length"] = len(values)
    if isinstance(metrics.get("head_acc_history"), dict):
        histories = {head: values for head, values in metrics["head_acc_history"].items()
                     if isinstance(values, (list, tuple))}
        result["head_acc_history"] = {
            head: deepcopy(list(values[-limits.history_entries:])) for head, values in histories.items()
        }
        result["head_acc_history_lengths"] = {head: len(values) for head, values in histories.items()}
    if include_validation and isinstance(metrics.get("validation_metrics"), dict):
        result["validation_metrics"] = _compact_metrics(metrics["validation_metrics"], False, limits)
    return result


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Preserve scores and evidence within the configured history and class limits."""
    return _compact_metrics(metrics, True, get_context_limits())


def compact_test_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep inference evidence once, including nested-only fields if necessary."""
    limits = get_context_limits()
    nested = result.get("metrics")
    values = {**nested, **result} if isinstance(nested, dict) else result
    compact = _scalar_fields(values)
    if isinstance(values.get("heads"), dict):
        compact["heads"] = {
            head: _compact_head(evidence, limits.max_classes_per_head) for head, evidence in values["heads"].items()
            if isinstance(evidence, dict)
        }
    if isinstance(values.get("img_size"), (list, tuple)):
        compact["img_size"] = list(values["img_size"])
    return compact


def _short_items(values: Any, max_items: int) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    return [str(value)[:240] for value in values[:max_items]]


def compact_experiment_history(records: list[dict[str, Any]], target_metric: str = "mean_acc") -> list[dict[str, Any]]:
    """Summarize the configured recent attempts and an older best verified attempt."""
    limits = get_context_limits()
    recent_indices = list(range(max(0, len(records) - limits.recent_experiments), len(records)))
    scored = [(index, experiment_score(record, target_metric)) for index, record in enumerate(records)]
    valid = [(index, score) for index, score in scored if score is not None]
    if valid:
        best_index = max(valid, key=lambda item: item[1])[0]
        if best_index not in recent_indices:
            recent_indices.insert(0, best_index)
    scores = dict(scored)
    result = []
    for index in recent_indices:
        record = records[index]
        config, metrics, evaluation = record.get("config", {}), record.get("metrics", {}), record.get("evaluation", {})
        result.append({
            "attempt": record.get("attempt"),
            "decision_split": record.get("decision_split", "val"),
            "test_csv": record.get("test_csv", ""),
            "arch": config.get("arch"),
            "training_status": record.get("training_status"),
            "target_metric": target_metric,
            "metric_value": metrics.get(target_metric),
            "score": scores[index],
            "config": {key: deepcopy(config[key]) for key in sorted(TUNABLE_CONFIG_KEYS)
                       if key in config and key != "arch"},
            "verdict": evaluation.get("verdict"),
            "issues": _short_items(evaluation.get("issues", record.get("issues_detected", [])), limits.max_items),
            "recommendations": _short_items(evaluation.get("recommendations", []), limits.max_items),
        })
    return result
