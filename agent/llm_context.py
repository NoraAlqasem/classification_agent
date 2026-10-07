"""Keep LLM context small while retaining measured scores and recent trends."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from memory.experiment_store import experiment_score
from tools.config_tools import TUNABLE_CONFIG_KEYS

_SCALAR_TYPES = (str, int, float, bool, type(None))
_HISTORY_KEYS = ("train_loss_history", "val_loss_history", "mean_acc_history")
_MAX_CLASSES_PER_HEAD = 50


def _scalar_fields(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if isinstance(value, _SCALAR_TYPES)}


def _compact_class_accuracy(records: Any) -> dict[str, Any] | None:
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
    selected = valid[:_MAX_CLASSES_PER_HEAD]
    return {
        "columns": ["class", "correct", "total", "accuracy"],
        "rows": [[label, correct, total, round(accuracy, 6)]
                 for label, correct, total, accuracy in selected],
        "total_classes": len(valid), "shown_classes": len(selected),
        "omitted_classes": len(valid) - len(selected),
        "selection": "all" if len(selected) == len(valid) else "lowest_accuracy",
        "macro_accuracy": round(sum(row[3] for row in valid) / len(valid), 6),
    }


def _compact_head(values: dict[str, Any]) -> dict[str, Any]:
    result = _scalar_fields(values)
    result.pop("per_class_accuracy", None)
    per_class = _compact_class_accuracy(values.get("per_class_accuracy"))
    if per_class is not None:
        result["per_class_accuracy"] = per_class
    return result


def _compact_metrics(metrics: dict[str, Any], include_validation: bool) -> dict[str, Any]:
    result = _scalar_fields(metrics)
    if isinstance(metrics.get("heads"), dict):
        result["heads"] = {
            head: _compact_head(values) for head, values in metrics["heads"].items()
            if isinstance(values, dict)
        }
    if isinstance(metrics.get("img_size"), (list, tuple)):
        result["img_size"] = list(metrics["img_size"])
    for key in _HISTORY_KEYS:
        values = metrics.get(key)
        if isinstance(values, (list, tuple)):
            result[key] = deepcopy(list(values[-5:]))
            result[f"{key}_length"] = len(values)
    if isinstance(metrics.get("head_acc_history"), dict):
        histories = {head: values for head, values in metrics["head_acc_history"].items()
                     if isinstance(values, (list, tuple))}
        result["head_acc_history"] = {head: deepcopy(list(values[-5:])) for head, values in histories.items()}
        result["head_acc_history_lengths"] = {head: len(values) for head, values in histories.items()}
    if include_validation and isinstance(metrics.get("validation_metrics"), dict):
        result["validation_metrics"] = _compact_metrics(metrics["validation_metrics"], False)
    return result


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Preserve scores and evidence, with only the last five history values."""
    return _compact_metrics(metrics, True)


def compact_test_result(result: dict[str, Any]) -> dict[str, Any]:
    """Keep inference evidence once, including nested-only fields if necessary."""
    nested = result.get("metrics")
    values = {**nested, **result} if isinstance(nested, dict) else result
    compact = _scalar_fields(values)
    if isinstance(values.get("heads"), dict):
        compact["heads"] = {
            head: _compact_head(evidence) for head, evidence in values["heads"].items()
            if isinstance(evidence, dict)
        }
    if isinstance(values.get("img_size"), (list, tuple)):
        compact["img_size"] = list(values["img_size"])
    return compact


def _short_items(values: Any) -> list[str]:
    if not isinstance(values, list):
        values = [values] if values else []
    return [str(value)[:240] for value in values[:3]]


def compact_experiment_history(records: list[dict[str, Any]], target_metric: str = "mean_acc") -> list[dict[str, Any]]:
    """Summarize the last three attempts and an older best verified attempt."""
    recent_indices = list(range(max(0, len(records) - 3), len(records)))
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
            "issues": _short_items(evaluation.get("issues", record.get("issues_detected", []))),
            "recommendations": _short_items(evaluation.get("recommendations", [])),
        })
    return result
