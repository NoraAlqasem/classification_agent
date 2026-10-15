"""Keep LLM context small while retaining measured scores and recent trends."""
from __future__ import annotations

from copy import deepcopy
from collections import Counter
from dataclasses import dataclass
import math
import os
from typing import Any

from memory.experiment_format import deserialize_experiment
from memory.experiment_store import ExperimentStore, experiment_score
from tools.config_tools import TUNABLE_CONFIG_KEYS

_SCALAR_TYPES = (str, int, float, bool, type(None))
_HISTORY_KEYS = ("train_loss_history", "val_loss_history", "mean_acc_history", "macro_acc_history")


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
    for key in ("head_acc_history", "head_macro_acc_history"):
        if isinstance(metrics.get(key), dict):
            histories = {head: values for head, values in metrics[key].items()
                         if isinstance(values, (list, tuple))}
            result[key] = {
                head: deepcopy(list(values[-limits.history_entries:])) for head, values in histories.items()
            }
            result[f"{key}_lengths"] = {head: len(values) for head, values in histories.items()}
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


def _history_comparison_context(record: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Keep unknown tasks separate instead of inventing comparable evidence."""
    config = record.get("config", {})
    context = {
        "target_metric": record.get("target_metric"),
        "decision_split": record.get("decision_split"),
        "heads": deepcopy(config.get("heads") or record.get("heads", [])),
        "train_csv": config.get("train_csv") or record.get("train_csv"),
        "val_csv": config.get("val_csv") or record.get("val_csv"),
        "test_csv": record.get("test_csv") or config.get("test_csv"),
    }
    known = (
        isinstance(context["target_metric"], str) and bool(context["target_metric"])
        and context["decision_split"] in {"val", "test"}
        and isinstance(context["heads"], list) and bool(context["heads"])
        and all(isinstance(head, str) and head for head in context["heads"])
        and all(isinstance(context[key], str) and context[key] for key in ("train_csv", "val_csv"))
        and (context["decision_split"] != "test"
             or isinstance(context["test_csv"], str) and bool(context["test_csv"]))
    )
    return context, ExperimentStore._comparison_key(record) if known else None


def summarize_older_experiments(records: list[dict[str, Any]], target_metric: str = "mean_acc") -> dict[str, Any]:
    """Describe every attempt before the detailed window using verified facts.

    A group's first row carries its tunable configuration; subsequent rows carry
    only changes from its previous summarized attempt. Score deltas require both
    attempts to have verified scores for the requested target. The older best
    can also appear in detailed history; it remains one summarized attempt.
    """
    limits = get_context_limits()
    cutoff = max(0, len(records) - limits.recent_experiments)
    columns = ["attempt", "group", "arch", "training_status", "metrics_status",
               "verified_score", "score_change", "config_changes", "verdict", "rationale", "error"]
    groups: list[dict[str, Any]] = []
    group_indices: dict[Any, int] = {}
    previous_configs: dict[int, dict[str, Any]] = {}
    previous_scores: dict[int, float | None] = {}
    issues: dict[int, Counter] = {}
    rows = []
    for index, raw_record in enumerate(records[:cutoff]):
        record = deserialize_experiment(raw_record)
        config = record.get("config", {})
        metrics = record.get("metrics", {})
        evaluation = record.get("evaluation", {})
        context, comparison_key = _history_comparison_context(record)
        key = comparison_key if comparison_key is not None else ("unknown", index)
        if key not in group_indices:
            group_id = len(groups) + 1
            group_indices[key] = group_id
            groups.append({
                "group": group_id, **context, "count": 0, "verified_count": 0,
                "best_attempt": None, "best_score": None, "min_score": None, "max_score": None,
                "training_status_counts": {}, "common_issues": [],
            })
            issues[group_id] = Counter()
        group_id = group_indices[key]
        group = groups[group_id - 1]
        score = experiment_score(record, target_metric)
        previous_score = previous_scores.get(group_id)
        score_change = score - previous_score if score is not None and previous_score is not None else None
        tunables = {key: deepcopy(config[key]) for key in sorted(TUNABLE_CONFIG_KEYS - {"arch"}) if key in config}
        previous_config = previous_configs.get(group_id)
        changes = tunables if previous_config is None else {
            key: deepcopy(tunables.get(key)) for key in sorted(previous_config.keys() | tunables.keys())
            if previous_config.get(key) != tunables.get(key)
        }
        previous_configs[group_id] = tunables
        previous_scores[group_id] = score
        status = record.get("training_status")
        status_label = status if isinstance(status, str) and status else "unknown"
        group["count"] += 1
        group["training_status_counts"][status_label] = group["training_status_counts"].get(status_label, 0) + 1
        if score is not None:
            group["verified_count"] += 1
            if group["best_score"] is None or score > group["best_score"]:
                group["best_attempt"], group["best_score"] = record.get("attempt"), score
            group["min_score"] = score if group["min_score"] is None else min(group["min_score"], score)
            group["max_score"] = score if group["max_score"] is None else max(group["max_score"], score)
        issue_items = _short_items(evaluation.get("issues", record.get("issues_detected", [])), limits.max_items)
        issues[group_id].update(list(dict.fromkeys(issue_items)))
        error = record.get("error") or metrics.get("error")
        if not error and evaluation.get("llm_error"):
            error = f"Evaluation unavailable: {evaluation['llm_error']}"
        rationale = record.get("optimisation_rationale") or evaluation.get("reasoning") or ""
        rows.append([
            record.get("attempt"), group_id, config.get("arch") or record.get("arch"),
            status, metrics.get("status"), score, score_change, changes, evaluation.get("verdict"),
            str(rationale)[:240], str(error)[:240] if error else None,
        ])
    for group in groups:
        group["common_issues"] = [{"issue": issue, "count": count}
                                  for issue, count in issues[group["group"]].most_common(limits.max_items)]
    return {
        "count": cutoff, "recent_count": len(records) - cutoff, "target_metric": target_metric,
        "groups": groups, "columns": columns, "rows": rows,
    }
