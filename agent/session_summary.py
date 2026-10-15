"""Write a concise session overview using verified recorded results only."""
from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Any

from memory.experiment_format import deserialize_experiment
from memory.experiment_store import ExperimentStore, decision_score, experiment_score
from tools.class_alpha import DEFAULT_ALPHA, load_class_alpha


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _percent(value: Any) -> str:
    if isinstance(value, bool):
        return "—"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value * 100:.2f}%" if math.isfinite(value) and 0 <= value <= 1 else "—"


def _head_name(head: str, *, short: bool = False) -> str:
    if head in {"vehicle_type", "type"}:
        return "Type" if short else "Vehicle Type"
    return head.replace("_", " ").title()


def _metric_name(metric: str, heads: list[str] | None = None) -> str:
    if metric == "mean_acc":
        return "Mean Accuracy"
    if metric == "macro_acc":
        return "Macro Accuracy"
    if metric.endswith("_macro_acc") and (heads is None or metric[:-10] in heads):
        return f"{_head_name(metric[:-10])} Macro Accuracy"
    return f"{_head_name(metric.removesuffix('_acc'))} Accuracy"


def _experiment_name(record: dict) -> str:
    attempt = record.get("attempt")
    return f"exp_{attempt:02d}" if type(attempt) is int and attempt > 0 else "—"


def _experiment_label(record: dict) -> str:
    name = _experiment_name(record)
    return f"`{name}`" if name != "—" else name


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    return ["| " + " | ".join(_cell(value) for value in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)]


def _class_weight_changes(records: list[dict[str, Any]]) -> list[str]:
    """Compare alpha values rather than snapshot names or per-head loss weights."""
    snapshots: dict[str | None, dict | None] = {}
    result = []
    previous_heads = None
    previous_values = None
    for index, record in enumerate(records):
        config = record.get("config", {})
        raw_heads = config.get("heads") or record.get("heads")
        heads = (frozenset(raw_heads) if isinstance(raw_heads, (list, tuple))
                 and raw_heads and all(isinstance(head, str) for head in raw_heads) else None)
        values = None
        if "alpha_json" in config and heads is not None:
            path = config["alpha_json"]
            if path is None or isinstance(path, str):
                if path not in snapshots:
                    try:
                        snapshots[path] = load_class_alpha(path)
                    except (OSError, ValueError, TypeError):
                        snapshots[path] = None
                alpha = snapshots[path]
                if alpha is not None:
                    # Missing alpha entries use 1.0 in the trainer. Explicit
                    # default values and unused heads do not change weighting.
                    values = {head: {label: value for label, value in alpha.get(head, {}).items()
                                     if value != DEFAULT_ALPHA}
                              for head in sorted(heads)}
        if index == 0:
            changed = "Baseline"
        elif heads is None or heads != previous_heads or values is None or previous_values is None:
            changed = "Unknown"
        else:
            changed = "Yes" if values != previous_values else "No"
        result.append(changed)
        previous_heads, previous_values = heads, values
    return result


def render_session_summary(session_dir: str | Path, experiments: list[dict[str, Any]], *,
                           status: str = "In Progress", heads: list[str] | None = None,
                           target_metric: str = "mean_acc", target_value: float | None = None,
                           decision_split: str = "test", stop_reason: str | None = None,
                           errors: list[str] | None = None, onnx_path: str = "",
                           last_model_path: str = "", llm_allowed_config_keys: list[str] | None = None) -> str:
    """Compare attempts without promoting incomplete or unverified test scores."""
    records = [deserialize_experiment(record) for record in experiments]
    if heads is None:
        heads = next((record.get("heads") or record.get("config", {}).get("heads")
                      for record in records if record.get("heads") or record.get("config", {}).get("heads")), [])
    heads = list(heads)
    scores = []
    comparison_key = None
    for record in records:
        score = experiment_score(record, target_metric, decision_split,
                                 record.get("test_csv") or record.get("config", {}).get("test_csv"))
        key = ExperimentStore._comparison_key(record)
        if score is not None and comparison_key is None:
            comparison_key = key
        scores.append(score if key == comparison_key else None)
    eligible = [(record, score) for record, score in zip(records, scores) if score is not None]
    best, best_score = max(eligible, key=lambda pair: pair[1], default=({}, None))
    target = _percent(target_value)
    reached = "—" if best_score is None or target == "—" else ("Yes" if best_score >= float(target_value) else "No")
    lines = ["# Session Summary", "", "## Session Information", ""]
    information = [
        ["Session", Path(session_dir).name], ["Heads", ", ".join(f"`{head}`" for head in heads)],
        ["Target Metric", _metric_name(target_metric, heads)], ["Target Value", target],
        ["Decision Split", decision_split.title()], ["Total Experiments", len(records)],
        ["Status", status], ["Target Reached", reached],
    ]
    if stop_reason is not None:
        information.append(["Stop Reason", stop_reason])
    if llm_allowed_config_keys is not None:
        information.append(["LLM Allowed Changes", ", ".join(llm_allowed_config_keys) or "none"])
    lines.extend(_table(["Field", "Value"], information))
    split_label = decision_split.title()
    show_macro = (target_metric == "macro_acc" or target_metric.endswith("_macro_acc")
                  or any("macro_acc" in record.get("metrics", {}) for record in records))
    macro_columns = ([*[f"{_head_name(head, short=True)} {split_label} Macro Accuracy" for head in heads],
                      f"Macro {split_label} Accuracy"] if show_macro else [])
    accuracy_keys = [*(f"{head}_acc" for head in heads), "mean_acc"]
    if show_macro:
        accuracy_keys.extend([*(f"{head}_macro_acc" for head in heads), "macro_acc"])
    lines.extend(["", "## Experiment Comparison", ""])
    headers = ["Experiment", "Architecture", "Epochs", "Batch Size", "LR",
               *[f"{_head_name(head, short=True)} Weight" for head in heads],
               "Class Weights Changed",
               *[f"{_head_name(head, short=True)} {split_label} Accuracy" for head in heads],
               f"Mean {split_label} Accuracy", *macro_columns, "Verdict", "Training Status"]
    rows = []
    for record, score, class_weights_changed in zip(records, scores, _class_weight_changes(records)):
        config = record.get("config", {})
        weights = config.get("head_loss_weights") or {}
        metrics = record.get("metrics", {})
        values = []
        for metric in accuracy_keys:
            value = decision_score(metrics, record.get("training_status"), metric, decision_split,
                                   record.get("test_result"), record.get("test_csv")) if score is not None else None
            values.append(_percent(value))
        rows.append([_experiment_label(record), config.get("arch"), config.get("epochs"),
                     config.get("batch_size"), config.get("lr"), *[weights.get(head, 1.0) if config else None for head in heads],
                     class_weights_changed,
                     *values, record.get("evaluation", {}).get("verdict"), record.get("training_status")])
    lines.extend(_table(headers, rows))
    lines.extend(["", "Only successful, finite, verified results for the requested decision split are shown as accuracies.",
                  "Epochs shows the configured limit. Incomplete attempts show — for unavailable results.",
                  "Class Weights Changed compares per-class alpha values with the preceding experiment; "
                  "the first experiment is Baseline. Unknown means the alpha data or matching heads are unavailable. "
                  "Head loss weights are shown separately in the Weight columns.",
                  "", "## Best Experiment", ""])
    mean_candidates = [(record, value) for record, _ in eligible
                       if (value := decision_score(record.get("metrics", {}), record.get("training_status"),
                                                   "mean_acc", decision_split, record.get("test_result"), record.get("test_csv"))) is not None]
    best_mean_record, best_mean = max(mean_candidates, key=lambda pair: pair[1], default=({}, None))
    if best:
        lines.extend([f"**Best Experiment:** {_experiment_label(best)}", "",
                      f"**Best Mean Accuracy:** {_percent(best_mean)}"
                      + (f" ({_experiment_label(best_mean_record)})" if best_mean_record != best else ""), ""])
        if target_metric != "mean_acc":
            lines.extend([f"**Target {_metric_name(target_metric, heads)}:** {_percent(best_score)}", ""])
        if best.get("best_weights"):
            lines.extend([f"Best checkpoint: `{best['best_weights']}`", ""])
    else:
        lines.extend(["**Best Experiment:** —", "", "**Best Mean Accuracy:** —", "",
                      "No verified result is available for the requested target.", ""])
    best_metrics = best.get("metrics", {})
    best_rows = []
    metric_labels = [*[(_head_name(head) + " Accuracy", f"{head}_acc") for head in heads],
                     ("Mean Accuracy", "mean_acc")]
    if show_macro:
        metric_labels.extend([*[(_head_name(head) + " Macro Accuracy", f"{head}_macro_acc") for head in heads],
                              ("Macro Accuracy", "macro_acc")])
    for label, metric in metric_labels:
        value = decision_score(best_metrics, best.get("training_status"), metric, decision_split,
                               best.get("test_result"), best.get("test_csv")) if best else None
        best_rows.append([label, _percent(value)])
    best_rows.append(["Best Epoch", best_metrics.get("best_epoch")])
    lines.extend(_table(["Metric", "Value"], best_rows))
    lines.append("")
    lines.extend(["### Configuration", ""])
    config = best.get("config", {})
    image_size = config.get("img_size")
    image_size = " × ".join(str(value) for value in image_size) if isinstance(image_size, (list, tuple)) else image_size
    weights = config.get("head_loss_weights") or {}
    lines.extend(_table(["Parameter", "Value"], [
        ["Architecture", config.get("arch")], ["Epochs", config.get("epochs")],
        ["Batch Size", config.get("batch_size")], ["Learning Rate", config.get("lr")],
        ["Weight Decay", config.get("weight_decay")], ["Gamma", config.get("gamma")],
        *[[f"{_head_name(head)} Loss Weight", weights.get(head, 1.0) if config else None] for head in heads],
        ["Image Size", image_size],
    ]))
    lines.extend(["", "## Best Results by Head", ""])
    result_rows = []
    result_labels = [*[(_head_name(head), f"{head}_acc") for head in heads], ("Mean", "mean_acc")]
    if show_macro:
        result_labels.extend([*[(_head_name(head) + " Macro", f"{head}_macro_acc") for head in heads],
                              ("Macro", "macro_acc")])
    for label, metric in result_labels:
        measured = [(record, value) for record, _ in eligible
                    if (value := decision_score(record.get("metrics", {}), record.get("training_status"),
                                                metric, decision_split, record.get("test_result"), record.get("test_csv"))) is not None]
        selected, value = max(measured, key=lambda pair: pair[1], default=({}, None))
        result_rows.append([label, _experiment_label(selected), _percent(value)])
    lines.extend(_table(["Head", "Best Experiment", "Accuracy"], result_rows))
    artifacts = []
    if last_model_path:
        artifacts.append(["Latest checkpoint for best experiment", last_model_path])
    if onnx_path:
        artifacts.append(["ONNX export", onnx_path])
    if artifacts:
        lines.extend(["", "## Artifacts", ""])
        lines.extend(_table(["Artifact", "Path"], artifacts))
    if errors:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {_cell(error)}" for error in dict.fromkeys(errors))
    return "\n".join(lines) + "\n"


def write_session_summary(session_dir: str | Path, experiments: list[dict[str, Any]], *,
                          status: str = "In Progress", heads: list[str] | None = None,
                          target_metric: str = "mean_acc", target_value: float | None = None,
                          decision_split: str = "test", stop_reason: str | None = None,
                          errors: list[str] | None = None, onnx_path: str = "",
                          last_model_path: str = "", llm_allowed_config_keys: list[str] | None = None) -> Path:
    """Atomically replace this session's deterministic Markdown overview."""
    directory = Path(session_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session_summary.md"
    content = render_session_summary(directory, experiments, status=status, heads=heads,
                                     target_metric=target_metric, target_value=target_value,
                                     decision_split=decision_split, stop_reason=stop_reason, errors=errors,
                                     onnx_path=onnx_path, last_model_path=last_model_path,
                                     llm_allowed_config_keys=llm_allowed_config_keys)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=directory,
                                         prefix=".session-summary-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
