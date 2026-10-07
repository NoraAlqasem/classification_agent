from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from training.artifact_paths import find_training_file, find_weights_file


def _read(run_dir: str, monitor: str = "mean") -> dict[str, Any]:
    from agent.nodes.monitor import _parse_results_csv
    path = find_training_file(run_dir, "training_metrics.csv")
    if not path.is_file():
        return {"error": f"training_metrics.csv not found in {run_dir}"}
    try:
        return _parse_results_csv(path, monitor)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"error": f"Invalid training_metrics.csv: {exc}"}


def _dump(value: Any) -> str:
    from memory.experiment_store import _clean_json
    return json.dumps(_clean_json(value), indent=2, default=str, allow_nan=False)


@tool
def read_results_csv(run_dir: str, monitor: str = "mean") -> str:
    """Read ResNet training_metrics.csv, selecting the saved epoch by mean or a head name.

    All primary accuracies refer to the same checkpoint epoch, not independent
    maxima. Final accuracy and loss fields are diagnostic values.
    """
    return _dump({**_read(run_dir, monitor), "split": "val"})


@tool
def read_test_metrics(run_dir: str) -> str:
    """Read measured test accuracy, full coverage, and checkpoint identity for an attempt."""
    path = Path(run_dir) / "test_evaluation" / "summary.json"
    if not path.is_file():
        return _dump({"error": f"No test evaluation summary in {run_dir}"})
    try:
        return _dump(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        return _dump({"error": f"Invalid test summary: {exc}"})


@tool
def compute_overfitting_score(run_dir: str) -> str:
    """Compare recent train/validation total-loss trends; return an evidence score."""
    metrics = _read(run_dir)
    if "error" in metrics:
        return _dump({**metrics, "score": -1})
    train, val = metrics["train_loss_history"], metrics["val_loss_history"]
    if metrics["nonfinite_loss"]:
        return _dump({"score": -1, "reasoning": "Nonfinite losses prevent a reliable trend comparison."})
    if len(train) < 5 or len(val) < 5:
        return _dump({"score": 0, "reasoning": "Too few epochs to assess loss trends."})
    window = min(10, len(train) // 2, len(val) // 2)
    train_trend = sum(train[-window:]) / window - sum(train[:window]) / window
    val_trend = sum(val[-window:]) / window - sum(val[:window]) / window
    if val_trend > 0.05 and train_trend < -0.05:
        score, reasoning = 2, "Validation total loss rose while training total loss fell."
    elif val_trend > 0 and train_trend < 0:
        score, reasoning = 1, "Mildly opposing train/validation total-loss trends."
    else:
        score, reasoning = 0, "No opposing total-loss trends detected."
    return _dump({"score": score, "reasoning": reasoning, "train_trend": train_trend, "val_trend": val_trend})


@tool
def get_final_metrics(run_dir: str) -> str:
    """Return final-epoch ResNet accuracies and total losses as diagnostics."""
    metrics = _read(run_dir)
    if "error" in metrics:
        return _dump(metrics)
    result = {"epoch": metrics["final_epoch"], "mean_acc": metrics["final_mean_acc"],
              "train_loss": metrics["train_loss"], "val_loss": metrics["val_loss"]}
    for head in metrics["heads"]:
        result[f"{head}_acc"] = metrics[f"final_{head}_acc"]
    return _dump(result)


@tool
def read_loss_history(run_dir: str) -> str:
    """Read the last twenty epochs of total losses and validation head accuracies."""
    metrics = _read(run_dir)
    if "error" in metrics:
        return _dump(metrics)
    return _dump({
        "train_loss": metrics["train_loss_history"][-20:],
        "val_loss": metrics["val_loss_history"][-20:],
        "mean_acc": metrics["mean_acc_history"][-20:],
        "head_acc": {head: history[-20:] for head, history in metrics["head_acc_history"].items()},
        "nonfinite_loss": metrics["nonfinite_loss"],
    })


@tool
def read_per_class_metrics(run_dir: str, monitor: str = "mean") -> str:
    """Read test per-class accuracy when available, otherwise validation head results.

    The supplied trainer exports per-head accuracy/loss, not per-class accuracy.
    This tool states that limitation instead of inventing class-level metrics.
    """
    test_path = Path(run_dir) / "test_evaluation" / "summary.json"
    if test_path.is_file():
        try:
            test = json.loads(test_path.read_text(encoding="utf-8"))
            per_class = {}
            for head in test.get("heads", {}):
                path = test_path.parent / head / "per_class_accuracy.csv"
                if path.is_file():
                    with path.open(newline="", encoding="utf-8-sig") as handle:
                        per_class[head] = list(csv.DictReader(handle))
            return _dump({"split": "test", "heads": test.get("heads", {}),
                          "per_class_accuracy": per_class, "full_coverage": test.get("full_coverage")})
        except (OSError, json.JSONDecodeError, csv.Error) as exc:
            return _dump({"error": f"Invalid test per-class results: {exc}"})
    metrics = _read(run_dir, monitor)
    if "error" in metrics:
        return _dump(metrics)
    class_path = find_weights_file(run_dir, "classes_by_head.json")
    classes = {}
    if class_path.is_file():
        try:
            classes = json.loads(class_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return _dump({"best_epoch": metrics["best_epoch"], "heads": metrics["heads"],
                  "classes_by_head": classes,
                  "note": "Per-class accuracy is not emitted by this trainer; these are head-level results."})


@tool
def list_run_artefacts(run_dir: str) -> str:
    """List logs, checkpoints, class mappings, and plots in a ResNet attempt."""
    path = Path(run_dir)
    if not path.is_dir():
        return _dump({"error": f"Directory not found: {run_dir}"})
    files = sorted(str(file.relative_to(path)) for file in path.rglob("*") if file.is_file())
    return _dump({"run_dir": run_dir, "files": files})
