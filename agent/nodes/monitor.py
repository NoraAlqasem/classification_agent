from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from agent.state import AgentState
from training.artifact_paths import find_training_file

console = Console()


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_results_csv(csv_path: Path, monitor: str = "mean") -> dict[str, Any]:
    """Read the ResNet log; all primary accuracies use one checkpoint epoch.

    Ties select the first epoch, matching the script's strict improvement rule.
    Final losses and final accuracies are diagnostic values.
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = [column.strip() for column in (reader.fieldnames or [])]
        missing = sorted({"Epoch", "Split", "TotalLoss", "MeanAcc"} - set(columns))
        if missing:
            raise ValueError(f"Missing training metric columns: {missing}")
        rows = [{key.strip(): value for key, value in row.items() if key is not None} for row in reader]

    heads = [column[:-4] for column in columns if column.endswith("_Acc")]
    for head in heads:
        if f"{head}_Loss" not in columns:
            raise ValueError(f"Missing loss column for head {head}")
    if monitor == "mean":
        score_column = "MeanAcc"
    elif monitor == "macro":
        score_column = "MacroAcc"
    elif monitor in heads:
        score_column = f"{monitor}_Acc"
    elif monitor.endswith("_macro") and monitor[:-6] in heads:
        score_column = f"{monitor[:-6]}_MacroAcc"
    else:
        raise ValueError(f"Checkpoint monitor '{monitor}' is absent from the metric log")
    if score_column not in columns:
        raise ValueError(f"Checkpoint monitor '{monitor}' requires missing metric column '{score_column}'")
    for row in rows:
        row["Epoch"] = int(row["Epoch"])
    train = sorted((row for row in rows if row["Split"].strip().lower() == "train"), key=lambda row: row["Epoch"])
    val = sorted((row for row in rows if row["Split"].strip().lower() == "val"), key=lambda row: row["Epoch"])
    if not val or not train:
        raise ValueError("Training metric log must contain train and val rows")
    train_epochs = [row["Epoch"] for row in train]
    val_epochs = [row["Epoch"] for row in val]
    if train_epochs != val_epochs or len(set(val_epochs)) != len(val_epochs):
        raise ValueError("Training metric log must contain exactly one train/val pair per epoch")

    valid = [row for row in val if _finite_number(row.get(score_column)) is not None]
    if not valid:
        raise ValueError(f"No finite validation values for checkpoint monitor {monitor}")
    best = max(valid, key=lambda row: float(row[score_column]))
    final = val[-1]
    loss_columns = ["TotalLoss", *(f"{head}_Loss" for head in heads)]
    nonfinite_loss = any(_finite_number(row.get(column)) is None for row in rows for column in loss_columns)
    macro_heads = [head for head in heads if f"{head}_MacroAcc" in columns]
    accuracy_columns = ["MeanAcc", *(f"{head}_Acc" for head in heads)]
    if "MacroAcc" in columns:
        accuracy_columns.append("MacroAcc")
    accuracy_columns.extend(f"{head}_MacroAcc" for head in macro_heads)
    nonfinite_accuracy = any(_finite_number(row.get(column)) is None for row in rows for column in accuracy_columns)

    metrics: dict[str, Any] = {
        "total_epochs": len(val),
        "final_epoch": final["Epoch"],
        "best_epoch": best["Epoch"],
        "monitor": monitor,
        "best_score": _finite_number(best[score_column]),
        "mean_acc": _finite_number(best["MeanAcc"]),
        "final_mean_acc": _finite_number(final["MeanAcc"]),
        "heads": {
            head: {"acc": _finite_number(best[f"{head}_Acc"]), "loss": _finite_number(best[f"{head}_Loss"])}
            for head in heads
        },
        "train_loss": _finite_number(train[-1]["TotalLoss"]),
        "val_loss": _finite_number(final["TotalLoss"]),
        "best_val_loss": _finite_number(best["TotalLoss"]),
        "loss_epoch": final["Epoch"],
        "train_loss_history": [_finite_number(row["TotalLoss"]) for row in train],
        "val_loss_history": [_finite_number(row["TotalLoss"]) for row in val],
        "mean_acc_history": [_finite_number(row["MeanAcc"]) for row in val],
        "head_acc_history": {head: [_finite_number(row[f"{head}_Acc"]) for row in val] for head in heads},
        "nonfinite_loss": nonfinite_loss,
        "nonfinite_accuracy": nonfinite_accuracy,
    }
    for head in heads:
        metrics[f"{head}_acc"] = _finite_number(best[f"{head}_Acc"])
        metrics[f"final_{head}_acc"] = _finite_number(final[f"{head}_Acc"])
    if "MacroAcc" in columns:
        metrics.update({
            "macro_acc": _finite_number(best["MacroAcc"]),
            "final_macro_acc": _finite_number(final["MacroAcc"]),
            "macro_acc_history": [_finite_number(row["MacroAcc"]) for row in val],
        })
    if macro_heads:
        metrics["head_macro_acc_history"] = {
            head: [_finite_number(row[f"{head}_MacroAcc"]) for row in val]
            for head in macro_heads
        }
        for head in macro_heads:
            value = _finite_number(best[f"{head}_MacroAcc"])
            metrics["heads"][head]["macro_acc"] = value
            metrics[f"{head}_macro_acc"] = value
            metrics[f"final_{head}_macro_acc"] = _finite_number(final[f"{head}_MacroAcc"])
    return metrics


def monitor_node(state: AgentState) -> AgentState:
    """Extract checkpoint-aligned metrics from training_metrics.csv."""
    console.rule("[bold]Monitor[/bold]")
    training_result = state.get("training_result", {})
    run_dir = Path(training_result.get("run_dir") or state.get("run_dir") or ".")
    if training_result.get("status") != "success":
        return {"metrics": {"status": "training_failed", "error": training_result.get("error", "Training did not complete successfully"), "run_dir": str(run_dir)}}
    results_csv = find_training_file(run_dir, "training_metrics.csv")
    if not results_csv.is_file():
        return {"metrics": {"status": "no_results_csv", "error": f"Missing log: {results_csv}", "run_dir": str(run_dir)}}
    try:
        metrics = _parse_results_csv(results_csv, state.get("current_config", {}).get("monitor", "mean"))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"metrics": {"status": "no_results_csv", "error": f"Invalid training log: {exc}", "run_dir": str(run_dir)}}
    metrics.update({"run_dir": str(run_dir), "status": "ok"})
    table = Table(title=f"Validation accuracy — attempt {state.get('attempt_number', '?')}")
    table.add_column("Metric", style="cyan")
    table.add_column(f"Best checkpoint (epoch {metrics['best_epoch']})")
    table.add_column(f"Final epoch ({metrics['final_epoch']})")
    display_keys = ["mean_acc", "macro_acc", *(f"{head}_acc" for head in metrics["heads"]),
                    *(f"{head}_macro_acc" for head in metrics["heads"])]
    for key in (key for key in display_keys if key in metrics):
        best_value, final_value = metrics.get(key), metrics.get(f"final_{key}")
        table.add_row(key, f"{best_value:.4f}" if best_value is not None else "—", f"{final_value:.4f}" if final_value is not None else "—")
    console.print(table)
    return {"metrics": metrics}
