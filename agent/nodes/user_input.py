from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.prompt import Prompt

from agent.llm_context import get_context_limits
from agent.state import AgentState
from tools.config_tools import get_optimizer_hyperparameter_limit

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULTS_PATH = PROJECT_ROOT / "config" / "defaults.yaml"


def _load_defaults() -> dict[str, Any]:
    with _DEFAULTS_PATH.open() as handle:
        data = yaml.safe_load(handle) or {}
    return {**data.get("agent", {}), **data.get("training", {})}


def _number(value: Any, name: str, errors: list[str], *, integer: bool = False,
            fallback: int | float = 0) -> int | float:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean values are not numbers")
        result = int(str(value)) if integer else float(value)
        if not math.isfinite(result):
            raise ValueError("must be finite")
        return result
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{name} must be {'an integer' if integer else 'a finite number'}; got {value!r}.")
        return fallback


def _head_weights(raw: Any, heads: list[str], errors: list[str]) -> dict[str, float]:
    weights = {head: 1.0 for head in heads}
    if raw is None or raw == "":
        return weights
    if isinstance(raw, dict):
        pairs = raw.items()
    elif isinstance(raw, (list, tuple, str)):
        entries = raw.split() if isinstance(raw, str) else raw
        pairs = []
        for item in entries:
            if not isinstance(item, str) or "=" not in item:
                errors.append(f"Invalid head loss weight {item!r}; use HEAD=WEIGHT.")
                continue
            name, value = item.split("=", 1)
            pairs.append((name.strip(), value))
    else:
        errors.append("head_loss_weights must be a mapping or HEAD=WEIGHT values.")
        return weights
    for head, value in pairs:
        if head not in weights:
            errors.append(f"Unknown head {head!r} in head_loss_weights. Valid heads: {heads}.")
            continue
        weights[head] = _number(value, f"Loss weight for {head}", errors, fallback=1.0)
    return weights


def user_input_node(state: AgentState) -> AgentState:
    """Collect CLI/environment inputs and apply defaults before prompting."""
    defaults = _load_defaults()
    interactive = state.get("interactive", sys.stdin.isatty())
    errors: list[str] = []
    for validate_settings in (get_context_limits, get_optimizer_hyperparameter_limit):
        try:
            validate_settings()
        except ValueError as exc:
            errors.append(str(exc))
    console.rule("[bold cyan]Autonomous ResNet Training Agent[/bold cyan]")
    console.print("[dim]Multi-head classification · PyTorch DistributedDataParallel[/dim]\n")

    explicit_dataset = str(state.get("dataset_path") or "").strip()
    dataset_path = explicit_dataset or os.getenv("DATASET_PATH", "").strip()
    train_csv = str(state.get("train_csv") or "").strip()
    val_csv = str(state.get("val_csv") or "").strip()
    test_csv = str(state.get("test_csv") or "").strip()
    decision_split = str(state.get("decision_split") or os.getenv("DECISION_SPLIT", "").strip()
                         or defaults.get("decision_split", "test")).strip().lower()
    # An explicit --dataset takes priority over environment CSV paths.
    if not explicit_dataset:
        train_csv = train_csv or os.getenv("TRAIN_CSV", "").strip()
        val_csv = val_csv or os.getenv("VAL_CSV", "").strip()
        test_csv = test_csv or os.getenv("TEST_CSV", "").strip()
    if dataset_path:
        train_csv = train_csv or str(Path(dataset_path).expanduser() / "train.csv")
        val_csv = val_csv or str(Path(dataset_path).expanduser() / "val.csv")
        if decision_split == "test" or test_csv:
            test_csv = test_csv or str(Path(dataset_path).expanduser() / "test.csv")
    if not train_csv and interactive:
        train_csv = Prompt.ask("Training CSV path").strip()
    if not val_csv and interactive:
        val_csv = Prompt.ask("Validation CSV path").strip()
    if decision_split == "test" and not test_csv and interactive:
        test_csv = Prompt.ask("Test CSV path (used to decide whether to retrain)").strip()

    raw_heads = state.get("heads") or os.getenv("HEADS", "").strip()
    if not raw_heads and interactive:
        raw_heads = Prompt.ask("Classification heads (space-separated CSV columns)").strip()
    if isinstance(raw_heads, str):
        raw_heads = raw_heads.split()
    if not isinstance(raw_heads, (list, tuple)) or any(not isinstance(head, str) for head in raw_heads):
        errors.append("heads must be a list of CSV column names.")
        raw_heads = []
    heads = list(dict.fromkeys(head.strip() for head in raw_heads if head.strip()))
    # The raw CompCars label column is named type, a reserved ModuleDict key.
    heads = list(dict.fromkeys("vehicle_type" if head == "type" else head for head in heads))

    arch = str(state.get("model_type") or defaults.get("arch", "resnet18")).strip().lower()
    weights = str(state.get("weights_path") or PROJECT_ROOT / "weights" / "pretrained" / f"{arch}.pth")
    target_metric = str(state.get("target_metric") or defaults.get("target_metric", "mean_acc")).strip()
    if "vehicle_type" in heads and target_metric in {"type", "type_acc"}:
        target_metric = "vehicle_type_acc"
    target_raw = state.get("target_value")
    if target_raw is None or target_raw == "":
        target_raw = defaults.get("target_value", 0.85)
    attempts_raw = state.get("max_attempts")
    if attempts_raw is None or attempts_raw == "":
        attempts_raw = defaults.get("max_attempts", 5)
    nproc_raw = state.get("nproc_per_node")
    if nproc_raw is None or nproc_raw == "":
        nproc_raw = defaults.get("nproc_per_node", 1)
    target_value = _number(target_raw, "target_value", errors, fallback=0.85)
    max_attempts = _number(attempts_raw, "max_attempts", errors, integer=True, fallback=5)
    nproc = _number(nproc_raw, "nproc_per_node", errors, integer=True, fallback=1)
    export_format = str(state.get("export_format") or defaults.get("export_format", "onnx")).strip().lower()
    loss_weights = _head_weights(state.get("head_loss_weights"), heads, errors)

    console.print(f"[green]Training CSV:[/green] {train_csv or '(missing)'}")
    console.print(f"[green]Validation CSV:[/green] {val_csv or '(missing)'}")
    console.print(f"[green]Test CSV:[/green] {test_csv or ('(unused)' if decision_split == 'val' else '(missing)')}")
    console.print(f"[green]Retraining decisions:[/green] {decision_split} accuracy")
    console.print(f"[green]Heads:[/green] {', '.join(heads) or '(missing)'}")
    console.print(f"[green]Model:[/green] {arch} | GPUs/processes: {nproc}")
    console.print(f"[green]Target:[/green] {target_metric} ≥ {target_value} | attempts: {max_attempts}\n")

    return {
        "dataset_path": dataset_path,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "test_csv": test_csv,
        "decision_split": decision_split,
        "data_root": str(state.get("data_root") or "").strip() or os.getenv("DATA_ROOT", "").strip(),
        "heads": heads,
        "model_type": arch,
        "weights_path": weights,
        "alpha_json": str(state.get("alpha_json") or "").strip(),
        "head_loss_weights": loss_weights,
        "nproc_per_node": nproc,
        "target_metric": target_metric,
        "target_value": target_value,
        "max_attempts": max_attempts,
        "export_format": export_format,
        "input_errors": errors,
        "attempt_number": state.get("attempt_number", 0),
        "config_history": [],
        "experiment_history": [],
        "messages": [],
    }
