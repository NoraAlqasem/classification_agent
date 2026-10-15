from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from agent.state import AgentState
from training.data import read_dataset_csv

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ARCHITECTURES = {"resnet18", "resnet34", "resnet50"}
# ModuleDict refuses keys that collide with its own attributes. Keep validation
# independent of importing torch so --validate-only works without CUDA/PyTorch.
_RESERVED_HEAD_NAMES = set("""
__annotations__ __call__ __contains__ __delitem__ __dict__ __getattr__
__getitem__ __iter__ __len__ __module__ __setitem__ __setstate__ __weakref__
T_destination _apply _backward_hooks _backward_pre_hooks _buffers _call_impl
_compiled_call_impl _forward_hooks _forward_hooks_always_called
_forward_hooks_with_kwargs _forward_pre_hooks _forward_pre_hooks_with_kwargs
_get_backward_hooks _get_backward_pre_hooks _get_name _is_full_backward_hook
_load_from_state_dict _load_state_dict_post_hooks _load_state_dict_pre_hooks
_maybe_warn_non_full_backward_hook _modules _named_members
_non_persistent_buffers_set _parameters _register_load_state_dict_pre_hook
_register_state_dict_hook _replicate_for_data_parallel _save_to_state_dict
_slow_forward _state_dict_hooks _state_dict_pre_hooks _version _wrapped_call_impl
add_module apply bfloat16 buffers call_super_init children clear compile cpu cuda
double dump_patches eval extra_repr float forward get_buffer get_extra_state
get_parameter get_submodule half ipu items keys load_state_dict modules mtia
named_buffers named_children named_modules named_parameters parameters pop
register_backward_hook register_buffer register_forward_hook
register_forward_pre_hook register_full_backward_hook register_full_backward_pre_hook
register_load_state_dict_post_hook register_load_state_dict_pre_hook register_module
register_parameter register_state_dict_post_hook register_state_dict_pre_hook
requires_grad_ set_extra_state set_submodule share_memory state_dict to to_empty
train training type update values xpu zero_grad
""".split()) | set(dir(object))


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def canonical_target_metric(metric: str, heads: list[str]) -> str:
    """Accept overall and macro accuracy aliases; retain the CSV head's case."""
    metric = metric.strip()
    # An explicit case-sensitive head metric takes priority over aggregate aliases.
    for head in heads:
        if metric == f"{head}_acc":
            return metric
    if metric.lower() in {"mean", "mean_acc", "accuracy"}:
        return "mean_acc"
    if metric.lower() in {"macro", "macro_acc", "macro_accuracy", "balanced_accuracy"}:
        return "macro_acc"
    for head in heads:
        if metric in {head, f"{head}_acc"}:
            return f"{head}_acc"
        if metric in {f"{head}_macro", f"{head}_macro_acc", f"{head}_macro_accuracy"}:
            return f"{head}_macro_acc"
    choices = ["mean_acc", "macro_acc", *(f"{head}_acc" for head in heads),
               *(f"{head}_macro_acc" for head in heads)]
    raise ValueError(f"Unknown target_metric {metric!r}. Choose one of {choices}.")


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _check_alpha(path: str, report: dict[str, Any], heads: list[str]) -> None:
    if not path:
        return
    try:
        with Path(path).open() as handle:
            alpha = json.load(handle)
        if not isinstance(alpha, dict):
            raise ValueError("expected a JSON object mapping heads to class weights")
        for head, values in alpha.items():
            if not isinstance(values, dict):
                raise ValueError(f"head {head!r} must map class names to alpha values")
            for label, value in values.items():
                numeric = _finite_number(value)
                if numeric is None or numeric < 0:
                    raise ValueError(f"alpha for {head}/{label} must be finite and nonnegative")
            if head not in heads:
                report["warnings"].append(f"Alpha JSON contains unused head {head!r}.")
            elif head in report["classes_by_head"]:
                unused = sorted(set(values) - set(report["classes_by_head"][head]))
                if unused:
                    report["warnings"].append(f"Alpha JSON for {head!r} contains unused classes: {unused}.")
    except (OSError, ValueError, TypeError) as exc:
        report["errors"].append(f"Invalid alpha JSON {path}: {exc}")


def _check_images(df: pd.DataFrame, split: str, data_root: str,
                  report: dict[str, Any]) -> None:
    missing: list[str] = []
    missing_count = 0
    readable_samples: list[Path] = []
    bad_rows: list[int] = []
    for index, row in df.iterrows():
        root, name = row["root_folder"], row["image_name"]
        if pd.isna(root) or pd.isna(name) or not str(root).strip() or not str(name).strip():
            if len(bad_rows) < 10:
                bad_rows.append(int(index))
            continue
        root_path = Path(str(root))
        if not root_path.is_absolute():
            root_path = Path(data_root) / root_path if data_root else PROJECT_ROOT / root_path
        image_path = root_path / str(name)
        if not image_path.is_file():
            missing_count += 1
            if len(missing) < 10:
                missing.append(str(image_path))
        elif len(readable_samples) < 8:
            readable_samples.append(image_path)
    if bad_rows:
        report["errors"].append(f"{split}: missing root_folder or image_name at row indices {bad_rows}.")
    if missing_count:
        report["errors"].append(
            f"{split}: {missing_count} image files do not exist. Examples: {missing}."
        )
    if readable_samples:
        from PIL import Image

        for image_path in readable_samples:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:
                report["errors"].append(f"{split}: unreadable image {image_path}: {exc}")


def dataset_validator_node(state: AgentState) -> AgentState:
    """Validate CSV labels, local files and training inputs before GPU/API work."""
    report: dict[str, Any] = {
        "dataset_path": str(state.get("dataset_path") or ""),
        "checks": {},
        "warnings": [],
        "errors": list(state.get("input_errors", [])),
        "classes_by_head": {},
        "class_to_idx_by_head": {},
        "num_classes_by_head": {},
        "split_stats": {},
    }
    updates: dict[str, Any] = {}
    console.rule("[bold]Dataset and Input Validation[/bold]")
    heads_raw = state.get("heads", [])
    if isinstance(heads_raw, str):
        heads_raw = heads_raw.split()
    if not isinstance(heads_raw, (list, tuple)) or any(not isinstance(head, str) for head in heads_raw):
        report["errors"].append("heads must be a list of CSV column names.")
        heads_raw = []
    heads = list(dict.fromkeys(heads_raw))
    updates["heads"] = heads
    report["heads"] = heads
    if not heads:
        report["errors"].append("No classification heads supplied. Use --heads with CSV label columns.")
    for head in heads:
        if head in {"mean", "macro"} or head.startswith("final_"):
            report["errors"].append(f"Head name {head!r} conflicts with the agent's accuracy metric names.")
        if f"{head}_macro" in heads:
            report["errors"].append(
                f"Heads {head!r} and {head + '_macro'!r} conflict with per-head macro metric names."
            )
        if not head or "." in head or head in _RESERVED_HEAD_NAMES:
            report["errors"].append(f"Head name {head!r} cannot be used as a PyTorch ModuleDict key.")
        if any(character in head for character in ",\n\r="):
            report["errors"].append(f"Head name {head!r} cannot contain commas, newlines, or '='.")

    arch = state.get("model_type", "resnet18")
    if arch not in _ARCHITECTURES:
        report["errors"].append(f"Unsupported architecture {arch!r}; choose resnet18, resnet34, or resnet50.")
    report["arch"] = arch
    decision_split = str(state.get("decision_split") or "test").strip().lower()
    report["decision_split"] = updates["decision_split"] = decision_split
    if decision_split not in {"test", "val"}:
        report["errors"].append("decision_split must be test or val.")
    target_metric = str(state.get("target_metric") or "mean_acc").strip()
    try:
        updates["target_metric"] = canonical_target_metric(target_metric, heads)
    except ValueError as exc:
        report["errors"].append(str(exc))
    target = _finite_number(state.get("target_value", 0.85))
    if target is None or not 0 <= target <= 1:
        report["errors"].append("target_value must be a finite accuracy between 0 and 1.")
    for field, default in (("max_attempts", 5), ("nproc_per_node", 1)):
        value = state.get(field, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            report["errors"].append(f"{field} must be an integer greater than zero.")
    if state.get("export_format", "onnx") not in {"onnx", "none"}:
        report["errors"].append("export_format must be onnx or none.")

    dataset = str(state.get("dataset_path") or "")
    if dataset:
        report["dataset_path"] = str(_resolve_path(dataset))
        updates["dataset_path"] = report["dataset_path"]
    raw_weights = str(state.get("weights_path") or PROJECT_ROOT / "weights" / f"{arch}.pth")
    weights = str(_resolve_path(raw_weights))
    report["weights_path"] = updates["weights_path"] = weights
    if not Path(weights).is_file():
        report["errors"].append(
            f"Pretrained weights not found: {weights}. Supply --weights with an ImageNet {arch} state_dict."
        )
    data_root = str(state.get("data_root") or "")
    if data_root:
        data_root = str(_resolve_path(data_root))
        if not Path(data_root).is_dir():
            report["errors"].append(f"data_root is not a directory: {data_root}")
    report["data_root"] = updates["data_root"] = data_root

    frames: dict[str, pd.DataFrame] = {}
    required_columns = {"root_folder", "image_name", *heads}
    splits = ["train", "val"]
    if decision_split == "test" or state.get("test_csv"):
        splits.append("test")
    for split in splits:
        field = f"{split}_csv"
        raw_path = str(state.get(field) or (str(Path(dataset) / f"{split}.csv") if dataset else ""))
        if not raw_path:
            report["errors"].append(
                f"Missing {field}. Supply --{split}-csv, {field.upper()} in .env, or --dataset."
            )
            continue
        path = _resolve_path(raw_path)
        report[field] = updates[field] = str(path)
        if not path.is_file():
            report["errors"].append(f"{split} CSV not found: {path}")
            continue
        try:
            df = read_dataset_csv(path, heads)
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            report["errors"].append(f"Cannot read {split} CSV {path}: {exc}")
            continue
        report["split_stats"][split] = {"images": len(df)}
        if df.empty:
            report["errors"].append(f"{split} CSV is empty: {path}")
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            report["errors"].append(f"{split} CSV is missing columns: {missing_columns}.")
            continue
        frames[split] = df
        for head in heads:
            missing_labels = df[head].isna() | (df[head].astype(str).str.strip() == "")
            if missing_labels.any():
                rows = df.index[missing_labels].tolist()[:10]
                report["errors"].append(f"{split}: head {head!r} has missing labels at row indices {rows}.")
        _check_images(df, split, data_root, report)

    if "train" in frames:
        for head in heads:
            train_labels = frames["train"][head].dropna().astype(str)
            classes = sorted(train_labels.unique().tolist())
            report["classes_by_head"][head] = classes
            report["class_to_idx_by_head"][head] = {label: index for index, label in enumerate(classes)}
            report["num_classes_by_head"][head] = len(classes)
            for split in ("val", "test"):
                if split not in frames:
                    continue
                eval_labels = set(frames[split][head].dropna().astype(str))
                unseen = sorted(eval_labels - set(classes))
                if unseen:
                    report["errors"].append(f"{split}: head {head!r} contains classes absent from training: {unseen}.")

    weights_by_head = state.get("head_loss_weights")
    if weights_by_head is None:
        weights_by_head = {}
    if not isinstance(weights_by_head, dict):
        report["errors"].append("head_loss_weights must map head names to nonnegative finite weights.")
    else:
        numeric_weights = {head: 1.0 for head in heads}
        for head, value in weights_by_head.items():
            numeric = _finite_number(value)
            if head not in heads:
                report["errors"].append(f"Loss weight supplied for unknown head {head!r}.")
            elif numeric is None or numeric < 0:
                report["errors"].append(f"Loss weight for {head!r} must be finite and nonnegative.")
            else:
                numeric_weights[head] = numeric
        if heads and not any(value > 0 for value in numeric_weights.values()):
            report["errors"].append("At least one head loss weight must be greater than zero.")
        updates["head_loss_weights"] = numeric_weights
    alpha_path = str(state.get("alpha_json") or "")
    if alpha_path:
        alpha_path = str(_resolve_path(alpha_path))
    updates["alpha_json"] = report["alpha_json"] = alpha_path
    _check_alpha(alpha_path, report, heads)

    table = Table(title="CSV Dataset")
    table.add_column("Split")
    table.add_column("Images", justify="right")
    for split, stats in report["split_stats"].items():
        table.add_row(split, str(stats["images"]))
    console.print(table)
    for head, count in report["num_classes_by_head"].items():
        console.print(f"[dim]{head}:[/dim] {count} classes")
    for warning in report["warnings"]:
        console.print(f"[yellow]WARN:[/yellow] {warning}")
    for error in report["errors"]:
        console.print(f"[red]ERROR:[/red] {error}")
    is_valid = not report["errors"]
    if is_valid:
        report["checks"] = {"inputs": True, "csv_columns": True, "labels": True, "image_paths": True}
        console.print("[bold green]✓ Dataset validation passed.[/bold green]\n")
    else:
        console.print("[bold red]✗ Dataset validation failed.[/bold red]\n")
    return {**updates, "validation_report": report, "is_valid": is_valid}
