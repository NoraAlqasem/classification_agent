from __future__ import annotations

import json
import math
import os
import re
from typing import Any

from langchain_core.tools import tool

VALID_RESNET_CONFIG_KEYS = frozenset({
    "arch", "weights", "train_csv", "val_csv", "data_root", "heads",
    "epochs", "batch_size", "num_workers", "img_size", "lr", "weight_decay",
    "gamma", "patience", "monitor", "head_loss_weights", "alpha_json",
    "nproc_per_node", "out", "name",
})
VALID_RESNET_CONFIG_KEYS_DOC = ", ".join(sorted(VALID_RESNET_CONFIG_KEYS))
HYPERPARAMETER_KEYS = frozenset({
    "epochs", "batch_size", "img_size", "lr", "weight_decay", "gamma",
    "patience", "head_loss_weights", "alpha_json",
})
MODEL_SELECTION_KEYS = frozenset({"arch"})
TUNABLE_CONFIG_KEYS = HYPERPARAMETER_KEYS | MODEL_SELECTION_KEYS
PROTECTED_CONFIG_KEYS = VALID_RESNET_CONFIG_KEYS - TUNABLE_CONFIG_KEYS


def get_llm_allowed_config_keys(raw: Any = None) -> frozenset[str]:
    """Resolve the user's editable fields; an explicit empty selection locks all."""
    if raw is None:
        raw = os.getenv("LLM_ALLOWED_CONFIG_KEYS", "all")
    if isinstance(raw, str):
        entries = [raw]
    elif isinstance(raw, (list, tuple)) and all(isinstance(key, str) for key in raw):
        entries = list(raw)
    else:
        raise ValueError("LLM_ALLOWED_CONFIG_KEYS must be a comma/space-separated list of configuration keys.")
    keys = frozenset(key for entry in entries for key in re.split(r"[,\s]+", entry.strip()) if key)
    if keys == {"all"}:
        return TUNABLE_CONFIG_KEYS
    if not keys or keys == {"none"}:
        return frozenset()
    if keys & {"all", "none"}:
        raise ValueError("LLM_ALLOWED_CONFIG_KEYS: use 'all' or 'none' alone, or list specific keys.")
    invalid = keys - TUNABLE_CONFIG_KEYS
    if invalid:
        hint = " Pretrained weights are selected automatically from weights/ when arch changes." if "weights" in invalid else ""
        raise ValueError(
            f"LLM_ALLOWED_CONFIG_KEYS contains unsupported or protected keys: {', '.join(sorted(invalid))}. "
            f"Choose from: {', '.join(sorted(TUNABLE_CONFIG_KEYS))}, all, or none.{hint}"
        )
    return keys


def get_optimizer_hyperparameter_limit() -> int:
    """Read the changed-key limit; any class-weight edits in alpha_json count as one."""
    name = "LLM_OPTIMIZATION_MAX_HYPERPARAMETERS"
    raw = os.getenv(name, "5")
    error = f"{name} must be a positive integer; got {raw!r}."
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(error) from exc
    if value <= 0:
        raise ValueError(error)
    return value


def sanitize_resnet_config(config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Filter unsupported keys and reject invalid parameters before GPU launch."""
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a dictionary")
    removed = [key for key in config if key not in VALID_RESNET_CONFIG_KEYS]
    clean = {key: value for key, value in config.items() if key in VALID_RESNET_CONFIG_KEYS}
    missing = VALID_RESNET_CONFIG_KEYS - set(clean)
    if missing:
        raise ValueError(f"Missing training configuration keys: {', '.join(sorted(missing))}")
    if clean["arch"] not in {"resnet18", "resnet34", "resnet50"}:
        raise ValueError("arch must be resnet18, resnet34, or resnet50")
    for key in ("weights", "train_csv", "val_csv", "out", "name"):
        if not isinstance(clean[key], str) or not clean[key].strip():
            raise ValueError(f"{key} must be a nonempty path/name string")
    if clean["name"] in {".", ".."} or any(c in clean["name"] for c in "/\\"):
        raise ValueError("name must be a directory name, not a path")
    for key in ("data_root", "alpha_json"):
        if clean[key] is not None and not isinstance(clean[key], str):
            raise ValueError(f"{key} must be a path string or null")
    heads = clean["heads"]
    if not isinstance(heads, list) or not heads or any(not isinstance(h, str) or not h for h in heads):
        raise ValueError("heads must be a nonempty list of column names")
    if len(set(heads)) != len(heads):
        raise ValueError("heads must be unique")
    if clean["monitor"] not in ["mean", "macro", *heads, *(f"{head}_macro" for head in heads)]:
        raise ValueError("monitor must be 'mean', 'macro', a requested head, or HEAD_macro")
    for key in ("epochs", "batch_size", "num_workers", "patience", "nproc_per_node"):
        value = clean[key]
        minimum = 0 if key == "num_workers" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    image_size = clean["img_size"]
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 32 for v in image_size
    ):
        raise ValueError("img_size must contain two integers >= 32: [height, width]")
    clean["img_size"] = list(image_size)
    for key in ("lr", "weight_decay", "gamma"):
        value = clean[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite and numeric")
        if value < 0 or (key == "lr" and value == 0):
            raise ValueError(f"{key} must be {'positive' if key == 'lr' else 'nonnegative'}")
    weights = clean["head_loss_weights"]
    if not isinstance(weights, dict) or set(weights) - set(heads):
        raise ValueError("head_loss_weights must map requested heads to numbers")
    complete_weights = {head: weights.get(head, 1.0) for head in heads}
    for value in complete_weights.values():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Head loss weights must be finite and nonnegative")
    if not any(complete_weights.values()):
        raise ValueError("At least one head loss weight must be positive")
    clean["head_loss_weights"] = complete_weights
    return clean, removed


@tool
def read_config(config_json: str) -> str:
    """Summarize multi-head ResNet hyperparameters and fixed settings."""
    try:
        config, removed = sanitize_resnet_config(json.loads(config_json))
        return json.dumps({
            "tunable": {k: config[k] for k in sorted(TUNABLE_CONFIG_KEYS)},
            "fixed": {k: config[k] for k in sorted(PROTECTED_CONFIG_KEYS)},
            "global_batch_size": config["batch_size"] * config["nproc_per_node"],
            "removed_keys": removed,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        }, indent=2)
    except (TypeError, ValueError) as exc:
        return json.dumps({"error": str(exc)})
