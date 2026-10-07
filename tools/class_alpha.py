from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep this equal to training.train_multihead_ddp.DEFAULT_ALPHA, without loading torch.
DEFAULT_ALPHA = 1.0


def _normalize_values(values: Any, *, strict_numbers: bool) -> dict[str, dict[str, float]]:
    if not isinstance(values, dict):
        raise ValueError("alpha_json must be an object mapping heads to class alpha values")
    normalized: dict[str, dict[str, float]] = {}
    for head, class_values in values.items():
        if not isinstance(head, str):
            raise ValueError("alpha_json head names must be strings")
        if not isinstance(class_values, dict):
            raise ValueError(f"alpha_json head {head!r} must map class names to alpha values")
        normalized[head] = {}
        for label, value in class_values.items():
            if not isinstance(label, str):
                raise ValueError(f"alpha_json class names for head {head!r} must be strings")
            error = f"alpha_json value for {head}/{label} must be a finite nonnegative number"
            if isinstance(value, bool) or (
                strict_numbers and not isinstance(value, (int, float))
            ):
                raise ValueError(error)
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(error) from exc
            if not math.isfinite(number) or number < 0:
                raise ValueError(error)
            normalized[head][label] = number
    return normalized


def load_class_alpha(path: str | None) -> dict[str, dict[str, float]]:
    """Read a baseline alpha file, retaining unused entries for compatibility."""
    if path is None or path == "":
        return {}
    if not isinstance(path, str) or not path.strip():
        raise ValueError("alpha_json must be a nonempty path string or null")
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    try:
        resolved = resolved.resolve()
        with resolved.open(encoding="utf-8") as handle:
            values = json.load(handle)
        return _normalize_values(values, strict_numbers=False)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid alpha_json file {str(resolved)!r}: {exc}") from exc


def _check_effective_values(
    values: dict[str, dict[str, float]],
    heads: list[str],
    classes_by_head: dict[str, list[str]],
) -> None:
    for head in heads:
        labels = classes_by_head.get(head, [])
        if labels and not any(values.get(head, {}).get(label, DEFAULT_ALPHA) > 0 for label in labels):
            raise ValueError(
                f"alpha_json for head {head!r} gives every training class zero weight; "
                "at least one class must have positive alpha"
            )


def validate_class_alpha_updates(
    values: Any,
    heads: list[str],
    classes_by_head: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Validate proposed class values using the verified training label names."""
    normalized = _normalize_values(values, strict_numbers=True)
    for head, class_values in normalized.items():
        if head not in heads:
            raise ValueError(f"alpha_json contains unknown head {head!r}; choose from {heads!r}")
        if head not in classes_by_head:
            raise ValueError(f"alpha_json cannot verify classes for head {head!r}: dataset classes are unavailable")
        unknown = sorted(set(class_values) - set(classes_by_head[head]))
        if unknown:
            raise ValueError(f"alpha_json contains unknown classes for head {head!r}: {unknown!r}")
    _check_effective_values(normalized, heads, classes_by_head)
    return normalized


def prepare_class_alpha(
    proposal: Any,
    current_values: dict,
    heads: list[str],
    classes_by_head: dict[str, list[str]],
) -> dict[str, dict[str, float]] | None:
    """Prepare a patch, replacement file, or disabled alpha without writing files."""
    if proposal is None:
        return None
    if isinstance(proposal, str):
        if not proposal.strip():
            raise ValueError("alpha_json must be a nonempty existing file path, a class-value object, or null")
        return validate_class_alpha_updates(load_class_alpha(proposal), heads, classes_by_head)
    if not isinstance(proposal, dict):
        raise ValueError("alpha_json must be a class-value object, an existing file path, or null")
    patch = validate_class_alpha_updates(proposal, heads, classes_by_head)
    merged = copy.deepcopy(_normalize_values(current_values, strict_numbers=False))
    for head, class_values in patch.items():
        merged.setdefault(head, {}).update(class_values)
    _check_effective_values(merged, heads, classes_by_head)
    return merged


def write_class_alpha(values: dict, session_dir: str | Path, attempt: int) -> str:
    """Save a unique alpha snapshot without overwriting user or earlier run files."""
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("Class-alpha snapshot attempt must be a positive integer")
    payload = json.dumps(values, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    alpha_dir = Path(session_dir).expanduser().resolve() / "alpha"
    alpha_dir.mkdir(parents=True, exist_ok=True)
    snapshot: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=alpha_dir,
            prefix=f"exp_{attempt:02d}_", suffix=".json", delete=False,
        ) as handle:
            snapshot = Path(handle.name)
            handle.write(payload)
        return str(snapshot.resolve())
    except Exception:
        if snapshot is not None:
            try:
                snapshot.unlink(missing_ok=True)
            except OSError:
                pass
        raise
